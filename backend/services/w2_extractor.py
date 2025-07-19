import pytesseract
import pdfplumber
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance, ImageFilter
import re
import os
import logging
from typing import Dict, Any, Optional, Tuple, List
import cv2
import numpy as np
import tempfile

logger = logging.getLogger(__name__)

class W2Extractor:
    def __init__(self):
        pass

    def extract_text_from_pdf_page(self, pdf_path: str, page_num: int) -> Tuple[str, float]:
        """Extract text from specific PDF page"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_num < len(pdf.pages):
                    page = pdf.pages[page_num]
                    text = page.extract_text()
                    if text and len(text.strip()) > 50:
                        logger.info(f"Page {page_num + 1} extracted {len(text)} characters")
                        return text, 0.9
            
            # Fallback to OCR
            with tempfile.TemporaryDirectory() as temp_dir:
                images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1, dpi=300)
                if images:
                    temp_image_path = os.path.join(temp_dir, f"page_{page_num}.png")
                    images[0].save(temp_image_path, 'PNG')
                    return self.extract_text_from_image(temp_image_path)
                    
        except Exception as e:
            logger.error(f"Error extracting from page {page_num}: {e}")
            
        return "", 0.0

    def extract_text_from_image(self, image_path: str) -> Tuple[str, float]:
        """Extract text from image"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return "", 0.0
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            temp_path = image_path.replace('.', '_processed.')
            cv2.imwrite(temp_path, thresh)
            
            text = pytesseract.image_to_string(temp_path, config='--oem 3 --psm 6')
            data = pytesseract.image_to_data(temp_path, config='--oem 3 --psm 6', output_type=pytesseract.Output.DICT)
            
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            return text, avg_confidence / 100.0
            
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return "", 0.0

    def is_w2_document(self, text: str) -> bool:
        """Check if document contains W2 indicators"""
        text_lower = text.lower()
        
        # W2 indicators
        w2_indicators = [
            'w-2', 'wage and tax statement', 'social security', 'medicare',
            'federal income tax', 'employer identification', 'wages, tips'
        ]
        
        w2_score = sum(1 for indicator in w2_indicators if indicator in text_lower)
        
        # Must have basic W2 structure
        has_structure = w2_score >= 3
        
        logger.info(f"W2 detection - indicators found: {w2_score}, is_w2: {has_structure}")
        
        return has_structure

    def extract_universal_data(self, text: str) -> Dict[str, Any]:
        """Extract W2 data using universal patterns"""
        extracted = {}
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        full_text = ' '.join(lines)
        
        logger.info(f"Processing {len(lines)} lines for universal extraction")
        
        # 1. Extract Employee SSN - any valid SSN format
        ssn_patterns = [
            r'(?i)employee.*?social security number[:\s]*(\d{3}-\d{2}-\d{4})',
            r'(?i)social security number[:\s]*(\d{3}-\d{2}-\d{4})', 
            r'(\d{3}-\d{2}-\d{4})',  # Standard format
            r'(\d{9})',  # 9 digits without separators
        ]
        
        for pattern in ssn_patterns:
            matches = re.findall(pattern, full_text)
            if matches:
                ssn = matches[0]
                if len(ssn) == 9:  # Format 9-digit SSN
                    ssn = f"{ssn[:3]}-{ssn[3:5]}-{ssn[5:]}"
                extracted['employee_ssn'] = ssn
                logger.info(f"✅ Found SSN: {ssn}")
                break
        
        # 2. Extract Employer EIN - any valid EIN format
        ein_patterns = [
            r'(?i)employer identification number[:\s]*(\d{2}-\d{7})',
            r'(?i)ein[:\s]*(\d{2}-\d{7})',
            r'(\d{2}-\d{7})',  # Standard EIN format
            r'([A-Z]{2,5}\d{7,10})',  # Alternative format
        ]
        
        for pattern in ein_patterns:
            matches = re.findall(pattern, full_text)
            if matches:
                extracted['employer_ein'] = matches[0]
                logger.info(f"✅ Found EIN: {matches[0]}")
                break
        
        # 3. Extract Employee Name
        name_patterns = [
            r'(?i)employee.*?name.*?address[:\s]*\n([A-Z][a-zA-Z\s\-\'\.]{2,40})',
            r'(?i)employee.*?name[:\s]*([A-Z][a-zA-Z\s\-\'\.]{2,40})',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)',  # First Last format
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            if matches:
                name = matches[0].strip()
                # Clean up the name
                name = re.sub(r'[^\w\s\-\']', '', name).strip()
                if len(name) >= 3 and not any(word in name.lower() for word in ['address', 'street', 'city']):
                    extracted['employee_name'] = name
                    logger.info(f"✅ Found employee name: {name}")
                    break
        
        # 4. Extract Employer Name
        employer_patterns = [
            r'(?i)employer.*?name.*?address[:\s]*\n([A-Z][a-zA-Z\s\-\'\.\&]{2,50})',
            r'(?i)employer.*?name[:\s]*([A-Z][a-zA-Z\s\-\'\.\&]{2,50})',
            r'([A-Z][a-zA-Z\s\&\.\,]{3,40}(?:\s+(?:Inc|Corp|LLC|Company|Co)\.?)?)',
        ]
        
        for pattern in employer_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            if matches:
                name = matches[0].strip()
                # Clean up employer name
                name = re.sub(r'[^\w\s\-\'\.\&\,]', '', name).strip()
                if len(name) >= 3 and not any(word in name.lower() for word in ['address', 'street', 'city', 'number']):
                    extracted['employer_name'] = name
                    logger.info(f"✅ Found employer name: {name}")
                    break
        
        # 5. Extract Tax Year
        year_patterns = [
            r'(?i)tax year[:\s]*(\d{4})',
            r'(?i)form w-2.*?(\d{4})',
            r'\b(20\d{2})\b',
        ]
        
        for pattern in year_patterns:
            matches = re.findall(pattern, full_text)
            if matches:
                years = [int(y) for y in matches if 2020 <= int(y) <= 2025]
                if years:
                    extracted['tax_year'] = max(years)  # Take the most recent valid year
                    logger.info(f"✅ Found tax year: {extracted['tax_year']}")
                    break
        
        # 6. Extract Wage and Tax Amounts
        wage_patterns = {
            'wages_tips_compensation': [
                r'(?i)1\.?\s*wages.*?compensation[:\s\$]*([0-9,]+\.?\d{0,2})',
                r'(?i)wages.*?tips.*?compensation[:\s\$]*([0-9,]+\.?\d{0,2})',
                r'(?i)box\s*1[:\s\$]*([0-9,]+\.?\d{0,2})',
            ],
            'federal_income_tax_withheld': [
                r'(?i)2\.?\s*federal.*?tax.*?withheld[:\s\$]*([0-9,]+\.?\d{0,2})',
                r'(?i)federal.*?income.*?tax.*?withheld[:\s\$]*([0-9,]+\.?\d{0,2})',
                r'(?i)box\s*2[:\s\$]*([0-9,]+\.?\d{0,2})',
            ],
            'social_security_wages': [
                r'(?i)3\.?\s*social.*?security.*?wages[:\s\$]*([0-9,]+\.?\d{0,2})',
                r'(?i)social.*?security.*?wages[:\s\$]*([0-9,]+\.?\d{0,2})',
                r'(?i)box\s*3[:\s\$]*([0-9,]+\.?\d{0,2})',
            ],
            'social_security_tax_withheld': [
                r'(?i)4\.?\s*social.*?security.*?tax.*?withheld[:\s\$]*([0-9,]+\.?\d{0,2})',
                r'(?i)social.*?security.*?tax.*?withheld[:\s\$]*([0-9,]+\.?\d{0,2})',
                r'(?i)box\s*4[:\s\$]*([0-9,]+\.?\d{0,2})',
            ],
            'medicare_wages': [
                r'(?i)5\.?\s*medicare.*?wages[:\s\$]*([0-9,]+\.?\d{0,2})',
                r'(?i)medicare.*?wages.*?tips[:\s\$]*([0-9,]+\.?\d{0,2})',
                r'(?i)box\s*5[:\s\$]*([0-9,]+\.?\d{0,2})',
            ],
            'medicare_tax_withheld': [
                r'(?i)6\.?\s*medicare.*?tax.*?withheld[:\s\$]*([0-9,]+\.?\d{0,2})',
                r'(?i)medicare.*?tax.*?withheld[:\s\$]*([0-9,]+\.?\d{0,2})',
                r'(?i)box\s*6[:\s\$]*([0-9,]+\.?\d{0,2})',
            ],
        }
        
        for field_name, patterns in wage_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, full_text)
                if matches:
                    try:
                        # Clean and convert amount
                        amount_str = matches[0].replace(',', '').replace('$', '')
                        amount = float(amount_str)
                        if 0 <= amount <= 1000000:  # Reasonable range
                            extracted[field_name] = amount
                            logger.info(f"✅ Found {field_name}: ${amount:,.2f}")
                            break
                    except (ValueError, IndexError):
                        continue
        
        logger.info(f"Universal extraction complete - found {len(extracted)} fields")
        return extracted

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process document with universal W2 extraction"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            logger.info(f"Processing {file_ext} document: {file_path}")

            all_text = ""
            best_confidence = 0.0
            
            if file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                text, confidence = self.extract_text_from_image(file_path)
                all_text = text
                best_confidence = confidence
                
            elif file_ext == '.pdf':
                # Combine text from all pages
                combined_text = ""
                total_confidence = 0
                page_count = 0
                
                for page_num in range(5):  # Check first 5 pages
                    try:
                        text, confidence = self.extract_text_from_pdf_page(file_path, page_num)
                        
                        if text and len(text.strip()) > 20:
                            combined_text += text + "\n\n"
                            total_confidence += confidence
                            page_count += 1
                            logger.info(f"Added page {page_num + 1} text ({len(text)} chars)")
                    except Exception as e:
                        logger.warning(f"Error processing page {page_num + 1}: {e}")
                        continue
                
                all_text = combined_text
                best_confidence = total_confidence / page_count if page_count > 0 else 0.0
                
            else:
                return {
                    'is_w2': False,
                    'error': f'Unsupported file type: {file_ext}',
                    'confidence': 0.0
                }

            if not all_text or len(all_text.strip()) < 50:
                return {
                    'is_w2': False,
                    'error': 'No sufficient text extracted from document',
                    'confidence': 0.0,
                    'raw_text': '',
                    'extracted_fields': {}
                }

            # Check if this is a W2 document
            is_w2 = self.is_w2_document(all_text)
            
            if not is_w2:
                return {
                    'is_w2': False,
                    'error': 'Document does not appear to be a W2 form',
                    'confidence': best_confidence,
                    'raw_text': all_text[:500],
                    'extracted_fields': {}
                }

            # Extract W2 data
            extracted_fields = self.extract_universal_data(all_text)

            # Consider it successful if we found key data
            has_meaningful_data = len(extracted_fields) >= 3 or any(key in extracted_fields for key in ['employee_ssn', 'employer_ein', 'wages_tips_compensation'])

            logger.info(f"🏁 UNIVERSAL W2 EXTRACTION:")
            logger.info(f"   Is W2: {is_w2}")
            logger.info(f"   Has meaningful data: {has_meaningful_data}")
            logger.info(f"   Fields found: {len(extracted_fields)}")
            logger.info(f"   Data: {extracted_fields}")

            return {
                'is_w2': is_w2 and has_meaningful_data,
                'confidence': best_confidence,
                'raw_text': all_text[:500],
                'extracted_fields': extracted_fields,
                'error': None
            }

        except Exception as e:
            logger.error(f"Error processing document: {e}")
            import traceback
            traceback.print_exc()
            return {
                'is_w2': False,
                'error': str(e),
                'confidence': 0.0,
                'raw_text': '',
                'extracted_fields': {}
            }
