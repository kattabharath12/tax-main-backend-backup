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
        # Generic patterns that work for any W2
        self.w2_patterns = {
            'employee_ssn': [
                r'(?i)(?:employee.*?social security number|ssn)[\s\n:]*(\d{3}-\d{2}-\d{4})',
                r'(?i)(?:employee.*?social security number|ssn)[\s\n:]*(\d{9})',
                r'(\d{3}-\d{2}-\d{4})',  # Standard SSN format
                r'\b(\d{9})\b',  # 9-digit SSN without hyphens
            ],
            'employer_ein': [
                r'(?i)(?:employer identification number|ein)[\s\n:]*(\d{2}-\d{7})',
                r'(?i)(?:employer identification number|ein)[\s\n:]*([A-Z0-9]{9,11})',
                r'(\d{2}-\d{7})',  # Standard EIN format
                r'\b([A-Z]{2,5}\d{7,10})\b',  # Alternative EIN format
            ],
            'employer_name': [
                r'(?i)employer.*?name.*?address[\s\n:]*([A-Z][A-Za-z\s,\.&\-\'Inc]{2,50})(?=\s*\n|\s*\d)',
                r'(?i)(?:company|corp|corporation|inc|llc)[\s\n]*([A-Z][A-Za-z\s,\.&\-\']{2,50})',
                r'([A-Z][A-Za-z\s&,\.]{2,40}(?:\s+(?:Inc|Corp|LLC|Company|Co)\.?))',
            ],
            'employee_name': [
                r'(?i)employee.*?name.*?address[\s\n:]*([A-Z][A-Za-z\s\-\'\.]{2,50})(?=\s*\n|\s*\d)',
                r'([A-Z][a-z]+\s+[A-Z][a-z]+)',  # First Last name pattern
            ],
            'employee_address': [
                r'(?i)employee.*?name.*?address[\s\n:]*[A-Za-z\s]+\n([0-9][0-9A-Za-z\s,\.\-]{10,100})',
            ],
            'employer_address': [
                r'(?i)employer.*?name.*?address[\s\n:]*[A-Za-z\s]+\n([0-9][0-9A-Za-z\s,\.\-]{10,100})',
            ],
            # Wage and tax patterns - look for actual amounts
            'wages_tips_compensation': [
                r'(?i)(?:1\.?\s*wages.*?compensation|box\s*1.*?wages|wages.*?tips.*?compensation)[\s\n:$]*([0-9,]+(?:\.\d{2})?)',
                r'(?i)wages.*?tips.*?other.*?compensation[\s\n:$]*([0-9,]+(?:\.\d{2})?)',
                r'\$?([1-9]\d{4,6}(?:,\d{3})*(?:\.\d{2})?)',  # Large amounts (wages)
            ],
            'federal_income_tax_withheld': [
                r'(?i)(?:2\.?\s*federal.*?tax.*?withheld|box\s*2.*?federal|federal.*?income.*?tax.*?withheld)[\s\n:$]*([0-9,]+(?:\.\d{2})?)',
                r'(?i)federal.*?income.*?tax.*?withheld[\s\n:$]*([0-9,]+(?:\.\d{2})?)',
            ],
            'social_security_wages': [
                r'(?i)(?:3\.?\s*social.*?security.*?wages|box\s*3.*?social|social.*?security.*?wages)[\s\n:$]*([0-9,]+(?:\.\d{2})?)',
                r'(?i)social.*?security.*?wages[\s\n:$]*([0-9,]+(?:\.\d{2})?)',
            ],
            'social_security_tax_withheld': [
                r'(?i)(?:4\.?\s*social.*?security.*?tax.*?withheld|box\s*4.*?social|social.*?security.*?tax.*?withheld)[\s\n:$]*([0-9,]+(?:\.\d{2})?)',
                r'(?i)social.*?security.*?tax.*?withheld[\s\n:$]*([0-9,]+(?:\.\d{2})?)',
            ],
            'medicare_wages': [
                r'(?i)(?:5\.?\s*medicare.*?wages|box\s*5.*?medicare|medicare.*?wages.*?tips)[\s\n:$]*([0-9,]+(?:\.\d{2})?)',
                r'(?i)medicare.*?wages.*?tips[\s\n:$]*([0-9,]+(?:\.\d{2})?)',
            ],
            'medicare_tax_withheld': [
                r'(?i)(?:6\.?\s*medicare.*?tax.*?withheld|box\s*6.*?medicare|medicare.*?tax.*?withheld)[\s\n:$]*([0-9,]+(?:\.\d{2})?)',
                r'(?i)medicare.*?tax.*?withheld[\s\n:$]*([0-9,]+(?:\.\d{2})?)',
            ],
        }

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
            
            configs = ['--oem 3 --psm 6', '--oem 3 --psm 4', '--oem 3 --psm 3']
            
            best_text = ""
            best_confidence = 0.0
            
            for config in configs:
                try:
                    text = pytesseract.image_to_string(temp_path, config=config)
                    data = pytesseract.image_to_data(temp_path, config=config, output_type=pytesseract.Output.DICT)
                    
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    if avg_confidence > best_confidence and text.strip():
                        best_text = text
                        best_confidence = avg_confidence
                        
                except Exception:
                    continue
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            return best_text, best_confidence / 100.0
            
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return "", 0.0

    def clean_amount(self, amount_str: str) -> Optional[float]:
        """Clean and convert amount strings to float"""
        try:
            # Remove $ signs, commas, and extra spaces
            cleaned = re.sub(r'[\$,\s]', '', amount_str)
            # Handle decimal points
            if '.' in cleaned:
                return float(cleaned)
            else:
                # If no decimal, assume it's in dollars
                return float(cleaned)
        except (ValueError, TypeError):
            return None

    def extract_w2_fields(self, text: str) -> Dict[str, Any]:
        """Extract W2 fields using generic patterns"""
        extracted_fields = {}
        
        # Normalize text for better matching
        text_normalized = re.sub(r'\s+', ' ', text)
        
        logger.info(f"Extracting from text length: {len(text_normalized)}")
        
        # Show first few lines for debugging
        lines = text_normalized.split('\n')[:10]
        logger.info(f"First lines: {lines}")
        
        for field_name, patterns in self.w2_patterns.items():
            found_value = None
            
            for pattern in patterns:
                try:
                    matches = re.findall(pattern, text_normalized, re.IGNORECASE | re.MULTILINE)
                    
                    if matches:
                        value = matches[0].strip()
                        
                        if not value or len(value) < 1:
                            continue
                        
                        # Handle numeric fields (wages, taxes)
                        if field_name in ['wages_tips_compensation', 'federal_income_tax_withheld', 
                                        'social_security_wages', 'social_security_tax_withheld',
                                        'medicare_wages', 'medicare_tax_withheld']:
                            
                            amount = self.clean_amount(value)
                            if amount is not None and 0 < amount <= 1000000:
                                found_value = amount
                                logger.info(f"✅ Found {field_name}: ${amount:,.2f}")
                                break
                        
                        # Handle SSN formatting
                        elif field_name == 'employee_ssn':
                            ssn_clean = re.sub(r'[^\d]', '', value)
                            if len(ssn_clean) == 9:
                                formatted_ssn = f"{ssn_clean[:3]}-{ssn_clean[3:5]}-{ssn_clean[5:]}"
                                found_value = formatted_ssn
                                logger.info(f"✅ Found {field_name}: {formatted_ssn}")
                                break
                        
                        # Handle EIN formatting
                        elif field_name == 'employer_ein':
                            if re.match(r'\d{2}-\d{7}', value):
                                found_value = value
                            else:
                                ein_clean = re.sub(r'[^\dA-Z]', '', value)
                                if len(ein_clean) >= 9:
                                    found_value = ein_clean
                            logger.info(f"✅ Found {field_name}: {found_value}")
                            break
                        
                        # Handle text fields (names, addresses)
                        else:
                            # Clean up text fields
                            cleaned_value = re.sub(r'[^\w\s\-\.,&\']', '', value)
                            if len(cleaned_value) >= 2:
                                found_value = cleaned_value.strip()
                                logger.info(f"✅ Found {field_name}: {found_value}")
                                break
                                
                except Exception as e:
                    logger.warning(f"Pattern error for {field_name}: {e}")
                    continue
            
            if found_value is not None:
                extracted_fields[field_name] = found_value

        # Extract tax year
        year_match = re.search(r'\b(20\d{2})\b', text_normalized)
        if year_match:
            extracted_fields['tax_year'] = int(year_match.group(1))
            logger.info(f"✅ Found tax_year: {year_match.group(1)}")

        logger.info(f"Total fields extracted: {len(extracted_fields)}")
        return extracted_fields

    def is_w2_document(self, text: str) -> bool:
        """Check if document appears to be a W2"""
        text_lower = text.lower()
        w2_indicators = [
            'w-2', 'wage and tax statement', 'social security', 'medicare', 
            'federal income tax', 'employer identification number'
        ]
        return any(indicator in text_lower for indicator in w2_indicators)

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process document with generic W2 extraction"""
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
                # Try all pages and combine text
                combined_text = ""
                total_confidence = 0
                page_count = 0
                
                for page_num in range(3):  # Check first 3 pages
                    text, confidence = self.extract_text_from_pdf_page(file_path, page_num)
                    if text and len(text.strip()) > 20:
                        combined_text += text + "\n"
                        total_confidence += confidence
                        page_count += 1
                        logger.info(f"Added page {page_num + 1} to combined text")
                
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
                    'error': 'No text could be extracted from document',
                    'confidence': 0.0,
                    'raw_text': '',
                    'extracted_fields': {}
                }

            # Check if this looks like a W2
            is_w2 = self.is_w2_document(all_text)
            
            if not is_w2:
                return {
                    'is_w2': False,
                    'error': 'Document does not appear to be a W2 form',
                    'confidence': best_confidence,
                    'raw_text': all_text[:500],
                    'extracted_fields': {}
                }

            # Extract W2 fields
            extracted_fields = self.extract_w2_fields(all_text)

            logger.info(f"🏁 FINAL RESULT:")
            logger.info(f"   Is W2: {is_w2}")
            logger.info(f"   Confidence: {best_confidence:.1%}")
            logger.info(f"   Fields Extracted: {len(extracted_fields)}")
            logger.info(f"   Data: {extracted_fields}")

            return {
                'is_w2': is_w2,
                'confidence': best_confidence,
                'raw_text': all_text[:500],  # First 500 chars
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
