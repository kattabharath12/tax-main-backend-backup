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

    def is_template_data(self, text: str) -> bool:
        """Check if this contains template/sample data that should be skipped"""
        text_lower = text.lower()
        
        template_indicators = [
            'this is a sample',
            'for testing purposes only',
            'not for actual tax filing',
            'copy a of this form is provided for informational purposes'
        ]
        
        return any(indicator in text_lower for indicator in template_indicators)

    def extract_flexible_w2_data(self, text: str) -> Dict[str, Any]:
        """Extract W2 data using flexible patterns for different formats"""
        extracted = {}
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        full_text = ' '.join(lines)
        
        logger.info(f"Extracting from {len(lines)} lines using flexible patterns")
        
        # Skip if this is explicitly marked as template/sample
        if self.is_template_data(text):
            logger.info("⚠️ Document marked as sample/template - extracting anyway but noting it")
        
        # 1. Extract SSN - any valid SSN
        ssn_patterns = [
            r'(?i)employee.*?social security number[:\s]*(\d{3}-\d{2}-\d{4})',
            r'(?i)social security number[:\s]*(\d{3}-\d{2}-\d{4})',
            r'\b(\d{3}-\d{2}-\d{4})\b',
            r'\b(\d{9})\b',
        ]
        
        for pattern in ssn_patterns:
            matches = re.findall(pattern, full_text)
            if matches:
                ssn = matches[0]
                if len(ssn) == 9:
                    ssn = f"{ssn[:3]}-{ssn[3:5]}-{ssn[5:]}"
                extracted['employee_ssn'] = ssn
                logger.info(f"✅ Found SSN: {ssn}")
                break
        
        # 2. Extract EIN - any valid EIN
        ein_patterns = [
            r'(?i)employer identification number[:\s]*(\d{2}-\d{7})',
            r'(?i)ein[:\s]*(\d{2}-\d{7})',
            r'\b(\d{2}-\d{7})\b',
            r'\b([A-Z]{2,5}\d{7,10})\b',
        ]
        
        for pattern in ein_patterns:
            matches = re.findall(pattern, full_text)
            if matches:
                extracted['employer_ein'] = matches[0]
                logger.info(f"✅ Found EIN: {matches[0]}")
                break
        
        # 3. Extract Employee Name - look after "Employee's name"
        employee_name_patterns = [
            r'(?i)employee.*?name.*?address[:\s]*\n([A-Z][a-zA-Z\s\-\'\.]{2,40})',
            r'(?i)employee.*?name[:\s]*\n([A-Z][a-zA-Z\s\-\'\.]{2,40})',
            r'(?i)employee.*?name[:\s]*([A-Z][a-zA-Z\s\-\'\.]{2,40})',
        ]
        
        for pattern in employee_name_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            if matches:
                name = matches[0].strip()
                # Clean the name and check it's not an address
                name = re.sub(r'[^\w\s\-\']', '', name).strip()
                
                # Skip if it looks like an address or form text
                skip_terms = ['street', 'avenue', 'road', 'city', 'st ', 'ave', 'main', 'business']
                if (len(name) >= 3 and 
                    not any(term in name.lower() for term in skip_terms) and
                    not re.search(r'\d', name)):  # Skip if contains numbers
                    
                    extracted['employee_name'] = name
                    logger.info(f"✅ Found employee name: {name}")
                    break
        
        # 4. Extract Employer Name - look after "Employer's name"
        employer_name_patterns = [
            r'(?i)employer.*?name.*?address[:\s]*\n([A-Z][a-zA-Z\s\-\'\.\&\,]{2,50})',
            r'(?i)employer.*?name[:\s]*\n([A-Z][a-zA-Z\s\-\'\.\&\,]{2,50})',
            r'(?i)employer.*?name[:\s]*([A-Z][a-zA-Z\s\-\'\.\&\,]{3,40})',
        ]
        
        for pattern in employer_name_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            if matches:
                name = matches[0].strip()
                name = re.sub(r'[^\w\s\-\'\.\&\,]', '', name).strip()
                
                # Skip addresses and form text
                skip_terms = ['street', 'avenue', 'road', 'city', 'st ', 'ave', 'business ave', 'main']
                if (len(name) >= 3 and 
                    not any(term in name.lower() for term in skip_terms) and
                    not re.search(r'\d{3,}', name)):  # Skip if contains 3+ digit numbers
                    
                    extracted['employer_name'] = name
                    logger.info(f"✅ Found employer name: {name}")
                    break
        
        # 5. Extract Tax Year
        year_patterns = [
            r'(?i)form w-2.*?(\d{4})',
            r'(?i)tax year[:\s]*(\d{4})',
            r'(?i)wage and tax statement\s+(\d{4})',
            r'\b(20\d{2})\b',
        ]
        
        for pattern in year_patterns:
            matches = re.findall(pattern, full_text)
            if matches:
                years = [int(y) for y in matches if 2020 <= int(y) <= 2025]
                if years:
                    extracted['tax_year'] = max(years)
                    logger.info(f"✅ Found tax year: {extracted['tax_year']}")
                    break
        
        # 6. Extract Wage Amounts - flexible patterns for different formats
        wage_fields = {
            'wages_tips_compensation': [
                # Format 1: "1. Wages, tips, other compensation: $75,000.00"
                r'(?i)1\.?\s*wages.*?compensation[:\s]*\$?([0-9,]+\.?\d{0,2})',
                # Format 2: "Box 1: Wages, tips, other compensation $75,000.00"
                r'(?i)box\s*1[:\s]*.*?\$?([0-9,]+\.?\d{0,2})',
                # Format 3: Just look for amounts after "wages"
                r'(?i)wages.*?compensation.*?\$?([0-9,]+\.?\d{0,2})',
            ],
            'federal_income_tax_withheld': [
                r'(?i)2\.?\s*federal.*?tax.*?withheld[:\s]*\$?([0-9,]+\.?\d{0,2})',
                r'(?i)box\s*2[:\s]*.*?\$?([0-9,]+\.?\d{0,2})',
                r'(?i)federal.*?income.*?tax.*?withheld.*?\$?([0-9,]+\.?\d{0,2})',
            ],
            'social_security_wages': [
                r'(?i)3\.?\s*social.*?security.*?wages[:\s]*\$?([0-9,]+\.?\d{0,2})',
                r'(?i)box\s*3[:\s]*.*?\$?([0-9,]+\.?\d{0,2})',
                r'(?i)social.*?security.*?wages.*?\$?([0-9,]+\.?\d{0,2})',
            ],
            'social_security_tax_withheld': [
                r'(?i)4\.?\s*social.*?security.*?tax.*?withheld[:\s]*\$?([0-9,]+\.?\d{0,2})',
                r'(?i)box\s*4[:\s]*.*?\$?([0-9,]+\.?\d{0,2})',
                r'(?i)social.*?security.*?tax.*?withheld.*?\$?([0-9,]+\.?\d{0,2})',
            ],
            'medicare_wages': [
                r'(?i)5\.?\s*medicare.*?wages[:\s]*\$?([0-9,]+\.?\d{0,2})',
                r'(?i)box\s*5[:\s]*.*?\$?([0-9,]+\.?\d{0,2})',
                r'(?i)medicare.*?wages.*?tips.*?\$?([0-9,]+\.?\d{0,2})',
            ],
            'medicare_tax_withheld': [
                r'(?i)6\.?\s*medicare.*?tax.*?withheld[:\s]*\$?([0-9,]+\.?\d{0,2})',
                r'(?i)box\s*6[:\s]*.*?\$?([0-9,]+\.?\d{0,2})',
                r'(?i)medicare.*?tax.*?withheld.*?\$?([0-9,]+\.?\d{0,2})',
            ],
        }
        
        for field_name, patterns in wage_fields.items():
            for pattern in patterns:
                matches = re.findall(pattern, full_text)
                if matches:
                    try:
                        amount_str = matches[0].replace(',', '').replace('$', '').strip()
                        if amount_str:  # Make sure it's not empty
                            amount = float(amount_str)
                            # Accept any reasonable amount (including 0)
                            if 0 <= amount <= 9999999:
                                extracted[field_name] = amount
                                logger.info(f"✅ Found {field_name}: ${amount:,.2f}")
                                break
                    except (ValueError, IndexError):
                        continue
        
        logger.info(f"Flexible extraction found {len(extracted)} fields")
        return extracted

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process document with flexible format handling"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            logger.info(f"Processing {file_ext} document with flexible patterns")

            if file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                text, confidence = self.extract_text_from_image(file_path)
                extracted_fields = self.extract_flexible_w2_data(text)
                best_confidence = confidence
                
            elif file_ext == '.pdf':
                # Combine all pages for comprehensive extraction
                all_text = ""
                total_confidence = 0
                page_count = 0
                
                for page_num in range(min(5, 10)):
                    try:
                        text, confidence = self.extract_text_from_pdf_page(file_path, page_num)
                        
                        if text and len(text.strip()) > 20:
                            all_text += text + "\n\n"
                            total_confidence += confidence
                            page_count += 1
                            logger.info(f"Added page {page_num + 1} ({len(text)} chars)")
                            
                    except Exception as e:
                        logger.warning(f"Error processing page {page_num + 1}: {e}")
                        continue
                
                if not all_text:
                    return {
                        'is_w2': False,
                        'error': 'No readable content found in PDF',
                        'confidence': 0.0
                    }
                
                best_confidence = total_confidence / page_count if page_count > 0 else 0.0
                extracted_fields = self.extract_flexible_w2_data(all_text)
                
            else:
                return {
                    'is_w2': False,
                    'error': f'Unsupported file type: {file_ext}',
                    'confidence': 0.0
                }

            # Check if this is a valid W2
            has_w2_structure = any(keyword in all_text.lower() for keyword in ['w-2', 'wage and tax statement'])
            has_key_data = any(key in extracted_fields for key in ['employee_ssn', 'employer_ein'])
            has_wage_data = any(key in extracted_fields for key in ['wages_tips_compensation', 'federal_income_tax_withheld'])
            
            is_w2 = has_w2_structure and (has_key_data or has_wage_data or len(extracted_fields) >= 3)

            logger.info(f"🏁 FLEXIBLE W2 EXTRACTION:")
            logger.info(f"   Is W2: {is_w2}")
            logger.info(f"   Has structure: {has_w2_structure}")
            logger.info(f"   Has key data: {has_key_data}")
            logger.info(f"   Has wage data: {has_wage_data}")
            logger.info(f"   Fields found: {len(extracted_fields)}")
            logger.info(f"   Data: {extracted_fields}")

            return {
                'is_w2': is_w2,
                'confidence': best_confidence,
                'raw_text': "Processed with flexible format handling",
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
