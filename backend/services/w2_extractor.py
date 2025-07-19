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
        # Patterns for extracting W2 data
        self.w2_patterns = {
            'employee_ssn': [
                r'(?i)(?:employee.*?social security number|a\s+employee.*?social security number)[\s\S]{0,200}?(\d{10})',
                r'(?i)social security number[\s\S]{0,100}?(\d{10})',
                r'(\d{10})',  # 10 consecutive digits
            ],
            'employer_ein': [
                r'(?i)(?:employer identification number|b\s+employer identification number|ein)[\s\S]{0,200}?([A-Z]{2,5}\d{7,10})',
                r'([A-Z]{2,5}\d{7,10})',  # Pattern like FGHU7896901
            ],
            'employer_name': [
                r'(?i)(?:c\s+employer.*?name|employer.*?name.*?address)[\s\S]{0,200}?([A-Z][A-Za-z\s,\.&\-\']{2,50})(?=\s*,|\s*\d|\s*\n)',
                r'([A-Z][A-Za-z\s&]{2,20})(?=\s*,\s*\d)',  # Name before address
            ],
            'employee_name': [
                r'(?i)(?:e\s+employee.*?first name|employee.*?first name.*?initial)[\s\S]{0,200}?([A-Z][A-Za-z\s\-\'\.]{2,30})',
                r'(?i)first name.*?initial[\s\S]{0,100}?([A-Z][A-Za-z\s\-\'\.]{2,30})',
            ],
            'employee_last_name': [
                r'(?i)last name[\s\S]{0,50}?([A-Z][A-Za-z\-\']{2,25})',
            ],
            'control_number': [
                r'(?i)(?:d\s+control number|control number)[\s\S]{0,100}?([A-Z0-9]{5,20})',
            ],
            'wages_tips_compensation': [
                r'(?i)1\s+wages.*?tips.*?other.*?compensation[\s\S]{0,100}?(\d{1,7})',
                r'1[\s\S]{0,50}?(\d{4,7})',  # Box 1 with 4+ digit number
            ],
            'federal_income_tax_withheld': [
                r'(?i)2\s+federal.*?income.*?tax.*?withheld[\s\S]{0,100}?(\d{1,6})',
                r'2[\s\S]{0,50}?(\d{2,6})',  # Box 2 with 2+ digit number
            ],
            'social_security_wages': [
                r'(?i)3\s+social.*?security.*?wages[\s\S]{0,100}?(\d{1,7})',
                r'3[\s\S]{0,50}?(\d{2,7})',  # Box 3 with 2+ digit number
            ],
            'social_security_tax_withheld': [
                r'(?i)4\s+social.*?security.*?tax.*?withheld[\s\S]{0,100}?(\d{1,6})',
                r'4[\s\S]{0,50}?(\d{2,6})',  # Box 4 with 2+ digit number
            ],
            'medicare_wages': [
                r'(?i)5\s+medicare.*?wages.*?tips[\s\S]{0,100}?(\d{1,7})',
                r'5[\s\S]{0,50}?(\d{2,7})',  # Box 5 with 2+ digit number
            ],
            'medicare_tax_withheld': [
                r'(?i)6\s+medicare.*?tax.*?withheld[\s\S]{0,100}?(\d{1,6})',
                r'6[\s\S]{0,50}?(\d{2,6})',  # Box 6 with 2+ digit number
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

    def is_real_w2_page(self, text: str, page_num: int) -> bool:
        """Determine if this page contains real W2 data (not template/instructions)"""
        text_lower = text.lower()
        
        # Skip instruction pages
        instruction_indicators = [
            'attention:',
            'future developments',
            'notice to employee',
            'instructions for employee',
            'do you have to file',
            'earned income tax credit',
            'corrections',
            'clergy and religious workers',
            'employers, please note'
        ]
        
        if any(indicator in text_lower for indicator in instruction_indicators):
            logger.info(f"Page {page_num + 1} appears to be instructions - skipping")
            return False
        
        # Look for template/example indicators
        template_indicators = [
            '123-45-6789',  # Template SSN
            '12-3456789',   # Template EIN
            'abc corp',     # Template employer
            'john doe',     # Template employee
        ]
        
        if any(indicator in text_lower for indicator in template_indicators):
            logger.info(f"Page {page_num + 1} appears to be template data - skipping")
            return False
        
        # Look for actual W2 structure with real data
        has_w2_structure = any(indicator in text_lower for indicator in [
            'form w-2', 'wage and tax statement', 'social security', 'medicare'
        ])
        
        # Look for realistic data patterns (not template)
        has_real_ssn = bool(re.search(r'(?!123-45-6789)\d{10}|\d{3}-\d{2}-\d{4}', text))
        has_real_ein = bool(re.search(r'(?!12-3456789)[A-Z]{2,5}\d{7,10}', text))
        
        is_real = has_w2_structure and (has_real_ssn or has_real_ein)
        
        logger.info(f"Page {page_num + 1} real W2 check: structure={has_w2_structure}, real_ssn={has_real_ssn}, real_ein={has_real_ein}, result={is_real}")
        
        return is_real

    def extract_w2_fields(self, text: str) -> Dict[str, Any]:
        """Extract W2 fields from text"""
        extracted_fields = {}
        
        logger.info(f"Extracting from text length: {len(text)}")
        
        # Show first few lines for debugging
        lines = text.split('\n')[:10]
        logger.info(f"First lines: {[line.strip() for line in lines if line.strip()]}")
        
        for field_name, patterns in self.w2_patterns.items():
            found_value = None
            
            for pattern in patterns:
                try:
                    matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                    
                    if matches:
                        value = matches[0].strip()
                        
                        if not value or len(value) < 1:
                            continue
                        
                        # Skip template values
                        template_values = ['123456789', '123-45-6789', '12-3456789', '123456789']
                        if value in template_values:
                            logger.info(f"Skipping template value for {field_name}: {value}")
                            continue
                        
                        # Handle wage/tax amounts
                        if field_name in ['wages_tips_compensation', 'federal_income_tax_withheld', 
                                        'social_security_wages', 'social_security_tax_withheld',
                                        'medicare_wages', 'medicare_tax_withheld']:
                            
                            amount = self.clean_amount(value)
                            if amount is not None and 0 < amount <= 1000000:
                                found_value = amount
                                logger.info(f"✅ Found {field_name}: ${amount:,.2f}")
                                break
                        
                        # Handle SSN
                        elif field_name == 'employee_ssn':
                            if len(value) == 10 and value != '1234567890':
                                ssn_formatted = f"{value[:3]}-{value[3:5]}-{value[5:]}"
                                found_value = ssn_formatted
                                logger.info(f"✅ Found {field_name}: {ssn_formatted}")
                                break
                        
                        # Handle other fields
                        else:
                            cleaned = re.sub(r'[^\w\s\-\.,&\']', '', value).strip()
                            if len(cleaned) >= 2:
                                found_value = cleaned
                                logger.info(f"✅ Found {field_name}: {cleaned}")
                                break
                                
                except Exception as e:
                    logger.warning(f"Pattern error for {field_name}: {e}")
                    continue
            
            if found_value is not None:
                extracted_fields[field_name] = found_value

        # Combine first and last name
        if 'employee_name' in extracted_fields and 'employee_last_name' in extracted_fields:
            full_name = f"{extracted_fields['employee_name']} {extracted_fields['employee_last_name']}"
            extracted_fields['employee_name'] = full_name
            del extracted_fields['employee_last_name']

        # Extract tax year
        year_match = re.search(r'\b(20\d{2})\b', text)
        if year_match:
            extracted_fields['tax_year'] = int(year_match.group(1))

        return extracted_fields

    def clean_amount(self, amount_str: str) -> Optional[float]:
        """Clean and convert amount strings to float"""
        try:
            cleaned = re.sub(r'[\$,\s]', '', amount_str)
            return float(cleaned)
        except (ValueError, TypeError):
            return None

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process document focusing on real W2 data pages"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            logger.info(f"Processing {file_ext} document: {file_path}")

            if file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                text, confidence = self.extract_text_from_image(file_path)
                extracted_fields = self.extract_w2_fields(text)
                best_confidence = confidence
                
            elif file_ext == '.pdf':
                # Check each page and find the one with real W2 data
                best_data = {}
                best_confidence = 0.0
                best_page = None
                
                for page_num in range(5):  # Check first 5 pages
                    text, confidence = self.extract_text_from_pdf_page(file_path, page_num)
                    
                    if not text or len(text.strip()) < 50:
                        continue
                    
                    # Check if this page has real W2 data
                    if self.is_real_w2_page(text, page_num):
                        logger.info(f"🎯 Page {page_num + 1} contains real W2 data - processing...")
                        
                        extracted_data = self.extract_w2_fields(text)
                        
                        if len(extracted_data) > len(best_data):
                            best_data = extracted_data
                            best_confidence = confidence
                            best_page = page_num + 1
                            logger.info(f"Page {page_num + 1} is now the best page with {len(extracted_data)} fields")
                    else:
                        logger.info(f"⏭️ Skipping page {page_num + 1} - not real W2 data")
                
                extracted_fields = best_data
                
                if best_page:
                    logger.info(f"🏆 Using data from page {best_page}")
                else:
                    logger.warning("❌ No real W2 data page found")
                
            else:
                return {
                    'is_w2': False,
                    'error': f'Unsupported file type: {file_ext}',
                    'confidence': 0.0
                }

            # Check if we found meaningful data
            is_w2 = len(extracted_fields) > 1

            logger.info(f"🏁 FINAL RESULT:")
            logger.info(f"   Is W2: {is_w2}")
            logger.info(f"   Confidence: {best_confidence:.1%}")
            logger.info(f"   Fields: {len(extracted_fields)}")
            logger.info(f"   Data: {extracted_fields}")

            return {
                'is_w2': is_w2,
                'confidence': best_confidence,
                'raw_text': f"Processed real W2 data",
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
