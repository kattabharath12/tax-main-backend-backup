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
        # Improved patterns for structured W2 forms
        self.w2_patterns = {
            'employee_ssn': [
                # Look for SSN in various contexts
                r'(?i)(?:employee.*?social security number|a\s+employee.*?social security number)[\s\S]{0,200}?(\d{3}-?\d{2}-?\d{4})',
                r'(?i)social security number[\s\S]{0,100}?(\d{3}-?\d{2}-?\d{4})',
                r'(\d{10})',  # 10 consecutive digits
                r'(\d{3}-\d{2}-\d{4})',  # Formatted SSN
            ],
            'employer_ein': [
                # Look for EIN in various contexts
                r'(?i)(?:employer identification number|b\s+employer identification number|ein)[\s\S]{0,200}?([A-Z]{2,5}\d{7,10})',
                r'(?i)identification number[\s\S]{0,100}?([A-Z]{2,5}\d{7,10})',
                r'(?i)identification number[\s\S]{0,100}?(\d{2}-\d{7})',
                r'([A-Z]{2,5}\d{7,10})',  # Pattern like FGHU7896901
                r'(\d{2}-\d{7})',  # Standard EIN format
            ],
            'employer_name': [
                # Look for employer name in various contexts
                r'(?i)(?:c\s+employer.*?name|employer.*?name.*?address)[\s\S]{0,200}?([A-Z][A-Za-z\s,\.&\-\']{2,50})(?=\s*,|\s*\d|\s*\n)',
                r'(?i)employer.*?name[\s\S]{0,100}?([A-Z][A-Za-z\s,\.&\-\']{2,30})',
                r'([A-Z][A-Za-z\s&]{2,20})(?=\s*,\s*\d)',  # Name before address
            ],
            'employee_name': [
                # Look for employee name in various contexts
                r'(?i)(?:e\s+employee.*?first name|employee.*?first name.*?initial)[\s\S]{0,200}?([A-Z][A-Za-z\s\-\'\.]{2,30})',
                r'(?i)employee.*?first name[\s\S]{0,100}?([A-Z][A-Za-z\s\-\'\.]{2,30})',
                r'(?i)first name.*?initial[\s\S]{0,100}?([A-Z][A-Za-z\s\-\'\.]{2,30})',
            ],
            'employee_last_name': [
                r'(?i)last name[\s\S]{0,50}?([A-Z][A-Za-z\-\']{2,25})',
            ],
            'employee_address': [
                r'(?i)(?:f\s+employee.*?address|employee.*?address.*?zip)[\s\S]{0,200}?([0-9][0-9A-Za-z\s,\.\-]{10,80})',
            ],
            'employer_address': [
                r'(?i)(?:c\s+employer.*?address|employer.*?name.*?address)[\s\S]{0,200}?([0-9][0-9A-Za-z\s,\.\-]{10,80})',
            ],
            'control_number': [
                r'(?i)(?:d\s+control number|control number)[\s\S]{0,100}?([A-Z0-9]{5,20})',
            ],
        }
        
        # Specific patterns for wage boxes
        self.wage_patterns = {
            'wages_tips_compensation': [
                r'(?i)1\s+wages.*?tips.*?other.*?compensation[\s\S]{0,100}?(\d{1,7}(?:,\d{3})*(?:\.\d{2})?)',
                r'(?i)wages.*?tips.*?other.*?compensation[\s\S]{0,100}?(\d{1,7}(?:,\d{3})*)',
                r'1[\s\S]{0,50}?(\d{4,7})',  # Box 1 with 4+ digit number
            ],
            'federal_income_tax_withheld': [
                r'(?i)2\s+federal.*?income.*?tax.*?withheld[\s\S]{0,100}?(\d{1,6}(?:,\d{3})*(?:\.\d{2})?)',
                r'(?i)federal.*?income.*?tax.*?withheld[\s\S]{0,100}?(\d{1,6}(?:,\d{3})*)',
                r'2[\s\S]{0,50}?(\d{2,6})',  # Box 2 with 2+ digit number
            ],
            'social_security_wages': [
                r'(?i)3\s+social.*?security.*?wages[\s\S]{0,100}?(\d{1,7}(?:,\d{3})*(?:\.\d{2})?)',
                r'(?i)social.*?security.*?wages[\s\S]{0,100}?(\d{1,7}(?:,\d{3})*)',
                r'3[\s\S]{0,50}?(\d{2,7})',  # Box 3 with 2+ digit number
            ],
            'social_security_tax_withheld': [
                r'(?i)4\s+social.*?security.*?tax.*?withheld[\s\S]{0,100}?(\d{1,6}(?:,\d{3})*(?:\.\d{2})?)',
                r'(?i)social.*?security.*?tax.*?withheld[\s\S]{0,100}?(\d{1,6}(?:,\d{3})*)',
                r'4[\s\S]{0,50}?(\d{2,6})',  # Box 4 with 2+ digit number
            ],
            'medicare_wages': [
                r'(?i)5\s+medicare.*?wages.*?tips[\s\S]{0,100}?(\d{1,7}(?:,\d{3})*(?:\.\d{2})?)',
                r'(?i)medicare.*?wages.*?tips[\s\S]{0,100}?(\d{1,7}(?:,\d{3})*)',
                r'5[\s\S]{0,50}?(\d{2,7})',  # Box 5 with 2+ digit number
            ],
            'medicare_tax_withheld': [
                r'(?i)6\s+medicare.*?tax.*?withheld[\s\S]{0,100}?(\d{1,6}(?:,\d{3})*(?:\.\d{2})?)',
                r'(?i)medicare.*?tax.*?withheld[\s\S]{0,100}?(\d{1,6}(?:,\d{3})*)',
                r'6[\s\S]{0,50}?(\d{2,6})',  # Box 6 with 2+ digit number
            ],
        }

    def extract_text_from_pdf_page(self, pdf_path: str, page_num: int) -> Tuple[str, float]:
        """Extract text from PDF page with enhanced handling"""
        try:
            # Try pdfplumber first
            with pdfplumber.open(pdf_path) as pdf:
                if page_num < len(pdf.pages):
                    page = pdf.pages[page_num]
                    text = page.extract_text()
                    if text and len(text.strip()) > 50:
                        logger.info(f"Page {page_num + 1} pdfplumber: {len(text)} chars")
                        return text, 0.9
            
            # Fallback to OCR with higher DPI for better accuracy
            with tempfile.TemporaryDirectory() as temp_dir:
                images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1, dpi=400)
                if images:
                    temp_image_path = os.path.join(temp_dir, f"page_{page_num}.png")
                    images[0].save(temp_image_path, 'PNG')
                    text, confidence = self.extract_text_from_image(temp_image_path)
                    logger.info(f"Page {page_num + 1} OCR: {len(text)} chars, {confidence:.1%} confidence")
                    return text, confidence
                    
        except Exception as e:
            logger.error(f"Error extracting from page {page_num}: {e}")
            
        return "", 0.0

    def extract_text_from_image(self, image_path: str) -> Tuple[str, float]:
        """Enhanced OCR with multiple preprocessing approaches"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return "", 0.0
            
            # Try multiple preprocessing approaches
            preprocessed_images = []
            
            # 1. Standard approach
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            preprocessed_images.append(("standard", thresh1))
            
            # 2. Adaptive threshold
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            preprocessed_images.append(("adaptive", adaptive))
            
            # 3. Morphological operations
            kernel = np.ones((1, 1), np.uint8)
            morph = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
            preprocessed_images.append(("morphological", morph))
            
            best_text = ""
            best_confidence = 0.0
            
            for approach_name, processed_img in preprocessed_images:
                temp_path = image_path.replace('.', f'_{approach_name}.')
                cv2.imwrite(temp_path, processed_img)
                
                # Try multiple OCR configurations
                configs = [
                    '--oem 3 --psm 6',  # Uniform block
                    '--oem 3 --psm 4',  # Single column
                    '--oem 3 --psm 3',  # Fully automatic
                ]
                
                for config in configs:
                    try:
                        text = pytesseract.image_to_string(temp_path, config=config)
                        data = pytesseract.image_to_data(temp_path, config=config, output_type=pytesseract.Output.DICT)
                        
                        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                        
                        if avg_confidence > best_confidence and text.strip():
                            best_text = text
                            best_confidence = avg_confidence
                            logger.info(f"Best result from {approach_name} + {config}: {avg_confidence:.1f}%")
                            
                    except Exception:
                        continue
                
                # Cleanup
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
            return best_text, best_confidence / 100.0
            
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return "", 0.0

    def parse_structured_data(self, text: str) -> Dict[str, Any]:
        """Parse structured W2 data with positional awareness"""
        extracted = {}
        
        # Clean and normalize text
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        full_text = ' '.join(lines)
        
        logger.info(f"Parsing structured data from {len(lines)} lines")
        
        # Extract using both personal info patterns and wage patterns
        all_patterns = {**self.w2_patterns, **self.wage_patterns}
        
        for field_name, patterns in all_patterns.items():
            found_value = None
            
            for pattern in patterns:
                try:
                    matches = re.findall(pattern, full_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                    
                    if matches:
                        value = matches[0].strip()
                        
                        if not value or len(value) < 1:
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
                            ssn_clean = re.sub(r'[^\d]', '', value)
                            if len(ssn_clean) >= 9:
                                ssn_formatted = f"{ssn_clean[:3]}-{ssn_clean[3:5]}-{ssn_clean[5:9]}"
                                found_value = ssn_formatted
                                logger.info(f"✅ Found {field_name}: {ssn_formatted}")
                                break
                        
                        # Handle EIN
                        elif field_name == 'employer_ein':
                            found_value = value
                            logger.info(f"✅ Found {field_name}: {value}")
                            break
                        
                        # Handle names and addresses
                        else:
                            # Clean text - remove common OCR artifacts
                            cleaned = re.sub(r'[^\w\s\-\.,&\']', '', value).strip()
                            if len(cleaned) >= 2 and not cleaned.lower() in ['and zip code', 'zip code', 'address']:
                                found_value = cleaned
                                logger.info(f"✅ Found {field_name}: {cleaned}")
                                break
                                
                except Exception as e:
                    logger.warning(f"Pattern error for {field_name}: {e}")
                    continue
            
            if found_value is not None:
                extracted[field_name] = found_value

        # Combine first and last name if both found
        if 'employee_name' in extracted and 'employee_last_name' in extracted:
            full_name = f"{extracted['employee_name']} {extracted['employee_last_name']}"
            extracted['employee_name'] = full_name
            del extracted['employee_last_name']

        # Extract tax year
        year_match = re.search(r'\b(20\d{2})\b', full_text)
        if year_match:
            extracted['tax_year'] = int(year_match.group(1))
            logger.info(f"✅ Found tax_year: {year_match.group(1)}")

        return extracted

    def clean_amount(self, amount_str: str) -> Optional[float]:
        """Clean and convert amount strings to float"""
        try:
            cleaned = re.sub(r'[\$,\s]', '', amount_str)
            return float(cleaned)
        except (ValueError, TypeError):
            return None

    def is_w2_document(self, text: str) -> bool:
        """Check if document is a W2"""
        text_lower = text.lower()
        w2_indicators = [
            'w-2', 'wage and tax statement', 'social security', 'medicare', 
            'federal income tax', 'employer identification'
        ]
        return any(indicator in text_lower for indicator in w2_indicators)

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process document with enhanced structured parsing"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            logger.info(f"Processing {file_ext} document: {file_path}")

            if file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                text, confidence = self.extract_text_from_image(file_path)
                all_text = text
                best_confidence = confidence
                
            elif file_ext == '.pdf':
                # Process first page primarily, but check others if needed
                text, confidence = self.extract_text_from_pdf_page(file_path, 0)  # First page
                all_text = text
                best_confidence = confidence
                
                # If first page doesn't have enough data, try others
                if len(text.strip()) < 100:
                    for page_num in range(1, 3):
                        text2, conf2 = self.extract_text_from_pdf_page(file_path, page_num)
                        if len(text2.strip()) > len(all_text.strip()):
                            all_text = text2
                            best_confidence = conf2
                            logger.info(f"Using page {page_num + 1} as primary")
                
            else:
                return {
                    'is_w2': False,
                    'error': f'Unsupported file type: {file_ext}',
                    'confidence': 0.0
                }

            if not all_text or len(all_text.strip()) < 50:
                return {
                    'is_w2': False,
                    'error': 'No sufficient text extracted',
                    'confidence': 0.0,
                    'raw_text': '',
                    'extracted_fields': {}
                }

            # Check if W2
            is_w2 = self.is_w2_document(all_text)
            
            if not is_w2:
                return {
                    'is_w2': False,
                    'error': 'Document does not appear to be a W2',
                    'confidence': best_confidence,
                    'raw_text': all_text[:500],
                    'extracted_fields': {}
                }

            # Extract structured data
            extracted_fields = self.parse_structured_data(all_text)

            logger.info(f"🏁 FINAL EXTRACTION RESULT:")
            logger.info(f"   Fields found: {len(extracted_fields)}")
            logger.info(f"   Data: {extracted_fields}")

            return {
                'is_w2': True,
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
