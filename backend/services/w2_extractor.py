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
        # Enhanced patterns with more specific targeting
        self.w2_patterns = {
            'employee_ssn': [
                r'(?i)a\s+employee.*?social security number[\s\n]*(\d{3,4}-?\d{2}-?\d{4})',
                r'(?i)employee.*?social security number[\s\n]*(\d{3,4}-?\d{2}-?\d{4})',
                r'(\d{10})',  # 10 digits without hyphens
                r'(\d{3}-\d{2}-\d{4})',  # Standard SSN format
            ],
            'employer_ein': [
                r'(?i)b\s+employer identification number.*?[\s\n]*([A-Z]{2,4}\d{7,10})',
                r'(?i)employer identification number.*?[\s\n]*([A-Z]{2,4}\d{7,10})',
                r'(?i)ein.*?[\s\n]*([A-Z]{2,4}\d{7,10})',
                r'(\d{2}-\d{7})',  # Standard EIN format
            ],
            'employer_name': [
                r'(?i)c\s+employer.*?name.*?address.*?zip.*?[\s\n]+([A-Z][A-Za-z\s,\.&\-\']+?)(?:,|\n|[\d])',
                r'(?i)employer.*?name.*?[\s\n]+([A-Z][A-Za-z\s,\.&\-\']{3,30})',
            ],
            'employee_name': [
                r'(?i)e\s+employee.*?first name.*?initial.*?[\s\n]+([A-Z][A-Za-z\s\-\'\.]+?)(?:\s+Suff\.|\n|Last name)',
                r'(?i)employee.*?first name.*?[\s\n]+([A-Z][A-Za-z\s\-\'\.]+)',
            ],
            'employee_last_name': [
                r'(?i)Last name.*?[\s\n]+([A-Z][A-Za-z\-\']+)',
                r'(?i)e\s+employee.*?first name.*?initial.*?Last name.*?[\s\n]+([A-Z][A-Za-z\-\']+)',
            ],
            'control_number': [
                r'(?i)d\s+control number[\s\n]*([A-Z0-9]{5,15})',
                r'(?i)control number[\s\n]*([A-Z0-9]{5,15})',
            ],
            'wages_tips_compensation': [
                r'(?i)1\s+wages.*?tips.*?other compensation[\s\n]*(\d{1,6}(?:,\d{3})*(?:\.\d{2})?)',
                r'(?i)box\s*1.*?wages.*?[\s\n]*(\d{1,6}(?:,\d{3})*(?:\.\d{2})?)',
                r'1\s+wages.*?tips.*?other compensation[\s\n]*(\d{1,6}(?:,\d{3})*)',
            ],
            'federal_income_tax_withheld': [
                r'(?i)2\s+federal income tax withheld[\s\n]*(\d{1,6}(?:,\d{3})*(?:\.\d{2})?)',
                r'(?i)box\s*2.*?federal.*?tax.*?[\s\n]*(\d{1,6}(?:,\d{3})*(?:\.\d{2})?)',
                r'2\s+federal income tax withheld[\s\n]*(\d{1,6}(?:,\d{3})*)',
            ],
            'social_security_wages': [
                r'(?i)3\s+social security wages[\s\n]*(\d{1,6}(?:,\d{3})*(?:\.\d{2})?)',
                r'(?i)box\s*3.*?social security wages[\s\n]*(\d{1,6}(?:,\d{3})*(?:\.\d{2})?)',
                r'3\s+social security wages[\s\n]*(\d{1,6}(?:,\d{3})*)',
            ],
            'social_security_tax_withheld': [
                r'(?i)4\s+social security tax withheld[\s\n]*(\d{1,6}(?:,\d{3})*(?:\.\d{2})?)',
                r'(?i)box\s*4.*?social security tax[\s\n]*(\d{1,6}(?:,\d{3})*(?:\.\d{2})?)',
                r'4\s+social security tax withheld[\s\n]*(\d{1,6}(?:,\d{3})*)',
            ],
            'medicare_wages': [
                r'(?i)5\s+medicare wages and tips[\s\n]*(\d{1,6}(?:,\d{3})*(?:\.\d{2})?)',
                r'(?i)box\s*5.*?medicare wages[\s\n]*(\d{1,6}(?:,\d{3})*(?:\.\d{2})?)',
                r'5\s+medicare wages and tips[\s\n]*(\d{1,6}(?:,\d{3})*)',
            ],
            'medicare_tax_withheld': [
                r'(?i)6\s+medicare tax withheld[\s\n]*(\d{1,6}(?:,\d{3})*(?:\.\d{2})?)',
                r'(?i)box\s*6.*?medicare tax[\s\n]*(\d{1,6}(?:,\d{3})*(?:\.\d{2})?)',
                r'6\s+medicare tax withheld[\s\n]*(\d{1,6}(?:,\d{3})*)',
            ],
        }

        # W2 detection keywords
        self.w2_indicators = [
            "wage and tax statement",
            "form w-2",
            "w-2",
            "employer identification number",
            "wages, tips, other compensation",
            "federal income tax withheld",
            "social security wages",
            "medicare wages",
            "copy a",
            "copy b",
            "copy c",
            "department of the treasury",
            "internal revenue service",
            "box 1",
            "box 2",
            "box 3"
        ]

    def extract_text_from_pdf_pages(self, pdf_path: str) -> List[Tuple[str, float]]:
        """Extract text from each page of PDF separately"""
        pages_text = []
        
        try:
            # Try direct text extraction first
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text and len(page_text.strip()) > 50:
                        pages_text.append((page_text, 0.9))
                        logger.info(f"Extracted text from page {page_num + 1} via pdfplumber")
                    else:
                        # Fallback to OCR for this page
                        try:
                            with tempfile.TemporaryDirectory() as temp_dir:
                                images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1, dpi=300)
                                if images:
                                    temp_image_path = os.path.join(temp_dir, f"page_{page_num}.png")
                                    images[0].save(temp_image_path, 'PNG')
                                    page_text, confidence = self.extract_text_from_image(temp_image_path)
                                    pages_text.append((page_text, confidence))
                                    logger.info(f"Extracted text from page {page_num + 1} via OCR")
                        except Exception as e:
                            logger.error(f"Error processing page {page_num + 1}: {e}")
                            pages_text.append(("", 0.0))
                            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            
        return pages_text

    def find_w2_pages(self, pages_text: List[Tuple[str, float]]) -> List[int]:
        """Find which pages contain actual W2 data (not instructions)"""
        w2_pages = []
        
        for i, (text, confidence) in enumerate(pages_text):
            if not text:
                continue
                
            text_lower = text.lower()
            
            # Skip instruction pages
            instruction_indicators = [
                "future developments",
                "notice to employee",
                "instructions for employee",
                "do you have to file",
                "earned income tax credit",
                "corrections",
                "clergy and religious workers",
                "employers, please note",
                "due dates",
                "e-filing"
            ]
            
            is_instruction_page = any(indicator in text_lower for indicator in instruction_indicators)
            if is_instruction_page:
                logger.info(f"Page {i + 1} appears to be instructions, skipping")
                continue
            
            # Look for actual W2 data indicators
            data_indicators = [
                "employee's social security number",
                "employer identification number",
                "wages, tips, other compensation",
                "federal income tax withheld",
                "social security wages",
                "medicare wages"
            ]
            
            # Also look for numeric patterns that suggest actual data
            has_numeric_data = bool(re.search(r'\d{1,6}(?:,\d{3})*(?:\.\d{2})?', text))
            has_ssn_pattern = bool(re.search(r'\d{3,4}-?\d{2}-?\d{4}', text))
            has_ein_pattern = bool(re.search(r'[A-Z]{2,4}\d{7,10}|\d{2}-\d{7}', text))
            
            data_score = sum(1 for indicator in data_indicators if indicator in text_lower)
            
            if (data_score >= 3) or (has_numeric_data and has_ssn_pattern) or (has_numeric_data and has_ein_pattern):
                w2_pages.append(i)
                logger.info(f"Page {i + 1} appears to contain W2 data (score: {data_score})")
        
        return w2_pages

    def enhance_image(self, image_path: str) -> str:
        """Enhanced image preprocessing"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return image_path

            # Convert to PIL for enhancement
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # Enhance contrast and sharpness
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(1.5)
            enhancer = ImageEnhance.Sharpness(pil_img)
            pil_img = enhancer.enhance(2.0)

            # Convert back to CV2
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply adaptive threshold
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            # Save enhanced image
            temp_path = image_path.replace('.', '_enhanced.')
            cv2.imwrite(temp_path, thresh)
            return temp_path
            
        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            return image_path

    def extract_text_from_image(self, image_path: str) -> Tuple[str, float]:
        """Extract text from image with multiple configurations"""
        try:
            enhanced_path = self.enhance_image(image_path)
            
            # Try multiple configurations
            configs = [
                '--oem 3 --psm 6',
                '--oem 3 --psm 4',
                '--oem 3 --psm 3',
                '--oem 3 --psm 11',
                '--oem 3 --psm 12'
            ]
            
            best_text = ""
            best_confidence = 0.0
            
            for config in configs:
                try:
                    data = pytesseract.image_to_data(enhanced_path, config=config, output_type=pytesseract.Output.DICT)
                    text = pytesseract.image_to_string(enhanced_path, config=config)
                    
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    if avg_confidence > best_confidence and text.strip():
                        best_text = text
                        best_confidence = avg_confidence
                        
                except Exception as e:
                    logger.warning(f"Config {config} failed: {e}")
                    continue
            
            # Clean up
            if enhanced_path != image_path and os.path.exists(enhanced_path):
                os.remove(enhanced_path)
                
            return best_text, best_confidence / 100.0
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return "", 0.0

    def is_w2_document(self, text: str) -> bool:
        """Check if document is W2"""
        text_lower = text.lower()
        
        strong_indicators = ["form w-2", "wage and tax statement"]
        box_indicators = ["box 1", "box 2", "box 3"]
        
        matches = 0
        for indicator in strong_indicators:
            if indicator in text_lower:
                matches += 3
                
        for indicator in box_indicators:
            if indicator in text_lower:
                matches += 2
                
        for indicator in self.w2_indicators:
            if indicator not in strong_indicators and indicator not in box_indicators:
                if indicator in text_lower:
                    matches += 1
        
        return matches >= 4

    def clean_numeric_value(self, value: str) -> Optional[float]:
        """Clean and convert numeric values"""
        try:
            # Remove non-numeric characters except dots and commas
            cleaned = re.sub(r'[^\d.,]', '', value)
            cleaned = cleaned.replace(',', '')
            return float(cleaned)
        except (ValueError, TypeError):
            return None

    def extract_w2_fields(self, text: str) -> Dict[str, Any]:
        """Extract W2 fields with improved pattern matching"""
        extracted_fields = {}
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        for field_name, patterns in self.w2_patterns.items():
            for pattern in patterns:
                try:
                    matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
                    
                    for match in matches:
                        value = match.group(1).strip()
                        
                        if len(value) < 1:
                            continue
                        
                        # Handle numeric fields
                        if field_name in ['wages_tips_compensation', 'federal_income_tax_withheld', 
                                        'social_security_wages', 'social_security_tax_withheld',
                                        'medicare_wages', 'medicare_tax_withheld']:
                            numeric_value = self.clean_numeric_value(value)
                            if numeric_value is not None and 0 <= numeric_value <= 999999:
                                extracted_fields[field_name] = numeric_value
                                logger.info(f"Extracted {field_name}: {numeric_value}")
                                break
                        
                        # Handle SSN
                        elif field_name == 'employee_ssn':
                            # Format SSN properly
                            ssn = re.sub(r'[^\d]', '', value)
                            if len(ssn) >= 9:
                                formatted_ssn = f"{ssn[:3]}-{ssn[3:5]}-{ssn[5:9]}"
                                extracted_fields[field_name] = formatted_ssn
                                logger.info(f"Extracted {field_name}: {formatted_ssn}")
                                break
                        
                        # Handle text fields
                        else:
                            if field_name in ['employer_name', 'employee_name']:
                                # Clean up name fields
                                value = re.sub(r'^[^A-Za-z]*', '', value)
                                value = re.sub(r'[^A-Za-z\s\-\.\&\']+.*$', '', value)
                                if len(value) >= 2:
                                    extracted_fields[field_name] = value
                                    logger.info(f"Extracted {field_name}: {value}")
                                    break
                            else:
                                extracted_fields[field_name] = value
                                logger.info(f"Extracted {field_name}: {value}")
                                break
                                
                except Exception as e:
                    logger.warning(f"Error processing pattern for {field_name}: {e}")
                    continue

        # Combine first and last name if available
        if 'employee_name' in extracted_fields and 'employee_last_name' in extracted_fields:
            full_name = f"{extracted_fields['employee_name']} {extracted_fields['employee_last_name']}"
            extracted_fields['employee_name'] = full_name
            del extracted_fields['employee_last_name']

        # Extract tax year
        year_patterns = [r'(20\d{2})', r'(?i)(?:tax\s*year|year)\s*(\d{4})']
        for pattern in year_patterns:
            match = re.search(pattern, text)
            if match:
                year = int(match.group(1))
                if 2015 <= year <= 2025:
                    extracted_fields['tax_year'] = year
                    break

        return extracted_fields

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process document with enhanced multi-page handling"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            logger.info(f"Processing document: {file_path} (type: {file_ext})")

            if file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                text, confidence = self.extract_text_from_image(file_path)
                pages_text = [(text, confidence)]
                w2_pages = [0] if self.is_w2_document(text) else []
            elif file_ext == '.pdf':
                pages_text = self.extract_text_from_pdf_pages(file_path)
                w2_pages = self.find_w2_pages(pages_text)
            else:
                return {
                    'is_w2': False,
                    'error': f'Unsupported file type: {file_ext}',
                    'confidence': 0.0,
                    'raw_text': '',
                    'extracted_fields': {}
                }

            if not w2_pages:
                logger.info("No W2 data pages found")
                return {
                    'is_w2': False,
                    'error': 'No W2 data found in document',
                    'confidence': 0.0,
                    'raw_text': '\n'.join([text for text, _ in pages_text]),
                    'extracted_fields': {}
                }

            # Process the first W2 page found
            w2_page_idx = w2_pages[0]
            w2_text, w2_confidence = pages_text[w2_page_idx]
            
            logger.info(f"Processing W2 data from page {w2_page_idx + 1}")
            
            extracted_fields = self.extract_w2_fields(w2_text)

            result = {
                'is_w2': True,
                'confidence': w2_confidence,
                'raw_text': w2_text,
                'extracted_fields': extracted_fields,
                'error': None,
                'page_processed': w2_page_idx + 1
            }

            logger.info(f"Successfully processed W2, extracted {len(extracted_fields)} fields")
            return result

        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return {
                'is_w2': False,
                'error': str(e),
                'confidence': 0.0,
                'raw_text': '',
                'extracted_fields': {}
            }
