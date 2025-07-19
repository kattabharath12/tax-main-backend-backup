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
        # More specific patterns targeting actual filled forms
        self.w2_patterns = {
            'employee_ssn': [
                # Look for patterns near "Employee's social security number" but not template data
                r'(?i)(?:employee.*?social security number|a\s+employee.*?social security number)[\s\S]{0,100}?(\d{10}|\d{4}[\s-]\d{2}[\s-]\d{4})',
                r'(\d{10})(?=\s)',  # 10 consecutive digits
                r'(\d{3,4}-\d{2}-\d{4})(?!\s*(?:VOID|void))',  # SSN format, not marked as VOID
            ],
            'employer_ein': [
                # Look for EIN patterns that aren't template examples
                r'(?i)(?:employer identification number|b\s+employer identification number)[\s\S]{0,100}?([A-Z]{2,5}\d{7,10})',
                r'([A-Z]{2,5}\d{7,10})(?=\s)',  # Letters followed by numbers
                r'(\d{2}-\d{7})(?!\s*(?:12-3456789))',  # Standard EIN but not the template
            ],
            'employer_name': [
                # Look for employer name that's not template data
                r'(?i)(?:employer.*?name.*?address.*?zip|c\s+employer.*?name)[\s\n]+([A-Z][A-Za-z\s,\.&\-\']{2,40})(?=\s*,|\s*\d|\s*\n)',
                r'(?<!ABC Corp)([A-Z]{2,}[A-Za-z\s]{2,20})(?=\s*,\s*\d)',  # Name before address numbers
            ],
            'employee_name': [
                # Look for employee name sections
                r'(?i)(?:employee.*?first name.*?initial|e\s+employee.*?first name)[\s\n]+([A-Z][A-Za-z\s\-\'\.]{2,25})',
                r'(?i)first name.*?initial[\s\n]+([A-Z][A-Za-z\s\-\'\.]{2,25})',
            ],
            'employee_last_name': [
                r'(?i)last name[\s\n]+([A-Z][A-Za-z\-\']{2,25})',
                r'(?i)(?:last name|Last name)[\s\n]*([A-Z][A-Z a-z\-\']{2,25})',
            ],
            'employee_address': [
                r'(?i)(?:employee.*?address.*?zip|f\s+employee.*?address)[\s\n]+([0-9][0-9A-Za-z\s,\.\-]{10,50})',
            ],
            'control_number': [
                r'(?i)(?:control number|d\s+control number)[\s\n]*([A-Z0-9]{5,20})',
            ],
            # Numeric fields - look for actual numbers, not zeros
            'wages_tips_compensation': [
                r'(?i)1\s+wages.*?tips.*?other compensation[\s\n]*([1-9]\d{1,5}(?:,?\d{3})*(?:\.\d{2})?)',
                r'(?i)wages.*?tips.*?other compensation[\s\n]*([1-9]\d{1,5}(?:,?\d{3})*)',
                r'1[\s]*wages[\s\S]{0,50}?([1-9]\d{1,5})',
            ],
            'federal_income_tax_withheld': [
                r'(?i)2\s+federal income tax withheld[\s\n]*([1-9]\d{0,5}(?:,?\d{3})*(?:\.\d{2})?)',
                r'(?i)federal income tax withheld[\s\n]*([1-9]\d{0,5}(?:,?\d{3})*)',
                r'2[\s]*federal[\s\S]{0,50}?([1-9]\d{0,5})',
            ],
            'social_security_wages': [
                r'(?i)3\s+social security wages[\s\n]*([1-9]\d{0,5}(?:,?\d{3})*(?:\.\d{2})?)',
                r'(?i)social security wages[\s\n]*([1-9]\d{0,5}(?:,?\d{3})*)',
                r'3[\s]*social security wages[\s\S]{0,50}?([1-9]\d{0,5})',
            ],
            'social_security_tax_withheld': [
                r'(?i)4\s+social security tax withheld[\s\n]*([1-9]\d{0,5}(?:,?\d{3})*(?:\.\d{2})?)',
                r'(?i)social security tax withheld[\s\n]*([1-9]\d{0,5}(?:,?\d{3})*)',
                r'4[\s]*social security tax[\s\S]{0,50}?([1-9]\d{0,5})',
            ],
            'medicare_wages': [
                r'(?i)5\s+medicare wages and tips[\s\n]*([1-9]\d{0,5}(?:,?\d{3})*(?:\.\d{2})?)',
                r'(?i)medicare wages and tips[\s\n]*([1-9]\d{0,5}(?:,?\d{3})*)',
                r'5[\s]*medicare wages[\s\S]{0,50}?([1-9]\d{0,5})',
            ],
            'medicare_tax_withheld': [
                r'(?i)6\s+medicare tax withheld[\s\n]*([1-9]\d{0,5}(?:,?\d{3})*(?:\.\d{2})?)',
                r'(?i)medicare tax withheld[\s\n]*([1-9]\d{0,5}(?:,?\d{3})*)',
                r'6[\s]*medicare tax[\s\S]{0,50}?([1-9]\d{0,5})',
            ],
        }

        self.w2_indicators = [
            "wage and tax statement",
            "form w-2",
            "w-2",
            "employer identification number",
            "wages, tips, other compensation",
            "federal income tax withheld",
            "social security wages",
            "medicare wages"
        ]

    def extract_text_from_pdf_pages(self, pdf_path: str) -> List[Tuple[str, float, int]]:
        """Extract text from each page with page numbers"""
        pages_text = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text and len(page_text.strip()) > 50:
                        pages_text.append((page_text, 0.9, page_num + 1))
                        logger.info(f"Extracted {len(page_text)} chars from page {page_num + 1}")
                    else:
                        # Use OCR for pages with little text
                        try:
                            with tempfile.TemporaryDirectory() as temp_dir:
                                images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1, dpi=300)
                                if images:
                                    temp_image_path = os.path.join(temp_dir, f"page_{page_num}.png")
                                    images[0].save(temp_image_path, 'PNG')
                                    page_text, confidence = self.extract_text_from_image(temp_image_path)
                                    pages_text.append((page_text, confidence, page_num + 1))
                        except Exception as e:
                            logger.error(f"OCR failed for page {page_num + 1}: {e}")
                            pages_text.append(("", 0.0, page_num + 1))
                            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            
        return pages_text

    def find_best_w2_page(self, pages_text: List[Tuple[str, float, int]]) -> Optional[Tuple[str, float, int]]:
        """Find the page with the most actual W2 data (not template/instructions)"""
        best_page = None
        best_score = 0
        
        for text, confidence, page_num in pages_text:
            if not text:
                continue
                
            text_lower = text.lower()
            
            # Skip obvious instruction/template pages
            skip_indicators = [
                "future developments",
                "notice to employee", 
                "instructions for employee",
                "do you have to file",
                "corrections",
                "employers, please note",
                "void",  # Template pages often have VOID
                "attention:",
                "copy a of this form is provided for informational purposes"
            ]
            
            if any(indicator in text_lower for indicator in skip_indicators):
                logger.info(f"Skipping page {page_num} - appears to be instructions/template")
                continue
            
            # Score based on actual data presence
            score = 0
            
            # Look for filled numeric data (not just zeros)
            actual_numbers = re.findall(r'[1-9]\d{2,5}', text)  # Numbers 100+ (likely wages)
            score += len(actual_numbers) * 3
            
            # Look for realistic SSN patterns (not template 123-45-6789)
            realistic_ssns = re.findall(r'(?!123-45-6789)\d{3,4}-?\d{2}-?\d{4}', text)
            score += len(realistic_ssns) * 5
            
            # Look for realistic EIN patterns (not template 12-3456789)  
            realistic_eins = re.findall(r'(?!12-3456789)[A-Z]{2,5}\d{7,10}|\d{2}-\d{7}', text)
            score += len(realistic_eins) * 5
            
            # Look for actual names (patterns that aren't common template text)
            potential_names = re.findall(r'[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}', text)
            real_names = [name for name in potential_names 
                         if name not in ['JOHN DOE', 'ABC Corp', 'New York', 'Main St']]
            score += len(real_names) * 2
            
            # Bonus for W2 structure without template markers
            if 'form w-2' in text_lower and 'void' not in text_lower:
                score += 10
                
            logger.info(f"Page {page_num} score: {score} (numbers: {len(actual_numbers)}, SSNs: {len(realistic_ssns)}, EINs: {len(realistic_eins)}, names: {len(real_names)})")
            
            if score > best_score:
                best_score = score
                best_page = (text, confidence, page_num)
        
        if best_page:
            logger.info(f"Selected page {best_page[2]} as best W2 page (score: {best_score})")
        else:
            logger.warning("No suitable W2 page found")
            
        return best_page

    def enhance_image(self, image_path: str) -> str:
        """Enhanced image preprocessing"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return image_path

            # Convert to PIL for enhancement
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(1.5)
            enhancer = ImageEnhance.Sharpness(pil_img)
            pil_img = enhancer.enhance(2.0)

            # Convert back to CV2 and apply threshold
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            temp_path = image_path.replace('.', '_enhanced.')
            cv2.imwrite(temp_path, thresh)
            return temp_path
            
        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            return image_path

    def extract_text_from_image(self, image_path: str) -> Tuple[str, float]:
        """Extract text from image"""
        try:
            enhanced_path = self.enhance_image(image_path)
            
            configs = [
                '--oem 3 --psm 6',
                '--oem 3 --psm 4', 
                '--oem 3 --psm 3'
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
                    continue
            
            if enhanced_path != image_path and os.path.exists(enhanced_path):
                os.remove(enhanced_path)
                
            return best_text, best_confidence / 100.0
            
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return "", 0.0

    def is_w2_document(self, text: str) -> bool:
        """Check if document is W2"""
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in self.w2_indicators)

    def clean_numeric_value(self, value: str) -> Optional[float]:
        """Clean numeric values"""
        try:
            cleaned = re.sub(r'[^\d.,]', '', value)
            cleaned = cleaned.replace(',', '')
            return float(cleaned)
        except:
            return None

    def extract_w2_fields(self, text: str) -> Dict[str, Any]:
        """Extract W2 fields with improved targeting"""
        extracted_fields = {}
        
        # Log the text being processed for debugging
        logger.info(f"Processing text snippet: {text[:200]}...")
        
        for field_name, patterns in self.w2_patterns.items():
            found_value = None
            
            for pattern in patterns:
                try:
                    matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL))
                    
                    for match in matches:
                        value = match.group(1).strip()
                        
                        if not value or len(value) < 1:
                            continue
                        
                        # Skip obvious template/example values
                        template_values = ['123-45-6789', '12-3456789', 'JOHN DOE', 'ABC Corp', '55000', '4800']
                        if value in template_values:
                            logger.info(f"Skipping template value for {field_name}: {value}")
                            continue
                        
                        # Process numeric fields
                        if field_name in ['wages_tips_compensation', 'federal_income_tax_withheld', 
                                        'social_security_wages', 'social_security_tax_withheld',
                                        'medicare_wages', 'medicare_tax_withheld']:
                            numeric_value = self.clean_numeric_value(value)
                            if numeric_value is not None and 0 < numeric_value <= 999999:
                                found_value = numeric_value
                                logger.info(f"Found {field_name}: {numeric_value}")
                                break
                        
                        # Process SSN
                        elif field_name == 'employee_ssn':
                            ssn = re.sub(r'[^\d]', '', value)
                            if len(ssn) >= 9 and ssn != '123456789':  # Not template SSN
                                formatted_ssn = f"{ssn[:3]}-{ssn[3:5]}-{ssn[5:9]}"
                                found_value = formatted_ssn
                                logger.info(f"Found {field_name}: {formatted_ssn}")
                                break
                        
                        # Process other fields
                        else:
                            if len(value) >= 2:
                                found_value = value
                                logger.info(f"Found {field_name}: {value}")
                                break
                                
                except Exception as e:
                    logger.warning(f"Pattern error for {field_name}: {e}")
                    continue
            
            if found_value is not None:
                extracted_fields[field_name] = found_value

        # Combine names if both first and last found
        if 'employee_name' in extracted_fields and 'employee_last_name' in extracted_fields:
            full_name = f"{extracted_fields['employee_name']} {extracted_fields['employee_last_name']}"
            extracted_fields['employee_name'] = full_name
            del extracted_fields['employee_last_name']

        # Extract tax year
        year_match = re.search(r'(20\d{2})', text)
        if year_match:
            year = int(year_match.group(1))
            if 2015 <= year <= 2025:
                extracted_fields['tax_year'] = year

        return extracted_fields

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process document with better page selection"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            logger.info(f"Processing document: {file_path}")

            if file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                text, confidence = self.extract_text_from_image(file_path)
                if self.is_w2_document(text):
                    best_page = (text, confidence, 1)
                else:
                    best_page = None
            elif file_ext == '.pdf':
                pages_text = self.extract_text_from_pdf_pages(file_path)
                best_page = self.find_best_w2_page(pages_text)
            else:
                return {
                    'is_w2': False,
                    'error': f'Unsupported file type: {file_ext}',
                    'confidence': 0.0
                }

            if not best_page:
                return {
                    'is_w2': False,
                    'error': 'No valid W2 data found',
                    'confidence': 0.0,
                    'raw_text': '',
                    'extracted_fields': {}
                }

            text, confidence, page_num = best_page
            logger.info(f"Processing page {page_num} for W2 extraction")
            
            extracted_fields = self.extract_w2_fields(text)

            return {
                'is_w2': True,
                'confidence': confidence,
                'raw_text': text,
                'extracted_fields': extracted_fields,
                'error': None,
                'page_processed': page_num
            }

        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return {
                'is_w2': False,
                'error': str(e),
                'confidence': 0.0,
                'raw_text': '',
                'extracted_fields': {}
            }
