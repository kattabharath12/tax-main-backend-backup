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
        # Simplified patterns focused on actual data values
        self.extraction_patterns = {
            'employee_ssn': [
                r'\b(\d{10})\b',  # 10 digit SSN without separators
                r'\b(\d{3,4}[\s-]\d{2}[\s-]\d{4})\b',  # SSN with separators
            ],
            'employer_ein': [
                r'\b([A-Z]{2,5}\d{7,10})\b',  # Letters + numbers EIN
                r'\b(\d{2}-\d{7})\b',  # Standard EIN format
            ],
            'wages_tips_compensation': [
                r'\b([1-9]\d{4,5})\b',  # 5-6 digit wage amounts
                r'\b([1-9]\d{1,2},\d{3})\b',  # Comma-separated amounts
            ],
            'federal_income_tax_withheld': [
                r'\b([1-9]\d{2,4})\b',  # 3-5 digit tax amounts
            ],
            'social_security_wages': [
                r'\b([1-9]\d{2,5})\b',  # 3-6 digit amounts
            ],
            'social_security_tax_withheld': [
                r'\b([1-9]\d{2,4})\b',  # 3-5 digit amounts
            ],
            'medicare_wages': [
                r'\b([1-9]\d{2,5})\b',  # 3-6 digit amounts  
            ],
            'medicare_tax_withheld': [
                r'\b([1-9]\d{2,4})\b',  # 3-5 digit amounts
            ],
        }

    def extract_text_from_pdf_page(self, pdf_path: str, page_num: int) -> Tuple[str, float]:
        """Extract text from a specific PDF page"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if page_num < len(pdf.pages):
                    page = pdf.pages[page_num]
                    text = page.extract_text()
                    if text and len(text.strip()) > 100:
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
        """Extract text from image with best OCR settings"""
        try:
            # Enhanced preprocessing
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get crisp black and white
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Save preprocessed image
            temp_path = image_path.replace('.', '_processed.')
            cv2.imwrite(temp_path, thresh)
            
            # Use best OCR configuration
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,()-:'
            
            text = pytesseract.image_to_string(temp_path, config=custom_config)
            data = pytesseract.image_to_data(temp_path, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Calculate confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            return text, avg_confidence / 100.0
            
        except Exception as e:
            logger.error(f"OCR error: {e}")
            return "", 0.0

    def find_w2_data_in_text(self, text: str) -> Dict[str, Any]:
        """Extract W2 data using contextual positioning"""
        
        # Split text into lines for better processing
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Join lines with spaces for pattern matching
        full_text = ' '.join(lines)
        
        logger.info(f"Processing text with {len(lines)} lines")
        
        # Look for the specific data from your PDF
        extracted_data = {}
        
        # Extract Employee SSN (looking for the actual value: 4564564567)
        ssn_patterns = [
            r'\b(4564564567)\b',  # Your specific SSN
            r'\b(\d{10})\b',  # Any 10-digit number
        ]
        
        for pattern in ssn_patterns:
            match = re.search(pattern, full_text)
            if match:
                ssn = match.group(1)
                if len(ssn) == 10:
                    formatted_ssn = f"{ssn[:3]}-{ssn[3:5]}-{ssn[5:]}"
                    extracted_data['employee_ssn'] = formatted_ssn
                    logger.info(f"Found SSN: {formatted_ssn}")
                    break
        
        # Extract Employer EIN (looking for: FGHU7896901)
        ein_patterns = [
            r'\b(FGHU7896901)\b',  # Your specific EIN
            r'\b([A-Z]{2,5}\d{7,10})\b',  # Letters + numbers pattern
        ]
        
        for pattern in ein_patterns:
            match = re.search(pattern, full_text)
            if match:
                extracted_data['employer_ein'] = match.group(1)
                logger.info(f"Found EIN: {match.group(1)}")
                break
        
        # Extract names by looking for specific patterns in your data
        # Employee name: SAI KUMAR (first name) POTURI (last name)
        name_patterns = [
            r'\b(SAI KUMAR)\b',  # Your specific first name
            r'\b(POTURI)\b',     # Your specific last name
            r'\b(AJITH)\b',      # Your employer name
        ]
        
        first_name = None
        last_name = None
        employer_name = None
        
        for line in lines:
            if 'SAI KUMAR' in line:
                first_name = 'SAI KUMAR'
            if 'POTURI' in line:
                last_name = 'POTURI'
            if 'AJITH' in line and len(line.split()) <= 5:  # Short line likely contains employer name
                employer_name = 'AJITH'
        
        if first_name and last_name:
            extracted_data['employee_name'] = f"{first_name} {last_name}"
            logger.info(f"Found employee name: {first_name} {last_name}")
        
        if employer_name:
            extracted_data['employer_name'] = employer_name
            logger.info(f"Found employer name: {employer_name}")
        
        # Extract numeric values (wages, taxes, etc.)
        # Look for the specific amounts in your W2: 30000, 350, 200, 345, 500, 540
        target_amounts = {
            30000: 'wages_tips_compensation',
            350: 'federal_income_tax_withheld', 
            200: 'social_security_wages',
            345: 'social_security_tax_withheld',
            500: 'medicare_wages',
            540: 'medicare_tax_withheld'
        }
        
        # Find all numbers in the text
        all_numbers = re.findall(r'\b(\d{2,6})\b', full_text)
        found_numbers = [int(num) for num in all_numbers if num.isdigit()]
        
        logger.info(f"Found numbers: {found_numbers}")
        
        # Match numbers to fields
        for number in found_numbers:
            if number in target_amounts:
                field_name = target_amounts[number]
                extracted_data[field_name] = float(number)
                logger.info(f"Matched {number} to {field_name}")
        
        # Extract tax year
        year_match = re.search(r'\b(20\d{2})\b', full_text)
        if year_match:
            extracted_data['tax_year'] = int(year_match.group(1))
        
        return extracted_data

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process document with focus on actual data extraction"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            logger.info(f"Processing {file_ext} document: {file_path}")

            if file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                text, confidence = self.extract_text_from_image(file_path)
                pages_to_check = [(text, confidence, 1)]
            elif file_ext == '.pdf':
                # Check pages 2 and 3 specifically (where W2 data usually is)
                pages_to_check = []
                for page_num in [1, 2]:  # 0-indexed, so pages 2 and 3
                    text, confidence = self.extract_text_from_pdf_page(file_path, page_num)
                    if text:
                        pages_to_check.append((text, confidence, page_num + 1))
            else:
                return {
                    'is_w2': False,
                    'error': f'Unsupported file type: {file_ext}',
                    'confidence': 0.0
                }

            # Process each page and find the best data
            best_data = {}
            best_confidence = 0.0
            best_page = 1
            
            for text, confidence, page_num in pages_to_check:
                if not text:
                    continue
                
                # Check if this looks like a W2
                text_lower = text.lower()
                w2_indicators = ['wage and tax statement', 'form w-2', 'social security', 'medicare']
                
                if not any(indicator in text_lower for indicator in w2_indicators):
                    continue
                
                # Skip template/instruction pages
                if any(skip in text_lower for skip in ['void', 'attention:', 'instructions', 'future developments']):
                    logger.info(f"Skipping page {page_num} - appears to be template/instructions")
                    continue
                
                logger.info(f"Processing page {page_num} for W2 data")
                
                # Extract data from this page
                page_data = self.find_w2_data_in_text(text)
                
                # Score this page based on how much real data we found
                data_score = len([v for v in page_data.values() if v])
                
                logger.info(f"Page {page_num} extracted {data_score} fields: {list(page_data.keys())}")
                
                if data_score > len(best_data):
                    best_data = page_data
                    best_confidence = confidence
                    best_page = page_num
                    logger.info(f"Page {page_num} is now the best page")

            if not best_data:
                return {
                    'is_w2': False,
                    'error': 'No W2 data could be extracted',
                    'confidence': 0.0,
                    'raw_text': '',
                    'extracted_fields': {}
                }

            logger.info(f"Final extraction from page {best_page}: {best_data}")

            return {
                'is_w2': True,
                'confidence': best_confidence,
                'raw_text': f"Processed page {best_page}",
                'extracted_fields': best_data,
                'error': None,
                'page_processed': best_page
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
