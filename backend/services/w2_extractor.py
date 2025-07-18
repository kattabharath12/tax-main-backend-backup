import pytesseract
import pdfplumber
from pdf2image import convert_from_path
from PIL import Image
import re
import os
import logging
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class W2Extractor:
    def __init__(self):
        self.w2_patterns = {
            'employer_name': [
                r'(?:Employer|Company).*?([A-Z][A-Za-z\s&.,]+)',
                r'b\s*Employer identification number.*?\n([A-Z][A-Za-z\s&.,]+)',
            ],
            'employer_ein': [
                r'(?:EIN|Employer identification number).*?(\d{2}-\d{7})',
                r'b\s*Employer identification number.*?(\d{2}-\d{7})',
            ],
            'employee_ssn': [
                r'(?:SSN|Social Security Number).*?(\d{3}-\d{2}-\d{4})',
                r'(?:Employee.*?social security number).*?(\d{3}-\d{2}-\d{4})',
            ],
            'wages_tips_compensation': [
                r'(?:Box\s*1|Wages, tips, other compensation).*?(\d+\.?\d*)',
                r'1\s*Wages, tips, other compensation.*?(\d+\.?\d*)',
            ],
            'federal_income_tax_withheld': [
                r'(?:Box\s*2|Federal income tax withheld).*?(\d+\.?\d*)',
                r'2\s*Federal income tax withheld.*?(\d+\.?\d*)',
            ],
            'social_security_wages': [
                r'(?:Box\s*3|Social security wages).*?(\d+\.?\d*)',
                r'3\s*Social security wages.*?(\d+\.?\d*)',
            ],
            'social_security_tax_withheld': [
                r'(?:Box\s*4|Social security tax withheld).*?(\d+\.?\d*)',
                r'4\s*Social security tax withheld.*?(\d+\.?\d*)',
            ],
            'medicare_wages': [
                r'(?:Box\s*5|Medicare wages and tips).*?(\d+\.?\d*)',
                r'5\s*Medicare wages and tips.*?(\d+\.?\d*)',
            ],
            'medicare_tax_withheld': [
                r'(?:Box\s*6|Medicare tax withheld).*?(\d+\.?\d*)',
                r'6\s*Medicare tax withheld.*?(\d+\.?\d*)',
            ],
        }

    def preprocess_image(self, image_path: str) -> str:
        """Preprocess image for better OCR results"""
        try:
            # Read image
            img = cv2.imread(image_path)

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply threshold to get image with only black and white
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Noise removal
            kernel = np.ones((1, 1), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPENING, kernel, iterations=1)

            # Save preprocessed image temporarily
            temp_path = image_path.replace('.', '_processed.')
            cv2.imwrite(temp_path, opening)

            return temp_path
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return image_path

    def extract_text_from_image(self, image_path: str) -> Tuple[str, float]:
        """Extract text from image using OCR"""
        try:
            # Preprocess image
            processed_path = self.preprocess_image(image_path)

            # Configure tesseract
            custom_config = r'--oem 3 --psm 6'

            # Extract text with confidence
            data = pytesseract.image_to_data(processed_path, config=custom_config, output_type=pytesseract.Output.DICT)
            text = pytesseract.image_to_string(processed_path, config=custom_config)

            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            # Clean up temporary file
            if processed_path != image_path and os.path.exists(processed_path):
                os.remove(processed_path)

            return text, avg_confidence / 100.0
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return "", 0.0

    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, float]:
        """Extract text from PDF"""
        try:
            text = ""
            confidence = 0.8  # PDFs generally have good text extraction

            # Try direct text extraction first
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            # If no text found, convert to images and use OCR
            if not text.strip():
                images = convert_from_path(pdf_path)
                total_confidence = 0
                page_count = 0

                for i, image in enumerate(images):
                    temp_image_path = f"/tmp/temp_page_{i}.png"
                    image.save(temp_image_path, 'PNG')

                    page_text, page_conf = self.extract_text_from_image(temp_image_path)
                    text += page_text + "\n"
                    total_confidence += page_conf
                    page_count += 1

                    # Clean up temp file
                    if os.path.exists(temp_image_path):
                        os.remove(temp_image_path)

                confidence = total_confidence / page_count if page_count > 0 else 0.0

            return text, confidence
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return "", 0.0

    def is_w2_document(self, text: str) -> bool:
        """Check if the document is likely a W2 form"""
        w2_indicators = [
            "wage and tax statement",
            "form w-2",
            "w-2",
            "employer identification number",
            "wages, tips, other compensation",
            "federal income tax withheld",
            "social security wages",
            "medicare wages"
        ]

        text_lower = text.lower()
        matches = sum(1 for indicator in w2_indicators if indicator in text_lower)
        return matches >= 3  # Require at least 3 indicators

    def extract_w2_fields(self, text: str) -> Dict[str, Any]:
        """Extract W2 fields from text using regex patterns"""
        extracted_fields = {}

        for field_name, patterns in self.w2_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    value = match.group(1).strip()

                    # Convert numeric fields to float
                    if field_name in ['wages_tips_compensation', 'federal_income_tax_withheld', 
                                    'social_security_wages', 'social_security_tax_withheld',
                                    'medicare_wages', 'medicare_tax_withheld']:
                        try:
                            # Remove commas and convert to float
                            value = float(value.replace(',', ''))
                        except ValueError:
                            continue

                    extracted_fields[field_name] = value
                    break

        # Extract tax year
        year_match = re.search(r'(20\d{2})', text)
        if year_match:
            extracted_fields['tax_year'] = int(year_match.group(1))

        return extracted_fields

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process any document and extract W2 data if it's a W2 form"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()

            # Extract text based on file type
            if file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                text, confidence = self.extract_text_from_image(file_path)
            elif file_ext == '.pdf':
                text, confidence = self.extract_text_from_pdf(file_path)
            else:
                return {
                    'is_w2': False,
                    'error': f'Unsupported file type: {file_ext}',
                    'confidence': 0.0
                }

            # Check if it's a W2 document
            is_w2 = self.is_w2_document(text)

            result = {
                'is_w2': is_w2,
                'confidence': confidence,
                'raw_text': text,
                'extracted_fields': {},
                'error': None
            }

            # If it's a W2, extract fields
            if is_w2:
                result['extracted_fields'] = self.extract_w2_fields(text)

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