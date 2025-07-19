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
        # Enhanced W2 patterns with more variations
        self.w2_patterns = {
            'employer_name': [
                r'(?i)(?:employer|company|employer name)[\s\n]*([A-Z][A-Za-z\s&.,\-\']+?)(?:\n|$|[0-9])',
                r'(?i)b\s*employer identification number.*?\n([A-Z][A-Za-z\s&.,\-\']+)',
                r'(?i)control number.*?\n([A-Z][A-Za-z\s&.,\-\']+)',
            ],
            'employer_ein': [
                r'(?i)(?:ein|employer identification number)[\s\n]*(\d{2}-?\d{7})',
                r'(?i)b\s*employer identification number[\s\n]*(\d{2}-?\d{7})',
                r'(\d{2}-\d{7})',
            ],
            'employee_ssn': [
                r'(?i)(?:ssn|social security number|employee.*?social security number)[\s\n]*(\d{3}-?\d{2}-?\d{4})',
                r'(?i)d\s*employee.*?social security number[\s\n]*(\d{3}-?\d{2}-?\d{4})',
                r'(\d{3}-\d{2}-\d{4})',
            ],
            'employee_name': [
                r'(?i)(?:employee|employee name)[\s\n]*([A-Z][A-Za-z\s\-\'\.]+?)(?:\n|$|[0-9])',
                r'(?i)e\s*employee.*?name[\s\n]*([A-Z][A-Za-z\s\-\'\.]+)',
            ],
            'wages_tips_compensation': [
                r'(?i)(?:box\s*1|1\s*wages.*?compensation|wages.*?tips.*?compensation)[\s\n]*[\$]?(\d{1,3}(?:,\d{3})*\.?\d{0,2})',
                r'(?i)1[\s]*wages.*?tips.*?other.*?compensation[\s\n]*[\$]?(\d{1,3}(?:,\d{3})*\.?\d{0,2})',
                r'(?i)wages.*?tips.*?other.*?compensation[\s\n]*[\$]?(\d{1,3}(?:,\d{3})*\.?\d{0,2})',
            ],
            'federal_income_tax_withheld': [
                r'(?i)(?:box\s*2|2\s*federal.*?tax.*?withheld|federal.*?income.*?tax.*?withheld)[\s\n]*[\$]?(\d{1,3}(?:,\d{3})*\.?\d{0,2})',
                r'(?i)2[\s]*federal.*?income.*?tax.*?withheld[\s\n]*[\$]?(\d{1,3}(?:,\d{3})*\.?\d{0,2})',
            ],
            'social_security_wages': [
                r'(?i)(?:box\s*3|3\s*social.*?security.*?wages|social.*?security.*?wages)[\s\n]*[\$]?(\d{1,3}(?:,\d{3})*\.?\d{0,2})',
                r'(?i)3[\s]*social.*?security.*?wages[\s\n]*[\$]?(\d{1,3}(?:,\d{3})*\.?\d{0,2})',
            ],
            'social_security_tax_withheld': [
                r'(?i)(?:box\s*4|4\s*social.*?security.*?tax.*?withheld|social.*?security.*?tax.*?withheld)[\s\n]*[\$]?(\d{1,3}(?:,\d{3})*\.?\d{0,2})',
                r'(?i)4[\s]*social.*?security.*?tax.*?withheld[\s\n]*[\$]?(\d{1,3}(?:,\d{3})*\.?\d{0,2})',
            ],
            'medicare_wages': [
                r'(?i)(?:box\s*5|5\s*medicare.*?wages|medicare.*?wages.*?tips)[\s\n]*[\$]?(\d{1,3}(?:,\d{3})*\.?\d{0,2})',
                r'(?i)5[\s]*medicare.*?wages.*?tips[\s\n]*[\$]?(\d{1,3}(?:,\d{3})*\.?\d{0,2})',
            ],
            'medicare_tax_withheld': [
                r'(?i)(?:box\s*6|6\s*medicare.*?tax.*?withheld|medicare.*?tax.*?withheld)[\s\n]*[\$]?(\d{1,3}(?:,\d{3})*\.?\d{0,2})',
                r'(?i)6[\s]*medicare.*?tax.*?withheld[\s\n]*[\$]?(\d{1,3}(?:,\d{3})*\.?\d{0,2})',
            ],
        }

        # Enhanced W2 detection keywords
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
            "employee's withholding allowance",
            "box 1",
            "box 2",
            "box 3"
        ]

    def enhance_image(self, image_path: str) -> str:
        """Enhanced image preprocessing for better OCR results"""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Could not read image: {image_path}")
                return image_path

            # Convert to PIL Image for enhancement
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # Enhance contrast
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(1.5)

            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(pil_img)
            pil_img = enhancer.enhance(2.0)

            # Convert back to CV2
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (1, 1), 0)

            # Apply adaptive threshold
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # Morphological operations to clean up
            kernel = np.ones((1, 1), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPENING, kernel, iterations=1)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

            # Save enhanced image
            temp_path = image_path.replace('.', '_enhanced.')
            cv2.imwrite(temp_path, closing)

            return temp_path
        except Exception as e:
            logger.error(f"Error enhancing image: {e}")
            return image_path

    def extract_text_from_image(self, image_path: str) -> Tuple[str, float]:
        """Enhanced text extraction from image using multiple OCR configurations"""
        try:
            # Enhance image first
            enhanced_path = self.enhance_image(image_path)

            # Try multiple PSM modes for better results
            psm_modes = [6, 4, 3, 8, 11, 12, 13]
            best_text = ""
            best_confidence = 0.0

            for psm in psm_modes:
                try:
                    # Custom configuration
                    custom_config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,()-$:'

                    # Extract text with confidence
                    data = pytesseract.image_to_data(enhanced_path, config=custom_config, output_type=pytesseract.Output.DICT)
                    text = pytesseract.image_to_string(enhanced_path, config=custom_config)

                    # Calculate confidence
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

                    # Keep the best result
                    if avg_confidence > best_confidence and text.strip():
                        best_text = text
                        best_confidence = avg_confidence

                except Exception as e:
                    logger.warning(f"Error with PSM mode {psm}: {e}")
                    continue

            # Clean up temporary file
            if enhanced_path != image_path and os.path.exists(enhanced_path):
                os.remove(enhanced_path)

            # If no good results, try without whitelist
            if best_confidence < 30:
                try:
                    custom_config = '--oem 3 --psm 6'
                    text = pytesseract.image_to_string(image_path, config=custom_config)
                    data = pytesseract.image_to_data(image_path, config=custom_config, output_type=pytesseract.Output.DICT)
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    confidence = sum(confidences) / len(confidences) if confidences else 0
                    if confidence > best_confidence:
                        best_text = text
                        best_confidence = confidence
                except Exception as e:
                    logger.warning(f"Fallback OCR failed: {e}")

            return best_text, best_confidence / 100.0

        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return "", 0.0

    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, float]:
        """Enhanced PDF text extraction with OCR fallback"""
        try:
            text = ""
            confidence = 0.8  # PDFs generally have good text extraction

            # Try direct text extraction first
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            # If no text found or very little text, convert to images and use OCR
            if not text.strip() or len(text.strip()) < 100:
                logger.info("PDF has little or no extractable text, using OCR...")
                
                # Create temporary directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    try:
                        # Convert PDF pages to images
                        images = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=2)  # Only first 2 pages
                        total_confidence = 0
                        page_count = 0
                        text = ""  # Reset text for OCR

                        for i, image in enumerate(images):
                            temp_image_path = os.path.join(temp_dir, f"temp_page_{i}.png")
                            image.save(temp_image_path, 'PNG')

                            page_text, page_conf = self.extract_text_from_image(temp_image_path)
                            text += page_text + "\n"
                            total_confidence += page_conf
                            page_count += 1

                        confidence = total_confidence / page_count if page_count > 0 else 0.0
                    except Exception as e:
                        logger.error(f"Error converting PDF to images: {e}")
                        confidence = 0.0

            return text, confidence
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return "", 0.0

    def is_w2_document(self, text: str) -> bool:
        """Enhanced W2 document detection"""
        text_lower = text.lower()
        
        # Count matches with weighted scoring
        matches = 0
        strong_indicators = ["form w-2", "wage and tax statement", "w-2"]
        box_indicators = ["box 1", "box 2", "box 3", "box 4", "box 5", "box 6"]
        
        # Strong indicators are worth more
        for indicator in strong_indicators:
            if indicator in text_lower:
                matches += 3
        
        # Box indicators
        for indicator in box_indicators:
            if indicator in text_lower:
                matches += 2
        
        # Regular indicators
        for indicator in self.w2_indicators:
            if indicator not in strong_indicators and indicator not in box_indicators:
                if indicator in text_lower:
                    matches += 1

        logger.info(f"W2 detection score: {matches}")
        return matches >= 4  # Require higher score

    def clean_numeric_value(self, value: str) -> Optional[float]:
        """Clean and convert numeric values"""
        try:
            # Remove common non-numeric characters
            cleaned = re.sub(r'[^\d.,\-]', '', value)
            # Handle comma as thousands separator
            cleaned = cleaned.replace(',', '')
            # Convert to float
            return float(cleaned)
        except (ValueError, TypeError):
            return None

    def extract_w2_fields(self, text: str) -> Dict[str, Any]:
        """Enhanced W2 field extraction with better pattern matching"""
        extracted_fields = {}
        
        # Clean text for better matching
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        for field_name, patterns in self.w2_patterns.items():
            for pattern in patterns:
                try:
                    matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        value = match.group(1).strip()
                        
                        # Skip very short or obviously wrong values
                        if len(value) < 2:
                            continue
                            
                        # Convert numeric fields to float
                        if field_name in ['wages_tips_compensation', 'federal_income_tax_withheld', 
                                        'social_security_wages', 'social_security_tax_withheld',
                                        'medicare_wages', 'medicare_tax_withheld']:
                            numeric_value = self.clean_numeric_value(value)
                            if numeric_value is not None and 0 <= numeric_value <= 999999:  # Reasonable range
                                extracted_fields[field_name] = numeric_value
                                logger.info(f"Extracted {field_name}: {numeric_value}")
                                break
                        else:
                            # Clean text fields
                            if field_name in ['employer_name', 'employee_name']:
                                # Remove common OCR artifacts
                                value = re.sub(r'^[^A-Za-z]*', '', value)  # Remove leading non-letters
                                value = re.sub(r'[^A-Za-z\s\-\.\&\']+.*$', '', value)  # Remove trailing artifacts
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

        # Extract tax year with multiple approaches
        if 'tax_year' not in extracted_fields:
            year_patterns = [
                r'(?i)(?:tax\s*year|year)\s*(\d{4})',
                r'(20\d{2})',
                r'(?i)for\s*calendar\s*year\s*(\d{4})',
            ]
            
            for pattern in year_patterns:
                match = re.search(pattern, text)
                if match:
                    year = int(match.group(1))
                    if 2015 <= year <= 2025:  # Reasonable year range
                        extracted_fields['tax_year'] = year
                        logger.info(f"Extracted tax_year: {year}")
                        break

        return extracted_fields

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Enhanced document processing with better error handling"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            logger.info(f"Processing document: {file_path} (type: {file_ext})")

            # Extract text based on file type
            if file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                text, confidence = self.extract_text_from_image(file_path)
            elif file_ext == '.pdf':
                text, confidence = self.extract_text_from_pdf(file_path)
            else:
                return {
                    'is_w2': False,
                    'error': f'Unsupported file type: {file_ext}',
                    'confidence': 0.0,
                    'raw_text': '',
                    'extracted_fields': {}
                }

            logger.info(f"Extracted text length: {len(text)}, confidence: {confidence}")
            
            # Check if it's a W2 document
            is_w2 = self.is_w2_document(text)
            logger.info(f"Is W2 document: {is_w2}")

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
                logger.info(f"Extracted fields: {list(result['extracted_fields'].keys())}")

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
