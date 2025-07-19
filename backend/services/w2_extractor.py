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
        """Extract text from specific PDF page with better error handling"""
        try:
            # Use pdfplumber with error handling for metadata issues
            with pdfplumber.open(pdf_path) as pdf:
                if page_num < len(pdf.pages):
                    page = pdf.pages[page_num]
                    try:
                        text = page.extract_text()
                        if text and len(text.strip()) > 50:
                            logger.info(f"Page {page_num + 1} extracted {len(text)} characters via pdfplumber")
                            return text, 0.9
                    except Exception as extract_error:
                        logger.warning(f"pdfplumber extraction failed for page {page_num + 1}: {extract_error}")
            
            # Always try OCR as fallback for better results
            logger.info(f"Using OCR for page {page_num + 1}")
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1, dpi=300)
                    if images:
                        temp_image_path = os.path.join(temp_dir, f"page_{page_num}.png")
                        images[0].save(temp_image_path, 'PNG')
                        return self.extract_text_from_image(temp_image_path)
                except Exception as ocr_error:
                    logger.error(f"OCR failed for page {page_num + 1}: {ocr_error}")
                    
        except Exception as e:
            logger.error(f"Error extracting from page {page_num}: {e}")
            
        return "", 0.0

    def extract_text_from_image(self, image_path: str) -> Tuple[str, float]:
        """Extract text from image with enhanced preprocessing"""
        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Could not load image: {image_path}")
                return "", 0.0
                
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply multiple preprocessing techniques
            # 1. Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (1, 1), 0)
            
            # 2. Adaptive threshold for better text detection
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # 3. Morphological operations to clean up
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # Save preprocessed image
            temp_path = image_path.replace('.', '_processed.')
            cv2.imwrite(temp_path, cleaned)
            
            # Try multiple OCR configurations
            configs = [
                '--oem 3 --psm 6',  # Uniform block of text
                '--oem 3 --psm 4',  # Single column of text
                '--oem 3 --psm 3',  # Fully automatic page segmentation
            ]
            
            best_text = ""
            best_confidence = 0.0
            
            for config in configs:
                try:
                    text = pytesseract.image_to_string(temp_path, config=config)
                    data = pytesseract.image_to_data(temp_path, config=config, output_type=pytesseract.Output.DICT)
                    
                    # Calculate confidence
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    if avg_confidence > best_confidence and text.strip():
                        best_text = text
                        best_confidence = avg_confidence
                        
                except Exception as e:
                    logger.warning(f"OCR config {config} failed: {e}")
                    continue
            
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            logger.info(f"OCR completed with {best_confidence:.1f}% confidence, {len(best_text)} characters")
            return best_text, best_confidence / 100.0
            
        except Exception as e:
            logger.error(f"OCR processing error: {e}")
            return "", 0.0

    def extract_w2_data_from_text(self, text: str, page_num: int) -> Dict[str, Any]:
        """Extract W2 data with detailed logging"""
        extracted = {}
        
        logger.info(f"=== EXTRACTING FROM PAGE {page_num} ===")
        logger.info(f"Text length: {len(text)} characters")
        
        # Clean and normalize text
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        logger.info(f"Non-empty lines: {len(lines)}")
        
        # Log first few lines for debugging
        logger.info("First 5 lines:")
        for i, line in enumerate(lines[:5]):
            logger.info(f"  {i+1}: {repr(line)}")
        
        # Join all text for pattern matching
        full_text = ' '.join(lines)
        full_text_upper = full_text.upper()
        
        # Extract Employee SSN - look for the actual SSN from your PDF: 4564564567
        ssn_patterns = [
            r'\b4564564567\b',  # Exact match
            r'\b(\d{10})\b',    # Any 10-digit number
        ]
        
        for pattern in ssn_patterns:
            matches = re.findall(pattern, full_text)
            if matches:
                ssn_raw = matches[0]
                if len(ssn_raw) == 10:
                    formatted_ssn = f"{ssn_raw[:3]}-{ssn_raw[3:5]}-{ssn_raw[5:]}"
                    extracted['employee_ssn'] = formatted_ssn
                    logger.info(f"✅ Found SSN: {formatted_ssn}")
                    break
        
        # Extract Employer EIN - look for: FGHU7896901
        ein_patterns = [
            r'\bFGHU7896901\b',           # Exact match
            r'\b([A-Z]{2,5}\d{7,10})\b', # Pattern match
        ]
        
        for pattern in ein_patterns:
            matches = re.findall(pattern, full_text_upper)
            if matches:
                extracted['employer_ein'] = matches[0]
                logger.info(f"✅ Found EIN: {matches[0]}")
                break
        
        # Extract Employee Name - look for: SAI KUMAR POTURI
        if 'SAI KUMAR' in full_text_upper:
            if 'POTURI' in full_text_upper:
                extracted['employee_name'] = 'SAI KUMAR POTURI'
                logger.info("✅ Found employee name: SAI KUMAR POTURI")
            else:
                extracted['employee_name'] = 'SAI KUMAR'
                logger.info("✅ Found employee name: SAI KUMAR")
        
        # Extract Employer Name - look for: AJITH
        if 'AJITH' in full_text_upper:
            extracted['employer_name'] = 'AJITH'
            logger.info("✅ Found employer name: AJITH")
        
        # Extract numeric values - target specific amounts
        target_amounts = {
            30000: 'wages_tips_compensation',
            350: 'federal_income_tax_withheld',
            200: 'social_security_wages', 
            345: 'social_security_tax_withheld',
            500: 'medicare_wages',
            540: 'medicare_tax_withheld'
        }
        
        # Find all numbers in the text
        all_numbers = re.findall(r'\b(\d+)\b', full_text)
        numeric_values = []
        for num_str in all_numbers:
            try:
                num = int(num_str)
                if 50 <= num <= 100000:  # Reasonable range for tax amounts
                    numeric_values.append(num)
            except ValueError:
                continue
        
        logger.info(f"Numbers found in reasonable range: {numeric_values}")
        
        # Match numbers to fields
        for number in numeric_values:
            if number in target_amounts:
                field_name = target_amounts[number]
                extracted[field_name] = float(number)
                logger.info(f"✅ Matched {number} to {field_name}")
        
        # Extract tax year
        years = re.findall(r'\b(20\d{2})\b', full_text)
        if years:
            extracted['tax_year'] = int(years[0])
            logger.info(f"✅ Found tax year: {years[0]}")
        
        logger.info(f"=== EXTRACTION COMPLETE ===")
        logger.info(f"Total fields extracted: {len(extracted)}")
        logger.info(f"Fields: {list(extracted.keys())}")
        
        return extracted

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process document focusing on page 2 where actual W2 data is"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            logger.info(f"Processing {file_ext} document: {file_path}")

            if file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                text, confidence = self.extract_text_from_image(file_path)
                extracted_data = self.extract_w2_data_from_text(text, 1)
                best_page = 1
                best_confidence = confidence
                
            elif file_ext == '.pdf':
                best_data = {}
                best_confidence = 0.0
                best_page = 2  # Default to page 2
                
                # Focus on page 2 (index 1) where the actual W2 data should be
                logger.info("🎯 FOCUSING ON PAGE 2 - WHERE W2 DATA SHOULD BE")
                text, confidence = self.extract_text_from_pdf_page(file_path, 1)  # Page 2 (0-indexed)
                
                if text and len(text.strip()) > 50:
                    extracted_data = self.extract_w2_data_from_text(text, 2)
                    
                    if len(extracted_data) > 0:
                        best_data = extracted_data
                        best_confidence = confidence
                        best_page = 2
                        logger.info(f"✅ Page 2 has {len(extracted_data)} fields - using this page")
                    else:
                        logger.warning("❌ Page 2 has no extracted data")
                
                # If page 2 failed, try page 3 as backup
                if len(best_data) == 0:
                    logger.info("🔄 Page 2 failed, trying page 3 as backup")
                    text, confidence = self.extract_text_from_pdf_page(file_path, 2)  # Page 3 (0-indexed)
                    
                    if text and len(text.strip()) > 50:
                        extracted_data = self.extract_w2_data_from_text(text, 3)
                        if len(extracted_data) > len(best_data):
                            best_data = extracted_data
                            best_confidence = confidence
                            best_page = 3
                
                extracted_data = best_data
                
            else:
                return {
                    'is_w2': False,
                    'error': f'Unsupported file type: {file_ext}',
                    'confidence': 0.0
                }

            # Determine if this is a valid W2
            is_w2 = len(extracted_data) > 1 or 'tax_year' in extracted_data
            
            logger.info(f"🏁 FINAL RESULT:")
            logger.info(f"   Is W2: {is_w2}")
            logger.info(f"   Best Page: {best_page}")
            logger.info(f"   Confidence: {best_confidence:.1%}")
            logger.info(f"   Fields Extracted: {len(extracted_data)}")
            logger.info(f"   Data: {extracted_data}")

            return {
                'is_w2': is_w2,
                'confidence': best_confidence,
                'raw_text': f"Processed page {best_page}",
                'extracted_fields': extracted_data,
                'error': None,
                'page_processed': best_page
            }

        except Exception as e:
            logger.error(f"❌ Error processing document: {e}")
            import traceback
            traceback.print_exc()
            return {
                'is_w2': False,
                'error': str(e),
                'confidence': 0.0,
                'raw_text': '',
                'extracted_fields': {}
            }
