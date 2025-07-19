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

    def extract_all_numbers_and_text(self, text: str) -> Dict[str, List]:
        """Extract all numbers and text chunks for analysis"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        all_numbers = []
        all_text_chunks = []
        
        for i, line in enumerate(lines):
            # Find all numbers in this line
            numbers = re.findall(r'\b(\d+(?:\.\d{2})?)\b', line)
            for num in numbers:
                if len(num) >= 2:  # At least 2 digits
                    all_numbers.append((num, i, line))
            
            # Find text chunks (potential names)
            text_chunks = re.findall(r'\b([A-Z][A-Za-z]{2,}(?:\s+[A-Z][A-Za-z]{2,})*)\b', line)
            for chunk in text_chunks:
                if len(chunk) >= 3:
                    all_text_chunks.append((chunk, i, line))
        
        return {
            'numbers': all_numbers,
            'text_chunks': all_text_chunks,
            'lines': lines
        }

    def smart_field_extraction(self, text: str) -> Dict[str, Any]:
        """Extract W2 fields using smart pattern matching"""
        extracted = {}
        
        # Get all data
        data = self.extract_all_numbers_and_text(text)
        numbers = data['numbers']
        text_chunks = data['text_chunks']
        lines = data['lines']
        
        logger.info(f"Found {len(numbers)} numbers and {len(text_chunks)} text chunks")
        
        # 1. Extract SSN - look for 10-digit or formatted SSN
        for num, line_idx, line in numbers:
            if len(num) == 10 and num != '1234567890':
                # Format as SSN
                formatted_ssn = f"{num[:3]}-{num[3:5]}-{num[5:]}"
                extracted['employee_ssn'] = formatted_ssn
                logger.info(f"✅ Found SSN: {formatted_ssn} (line {line_idx + 1})")
                break
        
        # Also check for already formatted SSNs
        ssn_matches = re.findall(r'\b(\d{3}-\d{2}-\d{4})\b', text)
        if ssn_matches and 'employee_ssn' not in extracted:
            for ssn in ssn_matches:
                if ssn != '123-45-6789':  # Skip template
                    extracted['employee_ssn'] = ssn
                    logger.info(f"✅ Found formatted SSN: {ssn}")
                    break
        
        # 2. Extract EIN - look for letter+number patterns
        ein_patterns = [
            r'\b([A-Z]{2,5}\d{7,10})\b',
            r'\b(\d{2}-\d{7})\b'
        ]
        
        for pattern in ein_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if match not in ['12-3456789']:  # Skip template
                    extracted['employer_ein'] = match
                    logger.info(f"✅ Found EIN: {match}")
                    break
            if 'employer_ein' in extracted:
                break
        
        # 3. Extract Names - look for meaningful text chunks
        potential_names = []
        for chunk, line_idx, line in text_chunks:
            # Skip obvious form labels
            skip_terms = [
                'employee', 'employer', 'identification', 'number', 'address', 'code',
                'wages', 'tips', 'compensation', 'federal', 'income', 'tax', 'withheld',
                'social', 'security', 'medicare', 'statement', 'treasury', 'revenue',
                'service', 'privacy', 'paperwork', 'reduction', 'notice', 'instructions',
                'official', 'administration', 'department'
            ]
            
            chunk_lower = chunk.lower()
            if (len(chunk.split()) <= 3 and  # Not too long
                not any(term in chunk_lower for term in skip_terms) and
                not re.search(r'\d', chunk)):  # No numbers
                
                potential_names.append((chunk, line_idx, line))
        
        # Sort names by line position and pick the most likely ones
        potential_names.sort(key=lambda x: x[1])
        
        for name, line_idx, line in potential_names:
            if 'employee_name' not in extracted:
                # Look for patterns that suggest this is an employee name
                if any(indicator in line.lower() for indicator in ['first name', 'employee']):
                    extracted['employee_name'] = name
                    logger.info(f"✅ Found employee name: {name} (line {line_idx + 1})")
                    continue
            
            if 'employer_name' not in extracted:
                # Look for patterns that suggest this is an employer name
                if any(indicator in line.lower() for indicator in ['employer', 'company']):
                    extracted['employer_name'] = name
                    logger.info(f"✅ Found employer name: {name} (line {line_idx + 1})")
                    continue
        
        # 4. Extract Wage Amounts - look for realistic wage numbers
        wage_candidates = []
        for num, line_idx, line in numbers:
            try:
                amount = float(num)
                # Look for realistic wage amounts
                if 100 <= amount <= 999999:
                    wage_candidates.append((amount, line_idx, line))
            except ValueError:
                continue
        
        # Sort by amount (larger amounts likely to be wages)
        wage_candidates.sort(key=lambda x: x[0], reverse=True)
        
        logger.info(f"Wage candidates: {[(amt, line) for amt, _, line in wage_candidates[:10]]}")
        
        # Map wage amounts based on patterns in the line
        for amount, line_idx, line in wage_candidates:
            line_lower = line.lower()
            
            # Look for box indicators or wage descriptions
            if 'wages' in line_lower and 'compensation' in line_lower:
                if 'wages_tips_compensation' not in extracted:
                    extracted['wages_tips_compensation'] = amount
                    logger.info(f"✅ Found wages: ${amount:,.2f} (line {line_idx + 1})")
            
            elif 'federal' in line_lower and 'withheld' in line_lower:
                if 'federal_income_tax_withheld' not in extracted:
                    extracted['federal_income_tax_withheld'] = amount
                    logger.info(f"✅ Found federal tax: ${amount:,.2f} (line {line_idx + 1})")
            
            elif 'social security' in line_lower and 'wages' in line_lower:
                if 'social_security_wages' not in extracted:
                    extracted['social_security_wages'] = amount
                    logger.info(f"✅ Found SS wages: ${amount:,.2f} (line {line_idx + 1})")
            
            elif 'social security' in line_lower and 'withheld' in line_lower:
                if 'social_security_tax_withheld' not in extracted:
                    extracted['social_security_tax_withheld'] = amount
                    logger.info(f"✅ Found SS tax: ${amount:,.2f} (line {line_idx + 1})")
            
            elif 'medicare' in line_lower and 'wages' in line_lower:
                if 'medicare_wages' not in extracted:
                    extracted['medicare_wages'] = amount
                    logger.info(f"✅ Found Medicare wages: ${amount:,.2f} (line {line_idx + 1})")
            
            elif 'medicare' in line_lower and 'withheld' in line_lower:
                if 'medicare_tax_withheld' not in extracted:
                    extracted['medicare_tax_withheld'] = amount
                    logger.info(f"✅ Found Medicare tax: ${amount:,.2f} (line {line_idx + 1})")
        
        # 5. Extract Tax Year
        year_matches = re.findall(r'\b(20\d{2})\b', text)
        if year_matches:
            years = [int(y) for y in year_matches if 2020 <= int(y) <= 2025]
            if years:
                extracted['tax_year'] = max(years)
                logger.info(f"✅ Found tax year: {extracted['tax_year']}")
        
        return extracted

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process document with spatial analysis"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            logger.info(f"Processing {file_ext} document with spatial analysis")

            if file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                text, confidence = self.extract_text_from_image(file_path)
                extracted_fields = self.smart_field_extraction(text)
                best_confidence = confidence
                
            elif file_ext == '.pdf':
                # Process all pages and combine results
                all_extracted = {}
                best_confidence = 0.0
                
                for page_num in range(min(3, 10)):  # First 3 pages
                    try:
                        text, confidence = self.extract_text_from_pdf_page(file_path, page_num)
                        
                        if not text or len(text.strip()) < 50:
                            continue
                        
                        logger.info(f"\n=== PROCESSING PAGE {page_num + 1} ===")
                        page_extracted = self.smart_field_extraction(text)
                        
                        # Merge results, preferring non-empty values
                        for key, value in page_extracted.items():
                            if key not in all_extracted or not all_extracted[key]:
                                all_extracted[key] = value
                        
                        best_confidence = max(best_confidence, confidence)
                        
                    except Exception as e:
                        logger.warning(f"Error processing page {page_num + 1}: {e}")
                        continue
                
                extracted_fields = all_extracted
                
            else:
                return {
                    'is_w2': False,
                    'error': f'Unsupported file type: {file_ext}',
                    'confidence': 0.0
                }

            # Determine if this is a valid W2
            has_structure = any(keyword in text.lower() for keyword in ['w-2', 'wage and tax statement'])
            has_data = len(extracted_fields) >= 3
            has_key_fields = any(key in extracted_fields for key in ['employee_ssn', 'employer_ein', 'wages_tips_compensation'])
            
            is_w2 = has_structure and (has_data or has_key_fields)

            logger.info(f"🏁 SPATIAL W2 EXTRACTION:")
            logger.info(f"   Is W2: {is_w2}")
            logger.info(f"   Has structure: {has_structure}")
            logger.info(f"   Has data: {has_data}")
            logger.info(f"   Has key fields: {has_key_fields}")
            logger.info(f"   Fields found: {len(extracted_fields)}")
            logger.info(f"   Final data: {extracted_fields}")

            return {
                'is_w2': is_w2,
                'confidence': best_confidence,
                'raw_text': "Processed with spatial analysis",
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
