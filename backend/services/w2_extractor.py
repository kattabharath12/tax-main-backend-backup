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

    def parse_w2_line_by_line(self, text: str) -> Dict[str, Any]:
        """Parse W2 text line by line using the actual structure"""
        extracted = {}
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        logger.info("=== PARSING W2 LINE BY LINE ===")
        logger.info(f"Total lines: {len(lines)}")
        
        # Print all lines for debugging
        for i, line in enumerate(lines):
            logger.info(f"Line {i+1}: {line}")
        
        # Parse each line for specific patterns
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Line 2: SSN (after VOID)
            if 'void' in line_lower and re.search(r'\d{10}', line):
                ssn_match = re.search(r'(\d{10})', line)
                if ssn_match:
                    ssn = ssn_match.group(1)
                    # Format as SSN (taking last 9 digits if 10 digits found)
                    if len(ssn) == 10:
                        ssn = ssn[1:]  # Remove first digit
                    extracted['employee_ssn'] = f"{ssn[:3]}-{ssn[3:5]}-{ssn[5:]}"
                    logger.info(f"✅ Found SSN: {extracted['employee_ssn']} (line {i+1})")
            
            # EIN and Box 1,2 amounts - Line like "FGHU7896901 30000 350"
            if re.match(r'^[A-Z]{4}\d{7}\s+\d+\s+\d+', line):
                parts = line.split()
                if len(parts) >= 3:
                    extracted['employer_ein'] = parts[0]
                    extracted['wages_tips_compensation'] = float(parts[1])
                    extracted['federal_income_tax_withheld'] = float(parts[2])
                    logger.info(f"✅ Found EIN: {parts[0]} (line {i+1})")
                    logger.info(f"✅ Found wages (Box 1): ${float(parts[1]):,.2f} (line {i+1})")
                    logger.info(f"✅ Found federal tax (Box 2): ${float(parts[2]):,.2f} (line {i+1})")
            
            # Employer name line - after EIN line, contains address
            if 'employer' in line_lower and 'name' in line_lower and i+1 < len(lines):
                employer_line = lines[i+1]
                # Extract SS wages and tax from the numbers at the end first
                numbers = re.findall(r'\b(\d{1,4})\b', employer_line)
                if len(numbers) >= 2:
                    extracted['social_security_wages'] = float(numbers[-2])
                    extracted['social_security_tax_withheld'] = float(numbers[-1])
                    logger.info(f"✅ Found SS wages (Box 3): ${float(numbers[-2]):,.2f} (line {i+2})")
                    logger.info(f"✅ Found SS tax (Box 4): ${float(numbers[-1]):,.2f} (line {i+2})")
                    
                    # Remove the SS numbers from the end to get clean employer name
                    clean_line = employer_line
                    for num in numbers[-2:]:  # Remove last 2 numbers
                        clean_line = clean_line.replace(num, '', 1)
                    
                    # Clean up the employer name
                    extracted['employer_name'] = clean_line.strip().rstrip(',').strip()
                    logger.info(f"✅ Found employer name: {extracted['employer_name']} (line {i+2})")
                else:
                    # No numbers found, use the whole line
                    extracted['employer_name'] = employer_line.strip()
                    logger.info(f"✅ Found employer name: {extracted['employer_name']} (line {i+2})")
            
            # Medicare line - Line like "500 540"
            if re.match(r'^\d{3,4}\s+\d{3,4}$', line):
                parts = line.split()
                if len(parts) == 2:
                    # Check if this follows a Medicare wages line
                    if i > 0 and 'medicare' in lines[i-1].lower():
                        extracted['medicare_wages'] = float(parts[0])
                        extracted['medicare_tax_withheld'] = float(parts[1])
                        logger.info(f"✅ Found Medicare wages (Box 5): ${float(parts[0]):,.2f} (line {i+1})")
                        logger.info(f"✅ Found Medicare tax (Box 6): ${float(parts[1]):,.2f} (line {i+1})")
            
            # Employee name - Line with names after "first name and initial Last name"
            if 'first name' in line_lower and 'last name' in line_lower and i+1 < len(lines):
                name_line = lines[i+1]
                # Extract names (filter out numbers and single letters)
                name_parts = re.findall(r'\b[A-Z][A-Za-z]{1,}\b', name_line)
                if len(name_parts) >= 2:
                    extracted['employee_name'] = ' '.join(name_parts)
                    logger.info(f"✅ Found employee name: {extracted['employee_name']} (line {i+2})")
                elif len(name_parts) == 1:
                    extracted['employee_name'] = name_parts[0]
                    logger.info(f"✅ Found employee name: {extracted['employee_name']} (line {i+2})")
        
        # Extract tax year from anywhere in the text
        year_matches = re.findall(r'\b(20\d{2})\b', text)
        if year_matches:
            years = [int(y) for y in year_matches if 2020 <= int(y) <= 2025]
            if years:
                extracted['tax_year'] = max(years)
                logger.info(f"✅ Found tax year: {extracted['tax_year']}")
        
        # Extract employee address if available
        for i, line in enumerate(lines):
            if 'employee' in line.lower() and 'address' in line.lower() and i+1 < len(lines):
                addr_line = lines[i+1]
                # Look for address pattern (numbers, street, apt, zip)
                if re.search(r'\d+.*(?:ST|AVENUE|AVE|STREET|ROAD|RD).*\d{5}', addr_line, re.IGNORECASE):
                    extracted['employee_address'] = addr_line.strip()
                    logger.info(f"✅ Found employee address: {extracted['employee_address']} (line {i+2})")
        
        # Fallback for missing fields using general patterns
        if 'employee_ssn' not in extracted:
            # Look for any SSN pattern
            ssn_patterns = [r'(\d{3}-\d{2}-\d{4})', r'(\d{9})']
            for pattern in ssn_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    clean_ssn = re.sub(r'[-\s]', '', match)
                    if len(clean_ssn) == 9 and clean_ssn not in ['123456789', '000000000']:
                        extracted['employee_ssn'] = f"{clean_ssn[:3]}-{clean_ssn[3:5]}-{clean_ssn[5:]}"
                        logger.info(f"✅ Found SSN (fallback): {extracted['employee_ssn']}")
                        break
                if 'employee_ssn' in extracted:
                    break
        
        if 'employer_ein' not in extracted:
            # Look for EIN patterns
            ein_matches = re.findall(r'\b([A-Z]{2,5}\d{7,10})\b', text)
            for ein in ein_matches:
                if len(ein) >= 9:
                    extracted['employer_ein'] = ein
                    logger.info(f"✅ Found EIN (fallback): {ein}")
                    break
        
        # Validate and clean up amounts
        for key in ['wages_tips_compensation', 'federal_income_tax_withheld', 
                   'social_security_wages', 'social_security_tax_withheld',
                   'medicare_wages', 'medicare_tax_withheld']:
            if key in extracted:
                # Ensure reasonable ranges
                amount = extracted[key]
                if amount < 0 or amount > 9999999:
                    logger.warning(f"Removing unrealistic amount for {key}: ${amount}")
                    del extracted[key]
        
        return extracted

    def smart_field_extraction(self, text: str) -> Dict[str, Any]:
        """Main extraction method using line-by-line parsing"""
        return self.parse_w2_line_by_line(text)

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process document with precise line-by-line W2 extraction"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            logger.info(f"Processing {file_ext} document with line-by-line W2 extraction")

            if file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                text, confidence = self.extract_text_from_image(file_path)
                extracted_fields = self.smart_field_extraction(text)
                best_confidence = confidence
                
            elif file_ext == '.pdf':
                # Focus on the W2 page (usually page 2)
                all_extracted = {}
                best_confidence = 0.0
                w2_text = ""
                
                # Try pages in order of likelihood for W2 content
                page_priorities = [1, 0, 2]  # Page 2 first, then 1, then 3
                
                for page_num in page_priorities:
                    try:
                        text, confidence = self.extract_text_from_pdf_page(file_path, page_num)
                        
                        if not text or len(text.strip()) < 50:
                            continue
                        
                        # Check if this page contains W2 structure
                        w2_indicators = ['wage and tax statement', 'w-2', 'social security number', 
                                       'employer identification', 'wages, tips', 'federal income tax']
                        has_w2_structure = any(indicator in text.lower() for indicator in w2_indicators)
                        
                        logger.info(f"\n=== PROCESSING PAGE {page_num + 1} (W2 Structure: {has_w2_structure}) ===")
                        
                        if has_w2_structure:
                            page_extracted = self.smart_field_extraction(text)
                            # Prefer results from W2 pages
                            all_extracted.update(page_extracted)
                            best_confidence = max(best_confidence, confidence)
                            w2_text = text
                            break  # Stop after finding W2 page
                        elif not all_extracted:  # Use as fallback if no W2 page found yet
                            page_extracted = self.smart_field_extraction(text)
                            all_extracted.update(page_extracted)
                            best_confidence = max(best_confidence, confidence)
                            w2_text = text
                        
                    except Exception as e:
                        logger.warning(f"Error processing page {page_num + 1}: {e}")
                        continue
                
                extracted_fields = all_extracted
                text = w2_text
                
            else:
                return {
                    'is_w2': False,
                    'error': f'Unsupported file type: {file_ext}',
                    'confidence': 0.0
                }

            # Determine if this is a valid W2
            w2_indicators = ['w-2', 'wage and tax statement', 'form w-2', 'employer identification', 
                           'social security number', 'wages, tips', 'federal income tax']
            has_structure = any(indicator in text.lower() for indicator in w2_indicators)
            has_key_fields = any(key in extracted_fields for key in ['employee_ssn', 'employer_ein', 'wages_tips_compensation'])
            has_minimum_data = len(extracted_fields) >= 3
            
            is_w2 = has_structure and (has_key_fields or has_minimum_data)

            logger.info(f"🏁 LINE-BY-LINE W2 EXTRACTION RESULTS:")
            logger.info(f"   Is W2: {is_w2}")
            logger.info(f"   Has structure: {has_structure}")
            logger.info(f"   Has key fields: {has_key_fields}")
            logger.info(f"   Fields found: {len(extracted_fields)}")
            logger.info(f"   Confidence: {best_confidence:.2f}")
            logger.info(f"   Final extracted fields:")
            for key, value in extracted_fields.items():
                if isinstance(value, float):
                    logger.info(f"     {key}: ${value:,.2f}")
                else:
                    logger.info(f"     {key}: {value}")

            return {
                'is_w2': is_w2,
                'confidence': best_confidence,
                'raw_text': text[:500] + "..." if len(text) > 500 else text,
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
