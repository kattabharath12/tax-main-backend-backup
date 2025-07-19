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
        # W2 box structure mapping
        self.w2_boxes = {
            'a': 'employee_ssn',
            'b': 'employer_ein', 
            'c': 'employer_name',
            'd': 'control_number',
            'e': 'employee_name',
            'f': 'employee_address',
            '1': 'wages_tips_compensation',
            '2': 'federal_income_tax_withheld',
            '3': 'social_security_wages',
            '4': 'social_security_tax_withheld',
            '5': 'medicare_wages',
            '6': 'medicare_tax_withheld',
            '7': 'social_security_tips',
            '8': 'allocated_tips',
            '10': 'dependent_care_benefits',
            '11': 'nonqualified_plans',
            '12': 'codes',
            '13': 'statutory_employee',
            '14': 'other'
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

    def parse_w2_structured_text(self, text: str) -> Dict[str, Any]:
        """Parse W2 text using structured box patterns"""
        extracted = {}
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        logger.info("=== PARSING W2 STRUCTURED TEXT ===")
        logger.info(f"Total lines: {len(lines)}")
        
        # Print first 20 lines for debugging
        for i, line in enumerate(lines[:20]):
            logger.info(f"Line {i+1}: {line}")
        
        # Look for specific W2 patterns in sequence
        
        # 1. Find SSN (box a) - look for patterns near employee info
        ssn_patterns = [
            r'a\s+Employee\'?s\s+social\s+security\s+number[^\d]*(\d{9}|\d{3}[-\s]\d{2}[-\s]\d{4})',
            r'VOID.*?(\d{9}|\d{3}[-\s]\d{2}[-\s]\d{4})',
            r'(\d{9}|\d{3}[-\s]\d{2}[-\s]\d{4})',  # Any 9-digit number or formatted SSN
        ]
        
        for pattern in ssn_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                clean_ssn = re.sub(r'[-\s]', '', match)
                if len(clean_ssn) == 9 and clean_ssn.isdigit() and clean_ssn not in ['123456789', '000000000']:
                    extracted['employee_ssn'] = f"{clean_ssn[:3]}-{clean_ssn[3:5]}-{clean_ssn[5:]}"
                    logger.info(f"✅ Found SSN: {extracted['employee_ssn']}")
                    break
            if 'employee_ssn' in extracted:
                break
        
        # 2. Find EIN (box b) - should be near employer info
        ein_patterns = [
            r'b\s+Employer\s+identification\s+number[^\w]*([A-Z0-9]{10,12})',
            r'EIN[:\s]*([A-Z0-9]{10,12})',
            r'([A-Z]{2,4}\d{7,10})',  # General EIN pattern
        ]
        
        for pattern in ein_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) >= 9 and match not in ['XX-XXXXXXX', '12-3456789']:
                    extracted['employer_ein'] = match
                    logger.info(f"✅ Found EIN: {match}")
                    break
            if 'employer_ein' in extracted:
                break
        
        # 3. Find Employee Name (box e) - look for name patterns
        name_patterns = [
            r'e\s+Employee\'?s\s+first\s+name\s+and\s+initial.*?Last\s+name.*?([A-Z][A-Za-z]+).*?([A-Z][A-Za-z]+)',
            r'first\s+name.*?([A-Z][A-Za-z]+).*?Last\s+name.*?([A-Z][A-Za-z]+)',
        ]
        
        # Also look for names in structured areas
        for i, line in enumerate(lines):
            if 'first name' in line.lower() and 'last name' in line.lower():
                # Look in next few lines for names
                for j in range(i+1, min(i+4, len(lines))):
                    potential_names = re.findall(r'\b([A-Z][A-Za-z]{1,})\b', lines[j])
                    if len(potential_names) >= 2:
                        extracted['employee_name'] = ' '.join(potential_names[:2])
                        logger.info(f"✅ Found employee name: {extracted['employee_name']}")
                        break
                if 'employee_name' in extracted:
                    break
        
        # If not found, look for standalone name patterns
        if 'employee_name' not in extracted:
            skip_words = {'VOID', 'Employee', 'Employer', 'Social', 'Security', 'Medicare', 'Federal', 'Tax', 'Wages', 'Tips', 'Income', 'Withheld'}
            for line in lines:
                names = re.findall(r'\b([A-Z][A-Za-z]{2,})\b', line)
                valid_names = [name for name in names if name not in skip_words and len(name) > 2]
                if len(valid_names) >= 1:
                    extracted['employee_name'] = ' '.join(valid_names[:2]) if len(valid_names) > 1 else valid_names[0]
                    logger.info(f"✅ Found employee name (fallback): {extracted['employee_name']}")
                    break
        
        # 4. Find Employer Name (box c) - usually a company name
        for line in lines:
            if 'employer' in line.lower() and 'name' in line.lower():
                # Look in next few lines
                for j, next_line in enumerate(lines[lines.index(line)+1:lines.index(line)+4], 1):
                    if re.match(r'^[A-Z][A-Za-z\s,\.]{5,}', next_line):
                        extracted['employer_name'] = next_line.strip()
                        logger.info(f"✅ Found employer name: {extracted['employer_name']}")
                        break
                if 'employer_name' in extracted:
                    break
        
        # 5. Extract wage amounts using box-specific patterns
        box_patterns = {
            '1': (r'1\s+Wages,\s+tips,\s+other\s+compensation[^\d]*([0-9,]+(?:\.\d{2})?)', 'wages_tips_compensation'),
            '2': (r'2\s+Federal\s+income\s+tax\s+withheld[^\d]*([0-9,]+(?:\.\d{2})?)', 'federal_income_tax_withheld'),
            '3': (r'3\s+Social\s+security\s+wages[^\d]*([0-9,]+(?:\.\d{2})?)', 'social_security_wages'),
            '4': (r'4\s+Social\s+security\s+tax\s+withheld[^\d]*([0-9,]+(?:\.\d{2})?)', 'social_security_tax_withheld'),
            '5': (r'5\s+Medicare\s+wages\s+and\s+tips[^\d]*([0-9,]+(?:\.\d{2})?)', 'medicare_wages'),
            '6': (r'6\s+Medicare\s+tax\s+withheld[^\d]*([0-9,]+(?:\.\d{2})?)', 'medicare_tax_withheld'),
        }
        
        for box_num, (pattern, field_name) in box_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                amount_str = matches[0].replace(',', '')
                try:
                    amount = float(amount_str)
                    extracted[field_name] = amount
                    logger.info(f"✅ Found {field_name} (Box {box_num}): ${amount:,.2f}")
                except ValueError:
                    pass
        
        # 6. Fallback: look for numbers in structured positions
        if not any(key.endswith('_compensation') or key.endswith('_withheld') or key.endswith('_wages') for key in extracted.keys()):
            logger.info("🔄 Using fallback number extraction...")
            
            # Extract all numbers and their context
            number_contexts = []
            for i, line in enumerate(lines):
                numbers = re.findall(r'\b(\d{1,6}(?:\.\d{2})?)\b', line)
                for num in numbers:
                    try:
                        amount = float(num)
                        if 1 <= amount <= 999999:  # Reasonable wage range
                            context = ' '.join(lines[max(0, i-1):i+2]).lower()
                            number_contexts.append((amount, i, line, context))
                    except ValueError:
                        pass
            
            # Sort by amount (largest first for wages)
            number_contexts.sort(key=lambda x: x[0], reverse=True)
            
            logger.info(f"Found {len(number_contexts)} potential amounts:")
            for amount, line_idx, line, context in number_contexts[:10]:
                logger.info(f"  ${amount:,.2f} - Line {line_idx+1}: {line}")
            
            # Smart assignment based on W2 structure
            field_assignments = [
                ('wages_tips_compensation', ['wages', 'compensation', 'tips'], True),  # Largest amount
                ('federal_income_tax_withheld', ['federal', 'tax', 'withheld'], False),
                ('social_security_wages', ['social', 'security', 'wages'], False),
                ('social_security_tax_withheld', ['social', 'security', 'tax'], False),
                ('medicare_wages', ['medicare', 'wages'], False),
                ('medicare_tax_withheld', ['medicare', 'tax'], False),
            ]
            
            used_amounts = set()
            
            for field_name, keywords, prefer_largest in field_assignments:
                if field_name in extracted:
                    continue
                
                candidates = []
                for amount, line_idx, line, context in number_contexts:
                    if amount in used_amounts:
                        continue
                    
                    # Score based on keyword matches
                    score = sum(1 for keyword in keywords if keyword in context)
                    if score > 0:
                        candidates.append((score, amount, line_idx, line))
                
                if candidates:
                    # Sort by score, then by amount preference
                    candidates.sort(key=lambda x: (x[0], x[1] if prefer_largest else -x[1]), reverse=True)
                    amount = candidates[0][1]
                    extracted[field_name] = amount
                    used_amounts.add(amount)
                    logger.info(f"✅ Found {field_name} (fallback): ${amount:,.2f}")
                elif prefer_largest and number_contexts:
                    # For wages, use largest unused amount
                    for amount, line_idx, line, context in number_contexts:
                        if amount not in used_amounts:
                            extracted[field_name] = amount
                            used_amounts.add(amount)
                            logger.info(f"✅ Found {field_name} (largest): ${amount:,.2f}")
                            break
        
        # 7. Extract tax year
        year_matches = re.findall(r'\b(20\d{2})\b', text)
        if year_matches:
            years = [int(y) for y in year_matches if 2020 <= int(y) <= 2025]
            if years:
                extracted['tax_year'] = max(years)
                logger.info(f"✅ Found tax year: {extracted['tax_year']}")
        
        return extracted

    def smart_field_extraction(self, text: str) -> Dict[str, Any]:
        """Main extraction method using structured parsing"""
        return self.parse_w2_structured_text(text)

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process document with structured W2 extraction"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            logger.info(f"Processing {file_ext} document with structured W2 extraction")

            if file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                text, confidence = self.extract_text_from_image(file_path)
                extracted_fields = self.smart_field_extraction(text)
                best_confidence = confidence
                
            elif file_ext == '.pdf':
                # Focus on the W2 page (usually page 2)
                all_extracted = {}
                best_confidence = 0.0
                all_text = ""
                
                # Try multiple pages, but prioritize page 2 for W2 forms
                page_priorities = [1, 0, 2]  # Page 2 first, then 1, then 3
                
                for page_num in page_priorities:
                    try:
                        text, confidence = self.extract_text_from_pdf_page(file_path, page_num)
                        
                        if not text or len(text.strip()) < 50:
                            continue
                        
                        # Check if this page contains W2 structure
                        w2_indicators = ['wage and tax statement', 'w-2', 'social security number', 'employer identification']
                        has_w2_structure = any(indicator in text.lower() for indicator in w2_indicators)
                        
                        if has_w2_structure or page_num == 1:  # Always try page 2, or if W2 indicators found
                            logger.info(f"\n=== PROCESSING PAGE {page_num + 1} (W2 Structure: {has_w2_structure}) ===")
                            page_extracted = self.smart_field_extraction(text)
                            
                            # Prefer results from pages with W2 structure
                            if has_w2_structure or not all_extracted:
                                for key, value in page_extracted.items():
                                    if key not in all_extracted or not all_extracted[key] or has_w2_structure:
                                        all_extracted[key] = value
                            
                            best_confidence = max(best_confidence, confidence)
                            all_text += text + "\n"
                        
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
            w2_indicators = ['w-2', 'wage and tax statement', 'form w-2', 'employer identification', 'social security number']
            has_structure = any(indicator in text.lower() for indicator in w2_indicators)
            has_key_fields = any(key in extracted_fields for key in ['employee_ssn', 'employer_ein', 'wages_tips_compensation'])
            has_minimum_data = len(extracted_fields) >= 3
            
            is_w2 = has_structure or (has_key_fields and has_minimum_data)

            logger.info(f"🏁 STRUCTURED W2 EXTRACTION RESULTS:")
            logger.info(f"   Is W2: {is_w2}")
            logger.info(f"   Has structure: {has_structure}")
            logger.info(f"   Has key fields: {has_key_fields}")
            logger.info(f"   Fields found: {len(extracted_fields)}")
            logger.info(f"   Confidence: {best_confidence:.2f}")
            logger.info(f"   Final extracted fields:")
            for key, value in extracted_fields.items():
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
