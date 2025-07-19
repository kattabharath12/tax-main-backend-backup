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
        # Standard W2 box mapping for reference
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
            '9': '',  # Verification code
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

    def extract_all_numbers_and_text(self, text: str) -> Dict[str, List]:
        """Extract all numbers and text chunks for analysis"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        all_numbers = []
        all_text_chunks = []
        
        for i, line in enumerate(lines):
            # Find all numbers in this line - improved pattern for currency
            numbers = re.findall(r'\b(\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+(?:\.\d{2})?)\b', line)
            for num in numbers:
                # Clean and validate number
                clean_num = num.replace(',', '')
                if len(clean_num) >= 2 and clean_num.replace('.', '').isdigit():
                    all_numbers.append((clean_num, i, line))
            
            # Find text chunks (potential names) - improved pattern
            text_chunks = re.findall(r'\b([A-Z][A-Za-z]{1,}(?:\s+[A-Z][A-Za-z]{1,})*)\b', line)
            for chunk in text_chunks:
                if len(chunk) >= 2:
                    all_text_chunks.append((chunk, i, line))
        
        return {
            'numbers': all_numbers,
            'text_chunks': all_text_chunks,
            'lines': lines
        }

    def is_ssn_format(self, text: str) -> bool:
        """Check if text matches SSN format"""
        # Remove any separators
        clean = re.sub(r'[-\s]', '', text)
        if len(clean) == 9 and clean.isdigit():
            # Avoid common test/placeholder SSNs
            invalid_ssns = ['123456789', '000000000', '111111111', '222222222']
            return clean not in invalid_ssns
        return False

    def is_ein_format(self, text: str) -> bool:
        """Check if text matches EIN format"""
        # EIN can be XX-XXXXXXX or similar patterns
        if re.match(r'^[A-Z0-9]{2,5}[-]?\d{7,10}$', text) or re.match(r'^\d{2}-\d{7}$', text):
            invalid_eins = ['12-3456789', 'XX-XXXXXXX']
            return text not in invalid_eins
        return False

    def extract_ssn(self, text: str) -> Optional[str]:
        """Extract SSN with improved logic"""
        # Look for formatted SSNs first
        ssn_patterns = [
            r'\b(\d{3}-\d{2}-\d{4})\b',
            r'\b(\d{3}\s\d{2}\s\d{4})\b',
            r'\b(\d{9})\b'
        ]
        
        for pattern in ssn_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if self.is_ssn_format(match):
                    # Format consistently
                    clean = re.sub(r'[-\s]', '', match)
                    return f"{clean[:3]}-{clean[3:5]}-{clean[5:]}"
        
        return None

    def extract_ein(self, text: str) -> Optional[str]:
        """Extract EIN with improved logic"""
        ein_patterns = [
            r'\b([A-Z]{2,5}\d{7,10})\b',
            r'\b(\d{2}-\d{7})\b',
            r'\b([A-Z0-9]{10,12})\b'
        ]
        
        for pattern in ein_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if self.is_ein_format(match):
                    return match
        
        return None

    def extract_names(self, data: Dict) -> Dict[str, str]:
        """Extract employee and employer names using context"""
        names = {}
        text_chunks = data['text_chunks']
        lines = data['lines']
        
        # Common words to skip
        skip_terms = {
            'employee', 'employer', 'identification', 'number', 'address', 'code',
            'wages', 'tips', 'compensation', 'federal', 'income', 'tax', 'withheld',
            'social', 'security', 'medicare', 'statement', 'treasury', 'revenue',
            'service', 'privacy', 'paperwork', 'reduction', 'notice', 'instructions',
            'official', 'administration', 'department', 'form', 'copy', 'state',
            'local', 'dependent', 'care', 'benefits', 'nonqualified', 'plans',
            'statutory', 'retirement', 'third', 'party', 'sick', 'pay', 'control',
            'first', 'middle', 'last', 'name', 'initial', 'suffix', 'street',
            'city', 'zip', 'county', 'country', 'telephone', 'email'
        }
        
        potential_names = []
        for chunk, line_idx, line in text_chunks:
            chunk_lower = chunk.lower()
            # Filter out obvious non-names
            if (not any(term in chunk_lower for term in skip_terms) and
                not re.search(r'\d', chunk) and  # No numbers
                len(chunk.split()) <= 4 and  # Not too long
                len(chunk) >= 2):  # Not too short
                
                potential_names.append((chunk, line_idx, line))
        
        # Sort by position in document
        potential_names.sort(key=lambda x: x[1])
        
        # Use heuristics to assign names
        for i, (name, line_idx, line) in enumerate(potential_names):
            line_lower = line.lower()
            
            # Employee name usually appears earlier and has certain context
            if ('employee' not in names and 
                (i < len(potential_names) // 2 or  # Earlier in document
                 any(indicator in line_lower for indicator in ['employee', 'first', 'last']) or
                 line_idx < 20)):  # Early line number
                names['employee'] = name
                logger.info(f"✅ Found employee name: {name} (line {line_idx + 1})")
            
            # Employer name often appears later or has company indicators
            elif ('employer' not in names and 
                  (any(indicator in line_lower for indicator in ['employer', 'company', 'corp', 'inc', 'llc']) or
                   (i >= len(potential_names) // 2 and 'employee' in names))):
                names['employer'] = name
                logger.info(f"✅ Found employer name: {name} (line {line_idx + 1})")
        
        return names

    def extract_wage_amounts(self, data: Dict) -> Dict[str, float]:
        """Extract wage amounts using improved context matching"""
        amounts = {}
        numbers = data['numbers']
        lines = data['lines']
        
        # Build context for each number
        wage_candidates = []
        for num, line_idx, line in numbers:
            try:
                amount = float(num)
                # Filter for realistic amounts
                if 0.01 <= amount <= 9999999:
                    # Get surrounding context
                    context_lines = []
                    for i in range(max(0, line_idx-2), min(len(lines), line_idx+3)):
                        context_lines.append(lines[i].lower())
                    context = ' '.join(context_lines)
                    
                    wage_candidates.append((amount, line_idx, line, context))
            except ValueError:
                continue
        
        # Sort by amount for analysis
        wage_candidates.sort(key=lambda x: x[0], reverse=True)
        
        logger.info(f"Processing {len(wage_candidates)} wage candidates")
        
        # Define field patterns with priorities
        field_patterns = [
            ('wages_tips_compensation', ['wages', 'tips', 'compensation', 'box 1', 'box1'], 1),
            ('federal_income_tax_withheld', ['federal', 'income', 'tax', 'withheld', 'box 2', 'box2'], 2),
            ('social_security_wages', ['social security', 'wages', 'box 3', 'box3'], 3),
            ('social_security_tax_withheld', ['social security', 'tax', 'withheld', 'box 4', 'box4'], 4),
            ('medicare_wages', ['medicare', 'wages', 'box 5', 'box5'], 5),
            ('medicare_tax_withheld', ['medicare', 'tax', 'withheld', 'box 6', 'box6'], 6),
        ]
        
        # First pass: look for exact matches
        for field_name, keywords, box_num in field_patterns:
            if field_name in amounts:
                continue
                
            for amount, line_idx, line, context in wage_candidates:
                # Check if this amount fits the pattern
                keyword_matches = sum(1 for keyword in keywords if keyword in context)
                
                if keyword_matches >= 2:  # Need at least 2 keyword matches
                    amounts[field_name] = amount
                    logger.info(f"✅ Found {field_name}: ${amount:,.2f} (line {line_idx + 1})")
                    break
        
        # Second pass: use amount heuristics if we have wages
        if 'wages_tips_compensation' in amounts:
            wages = amounts['wages_tips_compensation']
            
            # Look for tax amounts that make sense relative to wages
            for amount, line_idx, line, context in wage_candidates:
                if amount >= wages:  # Skip amounts larger than wages
                    continue
                    
                # Federal tax is typically 10-30% of wages
                if ('federal_income_tax_withheld' not in amounts and 
                    0.05 * wages <= amount <= 0.35 * wages and
                    ('federal' in context or 'tax' in context)):
                    amounts['federal_income_tax_withheld'] = amount
                    logger.info(f"✅ Found federal tax (heuristic): ${amount:,.2f}")
                
                # Social Security tax is 6.2% of wages (up to wage base)
                ss_expected = min(wages, 160200) * 0.062  # 2023 wage base
                if ('social_security_tax_withheld' not in amounts and
                    abs(amount - ss_expected) / ss_expected < 0.1 and  # Within 10%
                    ('social' in context or 'security' in context)):
                    amounts['social_security_tax_withheld'] = amount
                    logger.info(f"✅ Found SS tax (heuristic): ${amount:,.2f}")
                
                # Medicare tax is 1.45% of wages
                medicare_expected = wages * 0.0145
                if ('medicare_tax_withheld' not in amounts and
                    abs(amount - medicare_expected) / medicare_expected < 0.1 and
                    'medicare' in context):
                    amounts['medicare_tax_withheld'] = amount
                    logger.info(f"✅ Found Medicare tax (heuristic): ${amount:,.2f}")
        
        # Third pass: assign remaining large amounts as wages if not found
        if 'wages_tips_compensation' not in amounts:
            # Look for the largest reasonable amount
            for amount, line_idx, line, context in wage_candidates:
                if amount >= 1000:  # Minimum reasonable wage
                    amounts['wages_tips_compensation'] = amount
                    logger.info(f"✅ Found wages (largest amount): ${amount:,.2f}")
                    break
        
        return amounts

    def extract_tax_year(self, text: str) -> Optional[int]:
        """Extract tax year"""
        current_year = 2024  # Update as needed
        year_matches = re.findall(r'\b(20\d{2})\b', text)
        
        if year_matches:
            years = [int(y) for y in year_matches if 2020 <= int(y) <= current_year + 1]
            if years:
                # Return the most recent valid year
                year = max(years)
                logger.info(f"✅ Found tax year: {year}")
                return year
        
        return None

    def smart_field_extraction(self, text: str) -> Dict[str, Any]:
        """Extract W2 fields using improved smart pattern matching"""
        extracted = {}
        
        # Get all data
        data = self.extract_all_numbers_and_text(text)
        logger.info(f"Found {len(data['numbers'])} numbers and {len(data['text_chunks'])} text chunks")
        
        # Extract SSN
        ssn = self.extract_ssn(text)
        if ssn:
            extracted['employee_ssn'] = ssn
        
        # Extract EIN
        ein = self.extract_ein(text)
        if ein:
            extracted['employer_ein'] = ein
        
        # Extract names
        names = self.extract_names(data)
        if 'employee' in names:
            extracted['employee_name'] = names['employee']
        if 'employer' in names:
            extracted['employer_name'] = names['employer']
        
        # Extract wage amounts
        amounts = self.extract_wage_amounts(data)
        extracted.update(amounts)
        
        # Extract tax year
        year = self.extract_tax_year(text)
        if year:
            extracted['tax_year'] = year
        
        return extracted

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process document with improved extraction"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            logger.info(f"Processing {file_ext} document")

            if file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                text, confidence = self.extract_text_from_image(file_path)
                extracted_fields = self.smart_field_extraction(text)
                best_confidence = confidence
                
            elif file_ext == '.pdf':
                # Process all pages and combine results
                all_extracted = {}
                best_confidence = 0.0
                all_text = ""
                
                for page_num in range(min(3, 10)):  # First 3 pages
                    try:
                        text, confidence = self.extract_text_from_pdf_page(file_path, page_num)
                        
                        if not text or len(text.strip()) < 50:
                            continue
                        
                        all_text += text + "\n"
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
                
                # Also try processing all text together for better context
                if all_text:
                    combined_extracted = self.smart_field_extraction(all_text)
                    # Fill in any missing fields
                    for key, value in combined_extracted.items():
                        if key not in all_extracted or not all_extracted[key]:
                            all_extracted[key] = value
                
                extracted_fields = all_extracted
                
            else:
                return {
                    'is_w2': False,
                    'error': f'Unsupported file type: {file_ext}',
                    'confidence': 0.0
                }

            # Determine if this is a valid W2
            w2_indicators = ['w-2', 'wage and tax statement', 'form w-2', 'employer identification']
            has_structure = any(indicator in text.lower() for indicator in w2_indicators)
            has_key_fields = any(key in extracted_fields for key in ['employee_ssn', 'employer_ein', 'wages_tips_compensation'])
            has_minimum_data = len(extracted_fields) >= 2
            
            is_w2 = has_structure or (has_key_fields and has_minimum_data)

            logger.info(f"🏁 IMPROVED W2 EXTRACTION RESULTS:")
            logger.info(f"   Is W2: {is_w2}")
            logger.info(f"   Has structure: {has_structure}")
            logger.info(f"   Has key fields: {has_key_fields}")
            logger.info(f"   Fields found: {len(extracted_fields)}")
            logger.info(f"   Confidence: {best_confidence:.2f}")
            logger.info(f"   Extracted fields: {extracted_fields}")

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
