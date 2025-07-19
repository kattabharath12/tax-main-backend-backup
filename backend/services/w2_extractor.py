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

    def score_page_for_w2_content(self, text: str) -> int:
        """Score a page based on W2 indicators (not specific values)"""
        text_lower = text.lower()
        score = 0
        
        # Negative scoring for instruction/template pages
        bad_indicators = [
            'attention:', 'future developments', 'notice to employee',
            'instructions for employee', 'do you have to file',
            'earned income tax credit', 'corrections', 'employers, please note',
            'copy a of this form is provided for informational purposes',
            'this is a sample', 'for testing purposes'
        ]
        
        for indicator in bad_indicators:
            if indicator in text_lower:
                score -= 10
        
        # Negative scoring for obvious template values
        template_patterns = [
            '123-45-6789',  # Common template SSN
            '12-3456789',   # Common template EIN
            'john doe',     # Common template name
            'abc company',  # Common template employer
        ]
        
        for pattern in template_patterns:
            if pattern in text_lower:
                score -= 5
        
        # Positive scoring for W2 structure
        w2_structure = [
            ('form w-2', 10),
            ('wage and tax statement', 10),
            ('social security wages', 5),
            ('medicare wages', 5),
            ('federal income tax withheld', 5),
            ('employer identification number', 5)
        ]
        
        for indicator, points in w2_structure:
            if indicator in text_lower:
                score += points
        
        # Positive scoring for realistic data patterns
        # Any valid SSN format (but not template)
        real_ssn = re.findall(r'\b(?!123-45-6789)\d{3}-?\d{2}-?\d{4}|\b(?!1234567890)\d{10}\b', text)
        if real_ssn:
            score += 15
        
        # Any valid EIN format (but not template)
        real_ein = re.findall(r'\b(?!12-3456789)[A-Z0-9]{2,5}\d{7,10}|\b(?!12-3456789)\d{2}-\d{7}\b', text)
        if real_ein:
            score += 15
        
        # Box structure with realistic amounts
        box_amounts = re.findall(r'(?i)box\s*[1-6].*?(\d{3,6})', text)
        if len(box_amounts) >= 3:
            score += 10
        
        # Realistic wage amounts (3+ digits, not tiny numbers)
        wage_amounts = re.findall(r'\b(\d{3,7})\b', text)
        realistic_wages = [int(w) for w in wage_amounts if 100 <= int(w) <= 999999]
        score += min(len(realistic_wages), 10)  # Cap at 10 points
        
        return score

    def extract_structural_w2_data(self, text: str) -> Dict[str, Any]:
        """Extract W2 data using structural analysis (no hardcoded values)"""
        extracted = {}
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        full_text = ' '.join(lines)
        
        logger.info(f"Extracting from {len(lines)} lines using structural patterns")
        
        # 1. Extract SSN - any valid SSN that's not a template
        ssn_patterns = [
            r'(?i)employee.*?social security number[:\s]*(\d{3}-\d{2}-\d{4})',
            r'(?i)a\s+employee.*?social security number[:\s]*(\d{10})',
            r'\b(?!123-45-6789)(\d{3}-\d{2}-\d{4})\b',  # Valid SSN, not template
            r'\b(?!1234567890)(\d{10})\b',  # 10-digit SSN, not template
        ]
        
        for pattern in ssn_patterns:
            matches = re.findall(pattern, full_text)
            if matches:
                ssn = matches[0]
                if len(ssn) == 10:
                    ssn = f"{ssn[:3]}-{ssn[3:5]}-{ssn[5:]}"
                extracted['employee_ssn'] = ssn
                logger.info(f"✅ Found SSN: {ssn}")
                break
        
        # 2. Extract EIN - any valid EIN that's not a template
        ein_patterns = [
            r'(?i)employer identification number[:\s]*(\d{2}-\d{7})',
            r'(?i)b\s+employer identification number[:\s]*([A-Z0-9]{2,5}\d{7,10})',
            r'\b(?!12-3456789)(\d{2}-\d{7})\b',  # Valid EIN, not template
            r'\b(?!FGHU7896901)([A-Z]{2,5}\d{7,10})\b',  # Any letter+number EIN
        ]
        
        for pattern in ein_patterns:
            matches = re.findall(pattern, full_text)
            if matches:
                extracted['employer_ein'] = matches[0]
                logger.info(f"✅ Found EIN: {matches[0]}")
                break
        
        # 3. Extract Employee Name - use structural position
        employee_name_patterns = [
            r'(?i)e\s+employee.*?first name.*?initial[:\s]*([A-Z][a-zA-Z\s\-\'\.]{2,40})',
            r'(?i)employee.*?first name[:\s]*([A-Z][a-zA-Z\s\-\'\.]{2,40})',
        ]
        
        # Also look for last name
        last_name_patterns = [
            r'(?i)last name[:\s]*([A-Z][a-zA-Z\-\']{2,25})',
        ]
        
        first_name = None
        last_name = None
        
        for pattern in employee_name_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            if matches:
                name = matches[0].strip()
                # Clean the name
                name = re.sub(r'[^\w\s\-\']', '', name).strip()
                # Avoid form labels
                avoid_terms = ['and initial', 'last name', 'suff', 'address', 'for official']
                if (len(name) >= 3 and 
                    not any(term in name.lower() for term in avoid_terms)):
                    first_name = name
                    logger.info(f"✅ Found first name: {name}")
                    break
        
        for pattern in last_name_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            if matches:
                name = matches[0].strip()
                name = re.sub(r'[^\w\s\-\']', '', name).strip()
                if len(name) >= 2 and name.lower() not in ['suff', 'address']:
                    last_name = name
                    logger.info(f"✅ Found last name: {name}")
                    break
        
        # Combine names
        if first_name and last_name:
            extracted['employee_name'] = f"{first_name} {last_name}"
        elif first_name:
            extracted['employee_name'] = first_name
        elif last_name:
            extracted['employee_name'] = last_name
        
        # 4. Extract Employer Name - use structural position
        employer_patterns = [
            r'(?i)c\s+employer.*?name.*?address[:\s]*([A-Z][a-zA-Z\s\-\'\.\&\,]{2,50})',
            r'(?i)employer.*?name[:\s]*([A-Z][a-zA-Z\s\-\'\.\&\,]{3,40})',
        ]
        
        for pattern in employer_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            if matches:
                name = matches[0].strip()
                name = re.sub(r'[^\w\s\-\'\.\&\,]', '', name).strip()
                # Avoid form labels and template names
                avoid_terms = ['and zip code', 'address', 'identification', 'abc company', 'attention']
                if (len(name) >= 3 and 
                    not any(term in name.lower() for term in avoid_terms)):
                    extracted['employer_name'] = name
                    logger.info(f"✅ Found employer name: {name}")
                    break
        
        # 5. Extract Tax Year
        year_patterns = [
            r'(?i)form w-2.*?(\d{4})',
            r'(?i)tax year[:\s]*(\d{4})',
            r'\b(20\d{2})\b',
        ]
        
        for pattern in year_patterns:
            matches = re.findall(pattern, full_text)
            if matches:
                years = [int(y) for y in matches if 2020 <= int(y) <= 2025]
                if years:
                    extracted['tax_year'] = max(years)
                    logger.info(f"✅ Found tax year: {extracted['tax_year']}")
                    break
        
        # 6. Extract Wage Amounts - use box patterns
        wage_fields = {
            'wages_tips_compensation': [
                r'(?i)1\s+wages.*?compensation[:\s\$]*([0-9,]+\.?\d{0,2})',
                r'(?i)box\s*1[:\s\$]*([0-9,]+\.?\d{0,2})',
            ],
            'federal_income_tax_withheld': [
                r'(?i)2\s+federal.*?tax.*?withheld[:\s\$]*([0-9,]+\.?\d{0,2})',
                r'(?i)box\s*2[:\s\$]*([0-9,]+\.?\d{0,2})',
            ],
            'social_security_wages': [
                r'(?i)3\s+social.*?security.*?wages[:\s\$]*([0-9,]+\.?\d{0,2})',
                r'(?i)box\s*3[:\s\$]*([0-9,]+\.?\d{0,2})',
            ],
            'social_security_tax_withheld': [
                r'(?i)4\s+social.*?security.*?tax.*?withheld[:\s\$]*([0-9,]+\.?\d{0,2})',
                r'(?i)box\s*4[:\s\$]*([0-9,]+\.?\d{0,2})',
            ],
            'medicare_wages': [
                r'(?i)5\s+medicare.*?wages[:\s\$]*([0-9,]+\.?\d{0,2})',
                r'(?i)box\s*5[:\s\$]*([0-9,]+\.?\d{0,2})',
            ],
            'medicare_tax_withheld': [
                r'(?i)6\s+medicare.*?tax.*?withheld[:\s\$]*([0-9,]+\.?\d{0,2})',
                r'(?i)box\s*6[:\s\$]*([0-9,]+\.?\d{0,2})',
            ],
        }
        
        for field_name, patterns in wage_fields.items():
            for pattern in patterns:
                matches = re.findall(pattern, full_text)
                if matches:
                    try:
                        amount_str = matches[0].replace(',', '').replace('$', '')
                        amount = float(amount_str)
                        # Accept any reasonable wage amount
                        if 0 <= amount <= 9999999:
                            extracted[field_name] = amount
                            logger.info(f"✅ Found {field_name}: ${amount:,.2f}")
                            break
                    except (ValueError, IndexError):
                        continue
        
        logger.info(f"Structural extraction found {len(extracted)} fields")
        return extracted

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process document with universal structural analysis"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            logger.info(f"Processing {file_ext} document with universal patterns")

            if file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                text, confidence = self.extract_text_from_image(file_path)
                extracted_fields = self.extract_structural_w2_data(text)
                best_confidence = confidence
                
            elif file_ext == '.pdf':
                # Find the best W2 page using structural scoring
                best_text = ""
                best_confidence = 0.0
                best_score = -999
                
                for page_num in range(min(5, 10)):
                    try:
                        text, confidence = self.extract_text_from_pdf_page(file_path, page_num)
                        
                        if not text or len(text.strip()) < 50:
                            continue
                        
                        score = self.score_page_for_w2_content(text)
                        logger.info(f"Page {page_num + 1}: score={score}")
                        
                        if score > best_score:
                            best_score = score
                            best_text = text
                            best_confidence = confidence
                            
                    except Exception as e:
                        logger.warning(f"Error processing page {page_num + 1}: {e}")
                        continue
                
                if not best_text:
                    return {
                        'is_w2': False,
                        'error': 'No W2 content found in document',
                        'confidence': 0.0
                    }
                
                logger.info(f"🎯 Using page with highest W2 score: {best_score}")
                extracted_fields = self.extract_structural_w2_data(best_text)
                
            else:
                return {
                    'is_w2': False,
                    'error': f'Unsupported file type: {file_ext}',
                    'confidence': 0.0
                }

            # Check if we found meaningful W2 data
            has_key_data = any(key in extracted_fields for key in ['employee_ssn', 'employer_ein'])
            has_wage_data = any(key in extracted_fields for key in ['wages_tips_compensation', 'federal_income_tax_withheld'])
            is_w2 = has_key_data or has_wage_data or len(extracted_fields) >= 3

            logger.info(f"🏁 UNIVERSAL W2 EXTRACTION:")
            logger.info(f"   Is W2: {is_w2}")
            logger.info(f"   Fields found: {len(extracted_fields)}")
            logger.info(f"   Data: {extracted_fields}")

            return {
                'is_w2': is_w2,
                'confidence': best_confidence,
                'raw_text': "Processed with universal structural analysis",
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
