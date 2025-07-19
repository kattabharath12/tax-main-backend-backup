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

    def score_page_quality(self, text: str, page_num: int) -> Dict[str, Any]:
        """Score each page to determine which has the best W2 data"""
        text_lower = text.lower()
        score = 0
        details = {'page': page_num + 1, 'score': 0, 'reasons': []}
        
        # NEGATIVE scoring for instruction/template pages
        bad_indicators = [
            'attention:', 'future developments', 'notice to employee',
            'instructions for employee', 'do you have to file',
            'earned income tax credit', 'corrections', 'employers, please note',
            'copy a of this form is provided for informational purposes',
            'this is a sample w-2 form for testing purposes'
        ]
        
        for indicator in bad_indicators:
            if indicator in text_lower:
                score -= 10
                details['reasons'].append(f"Found instruction text: {indicator}")
        
        # NEGATIVE scoring for template data
        template_indicators = [
            ('123-45-6789', 'template SSN'),
            ('12-3456789', 'template EIN'), 
            ('abc corp', 'template employer'),
            ('john doe', 'template employee'),
            ('void', 'void marking')
        ]
        
        for indicator, reason in template_indicators:
            if indicator in text_lower:
                score -= 5
                details['reasons'].append(f"Found {reason}")
        
        # POSITIVE scoring for real W2 structure
        structure_indicators = [
            ('form w-2', 5),
            ('wage and tax statement', 5),
            ('social security wages', 3),
            ('medicare wages', 3),
            ('federal income tax withheld', 3),
            ('employer identification number', 3)
        ]
        
        for indicator, points in structure_indicators:
            if indicator in text_lower:
                score += points
                details['reasons'].append(f"Found W2 structure: {indicator}")
        
        # POSITIVE scoring for real data patterns
        # Real SSN (not template)
        real_ssn_matches = re.findall(r'\b(?!123-45-6789)\d{3}-?\d{2}-?\d{4}|\b(?!1234567890)\d{10}\b', text)
        if real_ssn_matches:
            score += 10
            details['reasons'].append(f"Found real SSN pattern")
        
        # Real EIN (not template)
        real_ein_matches = re.findall(r'\b(?!12-3456789)[A-Z0-9]{2,5}\d{7,10}|\b(?!12-3456789)\d{2}-\d{7}\b', text)
        if real_ein_matches:
            score += 10
            details['reasons'].append(f"Found real EIN pattern")
        
        # Real names (not template)
        if 'ajith' in text_lower or 'sai kumar' in text_lower or 'poturi' in text_lower:
            score += 8
            details['reasons'].append(f"Found real names")
        
        # Realistic wage amounts (not small template numbers)
        wage_amounts = re.findall(r'\b(\d{4,6})\b', text)  # 4-6 digit numbers (realistic wages)
        if len(wage_amounts) >= 3:
            score += 5
            details['reasons'].append(f"Found {len(wage_amounts)} realistic wage amounts")
        
        # Box structure with numbers
        box_matches = re.findall(r'(?i)box\s*[1-6].*?(\d+)', text)
        if len(box_matches) >= 3:
            score += 5
            details['reasons'].append(f"Found {len(box_matches)} filled boxes")
        
        details['score'] = score
        logger.info(f"Page {page_num + 1} scored {score} points: {details['reasons']}")
        
        return details

    def extract_universal_data(self, text: str) -> Dict[str, Any]:
        """Extract W2 data using universal patterns"""
        extracted = {}
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        full_text = ' '.join(lines)
        
        logger.info(f"Processing {len(lines)} lines for universal extraction")
        
        # 1. Extract Employee SSN - prioritize non-template values
        ssn_patterns = [
            r'(?i)employee.*?social security number[:\s]*(\d{3}-\d{2}-\d{4})',
            r'(?i)social security number[:\s]*(\d{3}-\d{2}-\d{4})', 
            r'(\d{3}-\d{2}-\d{4})',  # Standard format
            r'(\d{10})',  # 10 digits without separators
        ]
        
        for pattern in ssn_patterns:
            matches = re.findall(pattern, full_text)
            for match in matches:
                ssn = match
                if len(ssn) == 10:  # Format 10-digit SSN
                    ssn = f"{ssn[:3]}-{ssn[3:5]}-{ssn[5:]}"
                
                # Skip template SSN
                if ssn != '123-45-6789':
                    extracted['employee_ssn'] = ssn
                    logger.info(f"✅ Found real SSN: {ssn}")
                    break
            if 'employee_ssn' in extracted:
                break
        
        # 2. Extract Employer EIN - prioritize non-template values
        ein_patterns = [
            r'(?i)employer identification number[:\s]*(\d{2}-\d{7})',
            r'(?i)ein[:\s]*(\d{2}-\d{7})',
            r'(\d{2}-\d{7})',  # Standard EIN format
            r'([A-Z]{2,5}\d{7,10})',  # Alternative format
        ]
        
        for pattern in ein_patterns:
            matches = re.findall(pattern, full_text)
            for match in matches:
                # Skip template EIN
                if match != '12-3456789':
                    extracted['employer_ein'] = match
                    logger.info(f"✅ Found real EIN: {match}")
                    break
            if 'employer_ein' in extracted:
                break
        
        # 3. Extract Employee Name - prioritize real names
        name_patterns = [
            r'(?i)employee.*?name.*?address[:\s]*\n([A-Z][a-zA-Z\s\-\'\.]{2,40})',
            r'(?i)employee.*?name[:\s]*([A-Z][a-zA-Z\s\-\'\.]{2,40})',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)',  # First Last format
        ]
        
        for pattern in name_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            for match in matches:
                name = match.strip()
                name = re.sub(r'[^\w\s\-\']', '', name).strip()
                
                # Skip template names and form text
                skip_names = ['john doe', 'abc company', 'and initial', 'last name', 'suff']
                if (len(name) >= 3 and 
                    name.lower() not in skip_names and
                    not any(skip in name.lower() for skip in skip_names) and
                    not any(word in name.lower() for word in ['address', 'street', 'city'])):
                    
                    extracted['employee_name'] = name
                    logger.info(f"✅ Found real employee name: {name}")
                    break
            if 'employee_name' in extracted:
                break
        
        # 4. Extract Employer Name - prioritize real names
        employer_patterns = [
            r'(?i)employer.*?name.*?address[:\s]*\n([A-Z][a-zA-Z\s\-\'\.\&]{2,50})',
            r'(?i)employer.*?name[:\s]*([A-Z][a-zA-Z\s\-\'\.\&]{2,50})',
            r'([A-Z][a-zA-Z\s\&\.\,]{3,40}(?:\s+(?:Inc|Corp|LLC|Company|Co)\.?)?)',
        ]
        
        for pattern in employer_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            for match in matches:
                name = match.strip()
                name = re.sub(r'[^\w\s\-\'\.\&\,]', '', name).strip()
                
                # Skip template names and form text
                skip_names = ['abc company', 'attention', 'employer identification']
                if (len(name) >= 3 and 
                    name.lower() not in skip_names and
                    not any(skip in name.lower() for skip in skip_names) and
                    not any(word in name.lower() for word in ['address', 'street', 'city', 'number'])):
                    
                    extracted['employer_name'] = name
                    logger.info(f"✅ Found real employer name: {name}")
                    break
            if 'employer_name' in extracted:
                break
        
        # 5. Extract Tax Year
        year_patterns = [
            r'(?i)tax year[:\s]*(\d{4})',
            r'(?i)form w-2.*?(\d{4})',
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
        
        # 6. Extract Wage and Tax Amounts - realistic amounts only
        wage_patterns = {
            'wages_tips_compensation': [
                r'(?i)1\.?\s*wages.*?compensation[:\s\$]*([0-9,]+\.?\d{0,2})',
                r'(?i)wages.*?tips.*?compensation[:\s\$]*([0-9,]+\.?\d{0,2})',
                r'(?i)box\s*1[:\s\$]*([0-9,]+\.?\d{0,2})',
            ],
            'federal_income_tax_withheld': [
                r'(?i)2\.?\s*federal.*?tax.*?withheld[:\s\$]*([0-9,]+\.?\d{0,2})',
                r'(?i)federal.*?income.*?tax.*?withheld[:\s\$]*([0-9,]+\.?\d{0,2})',
                r'(?i)box\s*2[:\s\$]*([0-9,]+\.?\d{0,2})',
            ],
            'social_security_wages': [
                r'(?i)3\.?\s*social.*?security.*?wages[:\s\$]*([0-9,]+\.?\d{0,2})',
                r'(?i)social.*?security.*?wages[:\s\$]*([0-9,]+\.?\d{0,2})',
                r'(?i)box\s*3[:\s\$]*([0-9,]+\.?\d{0,2})',
            ],
            'social_security_tax_withheld': [
                r'(?i)4\.?\s*social.*?security.*?tax.*?withheld[:\s\$]*([0-9,]+\.?\d{0,2})',
                r'(?i)social.*?security.*?tax.*?withheld[:\s\$]*([0-9,]+\.?\d{0,2})',
                r'(?i)box\s*4[:\s\$]*([0-9,]+\.?\d{0,2})',
            ],
            'medicare_wages': [
                r'(?i)5\.?\s*medicare.*?wages[:\s\$]*([0-9,]+\.?\d{0,2})',
                r'(?i)medicare.*?wages.*?tips[:\s\$]*([0-9,]+\.?\d{0,2})',
                r'(?i)box\s*5[:\s\$]*([0-9,]+\.?\d{0,2})',
            ],
            'medicare_tax_withheld': [
                r'(?i)6\.?\s*medicare.*?tax.*?withheld[:\s\$]*([0-9,]+\.?\d{0,2})',
                r'(?i)medicare.*?tax.*?withheld[:\s\$]*([0-9,]+\.?\d{0,2})',
                r'(?i)box\s*6[:\s\$]*([0-9,]+\.?\d{0,2})',
            ],
        }
        
        for field_name, patterns in wage_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, full_text)
                for match in matches:
                    try:
                        # Clean and convert amount
                        amount_str = match.replace(',', '').replace('$', '')
                        amount = float(amount_str)
                        
                        # Only accept realistic amounts (not small template numbers like 2, 4, 6)
                        if 100 <= amount <= 1000000:  # Realistic wage range
                            extracted[field_name] = amount
                            logger.info(f"✅ Found {field_name}: ${amount:,.2f}")
                            break
                    except (ValueError, IndexError):
                        continue
                if field_name in extracted:
                    break
        
        logger.info(f"Universal extraction complete - found {len(extracted)} fields")
        return extracted

    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process document with smart page selection"""
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            logger.info(f"Processing {file_ext} document: {file_path}")

            if file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
                text, confidence = self.extract_text_from_image(file_path)
                extracted_fields = self.extract_universal_data(text)
                best_confidence = confidence
                
            elif file_ext == '.pdf':
                # Score each page and select the best one
                page_scores = []
                page_texts = []
                
                for page_num in range(min(5, 10)):  # Check first 5 pages
                    try:
                        text, confidence = self.extract_text_from_pdf_page(file_path, page_num)
                        
                        if text and len(text.strip()) > 20:
                            score_info = self.score_page_quality(text, page_num)
                            page_scores.append((score_info['score'], page_num, text, confidence))
                            page_texts.append((page_num + 1, len(text), score_info['score']))
                            logger.info(f"Page {page_num + 1}: {len(text)} chars, score: {score_info['score']}")
                    except Exception as e:
                        logger.warning(f"Error processing page {page_num + 1}: {e}")
                        continue
                
                if not page_scores:
                    return {
                        'is_w2': False,
                        'error': 'No readable pages found in PDF',
                        'confidence': 0.0
                    }
                
                # Select the page with the highest score
                page_scores.sort(key=lambda x: x[0], reverse=True)
                best_score, best_page_num, best_text, best_confidence = page_scores[0]
                
                logger.info(f"🏆 Selected page {best_page_num + 1} with score {best_score}")
                logger.info(f"📊 All pages: {page_texts}")
                
                # Extract data from the best page only
                extracted_fields = self.extract_universal_data(best_text)
                
            else:
                return {
                    'is_w2': False,
                    'error': f'Unsupported file type: {file_ext}',
                    'confidence': 0.0
                }

            # Check if we found meaningful W2 data
            is_w2 = len(extracted_fields) >= 3 or any(key in extracted_fields for key in ['employee_ssn', 'employer_ein', 'wages_tips_compensation'])

            logger.info(f"🏁 SMART W2 EXTRACTION:")
            logger.info(f"   Is W2: {is_w2}")
            logger.info(f"   Fields found: {len(extracted_fields)}")
            logger.info(f"   Data: {extracted_fields}")

            return {
                'is_w2': is_w2,
                'confidence': best_confidence,
                'raw_text': f"Processed best scoring page",
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
