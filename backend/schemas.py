from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional, Dict, Any

class UserCreate(BaseModel):
    email: EmailStr
    full_name: str
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    is_active: bool
    is_cpa: bool
    created_at: datetime

    class Config:
        from_attributes = True

class DocumentResponse(BaseModel):
    id: int
    filename: str
    file_type: str
    uploaded_at: datetime
    processed: bool
    extraction_status: str

    class Config:
        from_attributes = True

class W2FormCreate(BaseModel):
    document_id: int
    employer_name: Optional[str] = None
    employer_address: Optional[str] = None
    employer_ein: Optional[str] = None
    employee_ssn: Optional[str] = None
    employee_name: Optional[str] = None
    employee_address: Optional[str] = None
    wages_tips_compensation: Optional[float] = None
    federal_income_tax_withheld: Optional[float] = None
    social_security_wages: Optional[float] = None
    social_security_tax_withheld: Optional[float] = None
    medicare_wages: Optional[float] = None
    medicare_tax_withheld: Optional[float] = None
    social_security_tips: Optional[float] = None
    allocated_tips: Optional[float] = None
    dependent_care_benefits: Optional[float] = None
    nonqualified_plans: Optional[float] = None
    state_wages: Optional[float] = None
    state_income_tax: Optional[float] = None
    local_wages: Optional[float] = None
    local_income_tax: Optional[float] = None
    tax_year: Optional[int] = None
    raw_extracted_data: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = None

class W2FormResponse(BaseModel):
    id: int
    document_id: int
    employer_name: Optional[str]
    employer_address: Optional[str]
    employer_ein: Optional[str]
    employee_ssn: Optional[str]
    employee_name: Optional[str]
    employee_address: Optional[str]
    wages_tips_compensation: Optional[float]
    federal_income_tax_withheld: Optional[float]
    social_security_wages: Optional[float]
    social_security_tax_withheld: Optional[float]
    medicare_wages: Optional[float]
    medicare_tax_withheld: Optional[float]
    social_security_tips: Optional[float]
    allocated_tips: Optional[float]
    dependent_care_benefits: Optional[float]
    nonqualified_plans: Optional[float]
    state_wages: Optional[float]
    state_income_tax: Optional[float]
    local_wages: Optional[float]
    local_income_tax: Optional[float]
    tax_year: Optional[int]
    confidence_score: Optional[float]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class TaxReturnCreate(BaseModel):
    tax_year: int
    income: float
    deductions: Optional[float] = None
    withholdings: float = 0

class TaxReturnResponse(BaseModel):
    id: int
    tax_year: int
    income: float
    deductions: float
    withholdings: float
    tax_owed: float
    refund_amount: float
    amount_owed: float
    status: str
    created_at: datetime

    class Config:
        from_attributes = True

class PaymentCreate(BaseModel):
    tax_return_id: int
    amount: float

class PaymentResponse(BaseModel):
    id: int
    amount: float
    payment_method: str
    status: str
    created_at: datetime

    class Config:
        from_attributes = True

class W2ExtractionResult(BaseModel):
    success: bool
    message: str
    w2_data: Optional[W2FormResponse] = None
    confidence_score: Optional[float] = None
    raw_data: Optional[Dict[str, Any]] = None