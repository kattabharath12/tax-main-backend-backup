from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_cpa = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    documents = relationship("Document", back_populates="user")
    tax_returns = relationship("TaxReturn", back_populates="user")
    payments = relationship("Payment", back_populates="user")
    w2_forms = relationship("W2Form", back_populates="user")

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    filename = Column(String)
    file_path = Column(String)
    file_type = Column(String)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)
    extraction_status = Column(String, default="pending")  # pending, processing, completed, failed

    user = relationship("User", back_populates="documents")

class W2Form(Base):
    __tablename__ = "w2_forms"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    document_id = Column(Integer, ForeignKey("documents.id"))

    # Employer Information
    employer_name = Column(String)
    employer_address = Column(Text)
    employer_ein = Column(String)

    # Employee Information
    employee_ssn = Column(String)
    employee_name = Column(String)
    employee_address = Column(Text)

    # Box Information
    wages_tips_compensation = Column(Float)  # Box 1
    federal_income_tax_withheld = Column(Float)  # Box 2
    social_security_wages = Column(Float)  # Box 3
    social_security_tax_withheld = Column(Float)  # Box 4
    medicare_wages = Column(Float)  # Box 5
    medicare_tax_withheld = Column(Float)  # Box 6
    social_security_tips = Column(Float)  # Box 7
    allocated_tips = Column(Float)  # Box 8
    dependent_care_benefits = Column(Float)  # Box 10
    nonqualified_plans = Column(Float)  # Box 11

    # State and Local Information
    state_wages = Column(Float)  # Box 16
    state_income_tax = Column(Float)  # Box 17
    local_wages = Column(Float)  # Box 18
    local_income_tax = Column(Float)  # Box 19

    # Additional fields
    tax_year = Column(Integer)
    raw_extracted_data = Column(JSON)  # Store raw OCR results
    confidence_score = Column(Float)  # OCR confidence

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="w2_forms")
    document = relationship("Document")

class TaxReturn(Base):
    __tablename__ = "tax_returns"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    tax_year = Column(Integer)
    income = Column(Float)
    deductions = Column(Float)
    withholdings = Column(Float)
    tax_owed = Column(Float)
    refund_amount = Column(Float)
    amount_owed = Column(Float)
    status = Column(String, default="draft")  # draft, submitted, processed
    created_at = Column(DateTime, default=datetime.utcnow)
    submitted_at = Column(DateTime)

    user = relationship("User", back_populates="tax_returns")
    payments = relationship("Payment", back_populates="tax_return")

class Payment(Base):
    __tablename__ = "payments"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    tax_return_id = Column(Integer, ForeignKey("tax_returns.id"))
    amount = Column(Float)
    payment_method = Column(String)
    status = Column(String, default="pending")  # pending, completed, failed
    transaction_id = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="payments")
    tax_return = relationship("TaxReturn", back_populates="payments")