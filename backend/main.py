from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime, timedelta
from typing import Optional, List
import jwt
from passlib.context import CryptContext
import uvicorn
import os
import shutil
import aiofiles
from pathlib import Path
import logging

from database import SessionLocal, engine, Base
from models import User, Document, TaxReturn, Payment, W2Form
from schemas import (
    UserCreate, UserResponse, DocumentResponse, TaxReturnCreate, 
    TaxReturnResponse, PaymentCreate, PaymentResponse, W2FormResponse,
    W2ExtractionResult
)
from services.w2_extractor import W2Extractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database initialization
print("Initializing database...")

def init_database():
    """Initialize database with proper error handling"""
    try:
        print("Testing database connection...")
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            print("✅ Database connection established")
        
        print("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        
        # Verify tables exist
        print("Verifying table creation...")
        table_names = ['users', 'documents', 'tax_returns', 'payments', 'w2_forms']
        
        with SessionLocal() as db:
            for table_name in table_names:
                try:
                    result = db.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                    count = result.scalar()
                    print(f"✅ {table_name} table verified (count: {count})")
                except Exception as e:
                    print(f"❌ {table_name} table issue: {e}")
                    raise e
        
        print("✅ Database initialization completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Initialize the database
db_success = init_database()
if not db_success:
    print("⚠️ Starting app with database issues - some features may not work")

app = FastAPI(title="TaxBox.AI API", version="2.0.0")

@app.on_event("startup")
async def startup_event():
    """Log startup information"""
    try:
        db_url = os.getenv("DATABASE_URL", "sqlite:///./taxbox.db")
        db_type = "PostgreSQL" if "postgresql" in db_url else "SQLite"
        print(f"🚀 TaxBox.AI API started successfully using {db_type}")
        
        # Verify W2 extractor
        try:
            extractor = W2Extractor()
            print("✅ W2 Extractor initialized successfully")
        except Exception as e:
            print(f"❌ W2 Extractor initialization failed: {e}")
            
    except Exception as e:
        print(f"❌ Startup verification error: {e}")

@app.get("/")
def root():
    return {
        "message": "TaxBox.AI API is running!", 
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://tax-main-frontend-backup-production.up.railway.app",
        "https://taxbox-ai-enhanced-backend-1-production-2bcd.up.railway.app",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Security
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize W2 extractor
w2_extractor = W2Extractor()

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Auth functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user

# Enhanced background task for W2 processing
async def process_w2_extraction(document_id: int, file_path: str):
    """Background task to process W2 extraction with better error handling"""
    db = SessionLocal()
    try:
        # Update document status
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            logger.error(f"Document {document_id} not found")
            return

        document.extraction_status = "processing"
        db.commit()
        logger.info(f"Starting W2 processing for document {document_id}")

        # Process the document
        result = w2_extractor.process_document(file_path)
        logger.info(f"W2 processing result for document {document_id}: {result}")

        if result['is_w2'] and not result.get('error'):
            # Create W2 form record
            w2_data = W2Form(
                user_id=document.user_id,
                document_id=document_id,
                raw_extracted_data=result,
                confidence_score=result.get('confidence', 0.0),
                **{k: v for k, v in result['extracted_fields'].items() 
                   if k in ['employer_name', 'employer_address', 'employer_ein', 
                           'employee_ssn', 'employee_name', 'employee_address',
                           'wages_tips_compensation', 'federal_income_tax_withheld',
                           'social_security_wages', 'social_security_tax_withheld',
                           'medicare_wages', 'medicare_tax_withheld', 'social_security_tips',
                           'allocated_tips', 'dependent_care_benefits', 'nonqualified_plans',
                           'state_wages', 'state_income_tax', 'local_wages', 
                           'local_income_tax', 'tax_year']}
            )
            db.add(w2_data)
            document.extraction_status = "completed"
            document.processed = True
            logger.info(f"W2 data saved successfully for document {document_id}")
        else:
            document.extraction_status = "no_w2_detected" if not result['is_w2'] else "failed"
            logger.info(f"W2 processing completed with status: {document.extraction_status}")

        db.commit()

    except Exception as e:
        logger.error(f"Error processing W2 extraction for document {document_id}: {e}")
        # Update document status to failed
        try:
            document = db.query(Document).filter(Document.id == document_id).first()
            if document:
                document.extraction_status = "failed"
                db.commit()
        except Exception as db_error:
            logger.error(f"Error updating document status: {db_error}")
    finally:
        db.close()

# Routes
@app.post("/register", response_model=UserResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    logger.info(f"Registration attempt for: {user.email}")
    
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = get_password_hash(user.password)
    db_user = User(
        email=user.email,
        full_name=user.full_name,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    logger.info(f"✅ User registered successfully: {user.email}")
    return db_user

@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/me", response_model=UserResponse)
def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Validate file type
    allowed_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    # Validate file size (10MB limit)
    file_content = await file.read()
    if len(file_content) > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")

    # Create unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{current_user.id}_{timestamp}_{file.filename}"
    file_path = UPLOAD_DIR / filename

    # Save file
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(file_content)

    # Create document record
    document = Document(
        user_id=current_user.id,
        filename=file.filename,
        file_path=str(file_path),
        file_type=file_ext,
        extraction_status="pending"
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    # Add background task for W2 processing
    background_tasks.add_task(process_w2_extraction, document.id, str(file_path))
    logger.info(f"Document uploaded and W2 processing queued for document {document.id}")

    return document

@app.get("/documents", response_model=List[DocumentResponse])
def get_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    documents = db.query(Document).filter(Document.user_id == current_user.id).all()
    return documents

@app.get("/documents/{document_id}/w2", response_model=W2FormResponse)
def get_w2_data(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Verify document belongs to user
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == current_user.id
    ).first()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Get W2 data
    w2_form = db.query(W2Form).filter(W2Form.document_id == document_id).first()

    if not w2_form:
        raise HTTPException(status_code=404, detail="No W2 data found for this document")

    return w2_form

@app.get("/w2-forms", response_model=List[W2FormResponse])
def get_user_w2_forms(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    w2_forms = db.query(W2Form).filter(W2Form.user_id == current_user.id).all()
    return w2_forms

@app.post("/tax-returns", response_model=TaxReturnResponse)
def create_tax_return(
    tax_return: TaxReturnCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Calculate tax owed (simplified calculation)
    income = tax_return.income
    deductions = tax_return.deductions or 12950  # Standard deduction 2022
    taxable_income = max(0, income - deductions)

    # Simplified tax calculation (you'd use actual tax brackets)
    if taxable_income <= 10275:
        tax_owed = taxable_income * 0.10
    elif taxable_income <= 41775:
        tax_owed = 1027.50 + (taxable_income - 10275) * 0.12
    else:
        tax_owed = 4807.50 + (taxable_income - 41775) * 0.22

    refund_amount = max(0, tax_return.withholdings - tax_owed)
    amount_owed = max(0, tax_owed - tax_return.withholdings)

    db_tax_return = TaxReturn(
        user_id=current_user.id,
        tax_year=tax_return.tax_year,
        income=income,
        deductions=deductions,
        withholdings=tax_return.withholdings,
        tax_owed=tax_owed,
        refund_amount=refund_amount,
        amount_owed=amount_owed
    )

    db.add(db_tax_return)
    db.commit()
    db.refresh(db_tax_return)
    return db_tax_return

@app.get("/tax-returns", response_model=List[TaxReturnResponse])
def get_tax_returns(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    tax_returns = db.query(TaxReturn).filter(TaxReturn.user_id == current_user.id).all()
    return tax_returns

@app.post("/payments", response_model=PaymentResponse)
def create_payment(
    payment: PaymentCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Verify tax return belongs to user
    tax_return = db.query(TaxReturn).filter(
        TaxReturn.id == payment.tax_return_id,
        TaxReturn.user_id == current_user.id
    ).first()

    if not tax_return:
        raise HTTPException(status_code=404, detail="Tax return not found")

    db_payment = Payment(
        user_id=current_user.id,
        tax_return_id=payment.tax_return_id,
        amount=payment.amount,
        payment_method="credit_card",  # Default for now
        status="completed"  # Simplified for demo
    )

    db.add(db_payment)
    db.commit()
    db.refresh(db_payment)
    return db_payment

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.get("/health/db")
def database_health_check():
    """Check database connectivity and table existence"""
    try:
        with SessionLocal() as db:
            # Test connection
            db.execute(text("SELECT 1"))
            
            # Check each table
            tables_status = {}
            for table_name in ['users', 'documents', 'tax_returns', 'payments', 'w2_forms']:
                try:
                    result = db.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                    count = result.scalar()
                    tables_status[table_name] = {"exists": True, "count": count}
                except Exception as e:
                    tables_status[table_name] = {"exists": False, "error": str(e)}
            
            return {
                "database_status": "healthy",
                "tables": tables_status,
                "timestamp": datetime.utcnow()
            }
    except Exception as e:
        return {
            "database_status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow()
        }

# Test endpoint for W2 extraction
@app.post("/test-w2-extraction")
async def test_w2_extraction(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Test endpoint to check W2 extraction without saving to database"""
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.bmp'}
        file_ext = Path(file.filename).suffix.lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # Save temporary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"test_{timestamp}_{file.filename}"
        temp_path = UPLOAD_DIR / temp_filename

        content = await file.read()
        async with aiofiles.open(temp_path, 'wb') as f:
            await f.write(content)

        # Process with W2 extractor
        result = w2_extractor.process_document(str(temp_path))

        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()

        return {
            "success": True,
            "result": result,
            "message": "W2 extraction test completed"
        }

    except Exception as e:
        logger.error(f"Error in test W2 extraction: {e}")
        # Clean up temp file if it exists
        if 'temp_path' in locals() and temp_path.exists():
            temp_path.unlink()
        
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
