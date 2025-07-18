# TaxBox.AI Backend

FastAPI backend for TaxBox.AI tax filing application.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Copy environment file:
```bash
cp .env.example .env
```

3. Run the application:
```bash
uvicorn main:app --reload
```

## API Documentation

Visit http://localhost:8000/docs for interactive API documentation.

## Features

- User registration and authentication
- Document upload
- Tax return creation and calculation
- Payment processing (stub)
- JWT-based security
