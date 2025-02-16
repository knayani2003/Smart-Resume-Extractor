# Smart CV Extractor

## Project Architecture
![Smart CV Extractor Architecture](Smart%20Cv%20Extractor_page-0001.jpg)

## Overview
Smart CV Extractor is an intelligent resume parsing and analysis system that leverages AI to extract, process, and analyze resume data. The system uses Google's Gemini 1.5 Flash model for text processing, Elasticsearch for data storage and retrieval, and provides both traditional and vector-based semantic search capabilities.

## System Architecture
The system consists of three main layers:
1. **Input Layer**
   - Collect PDFs of Resumes
   - Extract Text from PDFs
   - Clean the Extracted Text
   - Convert Data into JSON Format
   - Store JSON into Elasticsearch
   - Convert Data into Vectors
   - Store Vectors into Elasticsearch

2. **Process Layer**
   - GEN AI Model "gemini-1.5-flash"
   - GEN AI Framework LangChain

3. **Output Layer**
   - API Framework FastAPI
   - User Interface with Streamlit
   - Admin or User Access

## Features
- PDF text extraction with fallback OCR support
- Intelligent resume data parsing and structuring
- Dual search mechanisms:
  - Traditional Elasticsearch-based search
  - Semantic vector search using sentence transformers
- FastAPI backend for robust API endpoints
- User-friendly Streamlit interface
- Vector embeddings for enhanced search accuracy

## Technology Stack
- **AI/ML**: 
  - Google Gemini 1.5 Flash
  - Sentence Transformers
  - LangChain
- **Backend**: 
  - FastAPI
  - Elasticsearch
- **Frontend**: 
  - Streamlit
- **PDF Processing**: 
  - PyPDF2
  - pdf2image
  - Tesseract OCR
- **Data Processing**: 
  - NLTK
  - NumPy

## Setup and Installation

1. Clone the repository:
```bash
git clone [repository-url]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
- Set up Google API key
- Configure Elasticsearch credentials
- Set Tesseract and Poppler paths

4. Start Elasticsearch service

5. Run the application:
```bash
# Start FastAPI server
uvicorn main:app --reload

# Start Streamlit interface
streamlit run userio.py
```

## Project Structure
- `ats_score_ai.py`: Core logic for CV processing and analysis
- `main.py`: FastAPI server implementation
- `userio.py`: Streamlit user interface
- `requirements.txt`: Project dependencies

## API Endpoints
- `/search/`: Traditional Elasticsearch-based search
- `/vector-search/`: Semantic vector search
- `/health`: System health check endpoint

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
