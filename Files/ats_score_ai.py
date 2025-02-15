import warnings
import os
import re
import json
from typing import Optional, List
import numpy as np
from pdf2image import convert_from_path
import pytesseract
from PyPDF2 import PdfReader
from nltk.tokenize import word_tokenize
import google.generativeai as genai
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import ChatGoogleGenerativeAI

# Suppress warnings
warnings.filterwarnings('ignore')


api_key="" # Enter your api key here
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")
es = Elasticsearch("https://localhost:9200/", basic_auth=("username", "password"), verify_certs=False)


def extract_text_from_pdfs() -> dict:
    """Extract text content from PDF files in the current directory."""
    page_content = {}


    pytesseract.pytesseract.tesseract_cmd=f" " #enter your tesseract.exe path
    # Specify the path to the poppler binaries
    poppler_path = r"" # enter your poppler path

# Get the full path to the current directory
    current_directory = os.getcwd()

# Get all the PDF files in the current directory
    pdf_files = [f for f in os.listdir() if f.endswith('.pdf')]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(current_directory, pdf_file)  # Join the current directory with the file name
    
        pdf_reader = PdfReader(pdf_path)
        if pdf_reader.pages:
            text = pdf_reader.pages[0].extract_text()
            if text:
                page_content[pdf_file] = text
            else:
                # Pass the full path to convert_from_path
                images = convert_from_path(pdf_path, poppler_path=poppler_path)
                text = pytesseract.image_to_string(images[0])
                page_content[pdf_file] = text

    return page_content

def preprocess_text(page_content: dict) -> dict:
    """Preprocess extracted text from PDFs."""
    import nltk
    nltk.download('punkt')
    nltk.download('punkt_tab')
    for pdf_file, text in page_content.items():
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = word_tokenize(text)
        page_content[pdf_file] = ' '.join(tokens)
    return page_content

def call_gemini_pro_for_json(cv_text: str) -> Optional[dict]:
    """Convert CV text to JSON format using Gemini Pro."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
    Convert the following text, which is extracted from 6 different CVs, into a JSON format.
    Try to extract as much information as possible, organizing it into 6 different keys like:

    - "ID" : Give a unique id to all person. type of id is integer
    - "name": The full name of the person.'all_can_json_data'
    - "contact": The phone number and email.
    - "address": The full address
    - "education": A list of educational qualifications, including the institution, degree, and years.
    - "total_experience" : over all work experience (thier total work experience) in number.
    - "experience": A list of work experiences, including the company, job title, description and years for years create two fileds one for work duration and second for total years of expeince in this you have to count total years of expeince in thier particular copany when they change thier job in different company then you have to again count thier total years of expeince(it must be integer without decimal point value) in thier particular copany.
    - "skills": A list of skills.
    - "summary": A summary or objective from the CV.
    - "projects" : A list of projects

    Here is the CV text:
    {cv_text}

    Ensure the output is valid JSON. 
    """

    try:
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip()
        
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:].strip()
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3].strip()

        return json.loads(cleaned_response)
    except Exception as e:
        print(f"Error processing JSON: {e}")
        return None




def process_and_index_data(es: Elasticsearch, cv_json: dict) -> None:
    """Process and index CV data in Elasticsearch."""
    index = 'all_can_json_data'

    try:
        if es.indices.exists(index=index):
            print(f"Index {index} is present")
            
            query = {"query": {"match_all": {}}}
            search_response = es.search(index=index, body=query)

            duplicate_found = False
            for hit in search_response['hits']['hits']:
                if hit['_source'] == cv_json:
                    print(f"Duplicate document found with ID: {hit['_id']}")
                    duplicate_found = True
                    break

            if not duplicate_found:
                response = es.index(index=index, document=cv_json)
                print(f"Document indexed with ID: {response['_id']}")
        else:
            print(f"Index {index} is not present")
            inp = input("Do you want to create the index? [Y/n] ")

            if inp.lower() == 'y':
                if es.indices.create(index=index):
                    print(f"Index {index} created successfully")
                    response = es.index(index=index, document=cv_json)
                    print(f"Document indexed with ID: {response['_id']}")

    except Exception as e:
        print(f"Error: {e}")


def search_data(prompt):
    """
    Fetch data from Elasticsearch and use it as context for generating a response based on the prompt.
    """
    index = "all_can_json_data"
    try:
        if es.indices.exists(index=index):
            query = {
                "query": {
                    "match_all": {}
                }
            }
            search_response = es.search(index=index, body=query)
            existing_documents = [hit['_source'] for hit in search_response['hits']['hits']]
        else:
            print(f"Index {index} is not present.")
            existing_documents = []

    except Exception as e:
        print(f"Error fetching data from Elasticsearch: {e}")
        existing_documents = []

    if existing_documents:
        context_text = "\n".join([str(doc) for doc in existing_documents])
        full_prompt = f'''
        Here is some data from Elasticsearch: \n{context_text}\nNow, answer the following: {prompt}.
        
        - "Name": The full name of the person.
        - "Phone Number": The phone number.
        - "Email": The email address.
        - "address": The full address.
        - "education": A list of educational qualifications, including highest degree and total number of years.
        - "experience": A list of latest work experiences, including the company, job title, description, and total years.
        - "skills": A list of skills.
        
        Ensure the output is must be above Dictonary formate. Do not wrap anser into [] or anything it must be under curly braces.'''
    else:
        full_prompt = prompt

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(full_prompt)
        cleaned_response = response.text.strip() if hasattr(response, 'text') else response.strip()

        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:].strip()
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3].strip()

        try:
            json_data = json.loads(cleaned_response)
            return json_data
        except json.JSONDecodeError as e:
            print(f"Problematic Response Text: {cleaned_response}")
            return None
    except Exception as e:
        print(f"Error generating content: {e}")
        return None


def create_vector_index(es: Elasticsearch, page_content: dict) -> None:
    """Create and populate vector index for semantic search."""
    sen_model = SentenceTransformer("all-MiniLM-L6-v2")
    sentences = list(page_content.values())
    embeddings = sen_model.encode(sentences)
    
    index_name = 'cv_vectors'
    mappings = {
        "properties": {
            "text": {"type": "text"},
            "vector": {"type": "dense_vector", "dims": 384}
        }
    }

    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body={"mappings": mappings})
        print(f"Index '{index_name}' created!")

    for i, (filename, sentence) in enumerate(page_content.items()):
        vector_list = embeddings[i].tolist() if isinstance(embeddings[i], np.ndarray) else embeddings[i]
        doc = {
            'text': sentence,
            'vector': vector_list
        }
        
        existing_doc = es.get(index=index_name, id=filename, ignore=404)
        if existing_doc.get('found'):
            if existing_doc['_source']['text'] != sentence:
                es.index(index=index_name, id=filename, document=doc)
                print(f"Updated document for {filename}")
        else:
            es.index(index=index_name, id=filename, document=doc)
            print(f"Indexed new document for {filename}")

def search_and_return_text(input_keyword: str) -> List[Document]:
    """Search for similar documents using vector similarity."""
    sen_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    vector_of_input_keyword = sen_model.encode(input_keyword)
    query = {
        "field": "vector",
        "query_vector": vector_of_input_keyword,
        "k": 8,
        "num_candidates": 8
    }
    
    res = es.knn_search(index="cv_vectors", knn=query, source=['text'])
    return [Document(page_content=hit['_source']['text'], metadata={}) for hit in res['hits']['hits']]

def retrieve_answers(query: str) -> Optional[dict]:
    """Retrieve answers using LangChain QA chain."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    chain = load_qa_chain(llm=llm, chain_type='stuff')
    
    doc_search = search_and_return_text(query)
    formatted_query = f'''
    Here is some data from Elasticsearch which is in list formate but you must be give answer on json fromate: Now, answer the following: {query}.
        
        - "Name": The full name of the person.
        - "Phone Number": The phone number.
        - "Email": The email address.
        - "address": The full address.
        - "education": A list of educational qualifications, including highest degree and total number of years.
        - "experience": A list of latest work experiences, including the company, job title, description, and total years.
        - "skills": A list of skills.
        
        Ensure the output is must be above Dictonary formate. Do not wrap anser into [] or anything it must be under curly braces.
'''
    
    try:
        response = chain.run(input_documents=doc_search, question=formatted_query)
        cleaned_response = response.strip()
        
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:].strip()
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3].strip()
            
        return json.loads(cleaned_response)
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    # Extract and preprocess text from PDFs
    page_content = extract_text_from_pdfs()
    processed_content = preprocess_text(page_content)
    
    # Convert to JSON using Gemini Pro
    cv_json = call_gemini_pro_for_json(processed_content)
    
    if cv_json:
        # Save JSON data
        with open("cv_data.json", "w") as json_file:
            json.dump(cv_json, json_file, indent=4)
        print("JSON data has been saved to 'cv_data.json'.")
        
        # Setup Elasticsearch and process data
        if es.ping():
            process_and_index_data(es, cv_json)
            create_vector_index(es, processed_content)
        else:
            print("Failed to connect to Elasticsearch")

if __name__ == "__main__":
    main()