import os
import json
import numpy as np
import fitz
import faiss
from sentence_transformers import SentenceTransformer
import ollama
import nltk
import re
from string import punctuation
from nltk.corpus import stopwords
from pydantic import BaseModel
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import logging
from PIL import Image
import pytesseract
from nltk.tokenize import sent_tokenize

# Setup logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

app = FastAPI()

# Global variables
faiss_index = None
embedding_model = None
text_chunks = []

# Pydantic models
class QueryInput(BaseModel):
    query: str

class ResponseOutput(BaseModel):
    answer: str

class SummaryOutput(BaseModel):
    summary: list[str]

# Helper functions
def sanitize_filename(filename: str) -> str:
    return re.sub(r'[^\w\-_.]', '_', filename)

# Core functionality
def load_faiss_index(index_path: str):
    global faiss_index
    try:
        if not os.path.exists(index_path):
            dimension = 384
            faiss_index = faiss.IndexFlatIP(dimension)
            faiss.write_index(faiss_index, index_path)
        else:
            faiss_index = faiss.read_index(index_path)
    except Exception as e:
        logger.error(f"Index error: {e}")
        raise

def load_embedding_model():
    global embedding_model
    try:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        logger.error(f"Model error: {e}")
        raise

def load_text_chunks(directory: str):
    global text_chunks
    try:
        for file in os.listdir(directory):
            if file.endswith('.json'):
                with open(os.path.join(directory, file), 'r') as f:
                    text_chunks.extend(json.load(f))
    except Exception as e:
        logger.error(f"Chunk error: {e}")
        raise

def extract_text_from_pdf(pdf_path: str) -> list:
    try:
        doc = fitz.open(pdf_path)
        extracted = []
        for page in doc:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img)
            extracted.append({
                'page_number': page.number + 1,
                'sentences': sent_tokenize(text)
            })
        return extracted
    except Exception as e:
        logger.error(f"PDF error: {e}")
        raise

def clean_text(extracted: list) -> list:
    stop_words = set(stopwords.words('english'))
    cleaned = []
    for page in extracted:
        clean_sents = [
            re.sub(f'[{punctuation}]', '', 
            re.sub(r'\b(?:{})\b'.format('|'.join(stop_words)), '', 
            sent.lower())).strip()
            for sent in page['sentences']
        ]
        cleaned.append({
            'page_number': page['page_number'],
            'sentences': ' '.join(clean_sents)
        })
    return cleaned

def generate_embeddings(chunks: list) -> np.ndarray:
    texts = [chunk["sentences"] for chunk in chunks]
    return embedding_model.encode(texts).astype('float32')

def retrieve_context(query: str, top_k: int = 3) -> str:
    try:
        query_embed = embedding_model.encode([query]).astype('float32')
        faiss.normalize_L2(query_embed)
        distances, indices = faiss_index.search(query_embed, top_k)
        return ' '.join([text_chunks[idx]["sentences"] for idx in indices[0]])
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return None

def generate_summary(context: str) -> list[str]:
    try:
        prompt = f"""Summarize this text in 6 concise bullet points.
        
        Text: {context}
        
        Format EXACTLY as:
        - Point 1
        - Point 2
        - Point 3
        - Point 4
        - Point 5
        - Point 6"""
        
        response = ollama.generate(
            model="llama3.2:latest",
            prompt=prompt,
            options={"temperature": 0.2}
        )['response'].strip()

        # Parse response
        summary = [line.strip('- ') for line in response.split('\n') if line.strip()]
        
        # Ensure exactly 6 summary points
        summary = summary[:6] + ['']*(6-len(summary)) if len(summary) < 6 else summary[:6]

        return summary
    except Exception as e:
        logger.error(f"Summary generation error: {e}")
        return ["Summary error"]

def generate_response(context: str, question: str) -> str:
    try:
        prompt = f"""Answer the question based on the context provided:
        
        Question: {question}
        
        Context: {context}
        
        Answer: <clear answer>"""
        
        response = ollama.generate(
            model="llama3.2:latest",
            prompt=prompt,
            options={"temperature": 0.2}
        )['response'].strip()

        return response if response else "Answer: Not found"
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return "Answer generation failed"

# API endpoints
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(..., max_size=200_000_000)):
    try:
        # Save PDF
        os.makedirs("uploads", exist_ok=True)
        file_path = f"uploads/{sanitize_filename(file.filename)}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Process PDF
        extracted = extract_text_from_pdf(file_path)
        cleaned = clean_text(extracted)
        
        # Save cleaned data
        data_dir = "./newdata"
        os.makedirs(data_dir, exist_ok=True)
        json_path = os.path.join(data_dir, f"{os.path.splitext(file.filename)[0]}.json")
        with open(json_path, "w") as f:
            json.dump(cleaned, f)
        
        # Update index
        embeddings = generate_embeddings(cleaned)
        faiss.normalize_L2(embeddings)
        faiss_index.add(embeddings)
        text_chunks.extend(cleaned)
        
        return JSONResponse({"message": "PDF processed successfully"})
    
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(500, str(e))

@app.post("/ask_question/", response_model=ResponseOutput)
async def ask_question(query: QueryInput):
    try:
        context = retrieve_context(query.query)
        if not context:
            raise HTTPException(404, "No context found")
        
        answer = generate_response(context, query.query)
        return {"answer": answer}
    
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(500, str(e))

@app.post("/get_summary/", response_model=SummaryOutput)
async def summarize(query: QueryInput):
    try:
        context = retrieve_context(query.query)
        if not context:
            raise HTTPException(404, "No context found")
        
        summary = generate_summary(context)
        return {"summary": summary}
    
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(500, str(e))

# Startup
@app.on_event("startup")
async def startup():
    new_data_dir = "./newdata"
    os.makedirs(new_data_dir, exist_ok=True)
    load_embedding_model()
    load_faiss_index("faiss_index.idx")
    load_text_chunks(new_data_dir)

# Run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
