# main.py - Final Submission Code

import os
import time
import requests
from fastapi import FastAPI, Request, HTTPException, status
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

# LangChain components for our RAG system
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# --- SETUP ---

# Create a global dictionary to cache the retrievers for repeat requests
retriever_cache = {}

# Load environment variables from the .env file
load_dotenv()

# The Bearer token required for hackathon API authentication
REQUIRED_BEARER_TOKEN = "9341a670a250629b02214383a13e8cd1a682a6989890906db0416dd2bcf537fb"

# Initialize our FastAPI application
app = FastAPI(
    title="HackRx Intelligent Query-Retrieval System",
    version="1.0",
    description="A system to answer questions about documents using RAG and a fast LLM.",
)

# --- MIDDLEWARE TO TIME REQUESTS ---
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    print(f"--- Request processed in: {process_time:.4f} seconds ---")
    return response

# --- Pydantic Models for API ---
class APIRequest(BaseModel):
    documents: str = Field(..., description="URL to the PDF document.")
    questions: List[str] = Field(..., description="List of questions to answer.")

class APIResponse(BaseModel):
    answers: List[str]

# --- CORE LOGIC ---
def process_document_and_create_retriever(doc_url: str):
    """
    Downloads a PDF, loads its content, splits it into chunks,
    creates embeddings using a fast local model, and sets up a FAISS vector store.
    """
    try:
        # 1. Download the PDF
        response = requests.get(doc_url)
        response.raise_for_status()
        temp_pdf_path = "temp_document.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(response.content)

        # 2. Load the document text
        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()

        # 3. Split the document into chunks
        # NEW, OPTIMIZED text splitter
        # NEW, MORE BALANCED text splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
        splits = text_splitter.split_documents(docs)

        # 4. Create embeddings using a fast, local, self-contained model.
        # This model is small and CPU-friendly, but will automatically use a GPU if available.
        # NEW, MORE ACCURATE embedding model
        embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

        # 5. Create the FAISS vector store
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

        # 6. Clean up the temporary file
        os.remove(temp_pdf_path)

        # 7. Return the retriever
        return vectorstore.as_retriever(search_kwargs={'k': 5})

    except Exception as e:
        print(f"Error processing document: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to process the document: {e}")

# --- API ENDPOINT ---
@app.post("/api/v1/hackrx/run", response_model=APIResponse)
async def run_submission(request: Request, body: APIRequest):
    """
    This is the main endpoint for the hackathon. It receives a document URL
    and a list of questions, processes them, and returns the answers.
    """
    # 1. Authentication
    auth_header = request.headers.get("Authorization")
    if not auth_header or auth_header.startswith("Bearer ") is False or auth_header.split(" ")[1] != REQUIRED_BEARER_TOKEN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid authentication token")
    
    print("Authentication successful.")

    # 2. Caching Logic
    doc_url = body.documents
    if doc_url in retriever_cache:
        print(f"CACHE HIT: Loading retriever for '{doc_url}' from cache.")
        retriever = retriever_cache[doc_url]
    else:
        print(f"CACHE MISS: Processing document '{doc_url}' for the first time.")
        retriever = process_document_and_create_retriever(doc_url)
        retriever_cache[doc_url] = retriever
    
    print(f"Answering {len(body.questions)} questions...")

    # 3. Setup LLM and RAG Chain
    # NEW, MORE ACCURATE language model
    llm = ChatGroq(model_name="llama3-70b-8192", temperature=0)

    # NEW, MORE FOCUSED prompt
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based ONLY on the provided context.
    Be direct and exact. If the information is not in the context, say "The answer is not available in the provided context."

    Context:
    {context}

    Question:
    {input}

    Answer:
    """)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # 4. Generate Answers
    all_answers = []
    for i, question_text in enumerate(body.questions):
        response = retrieval_chain.invoke({"input": question_text})
        all_answers.append(response["answer"])

    print("All questions answered successfully.")
    return APIResponse(answers=all_answers)