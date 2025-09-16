# DocLLM
It's an LLM which accepts a document and answers questions related to the document's text.
# Hackathon: Intelligent Query-Retrieval System

This project is an LLM-powered system that answers questions about a given document.

## Setup
1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `.\venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file and add your `GROQ_API_KEY`.
6. Run the server: `uvicorn main:app --reload`
