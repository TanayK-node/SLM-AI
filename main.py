from fastapi import FastAPI
from pydantic import BaseModel
from app.engine.model import generate_response
from app.engine.rag import retrieve
from app.engine.rag import ingest_folder

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
def chat(request: QueryRequest):
    context = retrieve(request.query)
    
    prompt = f"""
    Use the following context to answer the question.

    Context:
    {context}

    Question:
    {request.query}
    """

    answer = generate_response(prompt)
    return {"response": answer}

@app.post("/ingest")
def ingest():
    ingest_folder("data")  # create a data folder in root
    return {"status": "Ingestion complete"}