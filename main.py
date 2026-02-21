from fastapi import FastAPI
from pydantic import BaseModel
from app.engine.model import generate_response
from app.engine.rag import retrieve, ingest_folder

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat(request: QueryRequest):
    # Retrieve context from FAISS
    context_chunks = retrieve(request.query)
    context = "\n\n---\n\n".join(context_chunks)
    
    # Improved prompt to enforce critical evaluation rather than summarization
    prompt = f"""
    You are an expert analytical engine. Use the following retrieved context to comprehensively answer the user's question or evaluate the provided logic. 
    If the context contains code, evaluate its structure and methodology critically.

    Context:
    {context}

    Question:
    {request.query}
    
    Answer:
    """

    # Await the asynchronous model call
    answer = await generate_response(prompt)
    return {"response": answer}

@app.post("/ingest")
def ingest():
    ingest_folder("data")  # create a data folder in root
    return {"status": "Ingestion complete"}