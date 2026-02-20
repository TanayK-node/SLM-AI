import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

index = None
documents = []
def chunk_text(text, chunk_size=300):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

def load_full_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def ingest_folder(folder_path):
    global index, documents
    
    documents = []
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if filename.endswith(".txt") or filename.endswith(".py"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                chunks = chunk_text(text)
                documents.extend(chunks)


    embeddings = model.encode(documents)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

def retrieve(query, top_k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    
    results = [documents[i] for i in indices[0]]
    return results
