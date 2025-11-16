# production_rag_api.py
"""
Production RAG FastAPI backend.

Run:
    python production_rag_api.py

Dependencies:
    pip install fastapi uvicorn langchain-community chromadb sentence-transformers transformers pandas python-multipart beautifulsoup4 requests

Endpoints:
 - POST /ingest/upload/  (files: UploadFile[])
 - POST /ingest/text/    (name, text)
 - POST /query/          (question, top_k)
 - GET  /rbz/            (fetch RBZ USD rate)
 - GET  /status/         (service status)
"""

import os
import io
import time
import traceback
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import requests
from bs4 import BeautifulSoup
import logging

# langchain-community imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma

# transformers generation
from transformers import pipeline

# Document model
try:
    from langchain.schema import Document
except Exception:
    from langchain_community.docstore.document import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("production_rag_api")

# CONFIG
PERSIST_DIR = "prod_vector_db"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
GEN_MODEL = os.getenv("GEN_MODEL", "google/flan-t5-small")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
DEFAULT_K = int(os.getenv("TOP_K", 5))

os.makedirs(PERSIST_DIR, exist_ok=True)

app = FastAPI(title="Production RAG API - Agri")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Initialize embeddings and vector store (cached)
logger.info("Loading embeddings model...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})

logger.info("Initializing/loading Chroma vector store...")
try:
    vector_store = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    logger.info("✅ Chroma vector store loaded.")
except Exception as e:
    logger.warning("Chroma load failed, creating new store. Error: %s", e)
    vector_store = Chroma.from_documents(documents=[Document(page_content="Initialization")], embedding=embeddings, persist_directory=PERSIST_DIR)
    vector_store.persist()
    logger.info("✅ New Chroma store created.")

logger.info("Loading generator model...")
generator = pipeline("text2text-generation", model=GEN_MODEL, device=-1, max_length=256)

# Utilities
def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def fetch_rbz_rate():
    RBZ_URL = "https://www.rbz.co.zw/exchange-rates"
    try:
        r = requests.get(RBZ_URL, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        usd_rate = None
        for tr in soup.select("tr"):
            tds = [td.get_text(strip=True) for td in tr.find_all("td")]
            if not tds:
                continue
            if any("US" in t or "USD" in t or "US Dollar" in t for t in tds) and len(tds) > 1:
                usd_rate = tds[1]
                break
        return {"usd_rate": usd_rate, "source": RBZ_URL}
    except Exception as e:
        return {"usd_rate": None, "error": str(e), "source": RBZ_URL}

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    top_k: int = DEFAULT_K

@app.post("/ingest/text/")
async def ingest_text(name: str, text: str):
    try:
        chunks = chunk_text(text)
        docs = [Document(page_content=c, metadata={"source": name}) for c in chunks]
        vector_store.add_documents(docs)
        vector_store.persist()
        return {"status": "ok", "chunks": len(chunks)}
    except Exception as e:
        logger.error("Ingest text error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest/upload/")
async def ingest_upload(files: List[UploadFile] = File(...), farming_type: str = "general"):
    added = 0
    details = []
    for f in files:
        try:
            content = await f.read()
            name = f.filename
            text = ""
            if name.lower().endswith(".csv"):
                df = pd.read_csv(io.BytesIO(content))
                text = df.to_csv(index=False)
            elif name.lower().endswith(".xlsx") or name.lower().endswith(".xls"):
                sheets = pd.read_excel(io.BytesIO(content), sheet_name=None)
                combined = []
                if isinstance(sheets, dict):
                    for sname, sframe in sheets.items():
                        combined.append(f"--- sheet: {sname} ---\n")
                        combined.append(sframe.to_csv(index=False))
                text = "\n".join(combined)
            elif name.lower().endswith(".pdf"):
                text = f"[PDF uploaded: {name}]"
            else:
                text = content.decode(errors="ignore")
            chunks = chunk_text(text)
            docs = [Document(page_content=c, metadata={"source": name, "farming_type": farming_type}) for c in chunks]
            vector_store.add_documents(docs)
            added += len(chunks)
            details.append({"filename": name, "chunks": len(chunks)})
        except Exception as e:
            details.append({"filename": f.filename, "error": str(e)})
    vector_store.persist()
    return {"status": "ok", "added_chunks": added, "details": details}

@app.post("/query/")
async def query(req: QueryRequest):
    try:
        question = req.question
        top_k = req.top_k
        try:
            results = vector_store.similarity_search(question, k=top_k)
            context = "\n\n".join([getattr(d, "page_content", str(d)) for d in results if d])
            sources = [getattr(d, "metadata", {}) for d in results]
        except Exception as e:
            logger.warning("Vector search failed: %s", e)
            context = ""
            sources = []
        prompt = f"Use the context to answer the question. If unknown, say you don't know.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        out = generator(prompt, max_length=256)[0]["generated_text"]
        return {"answer": out, "sources": sources}
    except Exception as e:
        logger.error("Query error: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rbz/")
async def rbz():
    return fetch_rbz_rate()

@app.get("/status/")
async def status():
    return {"ok": True, "embedding_model": EMBEDDING_MODEL, "gen_model": GEN_MODEL, "persist_dir": PERSIST_DIR}

# Run
if __name__ == "__main__":
    import uvicorn
    print("Starting Production RAG API at http://0.0.0.0:8000")
    uvicorn.run("production_rag_api:app", host="0.0.0.0", port=8000, reload=False)
