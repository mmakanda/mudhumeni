# agribot_complete_streamlit.py
"""
AgriBot Streamlit - Single-file Streamlit dashboard + local RAG.

Run:
    python agribot_complete_streamlit.py

Dependencies (install in the same environment you run the app):
    pip install streamlit pandas plotly langchain-community chromadb sentence-transformers transformers faiss-cpu python-multipart openpyxl beautifulsoup4 requests

Notes:
 - Embedding model: sentence-transformers/all-mpnet-base-v2
 - Generator model: google/flan-t5-small (CPU-friendly)
 - Vector store persist: ./vector_db
"""

import os
import io
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from bs4 import BeautifulSoup

# LangChain Community (Chroma)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.chroma import Chroma

# Optional document type
try:
    from langchain.schema import Document
except Exception:
    from langchain_community.docstore.document import Document

# Transformers generator
from transformers import pipeline

# CONFIG
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
GEN_MODEL = os.getenv("GEN_MODEL", "google/flan-t5-small")
PERSIST_DIR = "vector_db"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 5

Path(PERSIST_DIR).mkdir(exist_ok=True)

st.set_page_config(page_title="AgriBot Streamlit", layout="wide")
st.title("🌾 AgriBot — Streamlit RAG Dashboard")

# ------------------------- Helpers / Initialization -------------------------
@st.cache_resource
def init_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})

@st.cache_resource
def init_vector_store(embeddings):
    # Attempt to load existing Chroma store; else create new
    try:
        store = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        return store
    except Exception:
        # create new
        doc = Document(page_content="Initialization document", metadata={"source": "system"})
        store = Chroma.from_documents(documents=[doc], embedding=embeddings, persist_directory=PERSIST_DIR)
        store.persist()
        return store

@st.cache_resource
def init_generator():
    return pipeline("text2text-generation", model=GEN_MODEL, device=-1, max_length=256)

embeddings = init_embeddings()
vector_store = init_vector_store(embeddings)
generator = init_generator()

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

def ingest_text(name: str, text: str, metadata: dict = None):
    metadata = metadata or {}
    chunks = chunk_text(text)
    docs = [Document(page_content=c, metadata={**metadata, "source": name}) for c in chunks]
    vector_store.add_documents(docs)
    vector_store.persist()
    return len(chunks)

def ingest_file_upload(uploaded_file, farming_type="general"):
    name = uploaded_file.name
    content = uploaded_file.read()
    text = ""
    try:
        if name.lower().endswith(".pdf"):
            # naive PDF fallback — recommend pre-processing with a PDF loader for production
            text = f"[PDF file uploaded: {name}]"
        elif name.lower().endswith(".csv"):
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
        else:
            text = content.decode(errors="ignore")
    except Exception as e:
        text = content.decode(errors="ignore")
    added = ingest_text(name, text, metadata={"farming_type": farming_type})
    return {"filename": name, "added_chunks": added}

def query_rag(question: str, top_k: int = TOP_K):
    if not question:
        return {"answer": "Please enter a question", "sources": []}
    try:
        results = vector_store.similarity_search(question, k=top_k)
        context = "\n\n".join([getattr(d, "page_content", str(d)) for d in results if d])
        prompt = f"Use the context to answer the question. If unknown, say you don't know.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        out = generator(prompt, max_length=256)[0]["generated_text"]
        sources = [getattr(d, "metadata", {}) for d in results]
        return {"answer": out, "sources": sources}
    except Exception as e:
        return {"answer": f"Error: {e}", "sources": []}

# ------------------------- UI Layout -------------------------
with st.sidebar:
    st.header("Tools")
    st.write("Embedding model:")
    st.caption(EMBEDDING_MODEL)
    if st.button("Fetch latest RBZ USD rate"):
        with st.spinner("Fetching RBZ..."):
            rbz = fetch_rbz_rate()
            st.json(rbz)
    st.markdown("---")
    st.write("Vector store:")
    try:
        # try to show basic info
        st.write(f"Persist dir: {PERSIST_DIR}")
    except Exception:
        st.write("Vector store not available")

tab1, tab2, tab3 = st.tabs(["Dashboard", "Knowledge Base (Ingest)", "Chat (RAG)"])

# Dashboard Tab
with tab1:
    st.header("Excel / Product Performance Insights")
    uploaded = st.file_uploader("Upload Excel / CSV for analysis", type=["xlsx", "xls", "csv"])
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded)
            else:
                sheets = pd.read_excel(uploaded, sheet_name=None)
                # pick first sheet
                first_sheet = next(iter(sheets.keys()))
                df = sheets[first_sheet]
            st.subheader("Preview")
            st.dataframe(df.head())

            # Basic product performance logic: look for typical columns
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            if numeric_cols:
                st.subheader("Numeric KPIs")
                kpi = {c: float(df[c].sum()) for c in numeric_cols}
                st.write(kpi)
                fig = px.line(df[numeric_cols].fillna(0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns detected for KPIs. Try uploading a sales/performance sheet.")
        except Exception as e:
            st.error(f"Failed to parse file: {e}\n{traceback.format_exc()}")

# Knowledge Base Tab
with tab2:
    st.header("Knowledge Base — Upload & Ingest")
    farm_type = st.selectbox("Farming type", options=["general", "crop_farming", "poultry_farming", "fish_farming"])
    uploaded_files = st.file_uploader("Upload files (CSV / XLSX / PDF / TXT)", accept_multiple_files=True)
    if st.button("Ingest selected files"):
        if not uploaded_files:
            st.warning("Select files first")
        else:
            results = []
            for f in uploaded_files:
                res = ingest_file_upload(f, farming_type=farm_type)
                results.append(res)
            st.success("Ingestion complete")
            st.json(results)
    st.markdown("---")
    st.write("Or paste text to ingest:")
    name = st.text_input("Source name", value="manual_text")
    text_area = st.text_area("Text to ingest", height=150)
    if st.button("Ingest text"):
        if not text_area.strip():
            st.warning("Enter text")
        else:
            added = ingest_text(name, text_area, metadata={"farming_type": farm_type})
            st.success(f"Added {added} chunks from pasted text")

# Chat Tab
with tab3:
    st.header("Ask questions about your data / knowledge base")
    q = st.text_input("Your question", "")
    k = st.slider("Top-K retrieval", min_value=1, max_value=12, value=TOP_K)
    if st.button("Ask"):
        with st.spinner("Searching and generating answer..."):
            res = query_rag(q, top_k=k)
            st.subheader("Answer")
            st.write(res["answer"])
            if res.get("sources"):
                st.subheader("Top retrieved sources (metadata snippets)")
                for i, s in enumerate(res["sources"], 1):
                    st.write(f"{i}. {s}")

st.markdown("---")
st.caption("AgriBot Streamlit — uses open-source models. Persisted vector DB: ./vector_db")
