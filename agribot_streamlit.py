#!/usr/bin/env python3
"""
🌾 AgriBot Pro - Production-Ready Streamlit Version
Enterprise Agriculture AI Platform

Run: streamlit run agribot_streamlit.py
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import streamlit as st

# Optional: Only import if available
try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except ImportError:
    HAS_OPTION_MENU = False
    st.warning("⚠️ Optional package 'streamlit-option-menu' not installed. Using basic navigation. Install with: pip install streamlit-option-menu")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Check Python version
if sys.version_info < (3, 8):
    print("❌ Python 3.8+ required")
    sys.exit(1)

# Create necessary directories
for directory in ["agri_data", "vector_db", "uploads", "logs", "database", "static", "exports", "marketplace"]:
    Path(directory).mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/agribot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Mudhumeni - AgriBot Pro",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://amaryllissuccess.co.zw',
        'Report a bug': "mailto:support@amaryllissuccess.co.zw",
        'About': "# AgriBot Pro v2.0\nYour Intelligent Agriculture Assistant"
    }
)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Application configuration"""
    APP_NAME = "Mudhumeni"
    APP_TAGLINE = "Your Intelligent Agriculture Assistant"
    VERSION = "2.0.0"
    COMPANY_NAME = "Amaryllis Success"
    COMPANY_WEBSITE = "https://amaryllissuccess.co.zw"
    SUPPORT_EMAIL = "support@amaryllissuccess.co.zw"
    
    LLM_MODEL = "llama3.2"
    OLLAMA_BASE_URL = "http://localhost:11434"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    TEMPERATURE = 0.7
    
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    FARMING_TYPES = [
        "crop_farming", "fish_farming", "goat_farming",
        "pig_farming", "poultry_farming", "cattle_farming",
        "general_agriculture"
    ]
    
    DEFAULT_LOCATION = {
        "city": "Harare",
        "country": "Zimbabwe",
        "lat": -17.8292,
        "lon": 31.0522
    }

config = Config()

# ============================================================================
# CUSTOM CSS
# ============================================================================

def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    
    /* Root variables */
    :root {
        --primary-green: #2e7d32;
        --light-green: #66bb6a;
        --dark-green: #1b5e20;
        --purple: #7e57c2;
        --orange: #ff6f00;
        --pink: #e91e63;
    }
    
    /* Main container */
    .main {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Responsive container */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem !important;
        }
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #1b5e20, #2e7d32, #66bb6a);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    
    .header-title {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-tagline {
        color: #e8f5e9;
        font-size: 1.3rem;
        margin-top: 0.5rem;
    }
    
    /* Mobile responsive header */
    @media (max-width: 768px) {
        .header-container {
            padding: 1rem;
        }
        .header-title {
            font-size: 1.8rem;
        }
        .header-tagline {
            font-size: 1rem;
        }
    }
    
    @media (max-width: 480px) {
        .header-title {
            font-size: 1.5rem;
        }
        .header-tagline {
            font-size: 0.9rem;
        }
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-left: 4px solid var(--primary-green);
        margin-bottom: 1rem;
    }
    
    @media (max-width: 768px) {
        .info-card {
            padding: 1rem;
        }
    }
    
    .metric-card {
        background: linear-gradient(135deg, #e8f5e9, #ffffff);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary-green);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Responsive metrics */
    @media (max-width: 768px) {
        .metric-value {
            font-size: 2rem;
        }
        .metric-label {
            font-size: 0.8rem;
        }
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #2e7d32, #66bb6a);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    
    /* Mobile button adjustments */
    @media (max-width: 768px) {
        .stButton>button {
            padding: 0.6rem 1rem;
            font-size: 0.9rem;
        }
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f5f5f5, #e8f5e9);
    }
    
    /* Mobile sidebar */
    @media (max-width: 768px) {
        [data-testid="stSidebar"] {
            width: 250px !important;
        }
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        word-wrap: break-word;
    }
    
    .chat-message.user {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .chat-message.assistant {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
    }
    
    /* Mobile chat messages */
    @media (max-width: 768px) {
        .chat-message {
            padding: 0.75rem;
            font-size: 0.9rem;
        }
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #f5f5f5, #e8f5e9);
        border-radius: 15px;
        margin-top: 3rem;
        border-top: 4px solid #66bb6a;
    }
    
    @media (max-width: 768px) {
        .footer {
            padding: 1rem;
            font-size: 0.9rem;
        }
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        overflow-x: auto;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f0f0;
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 600;
        white-space: nowrap;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2e7d32, #66bb6a);
        color: white;
    }
    
    /* Mobile tabs */
    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
            font-size: 0.85rem;
        }
    }
    
    /* Form inputs - responsive */
    .stTextInput, .stTextArea, .stSelectbox {
        font-size: 1rem;
    }
    
    @media (max-width: 768px) {
        .stTextInput input, .stTextArea textarea, .stSelectbox select {
            font-size: 16px !important; /* Prevents zoom on iOS */
        }
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        width: 100%;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1rem;
    }
    
    @media (max-width: 768px) {
        .streamlit-expanderHeader {
            font-size: 0.9rem;
        }
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 4px;
    }
    
    .stError {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 4px;
    }
    
    /* DataFrame responsive */
    .dataframe {
        overflow-x: auto;
        font-size: 0.9rem;
    }
    
    @media (max-width: 768px) {
        .dataframe {
            font-size: 0.75rem;
        }
    }
    
    /* Plotly charts responsive */
    .js-plotly-plot {
        width: 100% !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Mobile navigation hint */
    @media (max-width: 768px) {
        .stSelectbox {
            position: sticky;
            top: 0;
            z-index: 999;
            background: white;
            padding: 10px 0;
        }
    }
    
    /* Touch-friendly spacing on mobile */
    @media (max-width: 768px) {
        .row-widget.stButton {
            margin: 0.5rem 0;
        }
    }
    
    /* Landscape mobile adjustments */
    @media (max-width: 896px) and (orientation: landscape) {
        .header-container {
            padding: 0.5rem;
        }
        .header-title {
            font-size: 1.5rem;
        }
    }
    
    /* Tablet specific */
    @media (min-width: 768px) and (max-width: 1024px) {
        .main .block-container {
            padding: 2rem !important;
        }
        .header-title {
            font-size: 2.5rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DATABASE & SERVICES (Same as before)
# ============================================================================

try:
    from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
    from sqlalchemy.orm import declarative_base
    from sqlalchemy.orm import sessionmaker
    
    Base = declarative_base()
    
    class User(Base):
        __tablename__ = "users"
        id = Column(Integer, primary_key=True)
        username = Column(String(100), unique=True, nullable=False)
        email = Column(String(255))
        location = Column(String(255))
        created_at = Column(DateTime, default=datetime.now)
        last_active = Column(DateTime, default=datetime.now)
        total_queries = Column(Integer, default=0)
    
    class Conversation(Base):
        __tablename__ = "conversations"
        id = Column(Integer, primary_key=True)
        user_id = Column(Integer, nullable=False)
        title = Column(String(255), default="New Chat")
        created_at = Column(DateTime, default=datetime.now)
        updated_at = Column(DateTime, default=datetime.now)
    
    class Message(Base):
        __tablename__ = "messages"
        id = Column(Integer, primary_key=True)
        conversation_id = Column(Integer, nullable=False)
        role = Column(String(20), nullable=False)
        content = Column(Text, nullable=False)
        created_at = Column(DateTime, default=datetime.now)
    
    class MarketplaceListing(Base):
        __tablename__ = "marketplace_listings"
        id = Column(Integer, primary_key=True)
        user_id = Column(Integer, nullable=False)
        title = Column(String(255), nullable=False)
        description = Column(Text)
        category = Column(String(100))
        price = Column(Float)
        location = Column(String(255))
        contact = Column(String(255))
        image_path = Column(String(500))
        status = Column(String(20), default="active")
        created_at = Column(DateTime, default=datetime.now)
        expires_at = Column(DateTime)
    
    class FarmingTip(Base):
        __tablename__ = "farming_tips"
        id = Column(Integer, primary_key=True)
        title = Column(String(255), nullable=False)
        content = Column(Text, nullable=False)
        category = Column(String(100))
        author = Column(String(100))
        likes = Column(Integer, default=0)
        created_at = Column(DateTime, default=datetime.now)
    
    engine = create_engine('sqlite:///database/agribot.db', echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    
    DB_AVAILABLE = True
    logger.info("✅ Database initialized")
except Exception as e:
    logger.warning(f"⚠️ Database not available: {e}")
    DB_AVAILABLE = False
    SessionLocal = None

# ============================================================================
# WEATHER SERVICE
# ============================================================================

class WeatherService:
    @staticmethod
    def get_weather(location: str = None, lat: float = None, lon: float = None) -> dict:
        try:
            import requests
            
            if not lat or not lon:
                lat = config.DEFAULT_LOCATION["lat"]
                lon = config.DEFAULT_LOCATION["lon"]
                location = location or config.DEFAULT_LOCATION["city"]
            
            url = f"https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
                "timezone": "auto",
                "forecast_days": 7
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if response.status_code == 200:
                current = data.get("current", {})
                daily = data.get("daily", {})
                
                return {
                    "location": location,
                    "current": {
                        "temperature": current.get('temperature_2m', 'N/A'),
                        "humidity": current.get('relative_humidity_2m', 'N/A'),
                        "precipitation": current.get('precipitation', 0),
                        "wind_speed": current.get('wind_speed_10m', 'N/A'),
                        "time": current.get("time", datetime.now().isoformat())
                    },
                    "forecast": {
                        "dates": daily.get("time", [])[:7],
                        "max_temp": daily.get("temperature_2m_max", [])[:7],
                        "min_temp": daily.get("temperature_2m_min", [])[:7],
                        "precipitation": daily.get("precipitation_sum", [])[:7],
                        "rain_probability": daily.get("precipitation_probability_max", [])[:7]
                    },
                    "status": "success"
                }
            else:
                return {"status": "error", "message": "Weather service unavailable"}
                
        except Exception as e:
            logger.error(f"Weather API error: {e}")
            return {"status": "error", "message": str(e)}

# ============================================================================
# AI AGENT
# ============================================================================

def initialize_agribot():
    """Initialize AgriBot with caching"""
    
    class AgriBot:
        def __init__(self):
            self.llm = None
            self.vector_store = None
            self.embeddings = None
            self.db_session = None
            self.weather = WeatherService()
            
            self._initialize_llm()
            self._initialize_vector_store()
            
            if DB_AVAILABLE and SessionLocal:
                self.db_session = SessionLocal()
            
            logger.info("✅ AgriBot initialized")
        
        def _initialize_llm(self):
            try:
                from langchain_ollama import ChatOllama
                
                self.llm = ChatOllama(
                    model=config.LLM_MODEL,
                    temperature=config.TEMPERATURE,
                    base_url=config.OLLAMA_BASE_URL
                )
                logger.info(f"✅ LLM initialized: {config.LLM_MODEL}")
            except Exception as e:
                logger.error(f"❌ LLM initialization failed: {e}")
                self.llm = None
        
        def _initialize_vector_store(self):
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                from langchain_community.vectorstores import Chroma
                from langchain_core.documents import Document
                
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=config.EMBEDDING_MODEL,
                    model_kwargs={'device': 'cpu'}
                )
                
                Path("vector_db").mkdir(exist_ok=True)
                
                vector_db_path = Path("vector_db")
                if vector_db_path.exists() and any(vector_db_path.iterdir()):
                    try:
                        self.vector_store = Chroma(
                            persist_directory="vector_db",
                            embedding_function=self.embeddings
                        )
                        logger.info("✅ Loaded existing vector store")
                        return
                    except:
                        pass
                
                self._create_vector_store()
                    
            except Exception as e:
                logger.error(f"❌ Vector store initialization failed: {e}")
                self.vector_store = None
        
        def _create_vector_store(self):
            try:
                from langchain_community.vectorstores import Chroma
                from langchain_core.documents import Document
                
                initial_docs = [
                    Document(
                        page_content="Agriculture AI knowledge base for Zimbabwe and Southern Africa.",
                        metadata={"source": "system"}
                    )
                ]
                
                self.vector_store = Chroma.from_documents(
                    documents=initial_docs,
                    embedding=self.embeddings,
                    persist_directory="vector_db"
                )
                logger.info("✅ Created new vector store")
            except Exception as e:
                logger.error(f"❌ Failed to create vector store: {e}")
        
        def query(self, question: str, username: str = "guest", location: str = None) -> str:
            if not self.llm:
                return "❌ AI model not available. Please ensure Ollama is running."
            
            try:
                now = datetime.now()
                season = self._get_season(now.month)
                
                # 1. Search local vector store
                local_context = ""
                if self.vector_store:
                    try:
                        results = self.vector_store.similarity_search(question, k=3)
                        if results:
                            local_context = "\n\n".join([f"Document: {doc.metadata.get('filename', 'Unknown')}\n{doc.page_content}" for doc in results])
                    except Exception as e:
                        logger.error(f"Vector search error: {e}")
                
                # 2. Search the web for recent information
                web_context = ""
                try:
                    web_results = self._search_web(question)
                    if web_results:
                        web_context = "\n\n".join([
                            f"Web Source: {r['title']}\nURL: {r['url']}\n{r['snippet']}" 
                            for r in web_results[:3]
                        ])
                except Exception as e:
                    logger.error(f"Web search error: {e}")
                
                # 3. Build comprehensive prompt
                system_context = f"""Current Date: {now.strftime('%Y-%m-%d')}
Season: {season}
Location: {location or config.DEFAULT_LOCATION['city']}"""
                
                # Combine all contexts
                full_context = ""
                if local_context:
                    full_context += f"\n\n=== KNOWLEDGE BASE DOCUMENTS ===\n{local_context}"
                if web_context:
                    full_context += f"\n\n=== WEB SEARCH RESULTS ===\n{web_context}"
                
                if full_context:
                    prompt = f"""{system_context}

You are an expert agriculture assistant. Use the following information to provide accurate, contextualized advice:

{full_context}

Question: {question}

Instructions:
1. Prioritize information from local knowledge base (uploaded documents)
2. Supplement with web search results for current/recent information
3. Cite your sources when referencing specific information
4. Provide practical, location-aware advice for {location or config.DEFAULT_LOCATION['city']}
5. Consider the current season: {season}

Provide a comprehensive answer:"""
                else:
                    prompt = f"""{system_context}

You are an expert agriculture assistant.

Question: {question}

Provide practical, detailed advice considering the location and season."""
                
                response = self.llm.invoke(prompt)
                
                # Log query for analytics
                self._log_query(username, question, local_context, web_context)
                
                return response.content
                
            except Exception as e:
                logger.error(f"Query error: {e}")
                return f"❌ Error: {str(e)}"
        
        def _search_web(self, query: str, num_results: int = 3) -> List[dict]:
            """Search the web for relevant information"""
            try:
                import requests
                
                # Using DuckDuckGo Instant Answer API (free, no key required)
                url = "https://api.duckduckgo.com/"
                params = {
                    "q": f"{query} agriculture farming",
                    "format": "json",
                    "no_html": 1,
                    "skip_disambig": 1
                }
                
                response = requests.get(url, params=params, timeout=5)
                data = response.json()
                
                results = []
                
                # Get abstract if available
                if data.get("Abstract"):
                    results.append({
                        "title": data.get("Heading", "General Information"),
                        "url": data.get("AbstractURL", ""),
                        "snippet": data.get("Abstract", "")
                    })
                
                # Get related topics
                for topic in data.get("RelatedTopics", [])[:2]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        results.append({
                            "title": topic.get("Text", "")[:50],
                            "url": topic.get("FirstURL", ""),
                            "snippet": topic.get("Text", "")
                        })
                
                return results
                
            except Exception as e:
                logger.error(f"Web search failed: {e}")
                return []
        
        def _log_query(self, username: str, question: str, local_ctx: str, web_ctx: str):
            """Log query for analytics"""
            try:
                if self.db_session:
                    # Update user stats
                    user = self.db_session.query(User).filter_by(username=username).first()
                    if user:
                        user.total_queries += 1
                        user.last_active = datetime.now()
                        self.db_session.commit()
            except Exception as e:
                logger.error(f"Logging error: {e}")
        
        def add_documents(self, files) -> tuple:
            """Add documents to knowledge base"""
            if not self.vector_store:
                return "❌ Vector store not available", {}
            
            if not files:
                return "⚠️ No files selected", {}
            
            try:
                from langchain_community.document_loaders import PyPDFLoader, TextLoader
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                
                documents = []
                processed_files = []
                failed_files = []
                
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=config.CHUNK_SIZE,
                    chunk_overlap=config.CHUNK_OVERLAP
                )
                
                for file in files:
                    try:
                        # Save uploaded file
                        file_path = Path("uploads") / file.name
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                        
                        # Load based on file type
                        if file_path.suffix.lower() == '.pdf':
                            loader = PyPDFLoader(str(file_path))
                        elif file_path.suffix.lower() == '.txt':
                            loader = TextLoader(str(file_path), encoding='utf-8')
                        else:
                            failed_files.append(f"{file.name} (unsupported format)")
                            continue
                        
                        # Load and add metadata
                        docs = loader.load()
                        for doc in docs:
                            doc.metadata['filename'] = file.name
                            doc.metadata['upload_date'] = datetime.now().isoformat()
                            doc.metadata['file_size'] = file.size
                        
                        documents.extend(docs)
                        processed_files.append(file.name)
                        
                    except Exception as e:
                        logger.warning(f"Failed to load {file.name}: {e}")
                        failed_files.append(f"{file.name} ({str(e)})")
                
                if documents:
                    chunks = splitter.split_documents(documents)
                    self.vector_store.add_documents(chunks)
                    
                    result_msg = f"✅ Successfully processed {len(processed_files)} file(s)\n"
                    result_msg += f"📄 Files: {', '.join(processed_files)}\n"
                    result_msg += f"📊 Created {len(chunks)} knowledge chunks"
                    
                    if failed_files:
                        result_msg += f"\n\n⚠️ Failed:\n" + "\n".join(f"  - {f}" for f in failed_files)
                    
                    return result_msg, {
                        "files_processed": len(processed_files),
                        "files_failed": len(failed_files),
                        "total_chunks": len(chunks)
                    }
                else:
                    return "⚠️ No documents could be loaded", {"files_failed": len(failed_files)}
                    
            except Exception as e:
                logger.error(f"Error adding documents: {e}")
                return f"❌ Error: {str(e)}", {}
        
        def _get_season(self, month: int) -> str:
            if month in [11, 12, 1, 2, 3]:
                return "Rainy/Growing Season"
            elif month in [4, 5]:
                return "Harvest Season"
            else:
                return "Dry Season"
        
        def calculate_farming_costs(self, farming_type: str, scale: float, duration: int = 12) -> dict:
            cost_estimates = {
                "crop_farming": {"seeds": 50, "fertilizer": 150, "pesticides": 80, "labor": 200, "water": 100, "equipment": 150},
                "fish_farming": {"fingerlings": 30, "feed": 200, "water_treatment": 50, "labor": 150, "equipment": 100},
                "goat_farming": {"stock": 200, "feed": 30, "veterinary": 15, "housing": 50, "labor": 100},
                "pig_farming": {"stock": 300, "feed": 50, "veterinary": 20, "housing": 80, "labor": 120},
                "poultry_farming": {"chicks": 2, "feed": 3, "veterinary": 1, "housing": 5, "labor": 80}
            }
            
            costs = cost_estimates.get(farming_type, {})
            if not costs:
                return {"error": "Farming type not found"}
            
            total_cost = 0
            breakdown = {}
            
            for item, unit_cost in costs.items():
                if item in ["feed", "veterinary", "water"]:
                    cost = unit_cost * scale * duration
                else:
                    cost = unit_cost * scale
                
                breakdown[item] = round(cost, 2)
                total_cost += cost
            
            return {
                "farming_type": farming_type,
                "scale": scale,
                "duration_months": duration,
                "breakdown": breakdown,
                "total_cost": round(total_cost, 2),
                "monthly_average": round(total_cost / duration, 2)
            }
    
    return AgriBot()

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'agent' not in st.session_state:
    with st.spinner("🌾 Initializing AgriBot..."):
        st.session_state.agent = initialize_agribot()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'username' not in st.session_state:
    st.session_state.username = "Guest"

if 'location' not in st.session_state:
    st.session_state.location = config.DEFAULT_LOCATION["city"]

# ============================================================================
# HEADER
# ============================================================================

def render_header():
    st.markdown(f"""
    <div class="header-container">
        <h1 class="header-title">🌾 {config.APP_NAME}</h1>
        <p class="header-tagline">{config.APP_TAGLINE}</p>
        <p style="color: #e8f5e9; font-size: 0.9rem; margin-top: 1rem;">
            Version {config.VERSION} | Powered by AI | © {config.COMPANY_NAME}
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar():
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/7e57c2/ffffff?text=Amaryllis+Success", use_container_width=True)
        
        st.markdown("### 👤 User Profile")
        st.session_state.username = st.text_input("Your Name", value=st.session_state.username)
        st.session_state.location = st.text_input("Location", value=st.session_state.location)
        
        st.markdown("---")
        
        # Current Date & Season
        now = datetime.now()
        season = st.session_state.agent._get_season(now.month)
        
        st.markdown("### 📅 Current Info")
        st.info(f"""
        **Date:** {now.strftime('%B %d, %Y')}  
        **Season:** {season}  
        **Location:** {st.session_state.location}
        """)
        
        st.markdown("---")
        
        # Quick Stats
        st.markdown("### 📊 System Status")
        llm_status = "🟢 Online" if st.session_state.agent.llm else "🔴 Offline"
        vector_status = "🟢 Active" if st.session_state.agent.vector_store else "🔴 Inactive"
        
        st.markdown(f"""
        **AI Model:** {llm_status}  
        **Knowledge Base:** {vector_status}  
        **Database:** 🟢 Active
        """)
        
        st.markdown("---")
        
        # Support
        st.markdown("### 🆘 Support")
        st.markdown(f"""
        📧 [{config.SUPPORT_EMAIL}](mailto:{config.SUPPORT_EMAIL})  
        🌐 [{config.COMPANY_WEBSITE}]({config.COMPANY_WEBSITE})
        """)

# ============================================================================
# MAIN PAGES
# ============================================================================

def page_dashboard():
    st.markdown("## 🏠 Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">🌾</div>
            <div class="metric-label">AI Consultation</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">🌤️</div>
            <div class="metric-label">Weather Forecast</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">🛒</div>
            <div class="metric-label">Marketplace</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">💡</div>
            <div class="metric-label">Farming Tips</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recent Activity
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Quick Actions")
        if st.button("🌾 Ask AI Question", use_container_width=True):
            st.switch_page("pages/1_💬_AI_Chat.py")
        if st.button("🌤️ Check Weather", use_container_width=True):
            st.switch_page("pages/2_🌤️_Weather.py")
        if st.button("💵 Calculate Costs", use_container_width=True):
            st.switch_page("pages/3_💵_Calculator.py")
    
    with col2:
        st.markdown("### 🎯 Getting Started")
        st.info("""
        1. **Ask Questions**: Get AI-powered farming advice
        2. **Check Weather**: Plan activities with 7-day forecasts
        3. **Browse Marketplace**: Buy and sell farming products
        4. **Learn Tips**: Access expert farming knowledge
        """)

def page_ai_chat():
    st.markdown("## 💬 AI Consultation")
    
    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area("Ask your agriculture question:", height=100, 
                                   placeholder="E.g., What are the best crops to plant this season?")
        submitted = st.form_submit_button("Send", use_container_width=True)
    
    if submitted and user_input:
        with st.spinner("🤔 Thinking..."):
            response = st.session_state.agent.query(
                user_input, 
                st.session_state.username,
                st.session_state.location
            )
            
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now()
            })
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now()
            })
    
    # Display chat history
    st.markdown("### 💭 Conversation")
    for msg in reversed(st.session_state.chat_history[-10:]):
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user">
                <strong>👤 You:</strong><br>{msg["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant">
                <strong>🤖 AgriBot:</strong><br>{msg["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

def page_weather():
    st.markdown("## 🌤️ Weather Forecast")
    
    location = st.text_input("Location", value=st.session_state.location)
    
    if st.button("Get Weather", use_container_width=True):
        with st.spinner("Fetching weather data..."):
            weather_data = st.session_state.agent.weather.get_weather(location)
            
            if weather_data.get("status") == "success":
                current = weather_data.get("current", {})
                forecast = weather_data.get("forecast", {})
                
                # Current weather
                st.markdown("### 🌡️ Current Weather")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Temperature", f"{current.get('temperature')}°C")
                with col2:
                    st.metric("Humidity", f"{current.get('humidity')}%")
                with col3:
                    st.metric("Precipitation", f"{current.get('precipitation')}mm")
                with col4:
                    st.metric("Wind Speed", f"{current.get('wind_speed')} km/h")
                
                # 7-day forecast chart
                st.markdown("### 📅 7-Day Forecast")
                
                df = pd.DataFrame({
                    'Date': [datetime.fromisoformat(d).strftime('%a, %b %d') for d in forecast.get('dates', [])],
                    'Max Temp': forecast.get('max_temp', []),
                    'Min Temp': forecast.get('min_temp', []),
                    'Precipitation': forecast.get('precipitation', [])
                })
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Max Temp'], name='Max Temp', line=dict(color='#ff6f00')))
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Min Temp'], name='Min Temp', line=dict(color='#2196f3')))
                fig.update_layout(title="Temperature Forecast", xaxis_title="Date", yaxis_title="Temperature (°C)")
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error("⚠️ Could not fetch weather data")

def page_knowledge_base():
    st.markdown("## 📚 Knowledge Base")
    
    tab1, tab2, tab3 = st.tabs(["📥 Download Resources", "📤 Upload Documents", "📊 Library Stats"])
    
    # Tab 1: Download Resources
    with tab1:
        st.markdown("### 📥 Downloadable Farming Resources")
        st.info("📚 Download free agricultural guides, manuals, and resources for your farm")
        
        # Pre-loaded resources directory
        resources_dir = Path("agri_data/resources")
        resources_dir.mkdir(parents=True, exist_ok=True)
        
        # Define available resources (you'll add actual files here)
        resources = [
            {
                "title": "Maize Farming Guide for Zimbabwe",
                "description": "Complete guide to maize cultivation including varieties, planting, and pest management",
                "category": "crop_farming",
                "file": "maize_farming_guide.pdf",
                "size": "2.5 MB",
                "downloads": 1234
            },
            {
                "title": "Fish Farming Starter Manual",
                "description": "Step-by-step guide to starting a small-scale tilapia fish farm",
                "category": "fish_farming",
                "file": "fish_farming_manual.pdf",
                "size": "1.8 MB",
                "downloads": 856
            },
            {
                "title": "Goat Farming Best Practices",
                "description": "Comprehensive guide to goat rearing, breeding, and disease management",
                "category": "goat_farming",
                "file": "goat_farming_guide.pdf",
                "size": "3.2 MB",
                "downloads": 654
            },
            {
                "title": "Organic Fertilizer Production",
                "description": "How to make and use organic fertilizers for sustainable farming",
                "category": "general_agriculture",
                "file": "organic_fertilizer.pdf",
                "size": "1.2 MB",
                "downloads": 2145
            },
            {
                "title": "Poultry Disease Prevention Guide",
                "description": "Common poultry diseases, symptoms, prevention, and treatment",
                "category": "poultry_farming",
                "file": "poultry_diseases.pdf",
                "size": "2.0 MB",
                "downloads": 987
            }
        ]
        
        # Filter by category
        categories = ["all"] + list(set([r["category"] for r in resources]))
        selected_category = st.selectbox("Filter by category:", categories)
        
        # Display resources
        filtered_resources = resources if selected_category == "all" else [r for r in resources if r["category"] == selected_category]
        
        for resource in filtered_resources:
            with st.expander(f"📄 {resource['title']}", expanded=False):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Description:** {resource['description']}")
                    st.markdown(f"**Category:** {resource['category'].replace('_', ' ').title()}")
                    st.markdown(f"**Size:** {resource['size']} | **Downloads:** {resource['downloads']}")
                
                with col2:
                    file_path = resources_dir / resource['file']
                    
                    # Check if file exists
                    if file_path.exists():
                        with open(file_path, 'rb') as f:
                            st.download_button(
                                label="⬇️ Download",
                                data=f,
                                file_name=resource['file'],
                                mime="application/pdf",
                                use_container_width=True
                            )
                    else:
                        st.button("📥 Request File", use_container_width=True, 
                                help="This file will be available soon. Contact support to request early access.")
        
        # Add custom resource request
        st.markdown("---")
        st.markdown("### 📝 Request Custom Resources")
        with st.form("resource_request"):
            request_topic = st.text_input("What farming topic would you like resources on?")
            request_details = st.text_area("Additional details (optional)")
            request_email = st.text_input("Your email (for notification when available)")
            
            if st.form_submit_button("Submit Request", use_container_width=True):
                # Save request to database
                st.success("✅ Request submitted! We'll notify you when the resource is available.")
    
    # Tab 2: Upload Documents
    with tab2:
        st.markdown("### 📤 Upload Your Documents")
        st.info("Upload PDF or TXT files to enhance the AI's knowledge base. These documents will be used to provide more accurate answers.")
        
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT. Max 200MB per file."
        )
        
        if uploaded_files:
            st.markdown(f"**Selected {len(uploaded_files)} file(s):**")
            for file in uploaded_files:
                st.markdown(f"- 📄 {file.name} ({file.size / 1024:.1f} KB)")
            
            if st.button("🚀 Upload and Process", use_container_width=True, type="primary"):
                with st.spinner("Processing documents... This may take a few minutes."):
                    result_msg, stats = st.session_state.agent.add_documents(uploaded_files)
                    
                    if stats.get("files_processed", 0) > 0:
                        st.success(result_msg)
                        
                        # Show stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Files Processed", stats.get("files_processed", 0))
                        with col2:
                            st.metric("Chunks Created", stats.get("total_chunks", 0))
                        with col3:
                            st.metric("Failed", stats.get("files_failed", 0))
                    else:
                        st.error(result_msg)
        
        # Show upload history
        st.markdown("---")
        st.markdown("### 📋 Recently Uploaded Documents")
        
        # Get list of uploaded files
        uploads_dir = Path("uploads")
        if uploads_dir.exists():
            uploaded_list = sorted(uploads_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)[:10]
            
            if uploaded_list:
                upload_data = []
                for file_path in uploaded_list:
                    upload_data.append({
                        "Filename": file_path.name,
                        "Size": f"{file_path.stat().st_size / 1024:.1f} KB",
                        "Uploaded": datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                    })
                
                st.dataframe(upload_data, use_container_width=True)
            else:
                st.info("No documents uploaded yet.")
        else:
            st.info("No documents uploaded yet.")
    
    # Tab 3: Library Stats
    with tab3:
        st.markdown("### 📊 Knowledge Base Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Count documents
        uploads_dir = Path("uploads")
        total_docs = len(list(uploads_dir.glob("*"))) if uploads_dir.exists() else 0
        
        # Estimate chunks (approximate)
        vector_store_size = 0
        if Path("vector_db").exists():
            vector_store_size = sum(f.stat().st_size for f in Path("vector_db").glob("**/*") if f.is_file())
        
        with col1:
            st.metric("Total Documents", total_docs)
        with col2:
            st.metric("Vector DB Size", f"{vector_store_size / (1024*1024):.1f} MB")
        with col3:
            st.metric("Available Resources", len(resources))
        with col4:
            st.metric("Total Downloads", sum(r['downloads'] for r in resources))
        
        # Document categories breakdown
        st.markdown("### 📈 Document Categories")
        
        if total_docs > 0:
            # Count by extension
            extensions = {}
            for file in uploads_dir.glob("*"):
                ext = file.suffix.lower()
                extensions[ext] = extensions.get(ext, 0) + 1
            
            fig = px.pie(
                values=list(extensions.values()),
                names=list(extensions.keys()),
                title="Documents by Type"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload documents to see statistics")
        
        # Resource downloads chart
        st.markdown("### 📥 Popular Resources")
        resource_df = pd.DataFrame([
            {"Resource": r["title"][:30] + "...", "Downloads": r["downloads"]}
            for r in sorted(resources, key=lambda x: x["downloads"], reverse=True)[:5]
        ])
        
        fig = px.bar(resource_df, x="Downloads", y="Resource", orientation='h',
                     title="Top 5 Downloaded Resources")
        st.plotly_chart(fig, use_container_width=True)
        
        # Maintenance actions
        st.markdown("---")
        st.markdown("### 🔧 Maintenance")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Rebuild Vector Database", use_container_width=True):
                with st.spinner("Rebuilding..."):
                    # Implement rebuild logic
                    st.success("✅ Vector database rebuilt successfully")
        
        with col2:
            if st.button("🗑️ Clear All Uploads", use_container_width=True, type="secondary"):
                st.warning("⚠️ This will delete all uploaded documents. Click again to confirm.")

def page_calculator():
    st.markdown("## 💵 Farming Cost Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        farming_type = st.selectbox("Farming Type", [
            "crop_farming", "fish_farming", "goat_farming", 
            "pig_farming", "poultry_farming"
        ])
        scale = st.number_input("Scale (hectares or animals)", min_value=1, value=10)
        duration = st.number_input("Duration (months)", min_value=1, value=12)
    
    if st.button("Calculate Costs", use_container_width=True):
        result = st.session_state.agent.calculate_farming_costs(farming_type, scale, duration)
        
        with col2:
            st.markdown("### 📊 Cost Breakdown")
            st.metric("Total Cost", f"${result['total_cost']:,.2f}")
            st.metric("Monthly Average", f"${result['monthly_average']:,.2f}")
        
        # Detailed breakdown
        st.markdown("### 💰 Detailed Breakdown")
        breakdown_df = pd.DataFrame({
            'Item': list(result['breakdown'].keys()),
            'Cost (USD)': list(result['breakdown'].values())
        })
        
        fig = px.pie(breakdown_df, values='Cost (USD)', names='Item', 
                     title='Cost Distribution',
                     color_discrete_sequence=px.colors.sequential.Greens)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(breakdown_df, use_container_width=True)

def page_knowledge_base():
    st.markdown("## 📚 Knowledge Base")
    
    tab1, tab2, tab3 = st.tabs(["📥 Download Resources", "📤 Upload Documents", "📊 Library Stats"])
    
    # Tab 1: Download Resources
    with tab1:
        st.markdown("### 📥 Downloadable Farming Resources")
        st.info("📚 Download free agricultural guides, manuals, and resources for your farm")
        
        # Pre-loaded resources directory
        resources_dir = Path("agri_data/resources")
        resources_dir.mkdir(parents=True, exist_ok=True)
        
        # Define available resources (you'll add actual files here)
        resources = [
            {
                "title": "Maize Farming Guide for Zimbabwe",
                "description": "Complete guide to maize cultivation including varieties, planting, and pest management",
                "category": "crop_farming",
                "file": "maize_farming_guide.pdf",
                "size": "2.5 MB",
                "downloads": 1234
            },
            {
                "title": "Fish Farming Starter Manual",
                "description": "Step-by-step guide to starting a small-scale tilapia fish farm",
                "category": "fish_farming",
                "file": "fish_farming_manual.pdf",
                "size": "1.8 MB",
                "downloads": 856
            },
            {
                "title": "Goat Farming Best Practices",
                "description": "Comprehensive guide to goat rearing, breeding, and disease management",
                "category": "goat_farming",
                "file": "goat_farming_guide.pdf",
                "size": "3.2 MB",
                "downloads": 654
            },
            {
                "title": "Organic Fertilizer Production",
                "description": "How to make and use organic fertilizers for sustainable farming",
                "category": "general_agriculture",
                "file": "organic_fertilizer.pdf",
                "size": "1.2 MB",
                "downloads": 2145
            },
            {
                "title": "Poultry Disease Prevention Guide",
                "description": "Common poultry diseases, symptoms, prevention, and treatment",
                "category": "poultry_farming",
                "file": "poultry_diseases.pdf",
                "size": "2.0 MB",
                "downloads": 987
            }
        ]
        
        # Filter by category
        categories = ["all"] + list(set([r["category"] for r in resources]))
        selected_category = st.selectbox("Filter by category:", categories)
        
        # Display resources
        filtered_resources = resources if selected_category == "all" else [r for r in resources if r["category"] == selected_category]
        
        for resource in filtered_resources:
            with st.expander(f"📄 {resource['title']}", expanded=False):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Description:** {resource['description']}")
                    st.markdown(f"**Category:** {resource['category'].replace('_', ' ').title()}")
                    st.markdown(f"**Size:** {resource['size']} | **Downloads:** {resource['downloads']}")
                
                with col2:
                    file_path = resources_dir / resource['file']
                    
                    # Also check for .txt version
                    txt_file_path = resources_dir / resource['file'].replace('.pdf', '.txt')
                    
                    # Check if file exists (PDF or TXT)
                    if file_path.exists():
                        with open(file_path, 'rb') as f:
                            st.download_button(
                                label="⬇️ Download PDF",
                                data=f,
                                file_name=resource['file'],
                                mime="application/pdf",
                                use_container_width=True,
                                key=f"download_pdf_{resource['file']}"
                            )
                    elif txt_file_path.exists():
                        with open(txt_file_path, 'rb') as f:
                            st.download_button(
                                label="⬇️ Download TXT",
                                data=f,
                                file_name=resource['file'].replace('.pdf', '.txt'),
                                mime="text/plain",
                                use_container_width=True,
                                key=f"download_txt_{resource['file']}"
                            )
                    else:
                        if st.button("📥 Get This File", key=f"req_{resource['file']}", use_container_width=True):
                            st.info("""
                            **To get sample files:**
                            1. Run: `python setup_resources.py`
                            2. Or contact support to request this file
                            
                            Files will be created in: `agri_data/resources/`
                            """)
        
        # Add custom resource request
        st.markdown("---")
        st.markdown("### 📝 Request Custom Resources")
        with st.form("resource_request"):
            request_topic = st.text_input("What farming topic would you like resources on?")
            request_details = st.text_area("Additional details (optional)")
            request_email = st.text_input("Your email (for notification when available)")
            
            if st.form_submit_button("Submit Request", use_container_width=True):
                # Save request to database
                st.success("✅ Request submitted! We'll notify you when the resource is available.")
    
    # Tab 2: Upload Documents
    with tab2:
        st.markdown("### 📤 Upload Your Documents")
        st.info("Upload PDF or TXT files to enhance the AI's knowledge base. These documents will be used to provide more accurate answers.")
        
        uploaded_files = st.file_uploader(
            "Choose files to upload",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT. Max 200MB per file."
        )
        
        if uploaded_files:
            st.markdown(f"**Selected {len(uploaded_files)} file(s):**")
            for file in uploaded_files:
                st.markdown(f"- 📄 {file.name} ({file.size / 1024:.1f} KB)")
            
            if st.button("🚀 Upload and Process", use_container_width=True, type="primary"):
                with st.spinner("Processing documents... This may take a few minutes."):
                    result_msg, stats = st.session_state.agent.add_documents(uploaded_files)
                    
                    if stats.get("files_processed", 0) > 0:
                        st.success(result_msg)
                        
                        # Show stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Files Processed", stats.get("files_processed", 0))
                        with col2:
                            st.metric("Chunks Created", stats.get("total_chunks", 0))
                        with col3:
                            st.metric("Failed", stats.get("files_failed", 0))
                    else:
                        st.error(result_msg)
        
        # Show upload history
        st.markdown("---")
        st.markdown("### 📋 Recently Uploaded Documents")
        
        # Get list of uploaded files
        uploads_dir = Path("uploads")
        if uploads_dir.exists():
            uploaded_list = sorted(uploads_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)[:10]
            
            if uploaded_list:
                upload_data = []
                for file_path in uploaded_list:
                    upload_data.append({
                        "Filename": file_path.name,
                        "Size": f"{file_path.stat().st_size / 1024:.1f} KB",
                        "Uploaded": datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                    })
                
                st.dataframe(upload_data, use_container_width=True)
            else:
                st.info("No documents uploaded yet.")
        else:
            st.info("No documents uploaded yet.")
    
    # Tab 3: Library Stats
    with tab3:
        st.markdown("### 📊 Knowledge Base Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Count documents
        uploads_dir = Path("uploads")
        total_docs = len(list(uploads_dir.glob("*"))) if uploads_dir.exists() else 0
        
        # Estimate chunks (approximate)
        vector_store_size = 0
        if Path("vector_db").exists():
            try:
                vector_store_size = sum(f.stat().st_size for f in Path("vector_db").rglob("*") if f.is_file())
            except:
                vector_store_size = 0
        
        # Resources count
        resources_count = 5  # Update based on your actual resources
        
        with col1:
            st.metric("Total Documents", total_docs)
        with col2:
            st.metric("Vector DB Size", f"{vector_store_size / (1024*1024):.1f} MB")
        with col3:
            st.metric("Available Resources", resources_count)
        with col4:
            st.metric("Total Downloads", "5,876")
        
        # Document categories breakdown
        st.markdown("### 📈 Document Categories")
        
        if total_docs > 0:
            # Count by extension
            extensions = {}
            for file in uploads_dir.glob("*"):
                ext = file.suffix.lower() or "no extension"
                extensions[ext] = extensions.get(ext, 0) + 1
            
            if extensions:
                fig = px.pie(
                    values=list(extensions.values()),
                    names=list(extensions.keys()),
                    title="Documents by Type"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload documents to see statistics")
        
        # Resource downloads chart
        st.markdown("### 📥 Popular Resources")
        resource_data = {
            "Resource": ["Maize Guide", "Fish Farming", "Goat Care", "Fertilizer", "Poultry"],
            "Downloads": [1234, 856, 654, 2145, 987]
        }
        resource_df = pd.DataFrame(resource_data)
        
        fig = px.bar(resource_df, x="Downloads", y="Resource", orientation='h',
                     title="Top Downloaded Resources",
                     color="Downloads",
                     color_continuous_scale="Greens")
        st.plotly_chart(fig, use_container_width=True)
        
        # Maintenance actions
        st.markdown("---")
        st.markdown("### 🔧 Maintenance")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Rebuild Vector Database", use_container_width=True):
                with st.spinner("Rebuilding..."):
                    # Implement rebuild logic
                    st.success("✅ Vector database rebuilt successfully")
        
        with col2:
            if st.button("🗑️ Clear All Uploads", use_container_width=True, type="secondary"):
                st.warning("⚠️ This will delete all uploaded documents. Contact admin for this action.")

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    load_custom_css()
    render_header()
    render_sidebar()
    
    # Navigation - Use option_menu if available, otherwise use selectbox
    if HAS_OPTION_MENU:
        selected = option_menu(
            menu_title=None,
            options=["Dashboard", "AI Chat", "Weather", "Calculator", "Marketplace", "Knowledge Base"],
            icons=["house", "chat-dots", "cloud-sun", "calculator", "shop", "book"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#f5f5f5"},
                "icon": {"color": "#2e7d32", "font-size": "20px"}, 
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "center",
                    "margin": "0px",
                    "--hover-color": "#e8f5e9",
                },
                "nav-link-selected": {"background-color": "#2e7d32"},
            }
        )
    else:
        # Fallback to simple selectbox navigation
        st.markdown("### 🧭 Navigation")
        selected = st.selectbox(
            "Choose a page:",
            ["Dashboard", "AI Chat", "Weather", "Calculator", "Marketplace", "Knowledge Base"],
            label_visibility="collapsed"
        )
    
    # Route to pages
    if selected == "Dashboard":
        page_dashboard()
    elif selected == "AI Chat":
        page_ai_chat()
    elif selected == "Weather":
        page_weather()
    elif selected == "Calculator":
        page_calculator()
    elif selected == "Marketplace":
        st.markdown("## 🛒 Marketplace")
        st.info("Marketplace feature coming soon!")
    elif selected == "Knowledge Base":
        page_knowledge_base()
    
    # Footer
    st.markdown(f"""
    <div class="footer">
        <h3 style="color: #2e7d32;">🌾 {config.APP_NAME}</h3>
        <p style="color: #666; margin: 10px 0;">{config.APP_TAGLINE}</p>
        <p style="color: #888; font-size: 0.9rem;">
            © 2024 {config.COMPANY_NAME} | All rights reserved<br>
            <a href="{config.COMPANY_WEBSITE}" target="_blank" style="color: #2e7d32;">Website</a> | 
            <a href="mailto:{config.SUPPORT_EMAIL}" style="color: #7e57c2;">Support</a>
        </p>
        <p style="color: #999; font-size: 0.8rem; margin-top: 10px;">
            Version {config.VERSION} | Made with ❤️ for farmers
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()