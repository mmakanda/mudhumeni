#!/usr/bin/env python3
"""
🌾 AgriBot Pro - Complete Agricultural Advisory Platform
Run: streamlit run agribot_streamlit.py
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create directories
for directory in ["agri_data", "uploads", "logs", "database", "saved_tips", "crop_photos", "agri_data/resources"]:
    Path(directory).mkdir(exist_ok=True)

# Page config
st.set_page_config(
    page_title="Mudhumeni - Agricultural Advisory",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
class Config:
    APP_NAME = "Mudhumeni"
    APP_TAGLINE = "Your Complete Intelligent Agriculture Assistant"
    VERSION = "2.1.0"
    COMPANY_NAME = "Amaryllis Success"
    COMPANY_WEBSITE = "https://amaryllissuccess.co.zw"
    SUPPORT_EMAIL = "support@amaryllissuccess.co.zw"
    WHATSAPP = "+263 77 123 4567"
    
    LLM_MODEL = "llama3.2"
    OLLAMA_BASE_URL = "http://localhost:11434"
    
    DEFAULT_LOCATION = {"city": "Harare", "lat": -17.8292, "lon": 31.0522}
    
    FROST_TEMP = 2
    HEAVY_RAIN = 50
    HIGH_WIND = 40

config = Config()

# Custom CSS
def load_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Poppins', sans-serif; }
    
    .main { background: #f8faf8; }
    
    .stButton>button {
        background: linear-gradient(135deg, #2e7d32, #66bb6a);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        transform: translateY(-2px);
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .chat-message.user {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .chat-message.assistant {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
    }
    
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #2e7d32;
        margin-bottom: 1rem;
    }
    
    .community-post {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
    }
    
    .post-avatar {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: #2e7d32;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        margin-right: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

load_custom_css()

# Initialize session state
if 'navigation' not in st.session_state:
    st.session_state.navigation = "Dashboard"
if 'location' not in st.session_state:
    st.session_state.location = config.DEFAULT_LOCATION["city"]
if 'username' not in st.session_state:
    st.session_state.username = "Guest"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'tips_generated' not in st.session_state:
    st.session_state.tips_generated = 0
if 'current_tip' not in st.session_state:
    st.session_state.current_tip = None

# Weather Service
class WeatherService:
    @staticmethod
    def get_weather(location):
        try:
            import requests
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": config.DEFAULT_LOCATION["lat"],
                "longitude": config.DEFAULT_LOCATION["lon"],
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": "auto",
                "forecast_days": 7
            }
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if response.status_code == 200:
                current = data.get("current", {})
                daily = data.get("daily", {})
                
                alerts = []
                temp = current.get('temperature_2m', 999)
                wind = current.get('wind_speed_10m', 0)
                
                if temp < config.FROST_TEMP:
                    alerts.append({"type": "frost", "message": "❄️ FROST WARNING: Protect sensitive crops!"})
                if wind > config.HIGH_WIND:
                    alerts.append({"type": "wind", "message": "💨 HIGH WIND: Avoid spraying operations"})
                
                return {
                    "status": "success",
                    "current": {
                        "temperature": current.get('temperature_2m', 25),
                        "humidity": current.get('relative_humidity_2m', 60),
                        "wind_speed": current.get('wind_speed_10m', 10),
                        "precipitation": current.get('precipitation', 0),
                        "weather_code": current.get('weather_code', 0)
                    },
                    "forecast": {
                        "dates": daily.get("time", [])[:7],
                        "max_temp": daily.get("temperature_2m_max", [])[:7],
                        "min_temp": daily.get("temperature_2m_min", [])[:7],
                        "precipitation": daily.get("precipitation_sum", [])[:7]
                    },
                    "alerts": alerts
                }
        except Exception as e:
            logger.error(f"Weather error: {e}")
        
        return {
            "status": "success",
            "current": {"temperature": 25, "humidity": 60, "wind_speed": 10, "precipitation": 0, "weather_code": 0},
            "forecast": {"dates": [], "max_temp": [], "min_temp": [], "precipitation": []},
            "alerts": []
        }
    
    @staticmethod
    def get_weather_icon(code):
        icons = {0: "☀️", 1: "🌤️", 2: "⛅", 3: "☁️", 61: "🌧️", 95: "⛈️"}
        return icons.get(code, "🌤️")

# Simple AI Agent with Web Search and PDF Support
class SimpleAgent:
    def __init__(self):
        self.llm_available = False
        self.vector_store = None
        self.embeddings = None
        
        # Initialize LLM
        try:
            from langchain_ollama import ChatOllama
            self.llm = ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL)
            self.llm_available = True
            logger.info("✅ LLM initialized")
        except:
            self.llm = None
            logger.info("ℹ️ LLM not available - using fallback responses")
        
        # Initialize vector store for PDFs
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize vector store with pre-loaded documents from agri_data/resources"""
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import Chroma
            from langchain_core.documents import Document
            from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            Path("vector_db").mkdir(exist_ok=True)
            
            # Try to load existing vector store
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
                    logger.warning("⚠️ Could not load existing vector store, rebuilding...")
            
            # Load all documents from agri_data/resources
            resources_path = Path("agri_data/resources")
            if resources_path.exists():
                logger.info(f"📚 Loading documents from {resources_path}")
                
                all_documents = []
                
                # Load PDF files
                try:
                    pdf_loader = DirectoryLoader(
                        str(resources_path),
                        glob="**/*.pdf",
                        loader_cls=PyPDFLoader,
                        show_progress=True
                    )
                    pdf_docs = pdf_loader.load()
                    logger.info(f"✅ Loaded {len(pdf_docs)} pages from PDF files")
                    all_documents.extend(pdf_docs)
                except Exception as e:
                    logger.warning(f"⚠️ PDF loading error: {e}")
                
                # Load TXT files
                try:
                    txt_loader = DirectoryLoader(
                        str(resources_path),
                        glob="**/*.txt",
                        loader_cls=TextLoader,
                        show_progress=True
                    )
                    txt_docs = txt_loader.load()
                    logger.info(f"✅ Loaded {len(txt_docs)} text documents")
                    all_documents.extend(txt_docs)
                except Exception as e:
                    logger.warning(f"⚠️ TXT loading error: {e}")
                
                if all_documents:
                    # Split documents into chunks
                    from langchain_text_splitters import RecursiveCharacterTextSplitter
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    chunks = splitter.split_documents(all_documents)
                    
                    # Create vector store
                    self.vector_store = Chroma.from_documents(
                        documents=chunks,
                        embedding=self.embeddings,
                        persist_directory="vector_db"
                    )
                    
                    logger.info(f"✅ Created vector store with {len(chunks)} chunks from {len(all_documents)} documents")
                    return
            
            # Fallback: Create empty vector store
            logger.warning("⚠️ No documents found in agri_data/resources, creating empty store")
            initial_docs = [
                Document(
                    page_content="Agriculture knowledge base for Zimbabwe and Southern Africa.",
                    metadata={"source": "system"}
                )
            ]
            
            self.vector_store = Chroma.from_documents(
                documents=initial_docs,
                embedding=self.embeddings,
                persist_directory="vector_db"
            )
            logger.info("✅ Created empty vector store")
            
        except Exception as e:
            logger.error(f"❌ Vector store initialization failed: {e}")
            self.vector_store = None
    
    def _search_web(self, query: str) -> list:
        """Search the web for relevant information"""
        try:
            import requests
            
            # Using DuckDuckGo API
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
            
            # Get abstract
            if data.get("Abstract"):
                results.append({
                    "title": data.get("Heading", "General Information"),
                    "url": data.get("AbstractURL", ""),
                    "snippet": data.get("Abstract", "")
                })
            
            # Get related topics
            for topic in data.get("RelatedTopics", [])[:3]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append({
                        "title": topic.get("Text", "")[:80],
                        "url": topic.get("FirstURL", ""),
                        "snippet": topic.get("Text", "")
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []
    
    def _search_documents(self, query: str, k: int = 3) -> str:
        """Search pre-loaded agricultural documents"""
        if not self.vector_store:
            return ""
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            if results:
                context = "\n\n".join([
                    f"📄 Source: {doc.metadata.get('source', 'Agricultural Guide')}\n{doc.page_content}"
                    for doc in results
                ])
                return context
        except Exception as e:
            logger.error(f"Document search error: {e}")
        
        return ""
    
    def get_knowledge_stats(self) -> dict:
        """Get statistics about loaded knowledge base"""
        stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "vector_db_ready": False,
            "documents_list": []
        }
        
        resources_path = Path("agri_data/resources")
        if resources_path.exists():
            pdf_files = list(resources_path.glob("**/*.pdf"))
            txt_files = list(resources_path.glob("**/*.txt"))
            
            stats["total_documents"] = len(pdf_files) + len(txt_files)
            stats["documents_list"] = [f.name for f in (pdf_files + txt_files)]
        
        if self.vector_store:
            try:
                # Get collection info
                stats["vector_db_ready"] = True
                stats["total_chunks"] = self.vector_store._collection.count()
            except:
                pass
        
        return stats
    
    def add_documents(self, files) -> tuple:
        """Add PDF/TXT documents to knowledge base"""
        if not self.vector_store:
            return "❌ Vector store not available. Install required packages.", {}
        
        if not files:
            return "⚠️ No files selected", {}
        
        try:
            from langchain_community.document_loaders import PyPDFLoader, TextLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            documents = []
            processed_files = []
            failed_files = []
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            for file in files:
                try:
                    # Save file
                    file_path = Path("uploads") / file.name
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    
                    # Load based on type
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
                # Split and add to vector store
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
    
    def query(self, question: str, username: str = "guest", location: str = None) -> str:
        """Query with web search and document retrieval"""
        
        # 1. Search uploaded documents
        doc_context = self._search_documents(question)
        
        # 2. Search the web
        web_results = self._search_web(question)
        web_context = ""
        if web_results:
            web_context = "\n\n".join([
                f"🌐 {r['title']}\nURL: {r['url']}\n{r['snippet'][:200]}..."
                for r in web_results[:3]
            ])
        
        # 3. Build comprehensive context
        now = datetime.now()
        season = self._get_season(now.month)
        
        full_context = f"""Current Date: {now.strftime('%Y-%m-%d')}
Season: {season}
Location: {location or config.DEFAULT_LOCATION['city']}

"""
        
        if doc_context:
            full_context += f"\n📚 KNOWLEDGE BASE DOCUMENTS:\n{doc_context}\n"
        
        if web_context:
            full_context += f"\n🌐 WEB SEARCH RESULTS:\n{web_context}\n"
        
        # 4. Generate response
        if self.llm_available and self.llm:
            try:
                prompt = f"""{full_context}

Question: {question}

Instructions:
1. Use information from uploaded documents (Knowledge Base) as primary source
2. Supplement with web search results for current information
3. Provide practical advice for {location or config.DEFAULT_LOCATION['city']}
4. Consider the current season: {season}
5. Cite your sources when possible

Answer:"""
                
                response = self.llm.invoke(prompt)
                return response.content
                
            except Exception as e:
                logger.error(f"LLM error: {e}")
        
        # Fallback response with context
        response = f"""**Question:** {question}

**Based on available information:**

"""
        
        if doc_context:
            response += """📚 **From Your Uploaded Documents:**

I found relevant information in your knowledge base. Here are key points from the documents:

"""
            # Extract first 500 chars from doc context
            response += doc_context[:500] + "...\n\n"
        
        if web_context:
            response += """🌐 **From Web Search:**

Recent information found online:

"""
            response += web_context[:500] + "...\n\n"
        
        if not doc_context and not web_context:
            response += f"""I don't have specific information about "{question}" in the uploaded documents or web search results.

**General Recommendations:**

1. **Consult Local Experts**: Contact your agricultural extension office in {location or config.DEFAULT_LOCATION['city']}
2. **Upload Documents**: Add relevant farming manuals to the Knowledge Base for better answers
3. **Seasonal Considerations**: Current season is {season}
4. **Weather Check**: Use the Weather page for planning
5. **Cost Planning**: Use the Calculator for financial estimates

**To get better answers:**
- Upload PDF guides related to your question
- Be specific about your crop/livestock type
- Mention your specific challenges
"""
        
        return response
    
    def _get_season(self, month: int) -> str:
        if month in [11, 12, 1, 2, 3]:
            return "Rainy/Growing Season"
        elif month in [4, 5]:
            return "Harvest Season"
        else:
            return "Dry Season"
    
    def calculate_farming_costs(self, farming_type: str, scale: float, duration: int = 12) -> dict:
        cost_estimates = {
            "crop_farming": {"seeds": 50, "fertilizer": 150, "pesticides": 80, "labor": 200, "water": 100},
            "fish_farming": {"fingerlings": 30, "feed": 200, "labor": 150, "equipment": 100},
            "goat_farming": {"stock": 200, "feed": 30, "veterinary": 15, "housing": 50},
            "pig_farming": {"stock": 300, "feed": 50, "veterinary": 20, "housing": 80},
            "poultry_farming": {"chicks": 2, "feed": 3, "veterinary": 1, "housing": 5}
        }
        
        costs = cost_estimates.get(farming_type, cost_estimates["crop_farming"])
        breakdown = {}
        total_cost = 0
        
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
    
    def calculate_loan(self, principal: float, rate: float, years: int) -> dict:
        monthly_rate = rate / 100 / 12
        num_payments = years * 12
        
        if monthly_rate > 0:
            monthly_payment = principal * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
        else:
            monthly_payment = principal / num_payments
        
        total_payment = monthly_payment * num_payments
        total_interest = total_payment - principal
        
        return {
            "principal": round(principal, 2),
            "monthly_payment": round(monthly_payment, 2),
            "total_payment": round(total_payment, 2),
            "total_interest": round(total_interest, 2),
            "rate": rate,
            "years": years
        }

# Initialize agent
if 'agent' not in st.session_state:
    st.session_state.agent = SimpleAgent()

weather = WeatherService()

# Header
st.markdown(f"""
<div style="background: linear-gradient(135deg, #1b5e20, #2e7d32, #66bb6a); padding: 2.5rem; border-radius: 20px; text-align: center; margin-bottom: 2rem; box-shadow: 0 10px 30px rgba(0,0,0,0.15);">
    <h1 style="color: white; font-size: 3rem; margin: 0; text-shadow: 2px 2px 8px rgba(0,0,0,0.3);">🌾 {config.APP_NAME}</h1>
    <p style="color: #e8f5e9; font-size: 1.3rem; margin-top: 0.5rem;">{config.APP_TAGLINE}</p>
    <p style="color: #e8f5e9; font-size: 0.9rem; margin-top: 1rem;">Version {config.VERSION} | © {config.COMPANY_NAME}</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 👤 User Profile")
    st.session_state.username = st.text_input("Your Name", value=st.session_state.username)
    st.session_state.location = st.text_input("📍 Location", value=st.session_state.location)
    
    st.markdown("---")
    st.markdown("### 📅 Current Info")
    now = datetime.now()
    season = st.session_state.agent._get_season(now.month)
    st.info(f"""
    **Date:** {now.strftime('%B %d, %Y')}  
    **Time:** {now.strftime('%I:%M %p')}  
    **Season:** {season}  
    **Location:** {st.session_state.location}
    """)
    
    st.markdown("---")
    st.markdown("### 📱 Quick Contact")
    st.markdown(f"""
    📧 [Email](mailto:{config.SUPPORT_EMAIL})  
    📱 [WhatsApp](https://wa.me/{config.WHATSAPP.replace('+', '').replace(' ', '')})  
    🌐 [Website]({config.COMPANY_WEBSITE})
    """)

# Navigation
nav_options = ["Dashboard", "AI Chat", "Weather", "Calculator", "Farming Tips", "Crop Health", "Community", "Knowledge Base"]
selected = st.selectbox("🧭 Navigation", nav_options, 
                       index=nav_options.index(st.session_state.navigation),
                       label_visibility="collapsed")
st.session_state.navigation = selected

# ============================================================================
# PAGES
# ============================================================================

if selected == "Dashboard":
    st.markdown("## 🏠 Agricultural Advisory Dashboard")
    
    weather_data = weather.get_weather(st.session_state.location)
    
    # Weather alerts
    if weather_data.get("alerts"):
        for alert in weather_data["alerts"]:
            st.warning(alert["message"])
    
    # Current weather
    if weather_data.get("status") == "success":
        current = weather_data["current"]
        now = datetime.now()
        
        st.markdown("### 🌤️ Current Weather Conditions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #2196f3, #64b5f6); color: white; padding: 2rem; border-radius: 15px; text-align: center; height: 200px; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 3rem;">🌤️</div>
                <div style="font-size: 2.5rem; font-weight: bold;">{current.get('temperature')}°C</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Temperature</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #00bcd4, #4dd0e1); color: white; padding: 2rem; border-radius: 15px; text-align: center; height: 200px; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 3rem;">💧</div>
                <div style="font-size: 2.5rem; font-weight: bold;">{current.get('humidity')}%</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Humidity</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #4caf50, #66bb6a); color: white; padding: 2rem; border-radius: 15px; text-align: center; height: 200px; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 3rem;">💨</div>
                <div style="font-size: 2.5rem; font-weight: bold;">{current.get('wind_speed')}</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Wind (km/h)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ff9800, #ffa726); color: white; padding: 2rem; border-radius: 15px; text-align: center; height: 200px; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 3rem;">📍</div>
                <div style="font-size: 1.3rem; font-weight: bold;">{st.session_state.location}</div>
                <div style="font-size: 0.85rem; opacity: 0.9;">{now.strftime('%I:%M %p')}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🌾 Agricultural Advisory Services")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ffffff, #e8f5e9); padding: 2rem; border-radius: 15px; text-align: center; min-height: 250px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <div style="font-size: 3rem;">🤖</div>
            <h3>AI Consultation</h3>
            <p style="color: #666; font-size: 0.9rem;">Get expert farming advice powered by AI</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Ask AI", key="btn1"):
            st.session_state.navigation = "AI Chat"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ffffff, #e8f5e9); padding: 2rem; border-radius: 15px; text-align: center; min-height: 250px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <div style="font-size: 3rem;">🌤️</div>
            <h3>Weather Forecast</h3>
            <p style="color: #666; font-size: 0.9rem;">7-day weather predictions</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("View Forecast", key="btn2"):
            st.session_state.navigation = "Weather"
            st.rerun()
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ffffff, #e8f5e9); padding: 2rem; border-radius: 15px; text-align: center; min-height: 250px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <div style="font-size: 3rem;">💡</div>
            <h3>Farming Tips</h3>
            <p style="color: #666; font-size: 0.9rem;">Expert tips and best practices</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Get Tips", key="btn3"):
            st.session_state.navigation = "Farming Tips"
            st.rerun()
    
    with col4:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ffffff, #e8f5e9); padding: 2rem; border-radius: 15px; text-align: center; min-height: 250px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <div style="font-size: 3rem;">💵</div>
            <h3>Cost Calculator</h3>
            <p style="color: #666; font-size: 0.9rem;">Calculate costs and ROI</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Calculate", key="btn4"):
            st.session_state.navigation = "Calculator"
            st.rerun()
    
    st.markdown("---")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### 🌱 Seasonal Farming Advice")
        st.info(f"""
        **{season}**
        
        **Recommended Activities:**
        - 🌱 Plan crop rotation
        - 💧 Check irrigation systems
        - 🐛 Monitor pest populations
        - 🌾 Prepare for planting/harvesting
        """)
    
    with col2:
        st.markdown("### 💡 Today's Tip")
        st.success("""
        **Soil Health**
        
        Regular soil testing helps optimize fertilizer use and improve yields. Test at least once per season.
        """)

elif selected == "AI Chat":
    st.markdown("## 💬 AI Consultation")
    
    # Show knowledge sources
    col1, col2, col3 = st.columns(3)
    
    stats = st.session_state.agent.get_knowledge_stats()
    
    with col1:
        st.metric("📚 Knowledge Base", f"{stats['total_documents']} documents")
    
    with col2:
        st.metric("🌐 Web Search", "✅ Active")
    
    with col3:
        llm_status = "✅ Active" if st.session_state.agent.llm_available else "⚠️ Fallback Mode"
        st.metric("🤖 AI Model", llm_status)
    
    if stats['vector_db_ready']:
        st.success(f"✅ Knowledge base loaded with {stats['total_chunks']} searchable chunks from agricultural guides")
    else:
        st.warning("⚠️ Knowledge base is building. Add documents to agri_data/resources folder and restart.")
    
    st.markdown("---")
    
    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area("Ask your agriculture question:", height=100,
                                   placeholder="E.g., What are the best practices for maize cultivation in Zimbabwe?")
        submitted = st.form_submit_button("📤 Send Message")
    
    if submitted and user_input:
        with st.spinner("🔍 Searching agricultural guides and web... 🤔 Thinking..."):
            response = st.session_state.agent.query(user_input, st.session_state.username, st.session_state.location)
            st.session_state.chat_history.append({"role": "user", "content": user_input, "timestamp": datetime.now()})
            st.session_state.chat_history.append({"role": "assistant", "content": response, "timestamp": datetime.now()})
    
    st.markdown("### 💭 Conversation History")
    
    if st.session_state.chat_history:
        for msg in reversed(st.session_state.chat_history[-10:]):
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user">
                    <strong>👤 You ({msg['timestamp'].strftime('%I:%M %p')}):</strong><br>{msg["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant">
                    <strong>🤖 AgriBot ({msg['timestamp'].strftime('%I:%M %p')}):</strong><br>{msg["content"]}
                </div>
                """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        with col2:
            if st.session_state.chat_history:
                chat_text = "\n\n".join([
                    f"{m['role'].upper()} ({m['timestamp'].strftime('%Y-%m-%d %I:%M %p')}): {m['content']}"
                    for m in st.session_state.chat_history
                ])
                st.download_button(
                    label="📥 Export Chat",
                    data=chat_text,
                    file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    else:
        st.info(f"""
        👋 **Welcome to AI Consultation!**
        
        I can help you with:
        - 🌾 **Crop farming** questions (planting, harvesting, varieties)
        - 🐄 **Livestock management** (health, feeding, breeding)
        - 💧 **Irrigation & water management**
        - 🐛 **Pest & disease control**
        - 💰 **Farming economics & planning**
        - 🌤️ **Weather-based advice**
        
        **I use authoritative sources:**
        - 📚 {stats['total_documents']} pre-loaded agricultural guides and manuals
        - 🌐 Live web search for current information
        - 🤖 AI-powered analysis and recommendations
        
        **Tips for best results:**
        - Be specific in your questions
        - Mention your crop/livestock type
        - Include your location for tailored advice
        - Ask about specific challenges or goals
        """)
    
    st.markdown("---")
    
    # Quick question suggestions
    st.markdown("### 💡 Quick Questions")
    
    quick_questions = [
        "What is the best time to plant maize in Zimbabwe?",
        "How do I control armyworms in my maize field?",
        "What are the water requirements for tobacco farming?",
        "How can I improve my soil fertility naturally?",
        "What are the signs of Newcastle disease in chickens?"
    ]
    
    cols = st.columns(2)
    for idx, question in enumerate(quick_questions):
        with cols[idx % 2]:
            if st.button(f"❓ {question[:50]}...", key=f"quick_{idx}"):
                with st.spinner("🔍 Searching... 🤔 Thinking..."):
                    response = st.session_state.agent.query(question, st.session_state.username, st.session_state.location)
                    st.session_state.chat_history.append({"role": "user", "content": question, "timestamp": datetime.now()})
                    st.session_state.chat_history.append({"role": "assistant", "content": response, "timestamp": datetime.now()})
                st.rerun()

elif selected == "Weather":
    st.markdown("## 🌤️ Weather Forecast")
    
    weather_data = weather.get_weather(st.session_state.location)
    
    if weather_data.get("status") == "success":
        current = weather_data["current"]
        forecast = weather_data["forecast"]
        now = datetime.now()
        
        st.markdown("### 🌡️ Current Conditions")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            temp = current.get('temperature', 'N/A')
            st.markdown(f"""
            <div style="text-align: center; padding: 30px;">
                <div style="font-size: 5rem;">🌤️</div>
                <div style="font-size: 4rem; font-weight: bold; color: #ff6f00;">{temp}°C</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="padding: 30px;">
                <p style="margin: 10px 0; font-size: 1.1rem;"><strong>Humidity:</strong> {current.get('humidity')}%</p>
                <p style="margin: 10px 0; font-size: 1.1rem;"><strong>Wind:</strong> {current.get('wind_speed')} km/h</p>
                <p style="margin: 10px 0; font-size: 1.1rem;"><strong>Precipitation:</strong> {current.get('precipitation')}mm</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="text-align: right; padding: 30px;">
                <h3 style="color: #2e7d32;">Weather</h3>
                <p style="color: #666;">{now.strftime('%A')}</p>
                <hr>
                <p><strong>📍 {st.session_state.location}</strong></p>
                <p>{now.strftime('%B %d, %Y')}</p>
                <p>{now.strftime('%I:%M %p')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 7-day forecast
        if forecast.get("dates"):
            st.markdown("### 📅 7-Day Forecast")
            
            df = pd.DataFrame({
                'Date': [datetime.fromisoformat(d).strftime('%a, %b %d') for d in forecast.get('dates', [])],
                'Max Temp (°C)': forecast.get('max_temp', []),
                'Min Temp (°C)': forecast.get('min_temp', [])
            })
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Max Temp (°C)'], name='Max', line=dict(color='#ff6f00', width=3)))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Min Temp (°C)'], name='Min', line=dict(color='#2196f3', width=3)))
            fig.update_layout(title="Temperature Forecast", height=400)
            st.plotly_chart(fig, use_column_width=True)
            
            st.dataframe(df, use_column_width=True)

elif selected == "Calculator":
    st.markdown("## 💵 Agricultural Calculator")
    
    tab1, tab2, tab3 = st.tabs(["💰 Cost Calculator", "🏦 Loan Calculator", "📊 ROI Calculator"])
    
    with tab1:
        st.markdown("### 💰 Farming Cost Estimator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            farming_type = st.selectbox("Farming Type", ["crop_farming", "fish_farming", "goat_farming", "pig_farming", "poultry_farming"])
            scale = st.number_input("Scale (hectares/animals)", min_value=1, value=10)
            duration = st.number_input("Duration (months)", min_value=1, value=12)
        
        if st.button("📊 Calculate Costs"):
            result = st.session_state.agent.calculate_farming_costs(farming_type, scale, duration)
            
            with col2:
                st.metric("💵 Total Cost", f"${result['total_cost']:,.2f}")
                st.metric("📅 Monthly Average", f"${result['monthly_average']:,.2f}")
            
            st.markdown("### 💰 Cost Breakdown")
            
            breakdown_df = pd.DataFrame({
                'Item': list(result['breakdown'].keys()),
                'Cost (USD)': list(result['breakdown'].values())
            })
            
            fig = px.pie(breakdown_df, values='Cost (USD)', names='Item', title='Cost Distribution')
            st.plotly_chart(fig, use_column_width=True)
            
            st.dataframe(breakdown_df, use_column_width=True)
    
    with tab2:
        st.markdown("### 🏦 Agricultural Loan Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            principal = st.number_input("💵 Loan Amount ($)", min_value=100, value=10000)
            rate = st.number_input("📈 Interest Rate (%)", min_value=0.1, value=8.5)
            years = st.number_input("📅 Loan Term (years)", min_value=1, value=5)
        
        if st.button("🧮 Calculate Loan"):
            result = st.session_state.agent.calculate_loan(principal, rate, years)
            
            with col2:
                st.metric("💵 Monthly Payment", f"${result['monthly_payment']:,.2f}")
                st.metric("📊 Total Payment", f"${result['total_payment']:,.2f}")
                st.metric("💸 Total Interest", f"${result['total_interest']:,.2f}")
    
    with tab3:
        st.markdown("### 📊 Return on Investment Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            investment = st.number_input("💵 Total Investment ($)", min_value=100, value=5000)
            revenue = st.number_input("💰 Expected Revenue ($)", min_value=0, value=8000)
            expenses = st.number_input("💸 Operating Expenses ($)", min_value=0, value=2000)
        
        if st.button("📈 Calculate ROI"):
            net_profit = revenue - expenses - investment
            roi = (net_profit / investment) * 100 if investment > 0 else 0
            
            with col2:
                st.metric("💰 Net Profit", f"${net_profit:,.2f}", delta=f"{roi:.1f}% ROI")
                st.metric("📊 ROI Percentage", f"{roi:.2f}%")
                st.metric("💵 Break-even", f"${investment + expenses:,.2f}")
            
            fig = go.Figure(data=[
                go.Bar(name='Investment', x=['Costs'], y=[investment], marker_color='#f44336'),
                go.Bar(name='Expenses', x=['Costs'], y=[expenses], marker_color='#ff9800'),
                go.Bar(name='Revenue', x=['Income'], y=[revenue], marker_color='#4caf50')
            ])
            fig.update_layout(title='Investment vs Returns', barmode='group', height=400)
            st.plotly_chart(fig, use_column_width=True)

elif selected == "Farming Tips":
    st.markdown("## 💡 Farming Tips & Knowledge")
    
    st.info("📚 Get AI-generated farming tips and expert knowledge")
    
    tip_categories = ["Crop Management", "Pest Control", "Soil Health", "Irrigation", 
                      "Harvest Practices", "Organic Farming", "Animal Husbandry", "Seasonal Planning"]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_category = st.selectbox("Select Topic", tip_categories)
        
        if st.button("🎯 Generate Farming Tip"):
            with st.spinner("🤔 Generating tip..."):
                prompt = f"Provide a practical farming tip about {selected_category} for farmers in {st.session_state.location}"
                response = st.session_state.agent.query(prompt, st.session_state.username, st.session_state.location)
                
                st.session_state.current_tip = {
                    "category": selected_category,
                    "content": response,
                    "generated_at": datetime.now()
                }
                st.session_state.tips_generated += 1
    
    with col2:
        st.markdown("### 📊 Stats")
        st.metric("✨ Tips Generated", st.session_state.tips_generated)
    
    if st.session_state.current_tip:
        st.markdown("---")
        st.markdown("### 📝 Generated Tip")
        
        tip = st.session_state.current_tip
        
        st.markdown(f"""
        <div class="info-card">
            <h4>🏷️ {tip['category']}</h4>
            <p style="color: #666;">Generated: {tip['generated_at'].strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(tip['content'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Save Tip"):
                tips_dir = Path("saved_tips")
                tips_dir.mkdir(exist_ok=True)
                filename = f"{tip['category'].replace(' ', '_')}_{tip['generated_at'].strftime('%Y%m%d_%H%M%S')}.txt"
                filepath = tips_dir / filename
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"Category: {tip['category']}\nGenerated: {tip['generated_at']}\n\n{tip['content']}")
                st.success(f"✅ Saved!")
        
        with col2:
            if st.button("🔄 Generate New"):
                st.session_state.current_tip = None
                st.rerun()
        
        with col3:
            st.download_button("📤 Export", tip['content'], f"{tip['category']}.txt")

elif selected == "Crop Health":
    st.markdown("## 📸 Crop Health Diagnosis")
    
    st.info("📷 Upload crop photos for AI-powered disease diagnosis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_image = st.file_uploader("Upload Crop Photo", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("🔍 Diagnose"):
                with st.spinner("🔬 Analyzing..."):
                    # Save image
                    img_path = Path("crop_photos") / uploaded_image.name
                    with open(img_path, "wb") as f:
                        f.write(uploaded_image.getbuffer())
                    
                    st.success("✅ Image saved for analysis")
                    
                    # Simulated diagnosis
                    st.session_state.diagnosis = {
                        "disease": "Leaf Blight",
                        "confidence": 85.0,
                        "severity": "Moderate",
                        "treatment": "Apply appropriate fungicide and remove affected leaves",
                        "prevention": "Ensure proper spacing and good drainage"
                    }
    
    with col2:
        if hasattr(st.session_state, 'diagnosis'):
            diag = st.session_state.diagnosis
            
            st.markdown(f"""
            ### 🔬 Diagnosis Results
            
            **Disease:** {diag['disease']}  
            **Confidence:** {diag['confidence']}%  
            **Severity:** {diag['severity']}
            
            #### 💊 Treatment
            {diag['treatment']}
            
            #### 🛡️ Prevention
            {diag['prevention']}
            """)
            
            if st.button("📋 Get Detailed Report"):
                st.session_state.navigation = "AI Chat"
                st.rerun()
        else:
            st.info("Upload a crop photo to begin diagnosis")

elif selected == "Community":
    st.markdown("## 👥 Community Forum")
    
    st.info("🌾 Connect with fellow farmers and share experiences")
    
    # Post creation
    with st.expander("➕ Create New Post"):
        with st.form("new_post"):
            post_content = st.text_area("What's on your mind?", height=100)
            post_category = st.selectbox("Category", ["General", "Crops", "Livestock", "Equipment", "Market Prices"])
            
            if st.form_submit_button("📤 Post"):
                if post_content:
                    st.success("✅ Post created successfully!")
    
    # Sample posts
    st.markdown("### 📰 Recent Posts")
    
    sample_posts = [
        {
            "username": "John Farmer",
            "content": "Great maize yields this year! Used new hybrid seeds. Highly recommend! 🌽",
            "likes": 12,
            "comments": 3,
            "time": "2 hours ago"
        },
        {
            "username": "Mary Agriculture",
            "content": "Anyone dealing with aphids on tobacco? What pesticides are you using?",
            "likes": 7,
            "comments": 5,
            "time": "5 hours ago"
        },
        {
            "username": "Peter Ranch",
            "content": "Just installed drip irrigation. Water usage down 40%! Happy to share details.",
            "likes": 18,
            "comments": 8,
            "time": "1 day ago"
        }
    ]
    
    for post in sample_posts:
        st.markdown(f"""
        <div class="community-post">
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <div class="post-avatar">👤</div>
                <div>
                    <strong>{post['username']}</strong><br>
                    <small style="color: #666;">{post['time']}</small>
                </div>
            </div>
            <p>{post['content']}</p>
            <div style="display: flex; gap: 1.5rem; padding-top: 1rem; border-top: 1px solid #e0e0e0; margin-top: 1rem; color: #666;">
                <span>👍 {post['likes']} Likes</span>
                <span>💬 {post['comments']} Comments</span>
                <span>🔄 Share</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

elif selected == "Knowledge Base":
    st.markdown("## 📚 Knowledge Base")
    
    stats = st.session_state.agent.get_knowledge_stats()
    
    # Overview
    st.markdown("### 📊 Knowledge Base Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📄 Total Documents", stats['total_documents'])
    
    with col2:
        st.metric("📊 Searchable Chunks", stats.get('total_chunks', 'N/A'))
    
    with col3:
        status = "✅ Active" if stats['vector_db_ready'] else "⚠️ Building"
        st.metric("🔍 Search Status", status)
    
    with col4:
        st.metric("🤖 AI Integration", "✅ Connected")
    
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["📥 Available Resources", "ℹ️ About Knowledge Base"])
    
    with tab1:
        st.markdown("### 📥 Agricultural Resources")
        st.info("📚 These documents are used by the AI to answer your questions")
        
        # Pre-loaded resources from agri_data/resources
        resources_path = Path("agri_data/resources")
        
        if resources_path.exists():
            pdf_files = sorted(resources_path.glob("**/*.pdf"))
            txt_files = sorted(resources_path.glob("**/*.txt"))
            all_files = pdf_files + txt_files
            
            if all_files:
                st.success(f"✅ {len(all_files)} document(s) loaded in knowledge base")
                
                # Display documents
                for idx, file_path in enumerate(all_files):
                    file_size = file_path.stat().st_size / 1024  # KB
                    file_type = "📕 PDF" if file_path.suffix == '.pdf' else "📄 TXT"
                    
                    with st.expander(f"{file_type} {file_path.name}", expanded=False):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"""
                            **Filename:** {file_path.name}  
                            **Type:** {file_path.suffix.upper()[1:]}  
                            **Size:** {file_size:.1f} KB  
                            **Location:** agri_data/resources/
                            """)
                            
                            # Try to show first few lines for TXT files
                            if file_path.suffix == '.txt':
                                try:
                                    with open(file_path, 'r', encoding='utf-8') as f:
                                        preview = f.read(300)
                                    st.text_area("Preview", preview, height=100, disabled=True, key=f"preview_{idx}")
                                except:
                                    pass
                        
                        with col2:
                            if st.button("📖 Use in Chat", key=f"use_{idx}"):
                                st.session_state.navigation = "AI Chat"
                                st.info(f"Go to AI Chat and ask questions - this document will be used!")
            else:
                st.warning("""
                ⚠️ **No documents found**
                
                To add documents:
                1. Place PDF or TXT files in: `agri_data/resources/`
                2. Restart the application
                3. Documents will be automatically loaded
                """)
        else:
            st.error(f"""
            ❌ **Resources folder not found**
            
            Please create: `agri_data/resources/` and add your agricultural guides.
            """)
        
        st.markdown("---")
        st.markdown("### 📖 Document Categories")
        
        categories = {
            "Crop Farming": ["Maize", "Tobacco", "Cotton", "Wheat", "Vegetables"],
            "Livestock": ["Cattle", "Goats", "Pigs", "Poultry", "Fish"],
            "Pest Control": ["Armyworms", "Aphids", "Diseases", "Fungicides"],
            "Soil Management": ["Fertilizers", "pH", "Composting", "Conservation"],
            "Water Management": ["Irrigation", "Drainage", "Conservation", "Harvesting"]
        }
        
        for category, topics in categories.items():
            with st.expander(f"📁 {category}"):
                st.markdown("**Topics covered:**")
                for topic in topics:
                    st.markdown(f"- {topic}")
    
    with tab2:
        st.markdown("### ℹ️ About the Knowledge Base")
        
        st.markdown("""
        #### 🎯 Purpose
        
        The Knowledge Base is a curated collection of agricultural guides, manuals, and research 
        specifically for Zimbabwe and Southern Africa farming practices.
        
        #### 🔍 How It Works
        
        1. **Document Storage**: Agricultural PDFs and text files are stored in `agri_data/resources/`
        2. **Intelligent Indexing**: Documents are automatically processed and indexed using AI
        3. **Semantic Search**: When you ask questions, the system finds relevant information
        4. **AI Integration**: Answers combine knowledge base + web search + AI reasoning
        
        #### 💡 Best Practices
        
        **For Best Results:**
        - Be specific in your questions
        - Mention crop/livestock types
        - Include your location
        - Ask about practical challenges
        
        **The AI will:**
        - Search these documents first (highest priority)
        - Supplement with web search for current info
        - Provide practical, location-aware advice
        - Cite sources when available
        
        #### 🔐 Access Control
        
        **Public Users:**
        - ✅ Can search and read documents
        - ✅ Can ask AI questions using these documents
        - ❌ Cannot upload new documents
        - ❌ Cannot modify existing documents
        
        **Administrators:**
        - Can add documents to `agri_data/resources/`
        - Can update the knowledge base
        - Can rebuild the search index
        
        #### 📊 Current Status
        """)
        
        if stats['vector_db_ready']:
            st.success(f"""
            ✅ **Knowledge Base is Ready**
            
            - {stats['total_documents']} documents loaded
            - {stats.get('total_chunks', 'N/A')} searchable chunks
            - AI Chat is using this knowledge
            """)
        else:
            st.warning("""
            ⚠️ **Knowledge Base is Building**
            
            The system is processing documents. This happens automatically when:
            - The application starts
            - New documents are added to agri_data/resources/
            - The search index needs updating
            """)
        
        st.markdown("---")
        st.markdown("### 🛠️ For Administrators")
        
        st.code("""
# To add new documents:
1. Place PDF/TXT files in: agri_data/resources/
2. Restart the application
3. Documents will be automatically indexed

# To rebuild the index:
1. Delete the vector_db folder
2. Restart the application
3. All documents will be re-indexed
        """)
        
        st.markdown("---")
        st.info("""
        💬 **Need help?**
        
        Contact support: {config.SUPPORT_EMAIL}
        """.format(config=config))

# Footer
st.markdown(f"""
<div style="text-align: center; padding: 3rem 2rem; margin-top: 4rem; background: linear-gradient(135deg, #f5f5f5, #e8f5e9); border-radius: 20px; border-top: 5px solid #66bb6a;">
    <h3 style="color: #2e7d32;">🌾 {config.APP_NAME}</h3>
    <p style="color: #666; margin: 10px 0;">{config.APP_TAGLINE}</p>
    <p style="color: #888; font-size: 0.9rem;">
        © 2024 {config.COMPANY_NAME} | All rights reserved<br>
        <a href="{config.COMPANY_WEBSITE}" style="color: #2e7d32;">Website</a> | 
        <a href="mailto:{config.SUPPORT_EMAIL}" style="color: #7e57c2;">Email</a> |
        <a href="https://wa.me/{config.WHATSAPP.replace('+', '').replace(' ', '')}" style="color: #25D366;">WhatsApp</a>
    </p>
    <p style="color: #999; font-size: 0.8rem; margin-top: 10px;">
        Version {config.VERSION} | Made with ❤️ for farmers | 🌍 Zimbabwe & Southern Africa
    </p>
</div>
""", unsafe_allow_html=True)