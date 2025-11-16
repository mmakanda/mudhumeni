#!/usr/bin/env python3
"""
🌾 AgriBot Pro - Complete Single-File Version
Production-ready Agriculture AI Agent

Run: python agribot_complete.py
Access: http://localhost:7860
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Check Python version
if sys.version_info < (3, 8):
    print("❌ Python 3.8+ required")
    sys.exit(1)

# Create necessary directories
for directory in ["agri_data", "vector_db", "uploads", "logs", "database", "static"]:
    Path(directory).mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Application configuration"""
    APP_NAME = "Mudhumeni"
    APP_TAGLINE = "Your Intelligent Agriculture Assistant"
    VERSION = "1.0.0"
    COMPANY_NAME = "Amaryllis Success"
    COMPANY_WEBSITE = "https://amaryllissuccess.co.zw"
    SUPPORT_EMAIL = "support@amaryllissuccess.co.zw"
    
    GRADIO_PORT = 7860
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

config = Config()

# ============================================================================
# DATABASE SETUP
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
    
    # Initialize database
    engine = create_engine('sqlite:///database/agribot.db', echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    
    DB_AVAILABLE = True
    logger.info("✅ Database initialized")
except Exception as e:
    logger.warning(f"⚠️  Database not available: {e}")
    DB_AVAILABLE = False
    SessionLocal = None
    
# ============================================================================
# AI AGENT
# ============================================================================

class AgriBot:
    """Main Agriculture AI Agent"""
    
    def __init__(self):
        self.llm = None
        self.vector_store = None
        self.embeddings = None
        self.db_session = None
        
        self._initialize_llm()
        self._initialize_vector_store()
        
        if DB_AVAILABLE and SessionLocal:
            self.db_session = SessionLocal()
        
        logger.info("✅ AgriBot initialized")

    # ============================================================================
    # 1. DISEASE IMAGE ANALYSIS (Add to AgriBot class)
    # ============================================================================

    def analyze_disease_image(self, image_path: str, symptoms: str = "") -> str:
        """Analyze crop/animal disease from image"""
        if not self.llm:
            return "AI model not available"
        
        try:
            # Basic image analysis (can be enhanced with vision models)
            prompt = f"""As an agriculture disease expert, analyze this situation:

    Image Analysis Request: {Path(image_path).name if image_path else 'No image'}
    Reported Symptoms: {symptoms if symptoms else 'Not specified'}

    Provide:
    1. Possible diseases based on description
    2. Recommended treatments
    3. Prevention measures
    4. Urgency level (Low/Medium/High)
    5. When to contact a veterinarian/agronomist

    Be specific and practical."""
            
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"

    # ============================================================================
    # 2. MARKET PRICE TRACKER (Add to AgriBot class)
    # ============================================================================

    def get_market_prices(self, product: str, location: str = "general") -> str:
        """Get market price information"""
        if not self.llm:
            return "AI model not available"
        
        try:
            prompt = f"""Provide market price information for {product} in {location}:

    Include:
    1. Current average price range
    2. Price trends (increasing/stable/decreasing)
    3. Factors affecting price
    4. Best time to sell
    5. Regional variations

    Use your knowledge of agricultural markets."""
            
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"

    # ============================================================================
    # 3. FARMING CALENDAR (Add to AgriBot class)
    # ============================================================================

    def get_farming_calendar(self, crop_or_animal: str, location: str, month: int = None) -> dict:
        """Get detailed farming calendar"""
        if not month:
            month = datetime.now().month
        
        if not self.llm:
            return {"error": "AI model not available"}
        
        try:
            months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            
            prompt = f"""Create a farming calendar for {crop_or_animal} in {location}, focusing on {months[month-1]}:

    Provide monthly breakdown:
    - Current month ({months[month-1]}): Key activities
    - Next 3 months: What to prepare
    - Critical dates and deadlines
    - Weather considerations
    - Resource requirements

    Format as a practical action plan."""
            
            response = self.llm.invoke(prompt)
            return {
                "crop_or_animal": crop_or_animal,
                "location": location,
                "current_month": months[month-1],
                "calendar": response.content
            }
        except Exception as e:
            return {"error": str(e)}

    # ============================================================================
    # 4. COST CALCULATOR (Add to AgriBot class)
    # ============================================================================

    def calculate_farming_costs(
        self, 
        farming_type: str,
        scale: float,  # hectares or number of animals
        duration_months: int = 12
    ) -> dict:
        """Calculate estimated farming costs"""
        
        # Base costs per unit (these should be customized for your region)
        cost_estimates = {
            "crop_farming": {
                "seeds": 50,  # per hectare
                "fertilizer": 150,
                "pesticides": 80,
                "labor": 200,
                "water": 100,
                "equipment": 150
            },
            "fish_farming": {
                "fingerlings": 30,  # per 100
                "feed": 200,
                "water_treatment": 50,
                "labor": 150,
                "equipment": 100
            },
            "goat_farming": {
                "stock": 200,  # per animal
                "feed": 30,  # monthly per animal
                "veterinary": 15,
                "housing": 50,
                "labor": 100
            },
            "pig_farming": {
                "stock": 300,
                "feed": 50,
                "veterinary": 20,
                "housing": 80,
                "labor": 120
            },
            "poultry_farming": {
                "chicks": 2,  # per bird
                "feed": 3,  # monthly per bird
                "veterinary": 1,
                "housing": 5,
                "labor": 80
            }
        }
        
        costs = cost_estimates.get(farming_type, {})
        if not costs:
            return {"error": "Farming type not found"}
        
        total_cost = 0
        breakdown = {}
        
        for item, unit_cost in costs.items():
            if item in ["feed", "veterinary", "water"]:
                # Monthly recurring costs
                cost = unit_cost * scale * duration_months
            else:
                # One-time or annual costs
                cost = unit_cost * scale
            
            breakdown[item] = round(cost, 2)
            total_cost += cost
        
        return {
            "farming_type": farming_type,
            "scale": scale,
            "duration_months": duration_months,
            "breakdown": breakdown,
            "total_cost": round(total_cost, 2),
            "monthly_average": round(total_cost / duration_months, 2),
            "note": "These are estimates. Actual costs vary by region and market conditions."
        }

    # ============================================================================
    # 5. WEATHER INTEGRATION (Add to AgriBot class)
    # ============================================================================

    def get_weather_advice(self, location: str, farming_activity: str) -> str:
        """Get weather-based farming advice"""
        if not self.llm:
            return "AI model not available"
        
        try:
            current_month = datetime.now().strftime("%B")
            
            prompt = f"""Provide weather-based advice for {farming_activity} in {location} during {current_month}:

    Consider:
    1. Typical weather patterns for this month
    2. Rainfall expectations
    3. Temperature considerations
    4. Risk factors (drought, frost, floods)
    5. Timing recommendations
    6. Protective measures needed

    Provide practical, actionable advice."""
            
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"

    # ============================================================================
    # 6. EXPORT REPORTS (Add to AgriBot class)
    # ============================================================================

    def export_conversation(self, conversation_history: list, username: str) -> str:
        """Export conversation as formatted report"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"agribot_consultation_{username}_{timestamp}.txt"
            filepath = Path("exports") / filename
            
            # Create exports directory
            Path("exports").mkdir(exist_ok=True)
            
            # Generate report
            report = f"""
    {'='*70}
    AGRIBOT PRO - CONSULTATION REPORT
    {'='*70}
    User: {username}
    Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    {'='*70}

    """
            
            for i, (question, answer) in enumerate(conversation_history, 1):
                report += f"""
    {'='*70}
    QUESTION {i}:
    {question}

    ANSWER:
    {answer}

    """
            
            report += f"""
    {'='*70}
    End of Report
    Generated by {config.APP_NAME} v{config.VERSION}
    © {config.COMPANY_NAME}
    {'='*70}
    """
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report)
            
            return str(filepath)
        
        except Exception as e:
            return f"Error exporting: {str(e)}"

    # ============================================================================
    # 7. BATCH QUERY PROCESSING (Add to AgriBot class)
    # ============================================================================

    def batch_query(self, questions: List[str], username: str = "guest") -> List[Dict[str, str]]:
        """Process multiple questions at once"""
        results = []
        
        for i, question in enumerate(questions, 1):
            try:
                answer = self.query(question, username)
                results.append({
                    "question_number": i,
                    "question": question,
                    "answer": answer,
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "question_number": i,
                    "question": question,
                    "answer": None,
                    "status": "error",
                    "error": str(e)
                })
        
        return results
        


    
    def _initialize_llm(self):
        """Initialize LLM"""
        try:
            from langchain_ollama import ChatOllama
            
            self.llm = ChatOllama(
                model=config.LLM_MODEL,
                temperature=config.TEMPERATURE,
                base_url=config.OLLAMA_BASE_URL
            )
            logger.info(f"✅ LLM initialized: {config.LLM_MODEL}")
        except ImportError:
            logger.error("❌ langchain-ollama not installed. Run: pip install langchain-ollama")
            self.llm = None
        except Exception as e:
            logger.error(f"❌ LLM initialization failed: {e}")
            self.llm = None
    
    def _initialize_vector_store(self):
        """Initialize vector store"""
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import Chroma
            from langchain.docstore.document import Document
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'}
            )
            
            # Create or load vector store
            if Path("vector_db").exists():
                try:
                    self.vector_store = Chroma(
                        persist_directory="vector_db",
                        embedding_function=self.embeddings
                    )
                    logger.info("✅ Loaded existing vector store")
                except:
                    self._create_vector_store()
            else:
                self._create_vector_store()
                
        except ImportError:
            logger.error("❌ Vector store dependencies not installed")
            logger.error("Run: pip install langchain-community chromadb sentence-transformers")
            self.vector_store = None
        except Exception as e:
            logger.error(f"❌ Vector store initialization failed: {e}")
            self.vector_store = None
    
    def _create_vector_store(self):
        """Create new vector store"""
        try:
            from langchain_community.vectorstores import Chroma
            from langchain.docstore.document import Document
            
            dummy_doc = Document(
                page_content="Agriculture AI knowledge base initialized.",
                metadata={"source": "system"}
            )
            
            self.vector_store = Chroma.from_documents(
                documents=[dummy_doc],
                embedding=self.embeddings,
                persist_directory="vector_db"
            )
            self.vector_store.persist()
            logger.info("✅ Created new vector store")
        except Exception as e:
            logger.error(f"❌ Failed to create vector store: {e}")
    
    def query(self, question: str, username: str = "guest") -> str:
        """Main query method"""
        if not self.llm:
            return "❌ AI model not available. Please ensure:\n1. Ollama is running\n2. Run: ollama pull llama3.2\n3. Install: pip install langchain-ollama"
        
        try:
            # Search knowledge base
            context = ""
            if self.vector_store:
                try:
                    results = self.vector_store.similarity_search(question, k=3)
                    if results:
                        context = "\n\n".join([doc.page_content for doc in results])
                except:
                    pass
            
            # Create prompt
            if context:
                prompt = f"""You are an expert agriculture assistant. Use this information to answer:

Knowledge Base:
{context}

Question: {question}

Provide practical advice for farmers."""
            else:
                prompt = f"""You are an expert agriculture assistant helping farmers.

Question: {question}

Provide practical, detailed advice."""
            
            # Get response
            response = self.llm.invoke(prompt)
            
            # Log to database
            if self.db_session:
                try:
                    user = self.db_session.query(User).filter_by(username=username).first()
                    if not user:
                        user = User(username=username, created_at=datetime.now())
                        self.db_session.add(user)
                        self.db_session.commit()
                    
                    user.total_queries += 1
                    user.last_active = datetime.now()
                    self.db_session.commit()
                except:
                    pass
            
            return response.content
            
        except Exception as e:
            logger.error(f"Query error: {e}")
            return f"❌ Error: {str(e)}\n\nPlease check:\n1. Ollama is running\n2. Model is downloaded: ollama pull llama3.2"
    
    def add_documents(self, files, farming_type: str):
        """Add documents to knowledge base"""
        if not self.vector_store:
            return "❌ Vector store not available", {}
        
        try:
            from langchain_community.document_loaders import PyPDFLoader, TextLoader
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            
            documents = []
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP
            )
            
            for file in files:
                try:
                    file_path = Path(file.name if hasattr(file, 'name') else str(file))
                    
                    if file_path.suffix == '.pdf':
                        loader = PyPDFLoader(str(file_path))
                    elif file_path.suffix == '.txt':
                        loader = TextLoader(str(file_path))
                    else:
                        continue
                    
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata['farming_type'] = farming_type
                        doc.metadata['filename'] = file_path.name
                    documents.extend(docs)
                except Exception as e:
                    logger.warning(f"Failed to load {file}: {e}")
            
            if documents:
                chunks = splitter.split_documents(documents)
                self.vector_store.add_documents(chunks)
                self.vector_store.persist()
                
                return f"✅ Processed {len(files)} files, added {len(chunks)} chunks", {
                    "files": len(files),
                    "chunks": len(chunks)
                }
            else:
                return "⚠️  No documents could be loaded", {}
                
        except Exception as e:
            return f"❌ Error: {str(e)}", {}
    
    def calculate_feed(self, animal_type: str, count: int, weight: float) -> dict:
        """Calculate feed requirements"""
        feed_rates = {
            "goat": 0.03, "pig": 0.04, "chicken": 0.12,
            "fish": 0.02, "cattle": 0.025
        }
        
        rate = feed_rates.get(animal_type.lower(), 0.03)
        daily_per_animal = weight * rate if animal_type != "chicken" else rate
        total_daily = daily_per_animal * count
        monthly = total_daily * 30
        
        return {
            "animal_type": animal_type,
            "count": count,
            "daily_feed_kg": round(total_daily, 2),
            "monthly_feed_kg": round(monthly, 2),
            "yearly_feed_kg": round(monthly * 12, 2)
        }
    
    def get_stats(self) -> dict:
        """Get system statistics"""
        stats = {
            "llm_status": "✅ Active" if self.llm else "❌ Not available",
            "vector_store": "✅ Active" if self.vector_store else "❌ Not available",
            "database": "✅ Active" if self.db_session else "❌ Not available"
        }
        
        if self.db_session:
            try:
                total_users = self.db_session.query(User).count()
                stats["total_users"] = total_users
            except:
                pass
        
        return stats

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    logger.error("❌ Gradio not installed. Run: pip install gradio")
    GRADIO_AVAILABLE = False
    sys.exit(1)

# Initialize agent
agent = AgriBot()

# Custom CSS
CUSTOM_CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto;
}
.logo-header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #2e7d32, #66bb6a);
    border-radius: 10px;
    margin-bottom: 20px;
    color: white;
}
.logo-header h1 {
    margin: 10px 0;
    font-size: 2.5em;
}
"""

# Create interface
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="green"),
    css=CUSTOM_CSS,
    title="AgriBot Pro"
) as app:
    
    # Header
    gr.HTML(f"""
        <div class="logo-header">
            <h1>🌾 {config.APP_NAME}</h1>
            <p style="font-size: 1.2em;">{config.APP_TAGLINE}</p>
            <small>Version {config.VERSION}</small>
        </div>
    """)
    
    # User input
    with gr.Row():
        username = gr.Textbox(
            label="Your Name",
            placeholder="Enter your name...",
            value="Guest"
        )
    
    # Main tabs
    with gr.Tabs():
        
        # Chat Tab
        with gr.Tab("💬 AI Consultation"):
            gr.Markdown("### Ask agriculture questions and get expert advice")
            
            chatbot = gr.Chatbot(height=500, label="Conversation", type="messages")
            msg = gr.Textbox(
                label="Your Question",
                placeholder="E.g., What are the best practices for fish farming?",
                lines=2
            )
            
            with gr.Row():
                submit = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")
            
            gr.Examples(
                examples=[
                    "What are the best practices for tilapia fish farming?",
                    "How do I prevent diseases in goats?",
                    "What is the feeding schedule for pigs?",
                    "When should I plant maize in East Africa?",
                ],
                inputs=msg
            )
            
            def respond(message, chat_history, user):
                if not message.strip():
                    return chat_history, ""

                response = agent.query(message, user)

                # Use OpenAI-style message format required by Gradio type="messages"
                chat_history.append({"role": "user", "content": message})
                chat_history.append({"role": "assistant", "content": response})

                return chat_history, ""
            
            msg.submit(respond, [msg, chatbot, username], [chatbot, msg])
            submit.click(respond, [msg, chatbot, username], [chatbot, msg])
            clear.click(lambda: ([], ""), None, [chatbot, msg])
        
        # Knowledge Base Tab
        with gr.Tab("📚 Knowledge Base"):
            gr.Markdown("### Upload documents to expand knowledge")
            
            with gr.Row():
                with gr.Column():
                    farm_type = gr.Dropdown(
                        choices=config.FARMING_TYPES,
                        label="Farming Type",
                        value="general_agriculture"
                    )
                    files = gr.File(
                        label="Upload Files (PDF, TXT)",
                        file_count="multiple",
                        file_types=[".pdf", ".txt"]
                    )
                    upload_btn = gr.Button("Upload", variant="primary")
                
                with gr.Column():
                    upload_status = gr.Textbox(label="Status", lines=5)
                    upload_stats = gr.JSON(label="Statistics")
            
            upload_btn.click(
                agent.add_documents,
                [files, farm_type],
                [upload_status, upload_stats]
            )
            
        # Disease Analysis Tab
        with gr.Tab("🔬 Disease Analysis"):
            gr.Markdown("### Analyze crop or animal diseases")

            with gr.Row():
                disease_image = gr.Image(label="Upload Image (optional)", type="filepath")
                disease_symptoms = gr.Textbox(
                label="Describe Symptoms",
                placeholder="E.g., yellowing leaves, spots, wilting...",
                lines=5
                )

                analyze_btn = gr.Button("Analyze", variant="primary")
                disease_result = gr.Markdown(label="Analysis & Treatment")

            analyze_btn.click(
                lambda img, symp: agent.analyze_disease_image(img, symp),
                [disease_image, disease_symptoms],
                disease_result
            )


        # Market Prices Tab
        with gr.Tab("💰 Market Prices"):
            gr.Markdown("### Check current market prices")
    
            with gr.Row():
                product_name = gr.Textbox(label="Product", placeholder="e.g., Maize")
                product_location = gr.Textbox(label="Location", placeholder="e.g., Nairobi")
    
                price_btn = gr.Button("Check Prices", variant="primary")
                price_result = gr.Markdown(label="Price Information")
    
            price_btn.click(
                agent.get_market_prices,
                [product_name, product_location],
                price_result
            )

        # Cost Calculator Tab
        with gr.Tab("💵 Cost Calculator"):
            gr.Markdown("### Calculate farming costs")
    
            with gr.Row():
                cost_farm_type = gr.Dropdown(
                choices=["crop_farming", "fish_farming", "goat_farming", 
                    "pig_farming", "poultry_farming"],
                label="Farming Type",
                value="crop_farming"
            )
                cost_scale = gr.Number(label="Scale (hectares or animals)", value=10)
                cost_duration = gr.Number(label="Duration (months)", value=12)
    
                cost_calc_btn = gr.Button("Calculate Costs", variant="primary")
                cost_result = gr.JSON(label="Cost Breakdown")
    
            cost_calc_btn.click(
                agent.calculate_farming_costs,
                [cost_farm_type, cost_scale, cost_duration],
                cost_result
            )

        # Export Tab
        with gr.Tab("📥 Export Report"):
            gr.Markdown("### Export your consultation as a report")

            export_btn = gr.Button("Export Current Conversation", variant="primary")
            export_result = gr.Textbox(label="Export Status")

            def export_chat(history, user):
                filepath = agent.export_conversation(history, user)
                return f"✅ Report exported to: {filepath}"

            export_btn.click(
                export_chat,
                [chatbot, username],
                export_result
                )

        
        # Feed Calculator Tab
        with gr.Tab("🧮 Feed Calculator"):
            gr.Markdown("### Calculate livestock feed requirements")
            
            with gr.Row():
                animal = gr.Dropdown(
                    choices=["goat", "pig", "chicken", "fish", "cattle"],
                    label="Animal Type",
                    value="goat"
                )
                count = gr.Number(label="Number of Animals", value=10)
                weight = gr.Number(label="Average Weight (kg)", value=50)
            
            calc_btn = gr.Button("Calculate", variant="primary")
            result = gr.JSON(label="Feed Requirements")
            
            calc_btn.click(
                agent.calculate_feed,
                [animal, count, weight],
                result
            )
        
        # System Info Tab
        with gr.Tab("ℹ️ System Info"):
            gr.Markdown("### System Status & Configuration")
            
            stats_btn = gr.Button("Refresh Status", variant="primary")
            stats_display = gr.JSON(label="System Status")
            
            stats_btn.click(agent.get_stats, None, stats_display)
            
            gr.Markdown(f"""
            ### Configuration
            - **LLM Model:** {config.LLM_MODEL}
            - **Embedding Model:** {config.EMBEDDING_MODEL}
            - **Chunk Size:** {config.CHUNK_SIZE}
            
            ### Requirements
            1. Ollama installed and running
            2. Model downloaded: `ollama pull llama3.2`
            3. Python packages installed
            
            ### Support
            For issues, check:
            - Ollama status: `ollama list`
            - Python version: {sys.version_info.major}.{sys.version_info.minor}
            """)
    
    # Footer
    gr.HTML(f"""
        <div style="text-align: center; margin-top: 40px; padding: 20px; border-top: 2px solid #66bb6a;">
            <p><strong>{config.APP_NAME}</strong> - {config.APP_TAGLINE}</p>
            <p>© 2024 {config.COMPANY_NAME}. All rights reserved.</p>
            <p style="font-size: 0.9em; color: #666;">
                Version {config.VERSION} | {datetime.now().strftime("%Y-%m-%d")}
            </p>
        </div>
    """)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🌾 AGRIBOT PRO - STARTING UP")
    print("="*70 + "\n")
    
    # Check Ollama
    print("Checking Ollama...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            if models:
                print(f"✅ Ollama is running with {len(models)} model(s)")
                for model in models:
                    print(f"   - {model.get('name')}")
            else:
                print("⚠️  Ollama running but no models found")
                print("   Run: ollama pull llama3.2")
        else:
            print("⚠️  Ollama not responding properly")
    except Exception as e:
        print(f"⚠️  Cannot connect to Ollama: {e}")
        print("   Make sure Ollama is installed and running")
        print("   Download from: https://ollama.ai")
    
    print("\n" + "="*70)
    print("✅ LAUNCHING INTERFACE")
    print("="*70)
    print(f"🌐 URL: http://localhost:{config.GRADIO_PORT}")
    print("📝 Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    try:
        app.launch(
            server_name="0.0.0.0",
            server_port=config.GRADIO_PORT,
            share=True,
            show_error=True
        )
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check Ollama: ollama list")
        print("2. Install packages: pip install -r requirements.txt")
        print("3. Check port: netstat -an | findstr 7860")