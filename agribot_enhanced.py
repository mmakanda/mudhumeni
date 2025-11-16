#!/usr/bin/env python3
"""
🌾 AgriBot Pro - Enhanced Version with Location, Weather & Marketplace
Production-ready Agriculture AI Agent

Run: python agribot_enhanced.py
Access: http://localhost:7860
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

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
    VERSION = "2.0.0"
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
    
    # Default location (Zimbabwe)
    DEFAULT_LOCATION = {
        "city": "Harare",
        "country": "Zimbabwe",
        "lat": -17.8292,
        "lon": 31.0522
    }

config = Config()

# ============================================================================
# DATABASE SETUP (Enhanced)
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
    
    # Initialize database
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
    """Weather forecast service"""
    
    @staticmethod
    def get_weather(location: str = None, lat: float = None, lon: float = None) -> dict:
        """Get weather forecast (using Open-Meteo free API)"""
        try:
            import requests
            
            # Use provided coordinates or default
            if not lat or not lon:
                lat = config.DEFAULT_LOCATION["lat"]
                lon = config.DEFAULT_LOCATION["lon"]
                location = location or config.DEFAULT_LOCATION["city"]
            
            # Open-Meteo API (free, no key required)
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
                        "temperature": f"{current.get('temperature_2m', 'N/A')}°C",
                        "humidity": f"{current.get('relative_humidity_2m', 'N/A')}%",
                        "precipitation": f"{current.get('precipitation', 0)}mm",
                        "wind_speed": f"{current.get('wind_speed_10m', 'N/A')} km/h",
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
            return {
                "status": "error",
                "message": str(e),
                "fallback": True,
                "location": location or config.DEFAULT_LOCATION["city"]
            }
    
    @staticmethod
    def format_weather_display(weather_data: dict) -> str:
        """Format weather data for display"""
        if weather_data.get("status") == "error":
            return f"⚠️ Weather data unavailable: {weather_data.get('message')}"
        
        current = weather_data.get("current", {})
        forecast = weather_data.get("forecast", {})
        
        display = f"""
### 🌤️ Current Weather - {weather_data.get('location', 'Unknown')}
**Temperature:** {current.get('temperature', 'N/A')}  
**Humidity:** {current.get('humidity', 'N/A')}  
**Precipitation:** {current.get('precipitation', 'N/A')}  
**Wind Speed:** {current.get('wind_speed', 'N/A')}

### 📅 7-Day Forecast
"""
        
        dates = forecast.get("dates", [])
        max_temps = forecast.get("max_temp", [])
        min_temps = forecast.get("min_temp", [])
        precip = forecast.get("precipitation", [])
        
        for i in range(min(7, len(dates))):
            date_obj = datetime.fromisoformat(dates[i])
            day_name = date_obj.strftime("%A, %b %d")
            display += f"\n**{day_name}**  \n"
            display += f"🌡️ {min_temps[i]:.1f}°C - {max_temps[i]:.1f}°C | "
            display += f"🌧️ {precip[i]:.1f}mm\n"
        
        return display

# ============================================================================
# LOCATION SERVICE
# ============================================================================

class LocationService:
    """Location detection and geocoding service"""
    
    @staticmethod
    def get_location_from_ip() -> dict:
        """Get approximate location from IP (using ipapi.co free API)"""
        try:
            import requests
            response = requests.get("https://ipapi.co/json/", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return {
                    "city": data.get("city", "Unknown"),
                    "region": data.get("region", ""),
                    "country": data.get("country_name", "Unknown"),
                    "lat": data.get("latitude"),
                    "lon": data.get("longitude"),
                    "timezone": data.get("timezone", "UTC")
                }
        except Exception as e:
            logger.error(f"Location detection error: {e}")
        
        return config.DEFAULT_LOCATION
    
    @staticmethod
    def geocode_location(location_name: str) -> dict:
        """Convert location name to coordinates (using Nominatim free API)"""
        try:
            import requests
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                "q": location_name,
                "format": "json",
                "limit": 1
            }
            headers = {
                "User-Agent": "AgriBot/2.0"
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data:
                    return {
                        "name": data[0].get("display_name"),
                        "lat": float(data[0].get("lat")),
                        "lon": float(data[0].get("lon"))
                    }
        except Exception as e:
            logger.error(f"Geocoding error: {e}")
        
        return None

# ============================================================================
# MARKETPLACE SERVICE
# ============================================================================

class MarketplaceService:
    """Marketplace for farming products and services"""
    
    def __init__(self, db_session):
        self.db = db_session
    
    def create_listing(self, user_id: int, title: str, description: str, 
                      category: str, price: float, location: str, 
                      contact: str, duration_days: int = 30) -> dict:
        """Create new marketplace listing"""
        if not self.db:
            return {"error": "Database not available"}
        
        try:
            listing = MarketplaceListing(
                user_id=user_id,
                title=title,
                description=description,
                category=category,
                price=price,
                location=location,
                contact=contact,
                expires_at=datetime.now() + timedelta(days=duration_days)
            )
            self.db.add(listing)
            self.db.commit()
            
            return {
                "status": "success",
                "listing_id": listing.id,
                "message": "Listing created successfully"
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_listings(self, category: str = None, location: str = None, 
                    active_only: bool = True) -> List[dict]:
        """Get marketplace listings"""
        if not self.db:
            return []
        
        try:
            query = self.db.query(MarketplaceListing)
            
            if active_only:
                query = query.filter(
                    MarketplaceListing.status == "active",
                    MarketplaceListing.expires_at > datetime.now()
                )
            
            if category:
                query = query.filter(MarketplaceListing.category == category)
            
            if location:
                query = query.filter(MarketplaceListing.location.contains(location))
            
            listings = query.order_by(MarketplaceListing.created_at.desc()).limit(50).all()
            
            return [{
                "id": l.id,
                "title": l.title,
                "description": l.description,
                "category": l.category,
                "price": l.price,
                "location": l.location,
                "contact": l.contact,
                "created_at": l.created_at.strftime("%Y-%m-%d"),
                "expires_at": l.expires_at.strftime("%Y-%m-%d") if l.expires_at else None
            } for l in listings]
            
        except Exception as e:
            logger.error(f"Error fetching listings: {e}")
            return []
    
    def add_farming_tip(self, title: str, content: str, category: str, 
                       author: str = "AgriBot") -> dict:
        """Add farming tip"""
        if not self.db:
            return {"error": "Database not available"}
        
        try:
            tip = FarmingTip(
                title=title,
                content=content,
                category=category,
                author=author
            )
            self.db.add(tip)
            self.db.commit()
            
            return {"status": "success", "tip_id": tip.id}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def get_farming_tips(self, category: str = None, limit: int = 10) -> List[dict]:
        """Get farming tips"""
        if not self.db:
            return []
        
        try:
            query = self.db.query(FarmingTip)
            
            if category:
                query = query.filter(FarmingTip.category == category)
            
            tips = query.order_by(FarmingTip.created_at.desc()).limit(limit).all()
            
            return [{
                "id": t.id,
                "title": t.title,
                "content": t.content,
                "category": t.category,
                "author": t.author,
                "likes": t.likes,
                "created_at": t.created_at.strftime("%Y-%m-%d")
            } for t in tips]
            
        except Exception as e:
            logger.error(f"Error fetching tips: {e}")
            return []

# ============================================================================
# AI AGENT (Enhanced)
# ============================================================================

class AgriBot:
    """Main Agriculture AI Agent"""
    
    def __init__(self):
        self.llm = None
        self.vector_store = None
        self.embeddings = None
        self.db_session = None
        self.marketplace = None
        self.weather = WeatherService()
        self.location_service = LocationService()
        
        self._initialize_llm()
        self._initialize_vector_store()
        
        if DB_AVAILABLE and SessionLocal:
            self.db_session = SessionLocal()
            self.marketplace = MarketplaceService(self.db_session)
            self._seed_initial_tips()
        
        logger.info("✅ AgriBot initialized")
    
    def _seed_initial_tips(self):
        """Add some initial farming tips"""
        if not self.marketplace:
            return
        
        initial_tips = [
            {
                "title": "Soil Testing Importance",
                "content": "Always test your soil before planting. Proper pH levels (6.0-7.0 for most crops) ensure optimal nutrient absorption.",
                "category": "crop_farming"
            },
            {
                "title": "Water Management for Fish",
                "content": "Maintain water temperature between 25-30°C for tilapia. Test water quality weekly: pH, ammonia, and dissolved oxygen levels.",
                "category": "fish_farming"
            },
            {
                "title": "Goat Vaccination Schedule",
                "content": "Vaccinate kids at 3 months old against PPR, foot-and-mouth disease. Deworm every 3 months for optimal health.",
                "category": "goat_farming"
            },
            {
                "title": "Crop Rotation Benefits",
                "content": "Rotate crops annually to prevent soil depletion and disease buildup. Follow legumes with cereals for nitrogen enrichment.",
                "category": "crop_farming"
            }
        ]
        
        try:
            existing = self.db_session.query(FarmingTip).count()
            if existing == 0:
                for tip in initial_tips:
                    self.marketplace.add_farming_tip(**tip)
                logger.info(f"✅ Added {len(initial_tips)} initial farming tips")
        except Exception as e:
            logger.error(f"Error seeding tips: {e}")
    
    def get_current_datetime_info(self) -> dict:
        """Get current date/time information"""
        now = datetime.now()
        return {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "day": now.strftime("%A"),
            "month": now.strftime("%B"),
            "year": now.year,
            "formatted": now.strftime("%A, %B %d, %Y at %H:%M"),
            "week_of_year": now.isocalendar()[1],
            "season": self._get_season(now.month)
        }
    
    def _get_season(self, month: int) -> str:
        """Determine agricultural season (Southern Hemisphere - Zimbabwe)"""
        if month in [11, 12, 1, 2, 3]:
            return "Rainy/Growing Season"
        elif month in [4, 5]:
            return "Harvest Season"
        else:
            return "Dry Season"
    
    def get_weather_based_advice(self, location: str = None, farming_type: str = "general") -> str:
        """Get weather-based farming advice"""
        weather_data = self.weather.get_weather(location)
        date_info = self.get_current_datetime_info()
        
        if not self.llm:
            return "AI model not available"
        
        try:
            prompt = f"""As an agriculture expert, provide weather-based advice:

Current Date: {date_info['formatted']}
Season: {date_info['season']}
Location: {weather_data.get('location', 'Unknown')}
Current Weather: {weather_data.get('current', {})}
Farming Type: {farming_type}

Based on the current weather conditions and season, provide:
1. Immediate actions to take
2. What to plant/harvest now
3. Weather-related risks to watch for
4. Irrigation recommendations
5. Pest management for this weather

Be specific and actionable."""
            
            response = self.llm.invoke(prompt)
            return response.content
            
        except Exception as e:
            return f"Error: {str(e)}"

    def analyze_disease_image(self, image_path: str, symptoms: str = "") -> str:
        """Analyze crop/animal disease from image"""
        if not self.llm:
            return "AI model not available"
        
        try:
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

    def get_market_prices(self, product: str, location: str = "general") -> str:
        """Get market price information"""
        if not self.llm:
            return "AI model not available"
        
        try:
            date_info = self.get_current_datetime_info()
            
            prompt = f"""Provide market price information for {product} in {location}:

Current Date: {date_info['formatted']}
Season: {date_info['season']}

Include:
1. Current average price range (in USD)
2. Price trends (increasing/stable/decreasing)
3. Factors affecting price
4. Best time to sell
5. Regional variations

Use your knowledge of agricultural markets in Zimbabwe/Southern Africa."""
            
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"

    def get_farming_calendar(self, crop_or_animal: str, location: str, month: int = None) -> dict:
        """Get detailed farming calendar"""
        if not month:
            month = datetime.now().month
        
        if not self.llm:
            return {"error": "AI model not available"}
        
        try:
            months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            date_info = self.get_current_datetime_info()
            
            prompt = f"""Create a farming calendar for {crop_or_animal} in {location}, focusing on {months[month-1]}:

Current Date: {date_info['formatted']}
Season: {date_info['season']}

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
                "season": date_info['season'],
                "calendar": response.content
            }
        except Exception as e:
            return {"error": str(e)}

    def calculate_farming_costs(
        self, 
        farming_type: str,
        scale: float,
        duration_months: int = 12
    ) -> dict:
        """Calculate estimated farming costs"""
        
        cost_estimates = {
            "crop_farming": {
                "seeds": 50,
                "fertilizer": 150,
                "pesticides": 80,
                "labor": 200,
                "water": 100,
                "equipment": 150
            },
            "fish_farming": {
                "fingerlings": 30,
                "feed": 200,
                "water_treatment": 50,
                "labor": 150,
                "equipment": 100
            },
            "goat_farming": {
                "stock": 200,
                "feed": 30,
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
                "chicks": 2,
                "feed": 3,
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
                cost = unit_cost * scale * duration_months
            else:
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
            "note": "Estimates in USD. Actual costs vary by region and market."
        }

    def export_conversation(self, conversation_history: list, username: str) -> str:
        """Export conversation as formatted report"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"agribot_consultation_{username}_{timestamp}.txt"
            filepath = Path("exports") / filename
            
            Path("exports").mkdir(exist_ok=True)
            
            date_info = self.get_current_datetime_info()
            
            report = f"""
{'='*70}
{config.APP_NAME} - CONSULTATION REPORT
{'='*70}
User: {username}
Date: {date_info['formatted']}
Season: {date_info['season']}
{'='*70}

"""
            
            for i, msg in enumerate(conversation_history, 1):
                if msg.get("role") == "user":
                    report += f"\n{'='*70}\nQUESTION {i//2 + 1}:\n{msg.get('content')}\n\n"
                else:
                    report += f"ANSWER:\n{msg.get('content')}\n\n"
            
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
            from langchain_core.documents import Document  # Updated import
            
            logger.info("Initializing vector store...")
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'}
            )
            
            logger.info("Embeddings model loaded successfully")
            
            # Ensure vector_db directory exists
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
                except Exception as e:
                    logger.warning(f"Could not load existing vector store: {e}, creating new one...")
            
            # Create new vector store
            self._create_vector_store()
                
        except ImportError as e:
            logger.error("❌ Vector store dependencies not installed")
            logger.error("Run: pip install langchain-community chromadb sentence-transformers langchain-core")
            logger.error(f"Import error details: {e}")
            self.vector_store = None
        except Exception as e:
            logger.error(f"❌ Vector store initialization failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.vector_store = None
    
    def _create_vector_store(self):
        """Create new vector store"""
        try:
            from langchain_community.vectorstores import Chroma
            from langchain_core.documents import Document  # Updated import
            
            logger.info("Creating new vector store...")
            
            # Create initial documents with agriculture knowledge
            initial_docs = [
                Document(
                    page_content="Agriculture AI knowledge base initialized. This system helps farmers with crop management, livestock care, disease prevention, and market information.",
                    metadata={"source": "system", "category": "general"}
                ),
                Document(
                    page_content="Cattle farming in Zimbabwe: Cattle are typically grazed on natural pastures. Main breeds include Mashona, Nkone, and Tuli for beef, and Jersey and Friesian for dairy. Vaccination against diseases like foot-and-mouth is essential.",
                    metadata={"source": "system", "category": "cattle_farming"}
                ),
                Document(
                    page_content="Crop farming seasons in Zimbabwe: The rainy season (November-March) is the main growing season. Popular crops include maize, tobacco, cotton, wheat, and vegetables. Proper soil preparation and timely planting are crucial.",
                    metadata={"source": "system", "category": "crop_farming"}
                )
            ]
            
            self.vector_store = Chroma.from_documents(
                documents=initial_docs,
                embedding=self.embeddings,
                persist_directory="vector_db"
            )
            
            logger.info("✅ Created and persisted new vector store with initial knowledge")
            
        except Exception as e:
            logger.error(f"❌ Failed to create vector store: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.vector_store = None
    
    def query(self, question: str, username: str = "guest", user_location: str = None) -> str:
        """Main query method with location and time context"""
        if not self.llm:
            return "❌ AI model not available. Please ensure:\n1. Ollama is running\n2. Run: ollama pull llama3.2\n3. Install: pip install langchain-ollama"
        
        try:
            # Get date/time context
            date_info = self.get_current_datetime_info()
            
            # Get location context
            if not user_location:
                user_location = config.DEFAULT_LOCATION["city"]
            
            # Search knowledge base
            context = ""
            if self.vector_store:
                try:
                    results = self.vector_store.similarity_search(question, k=3)
                    if results:
                        context = "\n\n".join([doc.page_content for doc in results])
                except:
                    pass
            
            # Create enhanced prompt with context
            system_context = f"""Current Date & Time: {date_info['formatted']}
Season: {date_info['season']}
Location: {user_location}"""
            
            if context:
                prompt = f"""{system_context}

You are an expert agriculture assistant. Use this information to answer:

Knowledge Base:
{context}

Question: {question}

Provide practical, location-aware and season-appropriate advice for farmers."""
            else:
                prompt = f"""{system_context}

You are an expert agriculture assistant helping farmers.

Question: {question}

Provide practical, detailed advice considering the current date, season, and location."""
            
            # Get response
            response = self.llm.invoke(prompt)
            
            # Log to database
            if self.db_session:
                try:
                    user = self.db_session.query(User).filter_by(username=username).first()
                    if not user:
                        user = User(
                            username=username, 
                            created_at=datetime.now(),
                            location=user_location
                        )
                        self.db_session.add(user)
                        self.db_session.commit()
                    
                    user.total_queries += 1
                    user.last_active = datetime.now()
                    if user_location:
                        user.location = user_location
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
            return "❌ Vector store not available. Please install required packages:\npip install langchain-community chromadb sentence-transformers langchain-core", {}
        
        if not files:
            return "⚠️ No files selected", {}
        
        try:
            from langchain_community.document_loaders import PyPDFLoader, TextLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter  # Updated import
            
            logger.info(f"Processing {len(files)} file(s)...")
            
            documents = []
            processed_files = []
            failed_files = []
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP
            )
            
            for file in files:
                try:
                    # Get file path
                    if hasattr(file, 'name'):
                        file_path = Path(file.name)
                    else:
                        file_path = Path(str(file))
                    
                    logger.info(f"Loading file: {file_path.name}")
                    
                    # Load based on file type
                    if file_path.suffix.lower() == '.pdf':
                        loader = PyPDFLoader(str(file_path))
                    elif file_path.suffix.lower() == '.txt':
                        loader = TextLoader(str(file_path), encoding='utf-8')
                    else:
                        logger.warning(f"Unsupported file type: {file_path.suffix}")
                        failed_files.append(f"{file_path.name} (unsupported format)")
                        continue
                    
                    # Load and add metadata
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata['farming_type'] = farming_type
                        doc.metadata['filename'] = file_path.name
                        doc.metadata['upload_date'] = datetime.now().isoformat()
                    
                    documents.extend(docs)
                    processed_files.append(file_path.name)
                    logger.info(f"✅ Loaded {len(docs)} pages from {file_path.name}")
                    
                except Exception as e:
                    logger.warning(f"Failed to load {file}: {e}")
                    failed_files.append(f"{file_path.name if 'file_path' in locals() else 'unknown'} ({str(e)})")
            
            if documents:
                # Split documents into chunks
                logger.info("Splitting documents into chunks...")
                chunks = splitter.split_documents(documents)
                
                # Add to vector store
                logger.info(f"Adding {len(chunks)} chunks to vector store...")
                self.vector_store.add_documents(chunks)
                
                result_msg = f"✅ Successfully processed {len(processed_files)} file(s)\n"
                result_msg += f"📄 Files: {', '.join(processed_files)}\n"
                result_msg += f"📊 Created {len(chunks)} knowledge chunks"
                
                if failed_files:
                    result_msg += f"\n\n⚠️ Failed to process:\n" + "\n".join(f"  - {f}" for f in failed_files)
                
                return result_msg, {
                    "files_processed": len(processed_files),
                    "files_failed": len(failed_files),
                    "total_chunks": len(chunks),
                    "farming_type": farming_type
                }
            else:
                return "⚠️ No documents could be loaded. Please check file formats (PDF or TXT only)", {
                    "files_failed": len(failed_files)
                }
                
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
        date_info = self.get_current_datetime_info()
        
        stats = {
            "current_date": date_info['formatted'],
            "season": date_info['season'],
            "llm_status": "✅ Active" if self.llm else "❌ Not available",
            "vector_store": "✅ Active" if self.vector_store else "❌ Not available",
            "database": "✅ Active" if self.db_session else "❌ Not available",
            "marketplace": "✅ Active" if self.marketplace else "❌ Not available"
        }
        
        if self.db_session:
            try:
                total_users = self.db_session.query(User).count()
                total_listings = self.db_session.query(MarketplaceListing).filter_by(status="active").count()
                total_tips = self.db_session.query(FarmingTip).count()
                
                stats["total_users"] = total_users
                stats["active_listings"] = total_listings
                stats["farming_tips"] = total_tips
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

# Amaryllis Success Logo as Base64 (SVG converted)
AMARYLLIS_LOGO_SVG = """
<svg viewBox="0 0 200 280" xmlns="http://www.w3.org/2000/svg" style="width: 120px; height: auto; margin: 0 auto 20px; display: block;">
  <!-- Flame design with Amaryllis colors -->
  <defs>
    <linearGradient id="flame1" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#ff1744;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#ff6f00;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#ffa726;stop-opacity:1" />
    </linearGradient>
    <linearGradient id="flame2" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" style="stop-color:#e91e63;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#ab47bc;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#7e57c2;stop-opacity:1" />
    </linearGradient>
  </defs>
  
  <!-- Right flame (orange gradient) -->
  <path d="M 120 30 Q 140 50 145 80 Q 148 110 140 140 Q 135 160 125 170 Q 115 175 105 172 Q 100 168 100 160 Q 100 140 108 120 Q 115 100 120 80 Q 123 60 120 30 Z" 
        fill="url(#flame1)" />
  
  <!-- Left flame (purple gradient) -->
  <path d="M 80 90 Q 70 110 65 135 Q 63 155 68 175 Q 73 190 85 200 Q 95 205 105 200 Q 110 195 110 185 Q 110 170 105 155 Q 100 135 95 120 Q 88 105 80 90 Z" 
        fill="url(#flame2)" />
  
  <!-- Text -->
  <text x="100" y="240" font-family="Arial, sans-serif" font-size="36" font-weight="bold" fill="#7e57c2" text-anchor="middle">Amaryllis</text>
  <text x="100" y="270" font-family="Arial, sans-serif" font-size="28" font-weight="normal" fill="#666666" text-anchor="middle">Success</text>
</svg>
"""

# Custom CSS with Amaryllis and Agriculture branding
CUSTOM_CSS = """
:root {
    --amaryllis-purple: #7e57c2;
    --amaryllis-pink: #e91e63;
    --amaryllis-orange: #ff6f00;
    --agri-green: #2e7d32;
    --agri-light-green: #66bb6a;
    --agri-dark-green: #1b5e20;
}

.gradio-container {
    max-width: 1400px !important;
    margin: auto;
}

.logo-header {
    text-align: center;
    padding: 40px 30px;
    background: linear-gradient(135deg, var(--agri-dark-green), var(--agri-green), var(--agri-light-green));
    border-radius: 15px;
    margin-bottom: 20px;
    color: white;
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    position: relative;
    overflow: hidden;
}

.logo-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(45deg, var(--amaryllis-purple) 0%, transparent 40%, transparent 60%, var(--amaryllis-orange) 100%);
    opacity: 0.1;
    pointer-events: none;
}

.logo-header h1 {
    margin: 10px 0;
    font-size: 3em;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    background: linear-gradient(45deg, #ffffff, #e8f5e9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.info-badge {
    display: inline-block;
    padding: 8px 18px;
    margin: 5px;
    background: linear-gradient(135deg, rgba(255,255,255,0.25), rgba(255,255,255,0.15));
    border-radius: 25px;
    font-size: 0.9em;
    border: 1px solid rgba(255,255,255,0.3);
    backdrop-filter: blur(5px);
}

.footer {
    text-align: center;
    margin-top: 40px;
    padding: 30px;
    background: linear-gradient(135deg, #f5f5f5, #e8f5e9);
    border-radius: 15px;
    border-top: 4px solid var(--agri-light-green);
    box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
}

.footer-logo {
    margin: 20px auto;
    filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
}

/* Tab styling with Amaryllis colors */
.tab-nav button[aria-selected="true"] {
    border-bottom: 3px solid var(--amaryllis-purple) !important;
    color: var(--agri-green) !important;
}

/* Button styling */
button.primary {
    background: linear-gradient(135deg, var(--agri-green), var(--agri-light-green)) !important;
    border: none !important;
}

button.primary:hover {
    background: linear-gradient(135deg, var(--agri-dark-green), var(--agri-green)) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
}

/* Accent elements with Amaryllis colors */
.accent-element {
    border-left: 4px solid var(--amaryllis-purple);
    padding-left: 15px;
}
"""

# Create interface
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="green", secondary_hue="emerald"),
    css=CUSTOM_CSS,
    title=f"{config.APP_NAME} - Agriculture AI Assistant"
) as app:
    
    # Header with Amaryllis Logo
    gr.HTML(f"""
        <div class="logo-header">
            {AMARYLLIS_LOGO_SVG}
            <h1>🌾 {config.APP_NAME}</h1>
            <p style="font-size: 1.3em; margin: 10px 0; text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">{config.APP_TAGLINE}</p>
            <div style="margin-top: 15px;">
                <span class="info-badge">📍 Location-Aware</span>
                <span class="info-badge">🌤️ Weather Integration</span>
                <span class="info-badge">🛒 Marketplace</span>
                <span class="info-badge">💡 Smart Tips</span>
            </div>
            <small style="opacity: 0.9;">Version {config.VERSION} | Powered by AI & {config.COMPANY_NAME}</small>
        </div>
    """)
    
    # User context section
    with gr.Row():
        with gr.Column(scale=2):
            username = gr.Textbox(
                label="👤 Your Name",
                placeholder="Enter your name...",
                value="Guest"
            )
        with gr.Column(scale=2):
            user_location = gr.Textbox(
                label="📍 Your Location",
                placeholder="e.g., Harare, Bulawayo, Mutare",
                value=config.DEFAULT_LOCATION["city"]
            )
        with gr.Column(scale=1):
            detect_location_btn = gr.Button("🌍 Auto-Detect", size="sm")
    
    # Display current date/time
    datetime_display = gr.Markdown()
    
    def update_datetime_display():
        info = agent.get_current_datetime_info()
        return f"""### 📅 {info['formatted']} | Season: {info['season']}"""
    
    def auto_detect_location():
        loc = agent.location_service.get_location_from_ip()
        return f"{loc['city']}, {loc['country']}"
    
    detect_location_btn.click(auto_detect_location, None, user_location)
    
    # Update datetime on load
    app.load(update_datetime_display, None, datetime_display)
    
    # Main tabs
    with gr.Tabs():
        
        # Chat Tab
        with gr.Tab("💬 AI Consultation"):
            gr.Markdown("### Ask agriculture questions with location and weather context")
            
            chatbot = gr.Chatbot(height=500, label="Conversation", type="messages")
            msg = gr.Textbox(
                label="Your Question",
                placeholder="E.g., What should I plant this season in my area?",
                lines=2
            )
            
            with gr.Row():
                submit = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear")
            
            gr.Examples(
                examples=[
                    "What are the best crops to plant this season?",
                    "How do I prevent diseases in goats?",
                    "What is the feeding schedule for pigs?",
                    "When should I plant maize considering current weather?",
                    "What vegetables grow well in my location?",
                ],
                inputs=msg
            )
            
            def respond(message, chat_history, user, location):
                if not message.strip():
                    return chat_history, ""

                response = agent.query(message, user, location)

                chat_history.append({"role": "user", "content": message})
                chat_history.append({"role": "assistant", "content": response})

                return chat_history, ""
            
            msg.submit(respond, [msg, chatbot, username, user_location], [chatbot, msg])
            submit.click(respond, [msg, chatbot, username, user_location], [chatbot, msg])
            clear.click(lambda: ([], ""), None, [chatbot, msg])
        
        # Weather Tab
        with gr.Tab("🌤️ Weather & Advice"):
            gr.Markdown("### Get weather forecast and farming advice")
            
            with gr.Row():
                with gr.Column():
                    weather_location = gr.Textbox(
                        label="Location",
                        value=config.DEFAULT_LOCATION["city"]
                    )
                    farming_type_weather = gr.Dropdown(
                        choices=config.FARMING_TYPES,
                        label="Farming Type",
                        value="general_agriculture"
                    )
                    
                    with gr.Row():
                        get_weather_btn = gr.Button("🌤️ Get Weather", variant="primary")
                        get_advice_btn = gr.Button("💡 Get Advice", variant="secondary")
                
                with gr.Column():
                    weather_display = gr.Markdown(label="Weather Forecast")
                    weather_advice = gr.Markdown(label="Farming Advice")
            
            def show_weather(location):
                weather_data = agent.weather.get_weather(location)
                return agent.weather.format_weather_display(weather_data)
            
            def show_advice(location, farming_type):
                return agent.get_weather_based_advice(location, farming_type)
            
            get_weather_btn.click(show_weather, [weather_location], weather_display)
            get_advice_btn.click(show_advice, [weather_location, farming_type_weather], weather_advice)
        
        # Marketplace Tab
        with gr.Tab("🛒 Marketplace"):
            gr.Markdown("### Buy, sell, and discover farming products")
            
            with gr.Tabs():
                # Browse Listings
                with gr.Tab("Browse"):
                    with gr.Row():
                        market_category = gr.Dropdown(
                            choices=["all"] + config.FARMING_TYPES,
                            label="Category",
                            value="all"
                        )
                        market_location_filter = gr.Textbox(
                            label="Filter by Location",
                            placeholder="Optional"
                        )
                        browse_btn = gr.Button("🔍 Search", variant="primary")
                    
                    listings_display = gr.Dataframe(
                        headers=["ID", "Title", "Category", "Price (USD)", "Location", "Contact", "Posted"],
                        label="Available Listings"
                    )
                    
                    def browse_listings(category, location):
                        if not agent.marketplace:
                            return []
                        
                        cat = None if category == "all" else category
                        listings = agent.marketplace.get_listings(cat, location)
                        
                        return [[
                            l["id"], l["title"], l["category"], 
                            l["price"], l["location"], l["contact"], l["created_at"]
                        ] for l in listings]
                    
                    browse_btn.click(
                        browse_listings,
                        [market_category, market_location_filter],
                        listings_display
                    )
                    
                    # Auto-load on tab open
                    app.load(lambda: browse_listings("all", None), None, listings_display)
                
                # Post Listing
                with gr.Tab("Post Listing"):
                    gr.Markdown("### Create a new marketplace listing")
                    
                    with gr.Column():
                        listing_title = gr.Textbox(label="Title", placeholder="e.g., Fresh Tomatoes for Sale")
                        listing_desc = gr.Textbox(label="Description", lines=3, placeholder="Describe your product...")
                        
                        with gr.Row():
                            listing_category = gr.Dropdown(
                                choices=config.FARMING_TYPES,
                                label="Category",
                                value="general_agriculture"
                            )
                            listing_price = gr.Number(label="Price (USD)", value=0)
                        
                        with gr.Row():
                            listing_location = gr.Textbox(label="Location", placeholder="Your location")
                            listing_contact = gr.Textbox(label="Contact", placeholder="Phone/Email")
                        
                        listing_duration = gr.Slider(
                            minimum=7, maximum=90, value=30, step=1,
                            label="Listing Duration (days)"
                        )
                        
                        post_listing_btn = gr.Button("📤 Post Listing", variant="primary")
                        listing_status = gr.Markdown()
                    
                    def post_listing(user, title, desc, cat, price, loc, contact, duration):
                        if not agent.marketplace:
                            return "❌ Marketplace not available"
                        
                        # Get or create user
                        user_id = 1  # Simplified for demo
                        if agent.db_session:
                            try:
                                user_obj = agent.db_session.query(User).filter_by(username=user).first()
                                if user_obj:
                                    user_id = user_obj.id
                            except:
                                pass
                        
                        result = agent.marketplace.create_listing(
                            user_id, title, desc, cat, price, loc, contact, duration
                        )
                        
                        if result.get("status") == "success":
                            return f"✅ Listing posted successfully! ID: {result['listing_id']}"
                        else:
                            return f"❌ Error: {result.get('message')}"
                    
                    post_listing_btn.click(
                        post_listing,
                        [username, listing_title, listing_desc, listing_category, 
                         listing_price, listing_location, listing_contact, listing_duration],
                        listing_status
                    )
        
        # Farming Tips Tab
        with gr.Tab("💡 Farming Tips"):
            gr.Markdown("### Learn from expert farming tips")
            
            with gr.Row():
                tips_category = gr.Dropdown(
                    choices=["all"] + config.FARMING_TYPES,
                    label="Category",
                    value="all"
                )
                refresh_tips_btn = gr.Button("🔄 Refresh", variant="primary")
            
            tips_display = gr.Dataframe(
                headers=["ID", "Title", "Category", "Author", "Likes", "Date"],
                label="Farming Tips"
            )
            
            selected_tip = gr.Markdown(label="Tip Content")
            
            def load_tips(category):
                if not agent.marketplace:
                    return []
                
                cat = None if category == "all" else category
                tips = agent.marketplace.get_farming_tips(cat, limit=20)
                
                return [[
                    t["id"], t["title"], t["category"], 
                    t["author"], t["likes"], t["created_at"]
                ] for t in tips]
            
            def show_tip_content(evt: gr.SelectData):
                if not agent.marketplace:
                    return "No tips available"
                
                tips = agent.marketplace.get_farming_tips(limit=100)
                for tip in tips:
                    if tip["id"] == evt.value:
                        return f"""### {tip['title']}
**Category:** {tip['category']} | **Author:** {tip['author']} | **👍 {tip['likes']}**

{tip['content']}
"""
                return "Tip not found"
            
            refresh_tips_btn.click(load_tips, [tips_category], tips_display)
            tips_display.select(show_tip_content, None, selected_tip)
            
            # Auto-load tips
            app.load(lambda: load_tips("all"), None, tips_display)
        
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
                    upload_btn = gr.Button("📤 Upload", variant="primary")
                
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
                with gr.Column():
                    disease_image = gr.Image(label="Upload Image (optional)", type="filepath")
                    disease_symptoms = gr.Textbox(
                        label="Describe Symptoms",
                        placeholder="E.g., yellowing leaves, spots, wilting...",
                        lines=5
                    )
                    analyze_btn = gr.Button("🔍 Analyze", variant="primary")
                
                with gr.Column():
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
                with gr.Column():
                    product_name = gr.Textbox(label="Product", placeholder="e.g., Maize, Tomatoes")
                    product_location = gr.Textbox(
                        label="Location", 
                        placeholder="e.g., Harare",
                        value=config.DEFAULT_LOCATION["city"]
                    )
                    price_btn = gr.Button("💵 Check Prices", variant="primary")
                
                with gr.Column():
                    price_result = gr.Markdown(label="Price Information")
    
            price_btn.click(
                agent.get_market_prices,
                [product_name, product_location],
                price_result
            )

        # Calendar Tab
        with gr.Tab("📅 Farming Calendar"):
            gr.Markdown("### Get seasonal farming calendar")
            
            with gr.Row():
                with gr.Column():
                    calendar_crop = gr.Textbox(label="Crop/Animal", placeholder="e.g., Maize, Goats")
                    calendar_location = gr.Textbox(
                        label="Location",
                        value=config.DEFAULT_LOCATION["city"]
                    )
                    calendar_month = gr.Slider(
                        minimum=1, maximum=12, 
                        value=datetime.now().month,
                        step=1,
                        label="Month"
                    )
                    calendar_btn = gr.Button("📆 Get Calendar", variant="primary")
                
                with gr.Column():
                    calendar_result = gr.JSON(label="Farming Calendar")
            
            calendar_btn.click(
                lambda crop, loc, month: agent.get_farming_calendar(crop, loc, int(month)),
                [calendar_crop, calendar_location, calendar_month],
                calendar_result
            )

        # Cost Calculator Tab
        with gr.Tab("💵 Cost Calculator"):
            gr.Markdown("### Calculate farming costs")
    
            with gr.Row():
                with gr.Column():
                    cost_farm_type = gr.Dropdown(
                        choices=["crop_farming", "fish_farming", "goat_farming", 
                                "pig_farming", "poultry_farming"],
                        label="Farming Type",
                        value="crop_farming"
                    )
                    cost_scale = gr.Number(label="Scale (hectares or animals)", value=10)
                    cost_duration = gr.Number(label="Duration (months)", value=12)
                    cost_calc_btn = gr.Button("🧮 Calculate Costs", variant="primary")
                
                with gr.Column():
                    cost_result = gr.JSON(label="Cost Breakdown")
    
            cost_calc_btn.click(
                agent.calculate_farming_costs,
                [cost_farm_type, cost_scale, cost_duration],
                cost_result
            )

        # Feed Calculator Tab
        with gr.Tab("🧮 Feed Calculator"):
            gr.Markdown("### Calculate livestock feed requirements")
            
            with gr.Row():
                with gr.Column():
                    animal = gr.Dropdown(
                        choices=["goat", "pig", "chicken", "fish", "cattle"],
                        label="Animal Type",
                        value="goat"
                    )
                    count = gr.Number(label="Number of Animals", value=10)
                    weight = gr.Number(label="Average Weight (kg)", value=50)
                    calc_btn = gr.Button("📊 Calculate", variant="primary")
                
                with gr.Column():
                    result = gr.JSON(label="Feed Requirements")
            
            calc_btn.click(
                agent.calculate_feed,
                [animal, count, weight],
                result
            )
        
        # Export Tab
        with gr.Tab("📥 Export Report"):
            gr.Markdown("### Export your consultation as a report")

            export_btn = gr.Button("📄 Export Current Conversation", variant="primary")
            export_result = gr.Textbox(label="Export Status")

            def export_chat(history, user):
                # Convert messages format to list of tuples
                conv_list = []
                for i in range(0, len(history), 2):
                    if i + 1 < len(history):
                        conv_list.append((history[i].get('content', ''), history[i+1].get('content', '')))
                
                filepath = agent.export_conversation(history, user)
                return f"✅ Report exported to: {filepath}"

            export_btn.click(
                export_chat,
                [chatbot, username],
                export_result
            )
        
        # System Info Tab
        with gr.Tab("ℹ️ System Info"):
            gr.Markdown("### System Status & Configuration")
            
            stats_btn = gr.Button("🔄 Refresh Status", variant="primary")
            stats_display = gr.JSON(label="System Status")
            
            stats_btn.click(agent.get_stats, None, stats_display)
            
            # Auto-load stats
            app.load(agent.get_stats, None, stats_display)
            
            gr.Markdown(f"""
            ### ⚙️ Configuration
            - **LLM Model:** {config.LLM_MODEL}
            - **Embedding Model:** {config.EMBEDDING_MODEL}
            - **Default Location:** {config.DEFAULT_LOCATION["city"]}, {config.DEFAULT_LOCATION["country"]}
            
            ### 📋 Requirements
            1. Ollama installed and running
            2. Model downloaded: `ollama pull llama3.2`
            3. Python packages installed
            4. Internet connection (for weather and location services)
            
            ### 🆘 Support
            - **Email:** {config.SUPPORT_EMAIL}
            - **Website:** {config.COMPANY_WEBSITE}
            - Check Ollama: `ollama list`
            - Python version: {sys.version_info.major}.{sys.version_info.minor}
            """)
    
    # Footer with Amaryllis branding
    gr.HTML(f"""
        <div class="footer">
            <div class="footer-logo">
                {AMARYLLIS_LOGO_SVG}
            </div>
            <h3 style="color: #2e7d32; margin: 20px 0 10px 0;">🌾 {config.APP_NAME}</h3>
            <p style="font-size: 1.1em; margin: 10px 0; color: #555;"><strong>{config.APP_TAGLINE}</strong></p>
            <p style="color: #666; margin: 15px 0; line-height: 1.6;">
                Empowering farmers with AI-driven insights, real-time weather data,<br>
                and a vibrant agricultural marketplace.
            </p>
            <div style="margin: 25px 0; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <strong style="color: #7e57c2; font-size: 1.1em;">© 2024 {config.COMPANY_NAME}</strong><br>
                <span style="color: #888; margin-top: 8px; display: inline-block;">All rights reserved</span><br>
                <div style="margin-top: 15px;">
                    <a href="{config.COMPANY_WEBSITE}" target="_blank" style="color: #2e7d32; text-decoration: none; margin: 0 10px; font-weight: 500;">
                        🌐 Website
                    </a> | 
                    <a href="mailto:{config.SUPPORT_EMAIL}" style="color: #7e57c2; text-decoration: none; margin: 0 10px; font-weight: 500;">
                        ✉️ Support
                    </a>
                </div>
            </div>
            <p style="font-size: 0.9em; color: #999; margin-top: 15px;">
                Version {config.VERSION} | {datetime.now().strftime("%Y-%m-%d")} | Made with ❤️ for farmers
            </p>
        </div>
    """)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🌾 AGRIBOT PRO - ENHANCED VERSION - STARTING UP")
    print("="*70 + "\n")
    
    # Check required packages
    print("Checking dependencies...")
    required_packages = {
        "gradio": "Web interface",
        "langchain_ollama": "LLM integration",
        "langchain_community": "Document processing",
        "langchain_core": "LangChain core",
        "langchain_text_splitters": "Text splitting",
        "chromadb": "Vector database",
        "sentence_transformers": "Embeddings",
        "sqlalchemy": "Database",
        "requests": "API calls"
    }
    
    missing_packages = []
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"  ✅ {package} ({description})")
        except ImportError:
            print(f"  ❌ {package} ({description}) - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages detected!")
        print(f"Run: pip install {' '.join(missing_packages)}")
        print("\nOr install all at once:")
        print("pip install gradio langchain-ollama langchain-community langchain-core langchain-text-splitters chromadb sentence-transformers sqlalchemy requests pypdf")
        print("\n" + "="*70)
        
        user_input = input("\nContinue anyway? (y/n): ").strip().lower()
        if user_input != 'y':
            sys.exit(1)
    
    # Check Ollama
    print("\nChecking Ollama...")
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
    print("✅ LAUNCHING ENHANCED INTERFACE")
    print("="*70)
    print(f"🌐 URL: http://localhost:{config.GRADIO_PORT}")
    print(f"🏢 Company: {config.COMPANY_NAME}")
    print(f"📧 Support: {config.SUPPORT_EMAIL}")
    print("📍 Features: Location Services, Weather, Marketplace, Tips")
    print("🛑 Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    try:
        app.launch(
            server_name="0.0.0.0",
            server_port=config.GRADIO_PORT,
            share=True,
            show_error=True
        )
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
        if agent.db_session:
            agent.db_session.close()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check Ollama: ollama list")
        print("2. Install packages: pip install gradio langchain-ollama langchain-community langchain-core langchain-text-splitters chromadb sentence-transformers sqlalchemy requests pypdf")
        print("3. Check port: netstat -an | findstr 7860")
        print("4. Check internet connection for weather/location services")
        print("5. Ensure vector_db directory has write permissions")
        print("6. If using Python 3.13, ensure all packages are updated to latest versions")