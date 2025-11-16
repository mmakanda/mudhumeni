#!/usr/bin/env python3
"""
Setup script for AgriBot Knowledge Base Resources
Creates directory structure and sample resources

Run: python setup_resources.py
"""

from pathlib import Path
import shutil

def create_directory_structure():
    """Create necessary directories"""
    directories = [
        "agri_data/resources",
        "uploads",
        "exports",
        "vector_db",
        "logs",
        "database"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")

def create_sample_documents():
    """Create sample farming guides"""
    
    sample_docs = {
        "maize_farming_guide.pdf": """
# Maize Farming Guide for Zimbabwe

## Introduction
Maize is Zimbabwe's staple food crop and most important cereal grain.

## Varieties
- SC Series (SC403, SC513, SC637)
- ZM Series (ZM421, ZM523, ZM621)
- Local varieties

## Planting Season
- Best time: November to December (first rains)
- Soil temperature: Above 15°C
- Soil pH: 5.5 to 7.0

## Land Preparation
1. Plough deeply (20-25cm)
2. Remove weeds and crop residues
3. Apply lime if pH below 5.5
4. Make ridges or furrows

## Planting
- Row spacing: 75-90cm
- Plant spacing: 25-30cm
- Planting depth: 5cm
- Seeds per hole: 2-3

## Fertilizer Application
**Basal Fertilizer (at planting):**
- Compound D: 200-300 kg/ha
- Apply in furrow, cover with soil

**Top Dressing (4-6 weeks after planting):**
- Ammonium Nitrate: 200-250 kg/ha
- Apply along rows, mix with soil

## Weed Control
- First weeding: 2-3 weeks after planting
- Second weeding: 5-6 weeks after planting
- Use herbicides if labor is limited

## Pest Management
**Fall Armyworm:**
- Scout regularly
- Use biopesticides (Bt)
- Chemical control if severe

**Stalk Borers:**
- Remove and destroy egg masses
- Apply granular insecticides

## Diseases
**Grey Leaf Spot:**
- Use resistant varieties
- Rotate crops
- Apply fungicides if needed

**Maize Streak Virus:**
- Plant resistant varieties
- Control leafhopper vectors

## Irrigation
- Critical stages: flowering and grain filling
- Water requirement: 450-650mm per season
- Avoid water stress during tasseling

## Harvesting
- Maturity: 120-140 days after planting
- Moisture content: 20-25% for mechanical harvesting
- Moisture content: 12-14% for storage

## Post-Harvest Handling
1. Dry to 12.5% moisture
2. Shell and clean
3. Treat with appropriate chemicals
4. Store in dry, cool place

## Expected Yields
- Commercial farms: 6-10 tons/ha
- Small-scale farms: 3-5 tons/ha
- Factors: variety, management, rainfall

## Cost Estimation (per hectare)
- Seeds: $50-80
- Fertilizer: $200-300
- Pesticides: $50-100
- Labor: $150-200
- Equipment: $100-150
**Total: $550-830/ha**

## Marketing
- GMB (Grain Marketing Board)
- Private buyers
- Seed companies
- Local markets

## Contact Information
AGRITEX: +263 [number]
Seed Co: +263 [number]

---
© 2024 AgriBot Pro | Amaryllis Success
        """,
        
        "fish_farming_manual.pdf": """
# Fish Farming Starter Manual - Tilapia Production

## Introduction
Tilapia is ideal for Zimbabwe's climate and grows well in ponds and tanks.

## Getting Started

### Site Selection
- Reliable water source
- Good soil for pond construction
- Flat or gently sloping land
- Away from flood areas

### Pond Construction
**Size:** 200-500 m²
**Depth:** 1-1.5 meters
**Shape:** Rectangular preferred
**Bottom:** Slightly sloped for drainage

### Water Quality Requirements
- Temperature: 25-30°C (optimal)
- pH: 6.5-8.5
- Dissolved Oxygen: >5 mg/L
- Ammonia: <0.1 mg/L

## Stocking

### Fingerling Selection
- Size: 5-10 grams
- Source: Reputable hatcheries
- Health: Active, no deformities

### Stocking Density
- Extensive: 1-2 fish/m²
- Semi-intensive: 3-5 fish/m²
- Intensive: 5-10 fish/m²

## Feeding

### Feed Types
- Commercial pellets (32-40% protein)
- Farm-made feed
- Supplementary feeds

### Feeding Schedule
- Fingerlings: 4-5 times daily
- Juveniles: 3 times daily
- Adults: 2 times daily

### Feed Amount
- 3-5% of body weight daily
- Adjust based on growth

## Water Management

### Water Quality Monitoring
- Test pH weekly
- Check temperature daily
- Monitor oxygen levels
- Watch for ammonia buildup

### Pond Maintenance
- Remove debris regularly
- Partial water changes (10-20% weekly)
- Clean filters/aerators
- Control algae growth

## Disease Prevention

### Common Diseases
**Bacterial Infections:**
- Symptoms: Lesions, fin rot
- Treatment: Antibiotics (veterinary advice)

**Parasites:**
- Symptoms: Flashing, scratching
- Treatment: Salt baths, proper medication

**Fungal Infections:**
- Symptoms: Cotton-like growth
- Treatment: Antifungal medication

### Prevention
- Maintain good water quality
- Avoid overstocking
- Quarantine new fish
- Proper nutrition

## Growth and Harvesting

### Growth Timeline
- Fingerling to market size: 6-8 months
- Market size: 300-500 grams

### Harvesting Methods
- Total harvest: Drain pond
- Partial harvest: Seine net
- Selective harvest: Cast net

### Post-Harvest Handling
- Clean immediately
- Gut and scale
- Ice or process quickly
- Maintain cold chain

## Economics

### Initial Investment (0.05 ha pond)
- Pond construction: $1,500-2,500
- Fingerlings: $200-300
- Feed (6 months): $500-800
- Equipment: $300-500
**Total: $2,500-4,100**

### Revenue (per cycle)
- Production: 1,500-2,000 kg
- Price: $3-4/kg
- Gross income: $4,500-8,000
- Net profit: $2,000-4,000

### ROI Timeline
- Break-even: 1-2 years
- Good management: High profitability

## Marketing

### Market Channels
- Local markets
- Supermarkets
- Hotels and restaurants
- Fish traders
- Direct consumers

### Value Addition
- Smoked fish
- Filleted fish
- Fish powder
- Fish sausages

## Record Keeping
- Stock records
- Feeding records
- Water quality logs
- Financial records
- Mortality records

## Support Services
- Veterinary services
- Extension officers
- Fish farmers associations
- Training programs

## Contact Information
Department of Fisheries: +263 [number]
Aquaculture Association: +263 [number]

---
© 2024 AgriBot Pro | Amaryllis Success
        """
    }
    
    resources_dir = Path("agri_data/resources")
    
    for filename, content in sample_docs.items():
        filepath = resources_dir / filename
        
        # Create as text file (can be converted to PDF later)
        txt_filepath = filepath.with_suffix('.txt')
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Created: {txt_filepath}")
    
    print("\n📝 Note: Sample documents created as TXT files.")
    print("   For production, convert these to PDF using tools like:")
    print("   - wkhtmltopdf")
    print("   - Online converters")
    print("   - Or create professionally designed PDFs")

def create_readme():
    """Create README for resources directory"""
    readme_content = """
# AgriBot Pro - Knowledge Base Resources

## Directory Structure

```
agri_data/resources/  - Downloadable farming guides and manuals
uploads/             - User-uploaded documents
vector_db/           - Vector database for AI retrieval
exports/             - Exported chat histories and reports
```

## Adding New Resources

1. **Create or obtain farming guides** in PDF format
2. **Save to**: `agri_data/resources/`
3. **Update the resource list** in `agribot_streamlit.py` (search for "resources = [")
4. **Add metadata**: title, description, category, file size

## Resource Categories

- `crop_farming` - Crop cultivation guides
- `fish_farming` - Aquaculture manuals
- `goat_farming` - Goat rearing guides
- `pig_farming` - Pig production manuals
- `poultry_farming` - Poultry management guides
- `cattle_farming` - Cattle farming resources
- `general_agriculture` - General farming topics

## Content Guidelines

### What to Include:
- ✅ Practical, actionable advice
- ✅ Local/regional context (Zimbabwe, Southern Africa)
- ✅ Step-by-step instructions
- ✅ Visual aids (diagrams, photos)
- ✅ Cost estimates
- ✅ Contact information for support

### Quality Standards:
- Clear, simple language
- Accurate, up-to-date information
- Culturally appropriate
- Accessible to small-scale farmers
- Mobile-friendly format

## Recommended Resources to Add

1. **Crop Guides**:
   - Tobacco farming
   - Cotton production
   - Wheat cultivation
   - Vegetable gardening

2. **Livestock Guides**:
   - Dairy farming
   - Rabbit production
   - Bee keeping

3. **Business Guides**:
   - Farm business planning
   - Marketing strategies
   - Financial management

4. **Technology Guides**:
   - Irrigation systems
   - Solar energy for farms
   - Mobile apps for farmers

## Sources for Quality Content

1. **Government Resources**:
   - Ministry of Agriculture
   - AGRITEX publications
   - Research stations

2. **International Organizations**:
   - FAO (Food and Agriculture Organization)
   - CIMMYT (maize and wheat research)
   - WorldFish

3. **Universities**:
   - University of Zimbabwe
   - Midlands State University

4. **NGOs and Development Partners**:
   - CARE International
   - World Vision
   - Heifer International

## Legal Considerations

- Ensure you have rights to distribute content
- Credit original authors/sources
- Use open-access or licensed materials
- Create original content when possible

## Maintenance

- Review and update annually
- Remove outdated information
- Add new seasonal guides
- Track download statistics
- Gather user feedback

---
For questions: support@amaryllissuccess.co.zw
"""
    
    readme_path = Path("agri_data/README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✅ Created: {readme_path}")

def main():
    print("="*70)
    print("🌾 AgriBot Pro - Resource Setup")
    print("="*70)
    print()
    
    print("Step 1: Creating directory structure...")
    create_directory_structure()
    print()
    
    print("Step 2: Creating sample documents...")
    create_sample_documents()
    print()
    
    print("Step 3: Creating documentation...")
    create_readme()
    print()
    
    print("="*70)
    print("✅ Setup Complete!")
    print("="*70)
    print()
    print("Next steps:")
    print("1. Review sample documents in: agri_data/resources/")
    print("2. Add more professional PDF guides")
    print("3. Update resource list in agribot_streamlit.py")
    print("4. Run the app: streamlit run agribot_streamlit.py")
    print()

if __name__ == "__main__":
    main()