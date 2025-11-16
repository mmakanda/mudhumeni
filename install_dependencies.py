#!/usr/bin/env python3
"""
Quick installer for AgriBot Pro dependencies
Run: python install_dependencies.py
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    print(f"Installing {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    print("="*70)
    print("🌾 AgriBot Pro - Dependency Installer")
    print("="*70)
    print()
    
    # Core packages (required)
    core_packages = [
        "streamlit>=1.30.0",
        "pandas>=2.0.0",
        "plotly>=5.18.0",
        "requests>=2.31.0",
        "sqlalchemy>=2.0.0",
    ]
    
    # Optional packages (enhance experience)
    optional_packages = [
        "streamlit-option-menu>=0.3.6",  # Better navigation
    ]
    
    # AI packages
    ai_packages = [
        "langchain>=0.2.0",
        "langchain-ollama>=0.1.0",
        "langchain-community>=0.2.0",
        "langchain-core>=0.2.0",
        "langchain-text-splitters>=0.2.0",
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.0",
        "pypdf>=3.17.0",
    ]
    
    print("📦 Installing core packages...")
    for package in core_packages:
        install_package(package)
    
    print("\n📦 Installing optional packages (for better UI)...")
    for package in optional_packages:
        if not install_package(package):
            print("⚠️ Optional package failed, app will still work")
    
    print("\n🤖 Installing AI packages...")
    for package in ai_packages:
        install_package(package)
    
    print("\n" + "="*70)
    print("✅ Installation complete!")
    print("="*70)
    print("\nNext steps:")
    print("1. Ensure Ollama is running: ollama list")
    print("2. Pull the model: ollama pull llama3.2")
    print("3. Run the app: streamlit run agribot_streamlit.py")
    print()

if __name__ == "__main__":
    main()