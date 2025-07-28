#!/usr/bin/env python3
"""
Startup script for RAG Adaptive Data Analyst Agent
Validates configuration and provides helpful setup information.
"""

import os
import sys
from config import RAGConfig, ENV_EXAMPLES

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        ('fastapi', 'fastapi'),
        ('uvicorn', 'uvicorn'),
        ('openai', 'openai'),
        ('google.generativeai', 'google-generativeai'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('sklearn', 'scikit-learn'),
        ('duckdb', 'duckdb'),
        ('requests', 'requests'),
        ('bs4', 'beautifulsoup4'),
        ('chromadb', 'chromadb'),
        ('sentence_transformers', 'sentence-transformers'),
        ('langchain', 'langchain'),
        ('langchain_community', 'langchain-community'),
        ('langchain_openai', 'langchain-openai'),
        ('redis', 'redis'),
        ('pickle_mixin', 'pickle-mixin'),
        ('tiktoken', 'tiktoken'),
        ('nltk', 'nltk')
    ]
    
    missing_packages = []
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall missing packages with:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_environment():
    """Check environment variables and configuration"""
    print("\n🔧 Checking configuration...")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️  No .env file found. Creating example .env file...")
        with open('.env.example', 'w') as f:
            f.write(ENV_EXAMPLES)
        print("📝 Created .env.example file. Please copy it to .env and add your API keys.")
    
    # Validate configuration
    if not RAGConfig.validate():
        print("❌ Configuration validation failed!")
        return False
    
    print("✅ Configuration is valid")
    return True

def print_setup_instructions():
    """Print setup instructions"""
    print("\n🚀 RAG Adaptive Data Analyst Agent Setup")
    print("=" * 50)
    
    print("\n1. Set your API keys:")
    if RAGConfig.LLM_PROVIDER == "openai":
        print("   export OPENAI_API_KEY=sk-your-api-key-here")
    elif RAGConfig.LLM_PROVIDER == "google":
        print("   export GOOGLE_API_KEY=your-gemini-api-key-here")
    
    print("\n2. Run the server:")
    print("   python main.py")
    
    print("\n3. Test the system:")
    print("   python test_rag_agent.py")
    
    print("\n4. API Documentation:")
    print("   http://localhost:8000/docs")
    
    print("\n5. Health Check:")
    print("   curl http://localhost:8000/api/health")

def main():
    """Main startup function"""
    print("🧠 RAG Adaptive Data Analyst Agent")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment
    if not check_environment():
        print("\n❌ Setup incomplete. Please fix the issues above.")
        sys.exit(1)
    
    # Print configuration
    RAGConfig.print_config()
    
    # Print setup instructions
    print_setup_instructions()
    
    print("\n✅ System is ready to start!")
    print("\nTo start the server, run: python main.py")

if __name__ == "__main__":
    main() 