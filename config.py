import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class RAGConfig:
    """Configuration for RAG Adaptive Data Analyst Agent"""
    
    # LLM Configuration
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # RAG System Configuration
    RAG_PERSIST_DIRECTORY = os.getenv("RAG_PERSIST_DIRECTORY", "./rag_data")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Vector Database Settings
    CHROMA_SETTINGS = {
        "anonymized_telemetry": False,
        "allow_reset": True
    }
    
    # Context Retrieval Settings
    CONTEXT_TOP_K = int(os.getenv("CONTEXT_TOP_K", "3"))
    SIMILAR_INTERACTIONS_TOP_K = int(os.getenv("SIMILAR_INTERACTIONS_TOP_K", "2"))
    
    # Learning Settings
    SUCCESS_THRESHOLD = float(os.getenv("SUCCESS_THRESHOLD", "0.7"))
    MIN_SUCCESS_SCORE = float(os.getenv("MIN_SUCCESS_SCORE", "0.0"))
    MAX_SUCCESS_SCORE = float(os.getenv("MAX_SUCCESS_SCORE", "1.0"))
    
    # Code Execution Settings
    EXECUTION_TIMEOUT = int(os.getenv("EXECUTION_TIMEOUT", "120"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1500"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
    
    # API Settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Check LLM provider
        if cls.LLM_PROVIDER not in ["openai", "gemini"]:
            errors.append(f"Invalid LLM_PROVIDER: {cls.LLM_PROVIDER}")
        
        # Check API keys
        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required for OpenAI provider")
        elif cls.LLM_PROVIDER == "gemini" and not cls.GOOGLE_API_KEY:
            errors.append("GOOGLE_API_KEY is required for Gemini provider")
        
        # Check numeric settings
        if cls.CONTEXT_TOP_K < 1:
            errors.append("CONTEXT_TOP_K must be >= 1")
        if cls.SUCCESS_THRESHOLD < 0 or cls.SUCCESS_THRESHOLD > 1:
            errors.append("SUCCESS_THRESHOLD must be between 0 and 1")
        if cls.EXECUTION_TIMEOUT < 10:
            errors.append("EXECUTION_TIMEOUT must be >= 10 seconds")
        
        if errors:
            print("Configuration errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
    
    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        """Get LLM-specific configuration"""
        if cls.LLM_PROVIDER == "openai":
            return {
                "model": "gpt-4o",
                "temperature": cls.TEMPERATURE,
                "max_tokens": cls.MAX_TOKENS
            }
        elif cls.LLM_PROVIDER == "gemini":
            return {
                "model": "gemini-1.5-pro",
                "temperature": cls.TEMPERATURE,
                "max_tokens": cls.MAX_TOKENS
            }
        else:
            raise ValueError(f"Unsupported LLM provider: {cls.LLM_PROVIDER}")
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("RAG Adaptive Data Analyst Agent Configuration:")
        print("=" * 50)
        print(f"LLM Provider: {cls.LLM_PROVIDER}")
        print(f"API Keys Configured: {'Yes' if cls._check_api_keys() else 'No'}")
        print(f"RAG Persist Directory: {cls.RAG_PERSIST_DIRECTORY}")
        print(f"Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"Context Top-K: {cls.CONTEXT_TOP_K}")
        print(f"Similar Interactions Top-K: {cls.SIMILAR_INTERACTIONS_TOP_K}")
        print(f"Success Threshold: {cls.SUCCESS_THRESHOLD}")
        print(f"Execution Timeout: {cls.EXECUTION_TIMEOUT}s")
        print(f"Server Host: {cls.HOST}")
        print(f"Server Port: {cls.PORT}")
        print(f"Log Level: {cls.LOG_LEVEL}")
        print("=" * 50)
    
    @classmethod
    def _check_api_keys(cls) -> bool:
        """Check if required API keys are configured"""
        if cls.LLM_PROVIDER == "openai":
            return bool(cls.OPENAI_API_KEY)
        elif cls.LLM_PROVIDER == "gemini":
            return bool(cls.GOOGLE_API_KEY)
        return False

# Environment variable examples for easy setup
ENV_EXAMPLES = """
# Environment Variables for RAG Adaptive Data Analyst Agent

# LLM Configuration
LLM_PROVIDER=openai  # Supported values: 'openai' or 'gemini'
OPENAI_API_KEY=sk-your-openai-api-key-here
GOOGLE_API_KEY=your-gemini-api-key-here

# RAG System Configuration
RAG_PERSIST_DIRECTORY=./rag_data
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Context Retrieval Settings
CONTEXT_TOP_K=3
SIMILAR_INTERACTIONS_TOP_K=2

# Learning Settings
SUCCESS_THRESHOLD=0.7
MIN_SUCCESS_SCORE=0.0
MAX_SUCCESS_SCORE=1.0

# Code Execution Settings
EXECUTION_TIMEOUT=120
MAX_TOKENS=1500
TEMPERATURE=0.2

# API Settings
HOST=0.0.0.0
PORT=8000

# Logging
LOG_LEVEL=INFO
"""

def check_env_file():
    required_vars = [
        "LLM_PROVIDER", "OPENAI_API_KEY", "GOOGLE_API_KEY",
        "RAG_PERSIST_DIRECTORY", "EMBEDDING_MODEL",
        "CONTEXT_TOP_K", "SIMILAR_INTERACTIONS_TOP_K",
        "SUCCESS_THRESHOLD", "MIN_SUCCESS_SCORE", "MAX_SUCCESS_SCORE",
        "EXECUTION_TIMEOUT", "MAX_TOKENS", "TEMPERATURE",
        "HOST", "PORT", "LOG_LEVEL"
    ]
    missing = [var for var in required_vars if os.getenv(var) is None]
    if missing:
        print("Missing variables in .env file:")
        for var in missing:
            print(f"  - {var}")
    else:
        print("All required environment variables are set.")

# Usage:
check_env_file()