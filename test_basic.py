#!/usr/bin/env python3
"""
Basic test script for RAG Adaptive Data Analyst Agent
Tests core functionality without requiring API keys.
"""

import os
import sys
import tempfile
import shutil

def test_imports():
    """Test that all core modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from config import RAGConfig
        print("✅ Config module imported successfully")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from rag_system import RAGAdaptiveSystem, Interaction
        print("✅ RAG system module imported successfully")
    except Exception as e:
        print(f"❌ RAG system import failed: {e}")
        return False
    
    try:
        from utils import scrape_table_from_url, run_duckdb_query, plot_and_encode_base64
        print("✅ Utils module imported successfully")
    except Exception as e:
        print(f"❌ Utils import failed: {e}")
        return False
    
    return True

def test_rag_system_initialization():
    """Test RAG system initialization with temporary directory"""
    print("\n🧪 Testing RAG system initialization...")
    
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp()
    
    try:
        from rag_system import RAGAdaptiveSystem
        
        # Initialize RAG system
        rag_system = RAGAdaptiveSystem(persist_directory=temp_dir)
        print("✅ RAG system initialized successfully")
        
        # Test context retrieval
        contexts = rag_system.retrieve_relevant_context("test query")
        print(f"✅ Context retrieval works: {len(contexts)} contexts found")
        
        # Test interaction creation
        from datetime import datetime
        from rag_system import Interaction
        interaction = Interaction(
            timestamp=datetime.now(),
            user_query="test query",
            generated_code="result = 42",
            execution_result=42,
            success_score=0.8
        )
        rag_system.add_interaction(interaction)
        print("✅ Interaction creation and storage works")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG system test failed: {e}")
        return False
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_configuration():
    """Test configuration system"""
    print("\n🧪 Testing configuration...")
    
    try:
        from config import RAGConfig
        
        # Test configuration printing
        RAGConfig.print_config()
        print("✅ Configuration system works")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_utils():
    """Test utility functions"""
    print("\n🧪 Testing utility functions...")
    
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Test basic data operations
        data = [1, 2, 3, 4, 5]
        df = pd.DataFrame({'numbers': data})
        print("✅ Pandas operations work")
        
        # Test numpy operations
        arr = np.array(data)
        mean_val = np.mean(arr)
        print(f"✅ Numpy operations work: mean = {mean_val}")
        
        # Test matplotlib (without display)
        fig, ax = plt.subplots()
        ax.plot(data)
        plt.close(fig)  # Close to avoid display
        print("✅ Matplotlib operations work")
        
        return True
        
    except Exception as e:
        print(f"❌ Utils test failed: {e}")
        return False

def run_basic_tests():
    """Run all basic tests"""
    print("🚀 RAG Adaptive Data Analyst Agent - Basic Tests")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Configuration Tests", test_configuration),
        ("Utils Tests", test_utils),
        ("RAG System Tests", test_rag_system_initialization),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed! The system is ready for API key setup.")
        print("\nNext steps:")
        print("1. Set your OpenAI API key: export OPENAI_API_KEY=sk-your-key-here")
        print("2. Run the server: python main.py")
        print("3. Test with API: python test_rag_agent.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1) 