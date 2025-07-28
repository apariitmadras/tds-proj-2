import requests
import json
import time
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    response = requests.get(f"{BASE_URL}/api/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_basic_analysis():
    """Test basic data analysis without RAG"""
    print("📊 Testing basic data analysis...")
    
    # Test 1: Simple data analysis
    task = "Create a list of numbers from 1 to 10 and calculate their sum"
    
    response = requests.post(
        f"{BASE_URL}/api/analyze",
        data=task.encode('utf-8'),
        headers={'Content-Type': 'text/plain'}
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Result: {result['result']}")
        print(f"Interaction ID: {result['interaction_id']}")
        print(f"Success Score: {result['success_score']}")
        print(f"Context Used: {len(result['context_used'])} items")
        return result['interaction_id']
    else:
        print(f"Error: {response.text}")
        return None

def test_web_scraping_analysis():
    """Test web scraping analysis with RAG enhancement"""
    print("🌐 Testing web scraping analysis...")
    
    task = "Scrape the table from https://en.wikipedia.org/wiki/List_of_countries_by_population and return the top 5 countries by population"
    
    response = requests.post(
        f"{BASE_URL}/api/analyze",
        data=task.encode('utf-8'),
        headers={'Content-Type': 'text/plain'}
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Result: {result['result'][:3]}...")  # Show first 3 items
        print(f"Interaction ID: {result['interaction_id']}")
        print(f"Success Score: {result['success_score']}")
        return result['interaction_id']
    else:
        print(f"Error: {response.text}")
        return None

def test_visualization_analysis():
    """Test visualization analysis"""
    print("📈 Testing visualization analysis...")
    
    task = "Create a bar chart of the numbers [1, 2, 3, 4, 5] and return it as base64 image"
    
    response = requests.post(
        f"{BASE_URL}/api/analyze",
        data=task.encode('utf-8'),
        headers={'Content-Type': 'text/plain'}
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Result type: {type(result['result'])}")
        print(f"Result starts with: {str(result['result'])[:50]}...")
        print(f"Interaction ID: {result['interaction_id']}")
        print(f"Success Score: {result['success_score']}")
        return result['interaction_id']
    else:
        print(f"Error: {response.text}")
        return None

def test_feedback_system(interaction_id):
    """Test the feedback system"""
    print("💬 Testing feedback system...")
    
    feedback_data = {
        "interaction_id": interaction_id,
        "feedback": "This was a great analysis! Very helpful.",
        "success_score": 0.9
    }
    
    response = requests.post(
        f"{BASE_URL}/api/feedback",
        json=feedback_data
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_context_addition():
    """Test adding new context to the knowledge base"""
    print("📚 Testing context addition...")
    
    new_context = {
        "content": "Advanced statistical analysis using scipy.stats for hypothesis testing and correlation analysis",
        "metadata": {
            "type": "data_analysis",
            "category": "statistics",
            "difficulty": "advanced"
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/api/context",
        json=new_context
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_system_stats():
    """Test getting system statistics"""
    print("📊 Testing system statistics...")
    
    response = requests.get(f"{BASE_URL}/api/stats")
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        stats = response.json()
        print(f"Total Interactions: {stats['total_interactions']}")
        print(f"Successful Interactions: {stats['successful_interactions']}")
        print(f"Success Rate: {stats['success_rate']:.2%}")
        print(f"Average Success Score: {stats['average_success_score']:.2f}")
        print(f"Context Count: {stats['context_count']}")
        print(f"System Learning: {stats['system_learning']}")
    else:
        print(f"Error: {response.text}")
    print()

def test_learning_progression():
    """Test how the system learns and improves over multiple interactions"""
    print("🧠 Testing learning progression...")
    
    # First interaction
    task1 = "Calculate the mean of numbers [10, 20, 30, 40, 50]"
    response1 = requests.post(
        f"{BASE_URL}/api/analyze",
        data=task1.encode('utf-8'),
        headers={'Content-Type': 'text/plain'}
    )
    
    if response1.status_code == 200:
        result1 = response1.json()
        interaction_id1 = result1['interaction_id']
        print(f"First interaction - Success Score: {result1['success_score']}")
        
        # Provide feedback
        feedback1 = {
            "interaction_id": interaction_id1,
            "feedback": "Perfect calculation!",
            "success_score": 0.95
        }
        requests.post(f"{BASE_URL}/api/feedback", json=feedback1)
        
        # Second similar interaction
        time.sleep(1)  # Small delay
        task2 = "Calculate the mean of numbers [5, 15, 25, 35, 45]"
        response2 = requests.post(
            f"{BASE_URL}/api/analyze",
            data=task2.encode('utf-8'),
            headers={'Content-Type': 'text/plain'}
        )
        
        if response2.status_code == 200:
            result2 = response2.json()
            print(f"Second interaction - Success Score: {result2['success_score']}")
            print(f"Context used in second interaction: {len(result2['context_used'])} items")
            
            # Check if the system learned
            if result2['success_score'] >= result1['success_score']:
                print("✅ System shows learning improvement!")
            else:
                print("⚠️ System may need more training data")
        else:
            print(f"Error in second interaction: {response2.text}")
    else:
        print(f"Error in first interaction: {response1.text}")
    print()

def run_comprehensive_test():
    """Run all tests to demonstrate RAG Adaptive capabilities"""
    print("🚀 Starting RAG Adaptive Data Analyst Agent Tests")
    print("=" * 60)
    
    # Test 1: Health check
    test_health_check()
    
    # Test 2: Basic analysis
    interaction_id1 = test_basic_analysis()
    
    # Test 3: Web scraping
    interaction_id2 = test_web_scraping_analysis()
    
    # Test 4: Visualization
    interaction_id3 = test_visualization_analysis()
    
    # Test 5: Add context
    test_context_addition()
    
    # Test 6: Feedback system
    if interaction_id1:
        test_feedback_system(interaction_id1)
    
    # Test 7: System stats
    test_system_stats()
    
    # Test 8: Learning progression
    test_learning_progression()
    
    # Final stats
    print("📈 Final System Statistics:")
    test_system_stats()
    
    print("✅ All tests completed!")

if __name__ == "__main__":
    run_comprehensive_test() 