#!/usr/bin/env python3
"""
Example usage of RAG Adaptive Data Analyst Agent
Demonstrates various data analysis capabilities with learning and feedback.
"""

import requests
import json
import time
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"

def example_basic_analysis():
    """Example 1: Basic mathematical analysis"""
    print("📊 Example 1: Basic Mathematical Analysis")
    print("-" * 40)
    
    task = "Calculate the mean, median, and standard deviation of the numbers [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]"
    
    response = requests.post(
        f"{BASE_URL}/api/analyze",
        data=task,
        headers={'Content-Type': 'text/plain'}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success! Result: {result['result']}")
        print(f"📈 Success Score: {result['success_score']}")
        print(f"🔍 Context Used: {len(result['context_used'])} items")
        return result['interaction_id']
    else:
        print(f"❌ Error: {response.text}")
        return None

def example_web_scraping():
    """Example 2: Web scraping and data analysis"""
    print("\n🌐 Example 2: Web Scraping and Data Analysis")
    print("-" * 40)
    
    task = "Scrape the table from https://en.wikipedia.org/wiki/List_of_countries_by_population and return the top 10 countries by population"
    
    response = requests.post(
        f"{BASE_URL}/api/analyze",
        data=task,
        headers={'Content-Type': 'text/plain'}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success! Retrieved {len(result['result'])} countries")
        print(f"📈 Success Score: {result['success_score']}")
        print(f"🔍 Context Used: {len(result['context_used'])} items")
        
        # Show first few results
        if result['result'] and len(result['result']) > 0:
            print("Top 3 countries:")
            for i, country in enumerate(result['result'][:3]):
                print(f"  {i+1}. {country}")
        
        return result['interaction_id']
    else:
        print(f"❌ Error: {response.text}")
        return None

def example_visualization():
    """Example 3: Data visualization"""
    print("\n📈 Example 3: Data Visualization")
    print("-" * 40)
    
    task = "Create a bar chart showing sales data for Q1: 100, Q2: 150, Q3: 200, Q4: 175 and return it as base64 image"
    
    response = requests.post(
        f"{BASE_URL}/api/analyze",
        data=task,
        headers={'Content-Type': 'text/plain'}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success! Generated visualization")
        print(f"📈 Success Score: {result['success_score']}")
        print(f"🔍 Context Used: {len(result['context_used'])} items")
        print(f"🖼️  Image data length: {len(str(result['result']))} characters")
        return result['interaction_id']
    else:
        print(f"❌ Error: {response.text}")
        return None

def example_feedback_and_learning(interaction_id):
    """Example 4: Providing feedback to improve learning"""
    print("\n💬 Example 4: Feedback and Learning")
    print("-" * 40)
    
    if not interaction_id:
        print("⚠️  No interaction ID available for feedback")
        return
    
    feedback_data = {
        "interaction_id": interaction_id,
        "feedback": "Excellent analysis! The visualization is clear and informative.",
        "success_score": 0.95
    }
    
    response = requests.post(
        f"{BASE_URL}/api/feedback",
        json=feedback_data
    )
    
    if response.status_code == 200:
        print("✅ Feedback submitted successfully!")
        print("🧠 The system will learn from this feedback for future interactions")
    else:
        print(f"❌ Error submitting feedback: {response.text}")

def example_custom_context():
    """Example 5: Adding custom context to the knowledge base"""
    print("\n📚 Example 5: Adding Custom Context")
    print("-" * 40)
    
    custom_context = {
        "content": "Advanced time series analysis using pandas with seasonal decomposition and trend analysis",
        "metadata": {
            "type": "data_analysis",
            "category": "time_series",
            "difficulty": "advanced",
            "domain": "forecasting"
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/api/context",
        json=custom_context
    )
    
    if response.status_code == 200:
        print("✅ Custom context added successfully!")
        print("🔍 This context will be used for future time series analysis tasks")
    else:
        print(f"❌ Error adding context: {response.text}")

def example_system_stats():
    """Example 6: Checking system statistics"""
    print("\n📊 Example 6: System Statistics")
    print("-" * 40)
    
    response = requests.get(f"{BASE_URL}/api/stats")
    
    if response.status_code == 200:
        stats = response.json()
        print("📈 System Performance:")
        print(f"  • Total Interactions: {stats['total_interactions']}")
        print(f"  • Successful Interactions: {stats['successful_interactions']}")
        print(f"  • Success Rate: {stats['success_rate']:.2%}")
        print(f"  • Average Success Score: {stats['average_success_score']:.2f}")
        print(f"  • Context Items: {stats['context_count']}")
        print(f"  • System Learning: {'Active' if stats['system_learning'] else 'Inactive'}")
    else:
        print(f"❌ Error getting stats: {response.text}")

def example_learning_progression():
    """Example 7: Demonstrating learning progression"""
    print("\n🧠 Example 7: Learning Progression")
    print("-" * 40)
    
    # First interaction
    task1 = "Calculate the sum of numbers from 1 to 10"
    print(f"🔍 First task: {task1}")
    
    response1 = requests.post(
        f"{BASE_URL}/api/analyze",
        data=task1,
        headers={'Content-Type': 'text/plain'}
    )
    
    if response1.status_code == 200:
        result1 = response1.json()
        interaction_id1 = result1['interaction_id']
        print(f"✅ First interaction - Success Score: {result1['success_score']}")
        
        # Provide positive feedback
        feedback1 = {
            "interaction_id": interaction_id1,
            "feedback": "Perfect calculation! Very accurate.",
            "success_score": 0.95
        }
        requests.post(f"{BASE_URL}/api/feedback", json=feedback1)
        
        # Second similar interaction
        time.sleep(1)  # Small delay
        task2 = "Calculate the sum of numbers from 1 to 20"
        print(f"🔍 Second task: {task2}")
        
        response2 = requests.post(
            f"{BASE_URL}/api/analyze",
            data=task2,
            headers={'Content-Type': 'text/plain'}
        )
        
        if response2.status_code == 200:
            result2 = response2.json()
            print(f"✅ Second interaction - Success Score: {result2['success_score']}")
            print(f"🔍 Context used in second interaction: {len(result2['context_used'])} items")
            
            # Check if the system learned
            if result2['success_score'] >= result1['success_score']:
                print("🎉 System shows learning improvement!")
            else:
                print("⚠️  System may need more training data")
        else:
            print(f"❌ Error in second interaction: {response2.text}")
    else:
        print(f"❌ Error in first interaction: {response1.text}")

def run_all_examples():
    """Run all examples to demonstrate RAG Adaptive capabilities"""
    print("🚀 RAG Adaptive Data Analyst Agent - Example Usage")
    print("=" * 60)
    
    # Check if server is running
    try:
        health_response = requests.get(f"{BASE_URL}/api/health", timeout=5)
        if health_response.status_code != 200:
            print("❌ Server is not running. Please start the server with: python main.py")
            return
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to server. Please start the server with: python main.py")
        return
    
    print("✅ Server is running and ready!")
    
    # Run examples
    interaction_id1 = example_basic_analysis()
    interaction_id2 = example_web_scraping()
    interaction_id3 = example_visualization()
    
    # Provide feedback
    if interaction_id3:
        example_feedback_and_learning(interaction_id3)
    
    # Add custom context
    example_custom_context()
    
    # Check system stats
    example_system_stats()
    
    # Demonstrate learning
    example_learning_progression()
    
    # Final stats
    print("\n📈 Final System Statistics:")
    example_system_stats()
    
    print("\n✅ All examples completed!")
    print("\n🎯 The RAG Adaptive system has:")
    print("  • Learned from your interactions")
    print("  • Stored relevant context")
    print("  • Improved its response patterns")
    print("  • Built a knowledge base for future queries")

if __name__ == "__main__":
    run_all_examples() 