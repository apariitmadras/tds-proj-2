import os
import json
import pickle
from typing import List, Dict, Any, Optional
from datetime import datetime
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Interaction:
    """Represents a user interaction with the system"""
    timestamp: datetime
    user_query: str
    generated_code: str
    execution_result: Any
    user_feedback: Optional[str] = None
    success_score: Optional[float] = None
    context_used: List[str] = None

class RAGAdaptiveSystem:
    """
    RAG Adaptive System for Data Analysis Agent
    - Stores and retrieves relevant context
    - Learns from user interactions
    - Adapts responses based on feedback
    """
    
    def __init__(self, persist_directory: str = "./rag_data"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize vector database
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize collections
        self._init_collections()
        
        # Load interaction history
        self.interaction_history = self._load_interaction_history()
        
        # Initialize with default knowledge base
        self._initialize_knowledge_base()
    
    def _init_collections(self):
        """Initialize ChromaDB collections"""
        try:
            self.context_collection = self.client.get_or_create_collection(
                name="data_analysis_context",
                metadata={"description": "Context for data analysis tasks"}
            )
            self.interaction_collection = self.client.get_or_create_collection(
                name="user_interactions",
                metadata={"description": "User interaction history for learning"}
            )
        except Exception as e:
            logger.error(f"Error initializing collections: {e}")
            raise
    
    def _initialize_knowledge_base(self):
        """Initialize the knowledge base with common data analysis patterns"""
        default_contexts = [
            {
                "content": "Data scraping from web tables using scrape_table_from_url() function",
                "metadata": {"type": "data_source", "category": "web_scraping"}
            },
            {
                "content": "SQL queries using DuckDB with run_duckdb_query() function",
                "metadata": {"type": "data_analysis", "category": "sql"}
            },
            {
                "content": "Creating visualizations with matplotlib and encoding as base64",
                "metadata": {"type": "visualization", "category": "plotting"}
            },
            {
                "content": "Pandas data manipulation and analysis techniques",
                "metadata": {"type": "data_analysis", "category": "pandas"}
            },
            {
                "content": "Statistical analysis using numpy and scikit-learn",
                "metadata": {"type": "data_analysis", "category": "statistics"}
            }
        ]
        
        # Add default contexts if collection is empty
        if self.context_collection.count() == 0:
            self.add_contexts(default_contexts)
    
    def add_contexts(self, contexts: List[Dict[str, Any]]):
        """Add new contexts to the knowledge base"""
        try:
            for i, context in enumerate(contexts):
                embedding = self.embedding_model.encode(context["content"]).tolist()
                self.context_collection.add(
                    embeddings=[embedding],
                    documents=[context["content"]],
                    metadatas=[context.get("metadata", {})],
                    ids=[f"context_{len(self.context_collection.get()['ids']) + i}"]
                )
            logger.info(f"Added {len(contexts)} contexts to knowledge base")
        except Exception as e:
            logger.error(f"Error adding contexts: {e}")
    
    def retrieve_relevant_context(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve relevant context for a given query"""
        try:
            query_embedding = self.embedding_model.encode(query).tolist()
            results = self.context_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            return results['documents'][0] if results['documents'] else []
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def add_interaction(self, interaction: Interaction):
        """Add a new interaction to the learning system"""
        try:
            # Store in interaction collection
            interaction_embedding = self.embedding_model.encode(interaction.user_query).tolist()
            self.interaction_collection.add(
                embeddings=[interaction_embedding],
                documents=[json.dumps(interaction.__dict__, default=str)],
                metadatas=[{
                    "timestamp": interaction.timestamp.isoformat(),
                    "success_score": interaction.success_score,
                    "feedback": interaction.user_feedback
                }],
                ids=[f"interaction_{len(self.interaction_collection.get()['ids'])}"]
            )
            
            # Add to local history
            self.interaction_history.append(interaction)
            self._save_interaction_history()
            
            # Learn from successful interactions
            if interaction.success_score and interaction.success_score > 0.7:
                self._learn_from_successful_interaction(interaction)
                
        except Exception as e:
            logger.error(f"Error adding interaction: {e}")
    
    def _learn_from_successful_interaction(self, interaction: Interaction):
        """Extract patterns from successful interactions and add to knowledge base"""
        try:
            # Extract code patterns and add as context
            if interaction.generated_code:
                new_context = {
                    "content": f"Successful code pattern for: {interaction.user_query}\nCode: {interaction.generated_code}",
                    "metadata": {
                        "type": "learned_pattern",
                        "success_score": interaction.success_score,
                        "original_query": interaction.user_query
                    }
                }
                self.add_contexts([new_context])
                logger.info("Learned new pattern from successful interaction")
        except Exception as e:
            logger.error(f"Error learning from interaction: {e}")
    
    def get_similar_interactions(self, query: str, top_k: int = 2) -> List[Interaction]:
        """Retrieve similar past interactions"""
        try:
            query_embedding = self.embedding_model.encode(query).tolist()
            results = self.interaction_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            similar_interactions = []
            for doc in results['documents'][0]:
                try:
                    interaction_data = json.loads(doc)
                    interaction = Interaction(**interaction_data)
                    similar_interactions.append(interaction)
                except Exception as e:
                    logger.warning(f"Error parsing interaction: {e}")
                    continue
            
            return similar_interactions
        except Exception as e:
            logger.error(f"Error retrieving similar interactions: {e}")
            return []
    
    def _load_interaction_history(self) -> List[Interaction]:
        """Load interaction history from disk"""
        history_file = os.path.join(self.persist_directory, "interaction_history.pkl")
        try:
            if os.path.exists(history_file):
                with open(history_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading interaction history: {e}")
        return []
    
    def _save_interaction_history(self):
        """Save interaction history to disk"""
        history_file = os.path.join(self.persist_directory, "interaction_history.pkl")
        try:
            with open(history_file, 'wb') as f:
                pickle.dump(self.interaction_history, f)
        except Exception as e:
            logger.error(f"Error saving interaction history: {e}")
    
    def get_adaptive_prompt(self, query: str) -> str:
        """Generate an adaptive prompt using retrieved context and similar interactions"""
        # Retrieve relevant context
        relevant_context = self.retrieve_relevant_context(query)
        
        # Get similar past interactions
        similar_interactions = self.get_similar_interactions(query)
        
        # Build adaptive prompt
        prompt_parts = [
            "You are a data analyst agent with access to the following helpers:",
            "- scrape_table_from_url(url)",
            "- run_duckdb_query(query, files=None)", 
            "- plot_and_encode_base64(fig)",
            "",
            "Relevant context for this type of task:"
        ]
        
        for context in relevant_context:
            prompt_parts.append(f"- {context}")
        
        if similar_interactions:
            prompt_parts.append("\nSimilar successful patterns:")
            for interaction in similar_interactions:
                if interaction.success_score and interaction.success_score > 0.7:
                    prompt_parts.append(f"- Query: {interaction.user_query}")
                    prompt_parts.append(f"- Successful code: {interaction.generated_code[:200]}...")
        
        prompt_parts.extend([
            "",
            f"Task: {query}",
            "",
            "Generate Python code that uses the helpers above. The final answer must be stored in a variable named 'result'.",
            "Return only the code, no explanation."
        ])
        
        return "\n".join(prompt_parts)
    
    def calculate_success_score(self, result: Any, user_feedback: Optional[str] = None) -> float:
        """Calculate a success score based on result and feedback"""
        score = 0.5  # Default neutral score
        
        # Check if result exists and is not None
        if result is not None:
            score += 0.3
        
        # Check if result contains expected data types
        if isinstance(result, (dict, list)) and len(result) > 0:
            score += 0.2
        
        # Adjust based on user feedback
        if user_feedback:
            feedback_lower = user_feedback.lower()
            if any(word in feedback_lower for word in ['good', 'great', 'excellent', 'perfect']):
                score += 0.3
            elif any(word in feedback_lower for word in ['bad', 'wrong', 'error', 'failed']):
                score -= 0.3
        
        return max(0.0, min(1.0, score))  # Clamp between 0 and 1 