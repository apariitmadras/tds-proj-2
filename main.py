from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Any
import uvicorn
import io
import json
import os
import openai
import google.generativeai as genai
from utils import scrape_table_from_url, run_duckdb_query, plot_and_encode_base64
from rag_system import RAGAdaptiveSystem, Interaction
from config import RAGConfig
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import traceback
from datetime import datetime
import logging

app = FastAPI()

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=getattr(logging, RAGConfig.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Validate configuration
if not RAGConfig.validate():
    raise RuntimeError("Invalid configuration. Please check your environment variables.")

# Initialize RAG Adaptive System
rag_system = RAGAdaptiveSystem(persist_directory=RAGConfig.RAG_PERSIST_DIRECTORY)

# Configure LLM providers
if RAGConfig.LLM_PROVIDER == "openai":
    if not RAGConfig.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY environment variable is required for OpenAI provider.")
    openai.api_key = RAGConfig.OPENAI_API_KEY
elif RAGConfig.LLM_PROVIDER == "gemini":
    if not RAGConfig.GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY environment variable is required for Gemini provider.")
    genai.configure(api_key=RAGConfig.GOOGLE_API_KEY)
else:
    raise ValueError(f"Unsupported LLM_PROVIDER: {RAGConfig.LLM_PROVIDER}")

# Print configuration on startup
RAGConfig.print_config()

# Pydantic models for API
class FeedbackRequest(BaseModel):
    interaction_id: str
    feedback: str
    success_score: Optional[float] = None

class ContextRequest(BaseModel):
    content: str
    metadata: Optional[dict] = None

class AnalysisResponse(BaseModel):
    result: Any
    interaction_id: str
    code_generated: str
    context_used: list
    success_score: float

async def call_llm(task: str, use_rag: bool = True):
    """
    Use the selected LLM (OpenAI or Gemini) to generate Python code for the given task.
    Now supports RAG-enhanced prompts.
    """
    if use_rag:
        prompt = rag_system.get_adaptive_prompt(task)
    else:
        prompt = f"""
You are a data analyst agent. Given the following task, generate Python code that uses the following helpers if needed:
- scrape_table_from_url(url)
- run_duckdb_query(query, files=None)
- plot_and_encode_base64(fig)

The final answer must be stored in a variable named 'result'.

Task:
{task}

Return only the code, no explanation.
"""
    
    llm_config = RAGConfig.get_llm_config()
    
    if RAGConfig.LLM_PROVIDER == "openai":
        client = openai.OpenAI(api_key=RAGConfig.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=llm_config["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=llm_config["temperature"],
            max_tokens=llm_config["max_tokens"]
        )
        return response.choices[0].message.content
    elif RAGConfig.LLM_PROVIDER == "gemini":
        model = genai.GenerativeModel(llm_config["model"])
        response = model.generate_content(prompt)
        return response.text
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {RAGConfig.LLM_PROVIDER}")

def safe_exec(code, timeout=None):
    """
    Safely execute code with access to helpers and common libraries. Capture 'result'. Timeout after N seconds.
    """
    if timeout is None:
        timeout = RAGConfig.EXECUTION_TIMEOUT
        
    allowed_globals = {
        '__builtins__': __builtins__,
        'scrape_table_from_url': scrape_table_from_url,
        'run_duckdb_query': run_duckdb_query,
        'plot_and_encode_base64': plot_and_encode_base64,
        'pd': pd,
        'np': np,
        'plt': plt,
    }
    local_vars = {}
    def runner():
        exec(code, allowed_globals, local_vars)
        return local_vars.get('result', None)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(runner)
        try:
            return future.result(timeout=timeout), None
        except Exception as e:
            return None, traceback.format_exc()

@app.post("/api/analyze")
async def analyze(request: Request, file: UploadFile = File(None)):
    """
    Main analysis endpoint with RAG enhancement
    """
    # Only read the body or file ONCE to avoid 'Stream consumed' error
    if file is not None:
        task = (await file.read()).decode("utf-8")
    else:
        task = (await request.body()).decode("utf-8")

    # Generate interaction ID
    interaction_id = f"interaction_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(task) % 10000}"

    try:
        # 1. RAG-Enhanced LLM Orchestration: get code from LLM with context
        code = await call_llm(task, use_rag=True)

        import re
        if code.strip().startswith("```"):
        code = re.sub(r"^```[\w]*\n|\n```$", "", code.strip(), flags=re.DOTALL).strip()

        
        # 2. Safe code execution
        result, error = safe_exec(code)
        
        if error:
            # Create failed interaction
            interaction = Interaction(
                timestamp=datetime.now(),
                user_query=task,
                generated_code=code,
                execution_result=None,
                success_score=0.0,
                context_used=rag_system.retrieve_relevant_context(task)
            )
            rag_system.add_interaction(interaction)
            
            return JSONResponse(
                content={
                    "error": "Code execution failed", 
                    "details": error, 
                    "code": code,
                    "interaction_id": interaction_id
                }, 
                status_code=500
            )
        
        # 3. Calculate initial success score
        success_score = rag_system.calculate_success_score(result)
        
        # 4. Create and store interaction
        interaction = Interaction(
            timestamp=datetime.now(),
            user_query=task,
            generated_code=code,
            execution_result=result,
            success_score=success_score,
            context_used=rag_system.retrieve_relevant_context(task)
        )
        rag_system.add_interaction(interaction)
        
        logger.info(f"Analysis completed successfully. Success score: {success_score}")
        
        return JSONResponse(content={
            "result": result,
            "interaction_id": interaction_id,
            "code_generated": code,
            "context_used": interaction.context_used,
            "success_score": success_score
        })
        
    except Exception as e:
        return JSONResponse(
            content={"error": f"Analysis failed: {str(e)}", "interaction_id": interaction_id}, 
            status_code=500
        )

@app.post("/api/feedback")
async def submit_feedback(feedback_request: FeedbackRequest):
    """
    Submit feedback for a previous interaction to improve the system
    """
    try:
        # Find the interaction and update it with feedback
        for interaction in rag_system.interaction_history:
            if hasattr(interaction, 'interaction_id') and interaction.interaction_id == feedback_request.interaction_id:
                interaction.user_feedback = feedback_request.feedback
                if feedback_request.success_score is not None:
                    interaction.success_score = feedback_request.success_score
                
                # Re-learn from this interaction
                if interaction.success_score and interaction.success_score > 0.7:
                    rag_system._learn_from_successful_interaction(interaction)
                
                return JSONResponse(content={"message": "Feedback submitted successfully"})
        
        raise HTTPException(status_code=404, detail="Interaction not found")
        
    except Exception as e:
        return JSONResponse(
            content={"error": f"Failed to submit feedback: {str(e)}"}, 
            status_code=500
        )

@app.post("/api/context")
async def add_context(context_request: ContextRequest):
    """
    Add new context to the knowledge base
    """
    try:
        rag_system.add_contexts([{
            "content": context_request.content,
            "metadata": context_request.metadata or {}
        }])
        return JSONResponse(content={"message": "Context added successfully"})
    except Exception as e:
        return JSONResponse(
            content={"error": f"Failed to add context: {str(e)}"}, 
            status_code=500
        )

@app.get("/api/stats")
async def get_system_stats():
    """
    Get system statistics and learning metrics
    """
    try:
        total_interactions = len(rag_system.interaction_history)
        successful_interactions = len([
            i for i in rag_system.interaction_history 
            if i.success_score and i.success_score > 0.7
        ])
        avg_success_score = sum([
            i.success_score for i in rag_system.interaction_history 
            if i.success_score is not None
        ]) / max(1, total_interactions)
        
        context_count = rag_system.context_collection.count()
        
        return JSONResponse(content={
            "total_interactions": total_interactions,
            "successful_interactions": successful_interactions,
            "success_rate": successful_interactions / max(1, total_interactions),
            "average_success_score": avg_success_score,
            "context_count": context_count,
            "system_learning": True
        })
    except Exception as e:
        return JSONResponse(
            content={"error": f"Failed to get stats: {str(e)}"}, 
            status_code=500
        )

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint
    """
    return JSONResponse(content={
        "status": "healthy",
        "rag_system": "active",
        "llm_provider": RAGConfig.LLM_PROVIDER,
        "version": "1.0.0"
    })

if __name__ == "__main__":
    uvicorn.run(app, host=RAGConfig.HOST, port=RAGConfig.PORT)
