from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Any
import uvicorn, io, json, os, concurrent.futures, traceback, logging
from datetime import datetime

import openai
import google.generativeai as genai

# ⬇️  utils now also exports to_float
from utils import (
    scrape_table_from_url,
    run_duckdb_query,
    plot_and_encode_base64,
    to_float,                       # NEW
)

from rag_system import RAGAdaptiveSystem, Interaction
from config import RAGConfig

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
app = FastAPI()
load_dotenv()

logging.basicConfig(level=getattr(logging, RAGConfig.LOG_LEVEL))
logger = logging.getLogger(__name__)

if not RAGConfig.validate():
    raise RuntimeError("Invalid configuration. Check environment variables.")

rag_system = RAGAdaptiveSystem(persist_directory=RAGConfig.RAG_PERSIST_DIRECTORY)

if RAGConfig.LLM_PROVIDER == "openai":
    if not RAGConfig.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI provider.")
    openai.api_key = RAGConfig.OPENAI_API_KEY
elif RAGConfig.LLM_PROVIDER == "gemini":
    if not RAGConfig.GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is required for Gemini provider.")
    genai.configure(api_key=RAGConfig.GOOGLE_API_KEY)
else:
    raise ValueError(f"Unsupported LLM_PROVIDER: {RAGConfig.LLM_PROVIDER}")

RAGConfig.print_config()

# ------------------ Pydantic models -------------------------
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

# ------------------ LLM orchestrator ------------------------
async def call_llm(task: str, use_rag: bool = True):
    if use_rag:
        prompt = rag_system.get_adaptive_prompt(task)
    else:
        prompt = f"""
You are a data-analyst agent. Generate Python code that may use:
- scrape_table_from_url(url)
- run_duckdb_query(sql, files=None)
- plot_and_encode_base64(fig)
- to_float(series)              # helper to clean $ and commas → float

Store the final answer in a variable named 'result' and return only code.
Task:
{task}
"""

    llm_cfg = RAGConfig.get_llm_config()

    if RAGConfig.LLM_PROVIDER == "openai":
        client = openai.OpenAI(api_key=RAGConfig.OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=llm_cfg["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=llm_cfg["temperature"],
            max_tokens=llm_cfg["max_tokens"],
        )
        return resp.choices[0].message.content
    else:  # gemini
        model = genai.GenerativeModel(llm_cfg["model"])
        return model.generate_content(prompt).text

# ------------------ Sandboxed executor ----------------------
def safe_exec(code: str, timeout: int | None = None):
    if timeout is None:
        timeout = RAGConfig.EXECUTION_TIMEOUT

    allowed_globals = {
        "__builtins__": __builtins__,
        "scrape_table_from_url": scrape_table_from_url,
        "run_duckdb_query": run_duckdb_query,
        "plot_and_encode_base64": plot_and_encode_base64,
        "to_float": to_float,           # NEW
        "pd": pd,
        "np": np,
        "plt": plt,
    }
    local_vars: dict[str, Any] = {}

    def runner():
        exec(code, allowed_globals, local_vars)
        return local_vars.get("result")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(runner)
        try:
            return fut.result(timeout=timeout), None
        except Exception:
            return None, traceback.format_exc()

# ------------------ /api/analyze ----------------------------
@app.post("/api/analyze")
async def analyze(request: Request, file: UploadFile = File(None)):
    task = (
        (await file.read()).decode() if file else (await request.body()).decode()
    )
    interaction_id = (
        f"interaction_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(task)%10000}"
    )

    try:
        code = await call_llm(task, use_rag=True)

        # strip ``` fences if present
        if code.strip().startswith("```"):
            import re

            code = re.sub(r"^```[\w]*\n|\n```$", "", code.strip(), flags=re.DOTALL).strip()

        result, error = safe_exec(code)

        if error:
            rag_system.add_interaction(
                Interaction(
                    timestamp=datetime.now(),
                    user_query=task,
                    generated_code=code,
                    execution_result=None,
                    success_score=0.0,
                    context_used=rag_system.retrieve_relevant_context(task),
                )
            )
            return JSONResponse(
                {
                    "error": "Code execution failed",
                    "details": error,
                    "code": code,
                    "interaction_id": interaction_id,
                },
                status_code=500,
            )

        success_score = rag_system.calculate_success_score(result)
        interaction = Interaction(
            timestamp=datetime.now(),
            user_query=task,
            generated_code=code,
            execution_result=result,
            success_score=success_score,
            context_used=rag_system.retrieve_relevant_context(task),
        )
        rag_system.add_interaction(interaction)

        return JSONResponse(
            {
                "result": result,
                "interaction_id": interaction_id,
                "code_generated": code,
                "context_used": interaction.context_used,
                "success_score": success_score,
            }
        )

    except Exception as exc:
        return JSONResponse(
            {"error": f"Analysis failed: {exc}", "interaction_id": interaction_id},
            status_code=500,
        )

# ------------------ Remaining endpoints (unchanged) --------
@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    for inter in rag_system.interaction_history:
        if getattr(inter, "interaction_id", "") == req.interaction_id:
            inter.user_feedback = req.feedback
            if req.success_score is not None:
                inter.success_score = req.success_score
            if inter.success_score and inter.success_score > 0.7:
                rag_system._learn_from_successful_interaction(inter)
            return {"message": "Feedback submitted successfully"}
    raise HTTPException(status_code=404, detail="Interaction not found")

@app.post("/api/context")
async def add_context(req: ContextRequest):
    rag_system.add_contexts([{"content": req.content, "metadata": req.metadata or {}}])
    return {"message": "Context added successfully"}

@app.get("/api/stats")
async def get_system_stats():
    total = len(rag_system.interaction_history)
    successful = len([i for i in rag_system.interaction_history if i.success_score and i.success_score > 0.7])
    avg = sum(i.success_score or 0 for i in rag_system.interaction_history) / max(1, total)
    return {
        "total_interactions": total,
        "successful_interactions": successful,
        "success_rate": successful / max(1, total),
        "average_success_score": avg,
        "context_count": rag_system.context_collection.count(),
        "system_learning": True,
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "rag_system": "active",
        "llm_provider": RAGConfig.LLM_PROVIDER,
        "version": "1.0.0",
    }

if __name__ == "__main__":
    uvicorn.run(app, host=RAGConfig.HOST, port=RAGConfig.PORT)
