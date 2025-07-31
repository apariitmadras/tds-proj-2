# main.py  – RAG Adaptive Data-Analyst API
# ---------------------------------------
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Optional
import uvicorn, logging, concurrent.futures, traceback
from datetime import datetime
import os, json

# LLMs
import openai, google.generativeai as genai

# Data / plots
import pandas as pd, numpy as np, matplotlib.pyplot as plt

# Project helpers
from utils import (
    scrape_table_from_url,
    run_duckdb_query,
    plot_and_encode_base64,
    to_float,                     # NEW – robust numeric converter
)
from rag_system import RAGAdaptiveSystem, Interaction
from config import RAGConfig

# ------------------------------------------------------------------
# FastAPI setup
app = FastAPI()
load_dotenv()

logging.basicConfig(level=getattr(logging, RAGConfig.LOG_LEVEL))
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Configuration & RAG initialisation
if not RAGConfig.validate():
    raise RuntimeError("Invalid configuration – fix environment variables.")

rag_system = RAGAdaptiveSystem(persist_directory=RAGConfig.RAG_PERSIST_DIRECTORY)

# LLM provider
if RAGConfig.LLM_PROVIDER == "openai":
    openai.api_key = RAGConfig.OPENAI_API_KEY
else:  # gemini
    genai.configure(api_key=RAGConfig.GOOGLE_API_KEY)

RAGConfig.print_config()

# ------------------------------------------------------------------
# Pydantic models
class FeedbackRequest(BaseModel):
    interaction_id: str
    feedback: str
    success_score: Optional[float] = None


class ContextRequest(BaseModel):
    content: str
    metadata: Optional[dict] = None


# ------------------------------------------------------------------
# Utility – ensure result is JSON-serialisable
def _make_json_safe(obj: Any) -> Any:
    """Convert Pandas / NumPy / other complex objects into JSON-safe types."""
    import numpy as np, pandas as pd

    if obj is None:
        return None
    try:
        json.dumps(obj)
        return obj  # already serialisable
    except (TypeError, ValueError):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        return str(obj)  # fallback


# ------------------------------------------------------------------
# LLM orchestration
async def call_llm(task: str, use_rag: bool = True) -> str:
    prompt = (
        rag_system.get_adaptive_prompt(task)
        if use_rag
        else f"""You are a data-analyst agent. Generate Python code that may use:
- scrape_table_from_url(url)
- run_duckdb_query(sql, files=None)
- plot_and_encode_base64(fig)
- to_float(series)  # cleans $, commas, footnote letters; returns float

Store the final answer in a variable named 'result' and return ONLY code.
Task:
{task}
"""
    )
    cfg = RAGConfig.get_llm_config()

    if RAGConfig.LLM_PROVIDER == "openai":
        client = openai.OpenAI(api_key=RAGConfig.OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
        )
        return resp.choices[0].message.content
    else:
        return genai.GenerativeModel(cfg["model"]).generate_content(prompt).text


# ------------------------------------------------------------------
# Sandboxed execution
def safe_exec(code: str, timeout: int | None = None):
    allowed = {
        "__builtins__": __builtins__,
        "scrape_table_from_url": scrape_table_from_url,
        "run_duckdb_query": run_duckdb_query,
        "plot_and_encode_base64": plot_and_encode_base64,
        "to_float": to_float,
        "pd": pd,
        "np": np,
        "plt": plt,
    }
    locals_: dict[str, Any] = {}

    def _runner():
        exec(code, allowed, locals_)
        return locals_.get("result")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_runner)
        try:
            return fut.result(timeout or RAGConfig.EXECUTION_TIMEOUT), None
        except Exception:
            return None, traceback.format_exc()


# ------------------------------------------------------------------
@app.post("/api/analyze")
async def analyze(request: Request, file: UploadFile = File(None)):
    task = (await file.read()).decode() if file else (await request.body()).decode()
    iid = f"interaction_{datetime.now():%Y%m%d_%H%M%S}_{abs(hash(task))%10000}"

    # ---- 1) get code from LLM
    code = (await call_llm(task, use_rag=True)).strip()
    if code.startswith("```"):
        import re

        code = re.sub(r"^```[\w]*\n|\n```$", "", code, flags=re.DOTALL).strip()

    # ---- 2) execute
    result, error = safe_exec(code)
    result_safe = _make_json_safe(result)

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
            {"error": "Code execution failed", "details": error, "code": code},
            status_code=500,
        )

    # ---- 3) success path
    score = rag_system.calculate_success_score(result)
    rag_system.add_interaction(
        Interaction(
            timestamp=datetime.now(),
            user_query=task,
            generated_code=code,
            execution_result=result_safe,
            success_score=score,
            context_used=rag_system.retrieve_relevant_context(task),
        )
    )
    return {
        "result": result_safe,
        "interaction_id": iid,
        "code_generated": code,
        "success_score": score,
    }


# ------------------------------------------------------------------
@app.post("/api/feedback")
async def submit_feedback(data: FeedbackRequest):
    for inter in rag_system.interaction_history:
        if getattr(inter, "interaction_id", "") == data.interaction_id:
            inter.user_feedback = data.feedback
            if data.success_score is not None:
                inter.success_score = data.success_score
            if inter.success_score and inter.success_score > 0.7:
                rag_system._learn_from_successful_interaction(inter)
            return {"message": "Feedback stored"}
    raise HTTPException(status_code=404, detail="Interaction not found")


@app.post("/api/context")
async def add_context(req: ContextRequest):
    rag_system.add_contexts([{"content": req.content, "metadata": req.metadata or {}}])
    return {"message": "Context added"}


@app.get("/api/stats")
def stats():
    total = len(rag_system.interaction_history)
    success = len([i for i in rag_system.interaction_history if i.success_score and i.success_score > 0.7])
    avg = sum(i.success_score or 0 for i in rag_system.interaction_history) / max(1, total)
    return {
        "total_interactions": total,
        "successful_interactions": success,
        "success_rate": success / max(1, total),
        "average_success_score": avg,
        "context_count": rag_system.context_collection.count(),
        "system_learning": True,
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "rag_system": "active",
        "llm_provider": RAGConfig.LLM_PROVIDER,
        "version": "1.0.0",
    }


# ------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host=RAGConfig.HOST, port=RAGConfig.PORT)
