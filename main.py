from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Any
import uvicorn, concurrent.futures, traceback, logging, os
from datetime import datetime

import openai, google.generativeai as genai
import pandas as pd, numpy as np, matplotlib.pyplot as plt

# ⬇️ utils now provides to_float
from utils import (
    scrape_table_from_url,
    run_duckdb_query,
    plot_and_encode_base64,
    to_float,                # NEW
)

from rag_system import RAGAdaptiveSystem, Interaction
from config import RAGConfig

# ------------------------------------------------------------------
app = FastAPI()
load_dotenv()

logging.basicConfig(level=getattr(logging, RAGConfig.LOG_LEVEL))
logger = logging.getLogger(__name__)

if not RAGConfig.validate():
    raise RuntimeError("Bad configuration – check env variables.")

rag_system = RAGAdaptiveSystem(persist_directory=RAGConfig.RAG_PERSIST_DIRECTORY)

# LLM provider init
if RAGConfig.LLM_PROVIDER == "openai":
    openai.api_key = RAGConfig.OPENAI_API_KEY
else:  # gemini
    genai.configure(api_key=RAGConfig.GOOGLE_API_KEY)

# ------------------------------------------------------------------
class FeedbackRequest(BaseModel):
    interaction_id: str
    feedback: str
    success_score: Optional[float] = None

class ContextRequest(BaseModel):
    content: str
    metadata: Optional[dict] = None

# ------------------------------------------------------------------
async def call_llm(task: str, use_rag: bool = True):
    prompt = (
        rag_system.get_adaptive_prompt(task)
        if use_rag
        else f"""You are a data-analyst agent. Generate Python code that may use:
- scrape_table_from_url(url)
- run_duckdb_query(sql, files=None)
- plot_and_encode_base64(fig)
- to_float(series)  # cleans $, commas, footnote letters → float

Store the final answer in a variable named 'result' and return ONLY code.
Task:
{task}
"""
    )
    cfg = RAGConfig.get_llm_config()

    if RAGConfig.LLM_PROVIDER == "openai":
        client = openai.OpenAI(api_key=RAGConfig.OPENAI_API_KEY)
        resp   = client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg["temperature"],
            max_tokens=cfg["max_tokens"],
        )
        return resp.choices[0].message.content
    else:
        return genai.GenerativeModel(cfg["model"]).generate_content(prompt).text

# ------------------------------------------------------------------
def safe_exec(code: str, timeout: int | None = None):
    allowed_globals = {
        "__builtins__": __builtins__,
        "scrape_table_from_url": scrape_table_from_url,
        "run_duckdb_query": run_duckdb_query,
        "plot_and_encode_base64": plot_and_encode_base64,
        "to_float": to_float,        # NEW
        "pd": pd,
        "np": np,
        "plt": plt,
    }
    local_vars = {}

    def runner():
        exec(code, allowed_globals, local_vars)
        return local_vars.get("result")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(runner)
        try:
            return fut.result(timeout=timeout or RAGConfig.EXECUTION_TIMEOUT), None
        except Exception:
            return None, traceback.format_exc()

# ------------------------------------------------------------------
@app.post("/api/analyze")
async def analyze(request: Request, file: UploadFile = File(None)):
    task = (await file.read()).decode() if file else (await request.body()).decode()
    interaction_id = f"interaction_{datetime.now():%Y%m%d_%H%M%S}_{abs(hash(task))%10000}"

    # get code from LLM
    code = await call_llm(task, use_rag=True).strip()
    if code.startswith("```"):
        import re
        code = re.sub(r"^```[\w]*\n|\n```$", "", code, flags=re.DOTALL).strip()

    # execute
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
            {"error": "Code execution failed", "details": error, "code": code},
            status_code=500,
        )

    success = rag_system.calculate_success_score(result)
    rag_system.add_interaction(
        Interaction(
            timestamp=datetime.now(),
            user_query=task,
            generated_code=code,
            execution_result=result,
            success_score=success,
            context_used=rag_system.retrieve_relevant_context(task),
        )
    )
    return {
        "result": result,
        "interaction_id": interaction_id,
        "code_generated": code,
        "success_score": success,
    }

# ------------------------ health & misc ----------------------
@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "rag_system": "active",
        "llm_provider": RAGConfig.LLM_PROVIDER,
        "version": "1.0.0",
    }

# (feedback / context / stats endpoints unchanged – keep as in current file)

if __name__ == "__main__":
    uvicorn.run(app, host=RAGConfig.HOST, port=RAGConfig.PORT)
