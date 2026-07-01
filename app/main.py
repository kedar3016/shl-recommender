"""
app/main.py
FastAPI application — exposes GET /health and POST /chat.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app import retriever
from app.agent import run_agent
from app.schemas import ChatRequest, ChatResponse, RecommendationItem


# ── Startup / Shutdown ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load FAISS index and catalog into memory once at startup."""
    retriever.load_retriever()
    yield
    # (nothing to clean up)


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for recommending SHL Individual Test assessments.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health() -> dict:
    """Readiness check — returns 200 immediately once startup completes."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Stateless chat endpoint.
    Receives full conversation history, returns next agent reply + recommendations.
    """
    t0 = time.perf_counter()

    # Convert Pydantic messages to plain dicts for the agent
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    try:
        result = await run_agent(messages)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = time.perf_counter() - t0

    # Build validated recommendation objects
    recs = []
    for r in result.get("recommendations", []):
        try:
            recs.append(
                RecommendationItem(
                    name=r["name"],
                    url=r["url"],
                    test_type=r["test_type"],
                )
            )
        except Exception:
            # Skip any item that fails Pydantic validation (URL not in catalog)
            continue

    return ChatResponse(
        reply=result["reply"],
        recommendations=recs,
        end_of_conversation=result["end_of_conversation"],
    )
