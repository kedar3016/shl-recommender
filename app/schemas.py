"""
app/schemas.py
Pydantic models for API request/response.
The response schema is NON-NEGOTIABLE — matches the evaluator contract exactly.
"""
from typing import Literal
from pydantic import BaseModel, field_validator


# ── Catalog URL set (populated at startup by retriever) ─────────────────────
# We store it here so the validator can access it as a module-level set.
_catalog_url_set: set[str] = set()


def register_catalog_urls(urls: set[str]) -> None:
    """Called once at startup by retriever to populate the validation set."""
    global _catalog_url_set
    _catalog_url_set = urls


# ── Keys → test_type letter mapping ─────────────────────────────────────────
KEYS_TO_TYPE: dict[str, str] = {
    "Ability & Aptitude":             "A",
    "Biodata & Situational Judgment": "B",
    "Competencies":                   "C",
    "Development & 360":              "D",
    "Assessment Exercises":           "E",
    "Knowledge & Skills":             "K",
    "Personality & Behavior":         "P",
    "Simulations":                    "S",
}


def keys_to_test_type(keys: list[str]) -> str:
    """['Personality & Behavior', 'Competencies'] → 'P,C'"""
    codes = [KEYS_TO_TYPE[k] for k in keys if k in KEYS_TO_TYPE]
    return ",".join(codes) if codes else "K"


# ── API Request ──────────────────────────────────────────────────────────────
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v: list[Message]) -> list[Message]:
        if not v:
            raise ValueError("messages list cannot be empty")
        if v[-1].role != "user":
            raise ValueError("last message must be from user")
        return v


# ── API Response ─────────────────────────────────────────────────────────────
class RecommendationItem(BaseModel):
    name: str
    url: str
    test_type: str  # single code "K" OR comma-joined "P,C"

    @field_validator("url")
    @classmethod
    def url_must_be_in_catalog(cls, v: str) -> str:
        """Hard eval guard — any hallucinated URL is caught here."""
        if _catalog_url_set and v not in _catalog_url_set:
            raise ValueError(f"URL not in SHL catalog: {v}")
        return v


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[RecommendationItem]  # [] when clarifying/refusing
    end_of_conversation: bool
