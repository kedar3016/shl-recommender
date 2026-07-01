"""
app/config.py
Centralised settings loaded from .env
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    # API Keys
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-flash"
    openrouter_api_key: Optional[str] = None
    openrouter_api_keys: Optional[str] = None
    openrouter_model: str = "openai/gpt-oss-120b:free"
    grok_api_key: Optional[str] = None
    grok_model: str = "grok-2-1212"
    app_url: str = "http://localhost:8000"

    # FAISS / catalog paths
    catalog_path: str = "shl_product_catalog.json"
    faiss_index_path: str = "catalog/faiss.index"
    metadata_path: str = "catalog/metadata.pkl"

    # Retrieval
    retrieval_top_k: int = 15

    # Agent behaviour
    max_turns_before_force_recommend: int = 4  # force recommend after 4 user turns
    llm_timeout_seconds: float = 22.0  # leave headroom under the evaluator's 30s cap

    # ChromaDB Cloud (optional — kept for future use)
    chroma_api_key: Optional[str] = None
    chroma_tenant: Optional[str] = None
    chroma_database: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"   # silently ignore any unknown .env keys


@lru_cache()
def get_settings() -> Settings:
    return Settings()
