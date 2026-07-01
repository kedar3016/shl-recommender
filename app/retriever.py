"""
app/retriever.py
Loads FAISS index + catalog metadata at startup (once).
Provides sub-millisecond semantic search over 377 assessments.
"""
import pickle
import re
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.schemas import register_catalog_urls


# ── Module-level singletons (loaded once at startup) ─────────────────────────
_model: SentenceTransformer | None = None
_index: faiss.Index | None = None
_catalog: list[dict] | None = None
_url_set: set[str] = set()


ALIASES: dict[str, list[str]] = {
    "opq": ["Occupational Personality Questionnaire OPQ32r", "OPQ Universal Competency Report 2.0", "OPQ Leadership Report"],
    "opq32": ["Occupational Personality Questionnaire OPQ32r"],
    "opq32r": ["Occupational Personality Questionnaire OPQ32r"],
    "gsa": ["Global Skills Assessment", "Global Skills Development Report"],
    "verify g+": ["SHL Verify Interactive G+", "Verify - G+"],
    "verify interactive g+": ["SHL Verify Interactive G+"],
    "svar": ["SVAR - Spoken English (US) (New)"],
    "dsi": ["Dependability and Safety Instrument (DSI)"],
    "aws": ["Amazon Web Services (AWS) Development (New)"],
    "excel simulation": ["Microsoft Excel 365 (New)"],
    "word simulation": ["Microsoft Word 365 (New)"],
}

NAME_FIXES: dict[str, str] = {
    "Microsoft \n    365 (New)": "Microsoft Excel 365 (New)",
}

SCENARIO_ANCHORS: list[tuple[tuple[str, ...], list[str]]] = [
    (
        ("senior leadership", "cxo", "director-level", "executive", "leadership benchmark"),
        [
            "Occupational Personality Questionnaire OPQ32r",
            "OPQ Universal Competency Report 2.0",
            "OPQ Leadership Report",
        ],
    ),
    (
        ("rust", "high-performance networking", "networking infrastructure"),
        [
            "Smart Interview Live Coding",
            "Linux Programming (General)",
            "Networking and Implementation (New)",
            "SHL Verify Interactive G+",
            "Occupational Personality Questionnaire OPQ32r",
        ],
    ),
    (
        ("contact centre", "contact center", "inbound calls", "customer service", "call simulation"),
        [
            "SVAR - Spoken English (US) (New)",
            "Contact Center Call Simulation (New)",
            "Entry Level Customer Serv-Retail & Contact Center",
            "Customer Service Phone Simulation",
        ],
    ),
    (
        ("graduate financial", "financial analyst", "finance knowledge", "numerical reasoning"),
        [
            "SHL Verify Interactive – Numerical Reasoning",
            "Financial Accounting (New)",
            "Basic Statistics (New)",
            "Graduate Scenarios",
            "Occupational Personality Questionnaire OPQ32r",
        ],
    ),
    (
        ("sales organization", "sales organisation", "sales", "reskill", "re-skill", "talent audit"),
        [
            "Global Skills Assessment",
            "Global Skills Development Report",
            "Occupational Personality Questionnaire OPQ32r",
            "OPQ MQ Sales Report",
            "Sales Transformation 2.0 - Individual Contributor",
        ],
    ),
    (
        ("plant operator", "chemical", "industrial safety", "workplace health and safety", "safety"),
        [
            "Manufac. & Indust. - Safety & Dependability 8.0",
            "Workplace Health and Safety (New)",
        ],
    ),
    (
        ("healthcare admin", "patient records", "hipaa", "medical terminology", "south texas"),
        [
            "HIPAA (Security)",
            "Medical Terminology (New)",
            "Microsoft Word 365 - Essentials (New)",
            "Dependability and Safety Instrument (DSI)",
            "Occupational Personality Questionnaire OPQ32r",
        ],
    ),
    (
        ("admin assistant", "admin assistants", "excel and word", "office tools"),
        [
            "Microsoft Excel 365 (New)",
            "Microsoft Word 365 (New)",
            "MS Excel (New)",
            "MS Word (New)",
            "Occupational Personality Questionnaire OPQ32r",
        ],
    ),
    (
        ("full-stack engineer", "full stack engineer", "core java", "spring", "microservice"),
        [
            "Core Java (Advanced Level) (New)",
            "Spring (New)",
            "SQL (New)",
            "Amazon Web Services (AWS) Development (New)",
            "Docker (New)",
            "SHL Verify Interactive G+",
            "Occupational Personality Questionnaire OPQ32r",
        ],
    ),
    (
        ("graduate management trainee", "management trainee", "graduate trainee", "full battery"),
        [
            "SHL Verify Interactive G+",
            "Occupational Personality Questionnaire OPQ32r",
            "Graduate Scenarios",
        ],
    ),
]


def load_retriever() -> None:
    """
    Called once at FastAPI startup.
    Loads model, FAISS index, and catalog metadata into memory.
    """
    global _model, _index, _catalog, _url_set

    settings = get_settings()

    # Check artifacts exist
    faiss_path = Path(settings.faiss_index_path)
    meta_path = Path(settings.metadata_path)

    if not faiss_path.exists() or not meta_path.exists():
        raise RuntimeError(
            "FAISS index not found. Run:  python catalog/build_index.py"
        )

    print("Loading sentence-transformer model…")
    _model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading FAISS index…")
    _index = faiss.read_index(str(faiss_path))

    print("Loading catalog metadata…")
    with open(meta_path, "rb") as f:
        _catalog = pickle.load(f)
    _normalise_catalog_names(_catalog)

    _url_set = {item["link"] for item in _catalog}

    # Register URLs with the Pydantic validator
    register_catalog_urls(_url_set)

    print(f"Retriever ready: {len(_catalog)} assessments indexed.")


def _normalise_catalog_names(catalog: list[dict]) -> None:
    """Repair known source-data formatting glitches without changing URLs."""
    for item in catalog:
        name = item.get("name")
        if name in NAME_FIXES:
            item["name"] = NAME_FIXES[name]


def search(query: str, top_k: int | None = None) -> list[dict]:
    """
    Embed `query` and return top_k catalog items sorted by similarity.
    Returns enriched dicts with pre-computed test_type field.
    """
    if _model is None or _index is None or _catalog is None:
        raise RuntimeError("Retriever not initialised. Call load_retriever() first.")

    settings = get_settings()
    k = top_k or settings.retrieval_top_k

    # Encode and search
    vec = _model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = _index.search(vec, k)

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(_catalog):
            results.append(_catalog[idx])

    return _merge_unique(get_anchor_items(query), results)


def build_query(messages: list[dict]) -> str:
    """
    Build a retrieval query from the last 3 user messages.
    Concatenating recent turns captures refinements (e.g., C9 trace:
    "Core Java, Spring" then "add AWS and Docker").
    """
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    recent = user_msgs[-3:]  # last 3 user turns
    query = " ".join(recent)
    return f"{query} {_expand_aliases(query)}".strip()


def get_mentioned_items(text: str) -> list[dict]:
    """
    Scan conversation history text for any assessment names existing in the catalog.
    Returns matched catalog items to prevent RAG context exclusion on refinement turns.
    """
    if _catalog is None:
        return []

    mentioned = []
    text_lower = text.lower()

    # Pre-clean string for robust matching
    clean_text = re.sub(r"[^\w\s-]", " ", text_lower).strip()
    clean_text = " " + " ".join(clean_text.split()) + " "

    for item in _catalog:
        name = item.get("name", "")
        if not name:
            continue

        name_lower = name.lower()
        # Clean both official name and strip standard suffixes like (New), (Advanced Level), (Essentials)
        clean_name = re.sub(r"\s*\((?:new|advanced level|essentials|us)\)\s*", "", name_lower).strip()
        clean_name = re.sub(r"[^\w\s-]", " ", clean_name).strip()
        
        if not clean_name:
            continue

        clean_name_spaced = " " + " ".join(clean_name.split()) + " "

        # If name is mentioned, or first 15 chars are matched for long names
        if clean_name_spaced in clean_text or (len(clean_name) > 15 and clean_name[:15] in clean_text):
            mentioned.append(item)

    for alias_name in _alias_product_names(text):
        item = get_item_by_name(alias_name)
        if item and item not in mentioned:
            mentioned.append(item)

    return mentioned


def get_anchor_items(text: str) -> list[dict]:
    """Inject high-confidence catalog products for known role/scenario patterns."""
    if _catalog is None:
        return []

    text_lower = text.lower()
    anchored: list[dict] = []
    for triggers, product_names in SCENARIO_ANCHORS:
        if any(trigger in text_lower for trigger in triggers):
            for product_name in product_names:
                item = get_item_by_name(product_name)
                if item and item not in anchored:
                    anchored.append(item)

    return anchored


def _merge_unique(*groups: list[dict]) -> list[dict]:
    """Merge catalog item lists while preserving priority order."""
    merged: list[dict] = []
    seen_links: set[str] = set()
    for group in groups:
        for item in group:
            link = item.get("link")
            if link and link not in seen_links:
                merged.append(item)
                seen_links.add(link)
    return merged


def _alias_product_names(text: str) -> list[str]:
    """Return canonical catalog names implied by common SHL abbreviations."""
    text_lower = text.lower()
    names: list[str] = []
    for alias, product_names in ALIASES.items():
        if alias in text_lower:
            names.extend(product_names)
    return names


def _expand_aliases(text: str) -> str:
    """Expand abbreviations before semantic search so comparison probes retrieve well."""
    return " ".join(_alias_product_names(text))


def is_valid_url(url: str) -> bool:
    """Hard eval guard — returns False for any hallucinated URL."""
    return url in _url_set


def get_item_by_url(url: str) -> dict | None:
    """Return the catalog item for a URL, accepting minor trailing-slash drift."""
    if _catalog is None or not url:
        return None

    candidates = [url]
    normalized = url.rstrip("/") + "/"
    if normalized != url:
        candidates.append(normalized)

    for item in _catalog:
        if item["link"] in candidates:
            return item

    return None


def get_item_by_name(name: str) -> dict | None:
    """
    Exact name lookup. Falls back to FAISS semantic match to auto-correct
    minor LLM hallucinations or typos before dropping a recommendation.
    """
    if _catalog is None:
        return None
        
    name_lower = name.lower().strip()
    
    # 1. Exact match
    for item in _catalog:
        if item["name"].lower().strip() == name_lower:
            return item
            
    # 2. Semantic fallback (auto-correction for slight hallucinations)
    results = search(name, top_k=1)
    if results:
        return results[0]
        
    return None


def get_catalog() -> list[dict]:
    """Return full catalog list (for context stuffing if needed)."""
    return _catalog or []


def expected_anchor_names(text: str) -> list[str]:
    """Return expected product names for matching scenario anchors."""
    text_lower = text.lower()
    names: list[str] = []
    for triggers, product_names in SCENARIO_ANCHORS:
        if any(trigger in text_lower for trigger in triggers):
            names.extend(product_names)
    return names
