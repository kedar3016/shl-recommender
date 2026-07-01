"""
catalog/build_index.py
Run ONCE locally to build the FAISS index from the SHL catalog JSON.

    python catalog/build_index.py

Outputs:
    catalog/faiss.index   ← FAISS flat L2 index (binary)
    catalog/metadata.pkl  ← list of catalog dicts with pre-computed test_type
"""
import json
import os
import pickle
import sys

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATALOG_JSON = os.path.join(ROOT, "shl_product_catalog.json")
OUT_DIR = os.path.join(ROOT, "catalog")
FAISS_PATH = os.path.join(OUT_DIR, "faiss.index")
META_PATH = os.path.join(OUT_DIR, "metadata.pkl")

# ── key → single-letter mapping ─────────────────────────────────────────────
KEYS_TO_TYPE = {
    "Ability & Aptitude":             "A",
    "Biodata & Situational Judgment": "B",
    "Competencies":                   "C",
    "Development & 360":              "D",
    "Assessment Exercises":           "E",
    "Knowledge & Skills":             "K",
    "Personality & Behavior":         "P",
    "Simulations":                    "S",
}

NAME_FIXES = {
    "Microsoft \n    365 (New)": "Microsoft Excel 365 (New)",
}


def keys_to_test_type(keys: list[str]) -> str:
    codes = [KEYS_TO_TYPE[k] for k in keys if k in KEYS_TO_TYPE]
    return ",".join(codes) if codes else "K"


def build_text(item: dict) -> str:
    """
    Pack all searchable signals into one string for embedding.
    Ordering matters — put the most discriminative signals first.
    """
    name = item.get("name", "")
    desc = item.get("description", "")
    keys = ", ".join(item.get("keys", []))
    levels = ", ".join(item.get("job_levels", []))
    langs = ", ".join(item.get("languages", [])[:5])   # first 5 languages
    duration = item.get("duration", "")
    adaptive = "adaptive" if item.get("adaptive") == "yes" else ""

    parts = [
        f"Assessment: {name}.",
        f"Type: {keys}.",
        f"Description: {desc}",
        f"Job levels: {levels}.",
        f"Languages: {langs}.",
    ]
    if duration:
        parts.append(f"Duration: {duration}.")
    if adaptive:
        parts.append("Adaptive/IRT scored.")

    return " ".join(parts)


def main() -> None:
    print("Loading catalog…")
    with open(CATALOG_JSON, encoding="utf-8") as f:
        raw_catalog: list[dict] = json.loads(f.read(), strict=False)

    print(f"  {len(raw_catalog)} items loaded.")

    # Enrich each item with pre-computed test_type
    for item in raw_catalog:
        if item.get("name") in NAME_FIXES:
            item["name"] = NAME_FIXES[item["name"]]
        item["test_type"] = keys_to_test_type(item.get("keys", []))

    # Build embedding texts
    texts = [build_text(item) for item in raw_catalog]

    print("Loading sentence-transformer model (all-MiniLM-L6-v2)…")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Embedding 377 documents…")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    print(f"  Embedding shape: {embeddings.shape}")

    # Build FAISS flat L2 index (exact search, fine for 377 items)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"  FAISS index built: {index.ntotal} vectors @ dim={dim}")

    # Save artifacts
    os.makedirs(OUT_DIR, exist_ok=True)
    faiss.write_index(index, FAISS_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(raw_catalog, f)

    print(f"\nDone! Saved:")
    print(f"   {FAISS_PATH}  ({os.path.getsize(FAISS_PATH):,} bytes)")
    print(f"   {META_PATH}  ({os.path.getsize(META_PATH):,} bytes)")
    print("\nAll done. Commit catalog/faiss.index and catalog/metadata.pkl.")


if __name__ == "__main__":
    main()
