"""
Smoke-test the deployed SHL recommender API.

Usage:
    python smoke_deployed.py

Optional direct OpenRouter check:
    set OPENROUTER_API_KEY=sk-or-v1-...
    python smoke_deployed.py --openrouter
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = os.getenv(
    "BASE_URL",
    "https://adityasuhane01-shl-assessment-recommender.hf.space",
).rstrip("/")


def request_json(url: str, payload: dict | None = None, headers: dict | None = None) -> tuple[int, dict]:
    data = None
    final_headers = headers or {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        final_headers = {"Content-Type": "application/json", **final_headers}

    req = Request(url, data=data, headers=final_headers, method="POST" if payload is not None else "GET")
    with urlopen(req, timeout=35) as resp:
        raw = resp.read().decode("utf-8")
        return resp.status, json.loads(raw)


def print_json(title: str, value: dict) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(value, indent=2, ensure_ascii=False))


def test_health() -> bool:
    try:
        status, body = request_json(f"{BASE_URL}/health")
    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"HEALTH FAIL: {exc}")
        return False

    print_json("Health", body)
    return status == 200 and body == {"status": "ok"}


def test_chat() -> bool:
    payload = {
        "messages": [
            {
                "role": "user",
                "content": "I am hiring a Java developer who works with stakeholders. Mid-level, around 4 years.",
            }
        ]
    }

    try:
        status, body = request_json(f"{BASE_URL}/chat", payload)
    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"CHAT FAIL: {exc}")
        return False

    print_json("Chat", body)

    required_keys = {"reply", "recommendations", "end_of_conversation"}
    schema_ok = status == 200 and required_keys.issubset(body)
    fallback_used = "trouble reaching the language model" in body.get("reply", "").lower()

    if fallback_used:
        print("\nNOTE: API schema works, but deployed app used retrieval fallback instead of the LLM.")
        print("Restart the Space after setting OPENROUTER_API_KEY, then rerun this script.")

    return schema_ok and not fallback_used


def test_openrouter() -> bool:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("\nOPENROUTER SKIP: OPENROUTER_API_KEY is not set in this terminal.")
        return False

    payload = {
        "model": "google/gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Say ok"}],
        "max_tokens": 20,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": BASE_URL,
        "X-Title": "SHL Agent",
    }

    try:
        status, body = request_json("https://openrouter.ai/api/v1/chat/completions", payload, headers)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"\nOPENROUTER FAIL: HTTP {exc.code}")
        print(detail)
        return False
    except (URLError, TimeoutError) as exc:
        print(f"\nOPENROUTER FAIL: {exc}")
        return False

    print_json("OpenRouter", body)
    return status == 200 and bool(body.get("choices"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openrouter", action="store_true", help="also test the OpenRouter key in this terminal")
    args = parser.parse_args()

    print(f"Testing deployed API: {BASE_URL}")
    health_ok = test_health()
    chat_ok = test_chat()

    openrouter_ok = True
    if args.openrouter:
        openrouter_ok = test_openrouter()

    print("\n=== Summary ===")
    print(f"health: {'PASS' if health_ok else 'FAIL'}")
    print(f"chat with LLM: {'PASS' if chat_ok else 'FAIL'}")
    if args.openrouter:
        print(f"direct OpenRouter: {'PASS' if openrouter_ok else 'FAIL'}")

    return 0 if health_ok and chat_ok and openrouter_ok else 1


if __name__ == "__main__":
    sys.exit(main())
