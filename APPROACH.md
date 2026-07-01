# SHL Assessment Recommender - Approach

I built a stateless FastAPI service with the required `GET /health` and `POST /chat` endpoints. Each `/chat` call carries the full conversation history, and the service reconstructs temporary state from that payload only. No per-conversation state is stored.

The runtime is a simple three-step agent pipeline:

```text
retrieve -> generate -> validate
```

I kept this as a lightweight LangGraph-inspired flow instead of full LangGraph execution because the workflow is linear, stateless, and deployment reliability matters for the evaluator.

## Stack and Retrieval

I used FastAPI and Pydantic for API/schema enforcement, FAISS with `sentence-transformers/all-MiniLM-L6-v2` for semantic search, and Gemini 2.5 Flash via OpenRouter for conversational reasoning. The SHL catalog is preprocessed into searchable text using product name, description, assessment keys, job levels, languages, duration, and metadata. The 377 catalog items are embedded once and stored in a FAISS index.

At runtime, retrieval uses recent user turns plus conversation context. I added alias expansion for common SHL abbreviations such as OPQ, GSA, DSI, SVAR, AWS, and Verify G+. I also scan previous assistant/user messages for already-mentioned assessments so refinement requests like "drop OPQ" preserve the rest of the active shortlist. High-confidence scenario anchors boost likely products for known role patterns, while FAISS remains the general retrieval layer for unseen roles.

## Prompt and Validation

The prompt limits the assistant to SHL Individual Test assessment selection and supports five behaviors: clarify, recommend, refine, compare, and refuse. It asks one targeted question for vague requests, recommends 1-10 assessments when enough context is available, edits shortlists during refinements, compares only from retrieved catalog data, and refuses off-topic HR advice, legal questions, and prompt-injection attempts.

The LLM output is never trusted directly. A deterministic validator parses JSON, resolves every recommendation back to the catalog, corrects official names/URLs/test types, drops non-catalog items, caps recommendations at 10, and returns a schema-compliant fallback if the LLM times out or emits invalid JSON. The LLM timeout is below 30 seconds so the API can still respond within the evaluator limit.

## Evaluation and Iteration

I built local evaluation scripts for the 10 public traces and separate hidden-style checks that do not reuse those scripts. They test schema compliance, exact catalog URL membership, turn cap, Recall@10, off-topic refusal, vague-query clarification, refinement, comparison, and prompt-injection handling.

Current local results:

- Public traces: **100.0% Mean Recall@10**
- Hard checks: **0 schema errors, 0 hallucinated URLs, 0 turn-cap violations**
- Behavior probes: **7/7 passed**
- Hidden-style checks: **11/11 passed**

What did not work initially: pure semantic retrieval missed products during refinement turns, and prefix-only URL checks were too weak. I fixed these with history scanning, exact catalog validation, alias expansion, and targeted scenario anchors. I used AI coding assistance for code review, implementation, and evaluation-script generation, while manually reviewing catalog fields, failure cases, and final results.
