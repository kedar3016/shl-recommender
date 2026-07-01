# SHL Assessment Recommender - My Approach

When starting this assignment, my priority was to build a simple, reliable, and completely stateless FastAPI backend. Since the evaluator passes the entire conversation history in each POST request and stores no state on the server, I chose to implement a straightforward, linear pipeline: Retrieve -> Generate -> Validate. I decided against using heavy orchestration frameworks like LangGraph to keep the memory footprint small and reduce the risk of deployment issues on free-tier containers.

Instead, I reconstruct the active shortlist on the fly by parsing the conversation history dynamically on each incoming turn. 

---

## 🛠️ My Stack and Retrieval Design

For semantic search, I chose `faiss-cpu` combined with `sentence-transformers/all-MiniLM-L6-v2` to embed the 377 individual test solutions in the SHL catalog. During index creation, I combined each product's official name, description, categories, job levels, languages, and duration into a unified text string to ensure our embeddings captured both domain and role relevance.

At runtime, querying the vector database with just the latest user message is often not enough—especially during refinement turns where the user asks to "drop" or "add" specific assessments. To solve this, I built a custom history-scanner. This scanner reads the raw text of previous turns to identify which assessments are currently active in the shortlist and extracts common SHL abbreviations (like OPQ, GSA, SVAR, and Verify G+). 

If a user requests to remove a test, a regex-based proximity filter checks if the removal verb refers specifically to that assessment and excludes it. For common roles (such as entry-level trainees or plant operators), I implemented a soft-boosting layer that prioritizes high-confidence templates before passing candidates to the LLM.

---

## 🧠 LLM and Grounded Validation

I integrated xAI's `grok-2` (and set up Google's `gemini-2.5-flash` as a fallback) to handle the conversational reasoning. However, since LLMs are prone to formatting errors and link hallucinations, I treat all LLM outputs as untrusted.

I built a strict, deterministic validator in Python that post-processes the response. If the LLM generates a slightly wrong URL or returns an assessment name with a typo, the validator cross-references the item with our catalog database, corrects the official name/URL/test type, and filters out any hallucinated suggestions. 

To respect the evaluator's 30-second limit, I set an internal LLM call timeout of 22 seconds. If the LLM times out or returns bad JSON, the backend automatically triggers a deterministic fallback that generates recommendations using our local vector search and keyword matching. This ensures the API always returns a schema-compliant response within the limit.

---

## 🧪 How I Tested and Iterated

I wrote two local test suites to measure and guide my changes:
1. A replay harness (`evaluate.py`) that runs the 10 public conversation traces to verify Recall@10.
2. A behavior probe script (`eval2.py` / `holdout_eval.py`) containing 11 distinct multi-turn checks (verifying vague query handling, off-topic questions, injection attempts, comparisons, and refinements).

### What didn't work initially:
In my early prototypes, pure semantic search was too sensitive. If a user said "remove JavaScript," the word "JavaScript" would dominate the query embedding, and the FAISS retriever would keep returning JavaScript assessments at the top of the RAG context. Adding keyword extraction and the proximity-based removal guard completely solved this, bringing my local test pass rate to **100% Mean Recall@10** and **11/11 behavior checks**.

### AI Tooling:
I used AI assistance (Antigravity) for writing python templates, compiling the initial indexing code, and generating test scripts. However, I manually reviewed and adjusted all catalog keyword mappings, debugged the OOM memory limits on CPU containers, and audited the edge cases to ensure the final solution is completely robust.
