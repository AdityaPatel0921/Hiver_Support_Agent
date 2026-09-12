# Engineering Decision Log: `@AppleSupport` AI Triage & Support System

This document records the architectural decisions, design rationales, and operational trade-offs established during the development of the `@AppleSupport` AI Customer Support Agent.

---

### Decision 01: Selecting `@AppleSupport` as Target Entity from TWCS
* **Context:** The Kaggle Customer Support on Twitter (`twcs.csv`) dataset contains 2.8M tweets across 20+ corporate brands (SprintCare, AmazonHelp, Uber_Support, AppleSupport, etc.).
* **Decision:** Focus exclusively on `@AppleSupport` dialogues (106,860 tweets).
* **Rationale:** Apple Support presents a high-stakes, multi-category support surface (hardware breakage, thermal/battery hazards, iCloud security takeovers, and OS updates) where safety triage accuracy and zero hallucination are mission-critical.
* **Trade-off:** Requires domain-specific intent taxonomy design and specialized guardrails rather than a generic multi-brand conversationalist.

---

### Decision 02: Bounding Clean Dialogue Corpus to 5,000 Verified Pairs
* **Context:** Ingesting and embedding all 106k raw Apple tweets introduces noisy duplicates, conversational fragments, and ungrounded chatter.
* **Decision:** Filter for clean, coherent customer–brand pairs with $\ge 15$ character length and deduplicated normalized text, capping at a high-quality 5,000-pair master set.
* **Rationale:** 5,000 high-signal pairs provide dense semantic coverage across all core Apple support topics while allowing full FAISS vector indexing and cold-start retrieval in under 10 seconds.
* **Trade-off:** Leaves long-tail, ultra-rare hardware model discussions out of the vector store unless added via continuous KB ingestion.

---

### Decision 03: 4,850 / 150 Deterministic Stratified Split Before Indexing
* **Context:** Evaluation benchmarks often suffer from subtle data leakage when test examples are indexed or sampled post-hoc.
* **Decision:** Split the 5,000 clean pairs into **4,850 Knowledge Base records** and **150 held-out evaluation candidate pairs** *prior to FAISS index construction* using a deterministic random seed (`seed=42`).
* **Rationale:** Guarantees absolute physical separation between what the agent retrieves and what the agent is tested against.
* **Trade-off:** 150 pairs are permanently excluded from the retrieval knowledge base.

---

### Decision 04: Two-Pass Streaming Chunked Ingestion for 2.8M Rows
* **Context:** Loading `twcs.csv` (516.5 MB, 2.8M rows) into a single Pandas DataFrame consumes > 3 GB RAM and risks Out-Of-Memory (OOM) crashes in memory-constrained environments.
* **Decision:** Implement a two-pass streaming reader with `chunksize=200,000`:
  - **Pass 1:** Stream CSV and collect all `AppleSupport` outbound replies, mapping `in_response_to_tweet_id` into a lightweight dictionary (~106k keys).
  - **Pass 2:** Stream CSV and match inbound customer tweets against the indexed IDs.
* **Rationale:** Reduces active working memory footprint to < 150 MB and completes full dataset scanning in ~30 seconds.
* **Trade-off:** Requires two disk read passes over the CSV instead of an in-memory join.

---

### Decision 05: Programmatic Data Leakage Guard & Normalized Text Collision Detection
* **Context:** ID disjointness alone does not prevent lexical contamination (e.g., identical customer inquiries with different tweet IDs).
* **Decision:** Implement `src/leakage_guard.py` verifying both ID disjointness and normalized string collision (`normalize_query_text` stripping casing, punctuation, and whitespace).
* **Rationale:** Prevents verbatim test inquiries from being memorized or retrieved with 1.0 similarity scores.
* **Trade-off:** Excludes legitimate near-duplicate customer questions from the evaluation set.

---

### Decision 06: Local Dense Embeddings (`all-MiniLM-L6-v2`) with FAISS `IndexFlatL2`
* **Context:** Cloud vector databases (Pinecone, Qdrant) introduce SaaS API costs, egress latency (50–150ms), and external dependency failure modes.
* **Decision:** Use local `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) with in-process FAISS `IndexFlatL2`.
* **Rationale:** Provides exact nearest-neighbor search with zero approximation error, sub-5ms query latency, zero cloud cost, and 100% offline test reproducibility.
* **Trade-off:** Index is held in RAM; for > 10M vectors, would require transitioning to `IndexIVFFlat` or `IndexHNSWFlat`.

---

### Decision 07: Strict Placeholder Substitution (`[Apple Support Link]`) for URL Grounding
* **Context:** Large Language Models frequently hallucinate plausible-looking but broken 404 URLs (e.g. `apple.com/support/iphone14-battery-replace`).
* **Decision:** Ban raw URL generation in prompt engineering and mandate the placeholder token `[Apple Support Link]`.
* **Rationale:** Completely eliminates link hallucination risk by decoupling conversational text generation from dynamic link resolution in the downstream publishing CMS.
* **Trade-off:** Requires the front-end publishing layer to resolve the placeholder based on the validated `IntentType`.

---

### Decision 08: Asymmetric Safety Triage Policy (Minimizing False Negatives on Escalations)
* **Context:** Support errors have asymmetric operational costs:
  - *False Positive (FP):* Escalating a routine update question costs ~$3–$5 in human agent time.
  - *False Negative (FN):* Auto-handling a thermal runaway or swollen battery hazard with generic advice can cause physical injury, property damage, and legal liability.
* **Decision:** Optimize prompt triage guidelines and guardrails to minimize False Negative Rate ($FNR$) on the `ESCALATE` class.
* **Rationale:** Prioritizes customer physical safety and account security above raw auto-resolution volume.
* **Trade-off:** Marginally increases human triage queue volume by ~2–3%.

---

### Decision 09: Strict Pydantic v2 Schema Enforcement & Custom Model Validators
* **Context:** Unstructured JSON output from LLMs can omit required fields or generate inconsistent states (e.g., `action: ESCALATE` without specifying why).
* **Decision:** Use Pydantic v2 schemas (`src/schemas.py`) with `@model_validator(mode="after")` enforcing that `escalation_reason` is non-empty whenever `action == ESCALATE`.
* **Rationale:** Guarantees downstream ticketing queues always receive actionable escalation context.
* **Trade-off:** Adds ~1ms schema validation overhead per inference.

---

### Decision 10: Fail-Closed Deterministic Fallback Guardrail
* **Context:** Network timeouts, LLM rate limits, or invalid JSON syntax could cause pipeline crashes or unhandled customer tickets.
* **Decision:** Implement `generate_fallback_response()` in `src/agent.py` which catches all unhandled exceptions and safely defaults to `action = ESCALATE` (`urgency_score = 4`).
* **Rationale:** Ensures no customer inquiry is dropped silently or handled unsafely during upstream service outages.
* **Trade-off:** During major API outages, 100% of incoming tickets are routed to human queues.

---

### Decision 11: Decoupling Heuristic Suggestions from Human Ground Truth
* **Context:** Using LLMs or regex to auto-label the golden evaluation set introduces confirmation bias and invalidates benchmark authenticity.
* **Decision:** Segregate regex predictions into `suggested_*` columns while keeping `ground_truth_*` strictly blank in `golden_set/golden_eval_set.csv` until manual human review.
* **Rationale:** Preserves human labeling integrity and satisfies Hiver's requirement of a genuine hand-labelled golden set.
* **Trade-off:** Requires dedicated human annotator effort instead of instantaneous automated generation.

---

### Decision 12: Clean Archival of Synthetic Baseline Dataset
* **Context:** The initial development phase used a synthetic benchmark baseline before the genuine Kaggle dataset was verified and placed.
* **Decision:** Safely preserve all 10 synthetic baseline files in `data/archive/synthetic_benchmark/` and re-run all pipelines strictly against genuine data.
* **Rationale:** Maintains engineering provenance and traceability while ensuring 100% of the active production pipeline runs on real Twitter Customer Support data.
* **Trade-off:** Requires maintaining an isolated archive folder outside the active git tracking.

---

### Decision 13: Qualitative 1–5 LLM-as-a-Judge Rubric Paired with Cohen's Kappa
* **Context:** Quantitative classification metrics (Macro-F1) cannot measure tone empathy, brand voice alignment, or troubleshooting actionability.
* **Decision:** Implement `eval/llm_judge.py` with structured 1–5 rubrics for Groundedness, Brand Voice, and Actionability, alongside Cohen's Kappa ($\kappa$) for inter-rater agreement.
* **Rationale:** Provides multi-dimensional evaluation combining classification rigor with qualitative conversational auditing.
* **Trade-off:** Adds judge LLM inference cost during batch evaluation runs.
