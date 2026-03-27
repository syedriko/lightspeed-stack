|                          |                                                                       |
|--------------------------|-----------------------------------------------------------------------|
| **Date**                 | 2026-03-25                                                            |
| **Component**            | lightspeed-stack (RAG / BYOK / OKP)                                   |
| **Authors**              | Sergey Yedrikov                                                       |
| **Status**               | Draft — high-level design                                             |
| **Related docs**         | [rag_guide.md](../rag_guide.md), [byok_guide.md](../byok_guide.md)    |
| **Feature proposals**    | [rag-roadmap-feature-proposals.md](rag-roadmap-feature-proposals.md)  |
| **External**             | [rag-content](https://github.com/lightspeed-core/rag-content) (indexing / container images) |

# What

This document captures a **high-level design and roadmap** for the next phase of Retrieval-Augmented Generation (RAG) work across Lightspeed Core. It spans:

1. **Round out support of CPU architectures and GPU variants in the rag-content builder image**:
   - **Add support for ROCm GPUs on x86_64** in the `rag-content` build image.
   - **E2E automated testing in Konflux** to verify all RAG build image flavors, run integration tests on automatically provisioned hardware.
   - **Detection of available GPUs** to provide `rag-content` users with a way to import `rag-content` as a base image based on what GPU is available.
2. **RAG evaluations** in partnership with the LEAD team: repeatable benchmarks, clear baselines, and automation. RAG evaluations must be easily runnable by engineers and AI agents, as well as periodic Konflux integration tests.
3. **Cross-encoder reranking** so results from heterogeneous retrieval (BYOK vs OKP, different embedding models) can be fused meaningfully.
4. **“No RAG” / large-context** experiments: very large prompts (e.g. full doc chapters) with long context windows and **prompt caching**, compared to retrieval-based flows.
5. **Multiple OKP logical stores**: one OKP endpoint with distinct stored queries / filters so agents can target CVE lookup, docs for individual products, etc.
6. **Query rewriting** with entity resolution: when the user relies on the current conversation (e.g. pronouns—“How do I fix **it**?”), infer what they refer to and run RAG against a resolved, self-contained query; also broaden underspecified queries to improve recall over the index.
7. **BYOK ingestion**: a broader set of formats - HTML, PDF, and other Docling-supported formats.
8. **BYOK sources**: ingest from GitHub, GitLab, Confluence, Wiki.
9. **Event-driven BYOK pipelines**: rebuild or update BYOK vector DBs when upstream content changes.
10. **Progressive disclosure:** experiments: evaluate staged presentation and/or iterative retrieval as an alternative to traditional one-shot RAG (retrieve top‑k chunks and inject them in a single pass).
11. **RAG retrieval observability:** a supported way to inspect why the system retrieved specific chunks (queries used, stores, scores, filters, merge/rerank order) for engineering debug, eval replay, and support—including, for similarity search, why a chunk is similar to the query at a lexical or span level (e.g. which terms or substrings in the query and chunk drive the match), not only which chunks ranked highest.

# Why

- **Heterogeneous hardware**: Customers and internal builders use AMD as well as Nvidia GPUs; indexing today is biased toward CUDA-centric stacks in practice. ROCm support widens where indexes can be built at acceptable cost and speed.
- **GPU vs CPU image selection**: Operators extending `rag-content` need a deterministic way to map available hardware to the correct image tag (CPU, CUDA, ROCm) without ad hoc `Dockerfile` forks. Explicit detection (probes, documented precedence, and CI matrix parameters) reduces silent CPU fallback and wrong-stack pulls.
- **Evidence, not opinion**: New RAG techniques (rerankers, rewrites, multi-store routing) need measured lift on agreed tasks; without pre-merge evals, wins are invisible or open to interpretation. Without periodic evals, the gates for regressions become open.
- **Incomparable relevance scores**: Today BYOK merges multiple FAISS stores using weighted cosine-style scores, while OKP uses a different retrieval stack, weights do not apply to OKP ([byok_guide.md](../byok_guide.md)). A cross-encoder (or similar late-interaction score) on `(query, passage)` pairs is a standard way to produce a single ranking after retrieval.
- **Product shape**: Focused tools (“CVE endpoint”, “OpenShift 4.21 docs only”) improve precision and simplify prompts; that maps to multiple filtered OKP views or named retrieval tools.
- **Conversation-aware search and entity resolution**: Follow-ups that use pronouns or other implicit references (“How do I fix **that**?”) are not valid retrieval queries on their own. Entity resolution maps those references to concrete entities or phrases from the dialog (product names, errors, CVEs, etc.) so the vector index is queried with a resolved string. That complements query rewriting for recall (casting a wider net when the utterance is short but not coreferential). RHEL Lightspeed already validated rewriting for recall; we explicitly add resolution for coreference before RAG.
- **Enterprise content**: Enterprise knowledge lives in PDFs, wikis, and git repos — not only Markdown on disk. Docling and remote connectors close that gap.
- **Freshness**: Static snapshots go stale; change-triggered rebuilds keep BYOK aligned with source-of-truth.
- **Progressive disclosure vs one-shot RAG**: Traditional RAG often injects many chunks at once, which can add noise, burn tokens, and bury the right passage. Progressive disclosure (in the broad sense: outlines or summaries first, then deeper material only when needed—or multi-step retrieval driven by the model or the user) may improve focus and efficiency. This is worth a measured comparison on the same eval tasks as one-shot RAG, large-context, and hybrid approaches.
- **Explainability of retrieval:** When answers look wrong, support and engineering must reconstruct the retrieval path (which embedding query hit which stores, what weights/filters applied, how BYOK and OKP results merged). Without a first-class trace, tuning rerankers, rewrites, and multi-store configs is guesswork and incident response is slow.
- **Similarity is not self-explanatory:** A dense embedding score alone does not tell a human which words aligned the query and chunk. Operators need attribution — overlapping tokens, highlighted spans, BM25-style term hits, or token-level scores from a cross-encoder—so “why this chunk?” is answerable alongside “why this rank?”.

# Current state

- **lightspeed-stack** implements inline RAG and tool RAG (`file_search`), merging BYOK (Llama Stack `vector_io`) and OKP (Solr-backed retrieval) in `build_rag_context` and related helpers in `src/utils/vector_search.py`.
- **BYOK** is configured via `byok_rag` and `rag.inline` / `rag.tool` in service YAML; **OKP** is toggled with the reserved `okp` id and `OkpConfiguration` (`chunk_filter_query`, `offline`, `rhokp_url`) in `src/models/config.py`.
- **Index building** for BYOK used the `rag-content` repository; the stack consumes compatible vector stores and enriches `run.yaml` via `llama_stack_configuration.py`.

# Design themes by initiative

## 1. Round out CPU/GPU support

### 1.1 ROCm/x86_64 in the `rag-content` build image

**Goal:** Produce a supported path to build embeddings / indexes on AMD ROCm in the same way CUDA is supported today.

**Architecture policy:** ROCm is supported only on x86_64 (amd64). ROCm wheels, CI, and documentation for this initiative do not target arm64 as there is no support for it on the part of RHOAI base images.

**Scope:**

- Base image / wheel selection for PyTorch + ROCm, there is a supported ROCm base image and index of Python packages from RHOAI.
- CI matrix entry.
- Documentation: prerequisites, hopefully can be referred to from the RHOIA base image tech notes.

**Dependencies:** Hardware or CI runners with ROCm; alignment with RHOAI base images and Python packages where applicable.

**Risks:** difficulty automating tests on the correct hardware, creating toil if testing semi-manually.

**Acceptance criteria / Definition of done:**

- A published `rag-content` image variant targeting ROCm exists and is documented.
- A test in CI proves the image can perform embedding + minimal index generation on ROCm hardware.
- Documentation includes supported ROCm versions and other prerequisites, amd64-only statement.


### 1.2 E2E automated testing in Konflux

**Goal:** Ensure build and integration validation for all RAG image flavors (CPU arches and GPU variants) is automated in Konflux on automatically provisioned hardware.

**Scope:**

- Define a Konflux pipeline matrix that covers CPU architecture variants and GPU-tagged image variants, consistent with §1.1.
- Add integration tests that execute a minimal end-to-end indexing workflow per image flavor (build image boots, dependencies resolve, embedding/index step succeeds).
- Validate hardware selection/provisioning path for GPU runs so tests are not dependent on manual runner setup.
- Publish artifacts and pass/fail summaries suitable for release gating.

**Dependencies:** Konflux pipeline support for matrix builds, access to GPU-capable workers, and a stable small integration corpus for deterministic checks.

**Risks:** GPU runner scarcity/queue delays, flaky provisioning, and longer CI time when the image matrix expands.

**Acceptance criteria / Definition of done:**

- Konflux runs an automated matrix that covers agreed CPU architecture and GPU image flavors.
- Each matrix job executes a minimal end-to-end indexing integration test and publishes artifacts/logs.
- GPU jobs use automatically provisioned hardware (no manual runner setup in release workflow).
- A clear gating policy is documented (required vs periodic jobs).

**Open questions:** Which image matrix is mandatory for every change vs periodic-only, and whether Konflux results should gate merges or only releases.

### 1.3 Base image selection based on detection of available GPUs

**Goal:** Give `rag-content` consumers (custom Dockerfiles, internal derivatives, CI templates) a supported pattern to choose the correct `FROM rag-content:…` variant—CPU, CUDA, or ROCm—based on what GPU (if any) is available in their environment, so that GPU/CPU branching is decided at image import (or build-arg) time rather than buried in application code with ambiguous runtime fallbacks.

**Problem statement:** Multiple mutually incompatible user-space stacks (CUDA vs ROCm wheels, CPU-only) cannot be merged into one universal image without the costs described in §1.1. Downstream teams otherwise copy brittle logic (“try `import torch` and hope”) or pick a tag by guesswork. Detection formalizes probes, precedence rules, and documentation so the choice of base image is explicit and auditable.

**Layers of detection (all may be documented; not all apply in every environment):**

1. **Orchestrated / CI (recommended for builds):** Konflux, Tekton, or GitHub Actions **matrix parameters** (e.g. `GPU_FLAVOR=cuda|rocm|cpu`) select the Dockerfile `FROM` or `build-arg`. The build does **not** rely on autodiscovery on the build host, which often has **no GPU**. This aligns with **Initiative B** (matrix cells = declared flavor).
2. **Runtime on target node (bare metal / interactive):** Before building a derivative image, operators run a **documented probe** (shell script or small Python entrypoint shipped beside `rag-content` docs) that inspects:
   - **NVIDIA:** presence of `/dev/nvidia*`, `nvidia-smi` exit status, optional NVML-based checks; PyTorch `torch.cuda.is_available()` when using a **CUDA-capable** throwaway env (same class of image as final stack).
   - **AMD:** `rocm-smi` / `rocminfo`, `/dev/kfd`, `HIP_VISIBLE_DEVICES`; PyTorch built for ROCm reporting a usable HIP device (`torch.version.hip`).
   - **Neither:** fall back to **CPU** tag.
3. **Precedence policy (example, finalize in implementation):** If both NVIDIA and AMD devices are visible (unusual), fail closed with a clear error unless `RAG_CONTENT_GPU_VENDOR` (or equivalent) is set—avoid non-deterministic picks.

**Non-goals:** A single “smart” `rag-content:latest` that downloads CUDA or ROCm wheels inside one image at first run (reintroduces fat-image and supply-chain problems).

**Scope:**

- Published detection script or Python module (exact packaging TBD: ship in repo, embed minimal layer in docs, or distribute as OCI artifact sidecar) with machine-readable output (e.g. `cpu|cuda|rocm`) and human-readable log of which probes passed or failed.
- **Documentation:** matrix of **environment → recommended tag**; **Konflux** example `Pipeline` params; **bare-metal** troubleshooting when probes lie (driver loaded but wrong user-space in container).
- **Tests:** unit tests for probe logic with mocked sysfs / subprocess; smoke on real GPU nodes in CI where available.

**Dependencies:** Stable image tag naming across CPU/CUDA/ROCm (§1.1); coordination with **Initiative B** so matrix labels match detection output vocabulary.

**Risks:** Probes false-positive / false-negative?

**Acceptance criteria / Definition of done:**

- Users can follow documentation to derive the correct base tag without reading Python internals.
- A reference Dockerfile pattern exists: `ARG RAG_CONTENT_TAG` or `FROM rag-content:${TAG}` where `TAG` is produced by CI matrix or by the detection helper on GPU-capable hosts.
- Detection output vocabulary aligns with published image tags and Konflux matrix dimensions.
- Override and diagnostic modes are documented (verbosity, dry-run).

**Open questions:** Whether the probe ships inside the `rag-content` image (larger) vs standalone script with minimal deps.

---

## 2. RAG evaluations with LEAD + Prow periodics

**Goal:** Any proposed RAG change can be evaluated against shared benchmarks; results are automated and historical (trends, regressions).

**Suggested architecture:**

| Layer | Responsibility |
|-------|----------------|
| **Benchmark suite** | Task definitions (queries, expected evidence, optional LLM-as-judge), golden or semi-golden sets, metrics (recall@k, nDCG, answer correctness, citation overlap). Owned jointly with LEAD where methodology lives. |
| **Harness** | CLI or job that runs LCS (or retrieval-only slice) + fixed configs, produces JSON / JUnit for Prow. |
| **Infra** | KOnflux periodic (nightly/weekly) with secrets for OKP or fixture-only mode; artifact upload for dashboards. |
| **Baseline policy** | Every merge request that touches RAG-critical paths can optionally run a lighter subset; periodics run full suite. |

**Acceptance criteria / Definition of done:**

- A versioned benchmark suite and harness are available and runnable in CI.
- At least one Konflux periodic job publishes machine-readable eval artifacts and trend-ready outputs.
- Baseline metrics are established and documented for current default RAG behavior.
- A policy exists for evaluating RAG-impacting changes against the benchmark suite.

---

## 3. Cross-encoder reranking

**Goal:** After candidate retrieval from multiple stores (BYOK FAISS, OKP, future multi-OKP), rerank top-N chunks with a cross-encoder so the final ordering reflects query–passage relevance rather than raw vector scores.

**Design sketch:**

1. **Retrieve:** Wider candidate pool per source (e.g. top `k_i` per store, cap total `K`).
2. **Normalize:** Plain text passages + stable doc identifiers for telemetry.
3. **Rerank:** Cross-encoder scores all `(query, passage)` pairs.
4. **Merge:** Sort by reranker score; apply optional per-source quotas to avoid one corpus dominating.
5. **Inject:** Existing `_format_rag_context` / tool path consumes the reranked list.

**Config:** Model id, `K`, batch size, CPU vs GPU, timeout fallback (skip rerank if SLO exceeded).

**Dependencies:** Model packaging (ONNX / sentence-transformers CrossEncoder), separate from embedding model.

**Risks:** Latency and memory at high `K`; need caching for repeated queries in the same session (optional).

**Relation to today:** Complements `weighted_score` merging in `_fetch_byok_rag`; OKP and BYOK can be merged after reranking.

**Acceptance criteria / Definition of done:**

- Reranker can be enabled via configuration and safely disabled (fallback path intact).
- Retrieved candidates from BYOK and OKP are reranked into a single final ordering.
- Eval results show non-negative quality impact versus baseline on agreed metrics.
- Latency impact remains within an agreed SLO budget (or graceful timeout fallback is verified).

---

## 4. Large context + prompt caching vs RAG (“no RAG” study)

**Goal:** Quantify when stuffing large corpora (e.g. full chapters) into a 1M-token-class context with prompt caching beats or loses to RAG on quality, cost, and latency.

**Method (spike → design):**

- Controlled tasks: same questions, three arms—(A) RAG baseline, (B) full-text inline context, (C) hybrid.
- Measure: token usage, time-to-first-token, cost, answer quality from eval harness.
- LLM provider-specific caching behavior (what is cached, prefix stability, invalidation).

**Outcomes:** Guidance for product defaults (“when to recommend traditional RAG vs long-context”) and possible config presets for large-context deployments.

**Acceptance criteria / Definition of done:**

- Controlled experiments for RAG baseline, no-RAG large-context, and hybrid are executed on the same task set.
- Results include quality, latency, and token/cost measurements in a shareable report.
- Provider-specific prompt caching behavior is documented with practical constraints.
- A recommendation matrix (when to use each mode) is agreed and captured.

**Open questions:** Which providers and models are in scope for caching semantics; maximum practical chapter size for stable prefixes.

---

## 5. Multiple OKP vector stores (same endpoint, different stored queries)

**Goal:** Expose several named retrieval surfaces (e.g. `okp-openshift-docs`, `okp-layered-products`, `okp-cve`) that all talk to the same OKP base URL but differ by stored query / filter (Solr query, OKP “collection” concept, or equivalent).

**Config direction:**

- Extend configuration from a single `OkpConfiguration` + global `chunk_filter_query` to a list of named OKP profiles, each with: `rag_id`, `chunk_filter_query` (and any OKP-specific parameters).
- Reserved namespace: map profiles to synthetic `rag_id`s used in `rag.inline` / `rag.tool` (similar to multiple `byok_rag` entries).
- **lightspeed-stack:** Solr/OKP fetch path selects the correct filter per `rag_id`; enrichment script registers multiple tool resources if needed.

**Dependencies:** OKP/Solr capabilities for the stored queries; agreement on naming for agent-facing tools.

**Acceptance criteria / Definition of done:**

- Configuration supports multiple named OKP profiles without breaking existing single-OKP configs.
- `rag.inline` and `rag.tool` can target specific OKP profiles by id.
- Integration tests verify profile-specific filtering and retrieval behavior.
- Documentation includes examples for CVE-focused and product-doc-focused OKP profiles.

---

## 6. Query rewriting and entity resolution

**Goal:**

1. **Entity resolution (coreference):** When the user’s message depends on conversational context (pronouns, “that issue”, “the command above”), determine what they refer to and formulate a standalone retrieval query that encodes that referent. RAG then runs against this resolved text, not the raw utterance alone.
2. **Query rewriting (recall):** For underspecified but not necessarily coreferential queries, expand or rephrase to improve coverage over the vector index (synonyms, product terminology, multi-query retrieval).

**Design sketch:**

1. **Trigger:** Optional step before `build_rag_context` / tool retrieval (configurable).
2. **Input:** Latest user message + short window of prior turns (or compacted summary if compaction is enabled).
3. **Resolve / rewrite:** A small, fast model or template+LLM step that (a) resolves implicit references into explicit search terms where needed, then (b) optionally rephrases or splits into one or more search queries (multi-query optional). Resolution and recall-oriented rewriting may be one pass or two, depending on latency and evals.
4. **Retrieve:** Run BYOK/OKP with the resolved query (or fuse results from multiple queries).
5. **Observability:** Log original vs retrieval query (PII policy compliant); distinguish resolution from pure expansion in metrics if useful.

**Acceptance criteria / Definition of done:**

- Retrieval can run using resolved/re-written queries while preserving original user prompt semantics.
- Ambiguous coreference cases follow a defined policy (disambiguate, conservative fallback, or skip resolution).
- Telemetry distinguishes raw query, resolved query, and expansion usage.
- Eval results show measurable recall or answer-quality improvement on conversation-dependent queries.

**Open questions:** Whether rewriting is always on for tool RAG or only inline RAG; check against RHEL Lightspeed requirements; whether resolution should run only when heuristics detect pronouns/coreference vs always-on lightweight resolution.

---

## 7. BYOK: Docling-backed formats (HTML, PDF, …)

**Goal:** Ingest HTML, PDF, and other formats Docling supports into the same BYOK indexing pipeline as today’s Markdown-centric path.

**Scope (primarily `rag-content` + docs):**

- Format detection, extraction, chunking strategy per format (preserve tables, headings).
- Failure modes: scanned PDFs without OCR, huge binaries—quotas and skips.

**Acceptance criteria / Definition of done:**

- BYOK ingestion supports at least HTML and PDF through Docling-backed processing.
- Integration tests validate extraction/chunking for representative documents and failure cases.
- Dependency and licensing review for added format tooling is complete and documented.
- User documentation lists supported formats, limits, and known caveats (e.g., OCR limitations).

---

## 8. BYOK: Remote sources (GitHub, GitLab, Confluence, Wiki)

**Goal:** First-class connectors that pull content from common enterprise systems before indexing.

**Design sketch:**

- **Connector interface:** authenticate → list / enumerate → fetch raw bytes or HTML → normalize to internal doc records → pass to Docling / chunker.
- **Auth:** PAT, OAuth apps, or cluster secrets (per connector); never log secrets.
- **Incremental:** Track etags / commit SHAs / page versions for delta updates (feeds into initiative 9).

**Open questions:** Which Wiki flavor (MediaWiki, Confluence-only, etc.); rate limits; air-gapped fallback (bundle export).

**Acceptance criteria / Definition of done:**

- At least two prioritized connectors (e.g., GitHub + Confluence) are implemented end-to-end.
- Connector auth flows are supported with secure secret handling.
- Incremental sync metadata (etag/SHA/version) is captured and used for delta ingestion.
- Connector-level integration tests cover happy paths and rate-limit/error handling.

---

## 9. Event-driven BYOK ingestion pipeline

**Goal:** When a source changes (git push, wiki publish, Confluence update), automatically rebuild or update the BYOK vector database and publish artifacts consumable by LCS.

**Design sketch:**

- **Triggers:** Webhooks, polling, or CI notifications from source systems.
- **Pipeline stages:** fetch → extract → chunk → embed → write FAISS/SQLite (or incremental index strategy if feasible).
- **Outputs:** Versioned DB artifact + manifest (source SHA, build id, embedding model id).
- **Deployment:** Rollout policy (blue/green mount, or hot swap with brief reload).

**Orchestration options:** Tekton, GitHub Actions, OpenShift Builds — choice depends on where sources live.

**Acceptance criteria / Definition of done:**

- A production-candidate pipeline can be triggered by at least one real source-change event.
- Pipeline emits versioned vector DB artifacts with source and embedding provenance metadata.
- Deployment/update strategy is validated.

**Open questions:** Full rebuild vs incremental vector updates; size of corpora; SLAs for “time to freshness.”

---

## 10. Progressive disclosure as a traditional RAG alternative (investigation)

**Goal:** Determine whether progressive disclosure patterns — delivering knowledge in stages rather than in a single retrieval-and-inject step—are a viable alternative or complement to traditional RAG for Lightspeed Core, and under what conditions they win on quality, latency, and cost.

**Working definition:** In this document, progressive disclosure means any pattern where the system does not rely solely on “retrieve top‑k chunks once and stuff them into the prompt.” Examples to compare in the spike (not all need to be in scope for v1):

- **Staged content:** Start with short summaries, outlines, or section titles; fetch or expand to full passages only when the user or agent signals need (second turn, tool follow-up, or explicit “show more”).
- **Iterative / multi-hop retrieval:** First pass retrieves coarse anchors (TOC, doc ids, headings); second pass retrieves targeted chunks inside those anchors.
- **API-mediated disclosure:** Client shows a thin layer first; backend or agent loads detail on demand (may pair with tool RAG).

**Method (spike → design):**

- Define 1–2 concrete disclosure strategies to implement as prototypes (e.g., summary-first + drill-down tool vs two-stage retrieval).
- Run the same benchmark suite as one-shot inline/tool RAG (initiative 2), plus comparison to large-context arms where relevant (initiative 4).
- Measure answer quality, citation usefulness, token use, latency (including number of round-trips), and failure modes.

**Dependencies:** Agreement on which variant of progressive disclosure is in scope; eval harness from LEAD collaboration.

**Risks:** Extra round-trips hurt perceived latency; over-compressed first stages may omit critical facts; agent-driven disclosure can be harder to test deterministically.

**Relation to today:** Today’s inline and tool RAG paths primarily reflect batch context injection or on-demand `file_search`; progressive disclosure would add an explicit staged policy or retrieval loop.

**Acceptance criteria / Definition of done:**

- Written definition of progressive disclosure for LSC (narrowing from the working definition above).
- At least one prototype implements a staged or iterative flow end-to-end enough to run evals.
- Comparative results vs traditional one-shot RAG are published (same tasks, same metrics) with a clear recommendation: adopt, defer, or hybrid.
- Documentation captures trade-offs (tokens, turns, UX) and any required API or configuration hooks.

**Open questions:** Which interpretation we want to prioritize — UI-driven disclosure, agent/tool multi-step retrieval only, hierarchical index (doc → section → chunk), or something else (e.g., a specific reference architecture).

---

## 11. RAG retrieval observability (debug / “why did we retrieve this?”)

**Goal:** Provide users, support engineers, and automated evals with a structured, reproducible view of the retrieval decision path: what text was used to query each store (raw user query vs rewritten vs entity-resolved), which vector stores / OKP profiles participated, per-candidate identifiers and scores (pre- and post-merge / rerank), Solr / filter queries for OKP, and the final ordering injected into the prompt or returned from tool RAG.

**Similarity attribution (within the same initiative):** Beyond listing which chunks were retrieved, the system should support explaining what made a chunk similar to the query — at minimum which query terms and which chunk terms/spans best explain the match under an agreed interpretable signal. Pure bi-encoder cosine similarity does not decompose into per-word contributions; therefore the design should combine:
- **Lexical / sparse signals:** token overlap, BM25 or Solr explanation fragments, highlighted substrings (case- and normalization-aware), or stopword-stripped term lists common to query and chunk.
- **Late-interaction / reranker path (when present):** if a cross-encoder (initiative 3) scores the pair, explore token attribution (attention rollout, gradient-based saliency, or model-specific explain APIs) as a phase-2 or optional enhancement, subject to model support.

**Problem statement:** Retrieval is a pipeline (rewrite → parallel store query → merge → optional rerank → format). Today, diagnosing “wrong chunk” requires correlating logs across components or reproducing by hand. A retrieval trace collapses that into one artifact per request (or per RAG step) suitable for support tickets, regression analysis, and LEAD harness ground-truth checks. Separately, “this chunk looks unrelated” is often a similarity explainability gap, not only a logging gap.

**Design sketch:**

- **Structured trace schema** (versioned JSON): e.g. `retrieval_trace_id`, `timestamp`, `conversation_id` (if applicable), stages: `{ "rewrite": {...}, "byok": [...], "okp": [...], "merge": {...}, "rerank": {...} }` with stable **chunk_id** / **document_id** / **source** fields aligned with referenced-documents emission. Per chunk (or top‑k), optional **`similarity_explanation`**: `{ "kind": "lexical_overlap" | "solr_explain" | "cross_encoder_tokens", "query_terms": [...], "chunk_spans": [{ "start", "end", "text" }], "notes": "..." }`.
- **Emission paths:**
  - **Logs:** DEBUG or dedicated retrieval logger channel with redaction of query text when policy requires.
  - **API / response extension (opt-in):** admin-only or `debug=true` query parameter / header returning trace in response body or a sidecar field (never default-on for end users without review).
  - **Streaming:** optional SSE event type `rag_trace` or final event payload field for clients that opt in during development.
- **Privacy:** Traces may contain PII in queries; gate behind role, feature flag, or deployment mode; document retention.

**Dependencies:** Stable identifiers from Llama Stack `vector_io` and OKP responses; alignment with Initiative C harness (export trace for failed eval cases).

**Risks:** Payload size; accidental PII leakage if traces are too verbose; performance of JSON serialization on hot path — mitigate with async or sampled traces.

**Acceptance criteria / Definition of done:**

- A documented schema for retrieval traces exists and is versioned.
- Engineers can enable trace collection on a dev deployment and answer “why was chunk X ranked above Y?” from one artifact.
- Support runbook references how to collect a trace for a bad answer.
- Automated tests verify trace includes store id, filter (OKP), and scores when those features are enabled.
- For at least one interpretable similarity path (e.g. lexical overlap + optional Solr explain), traces or API fields expose which terms or spans tie the query to each top chunk (schema field `similarity_explanation` or equivalent).

---

# Suggested phasing

Phasing is indicative; exact order should follow business priority and LEAD alignment.

| Phase | Initiatives | Rationale |
|-------|-------------|-----------|
| **0 — Foundations** | Eval harness skeleton + Konflux periodic (fixture mode); GPU support spike (ROCm + Konflux E2E automation + **GPU detection** / base-tag selection) | Unblocks measurement, reliable image validation, and predictable downstream `FROM` usage |
| **1 — Retrieval quality** | Query rewriting + entity resolution; multiple OKP profiles; cross-encoder rerank; **RAG retrieval observability** (traces) | Improves relevance and makes tuning/support evidence-based |
| **2 — Content breadth** | Docling formats; remote connectors | Expands addressable knowledge |
| **3 — Operations** | Event-driven pipelines; large-context/caching study; progressive disclosure investigation | Freshness, strategic positioning, and RAG alternatives |
| **4 — Experiments** | Large-context/caching study; progressive disclosure investigation | RAG alternatives |


# Cross-cutting requirements

- **Telemetry:** Retrieval latency breakdown (by store), reranker timing, rewrite and resolution on/off (or trigger reason), cache hit rates where applicable. Retrieval traces (initiative 11): opt-in emission of structured RAG decision data; similarity_explanation (lexical / Solr / optional reranker attribution); redaction and retention policy.
- **Security:** Connector credentials in OCP secrets; audit log of fetched URLs (sanitized).
- **Backward compatibility:** Existing single-OKP configs must keep working; new features opt-in via YAML.

---

# References

- [BYOK feature documentation](../byok_guide.md)
- [RAG configuration guide](../rag_guide.md)
- External indexing: [rag-content](https://github.com/lightspeed-core/rag-content)
