# RAG roadmap — feature proposals

This document expands [`feature_proposal_template.md`](../../feature_proposal_template.md) into implementation-grade specifications for each initiative in [rag-next-steps-hld.md](rag-next-steps-hld.md). GPU-related work is split into ROCm image (A), GPU detection / base-tag selection (A2), and Konflux E2E matrix (B). RAG retrieval observability (debug traces) is Initiative L. Other initiatives follow the numbering in [rag-next-steps-hld.md](rag-next-steps-hld.md).

**Conventions:** “LCS” denotes Lightspeed Core Stack (`lightspeed-stack`). “`rag-content`” denotes the RAG builder image repository and its container artifacts. Where precise APIs are unknown, items are marked **TBD** with the owning team.

---

## Initiative A — ROCm GPU support in `rag-content` build image

**Author Name(s):** (fill in)
**Author Date:** (fill in)
**Due Date:** (fill in)
**Status:** Draft
**Community Consensus:** TBD

### Overview

**Goal:**
Deliver a supported, reproducible `rag-content` container image variant in which embedding and batch encoding for BYOK index construction execute on AMD GPUs through ROCm, with parity of functional outcomes (Llama Stack–compatible vector store artifacts) to existing CUDA-backed or CPU-backed build paths. The outcome is not merely “PyTorch sees a GPU” but end-to-end index generation validated in CI on ROCm hardware.

**Architecture policy:** ROCm is in scope only for x86_64 (amd64). There is no supported ROCm-on-arm64 image, CI cell, or documentation path under this initiative; operators on arm64 use CUDA or CPU. Rationale: PyTorch/ROCm wheel availability, platform QA focus, and support surface—revisit arm64 ROCm only if product and Platform explicitly expand scope later.

**Scope — in scope:**
- Selection and pinning of ROCm base image(s) and PyTorch wheels consistent with Red Hat platform guidance where applicable, for amd64 only.
- **Image tagging and naming** (e.g. `-rocm`, digest-pinned references in docs).
- **Smoke and integration** tests: import stack, single-batch embedding, minimal FAISS artifact write matching existing layout expectations consumed by LCS enrichment (`llama_stack_configuration.py` and BYOK paths).
- **Documentation:** driver and ROCm version matrix, known limitations, **CPU fallback** procedure, troubleshooting.

**Scope — out of scope:**
- **ROCm on arm64**; arm64 remains out of scope for this ROCm deliverable.
- Inference-time GPU acceleration inside LCS serving path unless a shared dependency is later mandated (this initiative targets **indexing**, not query-time embedding in production).
- Intel / Apple / other accelerators, until requested by PM and supported by RHOAI.

**Primary interfaces:**
- `rag-content` CLI / entrypoints used today for index build.
- Output artifacts: same vector store files and metadata conventions documented in [rag_guide.md](../rag_guide.md) / [byok_guide.md](../byok_guide.md).

**Constraints:**
- **ROCm artifacts are amd64-only;** Konflux matrix and docs must not imply arm64+ROCm support.

### Background & Motivation

BYOK adoption assumes LSC users can rebuild indexes on infrastructure they already run. In practice, PyTorch ecosystems standardize on CUDA-first wheels; ROCm requires different binaries, often different base OS assumptions, and distinct failure modes (driver/kernel coupling, `HSA_OVERRIDE_GFX_VERSION`, memory visibility). Without a first-class ROCm image, AMD sites either (a) run CPU indexing at prohibitive wall-clock cost, (b) maintain snowflake environments that drift from upstream `rag-content`, or (c) skip BYOK entirely.

From a supportability perspective, an unsupported ROCm path generates unbounded engineering tickets (“works on my box”) and undermines security posture (unvetted base images). A **tagged, CI-tested** ROCm artifact aligns with enterprise expectations: the same provenance and scanning pipeline as other release images.

### Design Options

**Option 1 — Dedicated ROCm image tag (recommended baseline)**
**Summary:** Produce **`rag-content:<version>-rocm`** (exact naming TBD) built from a ROCm-enabled base, installing only ROCm PyTorch and dependencies required for indexing.

**Technical approach:**
- **Multi-stage build** where feasible: compile-heavy or model-download stages separated from runtime stage to shrink attack surface.
- Pin **ROCm major.minor** and **PyTorch build** in lockfiles consumed by `uv` or equivalent; record hashes in build metadata.
- CI job matrix entry: `runs-on` or cluster node pool with ROCm; execute **smoke** (import, `torch.cuda.is_available()` equivalent for HIP, one forward pass) and **integration** (tiny corpus → artifact checksum or structural validator).  
- Document **minimum AMD driver** and **incompatible host** symptoms.

**Integration surface:**  
- Release README and [byok_guide.md](../byok_guide.md) cross-links.  
- Optional: Konflux build parameter `TARGET_GPU=rocm` mapping to Dockerfile target.

**Pros:**  
- **Isolation of ABI risk:** CUDA consumers never pull ROCm shared libraries accidentally.  
- **Clear support boundary:** “Issues on `-rocm`” are triaged against a known stack.  
- **Smaller images** than universal fat binaries.  
- Matches ecosystem norms (NVIDIA and AMD images are rarely merged upstream in PyTorch Docker land).

**Cons:**  
- **Operational multiplication:** N tags to scan, promote, and document.  
- **Drift risk** between tags if Dockerfiles are forked; mitigate with shared layers or build-args to a single Dockerfile with `ARG GPU_FLAVOR`.

**Complexity / Impact:**  
- **Engineering:** Medium — Dockerfile + CI + docs.  
- **Platform / QE:** Medium — access to ROCm runners, baseline performance expectations.  
- **Risk:** Wheel/base mismatch causing **silent CPU fallback** if not asserted in tests (mitigate with explicit device assertions in smoke tests).

---

**Option 2 — Single “universal GPU” image**  
**Summary:** One image containing both CUDA and ROCm user-space stacks, selected at runtime via environment variables or entrypoint branching.

**Technical approach:** Would require installing **both** vendor stacks or a complex dispatcher; runtime `LD_LIBRARY_PATH` switching is fragile and **security scanners** flag duplicate attack surfaces.

**Pros:**  
- Single tag for documentation simplicity.

**Cons:**  
- **Image size** often prohibitive for pull and storage.  
- **Conflicting dependencies** between CUDA and ROCm user-space in one filesystem tree are common.  
- **Harder CVE triage** (“which stack?”).  
- Violates principle of least artifact for regulated customers.

**Complexity / Impact:** High — **not recommended** unless a hard product constraint mandates one pull URL.

---

**Option 3 — ROCm via host-mounted drivers only (thin image)**  
**Summary:** Image assumes **host-provided** ROCm user-space mount (similar to some HPC patterns).

**Pros:** Smaller image.

**Cons:** **Non-portable** for Konflux/Kubernetes unless cluster guarantees mounts; breaks “pull and run” for customers. **Rejected** for primary path except as advanced appendix.

### Key Decision Points

1. **Base image lineage**  
   a. Align with **OpenShift AI / RHEL AI** golden images for predictable support.  
   b. Upstream minimal ROCm dev images for velocity — may complicate enterprise support statements.

2. **Version pinning strategy**  
   a. **Strict pin** (recommended): one ROCm + one PyTorch combo per release branch.  
   b. Floating minor updates: lower maintenance, higher surprise breakage.

3. **Test depth**  
   a. Smoke only on every build; full mini-index on nightly.  
   b. Full mini-index on every ROCm build (slower, stronger guarantee).

4. **CPU fallback semantics**  
   a. **Fail fast** if ROCm requested but unavailable (clear logs).  
   b. Silent CPU fallback (dangerous for SLA mis-estimation).

### Recommendation

Adopt **Option 1** with a **single Dockerfile** parameterized by `GPU_FLAVOR` to reduce fork drift, publish **`-rocm`** for **amd64 only** (and keep existing CUDA/CPU tags), enforce **fail-fast** when ROCm mode is selected but HIP device init fails, and run at least a **mini-index integration** on every merge to the release branch (PRs may use smoke-only if queue time demands). Document **strict pinning**, **amd64-only ROCm**, and upgrade playbook per ROCm release train. **Image tag names and `GPU_FLAVOR` values must stay aligned** with **Initiative A2** (detection vocabulary and CI matrix parameters).

### Stakeholder Feedback Log

| Stakeholder | Topic | Feedback |
|-------------|--------|------------|
| (fill in) | | |

---

## Initiative A2 — Detection of available GPUs (base image selection)

**Author Name(s):** (fill in)  
**Author Date:** (fill in)  
**Due Date:** (fill in)  
**Status:** Draft  
**Community Consensus:** TBD  

### Overview

**Goal:**  
Provide **documented, testable** mechanisms so that **downstream** `Dockerfile` authors and CI pipelines can **select the correct `rag-content` base image tag** (`cpu`, `cuda`, `rocm`, or the project’s exact naming scheme) **without** maintaining parallel Dockerfiles per site or relying on **silent** PyTorch CPU fallback. The selection is **deterministic** given explicit inputs (host probes, cluster pool, or build matrix parameter), and **fails loudly** when the environment is ambiguous or incompatible.

**Scope — in scope:**  
- **Vocabulary contract:** canonical string values for **flavor** (`cpu` | `cuda` | `rocm`) aligned with **Initiative A** and **B** image tags and Konflux matrix axes.  
- **Reference implementation** of detection: one of (a) POSIX shell script with **strict mode**, (b) small Python module using `subprocess` + optional `torch` probe when a CUDA/ROCm PyTorch is already on `PYTHONPATH`, or (c) both—**thin** script that calls Python for PyTorch checks.  
- **Probe ordering and precedence** documented in a **decision table** (e.g. explicit `RAG_CONTENT_GPU_VENDOR` overrides; dual-GPU conflict → error unless forced).  
- **CI patterns:** examples for **Konflux/Tekton** `params` / `matrix` that set `FROM rag-content:${TAG}` or `ARG TAG` without running GPU probes on **GPU-less** build hosts.  
- **Kubernetes patterns:** mapping **node pool** / **resource requests** (e.g. `nvidia.com/gpu`) to tag choice in **Pod** or **Pipeline** YAML.  
- **Diagnostics:** `--verbose`, `--dry-run`, exit codes (0 = success with stdout tag, non-zero = error with stderr reason).  
- **Unit tests** with mocked subprocess and filesystem fixtures; optional **integration** on GPU runners (same pools as Initiative B).

**Scope — out of scope:**  
- Auto-installing CUDA or ROCm drivers at container runtime.  
- A single **universal** image that bundles all GPU stacks (see Initiative A Option 2 rejection).  
- **LCS** inference runtime GPU selection (this initiative targets **`rag-content`** indexing consumers unless shared script is reused later by explicit decision).  
- **arm64 + ROCm** (Initiative A architecture policy).

**Primary interfaces:**  
- Published artifact: script path or `python -m rag_content.detect` (**TBD** package layout in `rag-content` repo).  
- **Environment variables** for override: e.g. `RAG_CONTENT_FORCE_FLAVOR`, `RAG_CONTENT_GPU_VENDOR`.  
- **Documentation** in `rag-content` README and [byok_guide.md](../byok_guide.md) cross-links.

**Constraints:**  
- Probes must not execute **privileged** operations beyond what a normal user on a GPU host can run (no `modprobe` in default path).  
- **No secrets** in logs; optional redaction of **PCI paths** in verbose mode for support bundles.  
- **Reproducible** CI: same matrix param → same tag on all builds.

### Background & Motivation

Separate **CUDA and ROCm** images exist because **user-space stacks are incompatible** in one filesystem tree at acceptable size. A consumer who **incorrectly** bases on the CUDA image on an AMD host (or vice versa) either fails at import time or, worse, **silently** runs CPU paths if error handling swallows initialization failures. **GPU detection** moves the **branch point** to the **earliest** explicit step: **which base image to pull**. That matches the product ask: **stop GPU/CPU branching deep in application code** and instead **fix the base image import**.

**Kubernetes** adds another dimension: the build often runs on a **node without GPUs**, while **runtime** workers have GPUs. Detection documentation must distinguish **build-time matrix selection** (parameter-driven `FROM`) from **runtime** verification (optional sanity check job on a GPU node using the same tag).

### Design Options

**Option 1 — CI matrix is source of truth; detection script for bare metal only (recommended)**  
**Summary:**  
- **Konflux/GitHub Actions:** never rely on autodiscovery during `docker build`; use `matrix` / `params` to set `FROM rag-content:<tag>`.  
- **Laptop / bare metal:** ship **detection script** for operators preparing a derivative.

**Technical approach:**  
- **Build:** `ARG RAG_CONTENT_TAG` defaulting to `cpu` for safety; pipeline sets `cuda` or `rocm` when building on pools that **validate** those images.  
- **Runtime probe:** `nvidia-smi` exit 0 → CUDA candidate; `rocm-smi` or `rocminfo` → ROCm candidate; else CPU. **PyTorch** checks optional second stage after confirming device nodes exist.

**Pros:**  
- **Deterministic** CI; no “wrong GPU on build host” flakiness.  
- Matches how OCI **multi-platform** builds already separate concerns.

**Cons:**  
- Operators must **maintain** matrix alignment with hardware pools; **documentation** is critical.

**Complexity / Impact:** Low–Medium — mostly docs + small script + tests.

---

**Option 2 — Always run detection script inside a `docker run --gpus all` preflight**  
**Summary:** Every build starts with a tiny container that runs probes, emits `TAG`, then `docker build --build-arg TAG=...`.

**Pros:**  
- **Single** entrypoint narrative for users.

**Cons:**  
- Requires **Docker-in-Docker** or privileged patterns in CI; **not** all Konflux environments support it.  
- **Heavier** and slower.

**Complexity / Impact:** Medium–High for CI integration.

---

**Option 3 — Runtime-only detection inside the fat application image**  
**Summary:** One image tries CUDA then ROCm then CPU at **Python import**.

**Pros:** One tag.

**Cons:**  
- Conflicts with **Initiative A** single-flavor images; **rejected** as primary pattern.

### Key Decision Points

1. **Dual visible GPUs** — error vs vendor priority vs user override.  
2. **PyTorch in probe** — require **torch** import (accurate but ties probe to wheel) vs **sysfs/nvidia-smi only** (faster, may miss user-space mismatch).  
3. **Shipping** — script in `rag-content` repo root `scripts/` vs `pip install rag-content[cli]` entry point.  
4. **Alignment** with **Initiative B** preflight: same probe library to avoid drift.

### Recommendation

Adopt **Option 1** as the **default story**: **matrix-driven `FROM` in CI**; **detection script** for interactive and bare-metal workflows. Share **probe logic** (or a thin wrapper) with **Initiative B** GPU health preflight where feasible. **Document** failure modes explicitly: **no `/dev/nvidia0`** in container despite GPU node → device request / runtime misconfiguration.

### Stakeholder Feedback Log

| Stakeholder | Topic | Feedback |
|-------------|--------|------------|
| Platform / K8s | Device plugin resource names | |
| (fill in) | | |

---

## Initiative B — E2E automated testing in Konflux (RAG build image matrix)

**Author Name(s):** (fill in)  
**Author Date:** (fill in)  
**Due Date:** (fill in)  
**Status:** Draft  
**Community Consensus:** TBD  

### Overview

**Goal:**  
Institutionalize **automated build-and-verify** for every **RAG-related container flavor** (multi-arch CPU and discrete GPU variants) inside **Konflux**, such that **hardware provisioning** for GPU jobs is **declarative** (Pod template, node selector, device plugin) rather than operator-dependent, and failures surface as **blocking or reported signals** per agreed policy.

**Scope — in scope:**  
- **Pipeline definition** (Tekton/Konflux primitives): parameterized builds producing each image flavor.  
- **Integration test task** per matrix cell: container start → dependency sanity → **deterministic mini-ingestion** (embed + write index subset).  
- **Artifact capture:** logs, JUnit or structured JSON results, image digest references.  
- **Policy document:** which cells are **merge gates** vs **periodic** vs **release-only**.  
- **Flake detection:** retry budget, **GPU health preflight** (probe semantics **aligned** with **Initiative A2** so the same “flavor” vocabulary and failure strings appear in CI logs and operator docs).

**Scope — out of scope:**  
- Full LEAD benchmark execution at scale (**Initiative C**).  
- Load/performance benchmarking unless explicitly added later.  
- LCS runtime E2E (Kubernetes deployment of LCS) unless later unified under a platform initiative.

**Primary interfaces:**  
- Konflux **Application** / **Component** definitions; cluster **MachineConfig** or **NodeFeatureDiscovery** rules for GPU pools (**TBD** per cluster).  
- Test harness repository location (**TBD**): may live in `rag-content` or a sibling `rag-content-ci` repo.

**Constraints:**  
- **ROCm GPU integration tests apply only to amd64** (see Initiative A); the matrix must not require or advertise ROCm on arm64.  
- GPU job **wall-clock SLO** (e.g. < 45 min per cell) to avoid blocking all merges.  
- Secrets: no long-lived credentials in logs; registry pull via workload identity.

### Background & Motivation

Container images are **release artifacts**. A broken `rag-content` image is a **supply-chain** failure: downstream teams cannot produce BYOK databases, blocking demos, customers, and internal dogfood. **Multi-arch** errors (wrong `GOARCH`, QEMU emulation gaps) and **GPU** errors (missing `libamdhip64.so`, wrong driver) typically appear **only** when the image runs on the target class of node. Konflux is the authoritative build path; therefore **verification must execute where the artifact is intended to run**, especially for GPU.

Manual provisioning does not scale and creates **non-reproducible** “green in my cluster.” Automation forces **node selectors**, **resource limits**, and **device requests** to be first-class, which is prerequisite to honest support statements.

### Design Options

**Option 1 — Full matrix on every PR**  
**Summary:** Each pull request that touches build inputs runs **all** arch × GPU cells.

**Technical approach:** Konflux `PipelineRun` matrix or fan-out tasks; aggregate pass/fail gate.

**Pros:**  
- **Minimum time-to-detect** for regressions.  
- Strong confidence before merge.

**Cons:**  
- **GPU queue contention** can stall contributor velocity.  
- **Cost** scales with PR frequency.  
- **Flaky GPU nodes** may false-negative merges without robust retry policy.

**Complexity / Impact:** High on **infra budget** and **SRE attention**.

---

**Option 2 — Tiered gating (recommended)**  
**Summary:**  
- **PR path:** fast subset — e.g. amd64 CPU + one **canonical** GPU flavor if pool available; skip GPU with explicit `skip-gpu` label only when justified.  
- **Periodic:** full matrix nightly/weekly.  
- **Release/tag:** mandatory full matrix before promotion.

**Technical approach:**  
- Pipeline parameters: `MATRIX_MODE=pr|nightly|release`.  
- **Signed attestation** or release checklist referencing digest verified by full matrix.

**Pros:**  
- Balances **latency** and **coverage**.  
- Release integrity remains high.

**Cons:**  
- **Late discovery** if PR subset misses an interaction bug (mitigate with good subset design: rotate GPU flavor per day).

**Complexity / Impact:** Medium — requires discipline in policy documentation and monitoring periodic failures.

---

**Option 3 — External GPU farm trigger**  
**Summary:** Konflux delegates GPU tests to an external Jenkins/GitLab runner with GPUs.

**Pros:** Uses existing GPU pools.

**Cons:** **Split brain** in provenance; harder to correlate build digest with test run unless tightly integrated.

**Complexity / Impact:** Medium–High integration tax.

### Key Decision Points

1. **Definition of “RAG build image flavors”** — enumerate amd64 CPU, arm64 CPU, amd64 + CUDA, amd64 + ROCm, etc. (**exact list TBD**). **ROCm applies only to amd64** per Initiative A; there is no arm64 + ROCm cell.  
2. **Mini-ingestion fixture** — vendored 200 KB corpus in git vs OCI artifact vs signed tarball from internal bucket.  
3. **Failure semantics** — whether a **missing GPU pool** fails open (dangerous) or blocks release (strict).  
4. **Attestation** — integrate with enterprise policy (SLSA, cosign) **TBD**.

### Recommendation

Implement **Option 2** with **release-blocking full matrix**, **PR subset** tuned to catch Dockerfile and Python dependency drift, and **nightly full matrix** as safety net. Publish a **runbook** for GPU preflight failures. Mandate **artifact upload** of test logs for any failed cell.

### Stakeholder Feedback Log

| Stakeholder | Topic | Feedback |
|-------------|--------|------------|
| (fill in) | | |

---

## Initiative C — RAG evaluations with LEAD + Prow periodics

**Author Name(s):** (fill in)  
**Author Date:** (fill in)  
**Due Date:** (fill in)  
**Status:** Draft  
**Community Consensus:** TBD  

### Overview

**Goal:**  
Establish a **shared, automated evaluation regime** for RAG-affecting changes: benchmark **tasks**, **metrics**, **datasets**, and a **harness** that can run on **Prow** on a schedule and emit **versioned artifacts** (scores, retrieved chunk traces, failure cases). LEAD owns or co-owns methodology; LCS provides **stable integration points** and **configuration profiles** for reproducible runs.

**Scope — in scope:**  
- **Task schema:** query, optional conversation history, expected evidence signatures (e.g. required doc ids or regex on titles), grading rubric.  
- **Harness modes:** at minimum **retrieval scoring**; optionally **full answer** scoring.  
- **Prow periodic** wiring, secrets for live OKP (if allowed), **fixture replay** mode when not.  
- **Baseline registration:** commit hash + config snapshot + metric vector for “known good.”  
- **Regression policy:** threshold deltas that fail the job (e.g. recall@10 drop > 2% absolute).

**Scope — out of scope:**  
- Owning all product-side UX evaluation of answers (unless LEAD scope includes it).  
- Training data collection from production without privacy review.

**Primary interfaces:**  
- LCS HTTP APIs or internal Python entrypoints for **retrieval-only** invocation (**TBD**: direct `vector_search` module vs live service).  
- `lightspeed-stack.yaml` / `run.yaml` **frozen configs** for eval.

**Constraints:**  
- **Determinism:** where models are stochastic, multiple samples + confidence intervals or fixed seeds.  
- **Cost ceiling** per run for LLM-as-judge.

### Background & Motivation

RAG pipelines are **high-dimensional systems**: embedding model, chunking, merge policy, OKP filters, rerankers (**Initiative D**), query rewrite (**Initiative G**). Human review of PRs cannot detect **silent recall regression** on edge queries. **Periodic** evaluation catches **drift** from upstream dependency updates (Llama Stack, Solr client, numpy) as well as intentional code changes.

Without LEAD alignment, engineering optimizes **local metrics** that may not match product goals; with alignment, “ship / no-ship” becomes evidence-based.

### Design Options

**Option 1 — End-to-end generative eval**  
**Summary:** Run full LCS + configured LLM; score final answers (exact match, F1, LLM judge, human spot checks).

**Technical approach:**  
- Containerized job with LCS, Llama Stack, and frozen model endpoints.  
- Capture **full traces**: prompts, tool calls, citations.

**Pros:**  
- **Highest fidelity** to user experience.  
- Surfaces **prompting** and **tool orchestration** bugs.

**Cons:**  
- **Expensive** and **noisy** (temperature, provider outages).  
- Harder to attribute failure to **retrieval vs generation**.

**Complexity / Impact:** High — suitable for **weekly** or **pre-release**, not every PR.

---

**Option 2 — Retrieval-first eval (recommended core)**  
**Summary:** For each task, run retrieval subgraph only; score **recall@k**, **nDCG**, **chunk overlap**, **citation presence** against labeled relevant passages or doc ids.

**Technical approach:**  
- Instrument `build_rag_context` / tool RAG path with **structured logging** of `vector_store_id`, `chunk_id`, score, filter applied.  
- Optional **synthetic OKP** fixture server returning recorded Solr payloads.

**Pros:**  
- **Cheaper**, faster iteration for retrieval PRs.  
- **Attribution** is clearer.

**Cons:**  
- Misses failures where retrieval is fine but **generation ignores** context.

**Complexity / Impact:** Medium.

---

**Option 3 — Dual-track**  
**Summary:** Default CI uses Option 2; **release** runs Option 1 on subset.

**Pros:** Best of both.

**Cons:** Two harnesses to maintain — mitigate with shared task library.

### Key Decision Points

1. **Dataset governance** — public synthetic only vs internal Red Hat docs samples (legal review).  
2. **OKP** — live dependency acceptable on Prow cluster vs mandatory replay (**availability / flake**).  
3. **Metric authority** — which metric is **merge-blocking** vs informational.  
4. **Multi-turn tasks** — required for **Initiative G** validation or phase 2.

### Recommendation

Implement **Option 2** first with **fixture replay** for OKP to stabilize CI; add **Option 1** on a **weekly** cadence or **release gate** once costs are bounded. Share **task YAML** or JSON schema with LEAD as the contract. Encode **regression thresholds** in harness config, not in shell scripts.

### Stakeholder Feedback Log

| Stakeholder | Topic | Feedback |
|-------------|--------|------------|
| LEAD | Metrics & datasets | |
| (fill in) | | |

---

## Initiative D — Cross-encoder reranking (BYOK + OKP fusion)

**Author Name(s):** (fill in)  
**Author Date:** (fill in)  
**Due Date:** (fill in)  
**Status:** Draft  
**Community Consensus:** TBD  

### Overview

**Goal:**  
Introduce a **late-interaction reranking** stage after **multi-source retrieval** so that the ordering of passages presented to the LLM reflects a **single commensurable relevance score** derived from **query–passage pairs**, mitigating the incompatibility between BYOK vector similarity scores and OKP ranking scores ([byok_guide.md](../byok_guide.md)).

**Scope — in scope:**  
- **Candidate generation:** configurable `k_per_source` and global cap `K_total`.  
- **Normalization:** text extraction for reranker input (strip markup, length cap per passage).  
- **Reranker model:** configurable identifier; batch inference; CPU and optional GPU execution path.  
- **Merge policy:** sort by reranker score; optional **per-source floor/ceiling** to limit dominance.  
- **Fallback:** if reranker exceeds **timeout** or errors, use **pre-rerank** ordering.  
- **Telemetry:** latency histogram, fallback rate, distribution of scores by source.  
- **Evaluation:** Initiative C harness compares baseline vs reranked on same tasks.

**Scope — out of scope:**  
- Training custom cross-encoders (unless ML platform later scopes it).  
- Replacing bi-encoder retrieval entirely (still need recall stage).  
- Reranking **after** LLM has already consumed context (this is strictly **pre-generation** unless tool loop is redesigned).

**Primary interfaces:**  
- `src/utils/vector_search.py` merge path after `_fetch_byok_rag` / Solr OKP fetch.  
- Configuration model in `src/models/config.py` (new subsection, e.g. `reranker:` with `enabled`, `model_id`, `max_candidates`, `timeout_ms`, `per_source_cap`).

**Constraints:**  
- **p95 latency budget** for inline RAG (TBD ms — must be agreed with SRE).  
- **Memory:** batch size × sequence length bounded for pod limits.

### Background & Motivation

Current BYOK multi-store fusion uses **weighted inner-product-like scores** per store, then global sort. OKP results use **different scoring semantics**; weights **do not apply** to OKP. The merged list is therefore **not** a unified relevance ranking—it is an **ad hoc interleaving**. Cross-encoders (or other late interaction models) are a well-studied remedy: they score **conditional** relevance—how likely a passage is to answer the query, jointly conditioned on query and passage text—in a shared space.

Without reranking, improvements in one store can **mask** better passages from another, and reranker-free tuning of `score_multiplier` remains **guesswork** across heterogeneous corpora.

### Design Options

**Option 1 — Single pooled rerank (recommended baseline)**  
**Summary:** Concatenate candidates from all sources into one list of up to `K_total` items; score each `(q, passage)` with cross-encoder; sort descending.

**Technical approach:**  
- Deduplicate by `(document_id, chunk_id)` where metadata permits.  
- **Batch** scoring (e.g. 16–32 pairs) with padding; use ONNX Runtime or native PyTorch depending on packaging.  
- **Quota extension:** after sort, enforce “at least `m` from OKP if available” via **reordering** or **constrained sort** (iterative refill).

**Pros:**  
- One model invocation graph; simpler observability.  
- Direct global optimum under reranker’s objective.

**Cons:**  
- Large single corpus can still dominate **recall** stage before reranker sees tail — mitigated by `k_per_source` tuning.

**Complexity / Impact:** Medium — localized to retrieval assembly; **high** if GPU packaging required in LCS image.

---

**Option 2 — Per-source rerank then merge**  
**Summary:** Rerank within each source independently; merge with explicit quotas.

**Pros:**  
- **Equity** across corpora; avoids single pool bias when raw scores differ in scale.

**Cons:**  
- **Cross-store** comparisons weaker: reranker scores across separately normalized lists may still be miscalibrated unless calibrated on joint data.  
- **Multiple** batch invocations → higher latency.

**Complexity / Impact:** Medium–High.

---

**Option 3 — Two-stage retrieve + rerank + retrieve**  
**Summary:** Coarse retrieval → rerank top windows → expand context around winners (similar to Maximal Marginal Relevance or hierarchical expansion).

**Pros:** Can improve **context coherence**.

**Cons:** Significantly more engineering; higher latency.

**Complexity / Impact:** High — **future** if Option 1 plateaus.

### Key Decision Points

1. **Model selection** — multilingual requirement? size vs accuracy? ONNX for CPU inference?  
2. **Placement** — inline RAG only vs also **tool RAG** `file_search` result sets.  
3. **Timeout policy** — degrade to pre-rerank vs drop OKP first (**policy**).  
4. **Caching** — cache reranker scores per `(query_hash, chunk_id)` within conversation TTL (**privacy** review).

### Recommendation

Ship **Option 1** with **configurable per-source `k`** and optional **post-sort quota** rules once evals show dominance problems. Use **CPU ONNX** first if it meets p95; introduce GPU only if batch latency fails SLO. Require **Initiative C** metrics to prove non-regression before default-on in production configs.

### Stakeholder Feedback Log

| Stakeholder | Topic | Feedback |
|-------------|--------|------------|
| (fill in) | | |

---

## Initiative E — Large context + prompt caching vs RAG (“no RAG” study)

**Author Name(s):** (fill in)  
**Author Date:** (fill in)  
**Due Date:** (fill in)  
**Status:** Draft  
**Community Consensus:** TBD  

### Overview

**Goal:**  
Produce an **empirical decision framework** comparing three strategies on **identical tasks**: (A) **RAG** with current or agreed baseline retrieval settings, (B) **no RAG** — full or large **inline** document bodies in prompt (chapter- or collection-scale where model allows), (C) **hybrid** (e.g. RAG for tail detail + pinned canonical excerpt). Incorporate **prompt caching** economics where the inference provider exposes stable-prefix caching.

**Scope — in scope:**  
- **Task set** aligned with LEAD (or engineering provisional set superseded later).  
- **Measurement protocol:** quality metrics (same as Initiative C arms), **time-to-first-token**, **total tokens in/out**, **estimated cost** from provider pricing tables, **cache hit** indicators when available.  
- **Provider matrix** documented per model: what constitutes cacheable prefix, invalidation rules, max prefix size.  
- **Written outputs:** internal report + executive summary + **recommendation matrix** (when to default RAG vs long-context).

**Scope — out of scope:**  
- Changing product defaults in YAML without leadership approval.  
- Guaranteeing 1M-token practical usability across all providers (investigation may conclude “not viable for X”).

**Primary interfaces:**  
- LCS request path for injecting large static prefix (may be **config-only** experiment harness, not user-facing).  
- Provider APIs: OpenAI-compatible caching headers or vendor-specific fields (**TBD**).

### Background & Motivation

**Context window inflation** and **prompt caching** alter the breakeven point between retrieval-augmented and **brute-force context** approaches. Without measured data, architecture debates default to ideology. The study should answer: for **our** query distribution and **our** corpus sizes, does stuffing chapters reduce **hallucination** or increase **lost-in-the-middle** failure? Does cache amortization make long-prefix multi-turn cheap enough to obsolete frequent re-retrieval?

### Design Options

**Option 1 — Ad hoc engineering spike**  
**Summary:** Time-boxed scripts, manual spreadsheets.

**Pros:** Fast early signal.

**Cons:** Weak **reproducibility**; hard to defend in review.

**Complexity / Impact:** Low.

---

**Option 2 — Harness-integrated study (recommended)**  
**Summary:** Extend Initiative C harness with **arms** and frozen configs; store results in artifact storage.

**Pros:**  
- **Reproducible** and **diffable** across commits.  
- Aligns metrics with production evaluation philosophy.

**Cons:** Longer setup.

**Complexity / Impact:** Medium.

### Key Decision Points

1. **Corpus slicing** — how chapters are selected (fixed TOC vs dynamic window).  
2. **Hybrid definition** — minimum viable hybrid to test.  
3. **Judge model** — same as production or fixed audit model.  
4. **Statistical rigor** — number of repeats for stochastic decoding.

### Recommendation

Use **Option 2** once Initiative C exists; until then, **Option 1** for exploratory curves only, with explicit **non-binding** label. Final recommendation must cite **confidence intervals** or repeated runs for stochastic arms.

### Stakeholder Feedback Log

| Stakeholder | Topic | Feedback |
|-------------|--------|------------|
| (fill in) | | |

---

## Initiative F — Multiple OKP logical stores (named profiles, same endpoint)

**Author Name(s):** (fill in)  
**Author Date:** (fill in)  
**Due Date:** (fill in)  
**Status:** Draft  
**Community Consensus:** TBD  

### Overview

**Goal:**  
Allow operators to declare **multiple OKP retrieval profiles** that share one **`rhokp_url`** (and shared auth if any) but differ in **Solr filter query** (or future OKP-specific stored-query parameters), each exposed as a distinct **`rag_id`** selectable in `rag.inline` and `rag.tool`, analogous to multiple `byok_rag` entries.

**Scope — in scope:**  
- **Pydantic models** extending `OkpConfiguration` or replacing it with a list of profiles while preserving **backward compatibility** for single-global-filter configs.  
- **Resolution logic** in OKP fetch path: map `rag_id` → filter string + any per-profile overrides (`offline` URL join behavior if ever divergent — default shared).  
- **`llama_stack_configuration.py`:** emit multiple registered tool/vector resources if required by Llama Stack semantics (**verify** against current enrichment patterns).  
- **Tests:** unit tests for mapping; integration tests with mocked Solr/OKP HTTP.  
- **Documentation:** examples for `okp-openshift-docs`, `okp-cve`, etc.

**Scope — out of scope:**  
- OKP server implementing new APIs unless partner team agrees (**separate** proposal).  
- Automatic discovery of profiles from OKP (**manual config** v1).

**Primary interfaces:**  
- `lightspeed-stack.yaml` schema.  
- `src/utils/vector_search.py` Solr query construction.  
- Constants for reserved ids (`okp` legacy behavior).

**Constraints:**  
- **Query length** limits for Solr; injection safety when composing filters (no string concat from untrusted input).  
- **Rate limits:** N profiles × concurrent searches per user request — cap parallel OKP calls.

### Background & Motivation

A single global `chunk_filter_query` forces **one** retrieval universe. Product requirements ask for **scoped** assistants: CVE corpus vs OpenShift core docs vs layered products. Today that implies **separate services** or **manual** config swap. Named profiles reduce operational friction and unlock **tool granularity** for agents.

### Design Options

**Option 1 — Explicit `okp_profiles` list (recommended)**  
**Summary:**  
```yaml
okp:
  rhokp_url: ...
  profiles:
    - rag_id: okp-ocp
      chunk_filter_query: '...'
    - rag_id: okp-cve
      chunk_filter_query: '...'
```
Legacy: if `profiles` absent, synthesize one implicit profile from top-level `chunk_filter_query`.

**Pros:**  
- **Validatable** schema; clear migration story.

**Cons:**  
- Verbose YAML.

**Complexity / Impact:** Medium.

---

**Option 2 — Encode filter in `rag_id` magic strings**  
**Summary:** `okp:ocp:product:openshift` parsed ad hoc.

**Pros:** Compact.

**Cons:**  
- **Error-prone**; escaping hell; poor UX in validation errors.

**Complexity / Impact:** Low short-term, **unbounded** long-term.

### Key Decision Points

1. Whether **`okp`** reserved id means “default profile” or **union** of profiles (**breaking** if wrong).  
2. Interaction with **tool RAG** naming in Llama Stack enriched `run.yaml`.  
3. **Max profiles** per deployment for supportability.

### Recommendation

**Option 1** with **legacy shim** and deprecation notice in docs. Implement **strict validation** (unique `rag_id`, allowed character set). Add **integration tests** that prove `rag.tool` only exposes intended stores.

### Stakeholder Feedback Log

| Stakeholder | Topic | Feedback |
|-------------|--------|------------|
| OKP team | Filters / rate limits | |
| (fill in) | | |

---

## Initiative G — Query rewriting + entity resolution

**Author Name(s):** (fill in)  
**Author Date:** (fill in)  
**Due Date:** (fill in)  
**Status:** Draft  
**Community Consensus:** TBD  

### Overview

**Goal:**  
Insert a **retrieval preprocessing** stage that (1) **resolves** conversational **coreference** and implicit references into **standalone search text**, and (2) **rewrites** underspecified queries to improve **recall** against BYOK and OKP indices, without altering the **user-visible** primary utterance unless product chooses to echo changes.

**Scope — in scope:**  
- **Inputs:** latest user message + bounded prior turns or **compaction summary** if conversation compaction is enabled (see separate compaction design).  
- **Outputs:** one or more **search strings** plus optional **structured hints** (product names, version pins) carried as metadata into retrieval.  
- **Configuration:** enable/disable, model selection, max tokens, temperature=0 preference, per-tenant overrides.  
- **Safety:** PII redaction rules for logs; **refusal** when resolution confidence low.  
- **Telemetry:** discrete events `resolution_applied`, `rewrite_applied`, `fallback_reason`.  
- **Eval:** multi-turn tasks in Initiative C.

**Scope — out of scope:**  
- General **dialogue management** or task planning beyond retrieval query formulation.  
- **Automatic** query translation to other human languages unless explicitly added.

**Primary interfaces:**  
- Call site immediately **before** `build_rag_context` and analogous tool-RAG query path.  
- Optional hook for **cached** rewrite per conversation turn id.

**Constraints:**  
- **Latency:** budgeted similarly to reranker (**TBD**); may use small local model vs LLM.  
- **Determinism:** prefer greedy decoding for eval reproducibility.

### Background & Motivation

Dense retrieval **requires** a query embedding aligned with training distribution of the bi-encoder. Pronominal follow-ups (“fix that”) embed poorly and retrieve **noise**. RHEL Lightspeed experience shows **rewriting** improves recall. **Entity resolution** addresses a distinct failure mode: the query is semantically incomplete **relative to dialog** but not necessarily **lexically** underspecified.

### Design Options

**Option 1 — Two-pass pipeline (recommended for observability)**  
**Summary:**  
1. **Resolution pass:** produce `q_resolved` (if coreference detected or always-on lightweight pass).  
2. **Expansion pass:** produce `q_final` or `{q1, q2}` for multi-query fusion.

**Technical approach:**  
- **Detector** (heuristic or small classifier) for coreference triggers vs blind pass.  
- **Fusion:** if multi-query, retrieve per query and **RRF** or score merge before reranker (**Initiative D**).

**Pros:**  
- **Debuggable** stages; metrics per stage.

**Cons:**  
- Higher **latency** and **cost** than single call.

**Complexity / Impact:** Medium.

---

**Option 2 — Single LLM call**  
**Summary:** One prompt returns JSON with `search_queries: [...]` and `entities: [...]`.

**Pros:** Lower round-trips.

**Cons:**  
- Harder to enforce schema; **harder** to test; mixing tasks may reduce quality.

**Complexity / Impact:** Medium.

### Key Decision Points

1. **Inline vs tool RAG** — same preprocessor or tool-specific (tool may need **shorter** queries).  
2. **Ambiguity policy** — ask user vs pick best guess vs **no rewrite**.  
3. **Model locality** — on-cluster small model vs remote LLM (**data residency**).  
4. **Interaction with moderation** — preprocessor must not bypass safety.

### Recommendation

Implement **Option 1** behind feature flag; ship **heuristic gating** for resolution to control cost, with eval-driven decision to widen. Integrate with **Initiative C** multi-turn suite before default-on.

### Stakeholder Feedback Log

| Stakeholder | Topic | Feedback |
|-------------|--------|------------|
| (fill in) | Alignment with RHEL Lightspeed prompts | |

---

## Initiative H — BYOK: Docling-backed formats (HTML, PDF, …)

**Author Name(s):** (fill in)  
**Author Date:** (fill in)  
**Due Date:** (fill in)  
**Status:** Draft  
**Community Consensus:** TBD  

### Overview

**Goal:**  
Extend the **ingestion** stage of BYOK index construction so that **HTML**, **PDF**, and additional **Docling-supported** formats are normalized into the same **internal document representation** (text blocks + structure metadata) currently produced for plain text / Markdown, then chunked and embedded identically to preserve **downstream compatibility** with LCS and Llama Stack.

**Scope — in scope:**  
- **Magic-byte / MIME** detection pipeline; explicit override flags for CLI.  
- **Docling** integration version pin; GPU/CPU execution policy for Docling internals (**TBD**).  
- **Chunking policy per format:** heading-aware splits for HTML; page or block splits for PDF.  
- **Metadata propagation:** source file path, page number, heading breadcrumb, table provenance.  
- **Quotas:** max pages, max MB, timeout per file, skip-with-warning behavior.  
- **Tests:** golden files for representative HTML tables, multi-column PDFs, broken encodings.  
- **Legal:** license compliance for Docling transitive deps.

**Scope — out of scope:**  
- OCR for scanned PDFs in v1 unless Docling pipeline already enables it with acceptable quality (**explicit** phase 2).  
- Ingestion of **executable** or macro-laden formats beyond static extraction.

**Primary interfaces:**  
- `rag-content` ingest CLI flags and config file schema.  
- Chunk JSONL or equivalent intermediate format consumed by embed step.

**Constraints:**  
- **Deterministic** chunk boundaries where possible for reproducible index diffs.  
- **Memory:** streaming parse for large HTML where libraries permit.

### Background & Motivation

Enterprise knowledge bases are predominantly **PDF** (contracts, manuals) and **HTML** (exported wikis, help sites). Limiting ingestion to plain text forces **manual pre-conversion**, which decays over time. Docling provides a **unified** extraction story reducing one-off parsers.

### Design Options

**Option 1 — Docling as primary extractor (recommended)**  
**Summary:** Route supported MIME types through Docling; maintain a **small** set of fallbacks for edge types.

**Pros:**  
- One dependency surface for many formats.  
- Active upstream for layout-aware parsing.

**Cons:**  
- **Image size** and **CVE** surface area increase.  
- Version coupling to PyTorch stack (**interacts with Initiatives A/B**).

**Complexity / Impact:** Medium–High on `rag-content` image.

---

**Option 2 — Per-format best-of-breed**  
**Summary:** PDF via `pdfplumber`, HTML via `readability`, etc.

**Pros:** Tunable per format.

**Cons:**  
- **N** dependency chains; inconsistent metadata schema; higher QA cost.

**Complexity / Impact:** High long-term.

### Key Decision Points

1. **Table handling** — keep as Markdown, HTML, or structured JSON in chunk text?  
2. **Image extraction** — strip vs caption via vision model (**out of scope** v1).  
3. **Failure taxonomy** — user-visible error codes for CI/CD.

### Recommendation

**Option 1** with **strict v1 scope**: digital PDFs + HTML; **document** scanned PDF as unsupported until OCR spike passes quality bar. Add **size/time quotas** early to prevent DOS via malicious files.

### Stakeholder Feedback Log

| Stakeholder | Topic | Feedback |
|-------------|--------|------------|
| (fill in) | | |

---

## Initiative I — BYOK: Remote sources (GitHub, GitLab, Confluence, Wiki)

**Author Name(s):** (fill in)  
**Author Date:** (fill in)  
**Due Date:** (fill in)  
**Status:** Draft  
**Community Consensus:** TBD  

### Overview

**Goal:**  
Provide **durable connectors** that enumerate and fetch authoritative content from **hosted systems** (Git hosting, wikis, team collaboration tools), normalize fetched objects into the **local ingest representation** consumed by Initiative H, and support **incremental sync** via stable remote revision identifiers.

**Scope — in scope:**  
- **Abstract connector interface:** `list_changes(since)`, `fetch_object(id)`, `normalize()` → file records.  
- **Authentication:** PAT, OAuth client credentials, or Kubernetes-mounted secrets; **rotation** guidance.  
- **Rate limiting:** exponential backoff, respect `Retry-After`, global concurrency cap.  
- **Security:** SSRF protections when given user-supplied URLs (**allowlist** hosts); no secret logging.  
- **Tests:** contract tests with **recorded HTTP** fixtures (VCR-style) where legal.

**Scope — out of scope:**  
- Write-back to sources.  
- Full SharePoint / Google Drive unless later prioritized.  
- Crawling the public internet without allowlists.

**Primary interfaces:**  
- `rag-content` subcommand `pull <connector>` or unified `sync` manifest YAML listing sources.

**Constraints:**  
- **Air-gapped** deployments may **disable** remote connectors; document **export bundle** path (**Initiative J** handoff).

### Background & Motivation

Manual export is **not** a workflow; it guarantees **stale** indexes. Connectors move BYOK from a **batch artifact** mindset to a **system integration** mindset, prerequisite for event-driven pipelines (**Initiative J**).

### Design Options

**Option 1 — Depth-first GA on two connectors (recommended)**  
**Summary:** Implement **GitHub** and **Confluence** first (highest demand **TBD** with PM); GitLab and MediaWiki next.

**Pros:**  
- Deep **edge-case** handling (pagination, attachments).  
- Establishes patterns for later connectors.

**Cons:**  
- Some users wait for their system.

**Complexity / Impact:** Medium per connector.

---

**Option 2 — Generic git + HTTP fetch only**  
**Summary:** Require users to clone or wget themselves.

**Pros:** Minimal code.

**Cons:**  
- **Support burden** shifts to customers; inconsistent metadata.

**Complexity / Impact:** Low now, **high** later.

### Key Decision Points

1. **Confluence:** Cloud vs Server API differences.  
2. **GitHub:** App vs PAT; org-level allowlist.  
3. **Wiki taxonomy** — which engines are in support matrix.  
4. **Legal** — storing fetched HTML in CI fixtures.

### Recommendation

**Option 1** with a **public roadmap**. Ship **manifest-driven** sync (`sources.yaml`) so pipelines remain declarative.

### Stakeholder Feedback Log

| Stakeholder | Topic | Feedback |
|-------------|--------|------------|
| (fill in) | | |

---

## Initiative J — Event-driven BYOK ingestion pipeline

**Author Name(s):** (fill in)  
**Author Date:** (fill in)  
**Due Date:** (fill in)  
**Status:** Draft  
**Community Consensus:** TBD  

### Overview

**Goal:**  
Automate **rebuild or update** of BYOK vector databases in response to **authoritative content changes**, producing **versioned artifacts** (vector store files + **manifest** with content hashes, embedding model id, chunking parameters, build id) suitable for promotion through dev → stage → prod.

**Scope — in scope:**  
- **Triggers:** repository webhooks, Confluence event hooks (if available), scheduled polling fallback, manual `workflow_dispatch`.  
- **Stages:** fetch (Initiative I) → extract (Initiative H) → chunk → embed → index write → **validate** (structural checks) → **publish** to artifact storage (OCI, S3-compatible, internal registry **TBD**).  
- **Deployment hooks:** documentation for mounting new DB version in LCS; **rollback** to prior manifest.  
- **Observability:** pipeline success metrics, freshness lag (time from commit to searchable), failure alerts.

**Scope — out of scope:**  
- Mandating a single CI vendor; reference implementation may be Konflux/Tekton while documenting patterns for others.  
- **Online** index mutation in running LCS without restart (**TBD** — likely out of v1).

**Primary interfaces:**  
- Artifact: `manifest.json` alongside `*.faiss` / sqlite paths.  
- LCS config: pointer to **immutable** artifact version.

**Constraints:**  
- **Reproducibility:** same inputs → same manifest hash (allowing only known non-deterministic steps, documented).  
- **Secrets** isolation per tenant in multi-tenant future (**TBD**).

### Background & Motivation

Without automation, BYOK **staleness** erodes trust (“assistant does not know the new doc”). Event-driven pipelines close the loop from **source of truth** to **retrieval surface**, analogous to CI/CD for software.

### Design Options

**Option 1 — Full rebuild per event (recommended v1)**  
**Summary:** On each qualifying change (or batched window), rebuild entire index from scratch.

**Pros:**  
- **Correctness** simple; **no tombstone** bugs in vector store.  
- Artifact is **immutable** and easy to roll back.

**Cons:**  
- **Compute** cost scales with corpus size; may require **batching** windows (e.g. hourly).

**Complexity / Impact:** Medium compute; **low** algorithmic risk.

---

**Option 2 — Incremental vector update**  
**Summary:** Delete/chunk-diff only affected documents.

**Pros:**  
- Lower steady-state cost at large scale.

**Cons:**  
- **Hard** with FAISS flat indices; may require **separate** incremental structure or periodic compaction.  
- **Consistency** bugs are subtle.

**Complexity / Impact:** High engineering.

### Key Decision Points

1. **Batching window** vs per-commit builds.  
2. **Artifact registry** authority (who signs manifests).  
3. **SLA** for freshness (minutes vs hours).  
4. **Multi-region** replication of artifacts.

### Recommendation

Ship **Option 1** with **configurable batching** and **size triggers** (if corpus > N docs, move from per-push to hourly). Begin **Option 2** spike only when cost data from Option 1 proves unsustainable.

### Stakeholder Feedback Log

| Stakeholder | Topic | Feedback |
|-------------|--------|------------|
| (fill in) | | |

---

## Initiative K — Progressive disclosure as a traditional RAG alternative (investigation)

**Author Name(s):** (fill in)  
**Author Date:** (fill in)  
**Due Date:** (fill in)  
**Status:** Draft  
**Community Consensus:** TBD  

### Overview

**Goal:**  
Characterize whether **progressive disclosure**—delivering evidence in **layers** (outline, summary, then targeted passages) and/or using **multi-step retrieval**—improves **answer quality**, **token efficiency**, and **user trust** versus **one-shot** top‑k chunk injection, under workloads representative of Lightspeed. Deliver **quantitative** comparisons using Initiative C harness extensions and **qualitative** UX notes where applicable.

**Scope — in scope:**  
- **Definition workshop** with Product/UX narrowing “progressive disclosure” to one or two implementable patterns.  
- **Prototype(s):** e.g. (P1) tool-mediated two-hop retrieval; (P2) summary-first inline + drill-down tool.  
- **Metrics:** same primary metrics as RAG evals plus **turn count**, **tokens per successful answer**, **abandonment proxy** if UI logs exist.  
- **Risk register:** latency, omitted critical facts in first layer, eval determinism.

**Scope — out of scope:**  
- Production enablement without eval sign-off.  
- Full redesign of client applications unless explicitly approved.

**Primary interfaces:**  
- May extend **tool RAG** definitions (`file_search` or custom tools) and/or streaming events to surface **staged** payloads (**TBD**).

**Constraints:**  
- Prototypes should be **feature-flagged** and **off** by default.

### Background & Motivation

One-shot RAG optimizes for **single-pass** recall but often injects **redundant** or **marginally relevant** chunks, increasing **noise** and **cost**. Progressive disclosure trades **additional round-trips** for **higher precision** in what the model reads first. The investigation determines whether that trade is net-positive **for our latency budgets and UX**.

### Design Options

**Option 1 — Backend/agent-mediated (recommended first prototype)**  
**Summary:**  
- **Hop 1:** retrieve doc titles + first paragraph or auto-summary per top doc.  
- **Hop 2:** user or model triggers **deep** `file_search` constrained to chosen doc ids.

**Technical approach:**  
- Constrain second query with **metadata filter** if BYOK supports it; OKP may require **filter query** composition (**Initiative F** synergy).

**Pros:**  
- **Minimal** client changes; evaluable in headless harness if model simulates second hop.

**Cons:**  
- **Longer** wall-clock for single user question unless parallelized cleverly.

**Complexity / Impact:** Medium.

---

**Option 2 — Client-mediated progressive UI**  
**Summary:** First response returns structured sections; expanding a section issues a **follow-up** API call with `expand_ref`.

**Pros:**  
- Strong **UX** alignment with disclosure metaphor.

**Cons:**  
- Requires **API schema** + client work; harder to test without UI automation.

**Complexity / Impact:** Medium–High cross-team.

---

**Option 3 — Hierarchical index navigation**  
**Summary:** Separate coarse index (titles) and fine index (chunks); always traverse coarse → fine.

**Pros:** Theoretically clean separation.

**Cons:** **Two indices** to build and maintain; higher **infra** cost.

**Complexity / Impact:** High.

### Key Decision Points

1. **Who controls hop 2** — always model vs explicit user vs hybrid.  
2. **Eval without UI** — acceptable approximations (forced tool policy).  
3. **Interaction with query rewriting (G)** — staged retrieval may **reduce** need for aggressive rewriting or **increase** it.

### Recommendation

Prototype **Option 1** under a flag; collect Initiative C numbers; pursue **Option 2** only if UX research shows client-driven disclosure is necessary for adoption. Defer **Option 3** unless coarse/fine indices are independently justified.

### Stakeholder Feedback Log

| Stakeholder | Topic | Feedback |
|-------------|--------|------------|
| Product / UX | Definition of progressive disclosure | |
| (fill in) | | |

---

## Initiative L — RAG retrieval observability (debug / explainability)

**Author Name(s):** (fill in)  
**Author Date:** (fill in)  
**Due Date:** (fill in)  
**Status:** Draft  
**Community Consensus:** TBD  

### Overview

**Goal:**  
Give **engineers, support, and automated eval harnesses** a **first-class, structured record** of *why* a given RAG invocation returned the chunks it returned: effective **queries** (after rewrite and entity resolution), **which stores** were queried, **filters** (OKP / Solr), **raw scores**, **merge and rerank** ordering, and **final** passages sent to the LLM or tool layer. The outcome is **actionable debugging** (“ranking dropped after reranker”) without reconstructing state from scattered log lines.

Additionally, for **similarity search**, expose not only **which** chunks matched but **what in the text** drove the match: **query terms** and **chunk spans** (or Solr **explain** text) that justify **why** a chunk is considered similar, alongside honest labeling that **dense embedding** similarity does not admit **per-token** decomposition without auxiliary signals.

**Scope — in scope:**  
- **Versioned JSON schema** for a **retrieval trace** (single object per RAG phase or per request), including: trace id, request id, optional conversation id, per-stage payloads (rewrite, BYOK per `vector_store_id`, OKP with `chunk_filter_query` / profile id, merge policy, reranker scores if Initiative D is enabled), and **ordered list** of chunk references with **stable ids** matching **referenced documents** / citation UX.  
- **Similarity explanation** payload per retrieved chunk (or top‑k), at least one of: **lexical overlap** (normalized token intersection, optional stemmed forms), **highlighted spans** in chunk text (character offsets), **Solr / OKP** `explain`-style or **highlight** fragments when available, **optional** cross-encoder **token saliency** when Initiative D model supports it.  
- **Opt-in emission:** configuration flag, admin role, or `Debug-RAG: true`-style header (**exact mechanism TBD**); never dump full trace to anonymous clients by default.  
- **Implementation locus:** `lightspeed-stack` retrieval assembly (`src/utils/vector_search.py` and streaming/tool paths); **structured logging** at minimum; **response extension** where API contract allows.  
- **Redaction:** hooks to strip or hash query text per **privacy** policy; document what fields are safe for INFO vs DEBUG.  
- **Tests:** unit tests that a synthetic retrieval produces a trace with expected fields; snapshot tests on schema version bumps.  
- **Runbook** for support: how to enable trace, where logs land, how to attach to Jira.

**Scope — out of scope:**  
- **End-user** “explain this answer” product UI (unless Product adds it; trace is the **backend prerequisite**).  
- Storing **full chunk text** in long-term telemetry without retention policy (**TBD** with compliance).  
- Claiming **word-level causes** for **pure bi-encoder** cosine scores without auxiliary signals (mathematically misleading); v1 must **label** explanation kind (`lexical_overlap` vs `neural_pair` vs `solr_explain`).  
- **Guaranteed** full **attention** attribution for every cross-encoder model in v1 (model-dependent; optional enhancement).

**Primary interfaces:**  
- HTTP response JSON field (e.g. `extensions.rag_trace`) or parallel **diagnostic** endpoint `GET /v1/debug/retrieval/{trace_id}` if persisted (**TBD**).  
- **OpenTelemetry** span attributes or log **trace correlation** (`retrieval_trace_id` = W3C trace id or internal uuid).  
- **Initiative C** harness: export trace on eval failure for LEAD.

**Constraints:**  
- **p99 overhead** bounded (e.g. serialize trace off hot path or truncate verbose fields in prod).  
- **GDPR / enterprise**: traces may contain **queries** with personal data—**TTL**, access control, and **regional** storage rules apply.

### Background & Motivation

RAG bugs are often **retrieval** bugs: wrong **rewrite**, wrong **OKP filter**, **score_multiplier** skew, **reranker** timeout fallback, or **OKP/BYOK** merge dominance. Today, proving which link failed requires **manual** correlation of application logs with Llama Stack and Solr behavior. A **single structured trace** per request collapses mean time to resolution and makes **A/B** config changes **auditable** (“trace diff” before/after).

This initiative also **unblocks** scientific rigor: LEAD and internal evals can **assert** on trace fields (“OKP must not appear when disabled”) instead of brittle string matching in logs.

**Similarity explainability:** Stakeholders routinely ask **“why is this chunk in the top 5?”** Vector similarity is **opaque**; **lexical** and **sparse** signals are **interpretable**. The recommended approach is **hybrid explanation**: always compute **cheap** lexical overlap between **effective query string** and **chunk text** for the trace; pass through **Solr** highlights/explain when OKP returns them; add **cross-encoder token attribution** only when Initiative D’s reranker model and latency budget allow.

### Design Options

**Option 1 — Ephemeral trace in logs + optional response body (recommended v1)**  
**Summary:** Build a `RetrievalTrace` object in memory; emit **JSON** to structured logger when flag enabled; optionally attach **subset** to HTTP response for admins.

**Technical approach:**  
- Pydantic model `RetrievalTraceV1` in `src/models/` or `src/utils/`.  
- Logger: `logger.info("retrieval_trace", extra={"trace": trace.model_dump()})` with redactor.  
- Response: field gated by config + auth.

**Pros:**  
- No new persistence service.  
- Fast to ship.

**Cons:**  
- Support must have **log access**; harder for customers to “export my trace” unless they use **response** field.

**Complexity / Impact:** Medium — touches hot path serialization carefully.

---

**Option 1b — Similarity explanation sub-pipeline (recommended alongside Option 1)**  
**Summary:** After candidate chunks are known, run a **deterministic** `explain_similarity(query_effective, chunk_text)` that returns **shared terms**, **span highlights** (simple regex or tokenizer-based), and **explain_kind**. For OKP, **merge** Solr `highlighting` / `explain` into the same trace field.  
**Pros:** Fast, no ML beyond existing retrieval; **honest** about what is being explained.  
**Cons:** Misses **semantic** matches with **no** shared tokens (label as `low_lexical_overlap` so users know the vector did the work).  
**Complexity / Impact:** Low–Medium.

---

**Option 2 — Persisted trace store (short TTL)**  
**Summary:** Write trace to Redis/SQLite/ object storage keyed by `trace_id`; return id to client; support fetches later.

**Pros:**  
- **Async** support can retrieve trace after the fact.  
- Enables **UI** “share debug bundle.”

**Cons:**  
- **Storage**, **PII**, **purge** jobs, **cost**.

**Complexity / Impact:** High — phase 2 unless compliance mandates.

---

**Option 3 — OpenTelemetry-only**  
**Summary:** Encode everything as span events and attributes.

**Pros:** Fits existing observability stacks.

**Cons:** **Schema** less ergonomic for support; **size limits** on attributes; harder to version a **documented** JSON contract for evals.

**Complexity / Impact:** Medium as **supplement**, weak as sole solution.

### Key Decision Points

1. **Default-off** globally vs **on in dev** always.  
2. **Streaming** path: same trace in **final** SSE event vs separate channel.  
3. **Chunk content** in trace: ids only vs **snippet** (first N chars).  
4. **Correlation** with Llama Stack **internal** request ids for vendor escalation.  
5. **Similarity explanation:** v1 **lexical + Solr** only vs **also** cross-encoder token maps when reranker exists.  
6. **Tokenization** for overlap: language-aware **split** vs naive whitespace for first ship.

### Recommendation

Ship **Option 1** with **Option 1b** (lexical + Solr explain/highlight) in the **same v1 schema** under `similarity_explanation`, a **frozen v1 schema**, **redaction** layer, and **Initiative C** hook to **persist traces to artifacts** on failed eval runs only. Add **cross-encoder** span attribution when Initiative D **and** model tooling allow, without blocking L on D. Revisit **Option 2** if Product requires **customer-facing** export or async support workflows.

### Stakeholder Feedback Log

| Stakeholder | Topic | Feedback |
|-------------|--------|------------|
| Security / Privacy | PII in traces | |
| Support | Runbook | |
| LEAD | Eval export | |
| (fill in) | | |

---

# References

- [rag-next-steps-hld.md](rag-next-steps-hld.md)  
- [BYOK feature documentation](../byok_guide.md)  
- [RAG configuration guide](../rag_guide.md)  
- External indexing: [rag-content](https://github.com/lightspeed-core/rag-content)  
- [`feature_proposal_template.md`](../../feature_proposal_template.md)
