# AdeptAI Full Integration Plan

## Goal
Integrate the entire `adeptai-master` folder (every module, asset, and configuration) into `backend/app/search` while preserving the current search API surface (`routes.py`, `service.py`, background initializers, etc.). The existing `adeptai_components` directory already mirrors a trimmed subset of the upstream repo; this plan tracks how to upgrade to the full upstream while keeping backward-compatible behavior for Kempian’s search endpoints.

## Upstream Inventory (adeptai-master)
- **`app/`**: Flask service layer (`services.py`, dependency injection container, schemas, auth, logging). Drives initialization order for SageMaker, search system, bias/bias subsystems.
- **`search_system.py`**: Central orchestrator that wires embeddings, dense retrieval, LTR, RL agent, bandit, explainability, caching, behavioural analysis, etc.
- **Advanced modules**: `behavioural_analysis/`, `bias_prevention/`, `explainable_ai/`, `market_intelligence/`, `semantic_function/`, `enhanced_models/`, `Sagemaker/`, `services/`, `scripts/`, `search/*.py`, `context_aware_weighting.py`, `custom_llm_models.py`, `learning_to_rank.py`, `rl_ranking_agent.py`, etc.
- **Support assets**: `cache/*.pkl`, `indexes/`, `logs/`, `templates/`, `tests/`, `accuracy_report.json`, `requirements.txt`, env template, multiple training scripts.
- **Frontend**: `adeptai-frontend/` (React) – optional, but note for completeness.

## Gap Analysis vs `backend/app/search/adeptai_components`
| Area | Present Today | Missing / Outdated |
| --- | --- | --- |
| Core orchestrator | `enhanced_recruitment_search.py`, `optimized_search_system.py` (trimmed) | Full `search_system.py`, `main_integrated.py`, `main.py`, `services/`, `app/` |
| Search engines | `search/fast_search.py`, `search/ultra_fast_search.py` | `search/instant_search.py`, `search/optimized_cache.py`, `search/performance.py` (full metrics), `search/cache.py` enhancements |
| Service container | None (handled ad-hoc in `service.py`) | `app/services.py`, `app/config.py`, validation schemas |
| Domain/market modules | Partially copied | Need synchronization with upstream (check for additional helper files and data assets) |
| Data assets | None | `/cache`, `/indexes`, `/logs`, `/accuracy_report.json`, training data scripts |
| Tests/docs | None | `/tests`, `/ALGORITHM_ARCHITECTURE.md`, `/README.md` etc. |
| Dependency manifest | Implicit | `adeptai-master/requirements.txt` (additional packages: lightgbm, xgboost, faiss, transformers, openai, anthropic, sentence-transformers, prophet, etc.) |

## Integration Strategy
1. **Vendor Package Layout**
   - Create `backend/app/search/adeptai_master` (or reuse `adeptai_components` but replace contents) and copy the entire upstream folder verbatim to preserve relative imports.
   - Ensure `__init__.py` files keep the package importable as `app.search.adeptai_master`.
   - Update `backend/app/search/service.py` and related helpers to import from `adeptai_master` instead of `adeptai_components` with a compatibility shim (e.g., symlink module names to avoid rewriting every import immediately).

2. **Path & Import Harmonization**
   - Adjust `sys.path` injections in `search/service.py`, background initializers, and any Celery/worker entrypoints to include the new directory.
   - Provide a compatibility alias module (e.g., `adeptai_components/__init__.py` importing everything from `adeptai_master`) to keep existing code operational during transition.

3. **Dependency & Environment Management**
   - Merge `adeptai-master/requirements.txt` into the backend Python dependency set (likely `backend/requirements.txt` or global installer). Highlight heavy packages (FAISS, transformers, lightgbm) that may require system prerequisites.
   - Promote `.env` variables from `env.template` into Kempian’s configuration management (OpenAI/Anthropic keys, AWS region, LLM model selections, feature flags like `ENABLE_BIAS_PREVENTION`, `USE_SAGEMAKER`, etc.).
   - Document GPU vs CPU variants (e.g., `faiss-gpu`, `torch` versions). Decide whether to gate optional modules (behavioural analysis, market intelligence) behind existing feature flags in `service.py`.

4. **Service Initialization Alignment**
   - Map `app/services.py` initialization order onto Kempian’s `search/service.py`. Identify overlaps (bias prevention, explainable AI, SageMaker) and determine whether to delegate to AdeptAI’s container or keep current orchestrator and call into AdeptAI APIs.
   - Ensure `service.py` continues to expose the same helper functions (`search_candidates`, `get_search_stats`, etc.) by wrapping the new AdeptAI objects rather than replacing public signatures.

5. **Data & Cache Handling**
   - Decide where to store AdeptAI cache/index assets (`cache/*.pkl`, `indexes/*.index`). Either keep under `backend/app/search/adeptai_master/` or move to a shared writable path configured via env vars.
   - Wire DynamoDB/table usage in `search_system.py` to existing data sources. Confirm table names, region (`ap-south-1` defaults), and IAM permissions.
   - Validate that candidate retrieval hooks (currently custom logic in `service.py`) can feed data into AdeptAI’s caching/indexing expectations (list of dicts with `skills`, `resume_text`, etc.).

6. **API / Route Compatibility**
   - Identify which functions the REST routes call today (e.g., `perform_search`, `get_candidate_insights`). Provide shims that translate existing request payloads into AdeptAI’s expected inputs (`search_system.search(query, filters, top_k=...)`) and transform responses back to Kempian’s schema.
   - Confirm asynchronous/background jobs (e.g., `background_initializer.py`, `search_initializer.py`) continue to warm caches or pre-build FAISS indexes using the new modules.

7. **Testing & Validation**
   - Bring over AdeptAI’s `tests/` and wire them into the backend test runner (pytest). Add integration tests that hit Kempian’s `/search` endpoints to ensure outputs remain stable after the swap.
   - Validate heavy dependencies behind feature flags so CI can run in a lightweight mode (e.g., mock FAISS/transformers when not installed).

## Detailed Task Breakdown
1. **Repository Sync**
   - Copy entire `adeptai-master` into `backend/app/search/adeptai_master/`.
   - Preserve hidden files (`.gitignore`, `.pre-commit-config.yaml`) for reference but update paths as needed.
2. **Bootstrap Script Updates**
   - Update `backend/app/search/background_initializer.py`, `search_initializer.py`, `service.py`, and any other modules referencing `adeptai_components` to import the new package namespace.
   - Expose convenience accessors (e.g., `from .adeptai_master.search_system import OptimizedSearchSystem`) so other code stays unchanged.
3. **Dependency Installation**
   - Extend backend requirements and runtime Dockerfiles/venv setup to include AdeptAI dependencies; note CPU/GPU variants and platform-specific wheels (FAISS).
   - Document optional installs (e.g., `bitsandbytes`, `stable-baselines3`) and guard imports in production if not available.
4. **Configuration Wiring**
   - Merge `env.template` keys into Kempian’s `.env` with prefixes if necessary (e.g., `ADEPTAI_OPENAI_API_KEY`). Update `app/settings.py` or config module to surface these.
   - Ensure secrets (API keys) are injected via Vault/Parameter Store rather than kept in repo.
5. **Data Source Integration**
   - Map AdeptAI’s expected candidate ingestion to existing DB layer. Provide adapter that converts Kempian resume records into AdeptAI’s normalized schema (see `_normalize_candidate_profile` in `service.py`).
   - Decide whether to keep AdeptAI’s DynamoDB usage or replace with our persistence layer.
6. **Feature Flag Matrix**
   - Align feature toggles across systems (e.g., `ENABLE_PARALLEL_PROCESSING`, `ULTRA_FAST_MODE`, `enable_behavioural_analysis`). Centralize into Kempian config so both legacy and AdeptAI logic read from same source.
7. **Testing & Observability**
   - Port AdeptAI’s monitoring hooks (`prometheus-client`, `PerformanceMonitor`) into our logging/metrics stack.
   - Add smoke tests ensuring search responses, ranking explanations, and bias prevention outputs remain identical pre/post integration.

## Environment & Secrets Checklist
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `HUGGINGFACE_TOKEN`
- AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`)
- Feature toggles (`ENABLE_BEHAVIOURAL_ANALYSIS`, `ENABLE_BIAS_PREVENTION`, etc.)
- SageMaker configuration (`SAGEMAKER_ENDPOINT`, `USE_SAGEMAKER`, per `app/config.py`)
- Cache paths and limits (`CACHE_DIR`, `CACHE_EXPIRY_HOURS`, `VECTOR_CACHE_SIZE`)

## Validation Plan
- Run AdeptAI unit tests via `pytest adeptai-master/tests` (after adjusting PYTHONPATH).
- Run existing backend search tests and end-to-end API checks.
- Benchmark latency/throughput before and after integration using `PerformanceMonitor` plus existing Kempian metrics.
- Verify data migration for caches/indexes (warm start without deleting current caches).

## Risks & Mitigations
- **Dependency Conflicts**: Transformers/torch versions may clash with existing backend libs; pin versions and test in isolated venv/Docker.
- **Startup Cost**: Loading full AdeptAI stack can slow boot; use lazy imports and background initializers (already supported) and ensure feature flags disable heavy modules when not needed.
- **Operational Footprint**: GPU/FAISS requirements might exceed current infrastructure; plan CPU-only fallback (existing `service.py` fast TF-IDF path) as safety net.
- **Security/Compliance**: External API keys stored in new configs—coordinate with DevSecOps for secret management and logging redaction.

## Current Integration Progress
- `adeptai-master` is mirrored under `backend/app/search/adeptai_master` so every upstream module, asset, and document is versioned alongside the backend.
- Core services (`app/__init__.py`, `service.py`, background/optimized initializers, and the legacy `backend/search1` stack) now import from `adeptai_master` first and automatically fall back to the prior `adeptai_components` implementations if a module is missing. This keeps the existing flow stable while enabling gradual adoption of the full upstream stack.
- Centralized helper logic (`_import_adeptai_module`) ensures newly added AdeptAI modules become available without changing downstream business logic. Future swaps can focus on wiring behaviour rather than repetitive import edits.
- Startup optimization now leverages the AdeptAI service container (`adeptai_master/app/services.py`) when available, so embedding bootstrap scripts can evolve toward the official provider configuration.
- Next steps: migrate runtime configuration to use the upstream settings schema, replace legacy `EnhancedRecruitmentSearchSystem` usage with the new `OptimizedSearchSystem`, and begin phasing in AdeptAI’s caching/indexing assets.

---
Use this document as the master checklist before beginning the actual copy/refactor work. Update it as modules get ported or when decisions (e.g., DynamoDB replacement) are finalized. Once the folder sync is complete, the next step is to modify `service.py` to instantiate AdeptAI’s `OptimizedSearchSystem` directly and ensure all existing routes continue to call the same wrapper functions.

