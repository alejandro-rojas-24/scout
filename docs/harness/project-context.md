# Shared project context (included in EVERY agent's prompt)

PROJECT: Self-use model forensics tool. Given model weights, a model card and
related metadata, it shows a model's architecture, which blocks derive from which
base model (e.g. Qwen-derived text encoder blocks), how to retrain those blocks,
and a calibrated probability of origin.

INVARIANTS (never violate; flag any plan or code that does):
1. The Fingerprint Card (JSON + Parquet, keyed by repo + commit SHA) is the only
   persisted artifact. Weights are never kept after COMPUTE.
2. Every byte fetched is logged. Stages: RESOLVE > HEADERS > META > SAMPLED_READ >
   [FULL_DOWNLOAD gate] > COMPUTE > COMPARE > REPORT > PURGE. Any read above the
   configured threshold requires explicit human confirmation with bytes, disk and
   reason shown.
3. System 1 = JEV-style calibrated classifier (may abstain).
   System 2 = reasoning model that reads Cards + System 1 output only, never weights.
4. Weight similarity uses permutation/rotation-robust measures (CKA, spectral,
   sigma-curves). Raw cosine is triage only and never drives a verdict.
5. Reports always state: distillation is invisible to weight forensics; tokenizer
   reuse alone is not proof; licensing is a human decision.
6. Intake handles: Hub repo @ pinned revision, local folder, sharded checkpoints,
   multi-component repos (model_index.json), MoE. GGUF/quantized from Phase 4.

ROADMAP:
- P1 Single architecture view (graph + depth strip, header-only, Card v0, live log)
- P2 Diff view + sigma-curve slice + download gate + reference suggestion
- P3 Analysis backend + labeled ground-truth set + JEV + System 2 + base library
- P4 Infra (batch jobs on DGX Spark, purge, Card store, quantized) + block attribution
- P5 Retrain-plan loop, report export, UX polish

STACK: Python backend. Go/Rust allowed where justified. Frontend: decided in P1.
