# Phase 2: later / out-of-phase ideas

## Phase 3
- Calibrated verdicts. P2 shows only continuous per-layer r and never labels a pair
  "derived". The 0.80 / 0.40 thresholds exist only in the exit check. JEV (System 1) should
  consume `median_r`, the r distribution, alignment kind and tokenizer J as features, and
  calibrate them on the labelled set.
- Hard negatives: same architecture, trained independently (e.g. two 7B Llama-shaped
  models from different orgs). The P2 null argument assumes independent runs share only
  smooth depth structure. Measure that on the labelled set, and report the null median r
  per architecture family.
- CKA, spectral top-k, norm/std tools, and claimed-vs-detected `base_model` checks. P2
  records `claimed_base_match` but does not verify claims.
- Base reference library. P2 suggests only from Cards already in the out dir.
- An off-diagonal null in the DiffView: the median of R over non-aligned layer pairs, as an
  in-run baseline. It is omitted in P2 to keep the view minimal.

## Phase 4
- Content-addressed Card store. `GET /api/cards` is a plain glob; it neither skips rescans
  nor deduplicates.
- FULL_DOWNLOAD gate (batch jobs, auto-purge) and per-block attribution (`stats.attribution`).
- Streaming COMPUTE (read → SVD → release per tensor) to cut peak RAM from ≈ bytes_planned
  (544 MiB for Llama-3.1-8B) to one tensor. This needs stage semantics that allow
  interleaving SAMPLED_READ and COMPUTE, or sub-stage events.
- Huge models where even `v_proj` exceeds 16 MiB per layer (e.g. 405B: 1024 × 16384 × 2 B =
  32 MiB). Options: a sampled-layer subset (every k-th layer), or a column-block read. A
  column block is not contiguous in row-major storage, and a row block is not
  permutation-invariant, so both need a new invariance argument.
- GGUF/quantized sampled reads (dequantization before SVD).

## Unscheduled
- Hub-side candidate search (`/api/models?search=`, model tree `base_model:` filters) for
  suggestions beyond the local pool.
- SentencePiece `tokenizer.model` (protobuf) parsing for MinHash. P2 reads only
  `tokenizer.json` / `vocab.json` and records a note otherwise.
- Fused-QKV roles (`qkv_proj`, `query_key_value`, `c_attn`): split by config head counts
  before the SVD.
- Sampling more than one stack per unit (e.g. a vision tower of a VL text encoder), and MoE
  expert sampling.
- Width-pruned models (Minitron-style): the grid on relative rank makes the curves
  comparable, but the statistic has not been calibrated for them. They show as
  `width_changed: true`.
- Interactive confirmation for over-threshold META reads. They are still refused (P1
  semantics), which is the safe subset of invariant 2. Tokenizers are ≤ ~10 MB, so this has
  not been needed.
- Persisting DiffViews. They are recomputable from Cards (invariant 1), so they are never
  stored.
- The diff UI could let the user override the stack pairing manually.
- Environment: this container's proxy blocks huggingface.co and *.hf.co. Allowlisting them
  (plus an HF_TOKEN secret for the gated Llama repo) would let E1–E4 run here.
