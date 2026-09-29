# Phase 2: later / out-of-phase ideas

## Phase 3
- Calibrated verdicts. P2 shows only continuous per-layer r and never labels a pair
  "derived". The 0.80 / 0.40 thresholds exist only in the exit check. JEV (System 1) should
  consume `median_r`, the r distribution, alignment kind and tokenizer J as features, and
  calibrate them on the labelled set.
- Hard negatives: same architecture, trained independently (e.g. two 7B Llama-shaped
  models from different orgs). The P2 null argument assumes independent runs share only
  smooth depth structure. P2 tests it per run (C14 `z_shift`, C18 off-diagonal gap) and
  reports the participation ratio, but only on the one unrelated exit pair. Measure it on the
  labelled set, and report the null median r and PR per architecture family.
- Depth-localised shared anomalies beyond the edge blocks. P2 fits the depth detrend on blocks 1..L−2 and scores
  only interior layers (`edge_blocks_excluded = 1`). An anomaly that also covers block 1 or L−2 still leaks
  (fixture, exit geometry: 8/200 null medians > 0.40, 9/200 z_shift > 3). A robust layer-weighted bisquare IRLS
  detrend closed it on the fixture (0/100, 1/100) but adds tuning constants. Measure both on real null pairs of
  the labelled set before choosing (validation r2 major; plan §2.3).
- Depth autocorrelation of residuals. The √n in the median-SD formula assumes independent layers; real
  residuals are autocorrelated. Estimate the effective n per architecture family on the labelled set.
- Selection bias of `dp-structure+weights`. For structurally compatible pairs the alignment minimises a cost
  containing (1 − R)/2, and `median_r` is then read on that alignment, which biases it upward. The exit is not
  affected (related = identity, unrelated = reldepth), but same-architecture hard negatives are. Report or
  calibrate the median on the structure-only alignment when the two differ (validation r2, minor 5).
- CKA, spectral top-k, norm/std tools, and claimed-vs-detected `base_model` checks. P2
  records `claimed_base_match` but does not verify claims.
- Base reference library. P2 suggests only from Cards already in the out dir.

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
- Retry slack scaled with the plan (e.g. `max(16 MiB, 4 × largest read)`), so two mid-tensor
  resets on 8 MiB reads do not fail a target. P2 accepts rerunning E3 whole (validation r1,
  minor 4; plan §11 deferred minors).

## Unscheduled
- Hub-side candidate search (`/api/models?search=`, model tree `base_model:` filters) for
  suggestions beyond the local pool.
- SentencePiece `tokenizer.model` (protobuf) parsing for MinHash. P2 reads only
  `tokenizer.json` / `vocab.json` and records a note otherwise.
- Fused-QKV roles (`qkv_proj`, `query_key_value`, `c_attn`): split by config head counts
  before the SVD.
- T5-style encoders (`encoder.block.#.layer.0.SelfAttention.{k,v}.weight`), the usual second text encoder of
  diffusers pipelines. They match no P2 role and are skipped without an architecture hint. Add the role patterns
  (or a "T5 attention" skip hint) together with a T5 null measurement (validation r2, minor 6).
- MLA attention (DeepSeek-V2/V3 style: `kv_a_proj_with_mqa`, `kv_b_proj`, `q_a_proj`/`q_b_proj`).
  There are no k/v_proj tensors. The low-rank `kv_b_proj` (or the product `kv_b_proj ·
  kv_a_proj`) is the candidate role, and it needs its own invariance argument. P2 skips it,
  and the skip note names "MLA attention".
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
