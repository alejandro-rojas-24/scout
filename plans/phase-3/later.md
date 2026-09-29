# Phase 3: later / out-of-phase ideas

## Phase 4
- **Card store.** A content-addressed Card store with caching. P3's library is a manifest plus Cards in a gitignored
  directory, and the set scanner only skips a repo whose complete v2 Card already exists at its pin.
- **Retry slack and streaming COMPUTE.** Scale the retry slack with the plan, and add streaming COMPUTE (carried over
  from P2). The P3 E4 run reads about 13.1 GiB in 51 plans, and a mid-read reset on a 16 MiB layer can exhaust the 16 MiB
  slack. The remedy is still to rerun that repo with the same confirm pair.
- **FP8 / quantized embeddings.** DeepSeek-V3-Base (library) uses them. If its `embed_tokens` is not BF16, the Card has no
  anchors and no σ (MLA), so only structure, config and tokenizer are available. This needs dtype normalisation first.
- **Per-block attribution (Zhu et al.).** It could reuse the P3 labeled set as its evaluation ground truth.

## Phase 5
- **UI.** An analysis and System 2 panel in the web UI. P3 is CLI-only (`scout analyze`, D47). This also needs HTTP
  routes (`POST /api/analyze`) and a rendering of the claims table and abstention reasons.
- **Report export.** Export the analysis plus System 2 output with the three standing caveats.
- **Direction heuristics.** Present which model the metadata calls the base, with release dates, as a clearly labelled
  claim. Weights cannot show direction (D34).

## Unscheduled
- **Robust depth detrend.** Bisquare IRLS as an alternative to `edge_blocks_excluded`. P3 only measures e = 1 vs e = 2 on
  train/calibration negatives (D49). Adopting any change unfreezes P2 `SIGMA_PARAMS` and needs a P2 plan revision plus
  a re-run of the P2 exit.
- **Depth autocorrelation.** Estimate the effective n per architecture family. No P3 threshold uses the √n formula,
  because JEV consumes the in-run `z_shift`.
- **More sampleable architectures.**
  - MLA roles (DeepSeek-V2/V3)
  - fused QKV (Phi-3, Falcon, GPT-NeoX/Pythia)
  - T5 encoder roles
  - width-pruned models (Llama-3.2-1B/3B vs 3.1-8B: excluded from the labeled set as undocumented and width-changed)
  - Each needs its own invariance argument and null measurement.
- **Output-side anchor CKA.** Run the same Gram over `lm_head` rows for untied models. It doubles the anchor bytes. Add it
  only if the held-out results show that input embeddings are too often re-initialised.
- **Anchor sets beyond English.** Byte tokens or multilingual anchors, for models whose vocabularies lack many English
  whole-word tokens (`min_present = 128` currently).
- **SentencePiece `tokenizer.model` parsing.** Parse it for repos without `tokenizer.json`. The labeled-set criteria
  currently require `tokenizer.json`.
- **A larger labeled set.** Grow it towards more than 100 pairs, especially more merges (only 1 in the held-out set, and
  none in train or calibration: validation r2 minor 2, deferred) and more depth-upscaled models. Also give the held-out set
  a second not_derived family per shape group, so that losing one family (dscoder or the mistral7b quote) does not
  force a plan revision. Tighter calibration needs about 3× the data. The ±0.15 probability caveat stays until then.
- **Black-box behavioural signal for distillation.** It is invisible to weight forensics (invariant 5); P5 lists it as
  optional.
- **Environment.** This container's proxy blocks `huggingface.co` and `*.hf.co`, and no `ANTHROPIC_API_KEY` is configured.
  Allowlisting both, plus an HF_TOKEN secret with the Llama, Gemma and Mistral licences accepted, would let E1–E7 run here.
