# Phase scopes

## Phase 1: Single architecture view
- Intake: Hub repo at a pinned revision, local folder, sharded checkpoints, and
  multi-component repos (read `model_index.json`, one Card per component).
- Header-only scan plus model card parsing, including the claimed `base_model`,
  license, and tags.
- Card v0 schema with empty stats slots; the Card is saved from day one and
  weights are never kept.
- Graph view and depth strip, MoE-aware (collapsed expert groups).
- The staged log and byte counter are live from the start, even when the count
  is 0 bytes.
- Decision point: frontend stack.
- Exit: Qwen3-8B, a Qwen3 MoE, and one diffusion pipeline each render correctly
  in under 10 seconds with zero weight bytes.

## Phase 2: Diff view + first weight signal
- The reference is picked manually, and the tool suggests candidates by matching
  config and tokenizer (tokenizer MinHash).
- Structural diff, including depth-upscaled and pruned alignment.
- Thin slice of weight evidence: sampled ranged reads -> sigma-curve correlation
  per layer, colored in the diff.
- The download gate lives here: estimated bytes shown and confirmation required
  before any read over a threshold.
- Exit: Qwen2.5-7B vs its Instruct version shows high per-layer correlation, and
  Qwen2.5-7B vs Llama-3.1-8B shows low correlation. Both correct without hand-tuning.

## Phase 3: Analysis backend + ground truth
- Labeled set: ~30-50 pairs with org-documented parentage, including merges and
  hard negatives (same architecture trained independently).
- Tools: norm and std, spectral top-k, CKA, claimed-vs-detected check against the
  model card.
- JEV as System 1, trained and calibrated on the labeled set, allowed to abstain.
- Reasoning model as System 2, reads only Cards and JEV output.
- Minimal reference library: base Cards for Qwen, Llama, DeepSeek, Mistral, Gemma.
- Exit: on held-out pairs, zero false "derived" verdicts, >=90% correct among
  non-abstentions, and an acceptable abstention rate (exact numbers set by human).

## Phase 4: Infrastructure + block attribution
- Full-download escalation as batch jobs on the DGX Spark, auto-purged afterward.
- Content-addressed Card store with caching; GGUF and quantized inputs, with
  dtype normalization before comparison.
- Block-level attribution via the Zhu et al. matching test, filling the depth
  strip per block.
- Exit: a 70B model and one diffusion text encoder both get per-block attribution
  end-to-end without manual intervention.

## Phase 5: Product refinement
- Retrain-plan panel as a loop (re-init flagged blocks, retrain, re-fingerprint,
  confirm the signal drops). Surfaces the license question rather than answering it.
- Report export with standing caveats: distillation is invisible to weight
  forensics; tokenizer reuse alone isn't proof.
- Optional: black-box behavioral signal to partially cover the distillation blind spot.
- UX polish from usage notes.
- Exit: you'd hand a report to someone else without editing it.
