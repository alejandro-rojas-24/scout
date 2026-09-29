# Phase 4: later / out-of-phase ideas

## Phase 5
- **UI for attribution jobs.** P4 starts attributions from the CLI only (`scout attribute`); the UI only lists and
  opens stored Cards and colours the depth strip. Starting a job from the UI needs a gate panel for full plans, and a
  live job panel with host, stage, disk and heartbeats.
- **Report export.** Include the per-block attribution table, the null summary per reference, `ATTRIBUTION_CAVEAT`
  and the three standing disclaimers.
- **Retrain-plan loop.** Use the per-block attribution: re-initialise the attributed blocks, retrain, re-fingerprint,
  and confirm that the signal drops. The attribution test is the natural "signal".

## Unscheduled
- **Forward-pass activations for the matching test.** This is paper-faithful: Zhu et al. use hidden states on real
  text. P4 uses embedding-probe inputs, which are weights only (plan.md 3.8, deviation 1). Activations need torch or
  transformers, a layer-by-layer runner for 70B in 128 GB, and probably the Spark GPU.
- **Window search over reference blocks.** For depth-upscaled or pruned subjects (SOLAR, Llama-3.2-3B from 3.1-8B),
  test each subject block against a window of reference blocks, with the Bonferroni count widened by the window.
  P4 uses window 0 with P3's structure-only alignment, and a tie-break alignment can miss moved blocks.
- **More block types.**
  - MoE: per-expert matching, and expert permutations.
  - Fused `gate_up_proj`: split by the config intermediate size.
  - Non-GLU MLPs, such as CLIP `fc1`/`fc2`: an in/out matching variant.
  - T5 `DenseReluDense.wi_0/wi_1`, together with SentencePiece `spiece.model` parsing for the probe vocabulary.
  - Attention blocks: head-level matching of q/k/v/o.
- **Attribution evaluation on the P3 labeled set.** Run all blocks for the 51 pairs on the Spark (about 60–80 GB of
  MLP reads) to measure the null on 9+ independent families and the recall on the derived pairs. P4 only measures the
  null on the two exit controls. Since plan r1 this is also the measurement of assumption A-null (plan.md 3.8 step 7):
  on real hard negatives, does an independent candidate's aligned z look like a documented control's at equal depth?
- **JEV on quantised pairs.** The P3 JEV was calibrated on unquantised pairs. Quantisation adds distance, as a light
  fine-tune would (plan.md 3.7), so it is fail-safe for false derived but can cost recall. Add quantised derived pairs
  to the labeled set, or an explicit abstain policy.
- **More quantisation formats.**
  - GPTQ (incl. act-order `g_idx`), AWQ (interleaved nibbles), bitsandbytes nf4/int8 (`quant_state`), AQLM, HQQ, EXL2.
  - GGUF Q2_K, Q3_K, IQ*, TQ*, MXFP4, Q4_1, Q5_0 and Q5_1.
  - Each needs a decoder with byte-exact test vectors.
- **FP8 in σ-sample and anchor plans.** Plan the companion scale reads in the P2/P3 sampler. Attribution already
  decodes FP8.
- **`.bin` pickle inputs.** These need a restricted unpickler (or torch with `weights_only=True`) and a full download
  just to see the tensor names.
- **Card store.**
  - gc of stale entries.
  - Reuse of weight-derived stats across repos with equal `content_digest` (byte-identical weights), instead of only
    reporting it.
  - Syncing the store between the client and the Spark.
  - NFS support.
- **Job infrastructure.**
  - GPU matmul and assignment on the Spark (cupy or torch).
  - More than one worker (a real queue).
  - Downloads resumable across job restarts, which needs persisted partial scratch and so a revisit of invariant 1.
- **Throughput.** More parallel chunks per file, and reuse of the CDN/Xet location across chunks (the P1
  FUTURE_IMPROVEMENTS item).
- **GGUF canonical names for more architectures.** gemma2/3 (`post_attention_norm`, `post_ffw_norm`), phi3, qwen2moe
  and others. Until then they keep raw names and are untestable, not wrong.
- **Re-deferred from P2/P3.**
  - Retry slack scaling and streaming COMPUTE for the σ-sample path: their caps are frozen P2/P3 evidence.
  - The 405B σ-sample limitation (v_proj > 16 MiB per layer).
- **Environment.** Allowlisting `huggingface.co` and `*.hf.co` in this container, plus an ssh route to the Spark,
  would let E1–E4 run from here.
- **From validation round 2 (plan §11.2 deferred minors).**
  - Shift guard: measure its false-veto rate with cross-layer neuron inheritance at 80 blocks; make it relative (a
    shift z must exceed the same subject block's aligned z) or per block.
  - C9: a timed ranged read at E2, recorded in the preflight, so a throughput below about 9.5 MB/s is visible before
    the 6 h job.
  - Per-block control veto (a control LOO hit makes only that depth abstain) instead of the global one.
