# Future improvements

This file collects non-blocking review findings per phase for a later wave.

## Phase 1

- **Map redirect targets by normalised URL; no unmapped CDN bodies** — T015 behavior 1, T016 C10, plan §2 C10. Compare `httpx.URL` objects, not strings. C10 should also FAIL if any 2xx record with `body_bytes > 0`, other than the API call, has `orig_path` None. Source: r3-m2.
- **Catch HTTP that bypasses the injected client** — T015 behavior 2, plan §2 C11. In `run()`, patch `httpx.HTTPTransport.handle_request` at class level, or guard `socket.connect`. Add an offline test in which a fetch goes through a fresh `httpx.Client`. Soften the "no unlogged fetch path exists" wording. Source: r3-m3.
- **Automated view checks per target** — T016 behavior 3, plan §2 E1/E3. Assert the following automatically instead of relying on the manual E5:
  - `view.disclaimers == DISCLAIMERS`
  - every stack has a strip whose cell count equals its depth, and a stack node whose `count` equals its depth
  - at least one "running" poll returned new events, proving the log is live

  Source: r3-m4.
- **Config-derived expected constants** — T016, plan §2 E1/E2 and §8. Compute the expected `n_tensors` and `params_total` from the Card's `config.raw` using the §2 formulas. Also check `params_total * 2 == index_total_size` for all-BF16 Cards. A constant then changes only when two independent sources agree. Source: r3-m5.
- **Qwen-Image critical path** — T010 steps 3 and 5, plan §8. Read the index files and the META files through the existing thread pool. State the Qwen-Image round-trip budget in §8; the relative 307 adds a hop to each of these reads. Source: r3-m6.
- **Reuse the CDN Location for the second header read** — T005, plan §8. Reuse the Location from the 8-byte read for the `[8, 8+N)` read, re-resolving on 403 or expiry. This saves one hop per file. It was rejected in r2 because the 10 s budget was not at risk. Source: r2-m6.
- **"Indexed" vs "repeated" stacks** — T008, T011, T014, plan E5.
  - Containers such as `mlp.0`/`mlp.2` with depth ≤ 3 and no repeating block signature should get kind "indexed", with no depth strip.
  - Add a test using `mlp.0`/`mlp.2`.
  - Make the E5 wording non-tautological.

  Source: r3-m7.
- **Schedule fused-expert MoE support** — T008, later.md, D12. v0 only warns. Fused support (gpt-oss or Llama-4 `experts.gate_up_proj` with a leading expert dimension) should be assigned to a numbered phase. Source: r3-m8 (the warning part is fixed).
- **Unbuffered local reads** — T004 behavior 4. Use `open(path, "rb", buffering=0)` plus a `readinto` loop, so that read-ahead never pulls tensor bytes into the process. Add a spy test on the raw read sizes. Source: r3-m9.
- **Warn on skipped root weights in pipelines** — T010 behavior 2. When `model_index.json` exists, add a pipeline warning listing the weight files at the repo root that were not scanned (e.g. a single-file checkpoint). Source: r3-m13.
- **Script E4; D3 wording** — plan §2 E4 and D3. Make E4 a network-marked test that asserts:
  - exit code 0
  - `totals.weight == 0`
  - exactly 1 card path
  - the stage lines in order

  Also change D3 from "E1–E4" to "E1–E5". Source: r3-m14.
- **Cut subtree collapse/expand from P1** — T014 behavior 3d. This is UX polish, planned for P5, and no exit check needs it. Move it to later.md. Source: r3-m15, r1-m10 (partly: made optional in r1).
- **HF cache snapshot SHA as the local key** — T004 behavior 2, plan §3.3. When a local folder is `.../snapshots/<40hex>/`, use that SHA as the key after verifying blob hashes. It was rejected in r1 because the folder can be modified locally. Source: r1-m7 (optional part).

## Phase 2

- **Scale retry slack with the plan size**: T106/T108, plan §11.3. The 16 MiB retry slack is a fixed amount, so E3 can fail when the network resets mid-read. Today's remedy is to rerun E3 whole with the same `--confirm-plans` line. Scaling the slack to the plan's size would avoid that. Source: r1-m4.
- **Selection bias from the alignment step**: T110/T111, plan §11.3. The alignment is chosen using the per-layer correlations and then r is reported on it, which pushes r up for same-architecture pairs. The exit pairs aren't affected. Measure this against Phase 3's hard negatives. Source: r2-m5.
- **T5-style text encoders**: T107/T108, plan §11.3. Add sampling roles for `SelfAttention.{k,v}`, or at least a skip hint, and measure the null before adding them to the statistic. Source: r2-m6.
- **Cut or keep `scout suggest`**: T112, plan §11.3. It was kept, with the justification recorded; revisit after real use. Source: r2-m9.
- **Robust detrend for anomalies that go deeper than the first and last layers**: T109, plan §2.3. When the edge anomaly also covers block 1, the null still leaks (8/200 medians > 0.40 and 9/200 z > 3). A bisquare detrend closed that leak (0/100) but adds tuning constants, and excluding 2 layers per edge widens the L=16 null. Evaluate both in Phase 3. Source: r2 major, residual risk.

## Phase 3

- **Report-only diagnostics**: T214, plan §11.3. The diagnostics and the EXIT `nulls` field are reported but feed no check. Wire them into a check, or cut them after first real use. Source: r1-m14.
- **Merge coverage**: T201, plan §5. L30 is the only merge pair, and it is held-out only. Grow the labeled set with merges in train and calibration. Source: r2-m2.
- **Unequal-depth, identical-structure alignment**: T209. These pairs get an arbitrary tie-break alignment, so their σ features carry little information and they rely on CKA. Source: r2-m4.
- **Platt pool scale mismatch**: T210. The Platt pool mixes out-of-fold logits with final-model logits. This is accepted as a fail-safe caveat; refit with consistent logits later. Source: r2-m5.
- **`scout analyze` can score held-out Cards at any time**: plan §2.1. The one-shot ledger covers the exit scripts but not ad-hoc analysis. Consider having analyze refuse held-out repos until EXIT is recorded. Source: planner r2 residual gap.
- **Single points of failure in the not_derived set**: plan §5. Dropping deepseek-coder-base cuts the C9 sweep from 104 to 74, and the mistral7b independence quote is uncertain. Fallbacks are named, and verification must happen before FREEZE. Source: r2-m1.
- **D36 human acknowledgement**: plan §4.3. The stored anchor Gram is treated as a statistic, not weights (invariant 1). The orchestrator accepted this provisionally, and the human should confirm it. Source: r1-m15.

## Phase 4

_pending planning_

## Phase 5

_pending planning_
