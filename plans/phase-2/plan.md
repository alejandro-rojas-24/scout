# Phase 2 plan: Diff view + first weight signal

Status: PLANNED — approved by orchestrator under human directive (review disabled), 2026-09-29.
(Planner revision r2, answering validation rounds 1 and 2; §11.) This plan builds on the Phase 1 plan
(`plans/phase-1/plan.md`, tasks T001–T016) and treats it as a contract. Section 9 lists every
Phase 1 interface that changes, and the task that makes each change.

## 1. Goal

scout compares two Cards side by side. It shows a structural diff whose block alignment
handles depth-upscaled and pruned models, suggests a reference by config match and
tokenizer MinHash, and colours each aligned layer by a sigma-curve correlation computed
from full k/v projection matrices read with gated, logged Range requests. A weight read is
issued only after the plan has been displayed and a human has confirmed that plan's id and exact byte cap. Weights live only in
memory and are released before the Card is written, so the Card (now `card.v1`) remains
the only persisted artifact.

## 2. Exit check (runnable, falsifiable)

### 2.1 Targets, pairs, pins (fixed)

The exit targets are exactly these three repos:
- `Qwen/Qwen2.5-7B`
- `Qwen/Qwen2.5-7B-Instruct`
- `meta-llama/Llama-3.1-8B` (gated)

The pairs are fixed as `(label, reference, subject)` in `scripts/exit_expectations_p2.py::PAIRS` (T115):
- `("related", "Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct")`
- `("unrelated", "Qwen/Qwen2.5-7B", "meta-llama/Llama-3.1-8B")`

The Phase 1 conventions carry over unchanged:
- **C0 target set.** `set(pins) == {repos in PAIRS}`, or the run FAILs before any scan. There
  is no fallback repo, and swapping a target is a logged human decision followed by a plan
  revision.
- **Fixed endpoint.** The endpoint is always `https://huggingface.co`, and a foreign
  `HF_ENDPOINT` makes the run exit 1.
- **Pins.** Pins live in `exit/pins_p2.json`, committed as `"UNPINNED"` and written by
  `scripts/pin_exit.py --phase 2`. That script cross-checks `scout resolve` against
  `git ls-remote https://huggingface.co/<repo> refs/heads/main`, passing HF_TOKEN to git
  through `GIT_CONFIG_*` env so the gated repo resolves.
- **Transport and wire checks.** Every target is scanned with its own `make_server` and a
  fresh `CountingTransport`, reused from `scripts/exit_check.py`.
- **Evidence line.** The run ends with one JSON line `{"step":"EXIT","phase":2,...}`, whose
  `hostname` must equal the P2 PIN line's hostname.

**Credentials.** `meta-llama/Llama-3.1-8B` is gated. The run needs `HF_TOKEN` for an account
that has accepted the Llama 3.1 licence. `exit_check_p2.py` and `pin_exit.py --phase 2`
exit 3 before any request when `HF_TOKEN` is unset or empty. This is an environment limit
only: implementation and the offline suite never need the token.

**No hand-tuning.** The following are fixed by this plan before any real-model run:
- every constant in `SIGMA_PARAMS` and `ALIGN_PARAMS` (§4.6)
- the roles sampled
- the thresholds in §2.3 (including the in-run null thresholds of C12, C14 and C18)
- the synthetic fixture seeds (`weights_fixtures.NULL_SEED_PAIRS`, `FINETUNE_EPS/SEED`, `REHEARSAL_BASE/OTHER`, T103).
  Changing a seed to make an offline test pass is hand-tuning by another route and is forbidden.

Check C10 asserts that the code's params equal the frozen literals in
`scripts/exit_expectations_p2.py`. A FAIL goes to the human. Constants are never changed
after a real run without a logged human decision and a plan revision.

**Phase status rule.** Phase 2 closes only when E0–E5 have all passed. The Phase 1 exit
(E0–E5 of the Phase 1 plan) must still pass, because Phase 2 leaves plain scans unchanged.
After each run, one `step:"EXIT"` line with the full stdout is appended to
`plans/phase-2/log.jsonl`.

### 2.2 Commands

```bash
# E0 offline (no network, no token): P1 + P2 suites, incl. the offline P2 exit rehearsal (T116)
pip install -e '.[dev]'
pytest -q                                   # expect exit 0, 0 failures; network tests deselected

# E1 pins (Hub host, HF_TOKEN set, Llama licence accepted)
python scripts/pin_exit.py --phase 2 --pins exit/pins_p2.json
# expect exit 0; last stdout line {"step":"PIN","phase":2,"hostname":...,"repos":{...}}

# E2 plan only: every gate plan is displayed in full (stderr), confirmation declined, 0 weight bytes
python scripts/exit_check_p2.py --pins exit/pins_p2.json --plan-only
# expect exit 0. Prints 3 full plans (format_plan: plan_id, bytes planned, cap, disk 0 B, memory, reason, per-file table),
# then "CONFIRM WITH: --confirm-plans <id1>:<cap1>,<id2>:<cap2>,<id3>:<cap3>" and "TOTAL CAP: <N> B".
# At plan time (published configs):
#   Qwen2.5-7B           bytes_planned 205520896  cap 222298112  (28 layers x 2 x 512x3584 x 2 B, + 16 MiB)
#   Qwen2.5-7B-Instruct  bytes_planned 205520896  cap 222298112
#   Llama-3.1-8B         bytes_planned 536870912  cap 553648128  (32 x 2 x 1024x4096 x 2 B, + 16 MiB)
#   N = sum(bytes_cap) = 947912704 + 3 x 16777216 = 998244352

# E3 the confirmed run. The human reads the three plans and copies the CONFIRM WITH line; that is the confirmation.
python scripts/exit_check_p2.py --pins exit/pins_p2.json --confirm-plans <id1>:<cap1>,<id2>:<cap2>,<id3>:<cap3> --out /tmp/p2cards
# expect exit 0; each plan is displayed again before its confirm POST; the table shows C0..C18 all PASS;
# the last line is the JSON evidence

# E4 CLI path, same Cards
scout diff /tmp/p2cards/Qwen__Qwen2.5-7B/<sha>/_model.card.json \
           /tmp/p2cards/Qwen__Qwen2.5-7B-Instruct/<sha>/_model.card.json --json | jq .weight.median_r
# expect: equal (|diff| < 1e-9) to the E3 "related" median; exit 0

# E5 human visual check (recorded in log.jsonl): see 2.4
```

`--plan-only` and `--confirm-plans` are mutually exclusive, and exactly one is required.
With `--confirm-plans`, the runner displays each plan (stderr) as soon as its scan reaches
`awaiting_confirmation`, then confirms it through `POST /api/scans/{id}/confirm` with
`bytes_confirmed = bytes_cap` only if that exact `(plan_id, bytes_cap)` pair is in the list. Otherwise it
displays `NOT CONFIRMED` and declines, and the affected checks FAIL. There is no running byte sum: a
plan that was never displayed cannot be approved, even if its total is smaller. Plan ids are
deterministic at a pin (§4.4), so E2 and E3 produce the same ids.

**Mid-read network faults.** `RETRY_SLACK_BYTES` stays 16 MiB per plan. Two mid-tensor resets on
Llama's 8 MiB reads can exhaust it; the target then fails with `WeightReadRefused` and no Card. The
accepted remedy is to rerun E3 as a whole with the same `--confirm-plans` line; the rerun is a fresh
human invocation of the confirmed plans (validation r1, minor 4; scaling the slack is deferred, §11).

### 2.3 Checks (P2 namespace; each is one row `target | check | expected | actual | PASS/FAIL`)

The per-target checks C1–C10 run for each of the 3 targets.

| # | Check | Threshold |
|---|---|---|
| C0 | target set: `set(pins) == {repos in PAIRS}` (before any scan) | equal, else FAIL and exit 1 |
| C1 | **gate holds**, snapshot taken when status first becomes `awaiting_confirmation`: server `totals.weight`, Σ `bytes_by_class.weight` over events, and wire records with orig_path `*.safetensors` whose range ends past `8+N` | all `== 0` |
| C2 | plan size derived from `config.raw`: `bytes_planned == 2 × L × kv_dim × hidden × 2`, where `kv_dim = num_key_value_heads × (head_dim or hidden_size // num_attention_heads)`; `n_reads == 2L`; every planned tensor is BF16 | equal |
| C3 | plan ranges: every read `== [8+N+data_begin, 8+N+data_end)` of a `model.layers.#.self_attn.{v,k}_proj.weight` row of the Card Parquet, and max bytes per layer `<= 16777216` | true |
| C4 | bytes after confirmation: Σ weight bytes of `fetch` events `== bytes_planned`, and `bytes_planned <= totals.weight <= bytes_cap` | true |
| C5 | wire: each `.safetensors` request has a `Range` that lies in `[0, 8+N)` or equals a planned range exactly; each planned range has ≥ 1 2xx record with `body_bytes == end-start` | true |
| C6 | wire total: Σ body bytes `==` ByteLog `meta+header+weight` | equal |
| C7 | wire hosts are `huggingface.co` or `*.hf.co`; `card.source.endpoint == "https://huggingface.co"` | true |
| C8 | Card v1 content: `schema_version == "card.v1"`; `stats.sigma_curves.tensors` has 2L entries; Parquet `stat_sigma_curve` is non-null for exactly those 2L rows, each of length `min(shape)` and non-increasing; `fetch_log.gate` has exactly 1 decision with `approved`, `via == "api-confirm"`, `bytes_confirmed == bytes_cap` (exact) and the plan_id of C2, and that `(plan_id, bytes_cap)` pair is in `--confirm-plans`; `stats.tokenizer_minhash.num_perm == 256` and `source_file == "tokenizer.json"` | true |
| C9 | purge: `fetch_log.events` of the Card holds a `purge` event with stage `COMPUTE` and note containing `resident=0`, and its `seq` is smaller than the `seq` of the `REPORT` `stage_start` event **read from the live log** (the live log always has it; whether the Card's `fetch_log` includes that event is not relied on); the live log has a `purge` event with stage `PURGE` and note `PURGE check: resident=0 names=0`; out dir holds only `*.card.json` / `*.tensors.parquet` | true |
| C10 | frozen params: `scout.sigma.SIGMA_PARAMS == FROZEN_SIGMA_PARAMS` and `scout.align.ALIGN_PARAMS == FROZEN_ALIGN_PARAMS` | equal |
| C11 | related structure: exactly 1 stack pair `model.layers`↔`model.layers`, `kind == "same"`, 28 layers, each `op == "match"` with `reference_index == subject_index`, `jumps == []` | true |
| C12 | related weight: `weight.n_with_r == 26` (28 − 2 edge blocks, §3.4), **`weight.median_r >= 0.80`** and **`weight.q10_r >= 0.50`** | true |
| C13 | unrelated structure: `kind == "incompatible"`, `mode == "reldepth"`, and `tensor_diff.only_in_reference ⊇ {model.layers.#.self_attn.{q,k,v}_proj.bias}` | true |
| C14 | unrelated weight: `weight.n_with_r == 30` (32 − 2 edge blocks; reldepth maps subject 1..30 into reference 1..26), **`weight.median_r <= 0.40`** and **`weight.z_shift <= 3.0`** (not null) | true |
| C15 | separation: `median_r(related) - median_r(unrelated) >= 0.40` | true |
| C16 | suggestion for the Instruct Card with pool = the other two Cards in the out dir: top-1 is `Qwen/Qwen2.5-7B@pin`, with `tokenizer_jaccard >= 0.90`, `config_score >= 0.80` and `claimed_base_match == true` | true |
| C17 | the diffs and suggestions are computed by a separate server whose CountingTransport records 0 requests; both DiffViews and the suggestion result carry `disclaimers == DISCLAIMERS` | true |
| C18 | related in-run null: **`weight.median_r − weight.offdiag_median_r >= 0.40`** and **`weight.diag_top1_frac >= 0.80`** | true |

`--plan-only` (E2) writes no Card. It evaluates only C0, C1 and a `P plan` check (`bytes_planned > 0` and `bytes_cap == bytes_planned + 16 MiB`). C2–C18 are evaluated in E3.

The EXIT evidence line records, per pair, `median_r`, `q10_r`, `offdiag_median_r`, `diag_top1_frac`,
`z_shift`, the shift-null SD and both participation ratios, whether or not a check uses them.

C15 follows from C12 and C14. It is kept as its own row so the evidence line states the margin.

**Why these thresholds (first principles plus fixture measurements; never fitted to the targets; see §5 D20
for the statistic and D32 for the in-run null).** For each layer, the statistic is the Pearson r between two
vectors. Each vector holds the *depth-detrended, per-layer-centred log singular values of the top half of the
spectrum* of that layer's `v_proj` and `k_proj`. The first and last block of each stack are excluded from the
detrend fit and from every summary (`edge_blocks_excluded = 1`, §3.4; validation r2 major). Only the interior
("scored") layers enter `median_r`, `q10_r`, the off-diagonal median, top-1, the shift null and the PR.

- **Related pair, median (X = 0.80).** A fine-tune changes W by ΔW. By Weyl, `|Δσ_k| ≤ ‖ΔW‖₂`, so on the top
  half of the spectrum (σ_k ≥ σ_median) each log σ_k moves by at most `‖ΔW‖₂/σ_median`. Write s for the RMS of a
  layer's residual vector (its layer-specific spectral structure) and ε for the RMS of the fine-tune perturbation
  in the same space. Only one side is perturbed (base = x, instruct = x + e). If e is uncorrelated with x, then
  `r ≈ s/√(s²+ε²)` (corrected in r2; the r1 text used `s²/(s²+ε²)`, which holds only when both sides are
  independent noisy copies). X = 0.80 therefore tolerates a perturbation up to `ε ≤ 0.75 s`.
- **Related pair, lower tail (q10 ≥ 0.50).** r = 0.50 is `ε = √3·s ≈ 1.73 s`: the perturbation is well above the
  structure. With numpy's linear quantile over the 26 scored layers, q10 sits at position 2.5, between the 3rd
  and 4th smallest value, so about 3 of 26 layers may fall below 0.50 before C12 fails. A broken role or depth
  band longer than that cannot hide behind the median (validation r1, minor 3). Both thresholds are therefore
  more lenient than the r1 text claimed. They are kept unchanged: Instruct post-training (SFT + RL) can move
  weights far, the Weyl bound is loose, and C18 (gap and top-1) is the check that separates lineage from shared
  structure. This decision is made now, before the freeze.
- **Unrelated pair, absolute (Y = 0.40), with a measured d_eff.** The statistic removes the generic structure
  (relative-rank grid, per-layer centring, cubic depth detrend fitted on the interior blocks), so under
  independence `E[r] = 0`. The per-layer null r of a d-dimensional residual has SD ≈ `1/√(d−1)`. The median of n
  layers has SD ≈ `1.2533·SD_r/√n`. **The √n assumes that the per-layer residuals are independent over depth.**
  Real residuals are autocorrelated over depth, so the effective n is smaller than the number of scored layers,
  and this formula-SD is optimistic for real models (validation r2 major, part 3). The in-run shift null (next
  bullet) is the check that does not rely on it. d_eff is measured as the participation ratio `PR = (Σλ)²/Σλ²`
  of the interior residual layer vectors. It is reported in every DiffView and in the evidence line, and it is
  never thresholded. With the fixture's PR (3.76 at exit geometry, 3.14 at L = 16), the formula gives a median
  SD of 0.138 at exit geometry (n = 30) and 0.229 at L = 16 (n = 14). The measured SDs are 0.125 and 0.197. At
  exit geometry Y = 0.40 sits 2.9 formula-SD (3.3 measured SD) above the null mean. At L = 16 it sits only 1.7
  formula-SD above it, which is why the offline L = 16 test uses a 95th percentile and the rehearsal uses the
  exit depths (28 vs 32).
- **Unrelated pair, relative (z_shift ≤ 3.0).** Y assumes that real independent models are as independent at
  matched relative depth as the fixture. That assumption is tested in the run itself. The cyclic-shift null
  takes the same median over the n−1 deliberately misaligned pairings `i ↦ a((i+k) mod n)` of the scored layers
  of the same two models. Its spread therefore reflects the run's actual d_eff and depth autocorrelation.
  `z_shift = (median_r − mean)/sd` ≤ 3 is one-sided, because the failure mode is a spuriously high r. The
  measured false-FAIL rate on the fixture null at exit geometry is 1/200 (max z 3.04). If real unrelated models
  have a PR near 2, the absolute Y is only about 1.8 SD from the null. C14 may then FAIL, and the FAIL goes to
  the human with the PR in the evidence. It is never answered by retuning.
- **Shared depth-localised anomalies (validation r2 major).** Real decoder LLMs have strongly anomalous first
  and last blocks, and the anomaly points the same way in every model. That is shared structure with no lineage
  in it. A least-squares cubic detrend over all L blocks does not remove an endpoint spike. Through the endpoint
  leverage (h₀₀ ≈ 0.57 for a cubic at L = 28) it spreads the spike into every layer's residual as the same depth
  pattern in both models, which raises the aligned r and z_shift. The r2 statistic fits the cubic on blocks
  1..L−2 only, applies it to all blocks, and scores only the interior layers. Interior residuals are then
  exactly independent of whatever blocks 0 and L−1 contain, so a shared edge anomaly of any size leaves every
  summary identical to within float rounding (tested to 1e-12, T109). Residual risk: an anomaly that also covers block 1 (or L−2) still leaks
  (measured below). A robust bisquare detrend closed that leak on the fixture, but it adds tuning constants; it
  is listed in later.md for P3, where real null pairs exist to measure it on.
- **Related pair, in-run null (C18: gap ≥ 0.40, top-1 ≥ 0.80).** This closes validation r1 major 1. The
  related pair shares architecture, so a high diagonal r alone cannot separate lineage from structure shared
  by the architecture. Structure common to every layer raises `R[i, j]` for all j, not only `j = a(i)`, so it
  shrinks `median_r − offdiag_median_r`. A gap of 0.40 requires the layer-specific part of r alone to be as
  large as the whole unrelated bound. `diag_top1_frac ≥ 0.80` requires that 80 % of scored subject layers find
  their own aligned partner as the best of all interior reference layers. The 20 % allowance covers real
  neighbouring layers that resemble each other. A low-dimensional residual, where many layers correlate at ±1,
  fails top-1.
- **Margin.** `X − Y = 0.40` (C15).

**Measured on the fixture design (planner, revision r2).** These values come from an independent numpy
reimplementation of the T103 generator and the T109 statistic (scratch code under /tmp/p2r2, not in the repo;
spectra are generated directly, 7 roles drawn per layer as in T103). The rng stream differs from T103's
tensor order, so the values are distributional. "e" is `edge_blocks_excluded`; e = 1 is the frozen r2
statistic, e = 0 is the r1 statistic. "Edge anomaly Δα" adds Δα to the power-law exponent of every role's
spectrum at the listed blocks, in **both** models of every pair (a shared, lineage-free anomaly; T103
`edge_anomaly`). Exit geometry = 28-layer qwen2-style (k/v 64×128) vs 32-layer llama-style (hidden 160, k/v
128×160), reldepth, 200 pairs `(2p+1, 2p+2)`.

| Setting | e | median_r: mean / SD / q95 / max / # > 0.40 | z_shift: max / # > 3 | top-1 max | PR mean |
|---|---|---|---|---|---|
| exit geometry, no anomaly | 0 | −0.005 / 0.118 / 0.171 / 0.351 / 0 | 2.95 / 0 | 0.12 | 3.81 |
| exit, blocks 0 and L−1, Δα = 0.6 | 0 | +0.184 / 0.124 / 0.380 / 0.491 / 7 | 4.70 / 16 | 0.22 | 2.62 |
| exit, blocks 0 and L−1, Δα = 1.0 | 0 | +0.330 / 0.117 / 0.518 / 0.579 / 56 | 4.70 / 25 | 0.22 | 1.78 |
| exit, blocks 0 and L−1, Δα = 2.0 | 0 | +0.604 / 0.088 / 0.722 / 0.804 / 195 | 3.11 / 5 | 0.28 | 1.21 |
| exit, block 0 only, Δα = 1.0 | 0 | +0.153 / 0.125 / 0.340 / 0.457 / 4 | 3.88 / 4 | 0.22 | 2.42 |
| **exit, no anomaly** | **1** | **−0.010 / 0.125 / 0.201 / 0.358 / 0** | **3.04 / 1** | **0.17** | **3.76** |
| exit, blocks 0 and L−1 (any Δα: 0.6, 1.0, 2.0) or block 0 only | 1 | identical to the row above (to float rounding) | identical | identical | identical |
| exit, blocks 0, 1 and L−1, Δα = 1.0 (residual risk) | 1 | +0.171 / 0.134 / 0.380 / 0.511 / 8 | 3.67 / 9 | 0.20 | 2.37 |
| same, e = 0 | 0 | +0.437 / 0.108 / 0.602 / 0.660 / 135 | 4.61 / 38 | 0.25 | 1.72 |
| exit, first 20 pairs (T109 geometry test), no anomaly | 1 | −0.001 / 0.154 / 0.249 / 0.265 / 0 | 2.77 / 0 | 0.17 | 3.76 |
| exit, first 20 pairs, blocks 0 and L−1, Δα = 1.0 (T109 sensitivity test) | 0 | +0.323 / 0.119 / 0.457 / 0.561 / 5 | 3.69 / 2 | 0.19 | 1.79 |
| L = 16 same architecture, 200 pairs, no anomaly | 1 | −0.023 / 0.197 / 0.299 / 0.535 / 6 | 4.00 / 4 | 0.29 | 3.20 |
| L = 16, first 50 pairs (`NULL_SEED_PAIRS`), no anomaly or edge anomaly | 1 | +0.008 / 0.171 / 0.281 / 0.405 / 1 | 2.58 / 0 | 0.29 | 3.14 |
| L = 16, first 50 pairs, blocks 0 and L−1, Δα = 1.0 | 0 | +0.566 / 0.102 / 0.721 / 0.754 / 47 | 3.81 / 9 | 0.38 | 1.53 |
| L = 16, first 10 pairs, no depth detrend (`degree=None`) | 1 | mean 0.916 | – | – | – |

Related pairs and structural variants, e = 1 (50 seeds each, real SVD of 64×128 float32 matrices):

| Setting | Result |
|---|---|
| fine-tune ε = 0.02 / 0.05, L = 16 | median ≥ 0.9999 / ≥ 0.9992; q10 ≥ 0.9997 / ≥ 0.9979; offdiag −0.240…+0.012; gap ≥ 0.988; top-1 = 1.0 in all; z_shift ≥ 2.60 |
| fine-tune ε = 0.02 / 0.05, L = 28 | median ≥ 0.9999 / ≥ 0.9993; offdiag −0.107…−0.006; gap ≥ 1.006; top-1 = 1.0; z_shift ≥ 4.63 |
| upscaled `upscale(A,16,12,4)`, true map, 10 seeds | scored-layer median 0.941–0.989 |
| pruned `prune(A,16,[5,6,7])`, true map, 10 seeds | scored-layer median 0.872–0.993 |

Alternatives measured in r2 and not adopted (100 pairs, exit geometry): a robust layer-weighted bisquare IRLS
detrend (c = 4.685·MAD, 10 iterations) with e = 1 gives no anomaly: 0/100 > 0.40, z > 3 in 2/100; blocks 0, 1
and L−1 at Δα = 1.0: 0/100 and 1/100. It closes the 3-block leak, but it adds two tuning constants and an
iteration, so it goes to later.md for P3. e = 2 gives SD 0.150 (n = 28) and removes the 3-block anomaly; it
widens the null at L = 16 (n = 12, max 0.553), so it was not adopted either.

Option (a) of r1 (a per-layer log-rank detrend) was measured then and rejected (PR 3.97, max 0.53 at L = 16).
The off-diagonal median of a related pair is slightly negative because the depth detrend makes each grid
column sum to about zero over the fitted layers. The fixture's fine-tunes are easy (r ≈ 1), so the fixture
cannot validate X. X rests on the Weyl argument above, and a real-run C12 FAIL goes to the human.

The product UI shows r on a continuous colour scale. These thresholds exist only in the
exit check. P2 renders no verdict, because the calibrated verdict belongs to System 1 in P3.

### 2.4 E5 human visual confirmation

1. Run `scout serve` and scan the three `repo@pin` targets with "weight sample" and
   "tokenizer" ticked.
2. Record per target:
   - The gate panel appears before any weight byte, and the counter shows `weight 0 B`.
   - The panel shows the plan_id, the planned bytes (exact and human-readable), the hard cap,
     `disk 0 B`, peak memory, the reason and a per-file table.
   - After Confirm, the weight counter rises to exactly the planned bytes.
   - Decline on a fourth scan of the Instruct repo ends in an error card with 0 weight bytes.
3. In the Diff panel, pick reference Qwen2.5-7B.
4. Click "Suggest" for subject Qwen2.5-7B-Instruct. The top suggestion must be
   Qwen2.5-7B, and the three disclaimers are shown under the list.
5. Diff Qwen2.5-7B vs the Instruct model:
   - two strips of 28 cells, with straight alignment lines
   - the 26 interior subject cells in the high-r colour; the first and last cell grey hatched ("edge block, not scored")
   - the median shown, with the in-run null line (q10, off-diagonal median, top-1, z_shift, PR)
6. Diff Qwen2.5-7B vs Llama:
   - the header reads "incompatible (relative-depth pairing)"
   - the cells are in the low-r colour
   - the bias tensors are listed as only in the reference
7. All three disclaimers are visible in each diff.

### 2.5 Environment and access

| Step | Needs | This container |
|---|---|---|
| E0, all implementation | PyPI (numpy, pyarrow, httpx, pyyaml, pytest) | available |
| E1 | `huggingface.co` git + API, and HF_TOKEN with the Llama 3.1 licence accepted | blocked (proxy 403 on huggingface.co) |
| E2 | as E1, plus `*.hf.co` CDN (`cdn-lfs*.hf.co`, `cas-bridge.xethub.hf.co`) for header ranges; about 30 MB meta+header | blocked |
| E3, E4 | as E2, plus about 1.0 GB of weight ranges, about 600 MB peak RAM and about 1 min of SVD CPU | blocked |
| E5 | a browser on the exit host | n/a |

These limits are not blockers. The offline suite covers every code path with FakeHub and
synthetic real-valued checkpoints. E1–E5 run on a host with Hub access, and the P2 PIN line
and EXIT line must come from the same host.

## 3. Definitions

### 3.1 Stages used in P2
The scan runs RESOLVE > HEADERS > META > SAMPLED_READ > COMPUTE > REPORT > PURGE (Phase 1
`Stage` enum, order enforced).

- **META** additionally reads the tokenizer file when `tokenizer=True`. That is meta-class
  bytes, counted against the 64 MiB non-weight threshold.
- **SAMPLED_READ** first builds the `SamplePlan`, then runs the gate (§3.2), and reads weight
  ranges only under an active grant.
- **COMPUTE** decodes each buffer, runs the SVD, and pops and releases the buffer
  immediately. It ends with a `purge` event noting `resident=0`.
- **REPORT** writes the Cards.
- **PURGE** follows REPORT. It checks the live `SampleBuffers` object, which is deliberately kept
  (empty) until then: if `resident_bytes != 0` or any name remains, it clears the buffers, rolls
  back the Cards and raises `SigmaError`. Otherwise it emits `purge` with the note
  `PURGE check: resident=0 names=0` in the live log. The note states what was checked; there is
  no "verified" wording (validation r1, minor 2).

Nothing weight-bearing exists after COMPUTE (invariant 1). FULL_DOWNLOAD is never entered
in P2, because full downloads belong to P4. COMPARE is not a scan stage: a diff reads two
local Cards and fetches 0 bytes.

### 3.2 Download gate (invariant 2)
The gate protects weight reads. Four definitions:
- **`bytes_planned`** is Σ `(end − start)` over the planned reads.
- **`bytes_cap`** is `bytes_planned + RETRY_SLACK_BYTES`, with slack = 16 MiB. It is the
  hard bound including retries.
- **`disk_bytes`** is always 0.
- **`memory_peak_bytes`** is `bytes_planned + 8 × max numel` (the float64 working copy).

The flow:
1. The **full plan text** (`format_plan`: plan_id, reason, bytes planned, hard cap, `disk: 0 B`,
   memory peak, threshold, per-file table, skipped units) is logged as a `gate` event, and it is
   passed to the interface's display (stderr for the CLI and the exit runner). This happens before
   any decision, on every path. The server's live log and `plan` field carry the same plan, and the
   UI renders it (validation r1 blocker, part a).
2. If `totals.meta + totals.header + bytes_cap <= gate_threshold_bytes` (default 64 MiB), the plan is
   auto-approved (`via "below-threshold"`, logged). The non-weight bytes already fetched by this scan count
   against the same threshold, so a scan fetches at most `gate_threshold_bytes` in total without a human
   confirmation (validation r2, minor 2). There are two configured limits, and they do different jobs:
   - the P1 non-weight budget (`--max-read-bytes`, 64 MiB, P1 D5) is **refuse-only**: a non-weight read
     over it is never confirmable and fails the scan;
   - the weight gate threshold (`--gate-threshold-bytes`, 64 MiB) decides only whether a weight plan needs
     confirmation, and it is compared with the scan's **total** (non-weight so far plus the weight cap).

   In r1 the two were independent, so one scan could fetch about 128 MiB unconfirmed. Now the unconfirmed
   total per scan is at most `gate_threshold_bytes`. That is "the configured threshold" of invariant 2.
3. Otherwise a `Confirmer` must return an approval with `plan_id == plan.plan_id` and
   `bytes_confirmed == bytes_cap` **exactly**. Anything else is a decline. A blanket number larger
   than the cap declines, so the number can only come from a plan that was displayed (blocker,
   part b).
4. On approval, `ByteLog.grant(plan_id, ranges, cap)` opens the grant.
   `ByteLog.preflight` then allows a weight range only if all of these hold:
   - the stage is SAMPLED_READ
   - the range is pure weight and lies inside one granted range
   - weight received since the grant, plus reserved weight, plus the request, is
     `<= bytes_cap`

   Retries call `ByteLog.recheck(reservation)` before every attempt after the first, so
   retries stay under the cap too.
5. The grant is revoked when the reads end (in a `finally` block) and on any stage change.
6. A decline raises `GateDeclined` (exit code 6). No weight byte is requested and no Card
   is written.

Confirmers by interface:

| Interface | Display | Confirmation |
|---|---|---|
| CLI | `format_plan` on stderr before the decision | `--confirm-plan PLAN_ID --confirm-bytes CAP`, both exact (`via "cli-flag"`). Without them the scan prints the plan, declines and prints the two values to rerun with. There is no interactive prompt. |
| UI / API | gate panel from `GET /api/scans/{id}` `plan`; plan text in the live log | `POST /api/scans/{id}/confirm` with `plan_id` and `bytes_confirmed == bytes_cap` (`via "api-confirm"`) |
| UI decline | as above | `POST .../decline`, or a 900 s timeout (`via "api-decline"` / `"timeout"`) |
| Exit runner | `format_plan_dict` on stderr for every plan | `--confirm-plans ID:CAP,...`: a plan is confirmed only if its exact pair is listed (blocker, part c) |

The TTY prompt of round 0 was cut: it was a third confirmation path that had to stay consistent
with the other two (validation r1, minor 8).

Every decision is stored in `fetch_log.gate` of the Card(s) when a Card is written. It is
always in the live log.

### 3.3 Sampled read (what, how, how much)
The sample covers the largest stack (`structure.stacks[0]`, depth ≥ 8) of each unit.

- **Roles.** Roles are taken in the order `attn.v`, then `attn.k`. A role matches rel name
  `self_attn.v_proj.weight` or `attn.v_proj.weight` or `attention.v_proj.weight`, and
  likewise for k. The rel name must exist in every block, be 2-D, have dtype BF16/F16/F32
  and have `min(shape) >= 16`. When a role is not found, the skip note names the architecture
  reason if it can be seen in the rel names: `MLA attention` (`kv_a_proj_with_mqa` / `kv_b_proj`,
  DeepSeek-style) or `fused QKV` (`qkv_proj`, `query_key_value`, `c_attn`). Both are listed in
  later.md.
- **Per-layer cap.** A role is included only while the per-layer total stays `<=
  MAX_SAMPLE_BYTES_PER_LAYER = 16 MiB`.
- **Reads.** Each tensor is read **whole**, with one Range request
  `[8+N+data_begin, 8+N+data_end)`.

Row-block sampling is rejected: a contiguous block of rows is not invariant to permuting
output units or heads, whereas the singular values of the full matrix are invariant to
`W → P W Q` for any orthogonal P, Q (permutations included). That is invariant 4.

Per-layer cost at plan time:

| Model | Per layer | Per model |
|---|---|---|
| Qwen2.5-7B | 2 × 3,670,016 B = 7 MiB | 196 MiB |
| Llama-3.1-8B | 2 × 8,388,608 B = 16 MiB | 512 MiB |

### 3.4 Sigma curve and per-layer r (the statistic; frozen in `SIGMA_PARAMS`)
Computing each sampled tensor's sigma curve (T107):
1. Decode the tensor. BF16 is decoded as `uint16 << 16 → float32`; the result is cast to
   float64.
2. Compute `σ = svd(W, compute_uv=False)`, sorted descending.
3. Store the full σ in the Parquet column `stat_sigma_curve`.

Comparing two Cards on their sampled stack (T109):
1. **Log curve on a grid.** For each role and layer, map the curve onto the grid
   `u_g = 0.5·(g+0.5)/128`, g = 0..127, so only the top half of the spectrum is used.
   Evaluate `y = log(max(σ_k, 1e-12·σ_0))` with linear interpolation over the points
   `u_k = (k+0.5)/K`.
2. **Centre each layer.** Subtract each layer's mean over g. This removes the per-layer
   scale.
3. **Detrend over depth, edge blocks excluded from the fit.** Let e = `edge_blocks_excluded` = 1. For each
   role and grid column, fit a least-squares cubic in relative depth `t = linspace(-1, 1, L)` **on rows
   e..L−1−e only**, and subtract it from all L rows. This removes the smooth depth structure common to all
   LLMs without letting the anomalous first and last blocks lever the fit (validation r2 major; §2.3). It
   requires `L >= 8` (so at least 6 fitted rows).
4. **Layer vector.** Concatenate the roles common to both Cards, in role order.
5. **Correlation matrix.** Compute `R[i, j] = pearson(subject_i, reference_j)`. A
   zero-variance vector gives r = 0.
6. **Scored layers and per-layer r.** R is computed for all layers (the alignment of §3.5 may use every
   entry). The **scored set** is `S = {i : e ≤ i ≤ nB−1−e and e ≤ a(i) ≤ nA−1−e}` (subject positions, in
   order); the interior reference columns are `J = e..nA−1−e`. Per-layer r is `R[i, a(i)]` for `i ∈ S` and
   null for every other layer. `median_r` is the median over S.
7. **In-run null (D32, T109 `in_run_null`).** On the same R, the same displayed alignment and only the
   scored set S (n = |S|) with the interior columns J: `q10_r` (10th percentile of the per-layer r),
   `offdiag_median_r` (median of `R[i, j]`, i ∈ S, j ∈ J, `j != a(i)`), `diag_top1_frac` (share of i ∈ S
   with `argmax_{j∈J} R[i, j] == a(i)`), the cyclic-shift null (`m_k = median_x R[S_x, a(S_{(x+k) mod n})]`
   for k = 1..n−1; mean, SD with ddof 1, max) and `z_shift = (median_r − mean)/SD`, plus the participation
   ratio of each side's interior residual layer vectors (rows e..L−1−e). None of these is a verdict; only
   the exit check applies thresholds to them.

No raw cosine is computed anywhere in P2 (invariant 4).

### 3.5 Structural diff and alignment (frozen in `ALIGN_PARAMS`; T110)
**Structural cost between blocks.** Each block's signature set is `{(rel, dtype, shape)}`,
built from Parquet rows. The cost `S[i, j]` between subject block i and reference block j
is:
- 0 when the sets are equal
- 0.5 when the rel-name sets are equal but some shape or dtype differs (width change)
- 1 otherwise

**Stack pairing.** Stacks with equal prefixes are paired first. Then the largest unpaired
stack of each side is paired.

**Alignment by pair type.**
- **Incompatible pair** (`S == 1` everywhere). There is no structural alignment. The mapping
  is by relative depth, `a(i) = floor(i·(nA−1)/(nB−1) + 0.5)`, with `mode "reldepth"`.
- **Any other pair.** Viterbi segment alignment is used. Each subject position is assigned
  one reference position. The cost is:
  - `C[i, j] = S[i, j]` when there is no weight evidence, and
    `C[i, j] = (S[i, j] + (1 − R[i, j])/2)/2` when there is
  - 0 for a transition `j → j+1`, and `λ = 2.0` for any other transition (a jump backward
    means duplication/upscaling, a jump forward means pruning)
  - `end_cost = 2.0` when `a(0) != 0` or when `a(nB−1) != nA−1`

  Ties are broken toward the continuation `j−1`, then toward the smaller j.

**Outputs.** From the mapping:
- `duplicated_reference` lists the reference positions used more than once.
- `pruned_reference` lists the reference positions never used.
- `jumps` lists the discontinuities.
- `kind` is `same` (nA == nB and identity), `upscaled` (nB > nA), `pruned` (nB < nA),
  `rearranged` (otherwise) or `incompatible`.
- `width_changed` is true when any matched cost is 0.5.
- `location_resolved` is true when weight evidence was used, or when the mapping has no
  jumps and no end costs.

A homogeneous stack's header alone cannot locate a duplicated or pruned segment, so the
diff says "location unresolved" instead of guessing.

### 3.6 Tokenizer MinHash and reference suggestion (T104, T105)
**Tokenizer file.** The file is taken from the first of these that exists:
1. `<unit dir>/tokenizer.json`
2. for a pipeline component `text_encoder<sfx>`, `tokenizer<sfx>/tokenizer.json`
3. `<unit dir>/vocab.json`

Only one file is read, with class meta. The cost at plan time is about 7.0 MB for
Qwen2.5-7B's `tokenizer.json` and about 9.1 MB for Llama-3.1-8B's. The exact size comes from
the API siblings and is logged. The fetch is opt-in (`tokenizer=True`), so plain P1 scans
are unchanged.

**What is hashed.** One shingle is one whole vocabulary entry, not a character k-gram.
Character shingles would make unrelated byte-level BPE vocabularies look alike. The element
set is:
- `"v:" + token` for every key of `model.vocab` (BPE/WordPiece/WordLevel) or every piece of
  Unigram `model.vocab`
- `"a:" + content` for every `added_tokens[*].content`

Merges are excluded, because for byte-level BPE they are implied by the vocabulary.

**Hashing.**
- `x = uint32` of `blake2b(element_utf8, digest_size=4)`.
- 256 permutations: `h_i(x) = (a_i·x + b_i) mod (2^61−1)`. The coefficient `a_i` is
  `uint32(blake2b(b"scout.minhash.a.%d" % i, 4)) | 1` and `b_i` is
  `uint32(blake2b(b"scout.minhash.b.%d" % i, 4))`. Since `a_i·x + b_i < 2^64`, the value
  fits in uint64 with no overflow.
- The signature is the per-permutation minimum. The Jaccard estimate is the mean equality
  of two signatures, with SE ≤ 0.5/√256 = 0.031.

**Config match.** The config score is `equal / compared` over the keys
`CONFIG_MATCH_KEYS` present in both configs. Keys are looked up in `config.raw`, then in
`config.raw["text_config"]`, and the score is null if fewer than 3 keys are compared.

**Suggestion.**
- The score is `0.5·J + 0.5·config_score`, where a null value counts as 0.
- The pool is every Card under the out dir except the subject.
- The result is the top 5, with ties broken by repo, then sha.
- The model card's claimed `base_model` never changes the score. It is shown as
  `claimed_base_match`. (`unscanned_claims` was cut in r1, minor 8.)
- The result carries `disclaimers == DISCLAIMERS`, and every interface renders all three next to
  the ranking (invariant 5; validation r1, minor 6). The result is
  `{subject: card_key, suggestions: [...], disclaimers: [str, str, str]}`.

## 4. Data structures (exact field names)

### 4.1 LogEvent (P1 §4.1, unchanged fields)
New `event` values:
- `gate`: 0 bytes. The note is the plan summary or the decision.
- `purge`: 0 bytes. The note is `released <n> weight bytes; resident=0` in COMPUTE, and
  `PURGE check: resident=0 names=0` in PURGE.

`ByteLog.note()` accepts `error`, `refused`, `card_written`, `gate` and `purge`.

### 4.2 Card v1 JSON (changes relative to P1 §4.2 only)
```
schema_version: "card.v1"
scan:      {scanned_at, scout_version, elapsed_s,
            options: {sample: bool, tokenizer: bool},        # NEW
            notes: [str]}                                    # NEW (e.g. "sigma: no sampleable roles in visual.blocks")
stats:     {tensor_stats: null, spectral_topk: null, attribution: null,     # unchanged (P3/P4)
            sigma_curves: null | SigmaCurves,                                 # FILLED when sample=True
            tokenizer_minhash: null | TokenizerMinHash}                       # FILLED when tokenizer=True
fetch_log: {counting, threshold_bytes, totals, events,
            gate: [GateDecision]}                                             # NEW ([] for plain scans)

SigmaCurves = {version: "sigma.v1", method: "full-tensor-svd", svd_dtype: "float64",
  stack_prefix: str, roles: {role: rel_name}, plan_id: str, bytes_read: int,
  tensors: [{name: str, role: str, block_index: int, file: str, range: [start, end_exclusive],
             dtype: str, shape: [m, n], n_sigma: int, sigma_max: float, sigma_sum_sq: float}]}
TokenizerMinHash = {version: "minhash.v1", source_file: str, source_bytes: int, source_sha256: str,
  tokenizer_type: "BPE"|"Unigram"|"WordPiece"|"WordLevel"|"vocab.json", n_vocab: int, n_added: int,
  n_elements: int, num_perm: 256, element_scheme: "v:<token>|a:<added content>",
  hash: "blake2b-32;(a*x+b) mod 2^61-1", signature: [str]}   # 256 lowercase 16-hex strings
GateDecision = {plan_id: str, approved: bool, via: "below-threshold"|"cli-flag"|
  "api-confirm"|"api-decline"|"timeout"|"no-confirmer"|"error", bytes_planned: int, bytes_cap: int,
  bytes_confirmed: int|null, threshold_bytes: int, disk_bytes: 0, memory_peak_bytes: int,
  n_reads: int, reason: str, decided_at: str (ISO-8601 Z)}
```
A plain scan (no sample, no tokenizer) still has every `stats` value null. The Phase 1 exit
check C7 is therefore unaffected.

`load_card` accepts `card.v0` and `card.v1`. Consumers read the new keys with `.get`, so a
v0 Card behaves like a v1 Card with neither option set.

### 4.3 Card v1 Parquet
The columns are unchanged from P1 §4.3.
- `stat_sigma_curve` (list<float64>) holds the full descending σ for sampled tensors and
  stays null elsewhere.
- The other `stat_*` columns stay null.
- The metadata `schema_version` is `card.v1`.

### 4.4 SamplePlan (served in the API and printed by the CLI; not persisted, except via GateDecision)
```
{plan_id: str(16 hex = sha256(json [repo, sha, [[path,start,end]...]])[:16]), repo, revision_sha,
 reads: [{component: str|null, path, tensor, role, block_index, start, end_exclusive, dtype, shape}],
 per_file: [{path, n_reads, bytes}], units: [{component, stack_prefix, roles: [str], depth,
 bytes_per_layer_max}], skipped: [{component, reason}], bytes_planned, bytes_cap, disk_bytes: 0,
 memory_peak_bytes, threshold_bytes, reason: str}
```

### 4.5 DiffView (served, never persisted; recomputed from two persisted Cards; T111)
```
{schema: "diff.v1",
 reference: {card_key, title}, subject: {card_key, title},
 params: {sigma: SIGMA_PARAMS, align: ALIGN_PARAMS},
 config_diff: {score: float|null, compared: int, architectures_equal: bool,
               rows: [{key, reference, subject, equal: bool}]},
 tensor_diff: {only_in_reference: [str], only_in_subject: [str],
               shape_changed: [{name, reference, subject}], dtype_changed: [{name, reference, subject}],
               truncated: bool},                    # each list capped at 200, sorted
 stack_pairs: [{reference_prefix, subject_prefix, reference_depth, subject_depth,
                kind: "same"|"upscaled"|"pruned"|"rearranged"|"incompatible", width_changed: bool,
                mode: "dp-structure"|"dp-structure+weights"|"reldepth", location_resolved: bool,
                jumps: [{subject_pos, from_reference_pos, to_reference_pos}],
                duplicated_reference: [int], pruned_reference: [int],      # literal block indices
                layers: [{subject_index, reference_index, op: "match"|"jump"|"reldepth",
                          struct_cost: float, r: float|null}]}],   # r null outside the scored set (§3.4 step 6)
 unpaired_stacks: {reference: [str], subject: [str]},
 weight: {available: bool, reason: str|null, roles: [str], pair_index: int|null,
          n_layers: int, n_with_r: int,                              # n_with_r == |scored set|
          median_r: float|null, q10_r: float|null,                  # over the displayed alignment a
          offdiag_median_r: float|null,                             # median of R[i, j], j != a(i)
          diag_top1_frac: float|null,                               # share of i with argmax_j R[i, j] == a(i)
          shift_null: {n: int, mean: float, sd: float, max: float}|null,   # medians over i -> a((i+k) mod n), k = 1..n-1
          z_shift: float|null,                                      # (median_r - shift_null.mean) / shift_null.sd
          participation_ratio: {reference: float, subject: float}|null},   # d_eff of the residual layer vectors
 tokenizer: {available: bool, reason: str|null, jaccard: float|null},
 disclaimers: [str, str, str]}                       # == scout.view.DISCLAIMERS
```

### 4.6 Frozen parameters
```
SIGMA_PARAMS = {"version": "sigma.v1", "roles": ["attn.v", "attn.k"],
  "role_patterns": {"attn.v": ["self_attn.v_proj.weight", "attn.v_proj.weight", "attention.v_proj.weight"],
                    "attn.k": ["self_attn.k_proj.weight", "attn.k_proj.weight", "attention.k_proj.weight"]},
  "dtypes": ["BF16", "F16", "F32"], "min_dim": 16, "min_layers": 8,
  "max_bytes_per_layer": 16777216, "retry_slack_bytes": 16777216, "svd_dtype": "float64",
  "grid_points": 128, "grid_u_max": 0.5, "log_floor_rel": 1e-12, "center": "per-layer-mean",
  "detrend_degree": 3, "edge_blocks_excluded": 1, "corr": "pearson"}
ALIGN_PARAMS = {"version": "align.v1", "lambda_jump": 2.0, "end_cost": 2.0,
  "struct_cost": {"equal": 0.0, "same_names": 0.5, "different": 1.0},
  "combine": "(S + (1 - r) / 2) / 2", "incompatible_mapping": "reldepth-round-half-up"}
```

### 4.7 API and CLI surface added
- **CLI flags:**
  - `scout scan TARGET [--sample] [--tokenizer] [--confirm-plan PLAN_ID --confirm-bytes N] [--gate-threshold-bytes N]`
  - `scout diff REF_CARD SUBJ_CARD [--json]`
  - `scout suggest CARD [--out DIR] [--top 5] [--json]`
- **HTTP routes** (T113):
  - `POST /api/scans` now also accepts `{"sample": bool, "tokenizer": bool}`.
  - `GET /api/scans/{id}` adds `plan` and `gate`, and `status` may be `awaiting_confirmation`.
  - `POST /api/scans/{id}/confirm` takes `{plan_id, bytes_confirmed}`; 409 unless `bytes_confirmed == bytes_cap`.
  - `POST /api/scans/{id}/decline`.
  - `GET /api/cards`.
  - `POST /api/diff` takes `{reference: card_key, subject: card_key}`.
  - `GET /api/suggest?repo=&revision_sha=&component=` (the result includes `disclaimers`).
  - `GET /diff.js`.

## 5. Assumptions and open decisions (each with a recommended default)

Assumptions:
- Phase 1 is implemented exactly as specified in T001–T016.
- numpy ≥ 1.26 is installable from PyPI.
- The published configs of the three targets match the plan-time constants quoted in §2.2.
  If they do not, C2 compares against the config at the pin, never against a hard-coded
  number.
- Qwen2.5-7B-Instruct was post-trained from Qwen2.5-7B, as its model card documents.
  Llama-3.1-8B was trained independently.

| # | Decision | Default (recommended) | Rationale |
|---|---|---|---|
| D14 | Candidate pool for suggestion | Cards under the out dir (glob `*/*/*.card.json`, JSON only). No Hub search. The result carries the 3 disclaimers. | "Picked manually, tool suggests candidates". A local glob is not a Card store: no content addressing, no cache and no rescan skipping, all of which are P4. |
| D15 | Tokenizer MinHash scheme | As §3.6: whole-token shingles, 256 permutations, blake2b-32 + Mersenne-61 universal hash, fixed coefficients derived by blake2b (no RNG, so stable across numpy versions) | SE ≤ 0.031. It is deterministic and portable. |
| D16 | Tokenizer fetch | Opt-in (`--tokenizer` / `"tokenizer": true`). One file of about 7–9 MB, class meta. | This keeps the P1 10 s exit path byte-identical. |
| D17 | Config score and ranking | 13 keys: `model_type, hidden_size, num_hidden_layers, intermediate_size, num_attention_heads, num_key_value_heads, head_dim, vocab_size, rope_theta, tie_word_embeddings, max_position_embeddings, hidden_act, rms_norm_eps`. Score `0.5·J + 0.5·config`. Claims never score. | A lying `base_model` cannot steer the ranking. Invariant 5 requires stating that tokenizer reuse alone is not proof, so the UI shows both components separately. |
| D18 | What is sampled | Whole `v_proj` and `k_proj` of every block of the largest stack. One Range read per tensor. | Their σ are exactly permutation- and rotation-invariant, and they are the smallest 2-D matrices per block, 7–16 MiB per layer for 7–8B models. |
| D19 | Byte budget | 16 MiB per layer; slack 16 MiB per plan; gate threshold 64 MiB (`--gate-threshold-bytes`), compared with `totals.meta + totals.header + bytes_cap` (r2), so a scan's unconfirmed total is ≤ 64 MiB; the P1 non-weight budget stays refuse-only (§3.2). On a slack overrun, E3 is rerun whole with the same `--confirm-plans` line. | Llama-3.1-8B fits both roles exactly at 16 MiB. 70B models keep `v_proj` only. Scaling the slack with the largest read is deferred (§11). |
| D20 | Statistic | As §3.4: top half of the spectrum, 128-point log grid, per-layer centring, cubic depth detrend **fitted on blocks 1..L−2 only (`edge_blocks_excluded = 1`, r2)**, Pearson, L ≥ 8; only the scored interior layers enter any summary. An extra per-layer log-rank detrend (r1) and a robust bisquare detrend and e = 2 (r2) were measured and not adopted (§2.3). | See the §2.3 justification. Without the detrend, the fixture's null median is 0.92 (measured). With a shared edge anomaly (Δα = 1.0 on blocks 0 and L−1), the r1 statistic gives a null median > 0.40 in 56/200 exit-geometry pairs; the r2 statistic equals the no-anomaly null (0/200). |
| D21 | Raw cosine | Not computed in P2 | Invariant 4. Nothing needs triage yet. |
| D22 | Alignment algorithm | Viterbi over reference positions (§3.5), λ = 2, end cost 2; relative-depth mapping for incompatible pairs | The segment-copy model covers SOLAR-style duplication (backward jump) and layer pruning (forward jump or end cost). A jump must be justified by ≥ 2 fully mismatched layers of evidence. An incompatible pair is never weight-optimised, so r cannot be inflated by the choice of alignment. |
| D23 | Gate confirmation | The plan is always displayed and logged before the decision. Approval needs `plan_id` equal and `bytes_confirmed == bytes_cap` exactly (CLI `--confirm-plan` + `--confirm-bytes`; API confirm; runner `--confirm-plans ID:CAP,...`). There is no interactive prompt. On decline: `GateDeclined` (exit 6), no Card, and the CLI prints the two values to rerun with. The server waits 900 s, then treats it as a decline. | Invariant 2: "confirmation with bytes, disk and reason shown". A number can approve only the plan it was copied from. |
| D24 | Where weights live | In RAM only, in `SampleBuffers`. Each buffer is popped right after its SVD. A `purge` event with `resident=0` is written at the end of COMPUTE. PURGE re-checks the live (empty) buffers object and fails the scan if anything is resident. Disk is 0. | Invariant 1. Peak RAM ≈ bytes_planned (544 MiB for Llama), which is acceptable for a self-use tool. Streaming per-tensor compute is listed in later.md. |
| D25 | Diff persistence | Not persisted. DiffView is a pure function of two Cards, so the per-layer r values live nowhere but are reproducible from `stat_sigma_curve`. | Invariant 1: the Card is the only persisted artifact. |
| D26 | Card version | `card.v1` (§4.2). v0 stays loadable. | New fields `scan.options`, `scan.notes` and `fetch_log.gate`, plus the filled slots. |
| D27 | Frontend (D1 revisit) | Stay no-build vanilla. Add `scout/web/diff.js`. | The diff is two SVG strips plus lines and tables, so no framework is needed. |
| D28 | Exit thresholds | X = 0.80 (related median ≥), related q10 ≥ 0.50, Y = 0.40 (unrelated median ≤), unrelated z_shift ≤ 3.0, margin 0.40, related gap to off-diagonal ≥ 0.40, related top-1 ≥ 0.80. Frozen in T115 before any real run. | §2.3; the fixture measurements are recorded there, and none of the values was fitted to the targets |
| D29 | P2 pins | `exit/pins_p2.json`; `pin_exit.py --phase 2`; HF_TOKEN passed to git via `GIT_CONFIG_COUNT/KEY_0/VALUE_0 = http.extraHeader` | The token never appears in argv. The P1 pins file is untouched. |
| D30 | Stacks sampled | One per unit (`stacks[0]`) | This keeps bytes minimal. A text encoder's vision tower is not sampled (noted in `scan.notes`). |
| D31 | `GET /api/cards` | Lists Card keys found in the out dir, for the manual reference picker | This is the minimum needed for "picked manually". |
| D32 | In-run null (new in r1) | For the weight pair, DiffView reports `offdiag_median_r`, `diag_top1_frac`, a cyclic-shift null of the median (`shift_null`, `z_shift`) and the residual participation ratios. They are computed from the R already built (0 bytes), and they are deterministic (no RNG, no bootstrap). | Validation r1 majors 1 and 2. The shift null measures the run's own d_eff, and the off-diagonal gap separates lineage from shared architecture. A bootstrap SD was the validator's alternative; the shift null was chosen because it needs no RNG and reuses the median statistic exactly. |
| D33 | Frozen fixture seeds (new in r1) | `NULL_SEED_PAIRS` (50 pairs), `FINETUNE_EPS/SEED`, `EDGE_ANOMALY = 1.0` (r2), `REHEARSAL_BASE/OTHER` (r2: both with `edge_anomaly=EDGE_ANOMALY`, so the end-to-end rehearsal carries a shared edge anomaly) are constants in `weights_fixtures` (T103), asserted literally by a test, and imported by T109/T111/T116. The rehearsal uses the exit depths (28 vs 32). | Changing seeds to pass a test would be hand-tuning. Exit depths shrink the null SD from 0.167 to 0.114 (measured). |

## 6. Architecture (files)

```
pyproject.toml (+numpy), scout/errors.py, scout/bytelog.py, scout/hub.py   T101  (P1 files, extended)
scout/card.py (card.v1)                                                     T102
tests/helpers/weights_fixtures.py                                           T103
scout/tokenizer.py                                                          T104
scout/suggest.py                                                            T105
scout/gate.py                                                               T106
scout/sigma.py                                                              T107
scout/scan.py (P1 file, extended)                                           T108
scout/layercorr.py                                                          T109
scout/align.py                                                              T110
scout/diff.py                                                               T111
scout/cli.py (P1 file, extended)                                            T112
scout/server.py (P1 file, extended)                                         T113
scout/web/index.html, scout/web/app.js (P1), scout/web/diff.js (new)         T114
scripts/exit_expectations_p2.py                                             T115
scripts/exit_check_p2.py, scripts/pin_exit.py (P1, extended), exit/pins_p2.json   T116
```

## 7. Inputs handled and where each is tested (all offline)

| Input | Handling | Test |
|---|---|---|
| Sharded | Planned reads span shards; `per_file` breakdown | T106, T108 `test_sample_sharded_hub` |
| Local folder | Same plan and gate; LocalSource reads the granted ranges | T108 `test_sample_local` |
| Multi-component (`model_index.json`) | One combined plan across components; `text_encoder` tokenizer from `tokenizer/`; components without roles are listed in `skipped` | T106, T108 `test_sample_pipeline`, T104 |
| MoE | Attention k/v are dense, so they are sampled; experts are not sampled | T106 `test_plan_moe` |
| Missing model card | Suggestion works; `claimed_base_match` is false | T105 |
| Lying `base_model` | Never affects the score | T105 `test_claim_does_not_score` |
| Gated repo | GatedRepoError before any plan; HF_TOKEN documented | T108 (P1 path), T116 `test_main_requires_token` |
| Network failure mid-weight-read | Retry within the cap succeeds; beyond the cap → `WeightReadRefused`, no Card | T101, T108 `test_sample_retry_*` |
| Server ignores Range on a weight read | `RangeNotSupported`, counted honestly, no Card | T108 |
| Decline / timeout | `GateDeclined`, 0 weight bytes, no Card | T106, T108, T112, T113 |
| Upscaled (SOLAR-like) / pruned | Alignment with and without weights | T110, T111 |
| Unsupported dtype (F8) / shallow stack | Role or stack skipped with a note; weight evidence unavailable | T106, T109 |
| MLA attention (DeepSeek `kv_a_proj_with_mqa`/`kv_b_proj`), fused QKV | Role skipped; the note names the architecture reason; weight evidence unavailable (later.md) | T106 `test_skip_hint` |

## 8. Risks and how each is tested

| Risk | Mitigation / test |
|---|---|
| A weight byte is read without confirmation | The gate sits in ByteLog preflight and is independent of callers (T101 tests: no grant, wrong stage, mixed range, outside grant, over the cap, after revoke). T108 asserts `fakehub.cdn_reads` has no weight range when declined. Exit C1 and C5 check the wire independently. |
| A plan is approved that nobody saw (validation r1 blocker) | run_gate logs and displays `format_plan` before the confirmer runs (T106 `test_plan_shown_first`). Approval needs the exact cap and plan_id: a blanket larger number, a wrong cap or a wrong plan_id declines (T106 `test_run_gate_flag`, T112 `test_confirm_blanket_number`, T113 `test_gate_confirm_rejects`). The CLI and runner show the plan on stderr before any weight read (T112 `test_plan_printed_before_read`, T116 `test_rehearsal_plan_displayed_first`). A plan missing from `--confirm-plans` is declined (T116 `test_rehearsal_missing_plan`, `test_rehearsal_wrong_cap`). C8 checks that the approved pair was listed. |
| Retries exceed the confirmed bytes | `recheck` before each retry (T101 `test_recheck_cap`); exit C4 |
| Unlogged weight bytes | All weight reads go through `Source.read_range`. Exit C6 compares the wire total with the ByteLog. T116 rehearsal includes a CountingTransport. |
| Weights persisted or kept | `SampleBuffers.resident_bytes == 0` after COMPUTE, and the purge event precedes REPORT (T108). `test_no_extra_files` extends to sampled scans (HOME/TMPDIR empty, out dir holds only Cards). Exit C9. |
| The statistic correlates everything, like a raw σ-curve would | T109 synthetic tests over the 50 frozen `NULL_SEED_PAIRS` (same architecture and a **shared** depth trend): 95th percentile of median r ≤ 0.40 (measured 0.281), z_shift ≤ 3 for ≥ 47/50 (measured 0/50 above 3; 4/200 over 200 pairs, so 47/50 keeps the by-construction failure chance below 2 %), top-1 ≤ 0.40 (measured max 0.29). Without the depth detrend (`residuals(degree=None)`) the mean median is ≥ 0.60 (measured 0.92). Exit C14 (absolute and in-run). |
| Shared architecture passes as lineage (validation r1 major 1) | C18 in the exit: related gap to the off-diagonal median ≥ 0.40 and top-1 ≥ 0.80, computed in-run at 0 bytes. T109 `test_in_run_null_math` shows that uniformly inflated R fails the gap. The true hard negative (same architecture, independent real training) remains a P3 labelled-set item (later.md). |
| Shared depth-localised anomalies (first/last block) inflate the unrelated r (validation r2 major) | The detrend is fitted on blocks 1..L−2 and only interior layers are scored (§3.4). T103 `edge_anomaly` fixture; T109 `test_edge_anomaly_invariance` (L = 16 and exit geometry: every in-run-null output with the anomaly equals the no-anomaly output to 1e-12, and the exit-geometry thresholds hold) and `test_edge_anomaly_sensitivity` (the same pairs with `edge=0` have a mean median ≥ 0.20, measured 0.323: the fixture has teeth). The rehearsal families carry the anomaly end to end (T116). Residual risk: an anomaly that also covers block 1 or L−2 (measured 8/200 > 0.40); a C14 FAIL from it goes to the human, and the robust detrend is in later.md for P3. |
| The null is wider than assumed (validation r1 major 2) | d_eff is measured (participation ratio in every DiffView and the evidence line). The shift null adapts to the run. The derivation in §2.3 uses the measured PR (3.3–3.9 on fixtures, not ≥ 5). |
| The statistic misses real fine-tunes (RL-heavy Instruct moves weights more) | Top-half grid (Weyl), synthetic fine-tunes at ε = 0.02 / 0.05 give ≥ 0.95 / ≥ 0.80, and permuted and rotated copies give r = 1 ± 1e-9 (T109). A real-run FAIL of C12 goes to the human; no retuning (C10). |
| Hand-tuning after the first real run | C10 frozen params; the evidence line records git commit + dirty flag; §2.1 rule |
| Alignment inflates r for unrelated models | Incompatible pairs use the fixed reldepth mapping (D22). T110 `test_no_jump_on_noise`: random R with no structure → identity or reldepth path with 0 jumps for nA == nB. |
| Upscale / prune location guessed from headers | `location_resolved == false` without weights (T110 test). With weights, the SOLAR-like and pruned synthetic families are recovered exactly (T111). |
| Tokenizer J misleads (Qwen and Llama both descend from cl100k-style BPE vocabularies) | J is only half of the suggestion score and never a verdict. Disclaimer 2 sits in every DiffView (exit C17). The exit sets no upper bound on the unrelated J. |
| Llama gated / token missing | Up-front exit 3 with instructions (T116). This is an environment limit and is documented (§2.5). |
| Memory ~544 MiB for Llama | Shown at the gate (`memory_peak_bytes`); later.md has streaming |
| Stage-order conflict (reads must precede compute) | All reads happen in SAMPLED_READ, then COMPUTE; enforced by P1 `StageOrderError`, and the grant is auto-revoked on any stage change (T101) |
| P1 regressions (P1 exit C2 "weight == 0") | Options default off; the P1 offline suite, including `test_exit_check_offline.py`, stays in E0 |

## 9. Changes to Phase 1 contracts

| P1 contract | Change | Task |
|---|---|---|
| `pyproject.toml` (T001) | add `numpy>=1.26` to dependencies | T101 |
| `scout/errors.py` (T002) | add `GateDeclined(ScoutError)` exit_code 6, attr `plan: dict | None`; add `SigmaError(ScoutError)`; `WeightReadRefused` gains optional keyword `reason` (message text unchanged) | T101 |
| `ByteLog` (T002) | new `grant`, `revoke_grant`, `grant_active`, `recheck`, `add_gate_decision`, `gate_decisions`; weight preflight allowed only under a grant in SAMPLED_READ; weight reservations tracked separately from the non-weight threshold; `note()` accepts `gate`, `purge`; `to_card_dict()` adds `"gate"`; `stage()` revokes an active grant when leaving SAMPLED_READ. **Without a grant, behaviour and refusal text are byte-identical to P1; the P1 test files are not edited.** | T101 |
| `HubSource._get` (T005) | calls `log.recheck(reservation)` before every attempt ≥ 2 | T101 |
| Card v0 → v1 (T009, plan P1 §4.2) | `SCHEMA_VERSION = "card.v1"`; `build_card` gains kwargs `options`, `notes`, `sigma_curves`, `sigma_by_tensor`, `tokenizer_minhash`; `load_card` accepts v0 and v1; Parquet metadata `card.v1`; the P1 tests asserting `card.v0` are updated | T102 |
| `scan()` / `ScanResult` (T010) | kwargs `tokenizer=False, sample=False, confirmer=None, gate_threshold_bytes=DEFAULT_GATE_THRESHOLD_BYTES, gate_display=None`; `ScanResult.gate: list[dict]`; new stages SAMPLED_READ, COMPUTE, PURGE when sampling | T108 |
| CLI (T012) | new flags and subcommands (§4.7); exit code 6 | T112 |
| Server (T013) | new routes and status (§4.7); `ScanState.plan`, `ScanState.gate`; `/diff.js` static | T113 |
| Frontend (T014) | options checkboxes, gate panel, diff panel (new `diff.js`) | T114 |
| `scripts/pin_exit.py` (T015) | `--phase {1,2}` (default 1); the P2 target set comes from `exit_expectations_p2.P2_TARGETS`; git gets the HF_TOKEN header via env; `--phase 2` requires HF_TOKEN | T116 |
| P1 exit scripts | unchanged; still pass (plain scans unchanged) | — |

## 10. Task list

| id | title | route | depends_on |
|---|---|---|---|
| T101 | ByteLog weight grant + retry recheck + gate/purge events + GateDeclined + numpy dep | opus | T002, T005 |
| T102 | Card v1 schema (options, notes, gate log, filled slots), v0 compatibility | opus | T009, T101 |
| T103 | Real-valued safetensors fixtures, synthetic model families, tokenizer fixtures | sonnet | T001, T101 |
| T104 | Tokenizer file resolution, parsing, MinHash, Jaccard | opus | T103 |
| T105 | Config match + reference suggestion over the out-dir pool (with disclaimers) | opus | T011, T102, T104 |
| T106 | Sample plan + download gate (plan always displayed, exact-cap confirmation) | opus | T008, T101, T107 |
| T107 | Sigma compute: dtype decode, SVD, SIGMA_PARAMS | opus | T101, T103 |
| T108 | Scan integration: tokenizer, SAMPLED_READ/COMPUTE/PURGE, Card stats fill | opus | T010, T102, T104, T106, T107 |
| T109 | Per-layer statistic: log grid, centring, cubic depth detrend, Pearson matrix, in-run nulls | opus | T102, T103, T107 |
| T110 | Structural cost, stack pairing, Viterbi segment alignment, reldepth | opus | T009, T101, T102 |
| T111 | Diff builder (DiffView) | opus | T011, T105, T108, T109, T110 |
| T112 | CLI: scan gate flags (`--confirm-plan` + `--confirm-bytes`, plan printed first), `diff`, `suggest` | opus | T012, T108, T111 |
| T113 | Server: gate wait/confirm/decline, cards/diff/suggest routes | opus | T013, T108, T111 |
| T114 | Frontend: options, gate panel, diff view (strips + alignment + r colours) | opus | T014, T113 |
| T115 | P2 exit expectations (C1–C18 pure checks, frozen params, thresholds) | opus | T016, T111 |
| T116 | P2 exit runner (`--confirm-plans`), pins_p2, `pin_exit --phase 2`, offline rehearsal | opus | T015, T112, T113, T115 |

Parallel waves (no shared files within a wave):
1. {T101}
2. {T102, T103}
3. {T104, T107, T110}
4. {T105, T106, T109}
5. T108
6. T111
7. {T112, T113, T115}
8. {T114, T116}

Most tasks are opus because they touch download gating, Card schema, similarity math or
exit evidence (routing rule). T103 is the only sonnet task, since it is fully specified test
scaffolding.

## 11. Validation responses

### 11.1 Validation responses — round 1

Validator round 1 (`validation.md`, 2026-09-28): verdict REVISE, with 1 blocker, 3 majors and 8 minors.
The environment limits (no Hub access in this container, gated Llama) were not findings. They stay
documented in §2.5.

| # | Severity | Finding (short) | Response | Where |
|---|---|---|---|---|
| B1 | blocker | `--confirm-bytes N` approves plans nobody saw; runner confirms by running sum | **Fixed.** (a) `run_gate` logs and displays `format_plan` before any decision, on every path. (b) Approval requires `plan_id` equal and `bytes_confirmed == bytes_cap` exactly, everywhere (CLI flag pair, API confirm, runner). (c) E2 prints `CONFIRM WITH: --confirm-plans ID:CAP,...`. E3 confirms a plan only if its exact pair is listed, and C8 asserts that. The TTY prompt was removed. Tests: blanket number declines, wrong cap or plan_id declines, a plan missing from the list declines, and the plan is on stderr before any weight read. | §2.2, §2.3 C8, §3.2, D23; T106, T108, T112, T113, T114, T115, T116 |
| M1 | major | Exit cannot tell lineage from shared architecture | **Fixed.** The off-diagonal null was moved from later.md into P2. DiffView.weight gains `offdiag_median_r` and `diag_top1_frac` (plus `q10_r`, `shift_null`, `z_shift`, `participation_ratio`). New exit **C18**, frozen: related `median_r − offdiag_median_r ≥ 0.40` and `diag_top1_frac ≥ 0.80`. Both values are in the EXIT evidence line. The true same-architecture hard negative stays a P3 item. | §2.3, §4.5, D32; T109, T111, T114, T115, T116; later.md |
| M2 | major | d_eff ≥ 5 is wrong; the null test sits on the edge; seeds could be hand-tuned | **Fixed.** Re-measured in numpy (§2.3 table). PR is 3.25 (L = 16) to 3.90 (L = 32). The null median SD is 0.167 (L = 16) and 0.114 (exit geometry), with max 0.41 in 200 pairs at L = 16. Option (a), a log-rank detrend, was measured and rejected (PR 3.97, max 0.53). Option (b) was adopted: C14 adds `z_shift ≤ 3.0` against the in-run cyclic-shift null (measured 0/600 null pairs above 3; max 2.85). The derivation is rewritten with the measured PR, and PR is reported in DiffView and the evidence line. The seeds are frozen as literals (D33). T109 uses 50 frozen pairs and a 95th percentile ≤ 0.40 (measured 0.28). The rehearsal moves to the exit depths 28 vs 32. Y = 0.40 is unchanged: the measurement did not require changing it, and nothing was fitted to the targets. | §2.1, §2.3, D20, D28, D33; T103, T109, T111, T115, T116 |
| M3 | major | T109 acceptance needs a T108 scan but runs a wave earlier | **Fixed.** `test_sigma_set_from_card` uses a hand-built minimal Card dict and a pyarrow table. The scan-based variant moved to T111 (`test_sigma_set_from_scan`). T109.depends_on += T102, T103. | T109, T111, §10 |
| m1 | minor | T101 edits P1 test files not in `files` | **Fixed.** Without a grant, the refusal text stays exactly P1's. The reason goes in a new `WeightReadRefused.reason` attribute. The P1 test files are untouched. | T101, §9 |
| m2 | minor | PURGE "resident=0 verified" checks nothing | **Fixed.** PURGE checks the live `SampleBuffers` (bytes and names) and fails with rollback if anything is resident. The note is `PURGE check: resident=0 names=0`; C9 checks it. | §3.1, §4.1, D24; T108, T115 |
| m3 | minor | C12 only checks the median | **Fixed.** C12 adds `q10_r ≥ 0.50`, frozen and justified as `ε ≤ s` (§2.3). | §2.3; T111, T115 |
| m4 | minor | 16 MiB retry slack can fail E3 under network faults | **Accepted and documented.** E3 is rerun whole with the same `--confirm-plans` line; that is a fresh human invocation. Slack scaling is deferred (below). | §2.2, D19 |
| m5 | minor | MLA and fused QKV silently fall out | **Fixed.** The skip note names `MLA attention` or `fused QKV`, and a T106 test covers it. MLA was added to later.md. | §3.3, §7; T106; later.md |
| m6 | minor | Suggestion list carries 1 or 0 disclaimers | **Fixed.** `suggest()` returns `disclaimers`; the CLI, API and UI render all three; C17 checks them. | §3.6, §4.7; T105, T112, T113, T114, T115 |
| m7 | minor | T110 test depends on T102's in-flight card.py | **Fixed.** T110.depends_on += T102, and T110 moved to wave 3. | T110, §10 |
| m8 | minor | Simplicity: TTY prompt, `unscanned_claims`, kind `rearranged` | **Fixed in part.** The TTY prompt and `unscanned_claims` were cut. `rearranged` is kept: it is the label for the remaining case `nB == nA` and non-identity, and it has no dedicated test or code path beyond that label. | §3.2, §3.6; T105, T106, T112 |

### 11.2 Validation responses — round 2

Validator round 2 (`validation.md`, fresh context, 2026-09-29): verdict REVISE, with 0 blockers, 1 major and 9
minors. Its clean checks (byte arithmetic, MinHash overflow, null replication, gate and purge against invariants
1–2, Phase 1 references) were not re-opened. Re-measurement scripts: /tmp/p2r2 (scratch, not in the repo).

| # | Severity | Finding (short) | Response | Where |
|---|---|---|---|---|
| M1 | major | Shared first/last-block anomalies lever the cubic detrend and inflate the unrelated median and z_shift; the fixture has none | **Fixed before the freeze.** (1) T103 gains `FamilySpec.edge_anomaly` and the frozen `EDGE_ANOMALY = 1.0` (no extra rng draws, so interior layers are bit-identical); both rehearsal families carry it. T109 adds `test_residuals_edge_fit`, `test_edge_anomaly_invariance` (L = 16 and exit geometry) and `test_edge_anomaly_sensitivity`. (2) The statistic fits the cubic on blocks 1..L−2 only and scores only interior layers (`SIGMA_PARAMS.edge_blocks_excluded = 1`); every summary (median, q10, off-diagonal, top-1, shift null, PR) uses the scored set. (3) Re-measured (§2.3): at exit geometry with Δα = 1.0 on blocks 0 and L−1 the r1 statistic gives 56/200 null medians > 0.40 and 25/200 z > 3; the r2 statistic equals the no-anomaly null (to float rounding): mean −0.010, SD 0.125, q95 0.201, max 0.358, 0/200 > 0.40, z max 3.04 (1/200 > 3). (4) §2.3 states that √n assumes depth-independent residuals and that the shift null is the check that does not. Residual risk (a 3-block anomaly: 8/200) and the robust-detrend alternative are recorded in §2.3 and later.md. No threshold changed; nothing was fitted to the targets. C12/C14 now expect `n_with_r` 26 and 30. | §2.3, §3.4, §4.5, §4.6, D20, D33, §8; T103, T107, T109, T111, T114, T115, T116; later.md |
| m1 | minor | Weyl formula: one-sided perturbation gives `r = s/√(s²+ε²)` | **Fixed.** Text corrected: X = 0.80 tolerates ε ≤ 0.75 s, q10 = 0.50 tolerates ε ≤ 1.73 s, and q10 over 26 layers lets about 3 layers fall below. X and q10 are kept, with the reason stated (C18 carries the lineage separation). | §2.3 |
| m2 | minor | Two independent 64 MiB budgets allow ~128 MiB unconfirmed | **Fixed (unified).** Auto-approval now requires `totals.meta + totals.header + bytes_cap <= gate_threshold_bytes`, so the unconfirmed total per scan is ≤ 64 MiB. The P1 non-weight budget stays refuse-only. T106 test `test_run_gate_counts_nonweight`. | §3.2, D19; T106 |
| m3 | minor | T108 lambda late-binds `r` | **Fixed.** `pool.submit(fetch_one, r)` with `r` passed as an argument. | T108 |
| m4 | minor | "all 20" z_shift in the geometry test fails by construction too easily | **Fixed.** "≥ 19 of 20". The L = 16 test moves from 49/50 to ≥ 47/50, because with e = 1 the measured rate is 4/200 (P(fail by construction) < 2 %). | T109, §8 |
| m5 | minor | dp-structure+weights median is selection-biased | **Deferred** (below). | later.md |
| m6 | minor | T5 encoders match no role and get no skip hint | **Deferred** (below). | later.md |
| m7 | minor | T115 imports T016 but does not depend on it | **Fixed.** T115.depends_on = [T016, T111]. | T115, §10 |
| m8 | minor | C9 depends on whether fetch_log includes the REPORT stage_start | **Fixed.** C9 takes the REPORT stage_start seq from the live log; a positive test covers a Card without it. | §2.3 C9; T115 |
| m9 | minor | `scout suggest` CLI is a third renderer no check needs | **Deferred** (kept as is; below). | T112 |

### 11.3 Deferred minors

- r1 m4: scale `RETRY_SLACK_BYTES` with the plan (e.g. `max(16 MiB, 4 × largest read)`); deferred to P4 with streaming COMPUTE, and P2 accepts rerunning E3 whole (D19).
- r2 m5: report or calibrate the median on the structure-only alignment for compatible pairs (selection bias of `dp-structure+weights`); P3 hard negatives, later.md; the exit pairs use identity/reldepth and are unaffected.
- r2 m6: T5 `SelfAttention.{k,v}` role patterns or a "T5 attention" skip hint, with a T5 null measurement; later.md (Unscheduled), because adding roles unmeasured before the freeze would widen the frozen statistic untested.
- r2 m9: `scout suggest` stays in P2 (about 20 lines over the same `suggest()` result, which carries `disclaimers`; `test_suggest_cmd` asserts all three), because the CLI is the self-use path; revisit in P5 UX polish if the renderers drift.

## 12. Out of scope for Phase 2
- CKA, spectral top-k, norm/std tools, JEV, System 2, the labelled set, the base library,
  and verdict wording (P3)
- full downloads, the FULL_DOWNLOAD stage, the Card store, GGUF/quantized inputs and
  per-block attribution (P4)
- report export (P5)

See `later.md`.
