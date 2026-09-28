# Phase 2 plan: Diff view + first weight signal

Status: DRAFT (planner round 1), 2026-09-28. This plan builds on the Phase 1 plan
(`plans/phase-1/plan.md`, tasks T001–T016) and treats it as a contract. Section 9 lists every
Phase 1 interface that changes, and the task that makes each change.

## 1. Goal

scout compares two Cards side by side. It shows a structural diff whose block alignment
handles depth-upscaled and pruned models, suggests a reference by config match and
tokenizer MinHash, and colours each aligned layer by a sigma-curve correlation computed
from full k/v projection matrices read with gated, logged Range requests. A weight read is
issued only after a human has confirmed the exact byte count. Weights live only in
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
- the thresholds in §2.3

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

# E2 plan only: gate plans are shown, confirmation declined, 0 weight bytes
python scripts/exit_check_p2.py --pins exit/pins_p2.json --plan-only
# expect exit 0. Prints 3 plans and "CONFIRM WITH: --confirm-bytes <N>".
# At plan time (published configs):
#   Qwen2.5-7B           bytes_planned 205520896  (28 layers x 2 x 512x3584 x 2 B)
#   Qwen2.5-7B-Instruct  bytes_planned 205520896
#   Llama-3.1-8B         bytes_planned 536870912  (32 x 2 x 1024x4096 x 2 B)
#   N = sum(bytes_cap) = 947912704 + 3 x 16777216 = 998244352

# E3 the confirmed run (the human copies N from E2; this is the human confirmation)
python scripts/exit_check_p2.py --pins exit/pins_p2.json --confirm-bytes 998244352 --out /tmp/p2cards
# expect exit 0; the table shows C0..C17 all PASS; the last line is the JSON evidence

# E4 CLI path, same Cards
scout diff /tmp/p2cards/Qwen__Qwen2.5-7B/<sha>/_model.card.json \
           /tmp/p2cards/Qwen__Qwen2.5-7B-Instruct/<sha>/_model.card.json --json | jq .weight.median_r
# expect: equal (|diff| < 1e-9) to the E3 "related" median; exit 0

# E5 human visual check (recorded in log.jsonl): see 2.4
```

`--plan-only` and `--confirm-bytes` are mutually exclusive, and exactly one is required.
With `--confirm-bytes N`, the runner confirms each plan through
`POST /api/scans/{id}/confirm` only while the running sum of `bytes_cap` stays `<= N`.
Otherwise it declines, and the affected checks FAIL.

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
| C8 | Card v1 content: `schema_version == "card.v1"`; `stats.sigma_curves.tensors` has 2L entries; Parquet `stat_sigma_curve` is non-null for exactly those 2L rows, each of length `min(shape)` and non-increasing; `fetch_log.gate` has exactly 1 decision with `approved`, `via == "api-confirm"`, `bytes_confirmed >= bytes_cap` and the plan_id of C2; `stats.tokenizer_minhash.num_perm == 256` and `source_file == "tokenizer.json"` | true |
| C9 | purge: a `purge` event with stage `COMPUTE` and note containing `resident=0` precedes the `REPORT` `stage_start` in `fetch_log.events`; out dir holds only `*.card.json` / `*.tensors.parquet` | true |
| C10 | frozen params: `scout.sigma.SIGMA_PARAMS == FROZEN_SIGMA_PARAMS` and `scout.align.ALIGN_PARAMS == FROZEN_ALIGN_PARAMS` | equal |
| C11 | related structure: exactly 1 stack pair `model.layers`↔`model.layers`, `kind == "same"`, 28 layers, each `op == "match"` with `reference_index == subject_index`, `jumps == []` | true |
| C12 | related weight: `weight.n_with_r == 28` and **`weight.median_r >= 0.80`** | true |
| C13 | unrelated structure: `kind == "incompatible"`, `mode == "reldepth"`, and `tensor_diff.only_in_reference ⊇ {model.layers.#.self_attn.{q,k,v}_proj.bias}` | true |
| C14 | unrelated weight: `weight.n_with_r == 32` and **`weight.median_r <= 0.40`** | true |
| C15 | separation: `median_r(related) - median_r(unrelated) >= 0.40` | true |
| C16 | suggestion for the Instruct Card with pool = the other two Cards in the out dir: top-1 is `Qwen/Qwen2.5-7B@pin`, with `tokenizer_jaccard >= 0.90`, `config_score >= 0.80` and `claimed_base_match == true` | true |
| C17 | the diffs and suggestions are computed by a separate server whose CountingTransport records 0 requests; both DiffViews carry `disclaimers == DISCLAIMERS` | true |

`--plan-only` (E2) writes no Card. It evaluates only C0, C1 and a `P plan` check (`bytes_planned > 0` and `bytes_cap == bytes_planned + 16 MiB`). C2–C17 are evaluated in E3.

C15 follows from C12 and C14. It is kept as its own row so the evidence line states the margin.

**Why these thresholds (first principles, not fitted; see §5 D20 for the statistic).** For
each layer, the statistic is the Pearson r between two vectors. Each vector holds the
*depth-detrended, per-layer-centred log singular values of the top half of the spectrum* of
that layer's `v_proj` and `k_proj`.

- **Related pair.** A fine-tune changes W by ΔW. By Weyl, `|Δσ_k| ≤ ‖ΔW‖₂`, so on the top
  half of the spectrum (σ_k ≥ σ_median) each log σ_k moves by at most `‖ΔW‖₂/σ_median`. Write
  s for the RMS of a layer's residual vector (its layer-specific spectral structure) and ε
  for the RMS of the fine-tune perturbation in the same space. If the perturbation is
  uncorrelated with the structure, then `r ≈ s²/(s²+ε²)`. **X = 0.80** therefore tolerates a
  perturbation up to half of the layer-specific structure (`ε ≤ 0.5 s`). Below that we
  would no longer call the signal "high".
- **Unrelated pair.** Independent training runs share only generic structure: the matrix
  shape, a smooth dependence on depth, and a per-layer scale. The statistic removes each of
  these (grid on relative rank, cubic detrend over relative depth, per-layer centring), so
  under independence `E[r] = 0`. The residual curves are smooth along rank, so take a
  conservative effective dimension `d_eff ≥ 5`. The per-layer null SD is then at most
  `1/√(d_eff−1) = 0.5`. The median of `n ≥ 28` layers has SD `≤ 1.2533·0.5/√28 = 0.118`.
  **Y = 0.40** sits more than 3.3 SD above the null mean, which absorbs some residual generic
  structure without letting an unrelated pair pass as related.
- **Margin.** `X − Y = 0.40`, so the two regimes cannot overlap.

The product UI shows r on a continuous colour scale. These thresholds exist only in the
exit check. P2 renders no verdict, because the calibrated verdict belongs to System 1 in P3.

### 2.4 E5 human visual confirmation

1. Run `scout serve` and scan the three `repo@pin` targets with "weight sample" and
   "tokenizer" ticked.
2. Record per target:
   - The gate panel appears before any weight byte, and the counter shows `weight 0 B`.
   - The panel shows the planned bytes (exact and human-readable), the hard cap, `disk 0 B`,
     peak memory, the reason and a per-file table.
   - After Confirm, the weight counter rises to exactly the planned bytes.
   - Decline on a fourth scan of the Instruct repo ends in an error card with 0 weight bytes.
3. In the Diff panel, pick reference Qwen2.5-7B.
4. Click "Suggest" for subject Qwen2.5-7B-Instruct. The top suggestion must be
   Qwen2.5-7B.
5. Diff Qwen2.5-7B vs the Instruct model:
   - two strips of 28 cells, with straight alignment lines
   - the subject cells in the high-r colour
   - the median shown
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
- **PURGE** follows REPORT and emits a second `purge` event (`resident=0 verified`) in the
  live log only.

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
1. The plan is logged as a `gate` event.
2. If `bytes_cap <= gate_threshold_bytes` (default 64 MiB), the plan is auto-approved
   (`via "below-threshold"`, logged).
3. Otherwise a `Confirmer` must return an approval with `plan_id == plan.plan_id` and
   `bytes_confirmed >= bytes_cap`. Anything else is a decline.
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

| Interface | Confirmation |
|---|---|
| CLI flag | `--confirm-bytes N` (`via "cli-flag"`) |
| CLI prompt | TTY only; the user types `yes` after the plan is printed (`via "cli-prompt"`) |
| UI | `POST /api/scans/{id}/confirm` with `plan_id` and `bytes_confirmed` (`via "api-confirm"`) |
| UI decline | `POST .../decline`, or a 900 s timeout (`via "api-decline"` / `"timeout"`) |

Every decision is stored in `fetch_log.gate` of the Card(s) when a Card is written. It is
always in the live log.

### 3.3 Sampled read (what, how, how much)
The sample covers the largest stack (`structure.stacks[0]`, depth ≥ 8) of each unit.

- **Roles.** Roles are taken in the order `attn.v`, then `attn.k`. A role matches rel name
  `self_attn.v_proj.weight` or `attn.v_proj.weight` or `attention.v_proj.weight`, and
  likewise for k. The rel name must exist in every block, be 2-D, have dtype BF16/F16/F32
  and have `min(shape) >= 16`.
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
3. **Detrend over depth.** For each role and grid column, remove the least-squares cubic
   fit in relative depth `t = linspace(-1, 1, L)`. This removes the smooth depth structure
   common to all LLMs. It requires `L >= 8`.
4. **Layer vector.** Concatenate the roles common to both Cards, in role order.
5. **Correlation matrix.** Compute `R[i, j] = pearson(subject_i, reference_j)`. A
   zero-variance vector gives r = 0.
6. **Per-layer r.** Per-layer r is `R[i, a(i)]` under the displayed alignment `a` (§3.5).
   `median_r` is the median over subject layers with an r.

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
  `claimed_base_match`. Claimed repos that are not in the pool are listed as
  `unscanned_claims`.

## 4. Data structures (exact field names)

### 4.1 LogEvent (P1 §4.1, unchanged fields)
New `event` values:
- `gate`: 0 bytes. The note is the plan summary or the decision.
- `purge`: 0 bytes. The note is `released <n> weight bytes; resident=0` in COMPUTE, and
  `resident=0 verified` in PURGE.

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
GateDecision = {plan_id: str, approved: bool, via: "below-threshold"|"cli-flag"|"cli-prompt"|
  "api-confirm"|"api-decline"|"timeout"|"no-confirmer", bytes_planned: int, bytes_cap: int,
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
                          struct_cost: float, r: float|null}]}],
 unpaired_stacks: {reference: [str], subject: [str]},
 weight: {available: bool, reason: str|null, roles: [str], pair_index: int|null,
          median_r: float|null, n_layers: int, n_with_r: int},
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
  "detrend_degree": 3, "corr": "pearson"}
ALIGN_PARAMS = {"version": "align.v1", "lambda_jump": 2.0, "end_cost": 2.0,
  "struct_cost": {"equal": 0.0, "same_names": 0.5, "different": 1.0},
  "combine": "(S + (1 - r) / 2) / 2", "incompatible_mapping": "reldepth-round-half-up"}
```

### 4.7 API and CLI surface added
- **CLI flags:**
  - `scout scan TARGET [--sample] [--tokenizer] [--confirm-bytes N] [--gate-threshold-bytes N]`
  - `scout diff REF_CARD SUBJ_CARD [--json]`
  - `scout suggest CARD [--out DIR] [--top 5] [--json]`
- **HTTP routes** (T113):
  - `POST /api/scans` now also accepts `{"sample": bool, "tokenizer": bool}`.
  - `GET /api/scans/{id}` adds `plan` and `gate`, and `status` may be `awaiting_confirmation`.
  - `POST /api/scans/{id}/confirm` takes `{plan_id, bytes_confirmed}`.
  - `POST /api/scans/{id}/decline`.
  - `GET /api/cards`.
  - `POST /api/diff` takes `{reference: card_key, subject: card_key}`.
  - `GET /api/suggest?repo=&revision_sha=&component=`.
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
| D14 | Candidate pool for suggestion | Cards under the out dir (glob `*/*/*.card.json`, JSON only) plus `unscanned_claims` from the subject's `base_model` claims. No Hub search. | "Picked manually, tool suggests candidates". A local glob is not a Card store: no content addressing, no cache and no rescan skipping, all of which are P4. |
| D15 | Tokenizer MinHash scheme | As §3.6: whole-token shingles, 256 permutations, blake2b-32 + Mersenne-61 universal hash, fixed coefficients derived by blake2b (no RNG, so stable across numpy versions) | SE ≤ 0.031. It is deterministic and portable. |
| D16 | Tokenizer fetch | Opt-in (`--tokenizer` / `"tokenizer": true`). One file of about 7–9 MB, class meta. | This keeps the P1 10 s exit path byte-identical. |
| D17 | Config score and ranking | 13 keys: `model_type, hidden_size, num_hidden_layers, intermediate_size, num_attention_heads, num_key_value_heads, head_dim, vocab_size, rope_theta, tie_word_embeddings, max_position_embeddings, hidden_act, rms_norm_eps`. Score `0.5·J + 0.5·config`. Claims never score. | A lying `base_model` cannot steer the ranking. Invariant 5 requires stating that tokenizer reuse alone is not proof, so the UI shows both components separately. |
| D18 | What is sampled | Whole `v_proj` and `k_proj` of every block of the largest stack. One Range read per tensor. | Their σ are exactly permutation- and rotation-invariant, and they are the smallest 2-D matrices per block, 7–16 MiB per layer for 7–8B models. |
| D19 | Byte budget | 16 MiB per layer; slack 16 MiB per plan; gate threshold 64 MiB (`--gate-threshold-bytes`) | Llama-3.1-8B fits both roles exactly at 16 MiB. 70B models keep `v_proj` only. |
| D20 | Statistic | As §3.4: top half of the spectrum, 128-point log grid, per-layer centring, cubic depth detrend, Pearson, L ≥ 8 | See the §2.3 justification. Without the detrend and the centring, every trained spectrum correlates at ≈ 0.9 with every other, because all are monotone and heavy-tailed. |
| D21 | Raw cosine | Not computed in P2 | Invariant 4. Nothing needs triage yet. |
| D22 | Alignment algorithm | Viterbi over reference positions (§3.5), λ = 2, end cost 2; relative-depth mapping for incompatible pairs | The segment-copy model covers SOLAR-style duplication (backward jump) and layer pruning (forward jump or end cost). A jump must be justified by ≥ 2 fully mismatched layers of evidence. An incompatible pair is never weight-optimised, so r cannot be inflated by the choice of alignment. |
| D23 | Gate confirmation | `bytes_confirmed >= bytes_cap`. On decline: `GateDeclined` (exit 6), no Card. The CLI prompt appears only on a TTY; non-TTY without `--confirm-bytes` declines and prints the flag to use. The server waits 900 s, then treats it as a decline. | `--sample` means "I want weight evidence". A header-only Card is what plain `scout scan` gives. |
| D24 | Where weights live | In RAM only, in `SampleBuffers`. Each buffer is popped right after its SVD. A `purge` event with `resident=0` is written at the end of COMPUTE, and a verification event in PURGE. Disk is 0. | Invariant 1. Peak RAM ≈ bytes_planned (544 MiB for Llama), which is acceptable for a self-use tool. Streaming per-tensor compute is listed in later.md. |
| D25 | Diff persistence | Not persisted. DiffView is a pure function of two Cards, so the per-layer r values live nowhere but are reproducible from `stat_sigma_curve`. | Invariant 1: the Card is the only persisted artifact. |
| D26 | Card version | `card.v1` (§4.2). v0 stays loadable. | New fields `scan.options`, `scan.notes` and `fetch_log.gate`, plus the filled slots. |
| D27 | Frontend (D1 revisit) | Stay no-build vanilla. Add `scout/web/diff.js`. | The diff is two SVG strips plus lines and tables, so no framework is needed. |
| D28 | Exit thresholds | X = 0.80 (related median ≥), Y = 0.40 (unrelated median ≤), margin 0.40 | §2.3 |
| D29 | P2 pins | `exit/pins_p2.json`; `pin_exit.py --phase 2`; HF_TOKEN passed to git via `GIT_CONFIG_COUNT/KEY_0/VALUE_0 = http.extraHeader` | The token never appears in argv. The P1 pins file is untouched. |
| D30 | Stacks sampled | One per unit (`stacks[0]`) | This keeps bytes minimal. A text encoder's vision tower is not sampled (noted in `scan.notes`). |
| D31 | `GET /api/cards` | Lists Card keys found in the out dir, for the manual reference picker | This is the minimum needed for "picked manually". |

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

## 8. Risks and how each is tested

| Risk | Mitigation / test |
|---|---|
| A weight byte is read without confirmation | The gate sits in ByteLog preflight and is independent of callers (T101 tests: no grant, wrong stage, mixed range, outside grant, over the cap, after revoke). T108 asserts `fakehub.cdn_reads` has no weight range when declined. Exit C1 and C5 check the wire independently. |
| Retries exceed the confirmed bytes | `recheck` before each retry (T101 `test_recheck_cap`); exit C4 |
| Unlogged weight bytes | All weight reads go through `Source.read_range`. Exit C6 compares the wire total with the ByteLog. T116 rehearsal includes a CountingTransport. |
| Weights persisted or kept | `SampleBuffers.resident_bytes == 0` after COMPUTE, and the purge event precedes REPORT (T108). `test_no_extra_files` extends to sampled scans (HOME/TMPDIR empty, out dir holds only Cards). Exit C9. |
| The statistic correlates everything, like a raw σ-curve would | T109 synthetic tests: independent families with a **shared** linear+U-shaped depth trend give median r ≤ 0.40 for every seed pair; without the depth detrend (`residuals(degree=None)` test hook) the mean median is ≥ 0.60, which proves the detrend matters. Exit C14. |
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
| `scout/errors.py` (T002) | add `GateDeclined(ScoutError)` exit_code 6, attr `plan: dict | None`; add `SigmaError(ScoutError)` | T101 |
| `ByteLog` (T002) | new `grant`, `revoke_grant`, `grant_active`, `recheck`, `add_gate_decision`, `gate_decisions`; weight preflight allowed only under a grant in SAMPLED_READ; weight reservations tracked separately from the non-weight threshold; `note()` accepts `gate`, `purge`; `to_card_dict()` adds `"gate"`; `stage()` revokes an active grant when leaving SAMPLED_READ. **Without a grant, behaviour is byte-identical to P1.** | T101 |
| `HubSource._get` (T005) | calls `log.recheck(reservation)` before every attempt ≥ 2 | T101 |
| Card v0 → v1 (T009, plan P1 §4.2) | `SCHEMA_VERSION = "card.v1"`; `build_card` gains kwargs `options`, `notes`, `sigma_curves`, `sigma_by_tensor`, `tokenizer_minhash`; `load_card` accepts v0 and v1; Parquet metadata `card.v1`; the P1 tests asserting `card.v0` are updated | T102 |
| `scan()` / `ScanResult` (T010) | kwargs `tokenizer=False, sample=False, confirmer=None, gate_threshold_bytes=DEFAULT_GATE_THRESHOLD_BYTES`; `ScanResult.gate: list[dict]`; new stages SAMPLED_READ, COMPUTE, PURGE when sampling | T108 |
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
| T105 | Config match + reference suggestion over the out-dir pool | opus | T102, T104 |
| T106 | Sample plan + download gate + confirmers | opus | T008, T101, T107 |
| T107 | Sigma compute: dtype decode, SVD, SIGMA_PARAMS | opus | T101, T103 |
| T108 | Scan integration: tokenizer, SAMPLED_READ/COMPUTE/PURGE, Card stats fill | opus | T010, T102, T104, T106, T107 |
| T109 | Per-layer statistic: log grid, centring, cubic depth detrend, Pearson matrix | opus | T107 |
| T110 | Structural cost, stack pairing, Viterbi segment alignment, reldepth | opus | T101, T009 |
| T111 | Diff builder (DiffView) | opus | T011, T105, T108, T109, T110 |
| T112 | CLI: scan gate flags, prompt confirmer, `diff`, `suggest` | opus | T012, T108, T111 |
| T113 | Server: gate wait/confirm/decline, cards/diff/suggest routes | opus | T013, T108, T111 |
| T114 | Frontend: options, gate panel, diff view (strips + alignment + r colours) | opus | T014, T113 |
| T115 | P2 exit expectations (C1–C17 pure checks, frozen params, thresholds) | opus | T111 |
| T116 | P2 exit runner, pins_p2, `pin_exit --phase 2`, offline rehearsal | opus | T015, T112, T113, T115 |

Parallel waves (no shared files within a wave):
1. {T101}
2. {T102, T103, T110}
3. {T104, T107}
4. {T105, T106, T109}
5. T108
6. T111
7. {T112, T113, T115}
8. {T114, T116}

Most tasks are opus because they touch download gating, Card schema, similarity math or
exit evidence (routing rule). T103 is the only sonnet task, since it is fully specified test
scaffolding.

## 11. Out of scope for Phase 2
- CKA, spectral top-k, norm/std tools, JEV, System 2, the labelled set, the base library,
  and verdict wording (P3)
- full downloads, the FULL_DOWNLOAD stage, the Card store, GGUF/quantized inputs and
  per-block attribution (P4)
- report export (P5)

See `later.md`.
