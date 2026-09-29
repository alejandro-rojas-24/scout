# Phase 3 plan: Analysis backend + ground truth

Status: PLANNED (planner round 2 = REVISE round 1 response, 2026-09-29). This plan builds on the Phase 1 plan
(`plans/phase-1/plan.md`, T001–T016) and the Phase 2 plan (`plans/phase-2/plan.md`, T101–T116), and treats both as
contracts. Section 9 lists every earlier interface that changes and the task that makes each change. Task ids are
T201–T217. Section 11 answers every validation-round-1 finding.

**Exit numbers set by planner under human delegation (2026-09-28 directive).** The phases.md phrase "exact numbers
set by human" was delegated by the human to the orchestrator/planner on 2026-09-28. The numbers in §2.3/§2.4 (D45)
are therefore final for this phase, and this is not a missing sign-off. The CHECKPOINT summary restates them.

## 1. Goal

scout turns two Cards into a calibrated lineage verdict: System 1 (JEV, a calibrated classifier over weight-only
pair features: sigma-curve correlation, spectral top-k, tensor norms and an activation-free anchor-embedding CKA)
outputs `derived`, `not_derived` or `abstain` with a probability, and System 2 (an LLM that reads only Card digests
and System 1 output) explains it, checks the model card's `base_model` claim, and may only keep or soften System 1's
verdict. Both are trained, calibrated and frozen on a 45-pair labeled set with org-documented parentage, family-grouped
into train / calibration / held-out, and a minimal library of base Cards (Qwen, Llama, DeepSeek, Mistral, Gemma) serves
as the default reference pool.

## 2. Exit check (runnable, falsifiable)

### 2.1 Targets, pins, freeze order (fixed)

**Target set.** The exit targets are exactly the 40 repos of the labeled set (§5) plus the 6 library-only repos (§3.6),
46 repos in total. They are the keys of `exit/pins_p3.json` (T213).
- **C0 target set.** `set(pins) == labeled repos ∪ library repos`, or the run FAILs before anything else.
- There is no fallback repo. Replacing a repo happens only before the labeled FREEZE (E3). It is a logged human
  decision followed by a plan revision that edits §5, and it needs the new repo's evidence. After FREEZE, the set
  never changes for this phase.

**Carried over from P1/P2 unchanged.**
- The endpoint is always `https://huggingface.co`. A foreign `HF_ENDPOINT` means exit 1.
- Pins are written by `scripts/pin_exit.py --phase 3`. It cross-checks `scout resolve` against `git ls-remote`,
  passing HF_TOKEN through `GIT_CONFIG_*` env.
- Every weight read goes through the P2 gate. The full plan is displayed first, and approval needs the exact
  `(plan_id, bytes_cap)` pair.
- Every run ends with one JSON evidence line. PIN, SCAN, FREEZE and EXIT lines must share one `hostname`.

**Credentials.** Three kinds of repo are gated: `meta-llama/*`, `google/gemma-*`, and `mistralai/*` (the Mistral gate is
to be verified). HF_TOKEN must belong to an account that has accepted each licence. Scripts that touch the Hub exit 3
before any request when HF_TOKEN is unset. System 2 (E7) needs `ANTHROPIC_API_KEY`, or any credential source the
`anthropic` SDK resolves. These are environment limits only. The offline suite needs neither.

**Freeze order (no hand-tuning).** The order is enforced by checks C1 and C4. All of the following are fixed by this
plan before any real-model run and asserted literally by tests and by C5:
- every constant in `CKA_PARAMS`, `TSTATS_PARAMS`, `FEATURE_PARAMS` and `JEV_PARAMS` (§4.8)
- the anchor word list (sha256 `466fcb07…`)
- the exit thresholds (§2.3)
- the synthetic fixture constants (T208)

P2's `SIGMA_PARAMS` and `ALIGN_PARAMS` are unchanged. The order of events is:
1. **PIN.** Pins are written, and a PIN line is logged.
2. **Plan-only scan (0 weight bytes).** The plans, the byte total and the model-card claims of every repo are displayed.
   Every `verify: true` entry of the labeled file is resolved: its evidence quote is retrieved and pasted, or the entry is
   replaced by a plan revision. A repo whose plan has **zero reads** (no σ role and no anchor rows; P2 then skips the
   gate, so C3 could never pass) is printed as `ZERO-READ PLAN` and E2 exits 1. The remedy is a plan revision that
   replaces the repo, before FREEZE. E2 must exit 0 before E3 runs.
3. **FREEZE labeled.** `sha256(groundtruth/labeled_v1.yaml)` and `sha256(exit/pins_p3.json)` are logged. The line's
   `ts` must not be later than the `scanned_at` of any sampled Card (C1). Plan mode forces the gate threshold to 0,
   so no Card and no weight byte exist before this line.
4. **Confirmed scans.** A SCAN line with the confirmed `(plan_id, cap)` pairs is logged.
5. **Train and calibrate** on the train and calibration splits only. **FREEZE model** logs `sha256(models/jev_v1.json)`,
   the git commit, and `code_dirty` over `CODE_PATHS = ("scout", "scripts", "pyproject.toml")`. FREEZE model refuses
   (exit 1) when `code_dirty` is true, so the code the model was trained with is a commit. The held-out features have
   not been computed at this point (`p3_train.py` never loads held-out Cards).
   - **Pre-held-out stop.** If the calibration pool shows a threshold warning (a pool negative at p ≥ t_hi, or a pool
     positive at p ≤ t_lo), FREEZE model exits 4 and prints the pool rows. This includes the hard-negative subset,
     which is reported separately (`pool_hard`). Continuing needs `--ack-warnings "<human reason>"`, which is recorded
     in the FREEZE line. This keeps a foreseeable hard-negative failure from burning the one-shot held-out set.
6. **Held-out evaluation, once.** A re-run is allowed only on identical inputs: the model sha and the labeled sha must
   match their FREEZE lines, and **no file under `CODE_PATHS` may differ from the FREEZE-model commit** (C4: `git diff
   --name-only <freeze commit> HEAD -- CODE_PATHS` is empty and the working tree is clean on those paths). Log lines,
   Cards and the model file are outside `CODE_PATHS`, so committing them after FREEZE is allowed. Any change to the
   model or the code after an EXIT line is a new phase revision and a human decision.

**Phase status rule.** Phase 3 closes only when E0–E7 have all passed. The P1 and P2 offline suites and their offline
exit rehearsals stay in E0. The P1 and P2 network exits are not re-run: scans without `anchors` are byte-identical to
P2 apart from the Card version string (§9), and T207 and T216 test that. One `step:"EXIT"` line per run is appended to
`plans/phase-3/log.jsonl`.

### 2.2 Commands

```bash
# E0 offline (no network, no tokens): P1 + P2 + P3 suites incl. the P3 synthetic exit rehearsal (T217)
pip install -e '.[dev]'
pytest -q                                   # expect exit 0, 0 failures; network tests deselected

# E1 pins (Hub host, HF_TOKEN with the Llama 3.x, Gemma and Mistral licences accepted)
python scripts/pin_exit.py --phase 3 --pins exit/pins_p3.json
# expect exit 0; last line {"step":"PIN","phase":3,"hostname":...,"repos":{46 entries}}

# E2 plan-only, 0 weight bytes: 46 gate plans displayed, claims table, byte totals, CONFIRM WITH line
python scripts/p3_set.py scan --pins exit/pins_p3.json --labeled groundtruth/labeled_v1.yaml \
       --manifest library/manifest.yaml --out cards/p3 --plan-only
# expect exit 0; "TOTAL PLANNED: <N> B" with N about 12.0e9 at plan time (§2.5); every Card-less repo lists its plan;
# 0 "ZERO-READ PLAN" rows (else exit 1 -> replace the repo by plan revision);
# totals.weight == 0 for every repo; last line {"step":"SCAN","phase":3,"mode":"plan",...,"zero_read":[]}

# E3 freeze the labeled set (refuses while any verify:true remains or the file fails validation)
python scripts/p3_freeze.py labeled --labeled groundtruth/labeled_v1.yaml --pins exit/pins_p3.json
# expect exit 0; prints {"step":"FREEZE","phase":3,"what":"labeled","sha256":...,"pins_sha256":...,"counts":{...}}
# -> append to plans/phase-3/log.jsonl and commit

# E4 confirmed scans (the human copies the CONFIRM WITH line printed by E2)
python scripts/p3_set.py scan --pins exit/pins_p3.json --labeled groundtruth/labeled_v1.yaml \
       --manifest library/manifest.yaml --out cards/p3 --confirm-plans <ID:CAP,...>
# expect exit 0; 46 Cards (card.v2) under cards/p3; last line {"step":"SCAN","phase":3,"mode":"confirm",
#  "confirm_plans":{...},"weight_bytes":{repo:n},...} -> append to log.jsonl

# E5 train + calibrate + freeze the model (train and calibration splits only)
python scripts/p3_train.py --labeled groundtruth/labeled_v1.yaml --pins exit/pins_p3.json --cards cards/p3 \
       --out models/jev_v1.json
python scripts/p3_freeze.py model --model models/jev_v1.json --labeled groundtruth/labeled_v1.yaml
# expect exit 0 both; FREEZE model line (code_dirty false, pool_hard reported) -> append to log.jsonl and commit
# models/jev_v1.json. p3_freeze model exit 4 = threshold warning -> human decision (--ack-warnings "<reason>") before E6;
# exit 1 = uncommitted changes under scout/, scripts/ or pyproject.toml

# E6 held-out evaluation, System 1 (0 network requests)
python scripts/exit_check_p3.py --pins exit/pins_p3.json --labeled groundtruth/labeled_v1.yaml --cards cards/p3 \
       --model models/jev_v1.json --log plans/phase-3/log.jsonl
# expect exit 0; table C0..C11 all PASS; "EXIT CHECK (S1): PASS"; last line {"step":"EXIT","phase":3,"stage":"s1",...}

# E7 System 2 on the same held-out pairs (ANTHROPIC_API_KEY; default model claude-opus-5-5)
python scripts/exit_check_p3.py ... --s2          # same flags as E6 plus --s2 [--s2-model claude-opus-5-5]
# expect exit 0; C0..C12 all PASS, incl. C12 "available 19/19" (0 unavailable pairs allowed) and 19 recorded
# request hosts == api.anthropic.com; last line {"step":"EXIT","phase":3,"stage":"s2",...}
```

`p3_set.py scan` takes exactly one of `--plan-only` and `--confirm-plans`, as in P2. The exit check never scans. It
reads the Cards written by E4 and FAILs if a Card is missing or its `revision_sha` differs from the pin.

### 2.3 Checks (P3 namespace; one row per check: `target | check | expected | actual | PASS/FAIL`)

| # | Check | Threshold |
|---|---|---|
| C0 | target set: `set(pins) == labeled repos ∪ library repos` | equal, else FAIL and exit 1 |
| C1 | labeled frozen: `labeled.v1` validates (T201); `synthetic == false`; no `verify: true`; `sha256(labeled) ==` and `sha256(pins) ==` the last FREEZE-labeled line; that line's `ts` ≤ min `scan.scanned_at` over Cards with a gate decision | all true |
| C2 | splits: families are disjoint across splits; both repos of every pair are in families of that pair's split; every derived pair is within one family; held-out has `n ≥ 16`, `derived ≥ 6`, `not_derived ≥ 8`, `hard ≥ 4` (hard = not_derived with equal `hidden_size`, computed from the Cards; must equal the YAML flag) | all true |
| C3 | Cards complete, one per repo: `card.v2`; `key.revision_sha == pin`; `scan.options == {sample, tokenizer, anchors: true}`; exactly 1 approved gate decision via `cli-flag` (with `bytes_confirmed == bytes_cap`) or `below-threshold`; its `(plan_id, bytes_cap)` is in a SCAN-confirm line of log.jsonl (not needed for below-threshold); `plan.bytes_planned ≤ totals.weight ≤ bytes_cap`; a COMPUTE `purge` event with `resident=0`. A Card with no gate decision and the note `sigma: nothing sampleable` FAILs with actual `zero-read plan: replace the repo (plan revision)`; E2 catches this before FREEZE | true per repo |
| C4 | model frozen: `sha256(model) ==` the last FREEZE-model line, which comes after the FREEZE-labeled line in log.jsonl (and `ts` ≥); `model.labeled_sha256 ==` FREEZE labeled sha; `train_pairs ⊆ train`; the calibration pool ids are `⊆ train ∪ calibration`; no held-out pair id or held-out repo appears anywhere in the model file; `model.features == JEV_FEATURES`; **code frozen**: the FREEZE-model line has a non-null `git_commit` and `code_dirty == false`, and at EXIT `code_dirty == false` and `git diff --name-only <that commit> HEAD -- CODE_PATHS` is empty | all true |
| C5 | frozen params: `CKA_PARAMS`, `TSTATS_PARAMS`, `FEATURE_PARAMS`, `JEV_PARAMS` equal the frozen literals of `exit_expectations_p3.py`; `SIGMA_PARAMS`/`ALIGN_PARAMS` equal the P2 frozen literals; `ANCHORS` sha256 `== 466fcb0770645851483e61a7c7a72041d80c7155a2f0e24530c2a1099c421516` | equal |
| C6 | **held-out false "derived"**: #(label `not_derived`, S1 verdict `derived`) | `== 0` |
| C7 | **held-out accuracy among non-abstentions**: correct / #(verdict ≠ abstain) | `≥ 0.90`, and #non-abstained `≥ 1` |
| C8 | **held-out abstention rate**: #abstain / n | `≤ 0.30` |
| C9 | **cross-family sweep**: every unordered pair of held-out repos from different families (reference = lexicographically smaller repo) is scored; #derived verdicts | count `≥ 100` (104 at plan time) and derived `== 0` |
| C10 | offline analysis: S1 evaluation made 0 network requests (httpx transport patched to raise; the counter stays 0); `scout.analysis`, `scout.features`, `scout.jev` and `scout.system2` do not import `scout.hub`, `scout.sources`, `scout.scan` or `scout.gate` (static AST check) | true |
| C11 | disclaimers: every analysis report has `disclaimers == DISCLAIMERS`, and `format_analysis` text contains all three | true |
| C12 | System 2 (E7 only): every held-out pair has one S2 result that validates against `s2.v1`; **`available == true` and `served_model` non-null for every pair (`S2_UNAVAILABLE_MAX = 0`; an unavailable pair is a FAIL row naming the pair and its `unavailable_reason`, never a pass-through)**; `final_verdict ∈ allowed(S1 verdict)` (derived → {derived, abstain}; not_derived → {not_derived, abstain}; abstain → {abstain}); `caveats == DISCLAIMERS`; `assert_no_weights(s2 input)` passes; **`assert_request_no_weights` passes on the request body actually sent** (captured per call from the SDK's raw response, §3.4); exactly one recorded request host per pair, all `∈ {"api.anthropic.com"}`; on S2 final verdicts, **false derived == 0**, **accuracy among non-abstentions ≥ 0.90**, **abstention ≤ 0.30** | all true |

The EXIT evidence line records the following, whether or not a check uses them:
- per held-out pair: `p_derived`, the verdict, the features, the missing features and the abstain reason
- per split and family: null statistics (median `r_med`, median `cka`, participation ratios) for the not_derived pairs
- the thresholds, the model sha and the labeled sha
- `git_commit`, `git_dirty`, `code_dirty`, `code_changed_since_freeze` and `hostname`
- for E7: the S2 model id requested and served, the per-pair `available`/`unavailable_reason`, the request hosts, and
  the token usage

### 2.4 Why these numbers (exit numbers set by planner under human delegation (2026-09-28 directive); frozen in T216)

- **Held-out size: n ≥ 16, derived ≥ 6, not_derived ≥ 8, hard negatives ≥ 4.** The draft (§5) has 19 pairs: 10
  derived, 9 not_derived, 7 of them hard. The minimums leave room to drop up to 3 entries that fail verification
  without re-planning. Below 16, a single error moves accuracy by more than 6 points, and the checks stop meaning
  anything.
- **False derived = 0 (C6).** This is the requirement. It is a falsification gate, not an estimate. Zero false positives
  in 9 negatives only bounds the one-sided 95 % false-derived rate at `1 − 0.05^(1/9) = 28.3 %`, which is weak. So
  **C9** adds the cross-family sweep. Every held-out repo pair from different families is documented not_derived
  (each family root is a from-scratch pretrain; the evidence is in `families[].independence_evidence`), and it costs
  0 bytes because the Cards exist. Zero derived in 104 sweep pairs bounds the rate at `1 − 0.05^(1/104) = 2.8 %`
  if pairs were independent. They are not: they come from only 6 family pairs, so the honest reading is "no false
  derived across 6 independent lineage contrasts and 104 model combinations". The sweep includes the hard contrasts
  Mistral/Zephyr/SOLAR vs Llama-3.1-8B and its derivatives (identical k/v shapes), and SmolLM2-1.7B vs
  deepseek-coder-1.3b (identical k/v shapes and depth).
- **Accuracy among non-abstentions ≥ 0.90 (C7).** With 19 pairs and at most 5 abstentions, at least 14 are
  committed, so at most 1 error is allowed (13/14 = 0.929; 12/14 = 0.857 fails). Together with C6, that one error
  can only be a false `not_derived`.
  - *Correlation caveat (as for C9).* The 10 derived held-out pairs come from only 4 families, and L27–L29 share one
    reference (Mistral-7B-v0.1). The pairs are not independent, so the effective n is closer to the number of
    families than to 19. C7 therefore shows "≥ 90 % correct on this held-out set". It does not show a population
    accuracy with a tight interval. The EXIT line reports accuracy per family next to the pooled value (reported only).
- **Abstention ≤ 0.30 (C8).** At n = 19 that is at most 5. Since at most 5 derived pairs can abstain and at most 1 can be
  wrong, at least 4 of 10 derived pairs must be called `derived`. "Always abstain" and "never say derived" therefore
  both fail. The 30 % allows the genuinely hard held-out cases to abstain: SOLAR depth upscale, the slerp merge, the RL-heavy
  R1 distill, and the width-different small pairs. Anything higher would let the classifier avoid commitment.
- **Same thresholds for System 2 (C12).** S2 can only keep a verdict or soften it to abstain (§3.4), so FD = 0 carries
  over by construction. The abstention ceiling stops S2 from abstaining on everything.
- **Zero unavailable S2 pairs at E7 (C12).** An unavailable call copies S1, so allowing any would let E7 PASS with S2
  never answering (bad key, wrong model id, SDK break, truncation). The exit therefore requires 19/19 available answers.
  The product path (`scout analyze --s2`) still degrades to S1 with a visible note and exit 0.
- **What the numbers cannot show.** A 45-pair set cannot estimate calibration quality to better than about ±0.15.
  `p_derived` is reported with that caveat (`JEV_PARAMS.calibration_note`), and the report never prints a probability
  with more than 2 decimals.

### 2.5 Environment, access and bytes

| Step | Needs | This container |
|---|---|---|
| E0, all implementation | PyPI (numpy, pyarrow, httpx, pyyaml, pytest, `anthropic>=1.8.0,<2` which pulls `httpx2`) | available (anthropic 1.9.0 verified 2026-09-29) |
| E1 | `huggingface.co` API + git; HF_TOKEN with the Llama 3.1/3.2, Gemma 2/3 and (verify) Mistral licences accepted | blocked (proxy 403) |
| E2 | as E1, plus `*.hf.co` CDN; meta+header only, about 0.45 GB (46 tokenizers at 4–11 MB, plus headers; DeepSeek-V3-Base has 163 shard headers) | blocked |
| E4 | as E2, plus **about 11.2 GiB of weight ranges** (below), peak RAM about 0.85 GiB (SOLAR), about 25 min of SVD CPU | blocked |
| E5, E6 | local only (Cards); 0 network | runs anywhere once the Cards exist |
| E7 | `api.anthropic.com`, `ANTHROPIC_API_KEY`; 19 calls. Worst case at `claude-opus-5-5` ($4 in / $20 out per MTok): 19 × 12k input + 19 × 16k output (max_tokens cap) ≈ $0.9 + $6.1 ≈ **$7**; typical is well under that. `claude-sonnet-5-5` ($2 / $10) halves it | not attempted here |

**Byte estimate (plan time, from published configs; the exact values are the E2 plans).** The per-repo weight bytes
are σ-sample bytes plus anchor rows:
- σ-sample bytes follow the P2 rule: both k and v while a layer stays ≤ 16 MiB, then v only, else nothing.
- anchor rows are ≤ 256 × hidden × 2 B.

| Group | Repos | Weight MiB (σ + anchors) |
|---|---|---|
| Qwen2.5 small (0.5B ×2, 1.5B ×5) | 7 | 21.9 + 214.2 |
| Gemma (2-2b ×2, 3-1b ×2) | 4 | 470.3 + 60.6 |
| TinyLlama ×3 | 3 | 135 |
| StableLM-2-1.6B ×2 | 2 | 770 |
| Qwen3 (0.6B ×2, 1.7B ×2) | 4 | 673 |
| OLMo-2-1B ×3 | 3 | 771 |
| Mistral-7B lineage (×5 at 7B, SOLAR 10.7B) | 6 | 2572 + 770 |
| Llama (3.1-8B ×3, 3.2-1B ×2) | 5 | 1542 + 130 |
| SmolLM2 (1.7B ×2, 360M ×2) | 4 | 770 + 76 |
| deepseek-coder-1.3b ×2 | 2 | 770 |
| Library only (Qwen2.5-7B, Qwen3-8B-Base, Llama-3.2-3B, gemma-2-9b, deepseek-llm-7b-base, DeepSeek-V3-Base) | 6 | 1708 |
| **Total** | **46** | **≈ 11,450 MiB ≈ 11.2 GiB** (hard caps add at most 46 × 16 MiB of retry slack) |

The gate (P2 §3.2) applies to each repo as its own plan. Everything above 64 MiB of total unconfirmed bytes needs the
exact `(plan_id, cap)` pair from the E2 display. Qwen2.5-0.5B (about 11 MiB σ + 0.4 MiB anchors + 7 MB tokenizer) and
Qwen2.5-1.5B-sized repos auto-approve below the threshold, and that is logged as `below-threshold`. Disk use is 0: bytes
live only in `SampleBuffers` (P2 D24).

## 3. Definitions

### 3.1 Verdict semantics
- **`derived`** means *shared weight lineage*: one model was initialised from the other's weights, or both from a
  common checkpoint. This covers fine-tunes, instruct post-training, continued pretraining, merges, depth upscales, and
  siblings sharing a base. Weight similarity is symmetric, so **direction is never asserted**. The report says which
  model the metadata presents as the base, and labels it as a claim.
- **`not_derived`** means the weight evidence is consistent with independent initialisation. Distillation, synthetic
  data and a shared tokenizer are all compatible with `not_derived` (invariant 5).
- **`abstain`** means System 1 does not commit. Every abstention carries a reason code (§3.3).
- The labeled set uses the same semantics. Pairs whose weight relation is undocumented are excluded, never labeled
  `not_derived`. Examples are different sizes of one release trained by the same org, and Llama-3.2-1B vs Llama-3.1-8B
  (pruned and distilled).

### 3.2 Tools (weight features; every one is 0 extra bytes except the anchor rows)

**Tensor norm and std (T204).** For every tensor already read by the P2 σ-sample (k/v of each block), COMPUTE also
stores `stat_mean`, `stat_std` (ddof 0) and `stat_fro_norm = sqrt(Σσ²)` in the Parquet row, from the same decoded
float64 array. **Only `stat_fro_norm` feeds a verdict**, because it is a function of the singular values and therefore
exactly invariant to `W → P W Q` (invariant 4). Mean and std are shown as descriptive stats only; they are not
rotation-invariant.

**Spectral top-k (T204).** `stat_spectral_topk = [σ_i / ‖W‖_F for i < min(16, K)]`. Normalising by the Frobenius norm
makes it scale-free and keeps it exactly invariant to orthogonal transforms. The top-k tail is a different slice from
the P2 σ-curve, which uses the top half of the spectrum on a 128-point grid, depth-detrended.

Pair features, on the structure-only alignment (§3.3) and the P2 scored set:
- `log_topk_dist = log10(median_{i,role} RMS_k |log σ̂_k^subj − log σ̂_k^ref| + 1e-4)`
- `log_fro_dlog = log10(median_{i,role} |log ‖W‖^subj − log ‖W‖^ref| + 1e-4)`

**Anchor-embedding CKA, activation-free (T203, T205, T206).** This is how CKA is computed without activations.
- **Examples.** A frozen set of 256 English anchor words, each with a leading space (`scout/data/anchors_v1.txt`,
  sha256 `466fcb07…`). Examples correspond across models by *token identity*, not by position. Each model's
  `tokenizer.json` maps anchor `" the"` to its own id: byte-level `Ġthe`, metaspace `▁the`, or plain `the`.
- **Features.** The model's input-embedding row for that id (`model.embed_tokens.weight[id, :]`).
- **Kernel.** Linear, `G = X Xᵀ / hidden`, over the anchors present (n ≤ 256). The Card stores G's upper triangle,
  not the rows.
- **Estimator.** Debiased linear CKA: the unbiased HSIC estimator of Song et al. (2012) on zero-diagonal Grams,
  `HSIC_u(K,L) = [tr(K̃L̃) + (1ᵀK̃1)(1ᵀL̃1)/((n−1)(n−2)) − (2/(n−2))·1ᵀK̃L̃1] / (n(n−3))`, then
  `CKA = HSIC_u(K,L)/sqrt(HSIC_u(K,K)·HSIC_u(L,L))`. It is computed on the common anchors (≥ 128 required), clipped
  to [−1, 1].
  - *Why debiased.* This matters because hidden ≫ n here. The planner measured the plain linear CKA of two
    independent random embeddings (n = 256): 0.335 at d = 128 and **0.889 at d = 2048**. The debiased estimator gives
    0.001 and 0.000, and 0.9996 for a 2 % fine-tune (scratch simulation, 20 pairs). Real hidden sizes (896–4096) are
    all in the biased regime.
- **Invariance (invariant 4).** For any orthogonal Q on the hidden axis, `E → E Q` leaves `G = E Eᵀ` exactly
  unchanged. That covers hidden-unit permutation, sign flips and rotations (a rotated E gives CKA 1.000000,
  measured). CKA is also invariant to isotropic scaling. Examples are matched by token string, so vocabulary
  re-ordering and tokenizer changes that keep the anchor words are handled too.
  - *Limitation (stated in the report).* A non-orthogonal invertible transform of the hidden basis, such as a
    per-dimension rescale absorbed into the next layer's weights, is not covered. It is function-preserving only
    together with compensating changes elsewhere (RMSNorm makes it inexact). Tokenizer replacement with
    re-initialised embeddings erases this feature but not the σ features.
- **Byte cost.** For each present anchor there is one embedding row of `hidden × 2 B` (BF16): 1.75 KiB at hidden 896
  and 8 KiB at 4096. At 256 anchors that is 0.44–2.0 MiB per model (3.5 MiB for DeepSeek-V3, hidden 7168). Reads are
  coalesced only across consecutive ids, so there are at most 256 ranged reads per model, all in the same gated plan
  as the σ-sample (role `embed.anchor`).
- **What it measures.** Independent LMs learn similar semantic geometry, so the unrelated baseline is positive. The
  synthetic fixture models this with a shared rank-16 semantic component (T208, `SEMANTIC_WEIGHT = 0.3`, giving an
  independent CKA of about 0.42–0.48, measured). That is why CKA is one JEV feature and never a verdict on its own.

**Claimed-vs-detected (T211).** For each entry of the subject Card's `model_card.base_model`, which is always a claim:
- find a reference Card with that repo, either in the library or in the cards dir given.
- the status is:
  - `confirmed` if S1 says derived
  - `contradicted` if S1 says not_derived
  - `undetermined` if S1 abstains
  - `no_card` if there is no Card; the report says which repo to scan
- Any library base that S1 calls `derived` but that is not claimed is listed as `unclaimed_detections`.
- The claim never enters JEV features, the reference ranking or the thresholds. A lying `base_model` can therefore
  change only the status label next to the verdict (tested with a lying fixture, T208/T211).

**Config consistency (T211; the P1 later.md item).** These are cheap checks from Card contents, and they are reported to
System 2 and in the analysis:
- `num_hidden_layers` equals `stacks[0].depth`
- `hidden_size` equals the embedding width
- `vocab_size` equals the embedding rows
- k_proj rows equal `num_key_value_heads × head_dim`

Each failure is an issue string. None of them is a verdict.

### 3.3 System 1: JEV (assumption: "Joint Evidence Vector" classifier)

**Assumption A1.** The project docs never expand "JEV". This plan reads it as a **Joint Evidence Vector** classifier:
- a fixed, versioned vector of per-tool pair features from two Cards
- a small calibrated probabilistic classifier over that vector
- abstention outside a calibrated confidence band

If the human meant something else, only T209 and T210 change; the tools and the Card fields stay.

**Features (`JEV_FEATURES`, jev.v1, frozen order).** All are computed by T209 from two Cards, weight-only:

| Feature | Definition | Source |
|---|---|---|
| `r_med` | median per-layer σ-curve r over the P2 scored set | P2 `layercorr.compare` + `in_run_null` |
| `r_gap` | `r_med − offdiag_median_r` | same |
| `r_top1` | `diag_top1_frac` | same |
| `z_shift` | `clip(z_shift, −5, 10)`; a null sd gives 0.0 | same |
| `cka` | debiased anchor CKA | T205 |
| `log_topk_dist` | as §3.2 | T204 columns |
| `log_fro_dlog` | as §3.2 | T204 columns |

- **Alignment.** The alignment is *structure-only*. It is `align.viterbi(align.combine_cost(S, None))`, or
  `align.reldepth` when S is **constant and non-zero** (`S.min() == S.max() > 0`). That covers incompatible stacks (all 1)
  and partially matching role sets such as all 0.5, where Viterbi on a flat cost would only produce tie-break jumps
  and a spurious "pruned"/"upscaled" kind. An all-0 S (identical block structure) keeps Viterbi, as in P2. The
  alignment summary carries `structure_informative = not (S constant)`, and it is shown in the report and in `s2in`.
  R is never used to choose the alignment. This removes the P2 selection bias of
  `dp-structure+weights` for same-architecture pairs (P2 later.md; validation r2 minor 5). The DiffView of P2 is
  unchanged.
- **Excluded.** Tokenizer Jaccard, config score and the `base_model` claim are shown to System 2, but they are not JEV
  features:
  - invariant 5: tokenizer reuse is not proof
  - hard negatives L10 (TinyLlama v1.1 vs 1431k-3T) and several fine-tune pairs share tokenizers, so a 26-pair fit
    would learn the shortcut
- **Width shortcut.** Every derived train pair has equal shapes. So that `log_fro_dlog`, `log_topk_dist` and `r_gap`
  cannot learn "same width ⇒ derived", at least half of the not_derived pairs in train (5/9) and calibration (3/4)
  are same-`hidden_size` hard negatives (§5). That matches the held-out mix (7/9).

**Model.**
- L2-regularised logistic regression on standardised features (train mean and sd, sd floor 1e-6), fitted by Newton's
  method in numpy.
- λ = 1.0 on the weights, none on the intercept; 100 iterations, tol 1e-10.
- No feature selection and no hyperparameter search: every choice is frozen now.

**Missing features.** A missing feature is imputed with the frozen mean of that feature over the *not_derived*
training pairs. This is conservative: a missing value pulls toward not_derived. Examples are MLA or over-cap models
without σ roles, a v1 Card without stats, and fewer than 128 common anchors.

**Calibration.**
1. **Out-of-fold logits.** Every train pair gets a logit from a model fitted on the train pairs that share **no
   family** with it. A pair whose fold has fewer than 2 examples of either class gets no out-of-fold logit, and that is
   recorded.
2. **Final model.** The final model is fitted on all train pairs.
3. **Calibration pool.** The pool is the out-of-fold train logits plus the final model's logits on the calibration
   split. It must contain ≥ 3 of each class.
4. **Platt fit.** Platt scaling `p = σ(A·logit + B)` is fitted on the pool with Platt's regularised targets
   `(N₊+1)/(N₊+2)` and `1/(N₋+2)`. A fit with A ≤ 0 aborts training and goes to the human.

**Thresholds (rule frozen now; values computed on the pool, then frozen in the model file).**
- `t_hi = min(0.99, max(0.80, max p over pool negatives + 0.05))`
- `t_lo = max(0.01, min(0.20, min p over pool positives − 0.05))`
- then `t_lo = min(t_lo, t_hi − 0.10)`
- If a pool negative reaches p ≥ t_hi or a pool positive reaches p ≤ t_lo, the training report prints a warning.
  Training still writes the model, but FREEZE model stops with exit 4 until a human acknowledges it (§2.1 step 5).
- The pool keeps each example's `hard` flag. `threshold_inputs` reports `n_pool_neg_hard` and `max_neg_hard_p`, and
  the FREEZE line repeats them as `pool_hard`, so a hard negative near t_hi is visible before the held-out run.
- **OOF feasibility for the §5 layout** (checked by hand; T210/T214 record it):
  - Every train pair except L15 and L17 has a family-disjoint fold with ≥ 2 examples of each class.
  - L15 and L17 are {tinyllama, qwen25}. Every other train negative touches one of those two families, so their
    folds have 0 negatives and the pairs are skipped with the recorded reason.
  - Result: the pool has 16 OOF logits (9 derived, 7 not_derived of which 5 hard) plus 8 calibration logits (4/4, 3
    hard), which is ≥ 3 per class.
  - If TinyLlama_v1.1 is dropped at E2 (§5 L10), L10 leaves the pool: 15 OOF logits, 6 negatives, 4 of them hard.
    Every fold still has ≥ 2 per class, except L15 and L17.
  - The OOF logits of the hard TinyLlama×StableLM negatives come from folds without any hard negative. That is
    deliberately pessimistic: it pushes t_hi up.

**Verdict and abstention, in order.**
1. `abstain("no weight evidence")` when both `r_med` and `cka` are missing.
2. `abstain("out of training range")` when any *available* standardised feature has |z| > 6.
3. Otherwise:
   - `derived` when p ≥ t_hi
   - `not_derived` when p ≤ t_lo
   - else `abstain("low confidence")`

**Output (`jev.v1`, per pair).**
```
{p_derived, verdict, abstain_reason, logit, features: {name: float|null},
 missing: [str], model_sha256, thresholds: {hi, lo}}
```

### 3.4 System 2: reasoning model over Cards + JEV only (invariant 3)
- **Input (`s2in.v1`, T212).**
  - the analysis report of §4.6: S1 results per pair, claims, config consistency, disclaimers
  - one **Card digest** per model: key; model-card claims; config summary; structure summary; parameter count and
    dtypes; which evidence exists
  - Local paths are redacted to `local:<sha256[:8]>`.

  `assert_no_weights` rejects any input containing:
  - a numeric list longer than 64
  - any key starting with `stat_`
  - any key named `gram`, `signature`, `sigma`, `tensors` or `token_ids`

  The builder takes loaded Cards and the report, never a Source, and T216 C10 checks that statically.
- **Call.** There is one call per analysis (subject + references). It goes through the official `anthropic` Python SDK,
  **pinned `anthropic>=1.8.0,<2`** (§3.4.1).
  - The default model is `claude-opus-5-5`. The cheaper option is `claude-sonnet-5-5` (current Sonnet; `claude-sonnet-5`
    is the previous generation). The model is set by `S2Config.model`, env `SCOUT_S2_MODEL`, or `--s2-model`.
  - `thinking={"type": "adaptive"}`
  - `output_config={"effort": "high", "format": {"type": "json_schema", "schema": S2_LLM_SCHEMA}}`. Effort is set
    explicitly, because the Opus 5.5 default is `medium`.
  - `max_tokens=16000`
  - no sampling parameters and no prefill
  - For `claude-opus-5-5` and `claude-sonnet-5-5` (`FALLBACK_MODELS`), the call goes through
    `client.beta.messages.with_raw_response.create(..., betas=["server-side-fallback-2026-07-01"], fallbacks="default")`
    (server-side refusal fallback). The non-beta `messages.create` has no `fallbacks` keyword and raises TypeError
    (verified). Other models use `client.messages.with_raw_response.create(...)`.
  - The served model (`message.model`) is recorded.
  - **Wire evidence from the real SDK, with no custom HTTP client.** `with_raw_response` returns the SDK's response
    wrapper, and its `.http_request` is the exact `httpx2.Request` that was sent. `AnthropicS2Client` returns
    `request_host = http_request.url.host` and `request_body = json.loads(http_request.content)` in `S2Raw`, next to
    the parsed message (`.parse()`). No event hook, transport or `http_client=` is used in production.
  - Unavailable outcomes all give `available: false` with a distinct `unavailable_reason`, and the final verdicts then
    equal S1 with a note:

    | Outcome | `unavailable_reason` |
    |---|---|
    | an exception | `error:<ExceptionType>` |
    | `stop_reason == "refusal"` | `refusal` |
    | `stop_reason == "max_tokens"` | `max_tokens` (truncated JSON) |
    | any other `stop_reason` except `end_turn` | `stop_reason:<value>` |
    | missing, unparseable or schema-invalid text | `malformed output` |

    The product path never crashes on these. The **exit** (C12) counts any unavailable pair as a FAIL.
- **Prompt (`s2p.v1`).** The literal text is in T212. It states:
  - the verdict semantics
  - "you never see weights"
  - the claim-is-not-evidence rule
  - the only allowed changes (keep, or soften to abstain, with one reason code)
  - the three standing caveats
- **Output (`s2.v1`).** Deterministic post-processing (T212) enforces:
  - `final_verdict ∈ allowed(S1)`. **An abstention can never become `derived`**, and nothing becomes `derived` unless
    S1 said so. Any violation is replaced by the S1 verdict and flagged `override_blocked`.
  - claim statuses are the deterministic ones from T211; S2 only adds comments
  - `caveats` is set by code to `DISCLAIMERS`, whatever the model returned
- **No weights on the wire.** `assert_request_no_weights(body)` checks the body that was sent:
  - body keys ⊆ `{model, max_tokens, system, messages, thinking, output_config, fallbacks}`
  - `assert_no_weights(body)`
  - every string message content parses as JSON, is an `s2in.v1` object, and passes `assert_no_weights`

  C12 runs it on every captured body.

  S2 runs outside the S1 network guard, so that guard (a patch on `httpx.HTTPTransport`) does not see SDK traffic.
  The SDK uses `httpx2`. The request-host record is the S2 wire evidence.
- **Offline testing (boundary fake, not an HTTP mock).** The seam is our own `S2Client` protocol:
  `complete(system, user, schema, config) -> S2Raw`.
  - `StubS2Client` returns canned answers: valid, trying to upgrade, refusal, `max_tokens`, malformed, and dropping
    caveats. It builds `request_body` from the same pure `request_kwargs()` that the real client uses, and sets
    `request_host = "stub.invalid"`. The offline rehearsal passes `expected_hosts={"stub.invalid"}` to C12. The real
    exit uses `{"api.anthropic.com"}`.
  - `RecordingS2Client(inner)` wraps any client and keeps every `S2Raw`, so the exit runner gets hosts and bodies the
    same way for the stub and for the real SDK.
  - `AnthropicS2Client` is covered in two ways:
    - (a) a pure test of `request_kwargs()`: exact route, kwargs and beta list
    - (b) one SDK-contract test that runs the real `anthropic.Anthropic` over `httpx2.Client(transport=httpx2.MockTransport(h))`.
      Test (b) is the only place that touches the SDK's HTTP layer. It exists to catch an SDK version that rejects our
      kwargs, and it also asserts that an old `httpx.Client` is rejected with TypeError, which documents why the pin
      has a lower bound.

#### 3.4.1 SDK facts verified by the planner (2026-09-29, scratch venv under /tmp)
- PyPI serves `anthropic` 1.9.0 (latest); 1.0.0 is the first 1.x. Every 1.x wheel `Requires-Dist: httpx2<3,>=2.0.0`
  and no longer depends on `httpx`, while 0.125.0 still depends on `httpx`. The 1.9.0 install pulled httpx2 2.13.1.
- `anthropic.Anthropic(http_client=httpx.Client())` raises `TypeError` ("`httpx.Client` is from the `httpx` package,
  but this SDK uses `httpx2`"; explicit check since 1.4.0, isinstance check before). `httpx2.Client(transport=
  httpx2.MockTransport(h))` is accepted.
- The following are present in every version from 1.0.0 to 1.9.0:
  - `client.beta.messages.create(..., fallbacks=...)`, typed `Union[Iterable[BetaFallbackParam], Literal["default"]]`
  - the beta literal `"server-side-fallback-2026-07-01"`
  - `ThinkingConfigAdaptiveParam`
  - `OutputConfigParam.effort` / `.format`
- The `claude-opus-5-5` model literal first appears in 1.8.0, and `claude-sonnet-5-5` in 1.9.0. Model ids are `str`
  unions, so 1.8 accepts both. **Pin `anthropic>=1.8.0,<2`.**
- Measured on 1.9.0 with a mock transport:
  - The beta call goes to `https://api.anthropic.com/v1/messages?beta=true`, with header
    `anthropic-beta: server-side-fallback-2026-07-01` and body `"fallbacks": "default"`.
  - `raw.http_request.url.host == "api.anthropic.com"`, and `json.loads(raw.http_request.content)` equals the body the
    transport received.
  - `raw.parse().model`, `.stop_reason` and `.usage` parse.
  - The non-beta call sends no beta header and no `fallbacks` key. The non-beta `create(fallbacks=...)` raises
    TypeError. The default `base_url` is `https://api.anthropic.com`.
- Model facts from the claude-api reference (cached 2026-09-25):
  - `claude-opus-5-5` costs $4/$20 per MTok, and effort defaults to `medium` there.
  - `claude-sonnet-5-5` costs $2/$10 per MTok and accepts `fallbacks: "default"` on the Claude API.
  - On both, adaptive is the only thinking mode we use.

### 3.5 Labeled ground-truth set
- **File.** `groundtruth/labeled_v1.yaml`, schema `labeled.v1` (§4.2). It is human-edited, validated by
  `scout.groundtruth.validate` (T201), and frozen by sha256 in log.jsonl (E3).
- **Parentage evidence.** Each pair cites ≥ 1 evidence entry:
  - `kind ∈ {paper, model_card_text, org_blog, mergekit_config}`, `url`, `quote` (≤ 400 chars, verbatim) and
    `retrieved` (date)
  - `hub_metadata` (the `base_model` field) is allowed only *in addition*; a pair whose only evidence is
    `hub_metadata` fails validation. **A `base_model` claim can lie**, so it is never evidence alone.
  - `not_derived` pairs cite each family's `independence_evidence` (the root is pretrained from scratch).
  - E2 records the claimed `base_model` of every Card next to the labels. A disagreement is printed and investigated
    before FREEZE, never auto-resolved.
- **Split, frozen before training.**
  - The split unit is the **family**: all repos that share an org lineage root (e.g. every Qwen2.5-small derivative,
    including DeepSeek-R1-Distill-Qwen-1.5B).
  - Each family is assigned to exactly one split in the file.
  - A pair belongs to the split of its families, and both must be in that split, so cross-family negatives are formed
    only within a split.
  - The assignment is hand-made now (§5) and frozen with the file's sha256 before any weight byte is read (C1).
  - Training code receives only train and calibration pairs (T214 refuses held-out pairs), and C4 checks the model file
    for any held-out id or repo.
- **Sizes (draft).** 45 pairs: train 18 (9/9, **5 hard**), calibration 8 (4/4, **3 hard**), held-out 19 (10/9, 7 hard).
  Hard negative means not_derived with equal `hidden_size`, computed from Cards. At least half of each split's
  negatives are hard, so train, calibration and held-out have the same negative mix (§3.3 "Width shortcut").

### 3.6 Minimal reference library
- **File.** `library/manifest.yaml` (schema `library.v1`, T213) lists `{repo, family, note, verify}` and nothing else.
  Pins live in `exit/pins_p3.json`.
- **Cards.** They are built by the same gated set scanner as the labeled set (`p3_set.py`, options sample, tokenizer and
  anchors) into the cards dir (default `cards/p3`, gitignored).
- **No Card store.** Nothing is content-addressed or cached (P4). Cards are reproducible from pins, and the manifest is
  committed.
- **Entries (16).** 10 are shared with the labeled set, so their bytes are counted once.

| family | repos | note |
|---|---|---|
| qwen | Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B, **Qwen/Qwen2.5-7B**, Qwen/Qwen3-0.6B-Base, Qwen/Qwen3-1.7B-Base, **Qwen/Qwen3-8B-Base** | |
| llama | meta-llama/Llama-3.1-8B, meta-llama/Llama-3.2-1B, **meta-llama/Llama-3.2-3B** | gated |
| deepseek | deepseek-ai/deepseek-coder-1.3b-base, **deepseek-ai/deepseek-llm-7b-base**, **deepseek-ai/DeepSeek-V3-Base** | 7b-base: MHA v_proj 32 MiB > 16 MiB cap, so anchors only; V3: MLA (P2 skip), anchors only if embed dtype is BF16 (**verify**). A non-BF16 embed makes V3 a zero-read plan: E2 flags it and it is replaced by plan revision before FREEZE (e.g. by deepseek-ai/DeepSeek-V2-Lite, same rule) |
| mistral | mistralai/Mistral-7B-v0.1 | gated (verify) |
| gemma | google/gemma-2-2b, google/gemma-3-1b-pt, **google/gemma-2-9b** | gated; 9b: v only (14 MiB/layer) (**verify** shapes) |

Bold entries are library-only. `scout analyze SUBJECT` uses all library Cards as the default references (T215).

## 4. Data structures (exact field names)

### 4.1 Card v2 (changes relative to P2 §4.2–4.3 only; T202)
```
schema_version: "card.v2"
scan.options:   {sample: bool, tokenizer: bool, anchors: bool}                       # anchors NEW
stats:          {tensor_stats, spectral_topk, sigma_curves, tokenizer_minhash, attribution,
                 anchor_embedding}                                                    # anchor_embedding NEW slot
stats.tensor_stats  = null | {version: "tstats.v1", n_tensors: int, columns: ["stat_mean","stat_std","stat_fro_norm"],
                              std_ddof: 0, invariant_columns: ["stat_fro_norm"]}
stats.spectral_topk = null | {version: "topk.v1", k: 16, normalization: "fro", n_tensors: int}
stats.anchor_embedding = null | {version: "anchor.v1", anchor_set: "anchors.v1", anchor_sha256: str,
      tensor: str, file: str, dtype: str, vocab_rows: int, hidden: int, tokenizer_file: str,
      normalization: "byte-level"|"metaspace"|"plain", present: [int],   # ascending indices into ANCHORS
      token_ids: [int],                                                  # same order as present
      n_present: int, gram_scale: "1/hidden", gram_layout: "upper-row-major-incl-diag",
      plan_id: str, bytes_read: int}
```
Parquet: the columns of P1 §4.3, plus **`stat_anchor_gram` list<float64>** (nullable) appended last.
- It is non-null only on the embedding tensor's row: `n_present·(n_present+1)/2` values.
- `stat_mean`, `stat_std`, `stat_fro_norm` and `stat_spectral_topk` are non-null exactly for the σ-sampled rows.

Parquet metadata `schema_version: card.v2`. `load_card` accepts v0, v1 and v2. A plain scan still has every stat
null (P1 C7 holds).

### 4.2 Labeled set `labeled.v1` (T201)
```yaml
schema: labeled.v1
synthetic: false                 # true only for the offline fixture set (T208)
splits: [train, calibration, heldout]
evidence:                        # id -> citation
  - {id: E01, kind: paper|model_card_text|org_blog|mergekit_config|hub_metadata|fixture,
     url: str, quote: str|null, retrieved: YYYY-MM-DD|null, verify: bool}
families:
  - {id: str, split: train|calibration|heldout, independence_evidence: [evidence id], note: str}
repos:
  - {repo: str, family: str, verify: bool, note: str}
pairs:
  - {id: str, reference: repo, subject: repo, label: derived|not_derived,
     relation: post_train|fine_tune|continued_pretrain|merge|depth_upscale|independent,
     hard_negative: bool, evidence: [evidence id], verify: bool, note: str}
```
Validation rules (T201):
- ids are unique; `quote` is non-null when `verify` is false.
- a derived pair's repos share a family, and a not_derived pair's repos do not.
- a pair's split is its family's split, and both repos must be in that split.
- `hard_negative` implies `not_derived`.
- every evidence id resolves.
- a pair's evidence includes ≥ 1 non-`hub_metadata` kind, and `fixture` is allowed only when `synthetic: true`.
- `independent` is used if and only if the label is not_derived.
- no pair appears twice as an unordered pair.

### 4.3 JEV model file `jev_model.v1` (`models/jev_v1.json`, T210)
```
{schema: "jev_model.v1", params: JEV_PARAMS, feature_params: FEATURE_PARAMS, features: [JEV_FEATURES],
 standardize: {mean: [float], sd: [float]}, impute: [float], weights: [float], intercept: float,
 platt: {A: float, B: float}, thresholds: {hi: float, lo: float},
 threshold_inputs: {max_neg_p: float, min_pos_p: float, n_pool: int, n_pool_pos: int, n_pool_neg: int,
                    n_pool_neg_hard: int, max_neg_hard_p: float|null, warnings: [str]},
 train_pairs: [id], oof: [{id, logit: float|null, skipped_reason: str|null}],
 calibration_pool: [{id, source: "oof"|"calibration", logit: float, p: float, y: 0|1, hard: bool}],
 labeled_sha256: str, card_keys: {pair_id: [ref card_key, subj card_key]},
 trained_at: ISO-8601 Z, scout_version: str, git_commit: str|null}
```
The JEV model and the labeled set are configuration, versioned in git like `SIGMA_PARAMS`. They hold no weights and no
per-model artifact. Invariant 1 governs scan outputs, and the Card stays the only persisted per-model artifact (D36).
**The human acknowledges this reading of invariant 1 (D36, and committing `models/jev_v1.json`) at CHECKPOINT.** The
orchestrator lists it among the checkpoint items. If the human rejects it, the model file becomes a gitignored,
recomputable output of E5 whose sha is still frozen by the FREEZE line. Only D46 and the "commit" note of E5 change.

### 4.4 System 1 result `jev.v1`: see §3.3.

### 4.5 Pair features `feat.v1` (T209)
```
{version: "feat.v1", values: {JEV_FEATURES: float|null}, missing: [str],
 alignment: {mode: "structure-only"|"reldepth", kind: str, n_scored: int, structure_informative: bool},
 cka_common: int|null, reasons: {feature: str}}
```

### 4.6 Analysis report `analysis.v1` (T211; served and printed, never persisted)
```
{schema: "analysis.v1", subject: {card_key, title},
 verdict_semantics: str,                        # the §3.1 text
 pairs: [{reference: {card_key, title}, system1: jev.v1, features: feat.v1,
          structure: {kind, mode, width_changed: bool, subject_depth: int, reference_depth: int},
          context: {tokenizer_jaccard: float|null, config_score: float|null}}],
 claims: [{claimed_base: str, status: "confirmed"|"contradicted"|"undetermined"|"no_card",
           reference_index: int|null}],
 unclaimed_detections: [{reference_index: int, p_derived: float}],
 config_consistency: {subject: {ok: bool, issues: [str]}, references: [{ok, issues}]},
 disclaimers: [str, str, str]}                  # == scout.view.DISCLAIMERS
```

### 4.7 System 2 `s2in.v1` / `s2.v1` (T212)
```
s2in.v1 = {schema: "s2in.v1", subject: CardDigest, references: [CardDigest],
           analysis: analysis.v1 minus features.values rounded to 4 dp, disclaimers}
CardDigest = {card_key, model_card: {base_model, base_model_relation, license, license_name, tags (<= 20),
              pipeline_tag, library_name}, config: {model_type, architectures, <CONFIG_MATCH_KEYS values>},
              structure: {stacks: [{prefix, depth}], n_expert_groups: int}, weights: {params_total, n_tensors,
              dtypes: [str]}, evidence: {sigma: bool, tensor_stats: bool, anchors: bool, tokenizer: bool}}
S2_LLM_SCHEMA (what the model returns) = {pairs: [{pair_index: int, final_verdict: enum, reason_code: enum,
              explanation: str}], claims: [{claimed_base: str, comment: str}], summary: str, needs_human: [str]}
s2.v1 = {schema: "s2.v1", prompt_version: "s2p.v1", requested_model: str, served_model: str|null,
         available: bool, unavailable_reason: null | "refusal" | "max_tokens" | "malformed output" |
                                            "error:<ExceptionType>" | "stop_reason:<value>",
         request_host: str|null,                     # from the SDK's raw http_request (or "stub.invalid")
         pairs: [{pair_index, system1_verdict, final_verdict, changed: bool, reason_code, explanation,
                  override_blocked: bool}],
         claims: [{claimed_base, status (deterministic), comment}], summary: str, needs_human: [str],
         caveats: [str, str, str], usage: {input_tokens, output_tokens}|null}
reason_code enum: agree | claim_conflict | weak_evidence | structure_mismatch | missing_weight_evidence |
                  config_inconsistent | s2_unavailable | s2_missing | other
```

### 4.8 Frozen parameters (literals; T203–T210, asserted by tests and C5)
```
ANCHORS: 256 words of scout/data/anchors_v1.txt (T203); file sha256 466fcb0770645851483e61a7c7a72041d80c7155a2f0e24530c2a1099c421516
CKA_PARAMS = {"version": "cka.v1", "anchor_set": "anchors.v1",
  "anchor_sha256": "466fcb0770645851483e61a7c7a72041d80c7155a2f0e24530c2a1099c421516", "n_anchors": 256,
  "embed_names": ["model.embed_tokens.weight"], "embed_suffix": "embed_tokens.weight",
  "dtypes": ["BF16", "F16", "F32"], "min_present": 128, "min_common": 128,
  "gram_scale": "1/hidden", "estimator": "debiased-linear-hsic-song2012", "coalesce": "consecutive-ids"}
TSTATS_PARAMS = {"version": "tstats.v1", "topk": 16, "topk_normalization": "fro", "std_ddof": 0}
FEATURE_PARAMS = {"version": "feat.v1", "alignment": "structure-only", "z_clip": [-5.0, 10.0], "log_eps": 0.0001,
  "log_base": 10, "cka_min_common": 128}
JEV_FEATURES = ("r_med", "r_gap", "r_top1", "z_shift", "cka", "log_topk_dist", "log_fro_dlog")
JEV_PARAMS = {"version": "jev.v1", "model": "logistic-l2-newton", "l2_lambda": 1.0, "max_iter": 100, "tol": 1e-10,
  "sd_floor": 1e-06, "impute": "train-not-derived-mean", "oof": "family-disjoint", "oof_min_per_class": 2,
  "calibration": "platt-1999-targets", "pool_min_per_class": 3,
  "t_hi_floor": 0.80, "t_hi_cap": 0.99, "t_lo_floor": 0.01, "t_lo_cap": 0.20, "t_margin": 0.05, "min_band": 0.10,
  "ood_abs_z": 6.0, "calibration_note": "45-pair set: probabilities are coarse (about +/-0.15)"}
```

### 4.9 Log lines (`plans/phase-3/log.jsonl`; harness evidence, not product artifacts)
```
{"step":"PIN","phase":3,"hostname","repos":{repo:{scout,git_ls_remote}}}
{"step":"SCAN","phase":3,"mode":"plan"|"confirm","hostname","ts","plans":{repo:{plan_id,bytes_planned,bytes_cap}},
 "confirm_plans":{plan_id:cap},"weight_bytes":{repo:int},"meta_bytes":{repo:int},"claims":{repo:[str]},
 "zero_read":[repo],"errors":[...],"git_commit","git_dirty"}
{"step":"FREEZE","phase":3,"what":"labeled","ts","hostname","sha256","pins_sha256","counts":{split:{derived,not_derived,hard}},"git_commit"}
{"step":"FREEZE","phase":3,"what":"model","ts","hostname","sha256","labeled_sha256","thresholds","warnings",
 "pool_hard":{n_neg_hard,max_neg_hard_p,n_neg_hard_ge_hi},"ack_warnings":str|null,"git_commit","code_dirty"}
{"step":"EXIT","phase":3,"stage":"s1"|"s2","ts","hostname","passed","failed","pairs":[...],"sweep":{n,derived},
 "metrics":{fd,acc_nonabstain,abstain_rate,n,n_nonabstained,derived_recall,acc_by_family:{family:float|null}},
 "nulls":{...},"thresholds":{...},"model_sha256","labeled_sha256",
 "s2":{requested_model,served_models,usage,n_available,unavailable:{pair_id:reason},request_hosts}|null,
 "git_commit","git_dirty","code_dirty","code_changed_since_freeze":[path]}
```
There are no new ByteLog event types. Anchor reads are `fetch` events of class weight under the same grant. The gate
note (the `format_plan` text) lists the `embed.anchor` reads per file.

## 5. Labeled set, first draft (transcribed verbatim into `groundtruth/labeled_v1.yaml` by T201)

**Selection criteria.** A candidate is admitted only if all of these hold:
- it is a single-unit repo with `*.safetensors` (no `.bin`-only) and a `tokenizer.json`
- the k/v roles are separate (no fused QKV, no MLA), dtype is BF16, F16 or F32, and `model.embed_tokens.weight` exists
- the relation is documented by the org in a paper, the model-card body text, an official blog, or a mergekit config
- each family root is documented as a from-scratch pretrain
- small models are preferred (bytes)
- in every split at least half of the not_derived pairs are same-`hidden_size` hard negatives

Entries marked **V** have `verify: true`. Their repo id, relation or safetensors availability was not checked from this
container (no Hub access). They must be verified on the E1 host before FREEZE, or replaced by a plan revision. Every
`quote` is `null` in the draft: this container cannot retrieve the documents, so all quotes are pasted at E2 (which
is why every evidence entry starts with `verify: true`).

**Families and splits.**

| family | split | repos | independence evidence (root is a from-scratch pretrain) |
|---|---|---|---|
| qwen25 | train | Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-0.5B-Instruct, Qwen/Qwen2.5-1.5B, Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-Math-1.5B, Qwen/Qwen2.5-Coder-1.5B, deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | Qwen2.5 technical report (arXiv 2412.15115) |
| gemma | train | google/gemma-2-2b, google/gemma-2-2b-it, google/gemma-3-1b-pt, google/gemma-3-1b-it | Gemma 2 (arXiv 2408.00118), Gemma 3 (arXiv 2503.19786) |
| tinyllama | train | TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T **V**, TinyLlama/TinyLlama-1.1B-Chat-v1.0 | TinyLlama (arXiv 2401.02385) |
| tinyllama11 | train | TinyLlama/TinyLlama_v1.1 **V** | TinyLlama_v1.1 model card text stating it was trained from scratch with a **different initialisation** from the 1431k-3T run (**V**; see L10) |
| stablelm2 | train | stabilityai/stablelm-2-1_6b **V** (tokenizer.json), stabilityai/stablelm-2-1_6b-chat **V** | Stable LM 2 1.6B report (arXiv 2402.17834) |
| qwen3 | calibration | Qwen/Qwen3-0.6B-Base, Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B-Base, Qwen/Qwen3-1.7B | Qwen3 technical report (arXiv 2505.09388) |
| olmo2 | calibration | allenai/OLMo-2-0425-1B **V**, allenai/OLMo-2-0425-1B-SFT **V**, allenai/OLMo-2-0425-1B-Instruct **V** | OLMo 2 (arXiv 2501.00656) |
| mistral7b | heldout | mistralai/Mistral-7B-v0.1, mistralai/Mistral-7B-Instruct-v0.1, HuggingFaceH4/zephyr-7b-beta, teknium/OpenHermes-2.5-Mistral-7B **V** (safetensors), upstage/SOLAR-10.7B-v1.0, Weyaxi/OpenHermes-2.5-neural-chat-v3-3-Slerp **V** | Mistral 7B (arXiv 2310.06825) |
| llama3 | heldout | meta-llama/Llama-3.1-8B, meta-llama/Llama-3.1-8B-Instruct, deepseek-ai/DeepSeek-R1-Distill-Llama-8B, meta-llama/Llama-3.2-1B, meta-llama/Llama-3.2-1B-Instruct | The Llama 3 Herd of Models (arXiv 2407.21783) |
| smollm2 | heldout | HuggingFaceTB/SmolLM2-1.7B, HuggingFaceTB/SmolLM2-1.7B-Instruct, HuggingFaceTB/SmolLM2-360M, HuggingFaceTB/SmolLM2-360M-Instruct | SmolLM2 (arXiv 2502.02737) |
| dscoder | heldout | deepseek-ai/deepseek-coder-1.3b-base **V** (safetensors), deepseek-ai/deepseek-coder-1.3b-instruct **V** (safetensors) | DeepSeek-Coder (arXiv 2401.14196) |

The pair evidence below uses these citations, in addition to each family's independence evidence:

| Tag | Source |
|---|---|
| [R1] | DeepSeek-R1, arXiv 2501.12948 (distilled models fine-tuned from Qwen2.5-Math-1.5B and Llama-3.1-8B) |
| [QM] | Qwen2.5-Math, arXiv 2409.12122 |
| [QC] | Qwen2.5-Coder, arXiv 2409.12186 |
| [Z] | Zephyr, arXiv 2310.16944 |
| [S] | SOLAR 10.7B depth up-scaling from Mistral 7B, arXiv 2312.15166 |
| [G2] | Gemma 2, arXiv 2408.00118 (pre-trained and instruction-tuned 2B) |
| [G3] | Gemma 3, arXiv 2503.19786 (PT and IT checkpoints) |
| [Q3] | Qwen3 technical report, arXiv 2505.09388 (base and post-trained models) |
| [L3] | The Llama 3 Herd of Models, arXiv 2407.21783 (pre-trained and post-trained 3.1 8B) |
| [L32] | Meta Llama 3.2 announcement blog, ai.meta.com (1B/3B pre-trained and instruction-tuned), kind org_blog **V** |
| [MC] | model-card text of the subject repo (added next to the paper where a paper exists) |
| [MK] | mergekit config in the subject's model card |

**Pairs (45).** Each row lists the reference, then the subject.

**Train, derived (9)**

| id | pair (reference → subject) | relation | evidence |
|---|---|---|---|
| L01 | Qwen2.5-0.5B → Qwen2.5-0.5B-Instruct | post_train | [MC] |
| L02 | Qwen2.5-1.5B → Qwen2.5-1.5B-Instruct | post_train | [MC] |
| L03 | Qwen2.5-1.5B → Qwen2.5-Math-1.5B | continued_pretrain | [QM] **V** |
| L04 | Qwen2.5-1.5B → Qwen2.5-Coder-1.5B | continued_pretrain | [QC] **V** |
| L05 | Qwen2.5-Math-1.5B → DeepSeek-R1-Distill-Qwen-1.5B | fine_tune | [R1] |
| L06 | gemma-2-2b → gemma-2-2b-it | post_train | [G2], [MC] |
| L07 | gemma-3-1b-pt → gemma-3-1b-it | post_train | [G3], [MC] |
| L08 | TinyLlama-1431k-3T → TinyLlama-1.1B-Chat-v1.0 | fine_tune | [MC] **V** |
| L09 | stablelm-2-1_6b → stablelm-2-1_6b-chat | post_train | [MC] **V** |

**Train, not_derived (9)**

| id | pair | hard | note |
|---|---|---|---|
| L10 | TinyLlama-1431k-3T vs TinyLlama_v1.1 | yes | same arch, tokenizer and org; families tinyllama vs tinyllama11. **V: needs evidence that v1.1's initialisation differs** (a shared random init is shared lineage under §3.1). Without such evidence, drop TinyLlama_v1.1, family tinyllama11 and L10 by plan revision before FREEZE. Train is then 9/8 with 4/8 hard, and OOF stays feasible (§3.3) |
| L11 | TinyLlama-1.1B-Chat-v1.0 vs stablelm-2-1_6b-chat | yes | hidden 2048 |
| L12 | TinyLlama-1431k-3T vs stablelm-2-1_6b | yes | hidden 2048, two bases |
| L13 | gemma-2-2b vs Qwen2.5-1.5B | no | |
| L14 | gemma-3-1b-pt vs Qwen2.5-0.5B | no | |
| L15 | TinyLlama-1431k-3T vs Qwen2.5-1.5B | no | OOF-skipped (§3.3) |
| L16 | TinyLlama-1431k-3T vs stablelm-2-1_6b-chat | yes | hidden 2048 |
| L17 | DeepSeek-R1-Distill-Qwen-1.5B vs TinyLlama-1.1B-Chat-v1.0 | no | "DeepSeek" name, Qwen weights; OOF-skipped |
| L18 | TinyLlama-1.1B-Chat-v1.0 vs stablelm-2-1_6b | yes | hidden 2048 |

**Calibration, derived (4)**

| id | pair (reference → subject) | relation | evidence |
|---|---|---|---|
| L19 | Qwen3-0.6B-Base → Qwen3-0.6B | post_train | [Q3], [MC] |
| L20 | Qwen3-1.7B-Base → Qwen3-1.7B | post_train | [Q3], [MC] |
| L21 | OLMo-2-0425-1B → OLMo-2-0425-1B-SFT | post_train | [MC] **V** |
| L22 | OLMo-2-0425-1B → OLMo-2-0425-1B-Instruct | post_train | [MC] **V** |

**Calibration, not_derived (4)**

| id | pair | hard | note |
|---|---|---|---|
| L23 | Qwen3-1.7B-Base vs OLMo-2-0425-1B | yes | hidden 2048 |
| L24 | Qwen3-1.7B-Base vs OLMo-2-0425-1B-SFT | yes | hidden 2048 |
| L25 | Qwen3-1.7B vs OLMo-2-0425-1B-Instruct | yes | hidden 2048 |
| L26 | Qwen3-0.6B vs OLMo-2-0425-1B-SFT | no | |

**Held-out, derived (10)**

| id | pair (reference → subject) | relation | evidence |
|---|---|---|---|
| L27 | Mistral-7B-v0.1 → Mistral-7B-Instruct-v0.1 | post_train | [MC] |
| L28 | Mistral-7B-v0.1 → zephyr-7b-beta | fine_tune | [Z] |
| L29 | Mistral-7B-v0.1 → SOLAR-10.7B-v1.0 | depth_upscale | [S] |
| L30 | OpenHermes-2.5-Mistral-7B → OpenHermes-2.5-neural-chat-v3-3-Slerp | merge | [MK] **V** |
| L31 | Llama-3.1-8B → Llama-3.1-8B-Instruct | post_train | [L3], [MC] |
| L32 | Llama-3.1-8B → DeepSeek-R1-Distill-Llama-8B | fine_tune | [R1] |
| L33 | Llama-3.2-1B → Llama-3.2-1B-Instruct | post_train | [L32], [MC] |
| L34 | SmolLM2-1.7B → SmolLM2-1.7B-Instruct | post_train | [MC] |
| L35 | SmolLM2-360M → SmolLM2-360M-Instruct | post_train | [MC] |
| L36 | deepseek-coder-1.3b-base → deepseek-coder-1.3b-instruct | post_train | [MC] **V** |

**Held-out, not_derived (9)**

| id | pair | hard | note |
|---|---|---|---|
| L37 | Mistral-7B-v0.1 vs Llama-3.1-8B | yes | identical k/v shapes, 32 layers |
| L38 | SmolLM2-1.7B vs deepseek-coder-1.3b-base | yes | identical k/v shapes, 24 layers |
| L39 | zephyr-7b-beta vs Llama-3.1-8B-Instruct | yes | two derivatives |
| L40 | Mistral-7B-Instruct-v0.1 vs DeepSeek-R1-Distill-Llama-8B | yes | "DeepSeek" name, Llama weights |
| L41 | SmolLM2-1.7B-Instruct vs deepseek-coder-1.3b-instruct | yes | |
| L42 | SOLAR-10.7B-v1.0 vs Llama-3.1-8B | yes | upscaled; tests the structure-only alignment |
| L43 | Llama-3.2-1B vs SmolLM2-360M | no | |
| L44 | deepseek-coder-1.3b-base vs Llama-3.2-1B | yes | hidden 2048 both (k/v shapes differ: 2048 vs 512 rows) |
| L45 | Mistral-7B-v0.1 vs SmolLM2-1.7B | no | |

Excluded on purpose (undocumented weight relation):
- different sizes of one release (e.g. Qwen2.5-0.5B vs 1.5B)
- Llama-3.2-1B/3B vs Llama-3.1-8B (pruning plus logit distillation)
- Qwen3 vs Qwen2.5 (org-internal, not documented as independent init)

Distillation-only relations are never labeled derived.

## 6. Assumptions and open decisions (each with a recommended default)

Assumptions:
- **A1.** "JEV" means Joint Evidence Vector classifier (§3.3).
- **A2.** P1 and P2 are implemented exactly as specified (T001–T116).
- **A3.** The labeled repos keep the file layout checked at E2. If a pin lacks safetensors or `tokenizer.json`, the
  entry is replaced before FREEZE.
- **A4 (verified, no longer an assumption; §3.4.1).** `anthropic>=1.8.0,<2` provides `output_config.format`/`effort`,
  adaptive thinking, `client.beta.messages.with_raw_response.create(..., fallbacks="default")` with the
  `server-side-fallback-2026-07-01` beta, and `.http_request` on the raw response. The SDK's HTTP stack is `httpx2`.
  A 2.x SDK is out of range, and adopting it needs a plan revision.

| # | Decision | Default (recommended) | Rationale |
|---|---|---|---|
| D34 | Meaning of `derived` | Shared weight lineage, no direction (§3.1); siblings count as derived | Weight similarity is symmetric. A directional label would be decided by metadata, which can lie. |
| D35 | CKA variant | Debiased linear CKA over a frozen 256-word anchor set of input-embedding rows, matched by token string; Gram stored in the Card (§3.2) | The only activation-free CKA whose examples correspond across models without trusting weight positions. It is exactly invariant to hidden permutation and rotation and costs ≤ 2 MiB per model. Layer-weight CKA (Gram over the hidden axis) needs hidden-basis correspondence and is not permutation-robust, so it was rejected. |
| D36 | Is the stored Gram "weights"? | No. It is a second-moment statistic of 256 of 32k–256k rows, stored as float64 upper triangle (≤ 33k values). The rows themselves are never stored. | It is analogous to σ-curves (P2). It determines those 256 rows only up to an unknown rotation, which is useless as weights. Invariant 1 holds: bytes are in RAM only and released in COMPUTE. |
| D37 | JEV model family | L2 logistic (λ = 1) on 7 weight-only features, conservative imputation, no search | 18 train pairs. Anything larger overfits. Freezing every hyperparameter now removes the tuning degree of freedom. |
| D38 | Calibration | Family-disjoint out-of-fold logits on train plus the calibration split, then Platt with regularised targets | A separate calibration split of 8 pairs alone is too small. Out-of-fold logits keep it honest: no pair is scored by a model that saw its family. |
| D39 | Abstention | A three-way band `[t_lo, t_hi]` set by the frozen rule (§3.3), plus "no weight evidence" and "out of range" | The rule is set now and the values are computed from the pool. The 0.80 floor on t_hi stops a well-separated pool from producing a lax threshold. |
| D40 | Features excluded from JEV | Tokenizer J, config score and claims | Invariant 5; the hard negatives share tokenizers. They go to System 2 as context. |
| D41 | Alignment for features | Structure-only (never R-selected) | Fixes the P2 selection bias for same-architecture hard negatives (P2 later.md). |
| D42 | System 2 authority | Keep or soften to abstain only; claims deterministic; caveats set by code | Invariant 3 and 5. An LLM must never create a `derived` verdict. |
| D43 | System 2 model | `claude-opus-5-5` (effort high, adaptive thinking, structured output, server-side refusal fallback); `claude-sonnet-5-5` as the cheaper option via config (also with fallback) | Human default. The served model is recorded because the fallback can change it. |
| D44 | S2 granularity | One call per analysis (subject + all its references) | Small inputs (under 10k tokens). E7 is about 19 calls. |
| D45 | Exit numbers | Held-out n ≥ 16 (≥ 6/≥ 8/≥ 4 hard), FD = 0, accuracy ≥ 0.90, abstention ≤ 0.30, sweep ≥ 100 with 0 derived, S2 unavailable = 0 | §2.4. Exit numbers set by planner under human delegation (2026-09-28 directive). |
| D46 | Library storage | `library/manifest.yaml` committed; Cards built into a gitignored cards dir by the gated set scanner; no store or cache | "Minimal". The Card store is P4. Reproducible from pins. |
| D47 | Where analysis runs | CLI (`scout analyze`) and exit scripts only; no UI panel or HTTP route in P3 | "Analysis backend". The UI is P5 polish (later.md). |
| D48 | Anchor option | New opt-in scan option `anchors` (requires `tokenizer`); off by default | The P1/P2 exits and plain scans stay byte-identical. |
| D49 | P2 robust-detrend follow-up | Measure only: `p3_train.py --diagnostics` reports `r_med` at `edge_blocks_excluded` 1 (frozen) and 2 on train/calibration not_derived pairs. No change to `SIGMA_PARAMS`. | Changing the frozen P2 statistic would invalidate P2 C10. The bisquare IRLS is re-deferred (later.md). |
| D50 | S2 privacy | Digests redact local paths; only Card metadata and S1 output are sent | Self-use tool. Hub metadata is public anyway. |
| D51 | S2 test seam and wire evidence | Fake at our own `S2Client` boundary (`StubS2Client`, `RecordingS2Client`); the real client uses the SDK's `with_raw_response` and reads host and body from `.http_request`; no custom HTTP client in production; `anthropic>=1.8.0,<2` | Works unchanged with the real SDK. No dependency on the SDK's transport internals (httpx vs httpx2). A single SDK-contract test (httpx2 MockTransport) catches kwargs drift. |
| D52 | Code freeze between FREEZE-model and EXIT | C4 compares `CODE_PATHS` (scout/, scripts/, pyproject.toml) at EXIT against the FREEZE-model commit; FREEZE model refuses a dirty code tree | Held-out Cards exist before training (E4 < E5), so the model sha alone cannot prove that feature or JEV code did not change afterwards. |

## 7. Architecture (files)

```
groundtruth/labeled_v1.yaml, scout/groundtruth.py                           T201
scout/card.py (card.v2)                                                     T202
scout/data/anchors_v1.txt, scout/anchors.py                                 T203
scout/tstats.py, scout/sigma.py (compute_with_stats)                        T204
scout/cka.py                                                                T205
scout/gate.py (anchor reads in the plan)                                    T206
scout/scan.py (anchors option, COMPUTE stats + Gram)                        T207
tests/helpers/p3_fixtures.py                                                T208
scout/features.py                                                           T209
scout/jev.py                                                                T210
scout/analysis.py                                                           T211
scout/system2.py, pyproject.toml (anthropic>=1.8.0,<2; dev: httpx2)        T212
scripts/p3_set.py, library/manifest.yaml, exit/pins_p3.json, scripts/pin_exit.py (--phase 3), .gitignore (cards/)   T213
scripts/p3_common.py, scripts/p3_freeze.py, scripts/p3_train.py, models/.gitkeep   T214
scout/cli.py (scan --anchors, analyze)                                      T215
scripts/exit_expectations_p3.py, scripts/exit_expectations_p2.py (C8)       T216
scripts/exit_check_p3.py, tests/test_exit_check_p3_offline.py              T217
```

## 8. Risks and how each is tested

| Risk | Mitigation / test |
|---|---|
| A label is wrong (misread documentation) | Verbatim quote plus URL per evidence entry; `verify` blocks FREEZE (T201 `test_freeze_refuses_verify`); E2 prints claims next to labels; C1 |
| `base_model` lies | Never a feature, never a ranking input. The T208 fixture `fh/ft` claims `fg/base` falsely; T211 `test_lying_claim` gives status `contradicted`, and the S1 verdict is identical with and without the claim (T209 `test_claim_not_a_feature`) |
| Family leak across splits | T201 validator rules (tests per rule); T214 refuses held-out ids; C2 and C4 (a model file containing a held-out repo string FAILs; T216 test) |
| Overfitting a 26-pair training set | 7 frozen features, λ = 1, no search, family-disjoint out-of-fold calibration; T210 `test_oof_family_disjoint` asserts no fold sees the family it scores |
| Held-out too small for confidence | C9 sweep of about 104 cross-family pairs at 0 bytes; §2.4 states the bounds and the correlation caveat |
| CKA biased upward when hidden ≫ n | Debiased HSIC. T205 `test_debiased_null`: independent random 256×2048 rows → \|CKA\| ≤ 0.05 in 20/20 (plain CKA ≥ 0.8, asserted, to prove the bias is real) |
| CKA not robust to hidden transforms | T205 `test_rotation_invariance` (random orthogonal Q, permutation, sign flips → CKA ≥ 0.999999); non-orthogonal rescale is documented as a limitation (§3.2) |
| Anchor words missing in some vocab | `min_present = 128`, else the feature is missing and a note is written (T207 `test_anchor_too_few`); normalisation tested for byte-level, metaspace and plain (T203) |
| Anchor reads miss the gate or are unlogged | Anchor reads are PlannedReads in the same SamplePlan, under the same grant (T206/T207); T207 `test_anchor_reads_gated` (declined → 0 weight bytes, no anchor range on the wire); C3 checks `totals.weight` against the cap |
| σ selection bias on same-architecture negatives | Structure-only alignment (T209 `test_structure_only_alignment`: R is never passed to `combine_cost`, spy) |
| Edge anomalies deeper than block 0 (P2 residual risk) | Diagnostics at e = 1 and 2 on train/calibration negatives (D49); a real problem shows up as held-out FD, and C6 then FAILs to the human |
| Missing σ for MLA or over-cap models | Conservative imputation and a "no weight evidence" abstain; T210 `test_missing_imputation`, `test_no_evidence_abstains` |
| System 2 upgrades a verdict | Deterministic post-processing (T212 `test_block_upgrade`: a stub returns `derived` for an S1 abstain, final is abstain with `override_blocked`); C12 |
| System 2 sees weights | `assert_no_weights` on every input (T212 tests: a planted 65-float list or a `stat_`/`token_ids` key raises); `assert_request_no_weights` on the body actually sent (C12; T217 `test_rehearsal_s2_request_leak`); static import check C10; recorded request hosts limited to `api.anthropic.com` (C12) |
| System 2 never answers but E7 passes | C12 requires 19/19 `available` with a served model (T217 `test_rehearsal_s2_unavailable` and `test_rehearsal_s2_max_tokens` expect C12 FAIL) |
| SDK incompatibility (anthropic 1.x uses httpx2) | Pin `>=1.8.0,<2` (§3.4.1); no custom HTTP client in production; T212 SDK-contract test on the real SDK with an httpx2 MockTransport |
| JEV learns "same width ⇒ derived" | ≥ half of the train/calibration negatives are hard (§5); `pool_hard` in the FREEZE line; FREEZE model stops (exit 4) on any pool threshold warning before the held-out run |
| Code changes between FREEZE-model and EXIT | C4 code-frozen row (D52); T217 `test_rehearsal_code_changed` |
| A zero-read plan makes C3 unpassable | E2 `ZERO-READ PLAN` exit 1 → replace before FREEZE (T213 `test_zero_read_flagged`) |
| Caveats dropped by the LLM | Code sets `caveats = DISCLAIMERS` (T212 `test_caveats_forced` with a stub returning none); C11/C12 |
| API refusal, outage or cost | Refusal, truncation or error gives S2 unavailable and final = S1 in the product (T212 tests), and a C12 FAIL at the exit; E7 is 19 calls, ≤ about $7 worst case (§2.5) |
| Byte volume (about 11 GiB) and gated repos | Per-repo gated plans, plan-only first (E2), exact-pair confirmation (P2 mechanism); §2.5 access table |
| P2 exit regression from card.v2 | P2 C8 accepts v1 or v2 (T216, §9); `anchors` defaults off; T207 `test_p2_plan_unchanged` (plan_id and reads identical with anchors=False) |
| Hand-tuning after the held-out eval | FREEZE lines (labeled, model), C1/C4 ordering, C4 code-frozen row (no change under `CODE_PATHS` since the FREEZE-model commit); the model is refused if its sha differs |
| The rehearsal's synthetic set is too easy or too hard | Fixture constants are frozen (T208) with a semantic CKA baseline of 0.42–0.48 (measured). The rehearsal overrides only test-only keywords, never a threshold: `sweep_min` (the synthetic held-out has 16 pairs: 6 derived, 10 not_derived, 10 hard, but only 20 sweep pairs), the expected S2 host (`stub.invalid`), and the git-state function (a fake clean state, so a developer's dirty tree cannot fail the rehearsal). Every threshold is identical. A rehearsal failure goes to the human (no seed changes). |

## 9. Changes to earlier contracts

| Earlier contract | Change | Task |
|---|---|---|
| `scout/card.py` (P1 T009, P2 T102) | `SCHEMA_VERSION = "card.v2"`; `SUPPORTED_SCHEMA_VERSIONS` += `"card.v2"`; `STATS_SLOTS` += `"anchor_embedding"`; `PARQUET_SCHEMA` += `stat_anchor_gram list<float64>` (last); `build_card` gains kwargs `tensor_stats`, `stats_by_tensor`, `anchor_embedding`, `anchor_gram`; `scan.options` gains `anchors`. The P1/P2 card tests are updated from `card.v1` to `card.v2` and to the new key sets (`tests/test_card.py`, `tests/test_card_v1.py`) | T202 |
| `scout/sigma.py` (P2 T107) | new `compute_with_stats(buffers, specs)`; `compute_all` and `SIGMA_PARAMS` are unchanged | T204 |
| `scout/gate.py` (P2 T106) | `plan_sample(..., anchors: list[AnchorUnit] \| None = None)`; PlannedRead role `"embed.anchor"` with `block_index = -1`; `reason` appends the anchor clause; `SamplePlan.to_dict()` adds `"anchors"` (list, `[]` when none). With `anchors=None`, the plan and plan_id are byte-identical to P2 | T206 |
| `scout/scan.py` (P1 T010, P2 T108) | `scan(..., anchors: bool = False)`; anchors require `tokenizer=True` (else ValueError); COMPUTE routes σ reads to `compute_with_stats` and anchor reads to `anchor_gram`; Card v2 fill. `tests/test_scan_p2.py::test_plain_unchanged` updated for `options.anchors` | T207 |
| `scripts/pin_exit.py` (P1 T015, P2 T116) | `--phase 3`: targets = `p3_targets()` (labeled ∪ library), default pins `exit/pins_p3.json`, HF_TOKEN required; phases 1 and 2 unchanged | T213 |
| `scout/cli.py` (P1 T012, P2 T112) | `scout scan --anchors`; new `scout analyze` | T215 |
| `scripts/exit_expectations_p2.py` C8 (P2 T115) | `schema_version == "card.v1"` becomes `schema_version in ("card.v1", "card.v2")`; the T115 test "card.v0 (C8)" still fails as before; a new case `card.v2` passes. This is not a threshold, and every P2 threshold stays frozen. | T216 |
| `pyproject.toml` | dependency `anthropic>=1.8.0,<2` (§3.4.1); dev extra `httpx2>=2.0,<3` (test import only; already a dependency of anthropic 1.x) | T212 |
| P2 later.md items addressed | see §12 | — |

## 10. Task list

| id | title | route | depends_on |
|---|---|---|---|
| T201 | Labeled set: schema, validator, split rules, draft `labeled_v1.yaml` | opus | T001 |
| T202 | Card v2 schema (stats slots, anchor_embedding, `stat_anchor_gram`, options.anchors) | opus | T102 |
| T203 | Anchor word set + tokenizer vocab-id mapping and normalisation | opus | T104 |
| T204 | Tensor stats + spectral top-k in COMPUTE (`compute_with_stats`) | opus | T107 |
| T205 | Anchor Gram + debiased linear CKA | opus | T203 |
| T206 | Gated plan: anchor-row reads (`AnchorUnit`, coalescing) | opus | T106, T203, T205 |
| T207 | Scan integration: `anchors` option, COMPUTE stats and Gram, Card v2 fill | opus | T108, T202, T204, T205, T206 |
| T208 | P3 synthetic fixtures (anchor vocab, semantic embeddings, merges, lying claim, synthetic labeled set) | sonnet | T103, T201, T203, T205 |
| T209 | Pair features (JEV input) on the structure-only alignment | opus | T111, T202, T204, T205, T207 |
| T210 | JEV: fit, family-disjoint OOF, Platt, thresholds, abstention, model file | opus | T201, T209 |
| T211 | Analysis report: S1 per pair, claimed-vs-detected, config consistency, disclaimers | opus | T210 |
| T212 | System 2: digest, no-weights guard, prompt, Anthropic client + stub, post-processing | opus | T211 |
| T213 | Set scanner (plan-only/confirm), library manifest, pins_p3, `pin_exit --phase 3` | opus | T116, T201, T207 |
| T214 | Freeze and train scripts, code-freeze git state, diagnostics | opus | T210, T211, T213 |
| T215 | CLI: `scout scan --anchors`, `scout analyze [--s2]` | sonnet | T112, T207, T211, T212 |
| T216 | P3 exit expectations C0–C12 (pure) + P2 C8 amendment | opus | T115, T210, T211, T212 |
| T217 | P3 exit runner + offline synthetic rehearsal | opus | T208, T213, T214, T216 |

Parallel waves (no shared files within a wave):
1. {T201, T202, T203, T204}
2. {T205}
3. {T206, T208}
4. {T207}
5. {T209, T213}
6. {T210}
7. {T211}
8. {T212, T214}
9. {T215, T216}
10. {T217}

Routing: nearly every task touches download gating, Card schema, similarity math, JEV features or calibration, or
verdict and invariant-5 wording, so they are opus. T208 (test fixtures with every constant given) and T215 (CLI
plumbing over finished functions, with exact output text) are sonnet. T215's library lookup is now fully specified:
Cards are resolved through the pins file, so no choice is left open.

T214 now depends on T211 because `LoadedCard` lives in `scout/analysis.py`. It moved from wave 7 to wave 8, next to
T212, and the two share no files.

## 11. Validation responses — round 1

Validator verdict: REVISE (1 blocker, 4 majors, 15 minors). Each finding is listed with what changed and where.

| # | Finding | Response | Where |
|---|---|---|---|
| B1 | L10 is a not_derived pair inside family `tinyllama`, so `validate()` rejects the draft | **Fixed.** TinyLlama_v1.1 is its own train family `tinyllama11` (11 families). L10 is now cross-family. OOF feasibility was re-derived for the new layout (§3.3): L15/L17 are skipped and the pool has ≥ 3 per class, with or without v1.1 | §5, §3.3, T201 |
| M1 | E7 can pass with S2 never answering | **Fixed.** C12 requires `available` and a non-null `served_model` for all held-out pairs (`S2_UNAVAILABLE_MAX = 0`), and each unavailable pair is a FAIL row. `max_tokens` has its own reason, as does any other non-`end_turn` stop. The rehearsal tests for unavailable and max_tokens now expect a C12 FAIL. Offline `analyze --s2` still degrades with exit 0 | §2.3, §2.4, §3.4, T212, T216, T217 |
| M2 | anthropic 1.x uses httpx2; `http_client=httpx.Client(...)` raises TypeError | **Fixed, verified (§3.4.1).** Pin `anthropic>=1.8.0,<2`. The production code has no custom HTTP client. Host and sent body come from `with_raw_response(...).http_request`, which works with the real SDK. Offline tests fake our own `S2Client` boundary, and one SDK-contract test uses `httpx2.MockTransport`. The S1 network guard stays on `httpx` (the Hub path), and the plan says it does not see SDK traffic | §3.4, §3.4.1, D51, T212, T217 |
| M3 | T214 imports `LoadedCard` (T211) without depending on it | **Fixed.** T214 depends on T211, and the waves are now 7 = {T211}, 8 = {T212, T214} | §10, T214 |
| M4 | Train/calibration negatives are mostly easy, while held-out negatives are mostly hard | **Fixed.** Train negatives are 5/9 hard (L10, L11, L12, L16, L18; all 2048-wide), and calibration is 3/4 hard (L23, L24, L25). The pool carries `hard`, and the FREEZE line reports `pool_hard`. FREEZE model stops (exit 4) on any pool threshold warning before the held-out run | §5, §3.3, §2.1, T201, T210, T214 |
| m1 | L10 shared-init risk | Fixed: L10's verify step requires evidence that the initialisation differs. Without it, v1.1 and L10 are dropped by plan revision (the layout stays feasible) | §5 L10 |
| m2 | No git check between FREEZE-model and EXIT | Fixed: D52 and the C4 code-frozen row over `CODE_PATHS`; FREEZE model refuses a dirty code tree | §2.1, §2.3, T214, T216, T217 |
| m3 | C3 is unpassable for zero-read plans | Fixed: E2 prints `ZERO-READ PLAN` and exits 1, and the remedy is replacement before FREEZE. C3 names the cause | §2.1, §2.3, §3.6, T213, T216 |
| m4 | Constant S (e.g. all 0.5) goes to Viterbi | Fixed: reldepth when S is constant and non-zero; `structure_informative` is reported | §3.3, T209 |
| m5 | [MC]-only evidence for L06/L07/L19/L20/L31/L33 | Fixed: [G2], [G3], [Q3], [L3] and [L32] are added. OpenHermes-2.5 safetensors and the stablelm-2-1_6b tokenizer.json are marked V | §5, T201 |
| m6 | T215 "lexicographically largest directory" | Fixed: library Cards are resolved through the pins file (`--pins`), and T215 stays sonnet | T215 |
| m7 | Cheaper model id and cost | Fixed: `claude-sonnet-5-5`, with fallback; E7 worst case is about $7 at Opus 5.5 prices | §2.5, §3.4, D43, T212 |
| m8 | Human sign-off for the exit numbers | Recorded: exit numbers set by planner under human delegation (2026-09-28 directive). This is not a missing sign-off | header, §2.4, D45 |
| m9 | no-weights check runs on a recomputed input, not on the request actually sent | Fixed: `assert_request_no_weights` on the captured body (C12) | §3.4, T212, T216, T217 |
| m10 | `token_ids` missing from the §3.4 forbidden keys | Fixed | §3.4 |
| m11 | No multi-component anchor test | Fixed: T207 `test_anchors_multicomponent` (diffusers `text_encoder/` + `tokenizer/tokenizer.json`) | T207 |
| m12 | C7 correlation caveat | Fixed: the caveat is stated, and the EXIT line reports `acc_by_family` | §2.4, §4.9 |
| m13 | `test_rehearsal_leak` undefined | Fixed: move held-out family `fi` to train in a copy, re-freeze, retrain, then evaluate against the original file → C1 and C4 FAIL | T217 |
| m14 | D49 diagnostics and EXIT `nulls` have no consumer | Deferred (see below) | — |
| m15 | D36 vs invariant 1 | Fixed: the human acknowledges D36 at CHECKPOINT, and a fallback is defined if they reject it | §4.3 |

**Deferred minors.**
- m14: T214 `--diagnostics` and the EXIT `nulls` are kept as reported-only output. They answer the P2 later.md item
  "null median r and PR per family on the labelled set" (§12) for under about 60 lines, and by design no exit check
  consumes them.

## 12. Items deferred "to P3" by earlier phases, and their disposition

| Source | Item | Disposition |
|---|---|---|
| P1 later | Claimed-vs-detected `base_model` check | **Included** (T211) |
| P1 later | Config vs header consistency as a formal tool | **Included** (T211, report only) |
| P2 later | Calibrated verdicts from `median_r`, r distribution, alignment kind, tokenizer J | **Included** (T209/T210). J is excluded from JEV by D40, and alignment kind goes to S2 context. |
| P2 later | Hard negatives; null median r and PR per family on the labelled set | **Included** (§5 hard negatives; T214 diagnostics; EXIT line `nulls`) |
| P2 later / FUTURE_IMPROVEMENTS | Depth-localised anomalies beyond the edge block: bisquare vs e = 2 | **Partly.** e = 2 is measured as a diagnostic (D49). Bisquare IRLS is re-deferred: it adds tuning constants, and adopting any change would unfreeze P2 `SIGMA_PARAMS`. |
| P2 later | Depth autocorrelation / effective n per family | **Re-deferred.** JEV consumes `z_shift`, which already uses the in-run shift null, so no P3 threshold relies on the √n formula. |
| P2 later / FUTURE_IMPROVEMENTS | Selection bias of `dp-structure+weights` | **Included** (D41, structure-only alignment for features) |
| P2 later | CKA, spectral top-k, norm/std tools | **Included** (T203–T207) |
| P2 later | Base reference library | **Included** (T213) |
| P2 later (unscheduled) | MLA roles, fused QKV, T5 encoders | **Re-deferred.** No labeled pair needs them. Anchor CKA covers MLA models' embeddings. |
| P2 later | Retry slack scaling, streaming COMPUTE | P4 (unchanged) |

## 13. Out of scope for Phase 3
- DGX batch jobs, FULL_DOWNLOAD, the Card store and caching, GGUF/quantized inputs, per-block (Zhu et al.)
  attribution: all P4
- UI panels for analysis or System 2, report export, the retrain loop: all P5
- behavioural or black-box signals

See `later.md`.
