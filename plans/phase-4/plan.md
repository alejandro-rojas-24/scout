# Phase 4 plan: Infrastructure + block attribution

Status: PLANNED — approved by orchestrator under human directive (review disabled), 2026-09-29.
Revision r2 (planner): revised after validation round 1 (0 blockers, 6 majors, 14 minors) and round 2 (0 blockers,
3 majors, 9 minors); §11.

This plan builds on the Phase 1–3 plans (`plans/phase-{1,2,3}/plan.md`, tasks T001–T218) and treats them as contracts:
Card v2 and its stats, the ByteLog and the P2 download gate (plan_id + exact cap), the frozen `SIGMA_PARAMS`,
`ALIGN_PARAMS`, `CKA_PARAMS`, JEV, System 2, and the P3 evaluation ledger. Section 9 lists every earlier interface that
changes and the task that makes the change. Task ids are T301–T321. Decision numbers continue from P3 (D56 onward).

## 1. Goal

scout escalates from sampled reads to full-tensor downloads that run as batch jobs on the DGX Spark. Every job is
confirmed through the P2 gate with bytes, disk and reason shown, streams its byte log back to the client, and deletes
every weight byte from disk as soon as COMPUTE ends, with a sweeper that removes the scratch files of crashed jobs.
Cards live in a content-addressed store that skips rescans. GGUF and FP8 inputs are read header-only and dequantised
before any comparison. Each block of the subject's depth strip is then attributed to a base model by the Zhu et al.
neuron-matching test, with a p-value calibrated against an independent control model at the same depths, a per-depth
paired margin against that control, and a family-wise error bound that holds under a stated assumption about that
control (§3.8).

## 2. Exit check (runnable, falsifiable)

### 2.1 Targets, references, pins (fixed)

The exit has exactly two attribution targets. Each has one documented base and one independent control reference;
the te job also carries one independent candidate (below). They are frozen in
`scripts/exit_expectations_p4.py::P4_TARGETS` (T320):

| label | subject (target, selector) | base (derived-from, documented) | control (independent) |
|---|---|---|---|
| `70b` | `bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF` `#DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf` (**V**: file name, single file) | `meta-llama/Llama-3.3-70B-Instruct` (gated). DeepSeek-R1 paper, arXiv 2501.12948: the 70B distilled model was fine-tuned from Llama-3.3-70B-Instruct. The GGUF is a llama.cpp Q4_K_M quantisation of `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` (the model card text states this, **V**) | `Qwen/Qwen2.5-72B`. Qwen2.5 technical report, arXiv 2412.15115, pretrained from scratch; same hidden size 8192 and depth 80 as the subject, and a different intermediate size (29568 vs 28672) |
| `te` | `Qwen/Qwen-Image` `#text_encoder` (diffusers; `Qwen2_5_VLForConditionalGeneration`; **V**: which tokenizer file `tokenizer/` ships, `tokenizer.json` or `vocab.json` + `merges.txt`; C15 accepts either and is evaluated at E3) | `Qwen/Qwen2.5-VL-7B-Instruct`. Qwen-Image technical report, arXiv 2508.02324: Qwen2.5-VL is the frozen condition encoder (**V**: quote) | `meta-llama/Llama-3.1-8B` (gated). Llama 3 herd paper, arXiv 2407.21783, pretrained from scratch |

- **Independent candidate (te job only; validation r2 major 2).** `allenai/OLMo-2-1124-7B` (ungated, Apache-2.0).
  OLMo 2 report, arXiv 2501.00656: pretrained from scratch by AI2 on its own data (OLMo-mix-1124). It enters the te job
  as a second candidate (role `reference`, **not** `control`), so it is tested by exactly the rule that attributes the
  base, and C10b requires 0 attributed blocks. It is the exit's specificity leg on real weights: the only live check
  that an unrelated model is not attributed. Shape (**V** at E3): 32 blocks, hidden 4096 (= its embedding width),
  intermediate 11008, vocab 100352, BF16 (**V**; F32 would double its bytes, and the E3 plan is exact either way). Its
  blocks use post-sublayer norms (`post_attention_layernorm`, `post_feedforward_layernorm`), so the frozen norm role
  matches `post_attention_layernorm`, which in OLMo 2 scales the attention output rather than the MLP input. That changes
  only its probe scaling; independence, and therefore the null, is unaffected. Alignment to the 28-block subject is
  reldepth (32 vs 28), like the te control. The 70b job has no independent candidate: a second 70B-class candidate would
  add about 77 GB, and the te leg tests the same rule.
- **Pins.** `exit/pins_p4.json` holds the 7 repos (subjects, bases, controls, the te independent candidate), committed
  as `"UNPINNED"`. It is written
  by `scripts/pin_exit.py --phase 4`, which cross-checks `scout resolve` against `git ls-remote` and passes HF_TOKEN to
  git through `GIT_CONFIG_*` env (P2 D29).
- **C0 target set.** The run FAILs before any request unless `set(pins) == P4_REPOS`, the 7 repos above.
  There is no fallback repo or file. Replacing a subject, a selector, a base or a control is a logged human decision
  followed by a plan revision, made before the `P4_PLAN` ledger entry (§2.2 E3). The same holds for the independent
  candidate.
- **Fixed endpoint.** The endpoint is always `https://huggingface.co` on both the client and the worker. A foreign
  `HF_ENDPOINT` makes the run exit 1.
- **Evidence line.** Every E-step prints one JSON line. PIN, PREFLIGHT, PLAN and EXIT must share the client
  `hostname`. The worker hostname is recorded separately.
- **Script-written ledger.** `exit/ledger_p4.jsonl` uses the P3 hash-chained format (`scripts/p3_ledger.py`, extended
  with a `kinds` argument by T321), with the kinds `P4_PREFLIGHT`, `P4_PLAN`, `P4_ATTEMPT` and `P4_EXIT`. Only
  `scripts/exit_check_p4.py` appends to it. `P4_ATTEMPT` is appended **before** the first job is submitted, and
  `P4_EXIT` afterwards, also when the run raises. Every rerun is therefore visible. Reruns are allowed, because the
  statistic is reproducible at fixed pins and code up to float nondeterminism (§2.4), but the EXIT evidence lists the
  number of earlier attempts with the same plan ids.
- **Frozen before any real run.** The following are fixed by this plan and asserted literally by tests and by C2:
  - `ATTRIB_PARAMS`, `QUANT_PARAMS`, `GGUF_PARAMS`, `JOB_PARAMS`, `STORE_PARAMS` (§4.9)
  - the thresholds in §2.4 and the target table above
  - the synthetic fixture constants (T303, T313)

  P2's `SIGMA_PARAMS` and `ALIGN_PARAMS` and P3's `CKA_PARAMS`, `TSTATS_PARAMS`, `FEATURE_PARAMS` and `JEV_PARAMS`
  stay unchanged. The code is frozen from the `P4_PLAN` entry to EXIT: `git diff --name-only <P4_PLAN commit> HEAD --
  scout scripts pyproject.toml` must be empty, and the working tree must be clean on those paths (C2).

**Credentials and hosts.**
- `HF_TOKEN` is needed on the client (pins, header scans) and on the worker (full reads). It must belong to an account
  that has accepted the Llama 3.1 and Llama 3.3 licences. The worker reads `HF_TOKEN` from its environment, or else
  from `~/.cache/huggingface/token`. The token is never placed in a job spec.
- The client needs ssh access to the Spark: key-based, `BatchMode=yes`, and the host name in `SCOUT_SPARK_HOST`.
- The client sets `SCOUT_SPARK_CMD="cd <spark checkout> && <venv>/bin/python -m scout.jobs.worker"` (validation r2
  minors): the `cd` puts the checkout root on `sys.path` (`python -m` adds the working directory), so `import
  scripts.exit_check` works for `wire_audit_available` even with a PEP 660 editable install, and the worker runs the
  checkout whose `code_id` C1 compares. The value is a shell prefix used verbatim (T315).
- Scripts exit 3 before any request when a required credential is missing.
- These are environment limits only. The offline suite needs neither the token nor the Spark.

**Phase status rule.** Phase 4 closes only when E0–E4 have all passed. The P1, P2 and P3 offline suites and their
offline exit rehearsals stay in E0 and are green after every task (§10). The P1–P3 network exits are not re-run:
- plain scans of safetensors repos are unchanged, apart from the Card version (`card.v3`)
- σ-sample and anchor plans of BF16/F16/F32 tensors are byte-identical to P3 (T308 `test_p3_plan_unchanged`)
- P2 C8 and P3 C3 accept `card.v3` (T306, the same change that switches the version)

### 2.2 Commands

```bash
# E0 offline (no network, no Spark, no token): P1-P4 suites incl. the P4 synthetic rehearsal (T321), which runs real
# worker subprocesses (LocalBackend) on a GGUF Q4_K subject and a diffusers text encoder, plus a crash and a sweep
# (FP8 subjects are covered by the T316 worker tests)
pip install -e '.[dev]'
pytest -q                                   # expect exit 0, 0 failures; network tests deselected

# E1 pins (client host, HF_TOKEN with the Llama 3.1 and 3.3 licences accepted)
python scripts/pin_exit.py --phase 4 --pins exit/pins_p4.json
# expect exit 0; last line {"step":"PIN","phase":4,"hostname":...,"repos":{7 entries}}

# E2 Spark preflight (0 weight bytes; ssh to the Spark)
python scripts/exit_check_p4.py preflight --pins exit/pins_p4.json --backend "ssh:$SCOUT_SPARK_HOST"
# expect exit 0; rows C0, C1, C2 and C7a PASS; ledger gains P4_PREFLIGHT; the JSON line has the worker's hello
# (at E2 no plan exists yet: C1's disk clause checks only disk_free_bytes >= the 50 GiB reserve; the full clause,
#  max(disk_bytes) + reserve, is evaluated at E3 and E4)
# (machine, mem_total_bytes, disk_free_bytes, scratch_root, git_commit, code_id, code_trees, git_dirty,
#  wire_audit_available, python, numpy/scipy versions, has_hf_token)
# Setup, not intervention: the Spark checkout must be at a commit whose CODE_PATHS trees equal the client's (C1 compares
# code_id = hash of `git rev-parse HEAD:<p>` over scout, scripts, pyproject.toml; the commits themselves may differ)

# E3 plan only (0 weight bytes): header scans into the store, the two full plans displayed in full
python scripts/exit_check_p4.py plan --pins exit/pins_p4.json --backend "ssh:$SCOUT_SPARK_HOST" --store cards/p4-store
# expect exit 0; 2 plans (format_full_plan: plan_id, bytes planned, hard cap, disk on <host>:<scratch_root> with
# free space, memory peak, reason, per-model and per-file tables); "CONFIRM WITH: --confirm-plans <id70b>:<cap>,<idte>:<cap>";
# "TOTAL CAP: <N> B"; client totals.weight == 0; te plan-time C15 rows (component, tokenizer file) PASS; C1 re-evaluated
# with disk_needed = max(disk_bytes) of the two plans (about 179 GB + 50 GiB reserve) PASS;
# ledger gains P4_PLAN {plans, git_commit, code_id}.
# At plan time (published configs; the exact values come from the E3 plans):
#   70b  bytes_planned 179000967168  cap 179700189696   (slack = bytes_planned/256)
#   te   bytes_planned  32549773312  cap  32818208768   (slack = 4 x 64 MiB; incl. the independent candidate)
# -> commit exit/ledger_p4.jsonl (exit/ is outside CODE_PATHS: HEAD moves, code_id does not, so C1 and C2 still hold
#    and the Spark checkout needs no update)

# E4 the confirmed run; the human copies the CONFIRM WITH line (invariant 2); nothing else is manual
python scripts/exit_check_p4.py run --pins exit/pins_p4.json --backend "ssh:$SCOUT_SPARK_HOST" --store cards/p4-store \
       --confirm-plans <id70b>:<cap70b>,<idte>:<capte>
# expect exit 0; both jobs run once, sequentially, with no prompt; table C0..C16 all PASS; then a second attribute
# call per target is served from the store (C13); ledger gains P4_ATTEMPT then P4_EXIT; the last line is the JSON evidence
```

- The subcommands `preflight`, `plan` and `run` are mutually exclusive, and `run` requires `--confirm-plans`.
- Plan ids are deterministic at a pin, so E3 and E4 produce the same ids. If they differ, the job is declined and
  C4 FAILs.
- `run` also performs the preflight and the planning steps, and FAILs if the plan ids differ from the last `P4_PLAN`
  entry.

### 2.3 Checks (P4 namespace; one row per check: `target | check | expected | actual | PASS/FAIL`)

C3–C15 are evaluated per target (`70b`, `te`); C10b only for `te`.

| # | Check | Threshold |
|---|---|---|
| C0 | target set: `set(pins) == P4_REPOS`, all pins 40-hex | equal, else FAIL and exit 1 before any request |
| C1 | Spark environment, from the worker hello: `machine == "aarch64"`, `system == "Linux"`, `mem_total_bytes >= 100 GiB`, `disk_free_bytes >= max(disk_bytes) + JOB_PARAMS.disk_reserve_bytes` (at E2, before any plan, `max(disk_bytes)` is 0; the full clause is evaluated at E3 and E4), `wire_audit_available`, `has_hf_token is True`; worker `code_id ==` client `code_id` (tree hashes of `CODE_PATHS`, not commits; D78), both clean on `CODE_PATHS` | all true (a FAIL means an assumption of §5 does not hold: human decision) |
| C2 | frozen params: `ATTRIB_PARAMS`, `QUANT_PARAMS`, `GGUF_PARAMS`, `JOB_PARAMS`, `STORE_PARAMS` equal the literals of `exit_expectations_p4.py`, on the client **and** in the worker hello (`params_digest`); P2/P3 params equal their frozen literals; code frozen since `P4_PLAN` (E4 only): client `code_id ==` the `P4_PLAN` entry's `code_id` and clean | equal |
| C3 | plan: models `== [subject, base, control]` (70b) or `[subject, base, independent, control]` (te); for each model the tested stack depth `==` the config's layer count (`num_hidden_layers`, or `text_config.num_hidden_layers`); every block planned; `bytes_planned ==` Σ `nbytes` over the planned Parquet rows (gate, up and norm of every block, the whole embedding, FP8 scales), recomputed independently by the check from the stored Cards; `bytes_cap == bytes_planned + max(16 MiB, 4 × min(chunk, largest read), ceil(bytes_planned / 256))`; `disk_bytes == bytes_planned` | equal |
| C4 | gate: the client's plan-time snapshot has `totals.weight == 0`; in the run, exactly 1 approved client decision via `cli-flag` with `bytes_confirmed == bytes_cap`, whose `(plan_id, bytes_cap)` is in `--confirm-plans`, and exactly 1 worker decision via `job-spec` with the same pair; worker `plan_id ==` client `plan_id` | true |
| C5 | bytes and streamed log: worker `bytes_planned <= totals.weight <= bytes_cap`; the ingested event count `==` the worker's `n_events`, and the worker seqs of the ingested events, in arrival order, are exactly `1..n_events` (the worker emits in seq order, D77); client totals of origin `<worker host>` `==` the worker's result totals | true |
| C6 | worker wire (`wire_audit`): Σ body bytes of the worker's CountingTransport `==` worker ByteLog `meta + header + weight`; hosts ⊆ {`huggingface.co`, `*.hf.co`}; every request carrying a `Range` header is attributed to `(repo, file)` through the redirect chain (P1 `CountingTransport.orig_path`, repo from the `/{owner}/{name}/resolve/` URL of the chain's first request) and lies within that file's registered header bound or inside one planned read of that model; an unattributable ranged request FAILs | true |
| C7 | disk and purge: (a) scratch audit before the job: 0 non-lock files; (b) worker `disk_peak_bytes <= disk_bytes`; (c) purge report `dir_absent`, `resident_after == 0`, `bytes_written == bytes_planned` and `bytes_removed >= bytes_written` (the removed bytes include `index.json`); (d) the live log has the COMPUTE `purge` event before the PURGE check event `PURGE check: disk_resident=0 files=0 scratch=absent`; (e) a scratch audit after the job, run over the backend: **0 non-lock files and 0 bytes** under `scratch_root` | all true |
| C8 | no manual intervention: exactly 1 job submitted per target in this run, `status == "succeeded"`, worker exit code 0; the runner's stdin is never read | true |
| C9 | resources: job wall-clock `<= max_wall_s` (70b 21600 s, te 5400 s); COMPUTE stage `<= max_compute_s` (70b 7200 s, te 1200 s); worker `rss_peak_bytes <= 48 GiB` | true |
| C10 | **positive attribution** (base): reference `status == "tested"`, `null_ok == true`, `control_ok == true`; `n_tested ==` depth; **attributed fraction `>= 0.95`**; the share of blocks whose `primary` is the base `>= 0.95`; median `z_adj >= 10.0` | true |
| C10b | **independent candidate** (te only; `allenai/OLMo-2-1124-7B`, role `reference`): `status == "tested"`, `null_ok == true`, `control_ok == true`, `n_tested ==` depth, **`n_attributed == 0`**, and it is the `primary` of 0 blocks | true |
| C11 | **negative control** (role `control`): `status == "tested"`, `n_tested ==` depth, **0 control tests with `z_adj ≥ crit` under leave-one-out**, `control_ok == true` | true |
| C12 | Card v3 and store: the attributed subject Card has `schema_version == "card.v3"`; `stats.attribution` validates as `attribution.v1` with `len(blocks) ==` depth; the store has it under `(repo, sha, component)` with `has.attribution`; recomputing the object id from the two files gives the stored id; `build_view` gives the tested stack's strip `depth` cells, each with `attribution.status` set, and `primary_title` equal to the base title for attributed cells | true |
| C13 | cache: a second `attribute()` for the same subject and references returns the same object id, submits 0 jobs, and its client CountingTransport records **0 requests** | true |
| C14 | 70b GGUF: subject Card `weights.format == "gguf"`, `quant.method == "gguf"`, `"Q4_K" in quant.types`; the subject scan's header bytes for the GGUF file `== gguf.infos_end <= gguf.data_offset` and its `totals.weight == 0`; tested stack prefix `model.layers`; gate/up shapes equal the base Card's for every block | true |
| C15 | te: subject `key.component == "text_encoder"`; the tested stack is the text stack (its depth equals `text_config.num_hidden_layers`, or `num_hidden_layers`); the vision stack is listed in `attribution.other_stacks` with the reason `no probe basis`; the subject Card's `stats.tokenizer_minhash.source_file` ∈ {`tokenizer/tokenizer.json`, `tokenizer/vocab.json`} (whichever the repo ships, **V**; the component and tokenizer clauses are also evaluated at E3 from the stored header-scan Card) | true |
| C16 | disclaimers: the `scout attribute` text output and the view carry the three `DISCLAIMERS` plus `ATTRIBUTION_CAVEAT` | true |

**E2** (`preflight`) evaluates C0, C1 (disk against the reserve only), C2 and C7a. **E3** (`plan`) re-evaluates C1 with
`disk_needed = max(disk_bytes)` of its plans and adds C3, the plan-time half of C4 and the plan-time C15 clauses (te).
**E4** (`run`) evaluates everything.

The `P4_EXIT` payload records, per target:
- plan_id, bytes_planned, bytes_cap, weight bytes, disk_peak, wall_s, compute_s and rss_peak
- `n_tested` and `n_attributed` per reference (base, independent candidate, control)
- the null summary per reference (`n_shift`, `shift_mean`, `shift_sd`, `shift_max`, `n_control`, `control_mean`,
  `control_sd`, `mu0`, `s0`, `kappa`, `df`, `guard`) and `crit`
- the median and minimum `z_adj` and the minimum `z_pair` for the base; the maximum `z_adj` and `z_pair` for the
  independent candidate; the maximum (leave-one-out) `z_adj` for the control
- the object id

It also records, per target, whether the subject's and the base's `weights.content_digest` are equal (a byte-identical
copy attributes trivially; validation r1 minor), and the worker hello, the backend string, `git_commit`, `code_id`,
`git_dirty` and `prior_attempts`, whether or not a check uses them.

### 2.4 Why these numbers (frozen in T320; exit numbers set by planner under the 2026-09-28 delegation)

- **α = 0.01, Bonferroni over all k = 0 tests of a job, Student-t critical values (§3.8 step 7).** 70b: 80 blocks ×
  (base + control) = 160 tests, n_c = 80, crit 4.036 (control LOO 4.038). te: 28 blocks × (base + independent
  candidate + control) = 84 tests, n_c = 28, crit 4.233 (control LOO 4.258). A candidate block is attributed only if
  both `z_adj ≥ crit` and the per-depth paired margin `z_pair = (z − z_ctrl at the same position)/(s0·√2) ≥ crit` hold
  (§3.8 step 7; validation r2 major 1). The family-wise bound (≤ 1 % false block attributions per job) is conservative
  under assumption A-null of §3.8: at each depth, an independent candidate's aligned z and the control's are
  exchangeable draws. The depth profile of their common mean may be anything, including a spike at block 0. It is not
  an unconditional guarantee.
- **Attributed fraction ≥ 0.95 and median z_adj ≥ 10 for the base (C10).**
  - *Synthetic measurement at the exit's m = 2048* (§3.8 step 6 table, r1 null): derived blocks at ε = 1.0 give raw z
    37.5–44.7 and median z_adj 25.0–43.7, 12/12 attributed in all 16 jobs, including the depth-local settings. At
    ε = 2.0 the median z_adj falls to 2.2–16.3.
  - *Scaling.* z scales with `√(m−1)`: the ceiling is `√2047 = 45.2` for an identical block, and z_adj = (z − μ0)/κ
    with κ ≥ 1.006 at n_c = 80. The control-calibrated null costs 5–37 % of z against raw z in the m = 2048 runs (larger when
    the control's own z is inflated, which is the point).
  - *What the targets imply.* R1-Distill is SFT on about 800k samples, and the Qwen-Image text encoder is expected to
    equal Qwen2.5-VL-7B (whether its weights are byte-identical is recorded, §2.3 payload). Both sit far below ε = 1.
    The 0.95 allows up to 4 of 80 (or 1 of 28) blocks to fail for a reason nobody anticipated; that failure is
    reported, not hidden. The median bound of 10 is about 2.5 × crit, so the base must be attributed by a wide margin.
- **Control: 0 tests above crit under leave-one-out and control_ok (C11).** The control shares width (70b) or
  architecture family (te) with the subject, but not training. It calibrates the null (§3.8), and its own blocks are
  tested against the rest of the pool. A control block above crit means the null is not homogeneous across depth or the
  control is not independent; every reference then abstains, and C10 and C11 FAIL. That goes to the human. Since r2 the
  control self-test no longer carries the family-wise bound (the paired margin does); it is a conservative homogeneity
  diagnostic, and its FAILs are abstentions, never false attributions.
- **Independent candidate: 0 attributed blocks (C10b, te).** The one live specificity check on real weights. Under
  A-null the Bonferroni share of its 28 tests is 28/84 × 0.01 ≤ 0.0034; measured with the r2 rule, its false-attribution
  rate is ≤ 0.0003 in every simulated case (§3.8 step 6 Monte Carlo).
- **False-FAIL probability of a correct implementation, and reruns (validation r1 minor; restated in r2, major 1).**
  Measured with /tmp/p4r2/fwer2.py (20,000 jobs per row, the T314 rule reimplemented; §3.8 step 6). These numbers are
  **conditional on the depth profile of the real null**, which nothing measures before E4:
  - *Depth-homogeneous null* (every row with a flat mean, including mean 3 with sd 1.5, and AR(1) correlation across
    depth up to φ = 0.8): the control self-test fires (C11 FAILs, and with it C10) with probability ≤ 0.0052 per job at
    L = 80 and ≤ 0.0027 at L = 28. The shift guard fires with ≤ 0.0025. C10b fails with ≤ 0.003 (control veto
    and guard included; ≤ 0.0003 from its own tests). Over both targets a correct implementation therefore FAILs with probability ≤ about 1.5 %.
  - *A depth spike* (the common mean of candidate and control raised at one depth, e.g. block 0, whose probe input is
    literally that block's real input): the control's own test fires at the spike, so every reference abstains, and
    C10/C11 FAIL. For one spiked block with mean 4 this happens with probability 0.37 (L = 80) and 0.24 (L = 28); with
    mean 5, 0.75 and 0.59; with mean ≥ 6, ≥ 0.87. For blocks 0–2 at mean 4 it is 0.52 and 0.05. For a smooth bump of
    height 4–6 over about L/5 depths it is ≤ 0.014.
  - *What the spike FAIL is.* It is an abstention, not a false attribution. With the r2 rule the false-attribution
    rate in every spike case is ≤ 5 × 10⁻⁵ (r1: up to 0.20). The exit is designed to FAIL loudly rather than attribute
    when the control reveals depth structure, and that FAIL goes to the human. The probability is stated honestly
    instead of being engineered away: no threshold, α or control was changed to lower it.
  - *Reruns.* The statistic is reproducible at fixed pins and code up to float nondeterminism (BLAS summation order
    across thread counts, and near-ties in the linear assignment), so a rerun almost always reproduces such a FAIL. It
    can only be resolved by a logged human decision (a new control, recorded with its documentation, followed by a plan
    revision before a new `P4_PLAN`), never by rerunning or by changing a threshold.
- **Resources (C9).** These are bounds, not targets.
  - *Wall clock.* 179 GB at ≥ 10 MB/s is ≤ 5 h, plus ≤ 1 h of compute (§2.5), for a 6 h cap on the 70b job. The te job
    reads 32.5 GB (54 min at 10 MB/s) plus ≤ 10 min of compute, for a 1.5 h cap. r1's 1 h cap was raised with the added
    candidate's 6.6 GB, before any real run; it is a byte-derived bound, not a fitted one.
  - *Memory.* 48 GiB is 4.5 × the §2.5 estimate of about 10.5 GiB and well under the 128 GB unified memory.
  - *What a FAIL means.* Something is wrong with the host or the code. It is never answered by raising the bound.
- **Quantised subject.** The 70b subject is Q4_K_M. By the measurements of §3.7, dequantisation noise does not move z
  (Q4_K at ε = 0.05: z = 16.0, the same as unquantised; null with a Q4_0 side: mean 0.10, sd 1.03). So the thresholds
  are the same for quantised and unquantised subjects.

### 2.5 Environment, access, bytes, time

**DGX Spark assumptions (each marked V: verify at E2; C1 checks the first four).**
- **V** One NVIDIA GB10 node, Linux `aarch64` (DGX OS, Ubuntu-based), 128 GB unified CPU+GPU memory. The hello reports
  `mem_total_bytes` of about 119–128 GiB.
- **V** About 4 TB NVMe, with the scratch root (`SCOUT_SCRATCH_ROOT`, default `~/.cache/scout/scratch`) on it, and at
  least `max(disk_bytes) + 50 GiB` (about 233 GB) free.
- **V** CUDA is installed. **P4 does not use the GPU**: numpy and scipy run on the 20-core Arm CPU (D74).
- **V** Python ≥ 3.11 with aarch64 wheels for numpy, scipy, pyarrow and httpx (manylinux aarch64 wheels exist for all
  four), and scout installed editable (`pip install -e .`) from a git checkout whose `CODE_PATHS` trees equal the
  client's (C1: `code_id`; `wire_audit_available` needs `import scripts.exit_check`, and `scripts` is not packaged).
- **V** `SCOUT_SPARK_CMD` on the client is `cd <spark checkout> && <venv>/bin/python -m scout.jobs.worker`, so the
  worker runs from the checkout root and `scripts` is importable (a PEP 660 editable install alone does not put the
  repo root on `sys.path`). C1's `wire_audit_available` checks the result at E2.
- **V** sshd is reachable from the client host, with key auth.
- **V** Outbound HTTPS to `huggingface.co`, `*.hf.co` (`cdn-lfs*.hf.co`, `cas-bridge.xethub.hf.co`).
- **V** `HF_TOKEN`, or `~/.cache/huggingface/token`, is present on the Spark (C1: hello `has_hf_token`), for an
  account that accepted the Llama 3.1 and 3.3 licences (only E4 can show the latter).
- **V** Internet throughput is unknown. The C9 bound assumes ≥ 10 MB/s sustained; below about 9.5 MB/s the 70b job
  cannot meet its 6 h cap (179 GB in 5.25 h after ≤ 0.75 h of compute). Nothing measures it before E4 (§11 deferred).

| Step | Needs | This container |
|---|---|---|
| E0, all implementation | PyPI: numpy, scipy (new, T314), pyarrow, httpx, pyyaml, pytest, anthropic | available (scipy 1.17.1 installed in a scratch venv) |
| E1, E3 (client) | `huggingface.co` API + git; HF_TOKEN; about 65 MB meta+header (seven tokenizers, about 30 + 37 + 3 shard headers, one GGUF header of about 8 MB) | blocked (proxy 403 on huggingface.co) |
| E2–E4 (worker) | ssh to the Spark; Spark egress to the Hub; HF_TOKEN on the Spark | no Spark from this container |
| E4 bytes | about 211.6 GB of weight ranges (179.0 GB + 32.5 GB), the same on the Spark's disk, peak 179 GB at a time | — |

**Byte estimate at plan time (published configs; the E3 plans are the exact values).**

| job | model | tensors planned | bytes |
|---|---|---|---|
| 70b | subject GGUF Q4_K_M | gate+up Q4_K 80 × 2 × 28672×8192 (144 B per 256 weights) = 21,139,292,160; `token_embd` Q4_K 591,003,648 (**V** type); `ffn_norm` F32 2,621,440 | 21,732,917,248 |
| 70b | base Llama-3.3-70B-Instruct BF16 | gate+up 75,161,927,680; embed 2,101,346,304; norms 1,310,720 | 77,264,584,704 |
| 70b | control Qwen2.5-72B BF16 | gate+up 77,510,737,920; embed 2,491,416,576; norms 1,310,720 | 80,003,465,216 |
| 70b | **total** / slack (bytes_planned/256) / cap | | **179,000,967,168** / 699,222,528 / 179,700,189,696 |
| te | subject Qwen-Image text_encoder (**V** BF16) | text stack only: gate+up 28 × 2 × 18944×3584 × 2 = 7,604,273,152; embed 1,089,994,752; norms 200,704 | 8,694,468,608 |
| te | base Qwen2.5-VL-7B-Instruct BF16 | same shapes | 8,694,468,608 |
| te | independent candidate OLMo-2-1124-7B (**V** BF16) | gate+up 32 × 2 × 11008×4096 × 2 = 5,771,362,304; embed 100352×4096 × 2 = 822,083,584; norms 262,144 | 6,593,708,032 |
| te | control Llama-3.1-8B BF16 | gate+up 7,516,192,768; embed 1,050,673,152; norms 262,144 | 8,567,128,064 |
| te | **total** / slack (4 × 64 MiB) / cap | | **32,549,773,312** / 268,435,456 / 32,818,208,768 |

**Compute estimate (70b; measured on this 4-core x86 container, scaled conservatively).**
- *Neuron responses.* A reference block's response matrix `X·Wᵀ` (2048 × 8192 × 28672, float32) takes 2.4 s. There
  are 2 per block, per reference and per probe set, so 320 in total: about 13 min.
- *Correlations.* The correlation products (2048 × 2048 × 29568) number 960 at about 1 s: about 16 min.
- *Matching.* A rectangular linear assignment of 2048 × 28672 takes 0.4–0.5 s (scipy 1.17.1). There are 960: about 8 min.
- *Decoding.* Decoding and reading 179 GB from NVMe: about 5 min.
- *Total and memory.* COMPUTE is about 45 min. The memory estimate is 10.5 GiB (§3.2).

The te job (3 references × 28 blocks at smaller widths) is about 10 % of this.

## 3. Definitions

### 3.1 Stages with full downloads

A job's worker log runs RESOLVE > HEADERS > META > FULL_DOWNLOAD > COMPUTE > PURGE (the P1 `Stage` enum; order
enforced). On the client, each header scan runs P1's stages in its own ByteLog, which is embedded in that scan's Card.
The client's attribution log then runs FULL_DOWNLOAD (the gate decision only; the client fetches no weight byte) >
REPORT > PURGE, and the ingested worker events join it without changing its stage.

- **FULL_DOWNLOAD** reads whole tensors to the job's scratch directory on disk, under a grant that is now allowed in
  this stage (T301).
- **COMPUTE** decodes tensors from scratch and runs the attribution. It ends in a `finally` block that deletes the
  scratch directory and emits a `purge` event: `released <n> weight bytes from disk; disk_resident=0 files=0`.
- **PURGE** re-checks that the directory is absent and that a sweep of the scratch root finds nothing for this job,
  and emits `PURGE check: disk_resident=0 files=0 scratch=absent`. Otherwise it raises `PurgeError` (exit 8).

No weight byte exists on disk or in memory after COMPUTE (invariant 1). The worker writes no Card; the client writes
the Card in REPORT (§3.4).

### 3.2 Full-download plan and gate (invariant 2; T310)

A full plan is built from **Cards only**. The client uses its stored Cards; the worker builds identical Cards in memory
from its own header reads (§3.3).

**What each model contributes.** The *tested stack* is the largest stack whose every block has the attribution roles
(`ATTRIB_PARAMS.roles`: gate, up, norm) as 2-D (norm 1-D) tensors of a decodable dtype, with gate/up input width equal
to the input-embedding width. For that stack the plan contains, as `FullRead`s:
- gate, up and norm of every block
- the whole input embedding (P3 `cka.find_embedding`), which is the probe basis
- for an FP8 tensor, its companion scale tensor

A model with no tested stack is `untestable`. If the subject is untestable the plan is refused. An untestable reference
is kept in the plan with no reads, and its status is reported.

**Bytes, cap, disk and memory.**
- `bytes_planned` = Σ `(end − start)`.
- `bytes_cap = bytes_planned + slack`, with `slack = max(16 MiB, 4 × min(chunk_bytes, largest read), ceil(bytes_planned / 256))`
  (D60). This resolves the P2/P3 deferred "retry slack scales with the plan" item for full plans. Chunks are 64 MiB, so
  one failed attempt costs at most 64 MiB, and the slack allows about 11 failed chunks per 179 GB.
- `disk_bytes = bytes_planned`, since tensors are stored in their on-disk dtype.
- `memory_peak_bytes = 4 × (4 × G + 4 × P × I + 2 × L_s × m² × S) + workers × chunk_bytes`, where:
  - G is the largest gate numel
  - P = `n_probe` and I is the largest intermediate size
  - L_s is the number of subject blocks, `m = m_test` and S is the number of distinct probe sets
  - For the 70b job this is ≈ 10.5 GiB.

**The gate flow.** The flow is P2's, with the same exactness rule.
1. `format_full_plan` is displayed and logged as a `gate` event before any decision. It shows:
   - plan_id, reason, bytes planned, hard cap, disk on the worker, memory peak and threshold
   - a per-model table (repo@sha, component, tested stack, blocks, bytes)
   - a per-file table
   - the untestable models
2. Auto-approval happens only below the threshold (`totals.meta + totals.header + bytes_cap <= gate_threshold_bytes`).
3. Otherwise a confirmer must return `plan_id` equal and `bytes_confirmed == bytes_cap` exactly.

The CLI uses `--confirm-plan ID --confirm-bytes CAP`, and the exit runner uses `--confirm-plans ID:CAP,...`.

**Where the gate is enforced (D57).** Bytes are fetched on the worker, so the worker enforces the gate:
- It re-derives the plan from its own header reads at the pinned sha. If its `plan_id` or `bytes_cap` differs from the
  spec, it raises `PlanMismatch` (exit 7) before any weight byte.
- It then runs the same gate with a confirmer that returns the spec's client decision (`via "job-spec"`). It checks the
  exact pair again, and grants `(plan_id, ranges, cap, stage=FULL_DOWNLOAD)` on its own ByteLog.

A spec can therefore never widen what the human saw: the ranges come from the pinned headers, and the confirmed
number must equal the cap of exactly those ranges.

### 3.3 Batch jobs on the DGX Spark (T311, T312, T315, T316)

**Transport (D56): ssh + JSON lines, no daemon.**
- `SshBackend(host)` runs `ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4 <host>
  "<prefix> <shlex-quoted args>"`, where `<prefix>` is `SCOUT_SPARK_CMD` used verbatim as a shell fragment, default
  `python3 -m scout.jobs.worker` (T315; validation r2 minor: the remote command line is the same module entry the
  offline rehearsal runs, never a CLI subcommand whose argument forwarding is a separate contract). It writes the job spec (`job.v1`, §4.4) on stdin and reads `jobproto.v1`
  messages (§4.5), one JSON object per stdout line. Remote stderr lines are passed to the client display, prefixed with
  the host.
- `LocalBackend` runs the same worker as a local subprocess (`sys.executable -m scout.jobs.worker`). It is used on the
  Spark itself and by the offline rehearsal.
- `InProcessBackend` runs `run_job` in a thread. It is used by unit tests with FakeHub.
- Rejected alternatives:
  - an HTTP job API on the Spark: another server, plus auth and TLS
  - Redis/Celery: infrastructure
  - Slurm: not installed on a Spark by default

  ssh is already there, authenticates with keys and streams both ways.

**Worker behaviour (T316 `run_job`).**
1. **Validate.** The spec is validated (`job.v1`). `params_digest` must equal the worker's own; otherwise `JobError`
   "worker code differs from client". This is how client and worker agree on code without trusting the spec.
2. **Lock and sweep.** The worker takes the global worker lock (`fcntl.flock` on `<scratch_root>/.worker.lock`), which
   makes jobs serial. A busy lock waits up to `lock_wait_s` = 3600 s, then raises `JobError("worker busy")`. It then
   **sweeps orphans** (below).
3. **Inspect.** `scan.inspect_all(targets)` runs RESOLVE, then HEADERS, then META across all models of the job (stages
   interleaved across targets, so the order never goes backwards). It builds in-memory Cards and reads tokenizer bytes.
   Nothing is written. Model i logs through `log.view(f"m{i}:")`, so identical file names in different repos never
   share a header registration or a grant range. The grant ranges are `(f"m{i}:{path}", start, end)`.
4. **Plan and check.** `fullplan.plan_attribution(...)` builds the plan, and the plan_id/cap check of §3.2 runs. The
   disk check follows: free space at `scratch_root` must be ≥ `disk_bytes + disk_reserve_bytes` (50 GiB), else
   `DiskSpaceError` (exit 9) before any weight byte.
5. **Download.** FULL_DOWNLOAD: gate, grant, `ScratchDir.create` (holding `.lock` in the job dir), then `download_plan`
   (64 MiB chunks, 4 threads, through `Source.read_range`, so every byte is preflighted, grant-checked and logged).
6. **Compute.** COMPUTE: `attrib.attribute(...)` over a `ScratchProvider`. It reads each tensor with `numpy.fromfile`
   (a copy, never `mmap`), so deleting the file frees its space at once, and decodes/dequantises it (§3.6). Then, in
   `finally`, `scratch.purge()` and the `purge` event.
7. **Purge check.** PURGE: the check of §3.1.
8. **Result.** A `result` message (`JobResult`, §4.5), then `bye`.

Throughout, the worker sends a heartbeat every 30 s from a thread. Every message is written with flush, under one
output lock, so heartbeat and event lines never interleave. Event lines leave in seq order: the worker's ByteLog calls
`on_event` (which writes the line) in strictly increasing seq order, also with 4 download threads (D77; a P1 contract
change owned by T301, §9). If stdout breaks (`BrokenPipeError`: the client or ssh died), the worker aborts. The
`finally` block of COMPUTE, or of FULL_DOWNLOAD, purges on every exception path.

The worker's gate threshold is `min(spec.plan.threshold_bytes, DEFAULT_GATE_THRESHOLD_BYTES)`: a spec can lower the
worker's auto-approval threshold (the rehearsal uses 0) but never raise it.

**Scratch, disk accounting and purge (T311).**
- **Creation without a sweep race.** The job directory is created as `<scratch_root>/.pending-<job_id>-<rand>/`, its
  `.lock` is flocked, and only then is it renamed to `<job_id>/`. A job directory therefore never exists under its final
  name without a held lock. The sweeper skips `.pending-*` directories younger than 300 s and treats older ones like
  any other directory (T311).
- **Layout.** `<scratch_root>/<job_id>/` (mode 0700) holds:
  - `.lock` (flock, held for the job's lifetime)
  - `index.json` (tensor names and sizes; no weights)
  - `m<model>/<sha1(tensor)[:16]>.bin` (raw on-disk bytes of one tensor)
- **Accounting.** Every write goes through `ScratchDir.write`, which updates `bytes_written` and calls
  `log.disk_add(n)`. `ByteLog` tracks `disk_resident` and `disk_peak`.
- **Purge.** `ScratchDir.purge()`:
  1. stats and unlinks every file, summing `bytes_removed`
  2. removes the directory and releases the lock
  3. verifies `dir_absent` and `resident_after == 0` from a fresh listing
  4. calls `log.disk_release`

  It returns `PurgeReport`. `ok` requires `dir_absent`, `resident_after == 0` and `bytes_removed >= bytes_written`.
- **Crash mid-job: the orphan sweeper.** A killed worker (SIGKILL, OOM, power loss) leaves its job directory, but the
  kernel releases its flock.
  - `sweep(root)` visits every `<root>/<job_id>/`. It tries `flock(LOCK_EX | LOCK_NB)` on its `.lock`:
    - acquired: the directory is an orphan, and the sweeper deletes it and reports the files and bytes
    - held: the job is active and is skipped
  - A directory without `.lock` is an orphan.
  - Files directly under the root, other than `.worker.lock`, are reported as stray.
  - The sweeper runs automatically at every worker start (step 2). It is also available as `scout jobs sweep`, with a
    documented cron line for the Spark (`*/15 * * * * scout jobs sweep --json > /dev/null 2>&1`). The cron writes no
    file: its result is visible in the next hello, job result (`sweep_before`) or `scout jobs audit`.
- **Client side.** On disconnect, the client marks the job failed and immediately runs `backend.sweep()` (for ssh:
  `ssh host "<prefix> sweep --json"`), then prints its report. A still-running worker keeps its lock and is not
  touched; it aborts on its next write to the broken pipe.
- `scout jobs audit` lists every non-lock file and byte under the root, read-only. C7a and C7e use it.

### 3.4 Logs streamed back into the ByteLog (T301, T315)

- Each `event` message carries a worker `LogEvent.to_dict()`. Its `stage` is a `Stage` name or `"NONE"` (P1's label for
  events logged before the first `stage()` call, such as the worker's sweep note); ingest accepts both. The client
  calls `ByteLog.ingest(event, origin=<worker hostname>)`, which appends a `LogEvent` with:
  - a new client `seq`
  - the worker's stage name, event, path, url, range, status, bytes, `bytes_by_class`, attempt, elapsed and note
  - `origin` set (a new optional field; `to_dict` emits it only when set)
  - the updated client totals
- Ingest never calls `stage()`, so the client's stage order is unaffected.
- **Ingest errors (T315).** Any exception while handling a worker message (a malformed line, an event `ingest`
  rejects, a missing key) kills the worker channel, marks the job failed, runs `backend.sweep()` and appends the
  JobRecord. `run_job_via` returns the failed outcome and never raises; T317 turns it into `JobError` and displays the
  bytes the worker fetched (`totals_by_origin`) with the sweep report. The killed worker fails its next write and
  purges in its `finally`.
- Client totals therefore include the bytes the worker fetched. `totals_by_origin()` splits them into `"local"` and
  each host.
- `gate` messages are recorded with `add_gate_decision(decision | {"origin": host})`.
- The CLI prints remote stage lines live as `[<host>] STAGE ...`.
- The subject Card written in REPORT carries the whole log: its scan events, the client's attribution events and the
  ingested worker events. It also carries a `fetch_log.jobs` record (§4.1).

### 3.5 Content-addressed Card store (T309; D62)

**Layout (`store.v1`).**
```
<root>/STORE                                   {"schema": "store.v1"}
<root>/.lock                                   writer lock (fcntl.flock, exclusive)
<root>/objects/<h[:2]>/<h>/<component_slug>.card.json      the Card files, byte-identical to the P1 layout
<root>/objects/<h[:2]>/<h>/<component_slug>.tensors.parquet
<root>/refs/<repo_slug>/<revision_sha>.json    {"schema": "storeref.v1", "repo", "revision_sha", "units": [component...] | null,
                                                "entries": [StoreEntry]}
<root>/by-weights/<digest_hex>.json            {"digest", "keys": [card_key]}   weights content digest -> Card keys
```

**Keys.**
- The **lookup key** is `(repo, revision_sha, component)`, as in P1. A Hub revision is an immutable 40-hex commit, and a
  local folder uses the P1 stat hash.
- The **object id** is the content hash
  `h = sha256("scout.card-object.v1\n" + sha256(json_bytes).hex + "\n" + sha256(parquet_bytes).hex)`. It is derivable
  from the P3 ledger's `card_files` hashes.
- The **weights content digest** (Card v3 `weights.content_digest`) is `"sha256:" + sha256("\n".join(sorted(lfs_sha256
  of the unit's weight files))).hex`, taken from the Hub API's `siblings[].lfs.sha256` (T302). It is null when any file
  has no LFS hash, as for local folders. It identifies byte-identical weights across repos at 0 extra bytes, and
  `by-weights/` indexes it. P4 only reports it; it never replaces a comparison.

**Status of the index files (invariant 1).** The Card objects under `objects/` are the only persisted artifact. `refs/`
and `by-weights/` are derived indexes: `CardStore.reindex()` (T309) rebuilds both from `objects/` alone (with
`refs.units` reset to null, which costs one rescan of a whole-repo target). `STORE` and `.lock` are a schema marker and
a lock file. The store holds no weights: `put_files` accepts only `.card.json` and `.tensors.parquet` files.

**Behaviour.**
- **Dedup.** Identical Card bytes make the same object, and a second `put` is a no-op. A Card carries its scan
  timestamp, so two scans of one pin are two objects. Caching (below) is what avoids the second scan.
- **Caching.** `cached_scan(target, store, ...)`:
  - For a Hub target pinned at 40 hex, and for a local folder (its stat-hash key needs no reads), it looks the key up
    first. If an entry exists for every needed unit with `has ⊇ required options` and the current `params_digest`, it
    returns the entries with **0 network requests**.
  - Otherwise it scans into a temporary directory, puts each Card and returns the entries.
  - An unpinned Hub target is always scanned; resolving would already cost the API call.
  - `refs.units` records the full component list of a whole-repo scan, so a pipeline hit is known to be complete.
- **Attribution cache.** `find_attribution(subject_key, reference_keys, control_keys)` returns an entry with
  `has.attribution`, `attribution_refs == sorted([f"reference:{k}" for the reference keys] + [f"control:{k}" for the
  control keys])` and the current `params_digest`. The role is part of the key (validation r2 minor), as it is part of
  `plan_id`: swapping a base and a control is a different attribution and misses the cache.
- **Invalidation.**
  - Pins never go stale, since a new commit is a new key.
  - Entries record `params_digest`, the sha256 of the canonical JSON of `{card: SCHEMA_VERSION, sigma, align, cka,
    tstats, attrib, quant, gguf}`. A lookup ignores entries with another digest, so a code change that changes a
    frozen parameter forces a rescan.
  - Nothing is ever deleted in P4. There is no gc (later.md), and `scout store ls --stale` shows the superseded entries.
- **Concurrency.**
  - Writers hold the exclusive flock on `<root>/.lock`. Object files are written to temp files and `os.replace`d, and
    refs are read, modified and replaced atomically.
  - Readers take no lock: they read whole files that are only ever replaced atomically.
  - A local POSIX filesystem is required; NFS is not supported (documented).
- **P3 compatibility.**
  - `store.import_tree(cards_dir)` imports P1-layout Cards (e.g. `cards/p3`) **byte-identically**, so every P3 ledger
    `card_files` hash still matches its object.
  - `store.export(entry, out_dir)` writes byte-identical files at the P1 path.
  - P3's scripts, library manifest and ledger are untouched and keep reading `cards/p3`.
  - `scout attribute` resolves library references through the store (`--base repo@pin`), so scanned library Cards are
    reused.

### 3.6 GGUF and quantised inputs (T303, T304, T305, T307, T308)

**Minimal set (D65).** Which quantised inputs are decoded, and why:

| input | decoded | why |
|---|---|---|
| GGUF (llama.cpp) `F32`, `F16`, `BF16`, `Q8_0`, `Q4_0`, `Q4_K`, `Q5_K`, `Q6_K` | yes | GGUF is the dominant format for community quantisations of large models. These eight types cover `Q8_0`, `Q4_0` and the `Q4_K_M`, `Q5_K_M` and `Q6_K` mixtures, which make up most GGUF downloads. All are self-contained blocks, with no companion tensors. |
| safetensors FP8 `F8_E4M3` / `F8_E5M2`, with a companion scale tensor (`<name>_scale_inv` block-wise, e.g. DeepSeek-V3, Qwen3-FP8; or `<name>_scale` per-tensor/per-row, compressed-tensors) | yes (attribution); σ/anchors deferred | Official FP8 releases, and the P3 later.md DeepSeek-V3 item. Decoding is a 256-entry table plus a broadcast multiply. |
| GPTQ, AWQ, bitsandbytes (nf4/int8), AQLM, HQQ, EETQ, EXL2, quanto, torchao | no: detected from `quantization_config.quant_method`, recorded in `weights.quant` with `dequantizable: false` and a reason; the roles are then not found (packed `qweight`), so σ, anchors and attribution report "untestable: <method> packed weights not decoded in P4" | Their packings multiply the test matrix: act-order `g_idx`, exllama v1/v2, AWQ's interleaved nibble order, bnb `quant_state`. Full-precision parents are almost always on the Hub. Deferred (later.md); never silently wrong. |
| other GGUF types (Q2_K, Q3_K, IQ*, TQ*, MXFP4, Q4_1, Q5_0/1) | no: sizes known, so the header parses and bytes are exact; tensors of these types are "not decodable" | Rare in practice, and each needs its own lattice or codebook. Deferred. |

**GGUF header-only reading with provably safe lookahead (D66; T304).**
- **The problem.** The GGUF header (KV pairs, including the 128k-entry token array, then the tensor infos) has no
  length prefix. A fixed-size read could run into the data section.
- **Lower bound.** The parser therefore works on a growing buffer. When it runs out, it computes a **lower bound** on the
  total header length from the minimum encoding of everything not yet parsed:
  - 13 B per unparsed KV (`u64` key length + ≥ 1 key byte + `u32` type + ≥ 1 value byte, minus the 1 key byte = 13)
  - the element size, or 8 B per string, for each remaining array element
  - 24 B per unparsed tensor info (`u64` name length + `u32` n_dims + `u32` type + `u64` offset)
- **Reads.** It reads exactly `[have, bound)`. Since `bound <= end of tensor infos <= data_offset` for any well-formed
  file, **no read can touch the data section**. A malformed file whose bound exceeds the file size raises
  `HeaderError`.
- **Registration.** Before each read, the bound is registered with `ByteLog.set_header_len(path, bound − 8)` (monotone
  for `.gguf`, T301), so every byte classifies as header. `.gguf` files get a default header bound of 24 bytes (the
  fixed prologue), as safetensors files get 8.
- **Round trips.** A string array loses about half of its remaining length per round, so a Llama-3 GGUF header
  (128k tokens, 280k merges, ≈ 8 MB) needs about 40 rounds. The cap is `GGUF_PARAMS.max_rounds` = 256.
- **Format.** Versions 2 and 3 are accepted; v1 or a big-endian file raises `HeaderError`. Alignment is
  `general.alignment` (default 32), and `data_offset = align_up(infos_end, alignment)`. Split files
  (`-NNNNN-of-NNNNN.gguf`) are read header by header.
- **Adapter.** Each GGUF file becomes a P1 `SafetensorsHeader`, so all P1–P3 code (structure, Card, σ plan, anchors)
  works unchanged:
  - `header_len = data_offset − 8`, so `8 + header_len + data_begin` is the absolute offset, and P1 C4 holds with
    `data_bytes = size − data_offset`
  - `TensorInfo.dtype` = the GGML type name, and `shape` = the reversed GGUF dims (PyTorch `(out, in)` order)
  - `nbytes` = numel / block_elems × block_bytes, from `dequant.GGML_TYPES`
  - names canonicalised to HF names for the architectures `llama`, `qwen2` and `qwen3` (`GGUF_PARAMS.name_map`). The
    original names go to the new Parquet column `source_name`. Other architectures keep their raw names, get a
    `scan.notes` entry, and then find no roles (untestable, not wrong).
- **Card fields.** The GGUF config is synthesised from KVs into `config.raw` with HF keys (`hidden_size`,
  `num_hidden_layers`, `intermediate_size`, heads, `rms_norm_eps`, `rope_theta`, `vocab_size`,
  `max_position_embeddings`, `model_type`), plus `gguf_kv` (the scalar KVs). The tokenizer comes from the header's
  `tokenizer.ggml.tokens` and `tokenizer.ggml.token_type`, so MinHash, anchors and probes cost 0 extra bytes. `gpt2`
  tokenizers normalise as byte-level and `llama` (SentencePiece) as metaspace.
- **Rotary rows.** llama.cpp permutes `attn_q`/`attn_k` rows for rotary. Singular values are invariant to row
  permutation, so σ is unaffected, and attribution does not use q/k.

**Dequantisation and dtype normalisation (D67; T305).**
- **GGUF decoders.** Every decoder follows the ggml reference `dequantize_row_*` bit layouts, spelled out in T305
  (**V** against `ggml-quants.c` at a pinned llama.cpp commit). They are vectorised in numpy, and the byte-exact test
  vectors come from hand-packed blocks (T303).
- **FP8.** The FP8 table decodes E4M3FN (bias 7, no inf, `0x7F`/`0xFF` NaN) and E5M2 (bias 15). The value is then
  multiplied by the broadcast companion scale. A scale shaped `(⌈m/b0⌉, ⌈n/b1⌉)` uses `weight_block_size` from the
  config (default `[128, 128]`); `numel == 1` is per-tensor; `(m, 1)` or `(m,)` is per-row. Any other shape is not
  decodable.
- **Normalisation.** All paths produce one normalised representation before any comparison:
  - float64 (σ-curves, anchors) or float32 (attribution)
  - canonical `(out, in)` orientation
  - a non-finite value after decode raises `SigmaError` (σ) or makes the tensor untestable (attribution)

  The Card records the storage dtype per tensor (Parquet `dtype`) and the quantisation (`weights.quant`).
- **σ/anchors on GGUF (T308).** The P2/P3 eligibility sets are extended by `QUANT_PARAMS.sample_dtypes` (the five GGUF
  quant types) without touching `SIGMA_PARAMS`. `sigma.decode_tensor` dispatches those dtypes to `dequant`. Anchor
  rows of a block-quantised embedding are `row_nbytes(dtype, hidden)` apart. With float dtypes, the plans are
  byte-identical to P3.

### 3.7 Effect of quantisation on sigma-curves, CKA, JEV features and the matching test (measured)

**Setup.** This is planner scratch code under `/tmp/p4` (numpy 2.4.6, scipy 1.17.1), not in the repo. Weights are
emulated with RTN quantisers matching each format's block structure:
- Q8_0: 32-blocks, fp16 d
- Q4_0: 32-blocks, d = max/−8
- Q4_K and Q5_K: 256 super-blocks of 8 × 32 with 6-bit scale/min
- Q6_K: 16 × 16 sub-blocks with 8-bit scales
- FP8 E4M3: 128×128 block scale
- two formats P4 does not decode, for reference: bnb NF4 (64-blocks) and a GPTQ-like RTN int4 (g128, without error
  feedback)

The emulations are slightly worse than ggml's scale search, which makes them conservative.

Relative Frobenius error on Gaussian (and Student-t, df 4) weights:

| format | Gaussian | t₄ |
|---|---|---|
| Q8_0 | 0.005 | 0.007 |
| Q6_K | 0.022 | 0.028 |
| FP8 E4M3 b128 | 0.026 | 0.026 |
| Q5_K | 0.038 | 0.046 |
| Q4_K | 0.078 | 0.094 |
| Q4_0 | 0.086 | 0.116 |
| NF4 | 0.092 | 0.112 |
| RTN int4 g128 | 0.118 | 0.195 |

**P2 σ-curve statistic.** This reimplements §3.4 of the P2 plan: top-half 128-point log grid, per-layer centring, cubic
detrend fitted on blocks 1..L−2, and median Pearson over the interior. The setup has 16 layers, v/k roles of 256 × 1024
with power-law spectra of exponent α₀ plus a shared depth trend, a fine-tune of ε = 0.02, and 10 independent null
pairs.

| α₀ (spectrum steepness) | Q8_0 | Q6_K | FP8 | Q5_K | Q4_K | Q4_0 | NF4 | RTN4 | null median r: none → Q4_0 on one side |
|---|---|---|---|---|---|---|---|---|---|
| 0.3 (flat) | 1.0000 | 1.0000 | 1.0000 | 0.9999 | 0.9995 | 0.9994 | 0.9994 | 0.9987 | mean −0.064 → −0.070, max 0.175 → 0.145 |
| 0.6 | 1.0000 | 0.9999 | 0.9999 | 0.9997 | 0.9983 | 0.9979 | 0.9976 | 0.9956 | mean −0.019 → −0.017, max 0.145 → 0.145 |
| 1.0 (steep) | 0.9999 | 0.9982 | 0.9961 | 0.9867 | 0.9324 | 0.9209 | 0.9063 | 0.8478 | mean −0.018 → +0.007, max 0.153 → 0.233 (one pair moved by 0.133) |

The values are base vs Q(base). Base vs Q(fine-tune) is lower by at most 0.009.

**Why the effect is small.** Quantisation adds an independent noise floor. P2 scores only the top half of the spectrum,
where the floor matters only for steep spectra.
- For α₀ ≤ 0.6 the null does not move (max |Δ| 0.030), because the noise is independent between models and cannot
  create correlation.
- At α₀ = 1.0 the floor reshapes the upper spectrum of the quantised side. One of 10 null pairs moved by 0.133, to a
  maximum null median of 0.233. That is still 0.17 below Y = 0.40, and the P2 in-run shift null (`z_shift`) sees the
  same run.

**P3 anchor CKA (debiased).** Base vs Q(base) is ≥ 0.9994 for every format; the lowest is RTN4 at 0.99939. Base vs
Q(fine-tune) is ≥ 0.9994. The independent-pair CKA is unchanged to 3 decimals (0.956 in this high-semantic setting,
with or without Q4_0).

**P3 JEV distance features** (`log_topk_dist`, `log_fro_dlog`):

| pair | log_topk_dist | log_fro_dlog |
|---|---|---|
| fine-tune ε = 0.02 | −3.37 | −3.50 |
| fine-tune ε = 0.05 | −2.90 | −2.86 |
| fine-tune ε = 0.2 | −1.83 | −1.70 |
| quantisation alone: Q8_0 | −3.77 | −3.94 |
| FP8 | −3.23 | −3.30 |
| Q5_K | −3.07 | −3.05 |
| Q4_K | −2.59 | −2.43 |
| Q4_0 | −2.51 | −2.89 |

Quantisation alone looks like a light fine-tune (ε ≈ 0.05–0.1). That is inside the derived range. It only adds
distance, which pushes towards `not_derived` or abstain, never towards a false `derived`.

**The matching test (§3.8), 8 blocks of 1024 neurons, m = 256.** Subjects were derived at ε = 0.05 and at ε = 1.0, then
quantised with every format:

| subject | z | gate/up agreement |
|---|---|---|
| ε = 0.05, every format | 16.0 in every block | 1.000 |
| ε = 1.0, every format | min 14.7 | 0.955–0.958 |

The unquantised ε = 1.0 subject gave z 14.7–15.3 and 0.958, so quantisation changes nothing measurable. The null with
Q4_0 on one side: mean z 0.10, sd 1.03, max 1.87 over 32 blocks.

**Verdict on the frozen thresholds.**
- The P2 exit thresholds (X = 0.80, q10 0.50, Y = 0.40, gap 0.40) and the P3 JEV thresholds and features **still hold
  for GGUF and FP8 inputs**. No constant changes.
- The steepest measured geometry at 4 bits (r 0.92–0.93) keeps a margin of 0.12 over X. Its null maximum (0.233)
  keeps a margin of 0.17 under Y.
- Lower-bit GGUF types (Q2_K, Q3_K, IQ*) are not decoded, so they never reach a comparison.
- **Documented caveat, no adjustment.** JEV was calibrated on unquantised pairs only. For a quantised pair its
  probabilities are not validated, and the shift is towards `not_derived`/abstain. Validation on quantised pairs is
  in later.md.
- The attribution thresholds (§2.4) are the same for quantised and unquantised inputs.

### 3.8 Block-level attribution: the Zhu et al. matching test (T314; D68, D69)

**Source (V).** S. Zhu, A. Ahmed, R. Kuditipudi, P. Liang, *Independence Tests for Language Models*, ICML 2025,
arXiv 2502.12292. The task brief cites "Zhu, Mu et al."; the author list above is the planner's recollection. **V**:
the authors, and the exact form of the unconstrained test statistic, are checked against the paper before the
`P4_PLAN` entry. A mismatch is fixed by a plan revision before any real run.

**The paper's unconstrained test.**
- For a GLU MLP, the gate and up projections share one permutation of the hidden neurons.
- The test matches one model's neurons to the other's twice, independently: by gate-projection activations, and by
  up-projection activations. Each matching is a linear assignment on activation correlations.
- It then computes the Spearman rank correlation of the two resulting permutations.
- Under independence the two matchings are unrelated, so ρ ≈ 0. If one MLP derives from the other, both recover the
  same true permutation, so ρ ≈ 1.
- The test is robust to any permutation of either model's neurons and to changes elsewhere in the architecture.
- It localises: one p-value per layer. The paper uses it to find which layers of Llama-3.2-3B come from
  Llama-3.1-8B.

**scout's statistic (frozen in `ATTRIB_PARAMS`).**

For a subject block ℓ and a reference block j = a(ℓ):
1. **Probe inputs (deviation 1: weights only, no forward pass).** Take the common tokens `C` of both vocabularies,
   matched by normalised text (P3 `anchors.vocab_ids`/`normalize_token`; for GGUF, the header's token list; `NORMAL`
   tokens only). Order them by `sha256(text)` and keep the first `n_probe = 2048`. Fewer than `min_probe = 512` makes
   the pair untestable. `X_s = E_s[ids_s]` and `X_r = E_r[ids_r]` are each model's own input-embedding rows for the
   same token strings. The probe input to block ℓ is `x̂ = x / sqrt(mean(x²) + 1e-6) ⊙ g` (the block's MLP RMSNorm
   weight g, or `1 + g` for Gemma model types).
   - *Why.* Correlation of responses is the paper's matching cost, but scout never runs models (P3 D35). Each model's
     inputs live in its own hidden basis, so any orthogonal transform of the residual stream (`E → EQ`, `W → WQ`,
     norm folding) leaves `x̂ Wᵀ` exactly unchanged. That is invariant 4 for hidden rotations and permutations.
   - *What it costs.* The probe feeds raw embeddings into deep layers instead of real hidden states. For a derived
     block, `W_B ≈ P W_A` makes the responses equal whatever the input, so power is unaffected (measured: z at the
     ceiling for ε ≤ 0.3). For independent blocks, the responses are unrelated.
   - The paper's forward-pass activations are later.md.
2. **Neurons tested (deviation 2: a row subset, for cost).** `m = min(m_test = 2048, n_s, n_r)` subject neurons
   `rows_k = ⌊k·n_s/m⌋`, k < m, are matched against **all** `n_r` reference neurons. A full square assignment on 28672
   neurons is O(n³) and too slow for 160 layers. The rectangular assignment takes 0.4 s at 2048 × 28672 (measured). Any
   neuron permutation of either model leaves the distribution unchanged.
3. **Responses.**
   - `A_g = x̂_s · W_gate,s[rows]ᵀ` (P × m) and `B_g = x̂_r · W_gate,rᵀ` (P × n_r), and likewise `A_u` and `B_u`.
   - Every column is centred over probes and scaled to unit norm; a zero-variance column is all zeros.
   - `C_g = A_gᵀ B_g` (m × n_r) is the Pearson matrix. It is a matching cost only. Raw cosine between weight rows is
     never computed, and **no correlation value ever drives a verdict** (invariant 4).
4. **Matching.** `π_g = linear_sum_assignment(C_g, maximize=True)`, giving the column per row in row order, and `π_u`
   likewise. m ≤ n_r, so every tested row is assigned.
5. **Statistic.** ρ = Pearson correlation of `rank(π_g)` and `rank(π_u)`, the Spearman correlation of the two matched
   index vectors (no ties: assignments are injective). Then `z = ρ·√(m−1)`; under the paper's null, `z ~ N(0, 1)`. The
   agreement fraction `mean(π_g == π_u)` is reported only.
6. **Null: control-calibrated and depth-matched (addition, measured necessity; D76, r1).**
   - *Why the paper's null is not enough.* The paper's null assumes that independent models share no neuron structure.
     The planner simulated independent models whose neurons read a shared rank-16 semantic subspace, with
     within-neuron gate/up correlation γ (8 blocks, m = 256):

     | setting | per-block null z |
     |---|---|
     | s = 0.3, a = 0.3, γ = 0.3 | mean 0.08, sd 1.10, max 2.74 |
     | s = 1.0, a = 2.0, γ = 0.9 | **mean 2.33**, sd 1.03, **max 4.78** |

     In the second setting the paper's parametric p-value alone would attribute independent blocks.
   - *Why the r0 shift null is not enough (validation r1 major 1).* r0 estimated the null from the same job's
     mismatched-depth pairs (subject block T[t] against reference block a(T[t+k]), k ∈ {1, 2}). That is valid only if a
     misaligned pair is exchangeable with an aligned independent pair. If independent models share **depth-local**
     features (a feature dictionary per depth, "universal neurons" at similar depths), aligned independent blocks are
     more alike than shifted ones. Reproduced at the exit's m = 2048 with the validator's model (/tmp/val4/depthlocal.py,
     per-layer dictionary of K features, s = 1, a = 2, γ = 0.9, drift 1.0, 12 blocks, every neuron tested, K = 64):
     aligned-null z mean 2.11–2.17 against a shift-null mean 0.66–0.78; the r0 rule attributed 1 independent block
     (z_adj 4.94 against z_crit 3.34) with `null_ok` true. The gap grows with m (validator: 0.08 at m = 256, 1.54 at
     m = 1024), so the fixture scale (m ≤ 512) could not reveal it.
   - *Rejected: a permutation or rotation null inside the aligned block.* The statistic is built to be invariant to
     neuron permutations of either model and to hidden rotations (step 1), so permuting or rotating inside the aligned
     block reproduces the observed z exactly and carries no null information. Breaking the gate/up pairing instead
     destroys the shared-feature signal as well, which gives back the paper's N(0, 1) null.
   - *A paired per-depth control: rejected as the sole statistic (r1), adopted as an additional clause (r2).* Used
     alone, `z_pair = (z − z_ctrl(ℓ))/(s0·√2)` is robust to any depth profile, but it loses about 30 % of z (median
     22–29 vs 31–39 at ε = 1) and leaves the control untestable, so the exit would have no negative check. Validation
     r2 (major 1) showed that the pooled rule alone does **not** bound the error when the null spikes at one depth: at
     the spike, the candidate's and the control's aligned z are exchangeable draws, so the control's leave-one-out test
     fires only about as often as the candidate's, and the candidate is falsely attributed in up to 20 % of jobs
     (reproduced below). r2 therefore requires **both** clauses. The pooled `z_adj` keeps the control testable and
     drives C10's median; the paired `z_pair` is what bounds the error under any depth profile.
   - *The r1 null.* Every job names at least one **control**: a model the human documents as trained independently of
     the subject and of every candidate, at least as close to the subject as an independent candidate could be (same
     width class and data era). The control is not fitted to anything; it is chosen from documentation, as the exit
     table does (§2.1). Its aligned k = 0 tests at the same subject positions give the depth-matched null sample.
     With `C` = the pooled k = 0 z values of all tested controls (n_c = |C|) and `S_r` = the shift values of reference r:
     - `μ0 = max(0, mean(S_r), mean(C))`, `s0 = max(1, sd(S_r), sd(C))` (ddof 1), `κ = s0 · √(1 + 1/n_c)`
     - `z_adj = (z − μ0)/κ`, `p = t.sf(z_adj, n_c − 1)` (Student t: `z_adj` is a prediction statistic for a new
       draw from the control's null, so the estimation error of μ and s is in the distribution, not ignored)
     - **control self-test, leave-one-out:** the control's own block i uses `C` without that value (df n_c − 2). If any
       control test reaches the critical value, `control_ok` is false and **every** reference abstains ("independent
       control attributed at block i: the in-run null is not homogeneous"). It is a conservative homogeneity
       diagnostic: it makes every reference abstain when the control reveals depth structure (a spike fires it with
       probability 0.24–0.75 for a spike mean of 4–5, §2.4) or is not independent. It does **not** carry the
       family-wise bound; the paired margin does (validation r2 major 1: the r1 text claimed otherwise).
     - **paired margin (r2):** for candidate r at subject position ℓ, `z_pair = (z_ℓ − zc_ℓ) / (s0_r · √2)`, where
       `zc_ℓ` is the largest k = 0 z of the tested controls at the same subject position ℓ (if no control was tested
       at ℓ, `max(C)`), and `s0_r` is the candidate's null scale above. Under A-null, `z_ℓ − zc_ℓ` has mean 0 at every
       depth, whatever the depth profile of the common mean, and sd `σ√2 ≤ s0·√2` in expectation. A control test
       has `z_pair = null`.
     - **shift guard:** `guard_r = (max(S_r) − max(0, mean(C))) / (max(1, sd(C)) · √(1 + 1/n_c))`. `null_ok_r` is
       `guard_r < crit`. It is measured against the control null alone, so a subject block that copies a reference block
       at another depth cannot hide by inflating the shift sd. (The r0 guard `(max − μ0)/s0` gave 0.97 on the frozen
       shifted-copy fixture, whose shift maximum is 22.6, and would not have fired; the r1 guard gives about 22.)
     - no tested control (none named, or all untestable) → every candidate abstains ("no tested independent control:
       the null is not calibrated"). The CLI and `attribute()` refuse a call without a control.
   - *Measured at m = 2048* (planner scratch /tmp/p4r1: null2048.py, analyse.py; numpy 2.4.6, scipy 1.17.1;
     the validator's depth-local generator; V = 4096 tokens, 2048 probes, hidden 256, 12 blocks; per job a subject, an
     independent candidate, a control and a second independent candidate, plus derived subjects at ε = 1 and 2):

     | setting (s, a, γ; drift; inter; K) | jobs | aligned-null z mean | shift-null mean | r0 rule: false blocks / max z_adj (crit 3.34) | r1 rule: false blocks / max z_adj (crit 4.55) | r1 control LOO: false / max (crit 4.71) | r1 power ε = 1: attributed, median z_adj |
     |---|---|---|---|---|---|---|---|
     | 1, 2, 0.9; 1.0; 2048; 64 (m = inter) | 2 | 2.11–2.17 | 0.66–0.78 | **1** / 4.94 | 0 / 3.19 | 0 / 1.89 | 12/12, 32.8–39.3 |
     | 1, 2, 0.9; 1.0; 2048; 16 | 3 | 0.07–0.86 | 0.44–0.79 | 0 / 2.30 | 0 / 1.34 | 0 / 1.70 | 12/12, 32.3–39.7 |
     | 1, 2, 0.9; 1.0; 4096; 64 | 2 | 1.28–1.43 | 0.58–0.74 | 0 / 2.21 | 0 / 1.28 | 0 / 2.90 | 12/12, 28.5–34.4 |
     | 1, 2, 0.9; 0.5; 4096 and 2048 | 2 | 1.39–2.00 | 0.57–1.03 | 0 / 2.76 | 0 / 1.74 | 0 / 1.87 | 12/12, 25.0–39.0 |
     | a ramps 0.5→2.5 with depth; 1.0; 4096 and 2048 | 3 | 0.37–1.65 | 0.10–0.60 | 0 / 2.88 | 0 / 2.10 | 0 / 1.83 | 12/12, 35.1–39.2 |
     | a spikes (2.5 every 3rd depth, else 0.5) | 2 | 0.28–0.94 | −0.13–0.78 | 0 / 2.08 | 0 / 1.16 | 0 / 1.86 | 12/12, 30.9–39.3 |
     | 0.5, 1, 0.6 (mild) and 0.3, 0.3, 0.3 (plain) | 2 | 0.11–0.55 | −0.16–0.21 | 0 / 2.41 | 0 / 2.32 | 0 / 1.05 | 12/12, 37.3–39.0 |
     | **total** | **16** | | | **1 of 384** null blocks | **0 of 384** | **0 of 192** | **12/12 in all 16** |

     At ε = 2 the r1 median z_adj is 2.2–16.3 (4–12 of 12 attributed): heavy retraining loses attribution, as it
     should. The shift guard statistic on these legitimate runs is at most 2.49 (crit 4.55).
   - *Measured at the fixture scale* (fixture_n1.py; the r0 generator, 16 blocks, inter 512, m = 512, 1024 probes,
     6 triples each, crit 4.305 at 32 tests): default family 0 false (max z_adj 1.28), inflated family 0 false (max
     2.43) while the paper's parametric rule attributes in 6 of 6; controls 0 false under LOO; derived subjects 16/16 at
     ε = 0.05 and 1.0 (min z_adj 20.3 and 21.3; ceiling √511 = 22.6 divided by κ), 16/16 at ε = 2.0 (min 6.2); the
     shifted copy abstains (guard ≈ 22); a subject with blocks 3 and 7 re-initialised attributes exactly the other 14.
     Depth-local at m = 1024 (inter 1024, K = 16, 4 triples, the new T313/T314 test dimensions): 0 of 48 candidate and
     0 of 48 control blocks (max z_adj 1.96, crit 4.548); ε = 1 subjects 12/12 (median 20.6–27.7).
   - *Monte Carlo of the decision rule, r1* (fwer_mc.py; 20,000 jobs per row, Gaussian nulls where an independent
     candidate and the control share mean μ and sd σ per block, shift null N(μ_s, σ_s) with μ_s ≤ μ, μ up to 6 and σ up
     to 2): family-wise error of the candidate's tests ≤ 0.0054, of the control LOO tests ≤ 0.0060, and of either
     ≤ 0.0111, at L = 12, 16, 28 and 80. **This covers depth-homogeneous means only** (validation r2 major 1).
   - *Monte Carlo of the decision rule, r2, including depth spikes* (/tmp/p4r2/fwer2.py, output fwer2_log.txt; the
     validator's generator /tmp/val4r2/fwer.py reimplemented with the T314 rule: candidate and control aligned z
     ~ N(μ_ℓ, σ) with the **same** per-depth profile μ_ℓ, shift z ~ N(0, 1), control LOO, shift guard, Bonferroni;
     20,000 jobs per row; "FWER" = P(≥ 1 false candidate block and control_ok and null_ok); "C11" = P(control LOO fires)):

     | null depth profile | L = 80, 160 tests: FWER r1 / **r2** / C11 | L = 28, 84 tests (te r2): FWER r1 / **r2** / C11 |
     |---|---|---|
     | flat μ = 0 | 0.0009 / **0.00015** / 0.0007 | 0.00015 / **0** / 0.0002 |
     | flat μ = 3, σ = 1.5 | 0.0057 / **0.0005** / 0.0052 | 0.0033 / **0.00025** / 0.0027 |
     | block-0 spike μ = 3 | 0.076 / **0.00005** / 0.096 | 0.027 / **0** / 0.045 |
     | block-0 spike μ = 4 | 0.201 / **0** / 0.378 | 0.081 / **0** / 0.238 |
     | block-0 spike μ = 5 | 0.169 / **0** / 0.748 | 0.118 / **0** / 0.586 |
     | block-0 spike μ = 6 | 0.043 / **0** / 0.953 | 0.063 / **0** / 0.875 |
     | block-0 spike μ = 8 | 0.00005 / **0** / 1.000 | 0.0011 / **0** / 0.999 |
     | mid-depth spike μ = 4 / 5 | 0.198, 0.170 / **≤ 0.00005** / 0.374, 0.747 | 0.083, 0.117 / **0** / 0.229, 0.589 |
     | blocks 0–2 at μ = 4 / 6 | 0.165, 0.015 / **0** / 0.521, 0.981 | 0.012, 0.004 / **0** / 0.052, 0.065 |
     | smooth bump, height 4 / 6 | 0.0076, 0.0015 / **0** / 0.013, 0.004 | 0.0036, 0.0006 / **0** / 0.014, 0.008 |
     | AR(1) across depth, φ = 0.5 / 0.8 | 0.0011, 0.0006 / **≤ 0.00015** / ≤ 0.0006 | ≤ 0.0001 / **0** / ≤ 0.0001 |

     The r1 validator's figures (0.196 / 0.104 at μ = 4, 0.168 / 0.123 at μ = 5) are reproduced; they used L = 28 with
     56 tests, where this script gives 0.107 and 0.126. **r2 FWER ≤ 0.0005 in every row, and ≤ 5 × 10⁻⁵ in every spike
     or bump row.** Power (same script; derived candidate z ~ N(D, 1) per block against a flat control): the r2 clause
     costs nothing at D ≥ 15 (P(attributed fraction ≥ 0.95) = 0.996–1.000 for both rules). At D = 10 it is 0.996
     (L = 80) and 0.944 (L = 28), against 0.997 and 1.000 for r1. At D = 8 it collapses (0.37 and 0.20), but there
     median z_adj ≈ 7.5 already fails C10's median ≥ 10. On the planner's m = 2048 synthetic runs
     (/tmp/p4r2/raw_r2.py over the 20 r1 raw jobs): null candidates 0 of 480 blocks for r1 and r2 (max z_pair 2.38);
     derived ε = 1: 240/240 for both (median z_pair 15.2–29.4); ε = 2: 210 → 185 of 240 (median z_pair 2.4–12.3).
7. **Multiple comparisons and the claim.** `n_tests` counts the k = 0 tests of **every candidate and every control that
   reached testing**, including a reference that later abstains (on `null_ok` or `control_ok`); it is fixed before any
   status is assigned. `crit = t.isf(α/n_tests, n_c − 1)` (candidates) or `t.isf(α/n_tests, n_c − 2)` (control LOO
   tests), α = 0.01. A candidate test is attributed iff `control_ok`, `null_ok`, `z_adj ≥ crit` **and** `z_pair ≥ crit`
   (r2). At the exit sizes: 70b `n_tests = 160`, `n_c = 80`, crit 4.036 (control 4.038); te `n_tests = 84` (base,
   independent candidate, control), `n_c = 28`, crit 4.233 (control 4.258).
   - **The family-wise error claim, stated honestly.** P(any false attribution in a job) ≤ α holds *if* (A-null) at
     every tested depth ℓ the aligned z of an independent candidate and that of the control are exchangeable,
     approximately normal draws with a common mean μ_ℓ (any depth profile) and spread at most the pooled control
     spread, drawn independently of each other given μ_ℓ. The paired clause bounds each test at a spike; the pooled
     clause bounds it where the profile is flat. The bound is **conservative under A-null (independent draws)**, not
     exact: the `max(0, ·)` and `max(1, ·)` floors, Bonferroni, and requiring two clauses each make it smaller than α.
     Nothing in a single job can verify A-null. What each job does check: the control's own blocks under leave-one-out
     (depth structure or a non-independent control makes every reference abstain) and the shift guard; the te exit
     job also tests a documented independent candidate (C10b). The claim is therefore **conditional on the control
     choice**: a candidate that is independent but much closer to the subject than the control (same
     organisation, same data, different initialisation) is not protected. Reports say so (`ATTRIBUTION_CAVEAT`, the
     `format_attribution` null line, §4.2), and p-values are described as "control-calibrated, assumes the candidate
     behaves like the control at equal depth", never as an unconditional FWER.
   - **Scaling with m** (next to the √(m−1) power argument of §2.4). A shared-feature inflation is a correlation ρ₀,
     so its z grows like ρ₀·√(m−1), exactly as the signal does. A shift null samples other depths and misses a
     depth-local ρ₀ entirely; the control null is taken at the same depths and the same m, so it scales with it. That
     is why the r1 null is measured at m = 2048 and at m = inter, the regime where the inflation is largest.

**What "block ℓ derives from base X" means.**
- *Attributed to X.* The MLP neuron set of subject block ℓ matches X's aligned block a(ℓ) up to permutation, at
  family-wise level α. It shares weight lineage (P3 §3.1 semantics: fine-tune, continued pretraining, copy, merge
  component). The direction is never asserted.
- *Primary.* If several references are attributed, the primary is the one with the largest `z_adj` (tie: lower
  reference index).
- *Not attributed.* No reference was attributed. This is **not evidence of independence**: the base may be missing
  from the references, or the block's MLP was re-initialised or retrained (`ATTRIBUTION_CAVEAT`).
- *Untestable.* Roles missing, not decodable, fused (`gate_up_proj`), an MoE block, or no probe basis. Always with a
  reason.
- Attention weights are not tested. A block whose attention was retrained but whose MLP was kept is still attributed.
  The report says "MLP".

**Alignment.**
- a is P3's structure-only alignment (`features.structure_alignment`; reldepth when S is constant and non-zero), with
  window 0: each subject block is tested against exactly one reference block.
- Equal-depth pairs get the identity: both exit bases (70b: 80 vs 80; te: 28 vs 28) and the 70b control (80). The te
  control Llama-3.1-8B and the te independent candidate OLMo-2-1124-7B each have 32 blocks against the subject's 28,
  so their alignment is reldepth (validation r1 minor). `zc_ℓ` for the paired margin is always read at the same
  **subject** position ℓ, so the two reldepth alignments need no further matching. The plan still downloads all 32
  blocks of each (about 0.94 GB + 0.72 GB more than the 28 aligned ones); planning only the aligned and shifted blocks
  is deferred (§11).
- Depth-upscaled or pruned subjects need a search window with its own correction. It is later.md, together with the
  P3 note that such alignments are tie-breaks.

### 3.9 Diffusion text encoders (D70)

- **Targets.** A pipeline component is addressed as `repo@sha#component` (T307 selector). The Card is the P1 component
  Card (`key.component = "text_encoder"`). Its model card claims come from the repo-root README, as in P1.
- **Tokenizer.** The probe vocabulary uses the P2 tokenizer resolution: for component `text_encoder<sfx>`,
  `tokenizer<sfx>/tokenizer.json`.
- **Which stack is tested.** The tested stack is the largest stack with GLU roles whose gate/up input width equals the
  embedding width. For a Qwen2.5-VL text encoder, that is the language stack (`model.layers` or
  `model.language_model.layers`, whichever the checkpoint uses). The vision tower (`visual.blocks`: GLU, width 1280 vs
  3584) has no probe basis and goes to `attribution.other_stacks` as `{prefix, depth, reason: "no probe basis (width
  1280 != embedding 3584)"}`.
- **Pairing with the reference.** The reference is a plain LLM or VLM repo. Its tested stack is chosen by the same rule,
  so checkpoint naming differences (`model.layers` vs `model.language_model.layers`) do not matter.
- **Covered.** Qwen-, Llama- and Gemma-based text encoders (Qwen-Image, HunyuanVideo's Llava-Llama, SANA and Lumina 2's
  Gemma 2) use `mlp.gate_proj` / `mlp.up_proj` and are covered.
- **Not covered.** T5 encoders (`DenseReluDense.wi_0/wi_1`, SentencePiece `spiece.model` without `tokenizer.json`) and
  CLIP encoders (non-GLU `fc1`/`fc2`) are `untestable` with a named reason. They are later.md: they need role
  patterns, SentencePiece parsing and their own null measurement.

## 4. Data structures (exact field names)

### 4.1 Card v3 (changes relative to P3 §4.1 only; T306)
```
schema_version: "card.v3"
key:        {repo, revision_sha, component}      # GGUF selection: component = "gguf:" + repo-relative path of the (first) file
weights:    {format: "safetensors"|"gguf",                                           # NEW
             index_path, index_total_size, n_tensors, params_total, tensor_bytes_total, params_by_dtype,
             files: [{path, size_bytes, header_len, data_bytes, n_tensors, metadata,
                      format: "safetensors"|"gguf",                                   # NEW
                      lfs_sha256: str|null,                                           # NEW (Hub API siblings[].lfs.sha256)
                      gguf: null | {version: int, alignment: int, n_kv: int, infos_end: int, data_offset: int,
                                    architecture: str|null, file_type: int|null, split_no: int|null, split_count: int|null}}],
             content_digest: str|null,                                                # NEW "sha256:<hex>" (§3.5)
             quant: null | {version: "quant.v1", method: str,                         # NEW; "gguf"|"fp8"|"compressed-tensors"|"gptq"|...
                            source: "gguf-header"|"config.quantization_config"|"dtype",
                            types: {dtype: n_tensors}, dequantizable: bool, reason: str|null,
                            block_size: [int, int]|null}}
stats:      {tensor_stats, spectral_topk, sigma_curves, tokenizer_minhash, anchor_embedding,
             attribution: null | Attribution}                                         # FILLED by attribute (§4.2)
fetch_log:  {counting, threshold_bytes, totals, events, gate,
             disk: {peak_bytes: int, resident_bytes: int},                             # NEW (client side; 0 for plain scans)
             jobs: [JobRecord]}                                                        # NEW ([] for plain scans)
JobRecord = {job_id, backend: "inproc"|"local"|"ssh", host: str, worker_hostname: str, plan_id, status: "succeeded"|"failed",
             exit_code: int, started_at, finished_at, wall_s: float, stage_s: {stage: float},
             totals: {meta, header, weight}, n_events: int,
             disk: {peak_bytes, bytes_written, bytes_removed, resident_after, free_before, free_after},
             purge_note: str, rss_peak_bytes: int, git_commit: str|null, scout_version: str,
             wire: {n_requests: int, body_bytes: int, hosts: [str]}|null}
```
- LogEvent dicts may carry `origin` (ingested worker events only).
- GateDecision `via` gains `"job-spec"`, and a decision may carry `origin`.
- Parquet v3: the P3 columns, then **`source_name` string (nullable)**, appended last. It holds the original GGUF
  tensor name, or null for safetensors. `dtype` holds the storage dtype (e.g. `Q4_K`, `F8_E4M3`).
- Parquet metadata: `schema_version: card.v3`.
- `load_card` accepts v0–v3.
- A plain scan still has every stat null, `jobs == []` and `disk == {0, 0}`, so P1 C7 holds.

### 4.2 `attribution.v1` (Card v3 `stats.attribution`; T314 builds it, T317 adds `job`)
```
{version: "attribution.v1", method: "zhu2025-match-glu-probe", params: ATTRIB_PARAMS,
 subject_stack: {prefix: str, depth: int, indices: [int]},
 other_stacks: [{prefix, depth, reason}],
 alpha: 0.01, n_tests: int,            # k = 0 tests of every candidate and control that reached testing (§3.8 step 7)
 null_model: "control-calibrated", control_ok: bool,
 references: [{card_key, title, role: "reference"|"control", content_digest: str|null, stack_prefix: str|null, depth: int|null,
               status: "tested"|"abstain"|"untestable", reason: str|null,
               alignment: {mode, kind, structure_informative}|null,
               probe: {n_common: int, n_probe: int, probe_sha256: str}|null,
               null: {n_shift, shift_mean, shift_sd, shift_max, n_control, control_mean, control_sd,
                      mu0, s0, kappa, df, guard}|null, null_ok: bool|null, crit: float|null, df: int|null,
               n_tested: int, n_attributed: int,   # control: tests with z_adj >= crit under leave-one-out (0 when control_ok)
               median_z_adj: float|null}],
 blocks: [{index: int, position: int, status: "attributed"|"not_attributed"|"untestable", reason: str|null,
           primary: int|null,
           tests: [{reference: int, reference_block: int|null, m: int|null, rho: float|null, z: float|null,
                    z_adj: float|null, p: float|null, crit: float|null, agree_frac: float|null,
                    z_pair: float|null,        # r2 paired margin (§3.8 step 6); null for a control test
                    attributed: bool,          # always false for a control test
                    reason: str|null}]}],
 caveat: ATTRIBUTION_CAVEAT,
 job: {job_id, host, plan_id}|null, computed_at: ISO-8601 Z}
```
All floats are rounded to 6 decimals. p = `t.sf(z_adj, df)` is stored as a float (for z_adj ≈ 40 at df 79 it is
about 1e-53; it underflows to 0.0 only far beyond the ceiling).

`ATTRIBUTION_CAVEAT` = "Block attribution tests whether a block's MLP neurons match a reference block's up to
permutation. 'Not attributed' is not evidence of independence: the base may be missing from the references, or the
block may have been retrained. p-values are calibrated against the named independent control model(s) and assume
that an independent reference would behave like them at the same depth; they are not an unconditional error rate.
Weights cannot show which model came first."

### 4.3 FullPlan (T310; served and printed, never persisted, except via GateDecision and JobRecord)
```
{plan_id: str (16 hex = sha256(json ["attrib.v1", [[repo, sha, component, [[path, start, end]...]] per model]])[:16]),
 reads: [FullRead dict] (all models' reads in plan order; a FIELD of FullPlan, so P2's gate.decision / flag_confirmer,
        which evaluate len(plan.reads), accept a FullPlan unchanged; validation r1 major 4),
 models: [{index, role: "subject"|"reference", card_key, target, stack_prefix: str|null, blocks: [int],
           embed: str|null, status: "planned"|"untestable", reason: str|null, bytes: int,
           reads: [{model, component, path, tensor, role: "embed"|"gate"|"up"|"norm"|"scale", block_index: int,
                    start, end_exclusive, dtype, shape: [int]}]}],
 per_file: [{model, path, n_reads, bytes}], bytes_planned, bytes_cap, slack_bytes, disk_bytes, memory_peak_bytes,
 threshold_bytes, host: str|null, scratch_root: str|null, disk_free_bytes: int|null, reason: str}
```

### 4.4 Job spec `job.v1` (T315; stdin of the worker; never contains a token)
```
{schema: "job.v1", job_id: str ("<yyyymmddThhmmssZ>-<plan_id[:8]>-<6 hex>"), kind: "attribute", created_at,
 client_hostname, scout_version, git_commit: str|null, params_digest: str,
 subject: {target: str, repo, revision_sha, component: str|null},
 references: [{target, repo, revision_sha, component}],          # candidate bases (role "reference")
 controls: [{target, repo, revision_sha, component}],            # >= 1 independent controls (role "control", D76)
 plan: {plan_id, bytes_planned, bytes_cap, disk_bytes, threshold_bytes},   # the worker uses min(threshold_bytes, its own default)
 confirmation: {plan_id, approved: true, via, bytes_confirmed: int|null, decided_at},
 endpoint: "https://huggingface.co", scratch_root: str|null, wire_audit: bool}
```
`target` is `repo@<40-hex>[#selector]` for Hub models, or an absolute local path that must exist on the worker host
(the offline rehearsal).

### 4.5 Job protocol `jobproto.v1` (T315; one JSON object per stdout line)
```
{"proto":"jobproto.v1","type":"hello","job_id"|null,"hostname","pid","machine","system","python","numpy","scipy",
 "scout_version","git_commit","git_dirty","code_id","code_trees","wire_audit_available","params_digest","mem_total_bytes",
 "disk_free_bytes","scratch_root","has_hf_token"}
{"proto":"jobproto.v1","type":"event","event":LogEvent dict}
{"proto":"jobproto.v1","type":"gate","decision":GateDecision dict}
{"proto":"jobproto.v1","type":"heartbeat","ts","stage","disk_resident_bytes","rss_bytes"}
{"proto":"jobproto.v1","type":"result","result":JobResult}
{"proto":"jobproto.v1","type":"error","error":{"type","message","exit_code"},"purge":PurgeReport|null}
{"proto":"jobproto.v1","type":"bye","exit_code":int}
JobResult = {job_id, hostname, plan_id, attribution: attribution.v1 (job null), purge: PurgeReport,
             disk: {peak_bytes, bytes_written, bytes_removed, resident_after, free_before, free_after},
             totals: {meta, header, weight}, n_events: int, stage_s: {stage: float}, wall_s: float,
             rss_peak_bytes: int, wire: {...}|null, git_commit, scout_version, sweep_before: SweepReport}
PurgeReport = {job_id, files_removed, bytes_removed, bytes_written, resident_after, dir_absent, ok}
SweepReport = {root, orphans: [{job_id, files, bytes, removed}], active: [job_id], stray: [str], bytes_found, bytes_removed}
```
A worker invoked with the argument `hello` prints only the hello line and exits 0. The preflight (E2) uses it.

### 4.6 Store records (T309)
```
StoreEntry = {component: str|null, object: str (content hash), schema_version, options: {sample, tokenizer, anchors},
              has: {sigma: bool, tokenizer: bool, anchors: bool, attribution: bool},
              attribution_refs: [str] ("<role>:repo@sha/component", role "reference"|"control", sorted) | [],
              params_digest: str, content_digest: str|null, created_at: ISO-8601 Z, scout_version}
```

### 4.7 ByteLog additions (T301)
- `LogEvent.origin: str | None = None`.
- `event` values gain `"job"`: 0-byte notes for job lifecycle (submitted, hello, finished, sweep).
- The new methods are listed in T301's interface.

### 4.8 View additions (T318)
```
depth_strips[].cells[].attribution: null | {status, primary_title: str|null, z_adj: float|null, p: float|null}
view.attribution: null | {stack_prefix, references: [{title, role, status, n_tested, n_attributed}], caveat: ATTRIBUTION_CAVEAT}
```

### 4.9 Frozen parameters (literals; asserted by tests and C2)
```
ATTRIB_PARAMS = {"version": "attrib.v1", "method": "zhu2025-match-glu-probe",
  "roles": {"gate": ["mlp.gate_proj.weight"], "up": ["mlp.up_proj.weight"],
            "norm": ["pre_feedforward_layernorm.weight", "post_attention_layernorm.weight"]},
  "norm_offset_model_types": ["gemma", "gemma2", "gemma3", "gemma3_text"],
  "rms_eps": 1e-06, "n_probe": 2048, "min_probe": 512, "probe_order": "sha256-normalized-text",
  "probe_token_types": "normal-only", "m_test": 2048, "min_blocks": 8, "shifts": [1, 2],
  "compute_dtype": "float32", "similarity": "pearson-of-probe-responses",
  "assignment": "linear_sum_assignment-maximize", "statistic": "spearman-of-matched-indices",
  "z": "rho*sqrt(m-1)", "null": "control-calibrated", "min_controls": 1,
  "null_adjust": "(z-max(0,mean_shift,mean_control))/(max(1,sd_shift,sd_control)*sqrt(1+1/n_control))",
  "critical": "student-t isf(alpha/n_tests, n_control-1); control leave-one-out n_control-2",
  "guard": "(max_shift-max(0,mean_control))/(max(1,sd_control)*sqrt(1+1/n_control)) < crit",
  "control_self_test": "leave-one-out; any control test >= crit -> all references abstain",
  "paired_margin": "(z-max_control_z_same_subject_position)/(s0*sqrt(2)) >= crit",
  "alpha": 0.01, "correction": "bonferroni-all-tests", "sided": "upper", "window": 0}
QUANT_PARAMS = {"version": "quant.v1",
  "gguf_decodable": ["F32", "F16", "BF16", "Q8_0", "Q4_0", "Q4_K", "Q5_K", "Q6_K"],
  "sample_dtypes": ["Q8_0", "Q4_0", "Q4_K", "Q5_K", "Q6_K"],
  "fp8_dtypes": ["F8_E4M3", "F8_E5M2"], "fp8_scale_suffixes": ["_scale_inv", "_scale"],
  "fp8_default_block": [128, 128], "fp8_scale_semantics": "multiply",
  "unsupported_methods": ["gptq", "awq", "bitsandbytes", "aqlm", "hqq", "eetq", "exl2", "quanto", "torchao"]}
GGUF_PARAMS = {"version": "gguf.v1", "versions": [2, 3], "prologue_bytes": 24, "default_alignment": 32,
  "min_kv_bytes": 13, "min_tensor_info_bytes": 24, "max_rounds": 256, "canonical_archs": ["llama", "qwen2", "qwen3"],
  "name_map": {"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight",
               "output.weight": "lm_head.weight",
               "blk.{i}.attn_norm.weight": "model.layers.{i}.input_layernorm.weight",
               "blk.{i}.ffn_norm.weight": "model.layers.{i}.post_attention_layernorm.weight",
               "blk.{i}.attn_q.weight": "model.layers.{i}.self_attn.q_proj.weight",
               "blk.{i}.attn_k.weight": "model.layers.{i}.self_attn.k_proj.weight",
               "blk.{i}.attn_v.weight": "model.layers.{i}.self_attn.v_proj.weight",
               "blk.{i}.attn_q.bias": "model.layers.{i}.self_attn.q_proj.bias",
               "blk.{i}.attn_k.bias": "model.layers.{i}.self_attn.k_proj.bias",
               "blk.{i}.attn_v.bias": "model.layers.{i}.self_attn.v_proj.bias",
               "blk.{i}.attn_output.weight": "model.layers.{i}.self_attn.o_proj.weight",
               "blk.{i}.attn_q_norm.weight": "model.layers.{i}.self_attn.q_norm.weight",
               "blk.{i}.attn_k_norm.weight": "model.layers.{i}.self_attn.k_norm.weight",
               "blk.{i}.ffn_gate.weight": "model.layers.{i}.mlp.gate_proj.weight",
               "blk.{i}.ffn_up.weight": "model.layers.{i}.mlp.up_proj.weight",
               "blk.{i}.ffn_down.weight": "model.layers.{i}.mlp.down_proj.weight"}}
JOB_PARAMS = {"version": "job.v1", "chunk_bytes": 67108864, "download_workers": 4,
  "slack": "max(16MiB, 4*min(chunk,largest_read), ceil(planned/256))", "disk_reserve_bytes": 53687091200,
  "heartbeat_s": 30, "client_idle_timeout_s": 900, "lock_wait_s": 3600,
  "scratch_root_env": "SCOUT_SCRATCH_ROOT", "scratch_root_default": "~/.cache/scout/scratch"}
STORE_PARAMS = {"version": "store.v1", "object_hash": "sha256(scout.card-object.v1|json_sha|parquet_sha)",
  "content_digest": "sha256(sorted lfs sha256)", "lock": "fcntl.flock"}
```

### 4.10 Ledger `exit/ledger_p4.jsonl` (P3 `p3ledger.v1` line format; kinds P4_*; T321)
```
P4_PREFLIGHT {hello, checks: {name: ok}, git_commit, code_id, code_dirty}
P4_PLAN      {plans: {label: {plan_id, bytes_planned, bytes_cap, disk_bytes}}, git_commit, code_id, code_dirty}
P4_ATTEMPT   {plans: {label: {plan_id, bytes_cap}}, confirm_plans: {plan_id: cap}, git_commit, code_id, code_dirty, prior_attempts: int}
P4_EXIT      {attempt_seq, passed, failed: [check], error: str|null, targets: {label: {...§2.3 payload}}, hello, backend,
              git_commit, code_id, git_dirty}
code_id = sha256("\n".join(f"{p} {git rev-parse HEAD:<p>}" for p in CODE_PATHS)) (T316 code_identity), the same
function on client and worker
```

## 5. Assumptions and open decisions (each with a recommended default)

Assumptions:
- **A5.** P1–P3 are implemented exactly as specified in T001–T218.
- **A6.** The DGX Spark facts in §2.5 (each marked V). C1 verifies the checkable ones at E2.
- **A7.** The §3.8 reading of Zhu et al. (V). The exit thresholds do not depend on the paper's details. They depend on
  the statistic as specified here, which is frozen.
- **A8.** The exit repos keep the layouts assumed in §2.1 (V items). A mismatch found at E1/E3 is a plan revision
  before `P4_PLAN`.
- **A9 (A-null).** At each depth, an independent candidate's aligned z and the control's are exchangeable draws with a
  common mean (any depth profile, including spikes) and a spread no larger than the pooled control spread (§3.8 step 7).
  It is what the family-wise bound rests on. Per job, the control self-test and the shift guard probe it, and the te
  exit job tests one documented independent candidate (C10b). r1 stated A-null as "distributed like the pooled control",
  i.e. depth-homogeneous; the r2 paired margin removes that requirement (validation r2 major 1). The P3
  labeled set's hard negatives are the natural place to measure it on real models (later.md, re-deferred §12).

| # | Decision | Default (recommended) | Rationale |
|---|---|---|---|
| D56 | Job transport | ssh + `jobproto.v1` JSON lines on stdout; spec on stdin; one job at a time via flock; backends ssh / local subprocess / in-process | Nothing to install or run on the Spark besides scout. Key auth. Streams both ways. HTTP API, Redis/Celery and Slurm were rejected (§3.3). |
| D57 | Gate across hosts | Client displays and confirms the exact `(plan_id, cap)`; the worker re-plans from its own pinned header reads, refuses a mismatch, and logs its own decision `via "job-spec"` | The gate is enforced where the bytes flow, and a spec can never widen what the human saw. |
| D58 | FULL_DOWNLOAD granularity | Whole tensors that attribution needs (gate/up/norm of every tested block, the embedding, FP8 scales), ranged, in 64 MiB chunks, to scratch disk; never whole files | 179 GB instead of 282 GB for the 70b job. Every range is planned and logged. |
| D59 | Scratch, purge, crash | Per-job directory + flock; purge in `finally` at the end of COMPUTE; PURGE check; sweeper at every worker start plus cron; `numpy.fromfile`, never mmap; reserve 50 GiB | Invariant 1. The kernel releases the flock on any crash, so orphans are exactly the unlockable directories. |
| D60 | Retry slack for full plans | `max(16 MiB, 4 × min(chunk, largest read), ceil(planned/256))`; P2/P3 σ plans keep 16 MiB | Resolves the deferred P2/P3 item for P4 plans. Changing P2/P3 caps would change frozen exit evidence (P2 E2 caps, P3 ledger pairs). |
| D61 | Log streaming | Worker events ingested with `origin`; client totals include worker bytes; the Card holds the full log and a JobRecord | "Every byte fetched is logged", in the one artifact that persists. |
| D62 | Card store | §3.5: refs by `(repo, sha)`, objects by content hash, weights digest from LFS hashes, `params_digest` invalidation, flock writers, byte-identical import/export | Reuses Cards across scans and attributions, keeps the P3 ledger hashes valid, and needs no database. |
| D63 | Card version | `card.v3` (§4.1); v0–v2 loadable; P2 C8 and P3 C3 accept v3 in the same task | New fields `format`, `quant`, `content_digest`, `lfs_sha256`, `source_name`, `disk`, `jobs`, plus the filled `attribution` slot. |
| D64 | Target syntax | `TARGET[#SELECTOR]`: a selector ending in `.gguf` selects a GGUF file (and its split set); otherwise it names a pipeline component | One grammar for the CLI, the store and job specs. |
| D65 | Quantised formats | GGUF F32/F16/BF16/Q8_0/Q4_0/Q4_K/Q5_K/Q6_K; safetensors FP8 with companion scales (attribution only); the rest detected and refused with a reason | §3.6 table. |
| D66 | GGUF header reads | Provably safe lookahead; monotone header bound; adapter `header_len = data_offset − 8`; canonical HF names for llama/qwen2/qwen3 | Zero weight bytes by construction, and every P1–P3 consumer works unchanged. |
| D67 | Dtype normalisation | Decode to float64 (σ/anchors) or float32 (attribution), `(out, in)`; thresholds unchanged (§3.7) | Measured: P2/P3 thresholds hold; the JEV caveat is documented. |
| D68 | Attribution statistic | Zhu et al. gate/up Spearman of rectangular LAP matchings on embedding-probe responses; control-calibrated null (D76); Bonferroni α = 0.01 with Student-t critical values | §3.8. Both deviations are measured and stated. |
| D69 | Block semantics | Attributed / not attributed (not evidence of independence) / untestable; MLP only; structure-only alignment, window 0; MoE blocks untestable | Honest about what the test sees. Windowed search and MoE experts are later.md. |
| D70 | Text encoders | Component selector; the tested stack is the GLU stack whose width equals the embedding; P2 tokenizer resolution; T5/CLIP untestable | §3.9. |
| D71 | Exit targets | §2.1: 70b = the R1-Distill-Llama-70B Q4_K_M GGUF (base Llama-3.3-70B-Instruct, control Qwen2.5-72B); te = the Qwen-Image text_encoder (base Qwen2.5-VL-7B-Instruct, independent candidate OLMo-2-1124-7B, control Llama-3.1-8B) | The GGUF subject exercises quantised input, full downloads and the 70B scale in one job, at 179 GB instead of 234 GB. Alternative: `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` (BF16) as the subject, +55 GB and no quantised path in the exit. |
| D72 | Spark assumptions | §2.5 (V) | Checked by C1 at E2. |
| D73 | Exit ledger | P3 line format and chain, kinds P4_*; `p3_ledger` gains a `kinds` argument | Reruns are visible, and the code freeze is anchored at `P4_PLAN`. |
| D74 | GPU | Not used in P4 | Compute is about 45 min on CPU (§2.5). CUDA wheels for aarch64 would add an environment dependency for no exit need (later.md). |
| D75 | New dependency | `scipy>=1.11` for `linear_sum_assignment` (and `scipy.stats.t`, D76) | A numpy Jonker–Volgenant would be about 150 lines of untested-in-the-wild code. scipy has aarch64 wheels. |
| D76 | Null for block attribution (r1) | Every job names ≥ 1 independent **control**; the null centre and scale are `max` over the shift null and the pooled aligned control z; Student-t prediction statistic; control self-test by leave-one-out (any hit → all abstain); shift guard against the control null; no control → abstain; CLI `--control` required. **Open for the human:** the control choice per job is a documented human decision; the default for the exit is the §2.1 table | The r0 shift null fails under depth-local shared features (validation r1 major 1; reproduced at m = 2048: 1 false in 384, r1 rule 0 in 384). Alternatives measured and rejected: in-block permutation/rotation (no information, the statistic is invariant), paired per-depth control **as the only statistic** (−30 % z, no negative check). r2 adds it as a second clause (D79). |
| D77 | Ordered log emission | `ByteLog.on_event` called in seq order under a re-entrant emit lock (P1 T002 contract change, T301); the worker writes each line under an output lock; the client requires seqs `1..n` | The stream is the byte log of invariant 2; out-of-order arrival would fail healthy jobs (validation r1 major 2). |
| D78 | Code identity across hosts | `code_id` = hash of the `CODE_PATHS` tree hashes at HEAD (plus a clean check), compared by C1 (client vs worker) and C2 (run vs `P4_PLAN`); commits are recorded, not compared | Committing the ledger or pins under `exit/` must not break the exit, and the Spark checkout need not track the client's HEAD (validation r1 major 5). |
| D79 | Paired per-depth margin (r2) | A candidate test also needs `z_pair = (z − zc_ℓ)/(s0·√2) ≥ crit`, with `zc_ℓ` the largest control z at the same subject position; the pooled `z_adj` clause, the control LOO self-test and the guard stay | The pooled rule alone gives FWER up to 0.20 under a one-depth spike (validation r2 major 1, reproduced); with both clauses it is ≤ 5 × 10⁻⁵ in every spike row and ≤ 0.0005 overall, with no power loss at ε = 1 (§3.8). Rejected: restating A-null as depth-homogeneous (keeps a known failure mode); per-block instead of global control veto (changes nothing for the exit, since C11 still requires 0 exceedances; later) |
| D80 | Specificity leg at the exit (r2) | te job gets one ungated independent candidate, `allenai/OLMo-2-1124-7B` (role `reference`); C10b requires 0 attributed blocks | Without it, no live check tests the rule on an unrelated real model (validation r2 major 2). About 6.6 GB of extra reads; te cap 32.8 GB, max_wall_s 5400. Alternative `mistralai/Mistral-7B-v0.1` is gated |

## 6. Architecture (files)

```
scout/bytelog.py, scout/errors.py                               T301  (P1/P2 files, extended)
scout/sources.py, scout/hub.py (lfs_sha256)                     T302
tests/helpers/gguf_fixtures.py                                  T303  GGUF writer + ggml block encoders (test only)
scout/dequant.py                                                T305  GGML_TYPES, decoders, FP8, QUANT_PARAMS, detect_quant
scout/gguf.py                                                   T304  header parser, adapter, config/vocab
scout/card.py (card.v3), P2 C8 / P3 C3 / p3_set edits           T306
scout/scan.py (selector, GGUF intake, inspect_all, v3 fill)     T307
scout/gate.py, scout/sigma.py, scout/cka.py (quantised σ/anchors)   T308
scout/store.py                                                  T309
scout/fullplan.py                                               T310
scout/jobs/__init__.py, scout/jobs/scratch.py                   T311
scout/jobs/download.py                                          T312
tests/helpers/attrib_fixtures.py                                T313
scout/attrib.py, pyproject.toml (scipy)                         T314
scout/jobs/spec.py, scout/jobs/protocol.py, scout/jobs/backend.py   T315
scout/jobs/runner.py, scout/jobs/worker.py                      T316
scout/attribute.py                                              T317
scout/view.py, scout/server.py, scout/web/index.html, scout/web/app.js   T318
scout/cli.py                                                    T319
scripts/exit_expectations_p4.py                                 T320
scripts/exit_check_p4.py, exit/pins_p4.json, scripts/pin_exit.py (--phase 4), scripts/p3_ledger.py (kinds), .gitignore   T321
```

## 7. Inputs handled and where each is tested (all offline)

| Input | Handling | Test |
|---|---|---|
| GGUF single file, Q4_K/Q6_K/Q8_0/F16 | Header-only, safe lookahead, canonical names, Card v3 | T304 `test_safe_lookahead_never_past_infos`, T307 `test_scan_gguf_local`, `test_scan_gguf_hub` |
| GGUF split set | All parts' headers; one unit | T304 `test_select_gguf`, T307 `test_gguf_selector` |
| Repo with several GGUF files | `AmbiguousWeightsError` listing them; `#file` selects | T307 `test_gguf_selector` |
| FP8 safetensors with block scales | Companion reads in the full plan; decode × scale | T305 `test_apply_scale`, T310 `test_plan_fp8_companions`, T316 `test_run_job_fp8_and_te` |
| GPTQ/AWQ config | `quant.dequantizable false`, reason; roles untestable | T305 `test_detect_quant`, T310 `test_plan_untestable_reference` |
| Diffusers text encoder with a vision stack | Component selector; text stack tested; vision in `other_stacks` | T310 `test_plan_text_encoder`, T314 `test_other_stacks_te`, T321 rehearsal |
| Pinned Hub target already in the store | 0 requests | T309 `test_cached_scan_zero_requests` |
| Local folder in the store | stat-hash key; hit without reads | T309 |
| Worker crash mid-download | Orphan directory swept at the next worker start; client reports | T311 `test_sweep_orphan_after_kill`, T321 `test_rehearsal_crash_then_sweep` |
| Client disconnect | Worker `BrokenPipeError` → purge; client `backend.sweep()` | T316 `test_broken_pipe_purges`, T315 `test_idle_timeout` |
| Plan mismatch (spec vs worker headers) | `PlanMismatch` before any weight byte | T316 `test_plan_mismatch_zero_weight` |
| Insufficient disk | `DiskSpaceError` before any weight byte | T316 `test_disk_space` |
| Gate declined | `GateDeclined`, no job submitted | T317 `test_decline_no_job` |
| Independent reference | 0 attributed at α, calibrated by an independent control | T314 `test_null_family`, T321 |
| Depth-local shared features (m = 1024) | Control-calibrated null; 0 false | T314 `test_depth_local_null` |
| A null that spikes at one depth (r2) | Paired margin: a candidate block never attributes on the shared spike | T314 `test_decide_depth_spike_mc` (decision rule, 2000 simulated jobs), `test_depth_spike_fixture` (weights, block 0 spiked) |
| Independent real model in an exit job (r2) | Tested as a candidate; must attribute 0 blocks | T320 C10b, T321 rehearsal te row (independent fixture candidate) |
| No control / a control that is not independent | Every reference abstains | T314 `test_no_control_abstains`, `test_control_heterogeneity_abstains`; T319 `test_control_required` |
| Partially re-initialised subject | Exactly the re-initialised blocks are not attributed | T314 `test_partial_reinit` |
| Neuron-permuted / hidden-rotated / norm-folded derivative | Attributed (invariance) | T314 `test_invariances` |
| Inflated shared-structure null | Adjusted away by the control-calibrated null; a shifted-copy subject makes the reference abstain (guard ≥ crit) | T314 `test_inflated_null_adjusted`, `test_shifted_copy_abstains` |

## 8. Risks and how each is tested

| Risk | Mitigation / test |
|---|---|
| A weight byte is read without confirmation on the worker | Worker re-plans and requires the exact pair; grant in FULL_DOWNLOAD only; `Source.read_range` preflight. T316 `test_spec_cap_widened_refused` (a spec with a larger cap or a changed plan_id is refused with 0 weight bytes), T301 grant-stage tests, C4/C6 |
| GGUF header read touches weight bytes | Provable lower bound; T304 `test_safe_lookahead_never_past_infos` (spy on every requested range: all `< infos_end`) over fixtures with 1..3 KV arrays and 2..200 tensors; C14 live |
| Weights left on disk (invariant 1) | Purge in `finally`; PURGE check; sweeper; no mmap. T311 `test_write_read_purge`, `test_sweep_orphan_after_kill`, `test_sweep_active_skipped`; T316 `test_purge_on_compute_error`, T321 rehearsal crash test, C7 (live audit over the backend) |
| Crash between download and purge | flock-based orphan detection; sweep at worker start; cron line documented. T311 `test_sweep_orphan_after_kill` (subprocess killed with SIGKILL) |
| Remote log incomplete or bytes unlogged | Ingest completeness (seqs `1..n` in arrival order, count equals `n_events`) and totals equal; worker wire audit. T315 `test_ingest_complete`, `test_out_of_order_seq_fails`, C5, C6 |
| Concurrent download threads emit events out of seq order, failing a healthy job (validation r1 major 2) | Ordered emission under the emit lock (D77, P1 contract change in T301); worker output lock. T301 `test_on_event_ordered_concurrent` (10 threads, randomised callback delay, exact order), T316 `test_emitted_seqs_ordered_under_concurrency` (hundreds of 4 KiB chunks over 4 threads and a 10 ms heartbeat, in-process and through LocalBackend) |
| A worker event the client cannot ingest (e.g. the pre-stage sweep note, stage "NONE"; validation r1 major 3) | T301 ingest accepts "NONE" (`test_ingest_none_stage`); any ingest or message error kills the worker, fails the job, sweeps and records it, never an uncaught exception (T315 `test_none_stage_first_event`, `test_ingest_error_kills_and_sweeps`) |
| The P2 confirmer cannot read a FullPlan (validation r1 major 4) | `FullPlan.reads` is a list field; T310 `test_p2_confirmer_contract` calls P2 `gate.decision` and `gate.flag_confirmer` on a FullPlan |
| Committing the ledger or pins breaks the exit's code check (validation r1 major 5) | `code_id` over the `CODE_PATHS` trees (D78); T316 `test_hello_code_identity` (a commit under exit/ keeps code_id), T320 `test_c1_code_identity`, T321 `test_rehearsal_code_identity` |
| The te tokenizer file differs from the assumed one (validation r1 major 6) | C15 accepts `tokenizer/tokenizer.json` or `tokenizer/vocab.json` (V) and is evaluated at E3 before `P4_PLAN`; T320 `test_c15_tokenizer_files` |
| A sweep deletes a job directory that is being created | Create under `.pending-*`, lock, then rename; T311 `test_create_sweep_race` |
| The null spikes at one depth (validation r2 major 1), so the pooled control null under-covers that depth | Paired per-depth margin (D79). Tests: T314 `test_decide_depth_spike_mc` (validator's generator, L = 28, block-0 spike μ = 4, 2000 jobs: ≤ 10 jobs with a false attribution, and the pooled clause alone fires in ≥ 100 of them, so the test shows the paired clause is what holds), `test_depth_spike_fixture` (≤ 1 false over the 4 depth-local triples with block 0 spiked), `test_paired_margin_values`. Measured: FWER ≤ 5 × 10⁻⁵ in every spike row (§3.8). Residual: a spike makes C10/C11 FAIL by abstention with the probabilities of §2.4 |
| No live exit check tests an unrelated real model (validation r2 major 2) | C10b on the te job's independent candidate (OLMo-2-1124-7B, 0 attributed blocks); T320 `test_each_check_pass_and_fail` (C10b), T321 rehearsal with an independent fixture candidate |
| The matching test's null is invalid on real independent models (shared neuron structure, including depth-local universality) | Control-calibrated depth-matched null (D76, §3.8), control self-test, shift guard; the FWER claim is stated as conditional on A-null (A9). Tests: T314 `test_inflated_null_adjusted` (the paper's rule attributes in ≥ 4 of 6 triples, measured 6; r1 ≤ 1 block in total, measured 0), `test_depth_local_null` (m = 1024; ≤ 1 false over 4 triples, measured 0 of 48, max z_adj 1.96 vs 4.548; power 12/12), `test_shifted_copy_abstains`, `test_control_heterogeneity_abstains`, `test_bonferroni`; planner measurement at m = 2048 in §3.8 (0 of 384 vs 1 of 384 for r0). Live: C11 (control self-test at the exit). A real FAIL goes to the human; A-null on real hard negatives is later.md. |
| Embedding-probe deviation loses power on real models | Measured power on fixtures (ε up to 1.0); C10 needs median z_adj ≥ 10; a FAIL goes to the human with z per block in the evidence. Forward-pass activations are later.md. |
| Wrong neuron roles for an architecture | Roles only from `ATTRIB_PARAMS.roles`; missing → untestable with a reason; C3 (every block planned), C14/C15 |
| Dequantisation bugs | Byte-exact decode of hand-packed blocks for every type (T303 encoders, T305 tests); encode→decode error bounds; GGUF subject vs its own BF16 source in the rehearsal: z at the ceiling |
| Quantisation shifts P2/P3 statistics | Measured (§3.7); T308 `test_quantised_sigma_r` (fixture Q4_K copy: related median r ≥ 0.95); JEV caveat documented |
| Store corruption under concurrent writers | flock + atomic replace; T309 `test_concurrent_put` (8 processes put different Cards: all refs present, JSON valid) |
| Stale cache after a code change | `params_digest` in every entry; T309 `test_params_digest_invalidates` |
| P3 ledger hashes break | Byte-identical import/export; T309 `test_export_import_bytes` |
| Client/worker version skew | `params_digest` and `git_commit` in the hello and spec; `JobError` on mismatch (T316); C1/C2 |
| P3 C10 import rule broken (`scout.analysis` must not reach `scout.gate`/`scan`/`hub`/`sources`) | `scout.card` and `scout.view` never import `scout.attrib` or `scout.fullplan`; `ATTRIBUTION_CAVEAT` is duplicated by value in `view.py` and tested equal; `scout.cka` imports only `scout.dequant`. Tests: T306 `test_card_imports`, T308 `test_cka_imports`, T314 `test_no_forbidden_imports`, T318 `test_caveat_equal`, and P3 C10 itself in the P3 rehearsal (E0) |
| P1–P3 regressions (card.v3) | T306 changes P2 C8 and P3 C3 in the same change, and updates every P1–P3 file that asserts a schema literal (`tests/test_scan_p3.py`, `scripts/p3_set.py`, `tests/test_p3_set.py`; validation r2 major 3, found by grep over all P1–P3 task specs); `p3_set` skips an existing `card.v2` or `card.v3` Card without re-reading weights (T306 `test_existing_v3_card_skipped`); T308 `test_p3_plan_unchanged`; every task runs the full `pytest -q` |
| Hub throughput too low for C9 | Bound documented (≥ 10 MB/s); the evidence records per-stage times; a FAIL goes to the human, and the bound is never raised after the fact |
| Memory on the unified 128 GB | Estimate in the plan display (≈ 10.5 GiB); `rss_peak_bytes` in the result; C9 ≤ 48 GiB |

## 9. Changes to earlier contracts

| Earlier contract | Change | Task |
|---|---|---|
| `scout/bytelog.py` (P1 T002, P2 T101) | `classify_range`: `.gguf` handled like `.safetensors` with a default header bound of 24 bytes (`GGUF_PROLOGUE`); `set_header_len` may be called repeatedly for `.gguf` with non-decreasing values (a decreasing value raises `ValueError`; safetensors unchanged); `grant(..., stage: Stage = Stage.SAMPLED_READ)`, and weight preflight allowed only in the grant's stage; `stage()` revokes the grant when leaving that stage; `LogEvent.origin` (optional; `to_dict` emits it only when set); `ingest()`, `totals_by_origin()`, `disk_add/disk_release/disk_resident/disk_peak`, `add_job_record/job_records`; `view(prefix)` returning a `LogView` that prefixes every path a Source logs (one job log holds several repos whose file names collide); `NOTE_EVENTS += {"job"}`; `ingest` accepts the P1 pre-stage label `"NONE"` as an event stage. `to_card_dict()` is **unchanged** (card.py composes the v3 fields). Without a GGUF path, a FULL_DOWNLOAD grant or ingest, behaviour is byte-identical to P2; no P1/P2 test file is edited | T301 |
| `scout/bytelog.py` **P1 T002 behavior 5 (on_event)** | **Ordered emission (D77):** `on_event` is now called in strictly increasing `seq` order under concurrent callers. A re-entrant `_emit_lock` is held from seq assignment through the `on_event` call; the data lock is still released before `on_event`, so readers are never blocked by a slow callback. P1's test "on_event receives every event in order" becomes a guarantee under concurrency (new T301 test); no P1 test changes | T301 |
| `scout/errors.py` | new `JobError` (exit 7), `PlanMismatch(JobError)`, `BackendError(JobError)`, `PurgeError` (exit 8), `DiskSpaceError` (exit 9), `StoreError` (exit 1), `QuantError` (exit 1) | T301 |
| `scout/sources.py` `RepoFile` (P1 T004) | new field `lfs_sha256: str \| None = None` (last, defaulted; P1 constructors unchanged) | T302 |
| `tests/helpers/fakehub.py` (P1 T003) | `add_repo(..., lfs_sha256: bool = False)`; default responses byte-identical | T302 |
| `scout/hub.py` (P1 T005) | `resolve()` fills `lfs_sha256` from `siblings[].lfs.sha256` when present | T302 |
| `scout/card.py` (P1 T009, P2 T102, P3 T202) | `card.v3` (§4.1); `component_slug` maps characters outside `[A-Za-z0-9._-]` to `_` (P1 component names are unaffected; GGUF components `gguf:<path>`); `SUPPORTED_SCHEMA_VERSIONS` += v3; `PARQUET_SCHEMA` += `source_name`; `build_card` kwargs `weights_format`, `quant`, `content_digest`, `gguf_files`, `source_names`; new `with_attribution()`, `object_id()`; the P1–P3 card tests are updated to v3 | T306 |
| `scripts/exit_expectations_p2.py` C8 (P2 T115, P3 T202) | accepts `card.v1`, `card.v2` or `card.v3`; not a threshold | T306 |
| `scripts/exit_expectations_p3.py` C3 (P3 T216) | `card.v2` becomes `card.v2` or `card.v3`, every other C3 condition unchanged; not a threshold | T306 |
| `scripts/p3_set.py` (P3 T213 behavior 2) | the skip rule "an existing Card loads with schema card.v2 and options all true" becomes "card.v2 or card.v3"; nothing else changes, and weights are never re-read for an existing Card (validation r2 major 3) | T306 |
| `tests/test_scan_p3.py` (P3 T207), `tests/test_p3_set.py` (P3 T213) | the `card.v2` literals (`test_anchors_approved_hub`, `test_confirm_mode`) become `card.v3`; `test_existing_card_skipped` unchanged (0 CDN weight requests on a rerun) | T306 |
| `scout/scan.py` (P1 T010, P2 T108, P3 T207) | selector (`split_selector`), GGUF intake (P1 D10 extended), GGUF config/tokenizer, Card v3 fill, `inspect_all()`; `parse_target` signature unchanged | T307 |
| `scout/gate.py` (P2 T106, P3 T206) | σ-role and anchor eligibility also accept `QUANT_PARAMS["sample_dtypes"]`; anchor row byte size via `dequant.row_nbytes`. Plans of float dtypes are byte-identical | T308 |
| `scout/cka.py` (P3 T205) | `find_embedding` also accepts dtypes in `QUANT_PARAMS["sample_dtypes"]`; `CKA_PARAMS` unchanged | T308 |
| `scout/sigma.py` (P2 T107, P3 T204) | `decode_tensor` dispatches `sample_dtypes` to `dequant.decode(..., out="float64")`; `SIGMA_PARAMS`, `SUPPORTED_DTYPES` and `compute_all` unchanged | T308 |
| `scout/view.py` (P1 T011) | attribution per cell and `view.attribution` (§4.8); `DISCLAIMERS` unchanged | T318 |
| `scout/server.py` (P1 T013, P2 T113) | `make_server(..., store: CardStore \| None = None)`; `GET /api/cards` lists store entries when a store is set; new `GET /api/view?repo=&revision_sha=&component=[&object=]` | T318 |
| `scout/web/index.html` (P1 T014, P2 T114) | adds `#stored`, `#stored-list`, `#stored-refresh`; P1/P2 ids kept | T318 |
| `scout/web/app.js` (P1 T014, P2 T114) | stored-Card list and view; strip cells coloured by attribution; legend with caveat | T318 |
| `scout/cli.py` (P1 T012, P2 T112, P3 T215) | `scout attribute`, `scout jobs {worker,hello,sweep,audit}` (`jobs worker` forwards its remaining argv to `scout.jobs.worker.main` unchanged), `scout store {import,ls}`, `scout scan --store DIR`, `#selector` targets, `scout serve --store` | T319 |
| `scripts/pin_exit.py` (P1–P3) | `--phase 4` (targets `P4_REPOS`, HF_TOKEN required) | T321 |
| `scripts/p3_ledger.py` (P3 T218) | `append`, `verify` and `read_ledger` take `kinds: tuple[str, ...] = KINDS`; P3 behaviour and P3 tests unchanged | T321 |
| `pyproject.toml` | dependency `scipy>=1.11` | T314 |
| `.gitignore` | += `cards/p4-store/` | T321 |

## 10. Task list

| id | title | route | depends_on |
|---|---|---|---|
| T301 | ByteLog P4: GGUF header class, grant stage, LogView, ingest/origin, disk accounting, job records; new errors | opus | T101 |
| T302 | `RepoFile.lfs_sha256` from the Hub API (+ FakeHub option) | sonnet | T003, T004, T005, T101 |
| T303 | GGUF fixture writer + ggml block encoders (test only) | sonnet | T103 |
| T305 | Dequantisation: GGML type table, GGUF decoders, FP8 + scales, QUANT_PARAMS, quant detection | opus | T303 |
| T306 | Card v3 schema, `with_attribution`, `object_id`; P2 C8 / P3 C3 / `p3_set` accept v3 | opus | T202, T207, T213, T216, T301, T302 |
| T311 | Scratch dir, disk accounting, purge, orphan sweeper, audit, worker lock | opus | T301 |
| T313 | Attribution fixtures: GLU families, derivations, GGUF/FP8/TE writers (test only) | sonnet | T103, T208, T303 |
| T304 | GGUF header parser (safe lookahead), adapter, config and vocab | opus | T203, T301, T303, T305 |
| T308 | Quantised σ-sample and anchor reads (gate/sigma/cka) | opus | T204, T205, T206, T303, T305, T313 |
| T310 | Full plan + full gate (`plan_attribution`, `format_full_plan`, `run_full_gate`) | opus | T106, T205, T305, T306, T311, T313 |
| T307 | Scan: selector, GGUF intake, Card v3 fill, `inspect_all` | opus | T207, T302, T304, T305, T306, T308 |
| T312 | Chunked FULL_DOWNLOAD to scratch | opus | T301, T310, T311 |
| T314 | Attribution statistic (probe, responses, LAP, Spearman, control-calibrated null, Bonferroni) + scipy | opus | T203, T205, T209, T305, T310, T313 |
| T315 | Job spec, protocol, backends (inproc/local/ssh), client ingest | opus | T301, T310, T311 |
| T309 | Card store + `cached_scan` | opus | T304, T306, T307, T314 |
| T316 | Worker `run_job` + `scout.jobs.worker` entrypoint | opus | T307, T310, T311, T312, T314, T315 |
| T317 | Attribution client (`attribute()`): cache, plan, gate, submit, ingest, Card v3, store | opus | T309, T315, T316 |
| T318 | View + server + frontend: attribution strip, stored Cards | sonnet | T011, T013, T014, T113, T114, T306, T309 |
| T319 | CLI: attribute, jobs, store, `scan --store`, selectors | sonnet | T215, T309, T311, T316, T317, T318 |
| T320 | P4 exit expectations C0–C16 (pure) + frozen literals | opus | T016, T115, T216, T306, T309, T310, T314, T317, T318 |
| T321 | P4 exit runner (preflight/plan/run), pins_p4, `pin_exit --phase 4`, ledger kinds, offline rehearsal | opus | T213, T218, T313, T317, T318, T319, T320 |

Parallel waves (no shared files within a wave; checked by script over the YAMLs):
1. {T301, T302, T303}
2. {T305, T306, T311, T313}
3. {T304, T308, T310}
4. {T307, T312, T314, T315}
5. {T309, T316}
6. {T317, T318}
7. {T319, T320}
8. {T321}

**File ownership.** Every file is in exactly one P4 task's `files`, and this was checked by script. The P1–P3 files
that P4 edits have these owners:

| File | Owner |
|---|---|
| `scout/bytelog.py`, `scout/errors.py` | T301 |
| `scout/sources.py`, `scout/hub.py`, `tests/helpers/fakehub.py` | T302 |
| `scout/card.py`, `tests/test_card*.py`, `scripts/exit_expectations_p2.py`, `scripts/exit_expectations_p3.py` and their tests, `scripts/p3_set.py`, `tests/test_p3_set.py`, `tests/test_scan_p3.py` | T306 |
| `scout/scan.py` | T307 |
| `scout/gate.py`, `scout/sigma.py`, `scout/cka.py` | T308 |
| `pyproject.toml` | T314 |
| `scout/view.py`, `scout/server.py`, `scout/web/index.html`, `scout/web/app.js` | T318 |
| `scout/cli.py` | T319 |
| `scripts/pin_exit.py`, `scripts/p3_ledger.py`, `.gitignore` | T321 |

**Suites stay green.** Every task's acceptance runs the full `pytest -q` (P1–P4, including the P1–P3 offline exit
rehearsals). T306 flips `card.v3` and, in the same change, amends P2 C8, P3 C3, the `p3_set` skip rule and the three
P1–P3 test literals that assert `card.v2` (a grep of every P1–P3 task spec for `card.v` found no others; P2 T115's
`card.v1` C8 literal is the P2 C8 edit above).

**Routing.**
- Opus: tasks that touch download gating (T301, T304, T310, T312, T315, T316, T317), Card schema (T306, T309), similarity
  or dequantisation math (T305, T308, T314), invariant 1 (T311), or exit evidence (T320, T321).
- Sonnet, each with every signature and output string given:
  - T302 (a defaulted field and one parse)
  - T303 and T313 (test fixtures with every constant and bit layout given)
  - T318 (view fields, one route, rendering over a finished schema)
  - T319 (CLI plumbing over finished functions)

## 11. Validation responses

### 11.1 Round 1

Validator verdict on r0: REVISE (0 blockers, 6 majors, 14 minors; `validation.md`). Task graph after r1, checked by
`/tmp/p4r1/validate_tasks.py` (YAML parse, required keys, deps exist in P1–P4, no cycle, wave order, single file
owner, plan table deps and routes equal the YAMLs): 21 tasks, waves 3/4/3/4/2/2/2/1, result OK.

**Majors (all resolved).**

| # | Finding | Fix | Where |
|---|---|---|---|
| 1 | The shift null is invalid under depth-local shared features; the FWER claim is unconditional | Reproduced at m = 2048 (validator's generator; aligned-null mean 2.11–2.17 vs shift 0.66–0.78; r0 rule 1 false block in 384, max z_adj 4.94). New **control-calibrated, depth-matched null** (D76): pooled aligned z of ≥ 1 documented independent control, Student-t prediction statistic, control self-test by leave-one-out, shift guard against the control null, abstain without a control. Measured at m = 2048: **0 of 384** null blocks (max z_adj 3.19 vs crit 4.55), control LOO **0 of 192** (max 2.90 vs 4.71), power ε = 1 **12/12 in all 16 jobs** (median z_adj 25.0–43.7); Monte Carlo FWER ≤ 0.0111. The FWER claim is now stated as conditional on A-null (A9), in §3.8, §2.4, the caveat and the text report. In-block permutation/rotation nulls were rejected with a reason (the statistic is invariant to them). Nothing is fitted to the exit targets: the rule, α and the controls are fixed before any real run | §3.8 steps 6–7, §2.4, §4.2, §4.9, D76, A9; T313 (`DEPTH_LOCAL*`, `NULL_TRIPLES`), T314 (`null_summary`, `test_depth_local_null` at m = 1024 and the null tests), T306, T310 (role `control`), T315–T317, T319 (`--control`), T320 (C10/C11), T321 |
| 2 | `on_event` outside the lock lets concurrent fetch events reach stdout out of order, failing healthy jobs | Fixed at the P1 layer: ordered emission under a re-entrant emit lock (P1 T002 contract change, §9, owner T301); worker output lock for heartbeat vs event lines; the client keeps the strict `1..n` check, which now verifies the guarantee | §3.3, §9, D77; T301 behavior 10, T316 behavior 12, T315 behavior 4f |
| 3 | The worker's first event (sweep note, stage "NONE") is rejected by `ingest`; T315 does not say what happens on an ingest error | `ingest` accepts the P1 label "NONE"; any exception while handling a worker message kills the channel, fails the job, sweeps, records the JobRecord and returns (never raises) | §3.4; T301 behavior 5, T315 behavior 4e2 |
| 4 | `FullPlan.reads` is a method, so P2's `flag_confirmer`/`decision` (`len(plan.reads)`) raises | `reads` is a list field; FullPlan satisfies the P2 SamplePlan attributes those functions read; contract test | §4.3; T310 interface, `test_p2_confirmer_contract` |
| 5 | C1 compares commits, but E3 commits the ledger | C1 and C2 compare `code_id` (hash of the `CODE_PATHS` tree hashes at HEAD) plus clean checks; `exit/` is outside `CODE_PATHS`; the Spark checkout only needs identical code trees; `P4_PLAN` records `code_id` | §2.2, §2.3, §2.5, §4.5, §4.10, D78; T316 `code_identity`, T320 C1/C2, T321 |
| 6 | C15 hard-codes an unverified `tokenizer/tokenizer.json` | Marked **V** in §2.1; C15 accepts `tokenizer/tokenizer.json` or `tokenizer/vocab.json` (P2 T104 resolution) and is evaluated at E3 from the stored header-scan Card, before `P4_PLAN` | §2.1, §2.2 E3, §2.3; T320 `TE_TOKENIZER_FILES`, T321 plan step |

**Found while fixing major 1.** The r0 test `test_shifted_copy_abstains` could not pass: with one shift an exact copy,
the shift values inflate their own sd and the r0 guard gives 0.97 (measured). The r1 guard measures the shift maximum
against the control null (≈ 22 on that fixture).

**Minors fixed.**
- `threshold_bytes` is in §4.4, and the worker caps it at its own `DEFAULT_GATE_THRESHOLD_BYTES` (T316 `test_threshold_capped`).
- Sweeper race: pending-name creation plus rename (§3.3; T311 `test_create_sweep_race`).
- Failed jobs display `totals_by_origin` and the JobRecord, and the `JobError` names the worker's bytes (T317).
- C6 attributes ranged requests through the redirect chain (`orig_path`, repo from the resolve URL), and an unattributable ranged request FAILs (T320 `test_c6_redirect_chain`).
- `wire_audit_available` is in the hello and C1, so a non-editable Spark install fails at E2 (§2.5, T316, T320).
- The evidence records whether the subject and base `content_digest` are equal (§2.3 payload).
- The false-FAIL probability of a correct implementation is stated (about 1–2 % over both targets), and resolving one is a logged human decision, never a rerun (§2.4).
- Missing edges added: T306 → T302, T320 → T318 (§10 table and YAMLs).
- The te control alignment statement is corrected: reldepth, 32 vs 28 (§3.8 Alignment).
- `n_tests` has one definition: k = 0 tests of every candidate and control that reached testing, abstaining or not, fixed before statuses (§3.8 step 7; T314 behavior 9c, `test_bonferroni` with an abstaining candidate).
- `scout attribute --json` carries `disclaimers` and `caveat` (T319 `test_attribute_json_disclaimers`).
- `refs/` and `by-weights/` are declared derived indexes, rebuildable by `CardStore.reindex()` (§3.5, T309 `test_reindex`), and the cron line writes no file.

**Deferred minors** (one line each; none affects the exit's correctness):
- Heartbeat `BrokenPipeError` does not abort COMPUTE: after a client death the worker computes for up to about 45 min before its next write fails and it purges; the bound is stated here, and an abort flag checked per block is later.
- The sweep cron is documented but not verified by the hello/C1; the worst case (worker crash with the client gone) leaves weights until the next worker start or cron run.
- The rehearsal has no partially re-initialised subject with per-block expected statuses (localisation is covered offline by T314 `test_partial_reinit`; C10's 0.95 bound makes such a subject a separate rehearsal row, later).
- The te job downloads all 32 control blocks although 28 are aligned (about 0.94 GB): planning only aligned and shifted reference blocks is later.
- The by-weights index and the stored-Card UI list are kept, knowingly accepting their small surface (P1 later.md asked for the store listing).

### 11.2 Validation responses — round 2

Validator verdict on r1: REVISE (0 blockers, 3 majors, 9 minors; `validation.md`). Task graph after r2, checked by
`/tmp/p4r2/validate_tasks.py` (YAML parse, required keys, deps exist, no cycle, wave order, single file owner, explicit
within-wave file disjointness, every edited P1–P3 file's original owner in the editing task's transitive deps, plan
table deps and routes equal the YAMLs) and by the validator's `/tmp/val4r2/deps.py`: 21 tasks, waves 3/4/3/4/2/2/2/1,
result OK, no finding. Statistics: `/tmp/p4r2/fwer2.py` (output `fwer2_log.txt`) and `/tmp/p4r2/raw_r2.py`, numpy
2.4.6 + scipy 1.17.1. No number below was fitted to an exit target: α, the controls, the independent candidate and every
threshold were fixed from documentation and the simulation before any real run, and no exit data exists.

**Majors (all resolved).**

| # | Finding | Fix | Where |
|---|---|---|---|
| 1 | Under a one-depth spike of the null, the pooled control rule does not bound the FWER (the control's LOO test fires only about as often as the candidate's) | Reproduced (L = 80 / 28: FWER 0.201 / 0.107 at μ = 4, 0.169 / 0.126 at μ = 5, validator 0.196 / 0.104 and 0.168 / 0.123). **Added the per-depth paired margin** as a second clause: a candidate test is attributed only if `z_adj ≥ crit` **and** `z_pair = (z − zc_ℓ)/(s0·√2) ≥ crit`, with `zc_ℓ` the control's z at the same subject position (D79). Measured over 15 null profiles × 2 layouts (block-0, mid-depth and 3-block spikes up to μ = 8, bumps, flat, AR(1)): **FWER ≤ 5 × 10⁻⁵ in every spike/bump row, ≤ 0.0005 overall** (L = 80 and L = 28). Power: no loss at D ≥ 15 or at m = 2048, ε = 1 (240/240); at D = 10, P(attributed fraction ≥ 0.95) is 0.996 / 0.944; ε = 2 drops 210 → 185 of 240. C10's median z_adj criterion is unchanged. A-null restated (A9: exchangeable per depth, any depth profile). The "catches a spike" claims were removed (§3.8, D76). The control LOO self-test is kept as a conservative homogeneity diagnostic. **C11 false-FAIL restated honestly** (§2.4): ≤ 0.0052 / 0.0027 per job under a flat null; **0.37 / 0.24 for one spiked block at μ = 4, 0.75 / 0.59 at μ = 5, ≥ 0.87 at μ ≥ 6** (L = 80 / 28). These are abstentions, never false attributions, and go to the human | §1, §2.4, §3.8 steps 6–7, §4.2 `z_pair`, §4.9 `paired_margin`, A9, D76, D79, §7, §8; T314 (`paired_margin`, `decide`, `test_paired_margin_values`, `test_decide_depth_spike_mc`, `test_depth_spike_fixture`, `test_attributed_tests_carry_z_pair`), T306 (`validate_attribution` z_pair rules) |
| 2 | The live exit tests no independent candidate, so specificity on real weights is never checked | te job gains the ungated, documented-independent `allenai/OLMo-2-1124-7B` as a second candidate (role `reference`); **C10b**: tested, `null_ok`, 0 attributed blocks, primary of 0 blocks. P4_TARGETS field `independent`, P4_REPOS and pins 7 repos, C0/C3 model lists, te `n_tests` 84 (crit 4.233, control 4.258), te bytes 25,956,065,280 → **32,549,773,312**, cap 26,224,500,736 → **32,818,208,768**, E4 total 211.6 GB, te `max_wall_s` 3600 → 5400 (byte-derived, §2.4). Added false-FAIL: ≤ 0.0034 by Bonferroni share under A-null; measured ≤ 0.0003 from its own tests (≤ 0.003 including the control veto and guard) | §2.1, §2.2, §2.3 (C0, C3, C9, C10b), §2.4, §2.5, §3.8 alignment, D71, D80, §7, §8; T320 (`independent`, `check_independent`, C1/C3/C11 indices, tests), T321 (pins, references, rehearsal independent fixture, C10b), T317/T309 (cache key covers it) |
| 3 | T306 flips `card.v3` but three P3 files hard-code `card.v2`, and `p3_set` would re-read weights for v3 Cards | T306 now owns `tests/test_scan_p3.py`, `scripts/p3_set.py`, `tests/test_p3_set.py` and depends on T207 and T213. `p3_set` skips an existing Card whose schema is in `P3_SET_SKIP_SCHEMAS = ("card.v2", "card.v3")` without re-reading weights (new `test_existing_v3_card_skipped`; `test_existing_card_skipped` unchanged). The two literals move to v3; no assertion weakened. Grep of every P1–P3 task spec for `card.v`: no other file. File ownership stays disjoint within wave 2 (script) | §2.1 status rule, §8, §9, §10 (table, ownership, suites); T306 files/deps/interface/behavior 8/tests |

**Minors fixed.**
- Cache key roles: `attribution_refs` entries are `"<role>:repo@sha/component"`; `find_attribution(subject, reference_keys, control_keys)`; swapped roles miss (T309 `test_find_attribution`, T317 `test_cache_roles_swapped`; §3.5, §4.6).
- T317 behavior 7 passes `control_targets=` to `make_spec` (T317 `test_spec_has_controls`).
- ssh argv: `SshBackend` runs `"<SCOUT_SPARK_CMD or 'python3 -m scout.jobs.worker'> <shlex-joined args>"` (T315 `test_ssh_argv`), exercised end to end through a local ssh shim before E2 (T316 `test_ssh_backend_runs_worker_module`); `scout jobs worker` forwards `argparse.REMAINDER` to `worker.main`, specified with tests (T319 stays sonnet: every signature and argv is given).
- C1 checks `has_hf_token is True`; `SCOUT_SPARK_CMD="cd <checkout> && <venv>/bin/python -m scout.jobs.worker"` makes `scripts` importable and is in the §2.5 V list (T320, §2.1 credentials).
- Wording: the bound is "conservative under A-null (independent draws)", not exact (§3.8 step 7); the statistic is "reproducible up to float nondeterminism", not deterministic (§2.1, §2.4).
- E2 disk: at E2 C1 checks only the 50 GiB reserve (no plan yet); `plan` (E3) and `run` (E4) re-evaluate C1 with `max(disk_bytes)` (§2.2, §2.3; T320 behavior 2, T321 behavior 3 and `test_rehearsal_plan_disk_row`).

**Found while fixing.**
- T314 `test_control_heterogeneity_abstains` spliced the subject's block 5 into a control whose own embedding lives in another hidden basis. Through the embedding probe that block would carry no response signal, so the test's expected `control_ok False` was not guaranteed. The control now also takes the subject's embedding.
- The ssh shim test sets `SCOUT_SCRATCH_ROOT` to a temp dir, so it never touches the real home directory.

#### Deferred minors
- Shift guard false veto under cross-layer neuron inheritance (validator minor 1): not measured at 80 blocks; a relative or per-block guard is later.md. A veto is an abstention, never a false attribution.
- C9 throughput: the implied minimum (about 9.5 MB/s for the 70b 6 h cap) is now stated in §2.5; a timed ranged read at E2 is later.
- The by-weights index, `same_weights`, `reindex`, `scout store import` and the stored-Card UI list are kept as accepted surface (validator minor 9; round 1 deferred minor).
- Per-block instead of global control veto (considered for major 1): it changes nothing for the exit while C11 requires 0 exceedances; later.
- Planning only the aligned and shifted reference blocks (the te control and independent candidate download 32 of which 28 are aligned, about 1.7 GB): later (round 1 item, extended).

## 12. Items deferred to P4 by earlier phases, and their disposition

| Source | Item | Disposition |
|---|---|---|
| P1 later | Card store with content addressing; reopen/list Cards in the UI; skip rescans | **Included** (T309, T318, `cached_scan`) |
| P1 later | GGUF / quantised inputs | **Included** (T303–T305, T307, T308) |
| P1 later | `.bin` pickle inputs (need the gate) | **Re-deferred.** Safe parsing needs a restricted unpickler or torch plus a full download just to see tensor names. Rare for current LLMs. later.md |
| P1 later | Per-block attribution filling the depth strip | **Included** (T314, T317, T318) |
| P2 later / FUTURE_IMPROVEMENTS | Scale retry slack with the plan | **Included for full plans** (D60). **Re-deferred for σ/anchor plans**: their 16 MiB slack is frozen P2/P3 exit evidence (P2 E2 caps, P3 ledger `(plan_id, cap)` pairs) |
| P2/P3 later | Streaming COMPUTE | **Included for full plans** (per-block decode from scratch; peak RAM ≈ one block's tensors). **Re-deferred for the σ-sample path**: its peak RAM of ≤ 0.85 GiB was never a problem, and changing it touches frozen P2 stage semantics |
| P2 later | Models where even `v_proj` exceeds 16 MiB per layer (405B) | **Re-deferred.** A σ-sample limitation. Attribution uses full reads instead, so it is covered for block attribution |
| P2 later | GGUF/quantised sampled reads | **Included** (T308, GGUF types) |
| P2 later | FULL_DOWNLOAD gate, batch jobs, auto-purge | **Included** (T310–T312, T315, T316) |
| P3 later | FP8 / quantised embeddings (DeepSeek-V3-Base) | **Partly.** FP8 decoding exists (T305) and attribution uses it. σ/anchor FP8 (companion reads in the P3 plan) is re-deferred. **V**: DeepSeek-V3 `embed_tokens` is BF16 per its checkpoint, in which case the P3 library entry is unaffected |
| P3 later | Per-block attribution evaluated on the P3 labeled set | **Re-deferred** (later.md). About 60–80 GB of full MLP reads on the Spark; valuable as a null calibration beyond the two exit controls and the real-model test of assumption A9 (A-null, §3.8), but not needed for the exit |
| FUTURE_IMPROVEMENTS P1–P3 | Minor items tied to P1–P3 files | Not P4 scope; unchanged |

## 13. Out of scope for Phase 4
- the retrain-plan loop, report export and UX polish (P5)
- GPU compute, a job queue beyond one serial worker, multi-node execution
- GPTQ/AWQ/bnb dequantisation, `.bin` pickles, T5/CLIP/MoE attribution, windowed alignment search
- JEV retraining or recalibration on quantised pairs

See `later.md`.
