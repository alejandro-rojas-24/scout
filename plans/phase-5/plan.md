# Phase 5 plan: Product refinement (retrain loop, report export, polish)

Status: DRAFT r0 complete (planner), 2026-10-01. Ready for VALIDATE.

This plan builds on the Phase 1–4 plans (`plans/phase-{1,2,3,4}/plan.md`, tasks T001–T321) and treats them as
contracts: Card v3 and its stats, the ByteLog and the download gates (P2 exact `(plan_id, cap)`, P4 full-download batch
jobs on the DGX Spark, purge and the orphan sweeper), the frozen `SIGMA_PARAMS`, `ALIGN_PARAMS`, `CKA_PARAMS`, JEV and
System 2, the P3 ledger, the P4 control-calibrated block attribution with its paired margin, the Card store, and the
exit-check conventions (fixed endpoint, pins cross-checked with `git ls-remote`, C-numbered checks, an exact target set,
script-written hash-chained ledgers, parameters frozen before real runs). Section 9 lists every earlier interface that
changes and the task that owns the change. Task ids are T401–T419. Decision numbers continue from P4 (D81 onward).

## 1. Goal

scout gets a retrain-plan loop that re-initialises the MLP blocks attributed to a base, retrains only those blocks as a
gated batch job on the DGX Spark, re-fingerprints them with the unchanged P4 attribution test at fixed intervals, and
stops at the first point where the flagged blocks are no longer attributed while perplexity stays within a stated
bound; the retrained weights are deleted like every other weight, only the subject's Card keeps the record, and the
panel surfaces the licence question without answering it. scout also exports a self-contained HTML report whose every
number carries a pointer to a Card, the frozen JEV model or a ledger entry, which a linter verifies mechanically, and
which two blind readers (an LLM and a person) must understand without any edit.

## 2. Exit check (runnable, falsifiable)

"You would hand a report to someone else without editing it" is made falsifiable by four things, all frozen in the
`P5_FREEZE` ledger entry before any real run:
1. a report linter with numeric checks (§3.7, C15);
2. a fixed set of three report targets (§2.1);
3. a blind-reader checklist with pass criteria, answered by an LLM reader (C16) and a human reader (C17) (§3.8);
4. one full retrain-loop run on a small real model on the Spark with numeric criteria (C3–C14).

### 2.1 Targets, pins, freeze (fixed)

**Retrain target (one job; `scripts/exit_expectations_p5.py::P5_RETRAIN_TARGET`, T416).**

| role | repo | evidence |
|---|---|---|
| subject | `Qwen/Qwen2.5-0.5B-Instruct` | Qwen2.5 technical report, arXiv 2412.15115: Instruct models are post-trained from the base models. P3 labeled pair L01 |
| base (reference) | `Qwen/Qwen2.5-0.5B` | same |
| control (independent) | `HuggingFaceTB/SmolLM2-360M` | SmolLM2, arXiv 2502.02737: pretrained from scratch by Hugging Face. P3 held-out family `smollm2` |
| data | `Salesforce/wikitext` (dataset), train file `wikitext-2-raw-v1/train-00000-of-00001.parquet`, eval file `wikitext-2-raw-v1/test-00000-of-00001.parquet` (**V**: paths, sizes about 6.4 MB and 0.7 MB, column `text`, licence CC BY-SA 3.0 / GFDL) | WikiText, Merity et al. 2016, arXiv 1609.07843 |
| flagged blocks | `(6, 12, 18)` of the 24 blocks of `model.layers` | three interior blocks, spread over depth; block 0 (whose probe input is its real input) and the last block are avoided |

- Shapes (**V** at E3): subject and base Qwen2 with 24 blocks, hidden 896, intermediate 4864, vocab 151936, tied
  embeddings, BF16; control Llama-style with 32 blocks, hidden 960, intermediate 2560, vocab 49152, BF16.
  Alignment to the control is reldepth (32 vs 24), as for the P4 te control.
- All four repos are ungated (Apache-2.0 for the models). The three model repos are already in `exit/pins_p3.json`:
  **their P5 pins must equal the P3 pins** (C0), so the P3 Cards (σ-sample, anchors, tokenizer) are reused and cited.
- The subject is a single-unit safetensors repo with a `tokenizer.json`, the base shares its architecture, and both are
  the P3 train pair L01. Its derived status is documented; the attribution at baseline is the P4 statistic unchanged.

**Report targets (three reports; `P5_REPORT_TARGETS`, T416).**

| label | subject | sections that must be present | where the evidence comes from |
|---|---|---|---|
| `retrain` | `Qwen/Qwen2.5-0.5B-Instruct` | System 1 analysis vs the P3 library, block attribution (baseline), retrain experiment | P3 Cards (imported), the E4 Card v4 |
| `te` | `Qwen/Qwen-Image#text_encoder` | block attribution; System 1 "not run" with its reason (the P4 te Card has no σ-sample or anchor evidence) | the P4 te attribution Card (imported from `cards/p4-store`) |
| `negative` | `HuggingFaceTB/SmolLM2-1.7B` | System 1 analysis vs the P3 library; block attribution and retrain "not run" | P3 Card (imported) |

The three reports together exercise every section, every "not run" path, abstentions, a multi-component subject and a
retrained subject. No report target needs a new network byte: all Cards come from the P3 and P4 exits (§2.5). System 2
commentary is **not** part of the exit reports (it is non-deterministic and costs API calls); the S2 section is covered
offline (T410, T411).

**Pins and target set.**
- `exit/pins_p5.json` holds exactly the 4 retrain repos (3 models and the dataset), committed as `"UNPINNED"` and
  written by `scripts/pin_exit.py --phase 5`, which cross-checks `scout resolve` against `git ls-remote` (for the
  dataset: `https://huggingface.co/datasets/Salesforce/wikitext`).
- **C0 target set.** `set(pins_p5) == P5_REPOS`, every pin 40-hex, every model pin equal to its `exit/pins_p3.json` pin,
  and the report subjects' pins equal to their P3/P4 pins. Otherwise the run FAILs before any request. There is no
  fallback repo; replacing one is a logged human decision followed by a plan revision before `P5_FREEZE`.
- **Fixed endpoint.** Always `https://huggingface.co`, on the client and the worker; a foreign `HF_ENDPOINT` exits 1.
- **Evidence line.** Every E-step prints one JSON line; the client `hostname` of PIN, FREEZE, PLAN, REPORTS and EXIT
  must agree.

**Script-written ledger.** `exit/ledger_p5.jsonl` uses the P3 hash-chained format (`scripts/p3_ledger.py` with
`kinds=P5_KINDS`): `P5_FREEZE`, `P5_PREFLIGHT`, `P5_PLAN`, `P5_ATTEMPT`, `P5_RETRAIN`, `P5_REPORTS`, `P5_READER`,
`P5_EXIT` (§4.9). Only `scripts/exit_check_p5.py` appends to it.

**Frozen before any real run (`P5_FREEZE`, E2).** The freeze entry records, and C2 asserts at every later step:
- the literals `RETRAIN_PARAMS`, `REPORT_PARAMS`, `LINT_PARAMS`, `READER_PARAMS` (the checklist questions and pass
  criteria), `P5_RETRAIN_TARGET`, `P5_REPORT_TARGETS` (§4.8) and their sha256;
- the sha256 of `exit/pins_p5.json`, of the JEV model file `models/jev_v1.json`, and of the P3 and P4 ledgers' last
  entries (the evidence the reports cite);
- the client `code_id` (P4 D78: tree hashes of `scout`, `scripts`, `pyproject.toml`). From `P5_FREEZE` to `P5_EXIT` the
  code is frozen: every later entry's `code_id` must equal the freeze's (C2).

`P5_FREEZE` must precede `P5_PLAN`, `P5_ATTEMPT` and `P5_REPORTS`. A second `P5_FREEZE` after a `P5_ATTEMPT` or a
`P5_REPORTS` entry is refused unless `--refreeze-reason "<human reason>"` is given; the reason is recorded, and C2 lists
every refreeze. P2's `SIGMA_PARAMS`/`ALIGN_PARAMS`, P3's `CKA_PARAMS`/`TSTATS_PARAMS`/`FEATURE_PARAMS`/`JEV_PARAMS` and
P4's `ATTRIB_PARAMS`/`QUANT_PARAMS`/`GGUF_PARAMS`/`JOB_PARAMS`/`STORE_PARAMS` are unchanged, and C2 checks them against
their frozen literals.

**Credentials and hosts.** As P4 (§2.1 there): `SCOUT_SPARK_HOST`, `SCOUT_SPARK_CMD`, ssh key auth, and a worker scratch
root. HF_TOKEN is not required by any P5 repo (all ungated) but is passed through when set. E6 needs
`ANTHROPIC_API_KEY` (any credential the `anthropic` SDK resolves). The worker additionally needs the `retrain` extra
(torch with CUDA for aarch64, transformers, tokenizers; §2.5). Scripts exit 3 before any request when a required
credential is missing. These are environment limits only; the offline suite needs none of them.

**Phase status rule.** Phase 5 closes only when E0–E8 have all passed. The P1–P4 offline suites and rehearsals stay in
E0 and are green after every task (§10). The P1–P4 network exits are not re-run: nothing in P5 changes a scan, a plan,
a Card v0–v3, a statistic or a gate (`card.v4` is written only by the retrain client; §4.1).

### 2.2 Commands

```bash
# E0 offline (no network, no Spark, no GPU, no API key): P1-P5 suites incl. the P5 rehearsal (T419): a real worker
# subprocess runs the whole retrain loop on a tiny CPU Llama family (8 blocks, hidden 128, trained on Markov text), the
# exit runner writes and lints the three kinds of report, and the stub LLM reader and a fixture human answer file are
# scored
pip install -e '.[dev]'          # dev now includes the retrain extra (torch, transformers, tokenizers)
pytest -q                        # expect exit 0, 0 failures; network tests deselected

# E1 pins (client host with huggingface.co access)
python scripts/pin_exit.py --phase 5 --pins exit/pins_p5.json
# expect exit 0; last line {"step":"PIN","phase":5,"hostname":...,"repos":{4 entries}}; the 3 model pins equal pins_p3

# E2 freeze (client, 0 network)
python scripts/exit_check_p5.py freeze
# expect exit 0; ledger gains P5_FREEZE {params, targets, sha256s, code_id}; commit exit/ledger_p5.jsonl

# E3 preflight + plan (0 weight bytes; ssh to the Spark)
python scripts/exit_check_p5.py plan --backend "ssh:$SCOUT_SPARK_HOST" --store cards/p5-store
# expect exit 0; the store is populated first (import cards/p3; export/import the P4 te Cards; idempotent);
# rows C0, C1, C2, C3 and the plan half of C4 PASS; the retrain plan is printed (format_retrain_plan: plan_id, bytes
# planned, hard cap, disk on <host>:<scratch_root>, memory, reason, per-model and per-file tables, the data files and
# the recipe) followed by "CONFIRM WITH: --confirm-plans <id>:<cap>" (the runner's line, the one to copy; the client
# display above it also shows the `scout retrain` form "--confirm-plan <id> --confirm-bytes <cap>" for the same pair);
# client totals.weight == 0; ledger gains
# P5_PREFLIGHT and P5_PLAN.
# At plan time (published configs; exact values come from the E3 plan):
#   bytes_planned about 2,087,765,760   cap about 2,356,201,216   (slack 4 x 64 MiB)   data (meta) about 7.1 MB

# E4 the confirmed retrain run (the human copies the CONFIRM WITH line; invariant 2)
python scripts/exit_check_p5.py run --backend "ssh:$SCOUT_SPARK_HOST" --store cards/p5-store --confirm-plans <id>:<cap>
# expect exit 0; one job, no prompt; rows C0..C14 PASS; the subject Card v4 (stats.retrain) is in the store;
# ledger gains P5_ATTEMPT (before the job) then P5_RETRAIN

# E5 reports (client, 0 network)
python scripts/exit_check_p5.py reports --store cards/p5-store --out reports/p5
# expect exit 0; reports/p5/{retrain,te,negative}.html built by the `scout report` code path (build_bundle, render,
# lint); each lints clean (C15, L1-L10);
# ledger gains P5_REPORTS {per report: sha256, lint rows, answer-key sha256}

# E6 LLM blind reader (api.anthropic.com; 3 calls)
python scripts/exit_check_p5.py read --reader llm
# expect exit 0; C16 PASS for all 3 reports; ledger gains P5_READER (reader llm)

# E7 human blind reader (a person who has not seen the plan, the code or scout's output)
python scripts/exit_check_p5.py reader-form --form exit/reader_human.yaml     # questions only, no answers
#   the reader opens the three HTML files, fills exit/reader_human.yaml, and signs the blind attestation
python scripts/exit_check_p5.py read --reader human --answers exit/reader_human.yaml
# expect exit 0; C17 PASS for all 3 reports; ledger gains P5_READER (reader human)

# E8 exit (client, 0 network): re-verifies every row from the ledger, the store and the report files
python scripts/exit_check_p5.py exit --store cards/p5-store --out reports/p5
# expect exit 0; table C0..C17 all PASS; "EXIT CHECK (P5): PASS"; ledger gains P5_EXIT; commit the ledger
```

- Subcommands are mutually exclusive; `run` requires `--confirm-plans`; `read` requires `--reader`; `read --reader
  human` requires `--answers`.
- `run` repeats the preflight and plan steps and FAILs ("plan changed since P5_PLAN") if the plan id differs.
- `reports`, `read` and `exit` refuse (exit 1) unless a `P5_RETRAIN` entry with `passed: true` exists after the last
  `P5_FREEZE`; `read` also needs a `P5_REPORTS` entry whose report sha256 values equal the files on disk.

### 2.3 Checks (P5 namespace; one row per check: `target | check | expected | actual | PASS/FAIL`)

C3–C14 are evaluated for the retrain job; C15–C17 per report.

| # | Check | Threshold |
|---|---|---|
| C0 | target set: `set(pins_p5) == P5_REPOS` (4 repos), all 40-hex; the 3 model pins `==` `pins_p3`; the report subjects' pins `==` their `pins_p3`/`pins_p4` pins | equal, else FAIL and exit 1 before any request |
| C1 | Spark environment: every P4 C1 clause (machine `aarch64`, Linux, `mem_total_bytes >= 100 GiB`, disk `>= max(disk_bytes) + reserve`, `wire_audit_available`, worker `code_id ==` client `code_id`, both clean); plus the `retrain-env` report: `torch` importable with version `>= 2.4`, `cuda_available is True`, `device_name` non-empty, `gpu_mem_total_bytes >= 32 GiB`, `transformers` version in `[4.51, 6)`, `tokenizers` importable | all true |
| C2 | frozen: the P5 literals equal `exit_expectations_p5.py` on the client and in the worker's `retrain-env` (`retrain_params_digest`); P2–P4 params equal their frozen literals; `P5_FREEZE` exists and precedes every `P5_PLAN`, `P5_ATTEMPT` and `P5_REPORTS`; every later entry's `code_id ==` the last `P5_FREEZE`'s; each refreeze lists its reason | equal |
| C3 | retrain plan: models `== [subject, base, control]`; the subject's reads are **every tensor** of its Card (Σ `nbytes` `== weights.tensor_bytes_total`), recomputed independently from the stored Card; the base's and control's reads `==` the P4 attribution plan for the same models (`fullplan.plan_attribution`); `bytes_cap == bytes_planned + slack` by the P4 rule; `disk_bytes == bytes_planned`; data files `==` the target's two files with sizes from the dataset API and `Σ <= RETRAIN_PARAMS.data_max_bytes` | equal |
| C4 | gate: plan-time client `totals.weight == 0`; in the run exactly 1 approved client decision via `cli-flag` with `bytes_confirmed == bytes_cap` and `(plan_id, cap)` in `--confirm-plans`; exactly 1 worker decision via `job-spec`; worker `plan_id ==` client `plan_id` | true |
| C5 | streamed log: worker `bytes_planned <= totals.weight <= bytes_cap`; ingested events `== n_events`, seqs exactly `1..n_events` in arrival order; client totals of origin `<worker>` `==` the worker's totals | true |
| C6 | worker wire: Σ body bytes `==` worker ByteLog `meta + header + weight`; hosts ⊆ {`huggingface.co`, `*.hf.co`}; every ranged request lies within a registered header bound or one planned read; every data-file request is for one of the two planned data files | true |
| C7 | disk and purge (P4 C7 a–e): scratch audit before 0 files; `disk_peak_bytes <= disk_bytes`; purge report ok with `bytes_written == bytes_planned` (**no checkpoint or other file was ever written**); COMPUTE `purge` event before the PURGE check event; audit after: 0 files, 0 bytes | all true |
| C8 | no manual intervention: 1 job submitted, `status == "succeeded"`, exit code 0, stdin never read | true |
| C9 | resources: wall `<= 7200 s`; COMPUTE `<= 5400 s`; worker `rss_peak_bytes <= 48 GiB` | true |
| C10 | **baseline** (the unmodified subject, computed in the job before any re-initialisation): base `status == "tested"`, `null_ok`, `control_ok`; base attributed fraction `>= 0.95`; control 0 leave-one-out hits; **every flagged block attributed to the base**; baseline perplexity in `[2.0, 60.0]` (forward-pass sanity) | true |
| C11 | **signal drop** at the stop step: `outcome == "success"`; at that step base `status == "tested"`, `null_ok`, `control_ok` (not an abstention); **0 of the 3 flagged blocks attributed to the base**; unflagged blocks attributed fraction `>= 0.95` (the model is still detected as derived); control 0 hits; stop step `<= max_steps` | true |
| C12 | **quality** at the stop step: `ppl_ratio = exp(eval_loss − eval_loss_baseline) <= 1.05`; top-1 agreement with the unmodified subject on the eval tokens `>= 0.90` | true |
| C13 | Card v4 and store: the subject Card written by the job has `schema_version == "card.v4"`; `validate_retrain(stats.retrain) == []`; `stats.attribution` is the baseline and validates as `attribution.v1`; `retrain.lineage.parent_object ==` the object id of the header-scan Card the plan used; `retrain.weights_kept is False`; the store holds it; recomputing its object id from the two files gives the stored id | true |
| C14 | caveats: the `scout retrain` text output and the view of the Card v4 carry the three `DISCLAIMERS`, `ATTRIBUTION_CAVEAT`, `RETRAIN_CAVEAT` and `LICENSE_QUESTION` | true |
| C15 | **report lint** (per report, §3.7): L1–L10 all at their thresholds; file sha256 `==` the `P5_REPORTS` entry; the sections required by §2.1 present and the others "not run" with a reason | all pass |
| C16 | **LLM blind reader** (per report, §3.8): available answer with a served model; request host `api.anthropic.com`; correct answers `>= 9` of 10; critical questions Q1, Q2, Q6, Q8 all correct; required edits `== 0` | all true |
| C17 | **human blind reader** (per report): blind attestation `true`; correct `>= 9` of 10; critical all correct; required edits `== 0` | all true |

**E2** evaluates C0 and C2 (freeze only). **E3** evaluates C0–C3 and the plan half of C4. **E4** evaluates C0–C14.
**E5** evaluates C15; **E6** C16; **E7** C17; **E8** re-evaluates every row from the ledger, the store and the files.

The `P5_RETRAIN` payload records: plan_id, bytes planned/cap, weight bytes, disk peak, wall/compute seconds, rss peak,
the `retrain-env` report, the baseline (ppl, per-block base z_adj and z_pair), the whole trace (§4.2), the outcome, the
stop step, the final flagged-block statistics, the `retrained_digest`, the Card object id, `prior_attempts`, the backend
string, `git_commit`, `code_id`.

### 2.4 Why these numbers (exit numbers set by planner under the 2026-09-28 delegation; frozen in T416)

- **Attribution thresholds (C10, C11) are P4's, unchanged.** At the exit sizes the job has 24 base tests and 24 control
  tests at m = 2048: `n_tests = 48`, `n_c = 24`, `crit = t.isf(0.01/48, 23) = 4.12` (control leave-one-out, df 22,
  slightly higher). The base is the documented parent of an Instruct post-training, far below the ε = 1 regime of P4
  §3.8, so every block should sit near the ceiling `√2047 = 45.2`. C10's 0.95 lets 1 of 24 blocks fail for an
  unforeseen reason (reported). The P4 false-FAIL analysis of the control self-test applies unchanged (≤ 0.3 % per job
  for a flat null; a depth spike makes every reference abstain, which FAILs C10 and goes to the human).
- **Unflagged blocks stay attributed (C11, ≥ 0.95).** Their weights are frozen, so their z values at the stop step equal
  the baseline's except through the null (the control's z at the three flagged positions changes, which moves μ0 and s0
  slightly). This row is the live localisation test that P4 deferred: exactly the re-initialised blocks lose the
  signal and the model is still detected as derived.
- **Quality bound (C12): ppl ratio ≤ 1.05 and top-1 agreement ≥ 0.90.** "Bounded quality loss" means: at most 5 %
  higher perplexity on the eval split and at most 10 % of next-token argmaxes changed relative to the unmodified
  subject. Perplexity alone would be too easy, because WikiText-2 test is in-domain for the retraining data (the toy
  below reaches a ratio below 1 by step 1000); agreement with the unmodified model has no such bias. The toy's "MLP
  removed" state (step 0) already has agreement 0.81, so a bound of 0.80 would not measure anything; 0.90 is above it and
  was reached by step 250.
- **Budget: 2000 steps of 16 × 512 tokens, eval every 250.** 16.4M token presentations, about 5.6 passes over the
  WikiText-2 train split (about 2.9M Qwen tokens, **V**); the stop rule takes the first eval step that meets both
  criteria, so the budget is a ceiling, not a target. The GPU time bound is §2.5.
- **Measured on a toy (planner scratch `/tmp/p5`, torch 2.14 on CPU; not in the repo).** A word-level LM on the Python
  standard library source (V = 1024 tokens, 2.5M tokens), 8 blocks, hidden 96, GLU intermediate 256, 4 heads, learned
  positions. Base: 2500 steps from scratch. Derivative: the base fine-tuned 300 steps at lr 3e-4 on a disjoint text
  split. Control: an independent initialisation with intermediate 384 trained on the **same** data (the worst case for
  shared features). The P4 statistic reimplemented (embedding probes, rectangular LAP on gate and up responses, Spearman,
  `z = ρ√(m−1)` with m = 256, control-calibrated `z_adj` and paired margin): derivative vs base z = 15.97 in all 8 blocks
  (the ceiling √255), control z −1.42…0.52, crit 5.20 (16 tests, df 7). Blocks 2 and 5 re-initialised as §3.1 and
  retrained (AdamW 1e-3, batch 32 × 64, cosine), cross-entropy (CE) or CE + ½ KL to the unmodified model (KD):

  | step | CE ppl ratio | CE top-1 | CE z (b2, b5) | KD ppl ratio | KD top-1 | KD z (b2, b5) |
  |---|---|---|---|---|---|---|
  | 0 (re-init) | 1.288 | 0.808 | −1.03, 0.43 | 1.288 | 0.808 | −1.03, 0.43 |
  | 100 | 1.074 | 0.880 | −0.89, 0.24 | 1.064 | 0.894 | −0.89, 0.31 |
  | 250 | 1.033 | 0.912 | −1.86, 0.08 | 1.021 | 0.932 | −1.09, 0.89 |
  | 500 | 1.020 | 0.923 | 1.12, −0.13 | 1.010 | 0.941 | −0.40, −0.88 |
  | 1000 | 1.003 | 0.921 | 1.74, 1.66 | 0.995 | 0.948 | 2.40, 0.78 |
  | 1500 | 0.999 | 0.925 | 2.01, 0.42 | 0.992 | 0.951 | 2.00, 0.28 |

  A second, harder toy (same text and statistic, hidden 128, GLU intermediate 1024, control intermediate 768, m = 1023
  probes; base and derivative trained as above) re-initialised **4 of 8** blocks (1, 3, 5, 6; 50 % of the MLPs) and
  ran the same recipe for 1000 steps (planner scratch logs `ce4.log`, `kd4.log`; the baseline z of this pair was not
  logged):

  | step | CE ppl ratio | CE top-1 | KD ppl ratio | KD top-1 | max flagged z_adj (CE / KD) |
  |---|---|---|---|---|---|
  | 0 (re-init) | 2.811 | 0.576 | 2.811 | 0.576 | 0.39 / 0.39 |
  | 250 | 1.102 | 0.833 | 1.063 | 0.868 | 0.81 / 2.49 |
  | 500 | 1.047 | 0.847 | 1.019 | 0.886 | 0.78 / 0.99 |
  | 1000 | 1.005 | 0.849 | 0.983 | 0.901 | 1.99 / 0.90 |

  What the two toys say for the exit: (1) the signal drops at the first eval point in every run (no flagged block
  attributed at any step; crit 5.20); (2) perplexity recovers in both; (3) **top-1 agreement is the binding
  criterion**, and it depends on how much of the MLP stack is removed: 2 of 8 blocks reach 0.91 by step 250, 4 of 8
  plateau at 0.85 under CE. The exit flags 3 of 24 blocks (12.5 %, half the smaller toy's fraction), which favours
  C12; but its whole budget presents only about 0.42 training tokens per trainable parameter (16.4M / 39.2M), against
  3.5 at the smaller toy's step 250 and 1.3 at the harder toy's step 1000. The toys therefore do not establish C12 at
  exit scale: **C12 is the exit's main risk** (§8). A C12 FAIL is a finding; the 0.90 bound and the budget are not
  changed after a run. (4) Under CE the flagged z drifts upward with training (max z_adj about 2 by step 1000, still
  far below crit); see later.md. The KD objective recovers agreement faster but is rejected for the product (§3.4: it
  pulls the new neurons towards the old function) and is not used.

### 2.5 Environment, access, bytes, time

**DGX Spark (each V at E3; C1 checks the first five).**
- **V** Everything P4 §2.5 assumes (GB10, aarch64, 128 GB unified memory, NVMe scratch, ssh, Hub egress).
- **V** A CUDA-enabled PyTorch for aarch64 that supports the GB10 GPU (`torch >= 2.4`; in practice a CUDA 12.8+/13
  build from the PyTorch index or NVIDIA's container), installed in the worker's venv with `pip install -e '.[retrain]'`.
  `torch.cuda.is_available()` must be true (C1). The CPU fallback exists in code (the offline rehearsal uses it) but the
  exit requires CUDA: on the Spark's 20 Arm cores the exit budget would take on the order of 14 h (below).
- **V** `transformers >= 4.51` (Qwen2/Qwen3 and Llama classes) and `tokenizers` wheels for aarch64 (both publish them).
- **V** GPU memory as reported by `torch.cuda.get_device_properties(0).total_memory` `>= 32 GiB` (unified memory).
- **V** The worker sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` for the retrain COMPUTE (§3.3), so transformers
  can never fetch anything outside the logged path.

| Step | Needs | This container |
|---|---|---|
| E0, implementation | PyPI: numpy, scipy, pyarrow, httpx, pyyaml, pytest, anthropic, torch (CPU is enough), transformers, tokenizers | available (torch 2.14.0 installed in a scratch venv from PyPI; download.pytorch.org is blocked, the PyPI wheel pulls about 3 GB of CUDA libraries on x86 but runs on CPU) |
| E1 (client) | `huggingface.co` API and git (3 model repos, 1 dataset repo) | blocked (proxy 403) |
| E2, E5, E7, E8 (client) | local files only: `cards/p3`, `cards/p4-store`, ledgers, `models/jev_v1.json` | runs anywhere once the P3/P4 exits have produced them |
| E3, E4 (worker) | ssh to the Spark; Spark egress to the Hub; the `retrain` extra with CUDA | no Spark from here |
| E6 | `api.anthropic.com`, `ANTHROPIC_API_KEY`; 3 calls at up to about 20k input and 4k output tokens each at `claude-opus-5-5` ($4 / $20 per MTok) ≈ $0.5 worst case | not attempted here |

**Preconditions from earlier exits.** The P3 exit (Cards in `cards/p3`, `models/jev_v1.json`, `exit/ledger_p3.jsonl`)
and the P4 exit (`cards/p4-store`, `exit/ledger_p4.jsonl`) must have run. If a report target's Card is missing, E3
FAILs C0 with "missing Card: run the P3/P4 exit first"; nothing is rescanned silently.

**Byte estimate at plan time (published configs; the E3 plan is exact).**

| model | tensors planned | bytes |
|---|---|---|
| subject Qwen2.5-0.5B-Instruct (BF16, all tensors, tied embeddings) | 494,032,768 params × 2 B | 988,065,536 |
| base Qwen2.5-0.5B (attribution tensors) | gate+up 24 × 2 × 4864 × 896 × 2 = 418,381,824; embed 151936 × 896 × 2 = 272,269,312; norms 43,008 | 690,694,144 |
| control SmolLM2-360M (attribution tensors) | gate+up 32 × 2 × 2560 × 960 × 2 = 314,572,800; embed 49152 × 960 × 2 = 94,371,840; norms 61,440 | 409,006,080 |
| **total** / slack (`4 × min(64 MiB, largest read)`) / cap | | **2,087,765,760** / 268,435,456 / **2,356,201,216** |
| data (meta class, below the 64 MiB meta threshold) | 2 parquet files | about 7.1 MB |

**Compute estimate (GB10; not measured, bounds only).** One step is about 6 × 0.49e9 × 8192 ≈ 2.4e13 FLOP (an upper
bound: the backward pass stops below the first flagged block); at an assumed 25–50 TFLOP/s effective in bf16 that is
0.5–1.0 s, so 2000 steps take 17–33 min. On the CPU (about 1 TFLOP/s) the same budget is about 4.8e16 FLOP ≈ 13 h,
which is why C1 requires CUDA. Each eval point costs about
2 s of evaluation and about 30 s of re-fingerprinting on the CPU (48 matchings of 2048 × 4864 or 2048 × 2560). With 8
eval points and the baseline, COMPUTE is at most about 45 min; the C9 bound is 5400 s (1.5 h), the job bound 7200 s.
Memory: the model in fp32 about 2 GB (bf16 autocast for compute), trainable fp32 state for 3 × 3 × 896 × 4864 =
39.2M parameters ≈ 0.63 GB, activations and logits for 16 × 512 tokens ≈ 8 GB; the worker's host RSS stays below the
48 GiB of C9 (the attribution side is the P4 estimate for this size, under 3 GiB).

## 3. Definitions

### 3.1 The retrain-plan loop: what it is for, and what it is not

The P4 depth strip says which blocks of a subject match a base. The retrain panel answers the follow-up question the
project context names ("how to retrain those blocks"): if these blocks were re-initialised and retrained, does the
matching-test signal for them go away, and what does it cost in quality? It answers with an experiment, not with an
estimate:

1. **Flag.** The human picks blocks among those attributed to a base (CLI `--blocks`; the panel lists the attributed
   ones). A flagged block that is not attributed to that base at baseline makes the job refuse to train (§3.3, COMPUTE
   step b).
2. **Re-initialise.** For each flagged block, the MLP `gate_proj` and `up_proj` are redrawn from `N(0, σ²)` with
   `σ = config.initializer_range` (default 0.02) from a fixed seed, and `down_proj` is set to zero. At step 0 the block
   therefore adds nothing to the residual stream (the model equals the subject with those MLPs removed), and its
   gate/up neurons are fresh.
3. **Retrain.** Only those three matrices per flagged block are trainable; every other weight is frozen. The objective is
   causal-LM cross-entropy on a pinned public text dataset (§3.4).
4. **Re-fingerprint.** Every `eval_every` steps, the unchanged P4 statistic (`scout.attrib.attribute`, `ATTRIB_PARAMS`)
   is recomputed with the subject's current gate/up tensors against the same base and control, and the eval perplexity
   and the top-1 agreement with the unmodified subject are measured.
5. **Confirm and stop.** The loop stops at the first eval step at which (a) no flagged block is attributed to the base,
   while the base is `tested`, `null_ok` and `control_ok` (an abstention never counts as a drop), and (b)
   `ppl_ratio <= max_ppl_ratio` and `top1_agreement >= min_top1_agreement`. If no eval step meets both before
   `max_steps`, the outcome names what failed (`signal_persists`, `quality_not_recovered`, or `neither`).

**What the loop does not claim.** A block that is no longer attributed has lost the matching-test signal; that is not
evidence of independent training (P4 §3.8). The retrained checkpoint is still derived from the subject: its attention
weights, norms, embeddings and every unflagged block are unchanged, and C11 requires the unflagged blocks to stay
attributed. The loop is a feasibility and cost experiment. It says nothing about licences, and the panel and the report
say so (`RETRAIN_CAVEAT`, `LICENSE_QUESTION`, §4.3).

**Why only the MLP and only gate/up/down.** The P4 test looks at gate/up neurons only (`ATTRIB_PARAMS.roles`).
Re-initialising attention would cost quality without changing the signal the test measures, and a re-initialised
`down_proj` would let a large random output perturb the frozen layers above it; the zero init keeps step 0 equal to
"MLP removed". Re-initialising more than the MLP is later.md.

### 3.2 Weights produced by retraining and invariant 1 (D83)

Invariant 1 says the Card is the only persisted artifact and weights are never kept after COMPUTE. P5 keeps it
**without an exception**:
- The retrained weights exist only in the worker's process memory (host RAM and GPU memory) during COMPUTE. The trainer
  never writes a checkpoint, optimizer state or any tensor to disk: `torch.save`, `save_pretrained` and
  `safetensors.torch.save_file` are never called (static test, T405), and C7 checks `bytes_written == bytes_planned`
  (only the downloaded parent tensors ever touched the scratch disk) and a 0-file audit after the job.
- At the end of COMPUTE (in `finally`, also on every error path) the model, optimizer and every tensor are deleted, the
  CUDA cache is emptied (`torch.cuda.empty_cache()`), and the scratch directory is purged as in P4.
- What is kept: the subject's **Card v4** with `stats.retrain` (§4.2): the recipe, the per-step trace, the outcome, the
  baseline and final attribution statistics, and `retrained_digest` = sha256 of the final flagged tensors (float32,
  little-endian, block then role order). The digest identifies the result without keeping it; no numeric list longer
  than 64 values (other than the attribution blocks, which are per-block records) is stored.
- **Reproduction instead of export.** The recipe pins the parent revision, the data files (repo, revision, sha256), the
  seed, every hyperparameter, the framework versions and the `code_id`, so the human can regenerate the checkpoint with
  their own tooling outside scout, up to GPU nondeterminism. scout itself never hands out retrained weights.
- **Rejected: an export-to-user path** (`--export-weights DIR`, gated like a download, logged). It would be the first
  scout code path that writes weights outside scratch, which is an invariant 1 change the human has not made, and a
  one-command "remove the fingerprint and give me the weights" path is exactly what the panel must not make easy
  without the licence question being answered by a human. It is later.md as a decision for the human, with the design
  sketched there.

### 3.3 The retrain job (T403, T405–T408)

**Plan (`scout.retrain.plan.plan_retrain`, T403).** Built from Cards only, like every P4 plan:
- The attribution part is `fullplan.plan_attribution(subject, [base], controls=[control])` unchanged. Its reads for base
  and control are used as they are.
- The subject additionally gets a `FullRead` for **every** tensor of its Card that the attribution plan does not already
  read (role `"train"`, `block_index` from the Parquet row or -1). Its reads are therefore exactly all its tensors.
- The subject must be a single-unit repo (no `#selector`), `config.model_type` in `RETRAIN_PARAMS.model_types`
  (`llama`, `qwen2`, `qwen3`, `mistral`), unquantised (`weights.quant` null) and with a `tokenizer.json`; otherwise
  `RetrainRefused` (exit 10) before any byte.
- `bytes_planned`, `slack`, `bytes_cap`, `disk_bytes` and `memory_peak_bytes` follow P4 §3.2 (the displayed memory
  estimate adds 16 bytes per subject parameter for the trainable model, T403).
- **Data files** are listed in the plan and counted in its display but are not weight reads: they are read in the META
  stage as meta-class bytes (P1 semantics), must total at most `RETRAIN_PARAMS.data_max_bytes` (32 MiB, so the job's
  meta stays under the 64 MiB threshold without a new byte class), and must match the dataset API's `lfs.sha256`
  after download (else `RetrainRefused`).
- `plan_id = sha256(json ["retrain.v1", attribution plan_id, [[path, start, end] of every subject read], [[repo, sha,
  path, size, lfs_sha256] of the data files], recipe_digest])[:16]`, where `recipe_digest` is the sha256 of the
  canonical recipe (blocks, seed, every `RETRAIN_PARAMS` value). The human confirms the exact recipe with the bytes.
- `format_retrain_plan` prints the P4 `format_full_plan` text, then a `DATA` table and a `RECIPE` block, then the
  licence question.

**Gate.** The P4 full gate unchanged (`run_full_gate`, exact `(plan_id, bytes_cap)`, `cli-flag` on the client,
`job-spec` on the worker, re-planning on the worker from its own header reads).

**Worker (`scout.retrain.job.run_retrain_job`, T407)**, in the P4 order with retrain steps in COMPUTE:
1. Validate the spec (`kind == "retrain"`, §4.4); `params_digest` (P4) and `retrain_params_digest` must equal the
   worker's.
2. Worker lock and orphan sweep (P4).
3. `scan.inspect_all([subject, base, control])` (RESOLVE, HEADERS, META); dataset resolve and the two data files read
   through `HubSource(..., repo_type="dataset")` (or `LocalSource` for a local data folder in the rehearsal) (META,
   logged, sha256 checked against the API's `lfs.sha256`).
4. Re-plan and compare `plan_id`/`bytes_cap` with the spec (`PlanMismatch`, 0 weight bytes); disk check (P4).
5. FULL_DOWNLOAD: gate, grant, scratch, `download_plan` (P4, unchanged).
6. COMPUTE (`try … finally purge`), with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` set in the process:
   a. **Baseline fingerprint**: `attrib.attribute(plan_attrib, …, ScratchProvider)` → attribution.v1 of the unmodified
      subject.
   b. **Baseline refusal**: any flagged block whose base test is not `attributed`, or base not `tested`/`null_ok`, or
      `control_ok` false → outcome `refused_baseline` with the reason; no training (the result is still a valid
      retrain.v1 and the Card records it).
   c. Build the model (§3.5) from the scratch tensors; tokenise the data; baseline eval (loss, perplexity, and the
      argmax token at every eval position, kept in RAM); baseline perplexity outside `baseline_ppl_range` → outcome
      `refused_forward_check` (the forward pass does not reproduce the model; no training).
   d. Re-initialise the flagged blocks (§3.1 step 2); freeze everything else.
   e. Train for up to `max_steps`; at every multiple of `eval_every`: eval metrics, re-fingerprint through a
      `RetrainProvider` that serves the flagged blocks' current gate/up tensors from the model (every other tensor is
      unchanged and comes from scratch); append a trace row; apply the stop rule (§3.1 step 5).
   f. `retrained_digest`, then delete every tensor and empty the CUDA cache.
   `finally`: `scratch.purge()` and the P4 `purge` event.
7. PURGE check (P4).
8. `result` message with the baseline `attribution.v1` and `retrain.v1` (both with job null), purge report, disk,
   totals, timings, rss and the `retrain-env` facts; then `bye`.

**Aborting on client death** (P4 deferred minor, closed for retrain jobs). The worker's `emit` wrapper sets an
`abort` event on any `OSError` (a broken pipe from the heartbeat thread included). The training loop checks it before
every step and every eval chunk, and raises `Aborted` (exit 7), so the `finally` purges within one step (about 1 s on the GPU)
instead of after the whole budget. The P4 attribution runner is unchanged (still re-deferred there).

**Client (`scout.retrain.client.retrain`, T408).** Mirrors P4 `attribute()`: store-cached header scans with
`tokenizer=True` (0 weight bytes), dataset resolve (0 weight bytes), plan, `plan_only` display, gate, spec, `run_job_via`,
then `card.with_retrain` on the subject's header-scan Card with the job's baseline attribution as `stats.attribution`
and `retrain.v1` as `stats.retrain` (`card.v4`), stored in the Card store; `format_retrain(card)` prints the recipe, the
trace, the outcome, `RETRAIN_CAVEAT`, `LICENSE_QUESTION` (filled with the Cards' claimed licences), `ATTRIBUTION_CAVEAT`
and the three `DISCLAIMERS`.

### 3.4 Data source policy (D84)

- The training and eval text comes from **one pinned public Hub dataset repository**, two named files (Parquet with a
  `text` column, or UTF-8 `.txt`), fetched by scout itself through the logged `HubSource(repo_type="dataset")` at a
  40-hex revision, verified against the API's `lfs.sha256`, and held in memory only.
- The dataset's licence string (from its README front matter `license`, if present) is recorded in the recipe and shown
  next to the model licences in the licence question. Whether the data licence permits this use is part of the same
  human decision.
- Size limit: `data_max_bytes` = 32 MiB in total, so data never needs a new byte class or gate (P1: meta reads under
  the threshold are logged and allowed). Larger corpora need a data byte class and a data gate (later.md).
- Text preparation: the `text` column (or the file) is joined with `"\n"`, tokenised with the subject's own
  `tokenizer.json` (`tokenizers.Tokenizer.from_str`), and cut into non-overlapping `seq_len` windows. Training batches
  sample windows with `numpy.random.default_rng(seed)`; the eval set is the first `n_eval_seqs` windows of the eval
  file (or all of them if fewer).
- The exit uses WikiText-2 (raw): train on the train split, evaluate on the test split. That eval is in-domain for the
  retraining data, which favours the retrained model on perplexity; the top-1 agreement with the unmodified subject
  does not share that bias, which is why C12 has both (§2.4). The report states it.
- Rejected: self-distillation data sampled from the subject (autoregressive sampling of millions of tokens costs more
  GPU time than the training, and fitting the subject's outputs pushes the new neurons towards the old function; see
  §2.4 for the measured effect of a distillation term); user-supplied local files (their bytes would not be fetched and
  logged by scout, and their provenance could not be pinned).

### 3.5 Training framework (D82)

- **PyTorch + transformers + tokenizers**, in a new optional extra `retrain` (`torch>=2.4,<3`,
  `transformers>=4.51,<6`, `tokenizers>=0.20,<1`), which `dev` includes. No `scout` module outside `scout/retrain/`
  imports them, and `scout/retrain/` imports them only inside functions (static import test, T405), so every P1–P4
  path runs without them.
- The model is `transformers.AutoModelForCausalLM.from_config(AutoConfig.for_model(**config.raw))` (no hub access:
  config and weights come from the Card and the scratch tensors). The state dict is built from the scratch tensors
  (decoded with `scout.dequant` to float32, then cast), loaded with `strict=True` except that `lm_head.weight` may be
  missing when `tie_word_embeddings` is true.
- Device: `cuda` when available (bf16 autocast for the forward, fp32 trainable parameters and optimizer state), else CPU
  (fp32). The offline rehearsal runs on CPU with the same code.
- Optimiser: AdamW (lr 1e-3, betas (0.9, 0.95), weight decay 0), 100 warmup steps, cosine decay to 10 % of the peak,
  gradient clipping at 1.0, batch 16 × 512 tokens, up to 2000 steps, eval every 250 steps. Every value is in
  `RETRAIN_PARAMS` (§4.8) and frozen.
- Rejected: a hand-written Qwen2/Llama forward in torch (about 150 lines of numerics to prove correct; the baseline
  perplexity check would still be the only guard) and a numpy trainer (no GPU; about 14 h on the Spark's CPU for the
  exit budget). transformers is used as a model library only; its hub, cache and download code is never called
  (offline env flags, a socket-guarded worker in the rehearsal, T418/T419).

### 3.6 Report export (D85, D86, D87)

**Format.** One self-contained HTML file per subject, produced by `scout report TARGET … --out FILE.html`:
- inline CSS (with a `@media print` sheet, so "print to PDF" in any browser gives the PDF), inline SVG for the depth
  strip, no JavaScript, no external URL in any `src`, `href` of a `<link>` or CSS `url()`;
- exactly one `<script type="application/json" id="scout-report-bundle">` holding the report bundle (`report.v1`,
  §4.5), from which the HTML is a pure function (`render(bundle)`, byte-deterministic).

Markdown was rejected as the primary format (no machine-checkable provenance attributes, no depth strip), and native
PDF generation would add a heavy dependency (WeasyPrint or a browser); the print sheet covers PDF.

**Sections, in order** (each has a fixed `id`; a section that does not apply is rendered with a one-line reason
instead of its body):
1. `header`: title "scout lineage report", the subject, generation time, scout version.
2. `summary`: a short list of template sentences filled from the bundle (§4.5 `summary`), e.g. the System 1 verdict per
   reference, the attribution count per reference, the retrain outcome.
3. `caveats`: the three `DISCLAIMERS`, verbatim, always.
4. `subject`: repo and pinned revision, component, the model-card claims (`base_model`, relation, licence, labelled as
   claims), the architecture summary (stacks and depths, parameter count, dtypes).
5. `system1`: `VERDICT_SEMANTICS`; the JEV decision thresholds `hi`/`lo` (2 decimals) and calibration note from the
   model mirror; one row per reference with the calibrated verdict, `p_derived` (2 decimals), the abstain reason, the
   structure alignment; the claims table (claimed vs detected); unclaimed detections. "Not run" when the subject Card
   has neither σ-sample nor anchor evidence, or no JEV model is given.
6. `system2` (optional, `--s2`): the S2 final verdicts, reason codes and explanations, headed "System 2 commentary
   (an LLM reading Cards and System 1 output; it can only keep or soften a verdict)".
7. `attribution`: the depth strip (SVG), one row per reference (role, status, `n_attributed/n_tested`, null summary,
   `crit`), one row per block (status, primary, `z_adj`, `p`, `z_pair`), `ATTRIBUTION_CAVEAT`. "Not run" when no Card
   of the subject has `stats.attribution`.
8. `retrain`: the recipe, the baseline, the trace table, the outcome and stop step, `RETRAIN_CAVEAT`. "Not run" when
   no Card v4 of the subject exists.
9. `license`: the licences claimed by the subject's and every reference's model card (and the dataset's, when a retrain
   section exists), then `LICENSE_QUESTION`. Always present.
10. `provenance`: every Card used (repo@sha, component, object id, schema version, and its ledger citation), the JEV
    model sha256 and its `FREEZE_MODEL` citation, every ledger cited (file, seq, sha256 of the line), `code_id`,
    `scout_version`, `REPORT_PARAMS` digest, the bundle sha256, and the exact command that regenerates the file.

**Traceability mechanism.** Every value that comes from data is rendered as
`<data value="RAW" data-src="SOURCE" data-fmt="FMT">TEXT</data>`, where `TEXT == format(RAW, FMT)`. `SOURCE` is one of:
- `card:<object_id>#<json-pointer>` into the Card JSON of that store object;
- `model:<sha256>#<json-pointer>` into the JEV model file whose sha256 is `<sha256>` (64 lowercase hex, equal to
  `inputs.model.sha256`). The bundle carries a mirror of the parts the report shows, `data.models[<sha256>]`, with the
  file's own key paths (`/params/calibration_note`, `/thresholds/hi`, `/thresholds/lo`), so one pointer resolves to
  the same value in the mirror (render) and in the file (lint). Only these three pointers are allowed (L2b). The
  linter resolves the pointer in the file given to it, and only if that file's sha256 equals `<sha256>`;
- `ledger:<p3|p4|p5>:<seq>#<json-pointer>` into that ledger entry;
- `analysis:<i>#<json-pointer>` into `analysis.v1` recomputed by the linter from the cited Cards and the model file
  (`inputs.analysis[i]` names them): the numbers are a deterministic function of Cards and the frozen model;
- `s2in:#<json-pointer>` and `s2:#<json-pointer>` only inside the System 2 section (what the LLM was shown and said);
- `bundle:#<json-pointer>` only for the report's own metadata (`/generated_at`, `/scout_version`, `/code_id`,
  `/params_digest`, `/regenerate`, `/inputs/...`); L10b verifies these by rebuilding the bundle from its inputs.

Identifiers (repo names, shas, object ids) use the same element with `data-fmt="id"`. Static template text contains no
digit at all (REPORT_PARAMS templates are digit-free, tested), so "a digit outside a `data` element" is exactly "an
untraced number".

**Ledger citations.** A Card is cited by the ledger entry that recorded its file hashes or object id: P3 `SCAN`
(confirm) `card_files`, P4 `P4_EXIT` `targets.<label>.object`, P5 `P5_RETRAIN` `card_object`. The JEV model is cited by
the P3 `FREEZE_MODEL` entry with its sha256. Outside the exits, a Card without a ledger entry is shown as "not in a
ledger" (allowed in product use; the exit linter requires every citation, `--require-ledger`).

### 3.7 Report linter (`scout report-lint`, T411; frozen `LINT_PARAMS`)

The linter reads the HTML file, the store, the JEV model file and the ledgers, and returns one row per check with a
count and a threshold. Every threshold is a count that must be 0, except where stated.

| id | what | counted | pass |
|---|---|---|---|
| L1 | caveats | the three `DISCLAIMERS` each exactly once inside `#caveats`; `ATTRIBUTION_CAVEAT` exactly once inside `#attribution` when it is run; `RETRAIN_CAVEAT` exactly once inside `#retrain` when run; `LICENSE_QUESTION` exactly once inside `#license`; count = missing or duplicated texts | 0 |
| L2 | traceability | (a) digits in visible text outside `data` elements and outside `#system2`; (b) `data` elements whose source is malformed or not allowed (`bundle:` outside the metadata pointers, `model:` outside the three mirrored pointers, `s2:`/`s2in:` outside `#system2`), does not resolve, or whose `format(resolved, fmt) != TEXT`, or whose `value` differs from the resolved value; (c) numbers in `#system2` prose that equal no numeric leaf of the `s2in.v1` input at the printed precision | 0 each |
| L3 | raw cosine | case-insensitive matches of `cosine` or `cos sim` in visible text and in bundle keys | 0 |
| L4 | calibrated verdicts only | (a) `.verdict` elements whose source is not `analysis:<i>#/pairs/<j>/system1/verdict` with a model whose sha256 equals the P3 ledger's last `FREEZE_MODEL` sha256; (b) the words `derived`, `not derived`, `not_derived`, `abstain` (any case, word-bounded) in visible text outside `.verdict`, `.semantics`, `.caveat`, `.attrib-status` (a block-attribution reference status, which can be `abstain`), `#system2` and `#license` | 0 each |
| L5 | abstentions shown | one row whose count is the sum of four absolute differences: (a) S1 abstentions in the recomputed analysis vs `.verdict[data-verdict=abstain]` elements **inside `#system1` only** whose table row carries a non-empty `.reason` (the `#summary` repeats each verdict, so counting it would double-count; summary verdicts are still subject to L4a); (b) attribution references with status `abstain`/`untestable` vs rendered reference rows with a reason; (c) `untestable` blocks vs rendered block rows with a reason; (d) "not run" sections without a reason | 0 |
| L6 | provenance | Cards used without repo@40-hex, component, object id (existing in the store, and equal to the id recomputed from its files) or (with `--require-ledger`) a verifying ledger citation; missing model sha or `FREEZE_MODEL` citation; missing `code_id` (64 hex), scout version or regenerate command | 0 |
| L7 | self-contained | `src=`, `<link`, `@import`, `url(` with an `http:`/`https:` target; `<script>` elements other than the one bundle | 0 |
| L8 | licence left open | matches of the frozen deny-list (§4.8) in visible text outside `.license-question` and `.caveat`; `#license` missing | 0 |
| L9 | precision | `p_derived` values and the JEV thresholds (`/thresholds/hi`, `/thresholds/lo`) rendered with other than `f2`; attribution `z_adj`/`z_pair` with other than `f1`; p-values with other than `e1` | 0 |
| L10 | unedited and reproducible | (a) `render(bundle) != file bytes`; (b) the bundle rebuilt from the store, model and ledgers (with the file's `generated_at`) differs from the embedded bundle | 0 each |

L10 is what makes "without editing" mechanical: any hand edit of the HTML breaks (a), and any edit of the bundle breaks
(b). The exit also pins each file's sha256 in `P5_REPORTS` (C15).

### 3.8 Blind-reader checklist (D88; frozen `READER_PARAMS`)

Ten questions per report, answered from the rendered report alone. The answer key is computed from the bundle by frozen
code (`answer_key(bundle)`), and its sha256 is written to `P5_REPORTS` **before** any reader answers. Questions 9 and 10
depend on whether the report has a retrain section.

| # | question | answer format | key |
|---|---|---|---|
| Q1 (critical) | Which model is this report about (repository and revision)? | `repo@sha` (≥ 7 hex) | subject key |
| Q2 (critical) | Which references, if any, does System 1 call derived? | list of repos, `none`, or `not run` | S1 derived set |
| Q3 | On which references did System 1 abstain, and why? | list of `repo: reason` | S1 abstain set with reason codes |
| Q4 | How many blocks does the block attribution attribute to each reference? | `repo: n of N` list, or `not run` | per reference `n_attributed`/`n_tested` |
| Q5 | Which independent control model calibrated the block-attribution test? | repo or `not run` | control repos |
| Q6 (critical) | Could the subject have been trained by distillation from another model without this report detecting it? | yes/no | yes |
| Q7 | Does the report treat a shared tokenizer as proof of derivation? | yes/no | no |
| Q8 (critical) | Does the report say whether the subject may be used or redistributed under the licences shown? | yes/no | no |
| Q9 (retrain) | Which blocks were retrained, and did their attribution signal drop below the threshold? | block list; yes/no | recipe blocks; outcome success |
| Q10 (retrain) | Is the retrained checkpoint still derived from the base, according to the report? | yes/no | yes |
| Q9 (no retrain) | Which sections were not run? | list of section names | sections with a "not run" reason |
| Q10 (no retrain) | Does "not attributed" for a block mean the block was trained independently? | yes/no | no |

Each reader also lists **required edits**: statements that are factually wrong, contradict another part of the report,
or would mislead a reader, each with a quote. Optional style edits are recorded but not scored.

**Pass (per report and reader):** correct `>= 9` of 10, all critical questions correct, required edits `== 0`.
Scoring is deterministic (T412): repos compared case-sensitively after trimming, shas by prefix, lists as sets, yes/no
exact, section names case-insensitive.

**Readers.**
- **LLM reader (C16).** One call per report through the P3 `S2Client` seam (`AnthropicS2Client`, default
  `claude-opus-5-5`, adaptive thinking, effort high, structured output with `READER_SCHEMA`), in a fresh context. The
  input is the report's **visible text** (HTML to text, the bundle removed) and the questions; the system prompt forbids
  outside knowledge. Offline, `StubS2Client` answers from fixtures.
- **Human reader (C17).** A person who has not seen the plan, the code or any scout output, recorded by name or
  initials with `blind: true` in the answers file. `reader-form` writes the questions without answers. **Open decision
  D89**: if no such person is available, the default is that Phase 5 stays open; the human may instead record a logged
  decision (`--waive-human "<reason>"`) that marks C17 `WAIVED` in the EXIT table, and the phase then closes as "PASS
  with waiver", listed as such.

### 3.9 Black-box behavioural signal: deferred (D90)

Not included. A behavioural signal (comparing generations of the subject and a candidate on a fixed prompt set to
detect distillation) needs generation on the GPU or through an API for every candidate, a prompt set, and above all a
labelled set of documented distillation pairs to calibrate against; none exists. An uncalibrated signal would sit
next to calibrated weight evidence and invite being read as a verdict. The report states the blind spot instead
(DISCLAIMERS 1; reader Q6 checks that a reader gets it). later.md records the design constraints for a future phase:
strictly separate section, never an input to JEV or System 2's verdict, its own calibration set, and never a verdict on
its own.

### 3.10 UX polish and usage notes (D91)

**Usage notes.** `docs/usage-notes.md` holds one note per line:
`- YYYY-MM-DD | area | kind | key | text`, with `area` ∈ {scan, diff, analyze, attribute, retrain, report, ui, cli,
jobs, store}, `kind` ∈ {bug, blocker, friction, wish}, and `key` a short kebab-case slug naming the problem.

**Triage rule** (`scripts/usage_notes.py triage`, T415). A key becomes a task candidate when any of these holds:
1. at least one note of that key has kind `bug` or `blocker`;
2. notes of that key (kind `friction` or `wish`) exist on at least 2 distinct dates;
3. a note of that key cites an exit check (`C<n>` or `L<n>` in its text) that it blocks.

The script prints the candidates (key, area, dates, notes) and exits 0; the next wave's planner turns each candidate
into a task or a later.md line with a reason. Notes are never deleted; a triage appends
`- YYYY-MM-DD | triage | - | <key> | -> <T### or later.md>`.

**Polish included in P5**, and only because an earlier phase deferred it to P5 and P5's own features need it:
- a read-only report view in the web UI (`GET /api/report`, a "report" link per stored subject), which is the P3
  "analysis and System 2 panel" item in its read-only form, and the P4 "report export" item;
- the retrain panel: depth-strip cells of retrained blocks outlined, with before/after status, and a retrain legend with
  the outcome, `RETRAIN_CAVEAT` and `LICENSE_QUESTION`.

Everything else is seeded into `docs/usage-notes.md` as `wish` notes (source: later.md) and waits for the triage rule:
subtree collapse/expand (P1), starting attribution or retrain jobs from the UI with a gate panel (P4), System 2 calls
from the UI (P3), release-date direction hints (P3).

## 4. Data structures (exact field names)

### 4.1 Card v4 (T401; only `card.with_retrain` writes it)

```
schema_version: "card.v4"
stats: {tensor_stats, spectral_topk, sigma_curves, tokenizer_minhash, anchor_embedding, attribution,
        retrain: retrain.v1}                                                   # NEW slot, the only difference to v3
```
- A Card v4 is the subject's header-scan Card v3 (key, source, scan, model_card, config, pipeline, weights, structure,
  Parquet) with `stats.attribution` = the job's baseline attribution (the unmodified subject), `stats.retrain` filled,
  and `fetch_log` = the retrain client's log (its scan events, the client's gate/job events and the ingested worker
  events, `disk`, `jobs`), exactly as P4 `with_attribution` composes it.
- The key is the subject's `(repo, revision_sha, component)`: the Card is about the subject; the retrain record is an
  experiment on it. The store keeps it as another object under the same key (P4 dedup rules).
- Parquet v4 = Parquet v3 (same columns); metadata `schema_version: card.v4`.
- `card.SCHEMA_VERSION` stays `"card.v3"`: every scan, attribution and store `params_digest` is unchanged, and no P1–P4
  check ever sees a v4 Card. `SUPPORTED_SCHEMA_VERSIONS` gains `"card.v4"`, so `load_card`, the store, `build_view`
  and the P3 `LoadedCard` accept it.

### 4.2 `retrain.v1` (Card v4 `stats.retrain`; T406 builds it, T408 adds `job`)
```
{version: "retrain.v1", params: RETRAIN_PARAMS, budget: "exit"|"rehearsal",
 recipe: {recipe_id: str (16 hex), recipe_digest: str (64 hex), stack_prefix: str, blocks: [int],
          reinit: {roles: ["gate", "up", "down"], init_std: float, seed: int, down: "zeros"},
          trainable: [str],                                     # the 3 x len(blocks) tensor names
          objective: "causal-lm-cross-entropy",
          optimizer: {name: "adamw", lr, betas: [float, float], weight_decay, warmup_steps, schedule, grad_clip},
          batch_size, seq_len, max_steps, eval_every, n_eval_seqs, seed,
          data: {repo, revision_sha, train_file, train_sha256, train_bytes, eval_file, eval_sha256, eval_bytes,
                 license: str|null, n_train_tokens: int, n_eval_tokens: int},
          base: {card_key, title}, controls: [{card_key, title}]},
 env: {device: "cuda"|"cpu", device_name: str|null, dtype: str, torch: str, transformers: str, tokenizers: str,
       cuda: str|null},
 baseline: {eval_loss: float|null, ppl: float|null,
            flagged: [{index, z, z_adj, z_pair, crit, attributed: bool}],
            ok: bool, reason: str|null},
 reinit: {eval_loss: float, ppl_ratio: float, top1_agreement: float}|null,   # right after re-initialisation (step 0)
 trace: [{step: int, train_loss: float, eval_loss: float, ppl_ratio: float, top1_agreement: float,
          base_status: "tested"|"abstain"|"untestable", null_ok: bool, control_ok: bool,
          flagged: [{index, z: float|null, z_adj: float|null, z_pair: float|null, crit: float|null, attributed: bool}],
          unflagged_attributed: int, unflagged_tested: int, control_hits: int,
          signal_dropped: bool, quality_ok: bool, wall_s: float}],
 outcome: "success"|"signal_persists"|"quality_not_recovered"|"neither"|"refused_baseline"|"refused_forward_check",
 stop_step: int|null, reason: str|null,
 final_attribution: attribution.v1|null,         # at stop_step, or at the last eval step; null when refused
 retrained_digest: str|null,                     # "sha256:<hex>" over the final flagged tensors; null when refused
 weights_kept: false,
 lineage: {parent: card_key, parent_object: str, relation: "partial-mlp-retrain", statement: RETRAIN_CAVEAT},
 license: {subject: str|null, references: [{title, license: str|null}], data: str|null, question: LICENSE_QUESTION},
 caveat: RETRAIN_CAVEAT,
 job: {job_id, host, plan_id}|null, computed_at: ISO-8601 Z}
```
- `signal_dropped` = base `tested` and `null_ok` and `control_ok` and no flagged block `attributed`.
- `quality_ok` = `ppl_ratio <= max_ppl_ratio` and `top1_agreement >= min_top1_agreement`.
- `outcome` = `success` at the first row with both; otherwise, after `max_steps`, from the last row: `signal_persists`
  (quality ok, signal not dropped), `quality_not_recovered` (dropped, quality not ok), `neither`.
- Floats are rounded to 6 decimals. `validate_retrain` (T401) checks every key and type, the enums, that the trace
  steps are strictly increasing multiples of `eval_every`, that `stop_step` is the first row with both flags when the
  outcome is `success`, and that `weights_kept is False`.

### 4.3 Texts (T401, `scout/caveats.py`; digit-free, tested)
```
RETRAIN_CAVEAT = "Retrain experiment: only the MLP gate, up and down projections of the listed blocks were
  re-initialised and retrained; every other weight (attention, norms, embeddings and all other blocks) is unchanged
  from the subject, so the retrained checkpoint remains derived from the subject's base. A block that is no longer
  attributed has lost the matching-test signal; that is not evidence of independent training, and longer training can
  bring the signal back, because the retrained neurons learn the function the rest of the model expects. The retrained
  weights were deleted after the experiment; only this record is kept."
LICENSE_QUESTION = "Licence question, not answered by scout: the model cards claim the licences listed here. Whether
  using, modifying, retraining or redistributing this model, or training on the listed data, is permitted under them
  is a decision for a human, and scout does not assess licence terms or compliance."
```
(Each is one line in code; wrapped here.) `DISCLAIMERS` (P1) and `ATTRIBUTION_CAVEAT` (P4) are unchanged and re-exported
by value; `scout/caveats.py` imports nothing from scout, and a test asserts equality with `scout.view.DISCLAIMERS` and
`scout.attrib.ATTRIBUTION_CAVEAT`.

### 4.4 Retrain job spec and messages (T407)
`job.v1` gains a second kind (§9). Fields that differ from the P4 attribute spec:
```
{schema: "job.v1", kind: "retrain", ..., params_digest, retrain_params_digest: str,
 references: [base target], controls: [control target],
 retrain: {blocks: [int], budget: "exit"|"rehearsal",
           data: {repo, revision_sha, train_file, eval_file},
           parent_object: str}}                   # the store object id of the subject's header-scan Card (lineage)
```
- `budget` names one of `RETRAIN_PARAMS.budgets`; no hyperparameter travels in the spec.
- `JobResult` for a retrain job: the P4 fields (`attribution` = the baseline attribution of the unmodified subject,
  `job` null) plus `retrain` (`retrain.v1`, `job` null) and `retrain_env` (below).
- New worker subcommand `python -m scout.jobs.worker retrain-env` prints one JSON line
  `{"type": "retrain_env", "torch": str|null, "torch_cuda": str|null, "cuda_available": bool, "device_name": str|null,
  "gpu_mem_total_bytes": int|null, "transformers": str|null, "tokenizers": str|null, "retrain_params_digest": str,
  "error": str|null}` and exits 0 (also when torch is missing: `error` says so).
- The retrain plan is a P4 `FullPlan` (same dataclass; subject reads carry role `"train"` for tensors outside the
  attribution roles), plus a `RetrainPlan` wrapper: `{plan: FullPlan, data_files: [{repo, revision_sha, path, size,
  lfs_sha256}], data_bytes: int, recipe: dict, recipe_digest: str, attrib_plan_id: str}`.

### 4.5 Report bundle `report.v1` (T409; embedded in the HTML, never stored by scout)
```
{schema: "report.v1", params_digest: str, generated_at: ISO-8601 Z, scout_version: str, code_id: str|null,
 subject: {target: str, card_key, title},
 inputs: {cards: [{role: "subject-analysis"|"subject-attribution"|"subject-retrain"|"reference", card_key, object,
                   schema_version, citation: {ledger: "p3"|"p4"|"p5", seq: int, line_sha256: str}|null}],
          model: {name: str, sha256: str, citation: {...}|null} | null,
          analysis: [{subject_object: str, reference_objects: [str], model_sha256: str}],   # 0 or 1 entries
          ledgers: [{name, file_name, n_entries, last_seq, last_line_sha256}]},
 sections: {system1: {status: "run"|"not_run", reason: str|null},
            system2: {status, reason, s2in_sha256: str|null},
            attribution: {status, reason, card_object: str|null},
            retrain: {status, reason, card_object: str|null}},
 data: {cards: {object_id: {key, model_card, config: {model_type, architectures}, structure: {stacks: [{prefix, depth}]},
                            weights: {params_total, n_tensors, params_by_dtype, format}, stats: {attribution, retrain}}},
        analysis: [analysis.v1], s2: s2.v1|null, s2in: s2in.v1|null,
        models: {"<model sha256>": {params: {calibration_note: str}, thresholds: {hi: float, lo: float}}},
                # mirror of the JEV model file, keyed by its sha256, same key paths; {} when System 1 is not run
        ledger_entries: {"<ledger>:<seq>": entry}},
 summary: [{template: str, slots: {name: SOURCE}}],
 regenerate: str}
```
`render(bundle) -> bytes` reads values only from `bundle.data`, addressed by the same `SOURCE` strings the linter resolves
against the store and the ledgers (§3.6). `answer_key(bundle)` (T412) derives the reader key from it.

### 4.6 `lint.v1` (T411)
```
{schema: "lint.v1", file_sha256: str, ok: bool,
 rows: [{id: "L1".."L10", name: str, count: int, threshold: 0, ok: bool, details: [str]}]}   # details <= 20 lines
```
L2, L4 and L10 report their sub-counts as `L2a`, `L2b`, `L2c`, `L4a`, `L4b`, `L10a`, `L10b` rows (same shape), so a
result has 14 rows in this order: L1, L2a, L2b, L2c, L3, L4a, L4b, L5, L6, L7, L8, L9, L10a, L10b.

### 4.7 Reader records (T412)
```
READER_SCHEMA (LLM output) = {answers: [{q: "Q1".."Q10", answer: str}],
                              required_edits: [{quote: str, reason: "factually_wrong"|"contradicts_report"|"misleading"|"unreadable"}],
                              optional_edits: [str]}
human answers file (YAML) = {reader: {id: str, blind: bool, date: YYYY-MM-DD},
                             reports: {label: {answers: {Q1: str, ..., Q10: str}, required_edits: [{quote, reason}]}}}
score = {reader: "llm"|"human", label, correct: int, n: 10, critical_ok: bool, required_edits: int,
         per_question: {Q: bool}, passed: bool}
```

### 4.8 Frozen parameters (literals; asserted by tests, by C2 and recorded in `P5_FREEZE`)
```
RETRAIN_PARAMS = {"version": "retrain.v1", "framework": "torch+transformers",
  "model_types": ["llama", "qwen2", "qwen3", "mistral"],
  "reinit_roles": ["gate", "up", "down"], "reinit_init": "normal(0,initializer_range):gate,up;zeros:down",
  "init_std_default": 0.02, "trainable": "reinit-only", "objective": "causal-lm-cross-entropy",
  "optimizer": "adamw", "lr": 0.001, "betas": [0.9, 0.95], "weight_decay": 0.0,
  "schedule": "linear-warmup-cosine-to-0.1", "grad_clip": 1.0, "seed": 20260929,
  "budgets": {"exit": {"batch_size": 16, "seq_len": 512, "max_steps": 2000, "eval_every": 250, "warmup_steps": 100,
                       "n_eval_seqs": 128},
              "rehearsal": {"batch_size": 16, "seq_len": 64, "max_steps": 400, "eval_every": 50, "warmup_steps": 10,
                            "n_eval_seqs": 16}},
  "dtype_cuda": "bf16-autocast-fp32-master", "dtype_cpu": "float32",
  "max_ppl_ratio": 1.05, "min_top1_agreement": 0.90, "baseline_ppl_range": [2.0, 60.0],
  "min_unflagged_attributed_frac": 0.95, "data_max_bytes": 33554432, "data_formats": ["parquet:text", "txt"],
  "offline_env": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
  "stop_rule": "first eval step with signal_dropped and quality_ok"}
REPORT_PARAMS = {"version": "report.v1", "format": "html-selfcontained", "bundle_script_id": "scout-report-bundle",
  "sections": ["header", "summary", "caveats", "subject", "system1", "system2", "attribution", "retrain", "license",
               "provenance"],
  "formats": {"p_derived": "f2", "z": "f1", "p": "e1", "ppl": "f2", "ratio": "f3", "agreement": "f3", "loss": "f4",
              "count": "int", "id": "id", "text": "str", "time": "str"},
  "sources": ["card", "model", "ledger", "analysis", "s2in", "s2", "bundle"]}
LINT_PARAMS = {"version": "lint.v1", "checks": ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10"],
  "number_regex": "[-+]?[0-9]+(?:\\.[0-9]+)?(?:[eE][-+]?[0-9]+)?",
  "cosine_regex": "(?i)cos(?:ine|\\s*sim)",
  "verdict_regex": "(?i)\\b(?:not[ _]derived|derived|abstain(?:s|ed)?)\\b",
  "verdict_containers": [".verdict", ".semantics", ".caveat", ".attrib-status", "#system2", "#license"],
  "license_deny": ["compliant", "complies", "compliance with", "violat", "infring", "permitted", "is allowed",
                   "are allowed", "you may ", "you can use", "free to use", "legal", "lawful", "no licence issue",
                   "no license issue"],
  "license_containers": [".license-question", ".caveat"],
  "external_regex": "(?i)(?:\\bsrc\\s*=|<link\\b|@import|url\\()\\s*['\"]?https?:"}
READER_PARAMS = {"version": "reader.v1", "n_questions": 10, "min_correct": 9, "critical": ["Q1", "Q2", "Q6", "Q8"],
  "max_required_edits": 0, "llm_model": "claude-opus-5-5", "llm_effort": "high", "llm_max_tokens": 8000,
  "edit_reasons": ["factually_wrong", "contradicts_report", "misleading", "unreadable"],
  "questions": <the §3.8 table, verbatim, as {id: {text, format, variant: "all"|"retrain"|"no_retrain", critical}}>}
```
The rehearsal budget changes only sizes and step counts; every threshold (`max_ppl_ratio`, `min_top1_agreement`,
`baseline_ppl_range`, the attribution rule) is the same for both budgets.

### 4.9 Ledger `exit/ledger_p5.jsonl` (P3 `p3ledger.v1` line format; kinds `P5_*`; T417)
```
P5_FREEZE    {params: {retrain, report, lint, reader}, params_sha256: {...}, retrain_target, report_targets,
              pins_sha256, model_sha256, cited_ledgers: {p3: {last_seq, last_line_sha256}, p4: {...}},
              refreeze_reason: str|null, git_commit, code_id, code_dirty}
P5_PREFLIGHT {hello, retrain_env, checks: {name: ok}, git_commit, code_id}
P5_PLAN      {plan: {plan_id, bytes_planned, bytes_cap, disk_bytes, data_bytes, recipe_digest}, git_commit, code_id}
P5_ATTEMPT   {plan_id, bytes_cap, confirm_plans, prior_attempts: int, git_commit, code_id}
P5_RETRAIN   {attempt_seq, passed, failed: [check], error: str|null, evidence: {...§2.3 payload}, card_object: str|null,
              git_commit, code_id}
P5_REPORTS   {reports: {label: {file_name, sha256, bytes, lint_ok, lint_rows: [{id, count, ok}], answer_key_sha256,
              bundle_sha256}}, passed, git_commit, code_id}
P5_READER    {reader: "llm"|"human", reports_seq: int (the P5_REPORTS entry read), reader_id: str, blind: bool|null, waived: str|null,
              scores: {label: score}, answers_sha256: str, s2: {requested_model, served_model, request_host,
              usage}|null, passed, git_commit, code_id}
P5_EXIT      {passed, failed: [check], rows: [{target, check, expected, actual, ok}], waivers: [str],
              prior_attempts: int, git_commit, code_id, git_dirty}
```

### 4.10 View additions (T413)
```
depth_strips[].cells[].retrain: null | {flagged: bool, before: str|null, after: str|null}   # attribution statuses
view.retrain: null | {outcome, stop_step, blocks: [int], ppl_ratio: float|null, top1_agreement: float|null,
                      caveat: RETRAIN_CAVEAT, license_question: LICENSE_QUESTION}
```

### 4.11 Log events
No new `event` type. The retrain worker logs through the P4 `job`, `gate` and `purge` notes plus one new note text per
eval step, `event: "job"`, `note: "retrain eval step <n>: ppl_ratio <r> top1 <a> flagged_attributed <k>/<K>"` (0 bytes),
and the data files as ordinary META `fetch` events of class meta. The ByteLog is unchanged.

## 5. Assumptions and open decisions (each with a recommended default)

Assumptions:
- **A10.** P1–P4 are implemented exactly as specified in T001–T321.
- **A11.** The P3 and P4 exits have run: `cards/p3`, `models/jev_v1.json`, `exit/ledger_p3.jsonl`, `cards/p4-store` and
  `exit/ledger_p4.jsonl` exist on the client host (C0 FAILs with a named missing input otherwise).
- **A12.** The DGX Spark facts of P4 §2.5 plus the retrain facts of §2.5 (**V**, checked by C1 at E3).
- **A13.** The WikiText-2 files exist at the stated paths in `Salesforce/wikitext` with a `text` column (**V** at E1/E3;
  a mismatch is a plan revision before `P5_FREEZE`).
- **A14.** transformers builds a Qwen2 or Llama model from `config.raw` and a state dict without any network access
  (`from_config` never downloads; the offline env flags and the rehearsal's socket guard make any attempt fail loudly).

| # | Decision | Default (recommended) | Rationale |
|---|---|---|---|
| D81 | Retrain loop | Re-initialise the MLP gate/up (normal, `initializer_range`) and down (zeros) of the flagged blocks; train only those; cross-entropy on pinned data; re-fingerprint with the unchanged P4 statistic every `eval_every`; stop at the first eval step with signal dropped (base tested, `null_ok`, `control_ok`) and quality within bounds (§3.1) | The P4 test sees gate/up neurons only; zero `down` makes step 0 "MLP removed", not "random noise injected"; the stop rule uses the P4 rule verbatim, so "dropped" means exactly what the depth strip means |
| D82 | Framework | torch + transformers + tokenizers in a `retrain` extra (in `dev`); CUDA on the Spark, CPU in the rehearsal; imports only inside `scout/retrain/` functions | Standard, supports every `model_types` entry, no hand-written numerics to verify; the core stays importable without it |
| D83 | Retrained weights and invariant 1 | **No exception.** Weights live in worker memory during COMPUTE only, are deleted in `finally`, and are never written to disk; the Card v4 keeps the recipe, trace, digest and statistics. **Export to the user is not built** (later.md, a human decision) | Invariant 1 holds unchanged, and scout does not become a one-command fingerprint-removal tool while the licence question is open. The recipe makes the checkpoint reproducible outside scout. **Open for the human**: if an export path is wanted, it needs an explicit invariant 1 amendment first |
| D84 | Data source | One pinned public Hub dataset, two files (train, eval), fetched and logged by scout, sha256-checked, ≤ 32 MiB, licence recorded; the exit uses WikiText-2 (§3.4) | Stays inside invariant 2 without a new byte class or gate; provenance is pinned like models |
| D85 | Report format | Self-contained HTML with an embedded JSON bundle; print CSS for PDF; no Markdown or native PDF | Provenance attributes and the depth strip need HTML; zero new dependency |
| D86 | Traceability | Every rendered value is a `data` element with a `SOURCE` pointer into a Card, the JEV model, a ledger entry, or the deterministic analysis of cited Cards; templates are digit-free | "Every number traceable" becomes "0 digits outside `data` elements and 0 unresolved pointers" (L2) |
| D87 | Linter | L1–L10 of §3.7, every threshold 0, frozen `LINT_PARAMS` | Each clause of the exit sentence becomes a count |
| D88 | Blind readers | 10 frozen questions (4 critical), key computed by code and hashed before answers; pass ≥ 9/10, all critical, 0 required edits; one LLM reader and one human reader | Comprehension and "no edit needed" are what the phase exit asks; the key cannot be adjusted after answers |
| D89 | Human reader availability | **Open for the human.** Default: required, and the phase stays open without one; a logged `--waive-human "<reason>"` marks C17 `WAIVED` and the phase closes "PASS with waiver" | A self-use tool may have no second person, but the exit claim is about another person |
| D90 | Black-box behavioural signal | Deferred (§3.9, later.md) | No calibration set exists; an uncalibrated signal must not sit next to verdicts |
| D91 | UX polish | Usage-notes file and triage rule (§3.10); in P5 only the report view and the retrain panel | No usage notes exist yet; both items are P5 features that earlier phases deferred to P5 |
| D92 | Card version | `card.v4` = v3 + `stats.retrain`, written only by `with_retrain`; `SCHEMA_VERSION` stays `card.v3` | No P1–P4 check or test literal changes (P4's v3 flip had to touch 8 files); scans and store digests are unchanged |
| D93 | Exit retrain target | §2.1: Qwen2.5-0.5B-Instruct, base Qwen2.5-0.5B, control SmolLM2-360M, blocks 6/12/18, WikiText-2 | Small (2.1 GB of reads), ungated, documented, P3-pinned Cards reused; GPU time well under an hour. Alternative: `HuggingFaceTB/SmolLM2-360M-Instruct` with control `Qwen/Qwen2.5-0.5B` (same documentation quality, 32 blocks) |
| D94 | Report targets | §2.1: `retrain`, `te`, `negative`; no System 2 in the exit reports | Together they cover every section and every "not run" path with 0 new bytes; S2 is non-deterministic, so L10 could not hold for it across a rebuild (the S2 section is linted offline) |
| D95 | Exit ledger and freeze | `exit/ledger_p5.jsonl`, kinds `P5_*`; `P5_FREEZE` before any real run records every literal, target and cited hash; code frozen from it (C2); refreeze needs a recorded reason | P3/P4 conventions; the reader key hash in `P5_REPORTS` precedes every `P5_READER` |
| D96 | Client death during retraining | Abort flag set on any emit `OSError`, checked every step; `finally` purges | A retrain COMPUTE is long; the P4 45-minute bound would become the whole budget |
| D97 | P5 store | `cards/p5-store`, populated by `import_tree(cards/p3)` and by export/import of the P4 te entries (byte-identical, so P3/P4 ledger hashes still cite them) | Reports need one store; the P3/P4 artifacts stay untouched |
| D98 | Quality measure | `ppl_ratio` on the eval split `<= 1.05` and top-1 agreement with the unmodified subject `>= 0.90`; both also recorded right after re-initialisation | Perplexity alone is biased by in-domain retraining; agreement measures preserved behaviour; the post-re-init values show what retraining recovered (§2.4) |
| D99 | Report files and invariant 1 | A report is a user-requested export, like stdout: scout never stores or reads one back (the linter only verifies), `reports/` is gitignored, the exit keeps each file's sha256 in the ledger. **Open for human acknowledgement**, like P3 D36 | It is fully recomputable from Cards, the frozen model and ledgers (L10b proves it) |

## 6. Architecture (files)

```
scout/caveats.py, scout/card.py (card.v4, with_retrain, validate_retrain),
  tests/helpers/retrain_records.py                                                           T401
scout/hub.py (repo_type "dataset"), tests/helpers/fakehub.py (datasets)                      T402
scout/retrain/__init__.py, scout/retrain/params.py, scout/retrain/plan.py                    T403
tests/helpers/retrain_fixtures.py, pyproject.toml (retrain extra)                            T404
scout/retrain/model.py, scout/retrain/train.py                                               T405
scout/retrain/loop.py                                                                        T406
scout/retrain/spec.py, scout/retrain/job.py, scout/jobs/spec.py (kind), scout/jobs/worker.py T407
scout/retrain/client.py                                                                      T408
scout/report/__init__.py, scout/report/bundle.py                                             T409
scout/report/render.py, scout/report/templates.py                                            T410
scout/report/lint.py                                                                         T411
scout/report/reader.py                                                                       T412
scout/view.py, scout/server.py, scout/web/index.html, scout/web/app.js                       T413
scout/cli.py                                                                                 T414
docs/usage-notes.md, scripts/usage_notes.py                                                  T415
scripts/exit_expectations_p5.py                                                              T416
scripts/exit_check_p5.py, exit/pins_p5.json, scripts/pin_exit.py (--phase 5), .gitignore     T417
tests/helpers/p5_rehearsal.py                                                                T418
tests/test_exit_check_p5_offline.py                                                          T419
```

## 7. Inputs handled and where each is tested (all offline)

| Input / situation | Handling | Test |
|---|---|---|
| Derived subject, flagged blocks attributed | Retrain loop runs, outcome `success` | T406 `test_loop_success`, T407 `test_run_retrain_job_local`, T419 rehearsal |
| Flagged block not attributed at baseline | `refused_baseline`, no training, Card still records it | T406 `test_refused_baseline`, T419 `test_rehearsal_refused_baseline` |
| Forward pass does not reproduce the model | `refused_forward_check` (baseline ppl outside range) | T406 `test_refused_forward_check` |
| Retraining pulls the neurons back to the base | `signal_persists` after `max_steps` | T406 `test_signal_persists` (a stub trainer that copies the base's gate/up into the flagged blocks) |
| Quality not recovered | `quality_not_recovered` | T406 `test_quality_not_recovered` (stub eval) |
| Selector, GGUF, quantised or unsupported `model_type` subject | `RetrainRefused` before any byte | T403 `test_refuse_subjects` |
| Data file over 32 MiB, wrong sha256, missing `text` column | refused before FULL_DOWNLOAD / after the META read, 0 weight bytes | T403 `test_data_limits`, T407 `test_data_sha_mismatch` |
| Tied embeddings (`lm_head` absent) | loaded with the tie | T405 `test_build_model_tied` |
| Client dies mid-training | abort within one step, scratch purged | T407 `test_abort_on_broken_pipe` |
| Worker killed mid-training | orphan swept at the next start (P4) | T419 `test_rehearsal_crash_then_sweep` |
| Something tries the network during COMPUTE | offline env flags; socket guard raises | T405 `test_no_network_in_compute`, T419 |
| Retrained weights written to disk | never: static test for save calls; `bytes_written == bytes_planned` | T405 `test_no_save_calls`, T407 `test_no_checkpoint_written` |
| Report subject with every section | all sections rendered and linted | T410, T411, T419 |
| Report subject without σ/anchors, without attribution, without retrain | "not run" with reasons | T409 `test_sections_not_run`, T411 `test_l5_not_run_reason` |
| Hand-edited report (a number, a caveat, a verdict word) | L10a, L2b, L1, L4b fail | T411 `test_tamper_*` |
| Report with S2 commentary quoting a number that was not in its input | L2c fails | T411 `test_l2c_s2_number` |
| Licence answered in text | L8 fails | T411 `test_l8_license_answered` |
| Reader wrong on a critical question / edits requested | score fails | T412 `test_score_*`; T419 `test_rehearsal_reader_fail` |

## 8. Risks and how each is tested

| Risk | Mitigation / test |
|---|---|
| Retraining recreates the base's neurons, so the signal never drops (exit C11) | Measured on a toy (§2.4). The loop reports `signal_persists` honestly; a FAIL goes to the human, and nothing is re-seeded or re-thresholded. T406 `test_signal_persists` proves the loop reports it |
| Quality does not recover within the budget (exit C12): **the main exit risk** | Zero-init `down` (step 0 = MLP removed); both quality measures reported per eval step and right after re-initialisation; measured on two toys (§2.4): 2 of 8 blocks reach top-1 0.91 by step 250, 4 of 8 plateau at 0.85, and the exit budget is only about 0.42 tokens per trainable parameter. The bounds and the budget are frozen in `P5_FREEZE`; a FAIL goes to the human with the trace, and changing either needs a refreeze with a recorded reason (C2) |
| Retrained weights persist (invariant 1) | No save call exists (static AST test over `scout/retrain`, T405); only scratch tensors of the plan are written (C7 `bytes_written == bytes_planned`); `finally` deletes tensors and purges; weights-free Card (T401 `test_retrain_no_weights`: no numeric list longer than 64 outside `final_attribution.blocks`) |
| transformers downloads something behind the ByteLog (invariant 2) | `from_config` only; offline env flags; the rehearsal runs COMPUTE with `socket.socket.connect` patched to raise (T419) and T405 `test_no_network_in_compute` |
| Data bytes bypass the gate | Data ≤ 32 MiB as logged META reads (P1 semantics); the plan display lists them; sha256 checked; C3 and C6 recount them |
| The worker trains a different recipe than the human confirmed | `recipe_digest` is inside `plan_id`; the worker re-derives both from its own inputs and the budget name (`PlanMismatch` otherwise; T407 `test_recipe_changed_refused`) |
| Client death leaves a long job holding weights | Abort flag (D96; T407 `test_abort_on_broken_pipe`), P4 sweep |
| A report's number is wrong or unattributed | L2 resolves every pointer against the store, model and ledgers, and L10b rebuilds the bundle (T411 tests per source kind) |
| A report reads as a licence opinion | `LICENSE_QUESTION` always present (L1), deny-list outside it (L8), reader Q8 (critical) |
| A verdict without calibration (e.g. a raw σ-curve r shown as "derived") | L4a/L4b; the renderer has no code path that prints a verdict from anything but `jev.v1` (T410 `test_verdicts_only_from_jev`) |
| The LLM reader is lenient or hallucinates | Fresh context, visible text only, structured answers, deterministic scoring; the human reader is a second, independent check (C17) |
| The answer key is adjusted after seeing answers | Key sha256 in `P5_REPORTS` before any `P5_READER`; code frozen since `P5_FREEZE` (C2) |
| P1–P4 regressions | `card.v4` written only by `with_retrain`; `SCHEMA_VERSION` unchanged; the job spec keeps kind `attribute` behaviour byte-identical; every task runs the full `pytest -q` |
| torch absent where P1–P4 run | Imports inside `scout/retrain` functions only; T405 `test_core_imports_without_torch` (subprocess with torch blocked via `sys.modules` poisoning: `import scout.cli, scout.jobs.worker, scout.report` succeeds) |
| GPU nondeterminism makes a rerun differ | Stated; reruns are recorded (`prior_attempts`); a FAIL is resolved only by a logged human decision and a refreeze with a reason |

## 9. Changes to earlier contracts

| Earlier contract | Change | Task |
|---|---|---|
| `scout/card.py` (P1 T009, P2 T102, P3 T202, P4 T306) | `SUPPORTED_SCHEMA_VERSIONS` += `"card.v4"`; new `RETRAIN_VERSION = "retrain.v1"`, `validate_retrain(r) -> list[str]`, `with_retrain(card, table, attribution, retrain, extra_log, out_dir) -> BuiltCard` (writes `card.v4`); `load_card` accepts v4. `SCHEMA_VERSION`, `build_card`, `with_attribution`, `object_id`, the Parquet schema and every v0–v3 behaviour are unchanged; card.py still imports none of `scout.attrib`, `fullplan`, `gate`, `scan`, `hub`, `sources` (P3 C10), and imports `scout.caveats` only | T401 |
| `scout/hub.py` (P1 T005, P4 T302) | `HubSource(..., repo_type: str = "model")`; `"dataset"` uses `GET {endpoint}/api/datasets/{repo}/revision/{rev}` and file URLs `{endpoint}/datasets/{repo}/resolve/{sha}/{path}`; the same logging, preflight, redirects, retries and error mapping; `repo_type="model"` is byte-identical to P4 (no P1–P4 test changes) | T402 |
| `tests/helpers/fakehub.py` (P1 T003, P4 T302) | `add_repo(..., repo_type: str = "model")`; dataset routes mirror the model routes under `/api/datasets/` and `/datasets/`; default responses byte-identical | T402 |
| `scout/jobs/spec.py` (P4 T315) | `validate_spec` accepts `kind == "retrain"` by delegating to `scout.retrain.spec.validate_retrain_spec` (imported inside the function); for `kind == "attribute"` behaviour and messages are unchanged; `params_digest()` unchanged | T407 |
| `scout/jobs/worker.py` (P4 T316) | `main` dispatches a job spec with `kind == "retrain"` to `scout.retrain.job.run_retrain_job` (import inside); new subcommand `retrain-env`; `hello`, `sweep`, `audit` and attribute jobs unchanged | T407 |
| `scout/view.py` (P1 T011, P4 T318) | `depth_strips[].cells[].retrain` and `view.retrain` (§4.10) for v4 Cards; `None` for v0–v3; `DISCLAIMERS`, `ATTRIBUTION_CAVEAT` unchanged | T413 |
| `scout/server.py` (P1 T013, P2 T113, P4 T318) | `make_server(..., report_config: ReportConfig \| None = None)`; new `GET /api/report?repo=&revision_sha=&component=` (text/html, built on request, never stored); `serve(..., report_config=None)`; every other route unchanged | T413 |
| `scout/web/index.html`, `scout/web/app.js` (P1 T014, P2 T114, P4 T318) | a "report" link per stored subject; retrain outline and legend on the strip; P1–P4 ids kept | T413 |
| `scout/cli.py` (P1 T012, P2 T112, P3 T215, P4 T319) | new `scout retrain`, `scout report`, `scout report-lint`; `scout resolve --repo-type {model,dataset}`; `scout serve --model/--library-manifest/--pins/--ledger`; new exit code 11 ("report lint failed"); everything else unchanged | T414 |
| `scripts/pin_exit.py` (P1–P4) | `--phase 5`: targets `P5_REPOS`; datasets resolved with `scout resolve --repo-type dataset` and cross-checked with `git ls-remote https://huggingface.co/datasets/<repo> refs/heads/main`; phases 1–4 unchanged | T417 |
| `scripts/p3_ledger.py` (P3 T218, P4 T321) | none: P5 calls it with `kinds=P5_KINDS` | — |
| `pyproject.toml` | optional extra `retrain = ["torch>=2.4,<3", "transformers>=4.51,<6", "tokenizers>=0.20,<1"]`; `dev` gains the same three | T404 |
| `.gitignore` | += `cards/p5-store/`, `reports/` | T417 |

No P1–P4 test file is edited. No frozen P1–P4 parameter changes.

## 10. Task list

| id | title | route | depends_on |
|---|---|---|---|
| T401 | Card v4 (`stats.retrain`), `retrain.v1` validator, `with_retrain`; `scout/caveats.py` texts | opus | T306, T314, T318 |
| T402 | Dataset repos in `HubSource` and FakeHub | opus | T003, T005, T302 |
| T403 | `RETRAIN_PARAMS`, retrain plan (all subject tensors + attribution plan + data files + recipe digest), plan display, subject/data refusals | opus | T310, T401, T402, T404 |
| T404 | Retrain fixtures: tiny trained Llama family (base, derived, control), WordLevel tokenizer, Markov text dataset (test only); `retrain` extra | sonnet | T003, T212, T313, T314 |
| T405 | Model build from scratch tensors, data tokenisation, re-init, training step and eval metrics (torch) | opus | T305, T314, T403, T404 |
| T406 | Retrain loop: baseline check, re-fingerprint via `RetrainProvider`, stop rule, outcomes, `retrain.v1` | opus | T401, T405 |
| T407 | Worker retrain job (`run_retrain_job`), retrain spec, worker dispatch and `retrain-env`, abort on client death | opus | T315, T316, T402, T406 |
| T408 | Retrain client `retrain()`: cached scans, plan, gate, submit, Card v4 into the store, text output | opus | T309, T317, T407 |
| T409 | Report bundle `report.v1`: Card selection, analysis inputs, ledger citations, summary slots | opus | T212, T218, T309, T401 |
| T410 | Report renderer: deterministic self-contained HTML with `data` provenance elements | opus | T409 |
| T411 | Report linter L1–L10 (`lint.v1`) | opus | T410 |
| T412 | Blind-reader checklist: answer key, scoring, LLM reader over `S2Client`, human answer files | opus | T410 |
| T413 | View, server, frontend: retrain panel and report route | sonnet | T318, T401, T410 |
| T414 | CLI: `scout retrain`, `scout report`, `scout report-lint`, `resolve --repo-type`, `serve` report options | sonnet | T319, T408, T411, T412, T413 |
| T415 | Usage notes file and triage script | sonnet | — |
| T416 | P5 exit expectations: frozen targets and literals, C0–C17 as pure functions | opus | T320, T403, T406, T411, T412 |
| T417 | P5 exit runner (freeze, plan, run, reports, reader-form, read, exit), `pins_p5`, `pin_exit --phase 5`, store population | opus | T321, T408, T414, T416 |
| T418 | P5 rehearsal world (test only): P3/P4 stand-in artifacts and ledgers, the rt_family retrain target, a socket-guarded worker, reader stubs | opus | T208, T213, T214, T218, T313, T317, T321, T404, T412 |
| T419 | P5 offline rehearsal: the whole exit on the rehearsal world with real worker subprocesses, plus failure paths | opus | T417, T418 |

Parallel waves (each task in the earliest wave its dependencies allow; no shared files within a wave; checked by
`/tmp/p5fin/validate_tasks.py`, which also checks YAML parsing, required keys, dependency existence and acyclicity,
file disjointness across all tasks, and that this table, the waves, §6 and the ownership table match the task files):
1. {T401, T402, T404, T415}
2. {T403, T409}
3. {T405, T410}
4. {T406, T411, T412, T413}
5. {T407, T416, T418}
6. {T408}
7. {T414}
8. {T417}
9. {T419}

**File ownership.** Every file is in exactly one P5 task's `files` (checked by script). The P1–P4 files P5 edits:

| File | Owner |
|---|---|
| `scout/card.py` | T401 |
| `scout/hub.py`, `tests/helpers/fakehub.py` | T402 |
| `pyproject.toml` | T404 |
| `scout/jobs/spec.py`, `scout/jobs/worker.py` | T407 |
| `scout/view.py`, `scout/server.py`, `scout/web/index.html`, `scout/web/app.js` | T413 |
| `scout/cli.py` | T414 |
| `scripts/pin_exit.py`, `.gitignore` | T417 |

**Suites stay green.** Every task's acceptance runs the full `pytest -q` (P1–P5 including the P1–P4 rehearsals).

**Routing.**
- Opus: download gating or the byte path (T402, T403, T407, T408, T417), Card schema (T401, T409), similarity and the
  attribution rule inside the loop (T406), training numerics and invariant 1 in memory (T405), verdict/caveat wording
  and invariant 5 checks (T410, T411), the reader key and scoring that decide an exit row (T412), exit evidence (T416,
  T418, T419).
- Sonnet, each with every signature, constant and output string given: T404 (fixtures), T413 (view fields, one route,
  rendering over finished schemas), T414 (CLI plumbing over finished functions), T415 (a line parser and a counting
  rule).

## 11. Items deferred "to P5" by earlier phases, and their disposition

| Source | Item | Disposition |
|---|---|---|
| phases.md P5 | Retrain-plan panel as a loop | **Included** (§3.1–3.5, T403–T408, T413) |
| phases.md P5, P3 later, P4 later | Report export with standing caveats, the attribution table, null summary per reference, `ATTRIBUTION_CAVEAT` | **Included** (§3.6, T409–T411) |
| phases.md P5 | Optional black-box behavioural signal | **Deferred** (D90, §3.9, later.md) |
| phases.md P5 | UX polish from usage notes | **Included as mechanism** (usage notes and triage, T415); polish limited to the report view and retrain panel (D91) |
| P3 later (P5) | Analysis and System 2 panel in the web UI (`POST /api/analyze`) | **Partly**: read-only report view (`GET /api/report`, System 1 and attribution, no LLM call from the UI). System 2 from the UI is seeded as a usage note |
| P3 later (P5) | Direction heuristics with release dates | **Partly**: the report shows the model-card `base_model` claims as claims (P3 claims table). Release dates need a Hub field and a Card change: seeded as a usage note, later.md |
| P4 later (P5) | UI for attribution jobs (gate panel, live job panel) | **Re-deferred**: a second confirmation surface for invariant 2 that no exit needs and no usage note asks for yet; seeded as a note |
| P4 later (P5) | Retrain loop using the per-block attribution | **Included** (the attribution test is the loop's signal) |
| FUTURE_IMPROVEMENTS P4 | No partially re-initialised subject tested live (block localisation) | **Included**: the retrain exit re-initialises 3 of 24 blocks of a real model and requires the other 21 to stay attributed (C10, C11) |
| FUTURE_IMPROVEMENTS P4 | Heartbeat `BrokenPipe` does not abort COMPUTE | **Included for retrain jobs** (D96); the attribution runner keeps its stated 45-minute bound (re-deferred) |
| FUTURE_IMPROVEMENTS P1 | Subtree collapse/expand ("planned for P5") | **Re-deferred** to the usage-notes queue (seeded note); no P5 feature needs it |
| FUTURE_IMPROVEMENTS P3 | D36 human acknowledgement | Carried to the CHECKPOINT summary with D83, D89 and D99 (no P5 work) |
| FUTURE_IMPROVEMENTS (all other P1–P4 items) | Minor items tied to P1–P4 files | Not P5 scope; unchanged |

## 12. Out of scope for Phase 5
- Exporting retrained weights (D83), re-initialising attention or whole blocks, retraining quantised, MoE or pipeline
  subjects, multi-GPU or multi-job retraining.
- Data corpora above 32 MiB (a data byte class and gate), self-distillation data.
- A black-box behavioural signal (D90).
- Starting jobs, System 2 calls or retrains from the web UI.
- Native PDF, Markdown or DOCX reports; persisting reports inside scout.
- Any change to a frozen P2–P4 statistic, threshold or the JEV model.

See `later.md`.
