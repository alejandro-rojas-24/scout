# Phase 5 plan: Product refinement (retrain loop, report export, polish)

Status: r1 (planner REVISE round 1 of 2), 2026-10-02. Responds to `validation.md` r1 (1 blocker, 6 majors, 13 minors);
see §11. Ready for VALIDATE.

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
stops at the first point where the flagged blocks are no longer attributed, quality is within stated bounds, and the drop
persists at the next interval; the retrained weights are deleted like every other weight, only the subject's Card keeps the record, and the
panel surfaces the licence question without answering it. scout also exports a self-contained HTML report whose every
number carries a pointer to a Card, the frozen JEV model or a ledger entry, which a linter verifies mechanically, and
which two blind readers (an LLM and a person) must understand without any edit, while failing a planted-defect copy.

## 2. Exit check (runnable, falsifiable)

"You would hand a report to someone else without editing it" is made falsifiable by five things, all frozen in the
`P5_FREEZE` ledger entry before any real run:
1. a report linter with numeric checks (§3.7, C15);
2. a fixed set of three report targets (§2.1);
3. a blind-reader checklist with pass criteria, answered by an LLM reader (C16) and a human reader (C17), plus a
   negative-control canary report with three planted defects that each reader must fail (§3.8);
4. one full retrain-loop run on a small real model on the Spark with numeric criteria (C3–C14). The loop-integrity
   rows (C10, C11a) gate the exit. The empirical outcome rows (C11b signal drop, C12 quality) are reported, not gated.
   A non-success outcome closes the phase only as **`PASS with finding (<outcome>)`**, never as a plain `PASS` (D100,
   §2.4);
5. one-shot gates: every gate (retrain run, reports, each reader) is recorded once per freeze. A retry exists only
   through a recorded refreeze, and the exit then says `PASS after refreeze (<reason>)`, never a plain PASS (C18, §2.1).

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
- the sha256 of `exit/pins_p5.json`, of the JEV model file `models/jev_v1.json`, and the P3 and P4 ledger heads
  (`head_seq`, `head_line_sha256`: the evidence the reports cite). C2 recomputes all three at every later step (pins and
  model bytes, and the line at each frozen head; later appends to P3/P4 are allowed) (T416 `check_frozen`);
- the client `code_id` (P4 D78: tree hashes of `scout`, `scripts`, `pyproject.toml`). From `P5_FREEZE` to `P5_EXIT` the
  code is frozen: every later entry's `code_id` must equal the freeze's (C2).

`P5_FREEZE` must precede `P5_PLAN`, `P5_ATTEMPT` and `P5_REPORTS`. A second `P5_FREEZE` after a `P5_ATTEMPT`, a
`P5_REPORTS` or a `P5_READER` entry is refused unless `--refreeze-reason "<human reason>"` is given; the reason is
recorded, and C2 lists every refreeze.

**One-shot gates (validation r1 major 3; the P3 one-shot pattern, P3 D54).** Under one freeze:
- `run`: once. After a `P5_RETRAIN` that passed, or that FAILed with a result (a finding: `crash: false`), `run` exits 5
  and appends nothing. Only a crash (no `result` message: transport or backend error, worker death, `PlanMismatch`,
  `DiskSpaceError`; `crash: true`) may be rerun without a refreeze, at most `MAX_CRASH_RERUNS = 2` times.
- `reports`: once. `read --reader llm`: once per `P5_REPORTS`, with the pre-registered 2-of-3 samples taken in that one
  invocation (§3.8); an invocation in which every call failed before any response (no answer seen) appends nothing and
  exits 3. `read --reader human` (or `--waive-human`): once; the answers are hashed into the ledger before any score is
  shown.
- Any other retry needs `freeze --refreeze-reason`. The exit status is then **`PASS after refreeze`** with every reason
  and the superseded entries listed (T416 `exit_status`); a refreeze is never reported as a plain `PASS`. C18 audits the
  whole ledger for these rules (a violation means a hand-edited ledger or a runner bug). P2's `SIGMA_PARAMS`/`ALIGN_PARAMS`, P3's `CKA_PARAMS`/`TSTATS_PARAMS`/`FEATURE_PARAMS`/`JEV_PARAMS` and
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
# expect exit 0; one job, no prompt; gating rows C0..C11a, C13, C14 PASS; outcome rows C11b (signal drop with
# persistence) and C12 (quality) print PASS or FINDING (never FAIL; D100); the subject Card v4 (stats.retrain) is in the
# store; ledger gains P5_ATTEMPT (before the job) then P5_RETRAIN {passed, finding}. One-shot: a second `run` under this
# freeze exits 5 unless the first was a crash (§2.1)

# E5 reports (client, 0 network). Precondition: the human has written exit/p5_ack.json (D99 acknowledgement);
# without it `reports` exits 5 and appends nothing
python scripts/exit_check_p5.py reports --store cards/p5-store --out reports/p5
# expect exit 0; reports/p5/{retrain,te,negative}.html built by the `scout report` code path (build_bundle, render,
# lint); each lints clean (C15, L1-L10); each pins the ledger heads it cites (the p5 head is the entry before this
# P5_REPORTS), so the E6-E8 appends never change a rebuild; reports/p5/canary.html = the retrain report with the three
# frozen planted defects (§3.8; never linted); ledger gains P5_REPORTS {per report: sha256, lint rows, answer-key
# sha256; canary sha256 and defects; neutral file names}

# E6 LLM blind reader (api.anthropic.com; 12 calls = 4 files x 3 pre-registered samples)
python scripts/exit_check_p5.py read --reader llm
# expect exit 0; C16 PASS for the 3 reports (>= 2 of 3 samples pass each) and for the canary (>= 2 of 3 samples fail it
# with a planted defect detected); ledger gains P5_READER (reader llm) with every sample. One-shot (§2.1)

# E7 human blind reader (a person who has not seen the plan, the code or scout's output)
python scripts/exit_check_p5.py reader-form --form exit/reader_human.yaml     # questions only, no answers, no labels
#   copies the 4 files to reports/p5/reader/report-{A..D}.html in a frozen order that hides which one is the canary;
#   the reader opens them, fills exit/reader_human.yaml, and signs the blind attestation
python scripts/exit_check_p5.py read --reader human --answers exit/reader_human.yaml
# expect exit 0; the answers are appended (P5_READER, reader human) BEFORE any score is printed; C17 PASS for the 3
# reports and the canary. One submission only (§2.1)

# E8 exit (client, 0 network): re-verifies every row from the ledger, the store and the report files
python scripts/exit_check_p5.py exit --store cards/p5-store --out reports/p5
# expect exit 0; every gating row PASS; last line exactly "EXIT CHECK (P5): PASS" on a success outcome. Qualified
# forms (T416 exit_status/status_line): "PASS with finding (<outcome>)" when C11b or C12 is FINDING (D100);
# "PASS with waiver (<reason>)" under D89; "PASS after refreeze (<reason>; ...)" when any gate was retried through a
# refreeze; the qualifiers combine in that fixed order, e.g. "PASS after refreeze (<r>), with finding (<o>)".
# Ledger gains P5_EXIT {status, notes}; commit the ledger
```

- Subcommands are mutually exclusive; `run` requires `--confirm-plans`; `read` requires `--reader`; `read --reader
  human` requires `--answers`.
- `run` repeats the preflight and plan steps and FAILs ("plan changed since P5_PLAN") if the plan id differs.
- `reports`, `read` and `exit` refuse (exit 1) unless a `P5_RETRAIN` entry with `passed: true` (every gating row
  passed; a FINDING does not block) exists after the last `P5_FREEZE`; `read` also needs a `P5_REPORTS` entry whose report and canary sha256 values equal the files on disk.
- Exit codes: 0 pass; 1 a check FAILed or a precondition is missing; 3 a credential or the reader endpoint is missing
  or unreachable (nothing appended); 5 a one-shot or freeze rule refused the step (nothing appended).

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
| C6 | worker wire: Σ body bytes `==` worker ByteLog `meta + header + weight`; hosts ⊆ {`huggingface.co`, `*.hf.co`}; every ranged request lies within a registered header bound or one planned read; **dataset requests** (T416 `check_dataset_wire`): every request under `/datasets/` or `/api/datasets/` is one of exactly the listing `GET /api/datasets/<repo>/revision/<pin>`, `README.md` (licence) and the two planned data files at `<pin>` (redirects on `*.hf.co` attributed to their origin), each data file read once in full | true |
| C7 | disk and purge (P4 C7 a–e): scratch audit before 0 files; `disk_peak_bytes <= disk_bytes`; purge report ok with `bytes_written == bytes_planned` (scout wrote only the planned tensors); during COMPUTE `HOME`, `TMPDIR`, `XDG_CACHE_HOME`, `HF_HOME`, `TORCH_HOME`, `TRITON_CACHE_DIR` and `tempfile.tempdir` point into the job dir (T407), so any file torch, transformers or triton writes is inside the audited tree; `env_files`/`env_bytes` shown; COMPUTE `purge` event before the PURGE check event; audit after: 0 files, 0 bytes | all true |
| C8 | no manual intervention: 1 job submitted, `status == "succeeded"`, exit code 0, stdin never read | true |
| C9 | resources: wall `<= 7200 s`; COMPUTE `<= 5400 s`; worker `rss_peak_bytes <= 48 GiB` | true |
| C10 | **baseline** (the unmodified subject, computed in the job before any re-initialisation): base `status == "tested"`, `null_ok`, `control_ok`; base attributed fraction `>= 0.95`; control 0 leave-one-out hits; **every flagged block attributed to the base**; baseline perplexity in `[2.0, 60.0]` (forward-pass sanity) | true |
| C11a | **loop integrity** (deterministic on a correct build): `validate_retrain == []`; outcome not refused; **re-initialisation changed the forward pass** (`reinit.ppl_ratio > 1.0` or `reinit.top1_agreement < 1.0`); **training changed the weights** (`retrained_digest != reinit_digest`); a trace row at every multiple of `eval_every` up to the last row; every row's flagged z non-null, and the last row's flagged z differ from the baseline's (re-fingerprinted on the live tensors); **at every trace row the unflagged blocks' attributed fraction `>= 0.95`** (localisation; the model is still detected as derived) | true |
| C11b | **signal drop** (empirical outcome row: PASS or FINDING, D100), persistence rule: `outcome == "success"`; at the stop row **and** the confirm row (`confirm_step == stop_step + eval_every <= max_steps`): base `status == "tested"`, `null_ok`, `control_ok` (not an abstention); **0 of the 3 flagged blocks attributed to the base**; control 0 hits; `post_stop_max_z_adj` shown | true |
| C12 | **quality** (empirical outcome row: PASS or FINDING, D100) at the stop step (at the last row when there is no stop): `ppl_ratio = exp(eval_loss − eval_loss_baseline) <= 1.05`; top-1 agreement with the unmodified subject on the eval tokens `>= 0.90` | true |
| C13 | Card v4 and store: the subject Card written by the job has `schema_version == "card.v4"`; `validate_retrain(stats.retrain) == []`; `stats.attribution` is the baseline and validates as `attribution.v1`; `retrain.lineage.parent_object ==` the object id of the header-scan Card the plan used; `retrain.weights_kept is False`; the store holds it; recomputing its object id from the two files gives the stored id | true |
| C14 | caveats: the `scout retrain` text output and the view of the Card v4 carry the three `DISCLAIMERS`, `ATTRIBUTION_CAVEAT`, `RETRAIN_CAVEAT` and `LICENSE_QUESTION` | true |
| C15 | **report lint** (per report, §3.7): L1–L10 all at their thresholds; file sha256 `==` the `P5_REPORTS` entry; the sections required by §2.1 present and the others "not run" with a reason | all pass |
| C16 | **LLM blind reader** (per file, §3.8): 3 samples per file in one invocation, each with a served model and request host `api.anthropic.com` (an unavailable sample counts as failed); a report passes a sample when correct `>= 9` of 10, critical Q1, Q2, Q4, Q6 all correct and required edits `== 0`; each of the 3 reports passes in `>= 2` of 3 samples; the **canary** fails as required (not passed, and `>= 1` planted defect detected) in `>= 2` of 3 samples | all true |
| C17 | **human blind reader** (one submission over the 4 neutrally named files): blind attestation `true`; each report passes as above; the canary fails as required | all true |
| C18 | **one-shot protocol** (§2.1): over the whole ledger, no `P5_ATTEMPT` after a passing or non-crash failing `P5_RETRAIN` without a refreeze in between; `<= 2` crash reruns per freeze; one `P5_REPORTS` per freeze; one `P5_READER` per (`P5_REPORTS`, reader); `actual` lists every refreeze reason, crash rerun and waiver | true |

**E2** evaluates C0 and C2 (freeze only). **E3** evaluates C0–C3 and the plan half of C4. **E4** evaluates C0–C14.
**E5** evaluates C15; **E6** C16; **E7** C17; **E8** re-evaluates every row from the ledger, the store and the files, adds
C18, and prints the status. `FAIL` when any gating row fails; otherwise `PASS`, qualified by `after refreeze`,
`with finding`, `with waiver` in that order (8 strings, enumerated in §4.9), each qualifier followed in the printed line
by its reasons or outcome in parentheses. C11b and C12 are the only outcome rows; every other row is gating.

The `P5_RETRAIN` payload records: plan_id, bytes planned/cap, weight bytes, disk peak, wall/compute seconds, rss peak,
the `retrain-env` report, the baseline (ppl, per-block base z_adj and z_pair), the whole trace (§4.2), the outcome, the
stop and confirm steps, `post_stop_max_z_adj`, the final flagged-block statistics, `reinit_digest` and
`retrained_digest`, `env_files`/`env_bytes`, the Card object id, `crash`, `prior_attempts`, the backend string,
`git_commit`, `code_id`.

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
  criteria and is confirmed at the next eval (§3.1 step 5), so the budget is a ceiling, not a target. A candidate can
  be at most step 1750. The GPU time bound is §2.5.
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

- **Measured at the exit's probe count (r1, validation major 1; planner scratch `/tmp/p5r1`, `m2048_retrain.py`,
  `starved.py`, `p4stat.py`).** A third toy family uses V = 4096 word tokens, 8 blocks, hidden 128 and GLU intermediate
  2048, so **m = 2048 neurons tested with 2048 probes**, as at the exit. The control has intermediate 2304 and is trained
  on the same data. The P4 statistic is re-implemented with Bonferroni over 16 tests, giving crit 5.20 (df 7). The
  baseline z_adj of derived vs base is 42.46 in every block (ceiling √2047 = 45.2), and the control LOO maximum is 0.83.
  The recipe is the exit's: AdamW 1e-3, warmup 100, cosine to 10 % over 2000 steps, clip 1.0, CE, eval every 250, run
  to 3000 steps. Batch 32 × 64.

  | flagged | step | ppl ratio | top-1 | flagged z_adj | unflagged attributed (min z_adj) |
  |---|---|---|---|---|---|
  | {3} (1 of 8 = 12.5 %, the exit's fraction) | 0 (re-init) | 1.033 | 0.922 | 1.40 | 7/7 (42.48) |
  | | 250 | 1.008 | 0.949 | −1.52 | 7/7 |
  | | 500 | 1.006 | 0.936 | 1.25 | 7/7 |
  | | 1000 / 2000 / 3000 | 1.002 / 0.995 / 0.988 | 0.915 / 0.901 / 0.897 | −1.76 / −1.91 / −0.79 | 7/7 (≥ 38.29) |
  | {2, 5} (2 of 8) | 0 (re-init) | 1.146 | 0.838 | −1.02, −0.62 | 6/6 |
  | | 250 | 1.016 | 0.914 | 0.18, 1.05 | 6/6 |
  | | 500 | 1.009 | 0.900 | −0.88, 0.37 | 6/6 |
  | | 1000 / 2000 / 3000 | 1.005 / 0.992 / 0.984 | 0.867 / 0.852 / 0.845 | ≤ 0.70 | 6/6 (≥ 40.35) |

  **Token-starved variant** (the same family, quality only, batch 2 or 4 × 64, so the first eval sees 0.04 training
  tokens per trainable parameter; the exit's whole budget is 0.42 and its first eval 0.05):

  | flagged, batch | tokens/param at step 250 | step 0 | step 125 | step 250 | step 375 | later |
  |---|---|---|---|---|---|---|
  | {3}, 2 × 64 | 0.041 | 1.033 / 0.922 | 1.019 / 0.939 | 1.018 / 0.919 | 1.018 / 0.924 | 500: 1.015 / 0.936 |
  | {2, 5}, 4 × 64 | 0.041 | 1.146 / 0.838 | 1.065 / 0.859 | 1.043 / 0.900 | 1.040 / 0.901 | 500: 1.033 / 0.901; 625: 1.026 / 0.893 |

  (cells are ppl ratio / top-1.)

- **What the measurements say for the exit.**
  1. **The signal drop is stable at m = 2048 (C11b).** No flagged block was attributed at any of the 39 flagged-block
     evaluations from re-initialisation to step 3000. z_adj had mean −0.59, sd 0.89 and maximum 1.40, against crit
     5.20 here and 4.12 at the exit. The validator's concern was that the m = 256 drift (z_adj about 2 by step 1000),
     rescaled by √(2047/255), would give 4.7–5.7 at the exit. It does not happen: the drift seen at m = 256 was null
     noise (sd about 1 at any m), not a constant ρ, so it does not grow with √m. The persistence rule (§3.1 step 5)
     gives `success` with stop 250 and confirm 500 in both m = 2048 runs (post-stop max z_adj 1.25 and 0.37), and stop
     250 in the m = 256 CE run. A false FINDING from noise at the exit crit is ≤ 0.002 per job (6 block-looks at
     t.sf(4.12, 23) = 2.1e-4).
  2. **Unflagged blocks stay attributed (C11a)**: minimum z_adj 38.3 over every row, against crit 5.20.
  3. **Quality (C12) passes at the first candidate in 3 of 4 configurations.** It passes for 1 of 8 and 2 of 8 at
     m = 2048 and for 2 of 8 at m = 256. It fails for 4 of 8, which plateaus at top-1 0.85. Token starvation to the
     exit's ratio did not change the 1-of-8 result. It moved 2 of 8 from 0.914 to 0.900 at step 250, exactly at the
     bound, with a plateau near 0.90 until step 500 and a decline after it (0.893 at step 625).
  4. **Agreement peaks early, then declines** (1 of 8: 0.949 at step 250, 0.897 at step 3000). In-domain retraining
     moves the argmax away from the unmodified model while perplexity keeps falling below 1. The stop rule's earliest
     confirmed candidate is therefore also the best quality point. A longer budget would not help C12, so no second
     budget is pre-registered.
  5. **The difference that remains is the exit model itself.** Its MLP-removed damage for 3 of 24 Qwen blocks is
     unknown. With 3 blocks it can be larger than one toy block, and the toy damage ranged from top-1 0.92 to 0.58.
     Hidden 896 against 128 and real text against code tokens can each move it. **Planner's estimate of P(success) on a
     correct build: 0.5–0.8** (not measured at exit scale; no GPU or Hub here). This is why D100 makes C11b and C12
     outcome rows: a correct build closes with `PASS` (success) or `PASS with finding (<outcome>)`. The rows that gate
     are deterministic on a correct build except C10 (≤ 0.003, P4), C16 (≤ 0.08 for the three reports, ≤ 0.03 for the
     canary, §3.8) and C17 (no rate claimed). The **pass rate of the gating rows on a correct build is ≥ 0.88** (1 − 0.003 − 0.11), with
     the LLM reader the dominant term.
  6. The KD objective of the first two toys recovers agreement faster, but it is rejected for the product (§3.4: it
     pulls the new neurons towards the old function) and is not used.

- **Rehearsal fixture (T404; validation r1 major 6), measured on the frozen recipe** (`rtfam.py`, `rt_measure.py`; P4
  statistic as above; transformers 5.18, torch 2.14 CPU). The fixture is a tied Llama with 8 blocks, hidden 128 and GLU
  intermediate 128, trained 400 steps on order-1 Markov text whose top successor has probability 0.7. The control has
  intermediate 96. Eval ppl: base 3.44, derived 2.93, control 3.34.
  - Attribution, derived vs base with the control: 8/8 attributed, z_adj 10.6 each (ceiling √127 = 11.3), crit 5.20,
    control z −1.43…0.98, LOO max 0.80. Redrawn block 3: z_adj −0.31, not attributed, the other 7 attributed. The
    control as subject: 0 attributed.
  - Re-initialising `RT_FLAGGED = (1, 2, 5, 6)` gives ppl ratio 1.324 and top-1 0.937, so `quality_ok` is false at
    step 0 and recovery is exercised.
  - Rehearsal budget (eval every 50): step 50 has ppl ratio 1.096 (not a candidate). Step 100 has 1.048 / 0.996 and is
    the candidate (**stop 100**). Step 150 has 1.022 and is the confirmation (**confirm 150**). The maximum flagged
    z_adj over 400 steps is 1.96.
  - Rejected variants, each measured: Dirichlet successor probabilities and intermediate 512 (re-init top-1 0.96, then
    retraining *lowered* it to 0.83: quality_not_recovered); flagging (2, 5) only (re-init 1.075 / 0.991, recovered by
    the first eval, so recovery is never exercised).
  - The real T404/T406 code must reproduce these numbers within the stated tolerances; if not, the fixture is a plan
    revision (T404 notes), never a silent constant edit.

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
| E6 | `api.anthropic.com`, `ANTHROPIC_API_KEY`; 12 calls (4 files × 3 pre-registered samples) at up to about 20k input and 4k output tokens each at `claude-opus-5-5` ($4 / $20 per MTok) ≈ $2 worst case | not attempted here |

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
5. **Confirm and stop (persistence rule, r1).** A row is a *candidate* when (a) no flagged block is attributed to the
   base while the base is `tested`, `null_ok` and `control_ok` (an abstention never counts as a drop: `signal_dropped`),
   and (b) `ppl_ratio <= max_ppl_ratio` and `top1_agreement >= min_top1_agreement` (`quality_ok`). The loop keeps
   training one more interval and stops only when the **next** row also has `signal_dropped`: `stop_step` = the
   candidate, `confirm_step` = the next eval step, and the confirm row's maximum flagged `z_adj` is recorded as
   `post_stop_max_z_adj`. A drop seen at a single look is never a stop (validation r1 major 2). Candidates are considered
   up to `max_steps − eval_every`. If no candidate is confirmed by `max_steps`, the outcome names what failed from the
   last row (`signal_persists`, `quality_not_recovered`, `neither`, or `unconfirmed` when only the last row is a
   candidate). A margin (for example `z_adj <= crit/2 = 2.06`) was considered and not adopted. Under the null, a
   re-initialised block's `z_adj` behaves like a standard score: measured at m = 2048 over 39 flagged-block evaluations
   (§2.4), it had mean −0.59, sd 0.89 and maximum 1.40. A `crit/2` margin would be crossed by noise with
   probability about 0.02 per block-look, so about 0.11 per job over 6 looks (3 blocks × 2 rows), which would add
   spurious FINDINGs and carry no information. Persistence over two looks costs nothing at the familywise crit (≤ 0.002
   per job). The confirm row's maximum is reported as `post_stop_max_z_adj` instead.

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
6. COMPUTE (`try … finally purge`), with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` set in the process, and
   `HOME`, `TMPDIR`, `XDG_CACHE_HOME`, `HF_HOME`, `TORCH_HOME`, `TRITON_CACHE_DIR` and `tempfile.tempdir` pointed into
   `<job dir>/env/` (r1, C7 scope: anything a library writes lands in the audited, purged tree; counted as
   `env_files`/`env_bytes` before the purge):
   a. **Baseline fingerprint**: `attrib.attribute(plan_attrib, …, ScratchProvider)` → attribution.v1 of the unmodified
      subject.
   b. **Baseline refusal**: any flagged block whose base test is not `attributed`, or base not `tested`/`null_ok`, or
      `control_ok` false → outcome `refused_baseline` with the reason; no training (the result is still a valid
      retrain.v1 and the Card records it).
   c. Build the model (§3.5) from the scratch tensors; tokenise the data; baseline eval (loss, perplexity, and the
      argmax token at every eval position, kept in RAM); baseline perplexity outside `baseline_ppl_range` → outcome
      `refused_forward_check` (the forward pass does not reproduce the model; no training).
   d. Re-initialise the flagged blocks (§3.1 step 2); freeze everything else; evaluate the "MLP removed" state
      (`reinit`) and record `reinit_digest` (sha256 of the re-initialised flagged tensors).
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
`tokenizer=True` for base and controls (0 weight bytes); for the subject, the **parent** is the newest store entry of its
key that is not a `card.v4`, has a tokenizer and has no attribution (`parent_entry`; a fresh header scan only when none
exists) - so a second retrain, or a rerun after a failed one, never plans from or composes onto a `card.v4` (validation
r1 major 5), and the P3 exit's `card.v2` subject Card is a valid parent (`with_retrain` upgrades v0–v2 like
`with_attribution`); dataset resolve (0 weight bytes), plan, `plan_only` display, gate, spec, `run_job_via`,
then `card.with_retrain` on that parent Card with the job's baseline attribution as `stats.attribution`
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
7. `attribution`, titled "Block attribution of the published subject (unmodified)" with one scope sentence (so a
   reader never confuses its counts with the retrain trace's): the depth strip (SVG), one row per reference (role,
   status, `n_attributed/n_tested`, null summary, `crit`, and the reason of an abstaining or untestable reference in a
   `.reason` span, which L5b counts), one row per block (status, primary, `z_adj`, `p`, `z_pair`), `ATTRIBUTION_CAVEAT`. "Not run" when no Card
   of the subject has `stats.attribution`.
8. `retrain`: the recipe, the baseline, the trace table, the outcome and stop step, `RETRAIN_CAVEAT`. "Not run" when
   no Card v4 of the subject exists.
9. `license`: the licences claimed by the subject's and every reference's model card (and the dataset's, when a retrain
   section exists), then `LICENSE_QUESTION`. Always present.
10. `provenance`: every Card used (repo@sha, component, object id, schema version, and its ledger citation), the JEV
    model sha256 and its `FREEZE_MODEL` citation, every ledger read with its **pinned head** (file, `head_seq`,
    `head_line_sha256`), every citation (ledger, seq, sha256 of the line), `code_id`,
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

**Pinned ledger heads (validation r1 blocker).** A report must stay valid after its ledgers grow. `build_bundle` reads
each ledger only up to a **head**: by default the file's last entry at build time, recorded in the bundle as
`inputs.ledgers[] = {name, file_name, head_seq, head_line_sha256}`; every citation ("the latest entry that...") is chosen
within that prefix. The bundle records no entry count or other property of the live file. L10b rebuilds the bundle with
`ledger_heads` taken from the embedded bundle, re-verifies each chain up to its head and requires the line at the head
to hash to `head_line_sha256`; entries appended later are ignored. So the E5 reports (whose p5 head is the entry before
their own `P5_REPORTS`) still lint clean at E8 after `P5_REPORTS`, `P5_READER` and `P5_EXIT` were appended, and a
product report citing a live ledger stays valid; rewriting the ledger below the head is detected (L10b, L6). Owners: T409
(`read_entries(..., upto_seq)`, `ledger_heads`, `build_bundle(..., ledger_heads=)`), T411 (L10b, L6), tested by T409
`test_heads_pinned`, T411 `test_l10b_ledger_append_ignored`, and the T419 rehearsal (E8 after appends).

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
| L10 | unedited and reproducible | (a) `render(bundle) != file bytes`; (b) the bundle rebuilt from the store, model and ledgers (with the file's `generated_at` and **its pinned ledger heads**; each chain verified up to its head, the head line hash equal) differs from the embedded bundle | 0 each |

L10 is what makes "without editing" mechanical: any hand edit of the HTML breaks (a), and any edit of the bundle breaks
(b). The exit also pins each file's sha256 in `P5_REPORTS` (C15).

### 3.8 Blind-reader checklist (D88; frozen `READER_PARAMS`)

Ten questions per report, answered from the rendered report alone. The answer key is computed from the bundle by frozen
code (`answer_key(bundle)`), and its sha256 is written to `P5_REPORTS` **before** any reader answers. Every report gets
the same ten questions (r1: no variants).

**Design rule (validation r1 major 4).** Every **critical** question has a key that depends on the report's content and
cannot be given from general knowledge (T412 asserts on its fixture bundles that no critical key is the same for every
report). The two fixed-answer questions (Q8, Q9) are not critical; they exist because each is linked to a planted
defect of the canary below, where the fixed answer becomes wrong.

| # | question | answer format | key |
|---|---|---|---|
| Q1 (critical) | Which model is this report about (repository and revision)? | `repo@sha` (≥ 7 hex) | subject key |
| Q2 (critical) | Which references, if any, does System 1 call derived? | list of repos, `none`, or `not run` | S1 derived set |
| Q3 | On which references did System 1 abstain, and with which reason code? | list of `repo: reason code` (codes as printed), `none` or `not run` | S1 abstentions with the P3 code (`no weight evidence`, `out of training range`, `low confidence`), matched exactly (case and spacing aside) |
| Q4 (critical) | In the section "Block attribution of the published subject (unmodified)", how many blocks are attributed to each reference? | `repo: n of N` list, or `not run` | per reference (role reference) `n_attributed`/`n_tested` of that section; never the retrain trace's counts |
| Q5 | Which independent control model calibrated the block-attribution test? | repo or `not run` | control repos |
| Q6 (critical) | Which of these sections were not run: System 1, System 2, block attribution, retrain experiment? | list or `none` | sections with status `not_run` |
| Q7 | Which licence does the report list for the subject? | licence string or `unknown` | the subject's claimed licence as shown |
| Q8 | Does the report state whether using, modifying or redistributing the subject is permitted? | yes/no | no (canary D2 makes it yes) |
| Q9 | According to the report, can weight forensics detect that a model was trained by distillation from another model? | yes/no | no (canary D3 removes the statement) |
| Q10 | Which blocks were re-initialised and retrained, and was their attribution signal below the threshold at the stop step (or at the last eval step when there was no stop)? | block list; yes/no, or `not run` | recipe blocks; `signal_dropped` of that row (r1 minor: not "outcome == success", which is wrong for `quality_not_recovered`); `not run` without a retrain section |

Each reader also lists **required edits**: statements that are factually wrong, contradict another part of the report,
or would mislead a reader, each with a quote. Optional style edits are recorded but not scored.

**Pass (per report and reading):** correct `>= 9` of 10, all critical questions (Q1, Q2, Q4, Q6) correct, required edits
`== 0`. Scoring is deterministic (T412): repos compared case-sensitively after trimming, shas by prefix, lists as sets,
yes/no on the first word (`not stated` is wrong), section names case-insensitive with fixed aliases, reason codes and
licences exact after case and whitespace folding.

**Negative control: the canary (validation r1 major 4).** A reader that answers from priors or approves anything must
be caught. At E5 the runner writes `reports/p5/canary.html` = the clean `retrain` report with three frozen defects
(`reader.make_canary`, visible HTML only; never linted):
- **D1 contradicted number**: the base's attribution count in `#summary` is changed by one; the `#attribution` table keeps
  the true value (linked question Q4);
- **D2 licence answered**: the sentence "Using, modifying and redistributing this model is permitted under the licences
  listed above." is inserted before the licence question (linked Q8);
- **D3 distillation disclaimer removed** (linked Q9).
A reading of the canary is scored against the clean key. It **fails as required** when it does not pass AND at least one
defect is detected (the linked question answered wrongly, or a required edit quoting at least 20 characters of the
defect text). A reader that copies the clean answers and requests no edit passes the canary and therefore FAILs C16/C17.

**Readers.**
- **LLM reader (C16).** Through the P3 `S2Client` seam (`AnthropicS2Client`, default `claude-opus-5-5`, adaptive
  thinking, effort high, structured output with `READER_SCHEMA`), one fresh context per call. The input is the file's
  **visible text** (HTML to text, the bundle removed) and the questions; the system prompt forbids outside knowledge.
  **Pre-registered k-of-n**: one `read --reader llm` invocation makes 3 independent calls per file (12 in total); a
  report passes C16 when `>= 2` of its 3 samples pass, the canary when `>= 2` of its 3 samples fail as required. An
  unavailable sample counts as a failed sample. The invocation is one-shot (§2.1): no resampling. Offline,
  `StubS2Client` answers from fixtures.
- **Human reader (C17).** A person who has not seen the plan, the code or any scout output, recorded by name or
  initials with `blind: true` in the answers file. `reader-form` copies the 4 files under neutral names
  (`report-A.html`…`report-D.html`, in an order frozen by the key hashes, so the reader cannot tell which is the
  canary) and writes the questions without answers. One submission: the answers are appended to the ledger before any
  score is shown. Pass: each report passes and the canary fails as required. **Open decision D89**: if no such person is
  available, the default is that Phase 5 stays open; the human may instead record a logged decision
  (`--waive-human "<reason>"`) that marks C17 `WAIVED`, and the phase then closes as "PASS with waiver", listed as such.

**Expected false-fail and false-pass rates (stated, not measured: no API key here; §2.5).** Let p be the probability
that one LLM sample fails a correct report (a careful reader slipping on 2 of 10 questions, a critical one, or
requesting an edit), and c the probability that one sample passes the canary. With 2-of-3, a report fails with
probability `3p²(1−p) + p³` and the canary is missed with `3c²(1−c) + c³`. Planning assumption p, c ≤ 0.10 per sample
(the critical questions are lookups of one printed value each, and D2 contradicts the always-present licence question
in the same section): ≤ 0.028 per file, so **≤ 0.11 per E6** over the four files (≤ 0.08 for the three reports alone).
Single-sample scoring would give ≤ 0.34 per E6 under the same assumption, which is why k-of-n was chosen; nothing here is
fitted to an observed reading. For the human (one reading) no rate is claimed: a FAIL there goes to the human
(refreeze with a reason, shown as "PASS after refreeze", or the D89 waiver). The rehearsal shows both directions with
stubs (T419 `test_rehearsal_reader_fail`, `test_rehearsal_canary_ignored`).

### 3.9 Black-box behavioural signal: deferred (D90)

Not included. A behavioural signal (comparing generations of the subject and a candidate on a fixed prompt set to
detect distillation) needs generation on the GPU or through an API for every candidate, a prompt set, and above all a
labelled set of documented distillation pairs to calibrate against; none exists. An uncalibrated signal would sit
next to calibrated weight evidence and invite being read as a verdict. The report states the blind spot instead
(DISCLAIMERS 1; reader Q9 asks for it, and canary defect D3 removes it to check that a reader notices). later.md records the design constraints for a future phase:
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
- A Card v4 is the subject's header-scan Card (the **parent**: card.v0–v3 without `stats.attribution`, upgraded to v3
  with `upgrade_to_v3` as P4 `with_attribution` does; never a card.v4; r1, validation major 5) (key, source, scan,
  model_card, config, pipeline, weights, structure, Parquet) with `stats.attribution` = the job's baseline attribution (the unmodified subject), `stats.retrain` filled,
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
 reinit_digest: str|null,                        # "sha256:<hex>" over the re-initialised flagged tensors (before any step)
 trace: [{step: int, train_loss: float, eval_loss: float, ppl_ratio: float, top1_agreement: float,
          base_status: "tested"|"abstain"|"untestable", null_ok: bool, control_ok: bool,
          flagged: [{index, z: float|null, z_adj: float|null, z_pair: float|null, crit: float|null, attributed: bool}],
          unflagged_attributed: int, unflagged_tested: int, control_hits: int,
          signal_dropped: bool, quality_ok: bool, wall_s: float}],
 outcome: "success"|"signal_persists"|"quality_not_recovered"|"neither"|"unconfirmed"|"refused_baseline"|
          "refused_forward_check",
 stop_step: int|null, confirm_step: int|null,    # success: the candidate row and the next eval row (persistence)
 post_stop_max_z_adj: float|null,                # max flagged z_adj of the confirm row
 reason: str|null,
 final_attribution: attribution.v1|null,         # at confirm_step (success), or at the last eval step; null when refused
 retrained_digest: str|null,                     # "sha256:<hex>" over the final flagged tensors; null when refused
 weights_kept: false,
 lineage: {parent: card_key, parent_object: str, relation: "partial-mlp-retrain", statement: RETRAIN_CAVEAT},
 license: {subject: str|null, references: [{title, license: str|null}], data: str|null, question: LICENSE_QUESTION},
 caveat: RETRAIN_CAVEAT,
 job: {job_id, host, plan_id}|null, computed_at: ISO-8601 Z}
```
- `signal_dropped` = base `tested` and `null_ok` and `control_ok` and no flagged block `attributed`.
- `quality_ok` = `ppl_ratio <= max_ppl_ratio` and `top1_agreement >= min_top1_agreement`.
- `outcome` = `success` at the first row with both flags whose next row has `signal_dropped` (stop_step = that row,
  confirm_step = the next, which is the trace's last row); otherwise, after `max_steps`, from the last row:
  `signal_persists` (quality ok, signal not dropped), `quality_not_recovered` (dropped, quality not ok), `neither`, or
  `unconfirmed` (both flags at the last row, no confirmation possible).
- Floats are rounded to 6 decimals. `validate_retrain` (T401) checks every key and type, the enums, that the trace
  steps are strictly increasing multiples of `eval_every`, the persistence rule (`stop_step`, `confirm_step`,
  `post_stop_max_z_adj`) when the outcome is `success`, that no row satisfies it otherwise, and that `weights_kept is
  False`.

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
          ledgers: [{name, file_name, head_seq: int|null, head_line_sha256: str|null, error: str|null}]},
          # pinned heads (§3.6): every ledger is read only up to head_seq; nothing about the live file is recorded
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
                             reports: {"report-A.html".."report-D.html": {answers: {Q1: str, ..., Q10: str},
                                                                     required_edits: [{quote, reason}]}}}
                                                                     # neutral names; the canary is one of the four
score = {reader: "llm"|"human", label, correct: int, n: 10, critical_ok: bool, required_edits: int,
         per_question: {Q: bool}, passed: bool}
detection (canary) = {score, detected: ["D1"|"D2"|"D3"], failed_as_required: bool}
verdict (per file) = {n, n_pass, n_failed_as_required, ok}           # ok by the READER_PARAMS k-of-n rule
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
  "stop_rule": "first eval step with signal_dropped and quality_ok whose next eval step has signal_dropped"}
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
READER_PARAMS = {"version": "reader.v1", "n_questions": 10, "min_correct": 9, "critical": ["Q1", "Q2", "Q4", "Q6"],
  "max_required_edits": 0, "llm_model": "claude-opus-5-5", "llm_effort": "high", "llm_max_tokens": 8000,
  "llm_samples": 3, "llm_min_passing": 2, "human_submissions": 1,
  "edit_reasons": ["factually_wrong", "contradicts_report", "misleading", "unreadable"],
  "reason_codes": ["no weight evidence", "out of training range", "low confidence"],
  "canary": {"source_label": "retrain", "min_detected": 1, "quote_min_chars": 20,
             "defects": {"D1": {"question": "Q4", "edit": "summary n_attributed of the base -1 (+1 when 0)"},
                         "D2": {"question": "Q8", "edit": "insert licence_sentence before .license-question"},
                         "D3": {"question": "Q9", "edit": "delete the #caveats item equal to DISCLAIMERS[0]"}},
             "licence_sentence": "Using, modifying and redistributing this model is permitted under the licences listed above."},
  "questions": <the §3.8 table, verbatim, as {id: {text, format, critical, canary_defect: "D1"|"D2"|"D3"|null}}>}
```
The rehearsal budget changes only sizes and step counts; every threshold (`max_ppl_ratio`, `min_top1_agreement`,
`baseline_ppl_range`, the attribution rule) is the same for both budgets.

### 4.9 Ledger `exit/ledger_p5.jsonl` (P3 `p3ledger.v1` line format; kinds `P5_*`; T417)
```
P5_FREEZE    {params: {retrain, report, lint, reader}, params_sha256: {...}, retrain_target, report_targets,
              pins_sha256, model_sha256, cited_ledgers: {p3: {head_seq, head_line_sha256}, p4: {...}},
              refreeze_reason: str|null, git_commit, code_id, code_dirty}
P5_PREFLIGHT {hello, retrain_env, checks: {name: ok}, git_commit, code_id}
P5_PLAN      {plan: {plan_id, bytes_planned, bytes_cap, disk_bytes, data_bytes, recipe_digest}, git_commit, code_id}
P5_ATTEMPT   {plan_id, bytes_cap, confirm_plans, prior_attempts: int, git_commit, code_id}
P5_RETRAIN   {attempt_seq, passed (every gating row ok), finding: str|null (the outcome when C11b or C12 is FINDING),
              crash: bool (no result message received), failed: [check], error: str|null,
              evidence: {...§2.3 payload}, card_object: str|null, git_commit, code_id}
P5_REPORTS   {reports: {label: {file_name, sha256, bytes, lint_ok, lint_rows: [{id, count, ok}], answer_key_sha256,
              bundle_sha256, ledger_heads: {name: head_seq}}},
              canary: {file_name, sha256, source_label, defects: [{id, question, text_sha256}]},
              ack_sha256: str (D99 acknowledgement file, a precondition of E5),
              neutral: {label_or_canary: "report-A.html"...}, passed, git_commit, code_id}
P5_READER    {reader: "llm"|"human", reports_seq: int (the P5_REPORTS entry read), reader_id: str, blind: bool|null,
              waived: str|null, answers_sha256: str,
              files: {label_or_canary: [{answers, required_edits, score | detection, s2: {requested_model,
                      served_model, request_host, usage}|null}]},      # 3 samples (llm) or 1 (human) per file
              verdicts: {label_or_canary: {n, n_pass, n_failed_as_required, ok}}, passed, git_commit, code_id}
P5_EXIT      {passed, status: "PASS"|"PASS with finding"|"PASS with waiver"|"PASS with finding, with waiver"|
                "PASS after refreeze"|"PASS after refreeze, with finding"|"PASS after refreeze, with waiver"|
                "PASS after refreeze, with finding, with waiver"|"FAIL", finding: str|null,
              notes: [str], failed: [check], rows: [{target, check, expected, actual, ok}], waivers: [str],
              refreezes: [{seq, reason, superseded: [seq]}], crash_reruns: [seq], prior_attempts: int, git_commit,
              code_id, git_dirty}
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
| D99 | Report files and invariant 1 | A report is a user-requested export, like stdout. scout never stores a report or uses one as an input: `report-lint`, `read` and `exit` read a report file back only to verify it. `reports/` is gitignored, and the exit keeps each file's sha256 in the ledger. **Open for human acknowledgement**, like P3 D36. The acknowledgement is a precondition of E5: `reports` exits 5 until `exit/p5_ack.json` records it (T417) | It is fully recomputable from Cards, the frozen model and ledgers (L10b proves it) |
| D100 | What the exit requires of the retrain outcome (validation r1 major 1) | **Report the outcome honestly.** C10 and C11a (baseline, loop integrity, reinit and training changed the weights, live re-fingerprinting, unflagged blocks still attributed) and every report and reader row gate the exit. C11b (signal drop) and C12 (quality) are outcome rows: PASS or FINDING. A non-success outcome closes the phase as `PASS with finding (<outcome>)`, listed like a waiver. A rerun after a finding needs a refreeze and shows `PASS after refreeze (<reason>)`. **Alternative**: gate on `success`. The planner's pass probability for that is 0.5–0.8 (§2.4), and the only route out of a FAIL is a post-hoc budget change | The phase asks for a loop that *confirms whether* the signal drops. Whether it drops on one model under one frozen budget is a fact about that model, not about the tool. The tool's ability to show a drop is proven deterministically in the rehearsal (T406, T419). Gating on an outcome with an unmeasured pass rate invites exactly the refreeze-until-pass the plan forbids |

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
| A reader that answers from priors (no reading) | fails the canary row (C16/C17) | T412 `test_canary_detection`; T419 `test_rehearsal_canary_ignored` |
| A drop seen at one eval only (transient) | not a stop; the next eval must confirm | T401 `test_validate_persistence`; T406 `test_transient_drop_not_a_stop` |
| Second retrain of the same subject; P3 card.v2 subject Card | parent = newest non-v4 header-scan Card; v0–v2 upgraded | T401 `test_with_retrain_parents`; T408 `test_retrain_twice`, `test_parent_from_p3_card` |
| Ledger grows after a report was rendered | heads pinned in the bundle; L10b reads the prefix | T409 `test_heads_pinned`; T411 `test_l10b_ledger_append_ignored`; T419 rehearsal E8 |
| Retry of a gate after a FAIL (rerun, resample, resubmit) | refused (exit 5) without a refreeze; refreeze shown as "PASS after refreeze" | T417 `test_run_one_shot`, `test_read_one_shot`; T416 `test_c18_one_shot`, `test_exit_status`; T419 `test_rehearsal_refreeze_status` |
| A library writes caches or files during COMPUTE | HOME/TMPDIR/HF_HOME/TORCH_HOME/TRITON_CACHE_DIR/XDG_CACHE_HOME inside the job dir; audited and purged | T407 `test_cache_dirs_in_job_dir` |

## 8. Risks and how each is tested

| Risk | Mitigation / test |
|---|---|
| Retraining recreates the base's neurons, so the signal comes back (exit C11b) | Measured at the exit's probe count m = 2048 (§2.4): max flagged z_adj 1.40 over 3000 steps and 3 flagged-block runs (crit 5.20 toy, 4.12 exit); the √m extrapolation of the m = 256 toy does not hold. Under the null, the false-FINDING rate from noise is ≤ 0.002 per job (6 block-looks at t.sf(4.12, 23) ≈ 2e-4). The loop reports `signal_persists` honestly, as `PASS with finding` (D100), and nothing is re-seeded or re-thresholded. T406 `test_signal_persists`; T419 `test_rehearsal_finding` |
| Quality does not recover within the budget (exit C12): **the main empirical risk** | Zero-init `down` (step 0 = MLP removed). Both quality measures are reported per eval step and right after re-initialisation. Measured (§2.4): at m = 2048 the 1-of-8 toy (the exit's 12.5 %) passes at the first eval, also when token-starved to the exit's 0.04 tokens per trainable parameter; 2 of 8 reaches 0.914 at step 250 (0.900 when starved); 4 of 8 plateaus at 0.85; agreement *declines* after its peak, so the earliest confirmed candidate is the best stop. The planner's estimate of P(success) on a correct build is 0.5–0.8. A non-success outcome is `PASS with finding` (D100), not a FAIL; the bounds and the budget stay frozen (C2) |
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
| GPU nondeterminism makes a rerun differ | Stated. Only crash reruns are allowed under one freeze (≤ 2, C18); any other rerun needs a refreeze with a reason and shows as `PASS after refreeze (<reason>)` (T416 `test_exit_status`, T419 `test_rehearsal_refreeze_status`) |
| A gate is retried until it passes (LLM resampling, human resubmission, run repetition) | One-shot gates (§2.1, validation r1 major 3): k-of-n pre-registered in `READER_PARAMS`, one human submission hashed before scores are shown, `run` refused after a completed retrain; C18 audits the ledger (T416 `test_c18_one_shot`, T417 `test_run_one_shot`, `test_read_one_shot`) |
| The readers cannot fail (constant keys, lenient LLM) | Content-dependent critical keys (T412 asserts they vary across fixture reports); the canary with three planted defects must FAIL in ≥ 2 of 3 LLM samples and for the human (C16, C17); stated rates in §3.8 |

## 9. Changes to earlier contracts

| Earlier contract | Change | Task |
|---|---|---|
| `scout/card.py` (P1 T009, P2 T102, P3 T202, P4 T306) | `SUPPORTED_SCHEMA_VERSIONS` += `"card.v4"`; new `RETRAIN_VERSION = "retrain.v1"`, `validate_retrain(r) -> list[str]`, `with_retrain(card, table, attribution, retrain, extra_log, out_dir) -> BuiltCard` (writes `card.v4` from a header-scan parent card.v0–v3, upgraded like `with_attribution`); `load_card` accepts v4. `SCHEMA_VERSION`, `build_card`, `with_attribution`, `object_id`, the Parquet schema and every v0–v3 behaviour are unchanged; card.py still imports none of `scout.attrib`, `fullplan`, `gate`, `scan`, `hub`, `sources` (P3 C10), and imports `scout.caveats` only | T401 |
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
| T404 | Retrain fixtures: tiny trained tied-embedding Llama family (base, derived, control, redrawn), WordLevel tokenizer, low-entropy Markov text dataset (test only), measured against P4 attribution and re-init damage; `retrain` extra | opus | T003, T212, T313, T314, T317 |
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
`/tmp/p5r1/validate_tasks.py`, which also checks YAML parsing, required keys, dependency existence and acyclicity,
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
- Opus since r1: T404 (the rehearsal fixture is a measured design that the loop and rehearsal evidence depends on;
  validation r1 major 6).
- Sonnet, each with every signature, constant and output string given: T413 (view fields, one route, rendering over
  finished schemas), T414 (CLI plumbing over finished functions), T415 (a line parser and a counting rule).

## 11. Validation responses — round 1

Answers to `validation.md` r1. Every fix below is checked by `/tmp/p5r1/validate_tasks.py`, which parses the YAML and
checks deps, waves, file disjointness, plan/task consistency and the r1 items. Measurements are in planner scratch
`/tmp/p5r1` and are not in the repo.

| Finding | Resolution | Where |
|---|---|---|
| **Blocker**: an embedded ledger summary goes stale after later appends | **Ledger heads are pinned.** The bundle records `inputs.ledgers[] = {name, file_name, head_seq, head_line_sha256}`, with no entry count. Every citation is chosen within that prefix. L10b rebuilds with the embedded heads, verifies each chain up to its head and checks the head line hash, and ignores later appends. The E5 reports pin the p5 head just before their own `P5_REPORTS`, so E6–E8 appends never change a rebuild. | §3.6 "Pinned ledger heads", §3.7 L10, §4.5; T409 `read_entries(upto_seq)`, `ledger_heads`, `test_heads_pinned`; T411 `test_l10b_ledger_append_ignored`; T417 behaviors 6 and 9; T419 `test_rehearsal_pass` (lint clean at E8) |
| **Major 1**: C11 drift and C12 unmeasured at exit scale; refreeze shown as a plain PASS | (a) Measured at the exit's m = 2048 (§2.4). Over 39 flagged-block evaluations to step 3000, no flagged block was attributed: max z_adj 1.40, mean −0.59, sd 0.89, crit 5.20 (4.12 at the exit). The predicted √m growth (4.7–5.7) does not occur, so the C11b noise false-FINDING rate is ≤ 0.002 per job. (b) C12 measured near exit scale (1 of 8 = 12.5 %, m = 2048, starved to 0.04 tokens/param): it passes at the first eval (top-1 0.94). 2 of 8 starved is at the bound (0.900). Agreement declines after its peak, so no fallback budget is pre-registered. Planner P(success) = 0.5–0.8; gating-row pass rate on a correct build ≥ 0.88. (c) **Decision D100: report the outcome honestly.** C11b and C12 are outcome rows (PASS/FINDING). A non-success outcome closes as `PASS with finding (<outcome>)`, and a refreeze shows `PASS after refreeze (<reason>)`, never a plain PASS. Justification: the phase asks for a loop that confirms *whether* the signal drops. Gating on an outcome with an unmeasured pass rate leaves only the forbidden post-hoc budget change as a way out. The loop's ability to show a drop is proven deterministically offline. | §2 items 4–5, §2.2 E4/E8, §2.3 C11a/C11b/C12, §2.4, §4.9, D100, §8; T416 `OUTCOME_ROW_PREFIXES`, `row_verdict`, `exit_status` (8 statuses + printed line), `test_outcome_rows`; T417 behaviors 5 and 9, `test_run_finding`; T418 `finding-worker`; T419 `test_rehearsal_finding`, `test_rehearsal_refreeze_status` (exact line `PASS after refreeze (reader stub fixed)`) |
| **Major 2**: the stop rule had no persistence | **Persistence:** a candidate row (signal dropped and quality ok) is a stop only when the next eval row also has `signal_dropped`. `stop_step`, `confirm_step` and `post_stop_max_z_adj` are recorded; candidates are considered up to `max_steps − eval_every`; `unconfirmed` is the outcome when only the last row qualifies. A `crit/2` margin was rejected with numbers: noise crosses it with probability 0.02 per look, about 0.11 per job. Measured result: stop 250 / confirm 500 in both m = 2048 runs (post-stop max z_adj 1.25 and 0.37), stop 250 at m = 256, stop 100 / confirm 150 on the rehearsal fixture. | §3.1 step 5, §4.2, §4.8 `stop_rule`; T401 `test_validate_persistence`; T406 behavior 5, `test_transient_drop_not_a_stop`, `test_unconfirmed`; T416 C11b `test_c11b_persistence` |
| **Major 3**: gates could be retried until they pass | **One-shot gates (P3 pattern).** `run` runs once per freeze; only crash reruns are allowed, ≤ 2. `reports` runs once. `read --reader llm` runs once with pre-registered 2-of-3 samples in one invocation. `read --reader human` takes one submission, appended to the ledger before any score is printed. Every other retry needs `freeze --refreeze-reason` and is shown as `PASS after refreeze (<reason>)`. Refused steps exit 5 and append nothing. C18 audits the whole ledger. | §2.1 "One-shot gates", §2.3 C16–C18, §4.9; T416 `check_one_shot`, `test_c18_one_shot`; T417 behaviors 5, 6 and 8, `test_run_one_shot`, `test_read_one_shot`; T419 `test_rehearsal_reader_fail` (second read → 5), `test_rehearsal_failed_run_needs_refreeze` |
| **Major 4**: readers shown unable to fail | **Content-dependent critical questions** Q1, Q2, Q4 and Q6 (subject@sha, the S1 derived set, attribution counts per reference, the sections not run). T412 asserts that no critical key is constant across the fixture reports. The fixed-answer Q8 and Q9 are non-critical and are linked to canary defects. Q3 matches the printed reason codes exactly. Q4 is scoped to the section titled "Block attribution of the published subject (unmodified)". Q10 is keyed on the row's `signal_dropped`. **Negative control:** a canary copy of the retrain report with three frozen defects (D1 a contradicted count, D2 a sentence answering the licence question, D3 the distillation disclaimer removed) must FAIL in ≥ 2 of 3 LLM samples and for the human; neutral file names hide it. **Rates:** with p, c ≤ 0.10 per sample, 2-of-3 gives ≤ 0.028 per file, ≤ 0.08 false-fail for the three reports and ≤ 0.03 canary miss (single-sample scoring: ≤ 0.34). | §3.8, §4.7, §4.8 `READER_PARAMS`; T412 `make_canary`, `canary_detection`, `llm_verdict`, `neutral_names`; T416 C16/C17 `test_c16_k_of_n`; T418 stubs (`canary_mode`); T419 `test_rehearsal_canary_ignored` |
| **Major 5**: a second retrain planned from a card.v4 | **The parent is chosen by role, not recency.** `parent_entry(store, key)` returns the newest entry of the key that is not `card.v4`, has a tokenizer and has no attribution; a fresh header scan is made only when none exists. `with_retrain` refuses a v4 or an attribution Card and upgrades v0–v2 (the P3 `card.v2` subject Card is a valid parent). | §3.3 Client, §4.1; T401 `test_with_retrain_parents`; T408 `parent_entry`, `test_retrain_twice`, `test_parent_from_p3_card`, `test_parent_entry_rules` |
| **Major 6**: rehearsal fixture unmeasured; T404 on sonnet | **T404 re-routed to opus.** The fixture was measured before freezing RT_* (§2.4 "Rehearsal fixture"): derived vs base 8/8 at z_adj 10.6 (crit 5.20), control 0, redrawn block 3 at z_adj −0.31 not attributed. Re-init of (1, 2, 5, 6) gives ppl ratio 1.324 and top-1 0.937, so recovery is exercised. Stop 100, confirm 150. The order-1 Markov text now uses peaked successors (0.7/0.1/0.1/0.1); the Dirichlet variant failed quality and the 2-block variant never exercised recovery. T404 acceptance runs the real P4 `attribute` (`test_family_attribution`) and `test_reinit_damages_quality`. T406 `test_loop_success` requires reinit quality not ok and a first trace row not ok. A miss is a plan revision, never a silent constant edit. | §2.4, §10 routing; T404; T406 |

**Cheap minors fixed in r1:**
- L5b: abstaining or untestable reference rows render `.reason` (T410 behavior 9, T411 `test_l5b_reference_reason`).
- C2 compares the recorded pins sha, model sha and the P3/P4 line at each frozen head at every later step (T416
  `check_frozen`, `test_c2_freeze_inputs`).
- C7: HOME, TMPDIR, XDG_CACHE_HOME, HF_HOME, TORCH_HOME, TRITON_CACHE_DIR and `tempfile.tempdir` point into the job dir,
  so they are audited and purged (T407 `test_cache_dirs_in_job_dir`).
- C6 dataset row: exactly the API listing, `README.md` and the two data files (T416 `check_dataset_wire`).
- C11a proves the weights changed: `reinit.ppl_ratio > 1` or `reinit.top1_agreement < 1`, and
  `retrained_digest != reinit_digest` (T406 records `reinit_digest`; T416 `test_c11a_integrity`).

**Also fixed:**
- The trace table renders `base_status`, `null_ok`, `control_ok` and `signal_dropped` per row (T410).
- Q10 is keyed on `signal_dropped`, not on `outcome == success` (T412).
- D99 wording is corrected, and its acknowledgement (`exit/p5_ack.json`) is a precondition of E5 (T417).

**Deferred minors** (one line each; carried to `later.md` at CHECKPOINT):
- Data bytes vs the cumulative 64 MiB meta threshold: fail-safe (`ReadThresholdExceeded`); state the budget against
  the job's cumulative log or lower `data_max_bytes`.
- `license_deny` is lexical ("allows", "permits", number words pass); low risk, since only templates and model-card
  strings reach the text.
- Report fixtures live in T409's test module; move them to `tests/helpers/report_fixtures.py` when T410–T413 need
  them across modules.
- The `P4_EXIT targets.<label>.object` key name is assumed; pin it against P4 T321 at implementation, or match any
  64-hex value.
- The System 2 report section (L2c) and `GET /api/report` are covered offline only, with no exit row; keep them or
  move them to later.md at CHECKPOINT.

## 12. Items deferred "to P5" by earlier phases, and their disposition

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

## 13. Out of scope for Phase 5
- Exporting retrained weights (D83), re-initialising attention or whole blocks, retraining quantised, MoE or pipeline
  subjects, multi-GPU or multi-job retraining.
- Data corpora above 32 MiB (a data byte class and gate), self-distillation data.
- A black-box behavioural signal (D90).
- Starting jobs, System 2 calls or retrains from the web UI.
- Native PDF, Markdown or DOCX reports; persisting reports inside scout.
- Any change to a frozen P2–P4 statistic, threshold or the JEV model.

See `later.md`.
