# Phase 4 plan: Infrastructure + block attribution

Status: DRAFT r0 (planner), 2026-09-29. Not yet validated.

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
neuron-matching test, with a p-value and a family-wise error bound.

## 2. Exit check (runnable, falsifiable)

### 2.1 Targets, references, pins (fixed)

The exit has exactly two attribution targets. Each has one documented base and one independent control reference.
They are frozen in `scripts/exit_expectations_p4.py::P4_TARGETS` (T320):

| label | subject (target, selector) | base (derived-from, documented) | control (independent) |
|---|---|---|---|
| `70b` | `bartowski/DeepSeek-R1-Distill-Llama-70B-GGUF` `#DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf` (**V**: file name, single file) | `meta-llama/Llama-3.3-70B-Instruct` (gated). DeepSeek-R1 paper, arXiv 2501.12948: the 70B distilled model was fine-tuned from Llama-3.3-70B-Instruct. The GGUF is a llama.cpp Q4_K_M quantisation of `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` (the model card text states this, **V**) | `Qwen/Qwen2.5-72B`. Qwen2.5 technical report, arXiv 2412.15115, pretrained from scratch; same hidden size 8192 and depth 80 as the subject, and a different intermediate size (29568 vs 28672) |
| `te` | `Qwen/Qwen-Image` `#text_encoder` (diffusers; `Qwen2_5_VLForConditionalGeneration`) | `Qwen/Qwen2.5-VL-7B-Instruct`. Qwen-Image technical report, arXiv 2508.02324: Qwen2.5-VL is the frozen condition encoder (**V**: quote) | `meta-llama/Llama-3.1-8B` (gated). Llama 3 herd paper, arXiv 2407.21783, pretrained from scratch |

- **Pins.** `exit/pins_p4.json` holds the 6 repos (subjects, bases, controls), committed as `"UNPINNED"`. It is written
  by `scripts/pin_exit.py --phase 4`, which cross-checks `scout resolve` against `git ls-remote` and passes HF_TOKEN to
  git through `GIT_CONFIG_*` env (P2 D29).
- **C0 target set.** The run FAILs before any request unless `set(pins) == P4_REPOS`, the 6 repos of the table.
  There is no fallback repo or file. Replacing a subject, a selector, a base or a control is a logged human decision
  followed by a plan revision, made before the `P4_PLAN` ledger entry (§2.2 E3).
- **Fixed endpoint.** The endpoint is always `https://huggingface.co` on both the client and the worker. A foreign
  `HF_ENDPOINT` makes the run exit 1.
- **Evidence line.** Every E-step prints one JSON line. PIN, PREFLIGHT, PLAN and EXIT must share the client
  `hostname`. The worker hostname is recorded separately.
- **Script-written ledger.** `exit/ledger_p4.jsonl` uses the P3 hash-chained format (`scripts/p3_ledger.py`, extended
  with a `kinds` argument by T321), with the kinds `P4_PREFLIGHT`, `P4_PLAN`, `P4_ATTEMPT` and `P4_EXIT`. Only
  `scripts/exit_check_p4.py` appends to it. `P4_ATTEMPT` is appended **before** the first job is submitted, and
  `P4_EXIT` afterwards, also when the run raises. Every rerun is therefore visible. Reruns are allowed, because the
  statistic is deterministic at fixed pins and code (§2.5), but the EXIT evidence lists the number of earlier attempts
  with the same plan ids.
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
# expect exit 0; last line {"step":"PIN","phase":4,"hostname":...,"repos":{6 entries}}

# E2 Spark preflight (0 weight bytes; ssh to the Spark)
python scripts/exit_check_p4.py preflight --pins exit/pins_p4.json --backend "ssh:$SCOUT_SPARK_HOST"
# expect exit 0; rows C0, C1, C2 and C7a PASS; ledger gains P4_PREFLIGHT; the JSON line has the worker's hello
# (machine, mem_total_bytes, disk_free_bytes, scratch_root, git_commit, python, numpy/scipy versions, has_hf_token)

# E3 plan only (0 weight bytes): header scans into the store, the two full plans displayed in full
python scripts/exit_check_p4.py plan --pins exit/pins_p4.json --backend "ssh:$SCOUT_SPARK_HOST" --store cards/p4-store
# expect exit 0; 2 plans (format_full_plan: plan_id, bytes planned, hard cap, disk on <host>:<scratch_root> with
# free space, memory peak, reason, per-model and per-file tables); "CONFIRM WITH: --confirm-plans <id70b>:<cap>,<idte>:<cap>";
# "TOTAL CAP: <N> B"; client totals.weight == 0; ledger gains P4_PLAN {plans, git_commit}.
# At plan time (published configs; the exact values come from the E3 plans):
#   70b  bytes_planned 179000967168  cap 179700189696   (slack = bytes_planned/256)
#   te   bytes_planned  25956065280  cap  26224500736   (slack = 4 x 64 MiB)
# -> commit exit/ledger_p4.jsonl

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

C3–C15 are evaluated per target (`70b`, `te`).

| # | Check | Threshold |
|---|---|---|
| C0 | target set: `set(pins) == P4_REPOS`, all pins 40-hex | equal, else FAIL and exit 1 before any request |
| C1 | Spark environment, from the worker hello: `machine == "aarch64"`, `system == "Linux"`, `mem_total_bytes >= 100 GiB`, `disk_free_bytes >= max(disk_bytes) + JOB_PARAMS.disk_reserve_bytes`; worker `git_commit ==` client `git_commit`, both clean on `CODE_PATHS` | all true (a FAIL means an assumption of §5 does not hold: human decision) |
| C2 | frozen params: `ATTRIB_PARAMS`, `QUANT_PARAMS`, `GGUF_PARAMS`, `JOB_PARAMS`, `STORE_PARAMS` equal the literals of `exit_expectations_p4.py`, on the client **and** in the worker hello (`params_digest`); P2/P3 params equal their frozen literals; code frozen since `P4_PLAN` (E4 only) | equal |
| C3 | plan: models `== [subject, base, control]`; for each model the tested stack depth `==` the config's layer count (`num_hidden_layers`, or `text_config.num_hidden_layers`); every block planned; `bytes_planned ==` Σ `nbytes` over the planned Parquet rows (gate, up and norm of every block, the whole embedding, FP8 scales), recomputed independently by the check from the stored Cards; `bytes_cap == bytes_planned + max(16 MiB, 4 × min(chunk, largest read), ceil(bytes_planned / 256))`; `disk_bytes == bytes_planned` | equal |
| C4 | gate: the client's plan-time snapshot has `totals.weight == 0`; in the run, exactly 1 approved client decision via `cli-flag` with `bytes_confirmed == bytes_cap`, whose `(plan_id, bytes_cap)` is in `--confirm-plans`, and exactly 1 worker decision via `job-spec` with the same pair; worker `plan_id ==` client `plan_id` | true |
| C5 | bytes and streamed log: worker `bytes_planned <= totals.weight <= bytes_cap`; the ingested event count `==` the worker's `n_events`, and the worker seqs of the ingested events are consecutive from the worker log's first event; client totals of origin `<worker host>` `==` the worker's result totals | true |
| C6 | worker wire (`wire_audit`): Σ body bytes of the worker's CountingTransport `==` worker ByteLog `meta + header + weight`; hosts ⊆ {`huggingface.co`, `*.hf.co`}; every request to a `.safetensors`/`.gguf` path lies within the registered header bound or inside one planned read | true |
| C7 | disk and purge: (a) scratch audit before the job: 0 non-lock files; (b) worker `disk_peak_bytes <= disk_bytes`; (c) purge report `dir_absent`, `resident_after == 0`, `bytes_written == bytes_planned` and `bytes_removed >= bytes_written` (the removed bytes include `index.json`); (d) the live log has the COMPUTE `purge` event before the PURGE check event `PURGE check: disk_resident=0 files=0 scratch=absent`; (e) a scratch audit after the job, run over the backend: **0 non-lock files and 0 bytes** under `scratch_root` | all true |
| C8 | no manual intervention: exactly 1 job submitted per target in this run, `status == "succeeded"`, worker exit code 0; the runner's stdin is never read | true |
| C9 | resources: job wall-clock `<= max_wall_s` (70b 21600 s, te 3600 s); COMPUTE stage `<= max_compute_s` (70b 7200 s, te 1200 s); worker `rss_peak_bytes <= 48 GiB` | true |
| C10 | **positive attribution** (base): reference `status == "tested"`, `null_ok == true`; `n_tested ==` depth; **attributed fraction `>= 0.95`**; the share of blocks whose `primary` is the base `>= 0.95`; median `z_adj >= 10.0` | true |
| C11 | **negative control**: reference `status == "tested"`, `null_ok == true`, `n_tested ==` depth, **`n_attributed == 0`** | true |
| C12 | Card v3 and store: the attributed subject Card has `schema_version == "card.v3"`; `stats.attribution` validates as `attribution.v1` with `len(blocks) ==` depth; the store has it under `(repo, sha, component)` with `has.attribution`; recomputing the object id from the two files gives the stored id; `build_view` gives the tested stack's strip `depth` cells, each with `attribution.status` set, and `primary_title` equal to the base title for attributed cells | true |
| C13 | cache: a second `attribute()` for the same subject and references returns the same object id, submits 0 jobs, and its client CountingTransport records **0 requests** | true |
| C14 | 70b GGUF: subject Card `weights.format == "gguf"`, `quant.method == "gguf"`, `"Q4_K" in quant.types`; the subject scan's header bytes for the GGUF file `== gguf.infos_end <= gguf.data_offset` and its `totals.weight == 0`; tested stack prefix `model.layers`; gate/up shapes equal the base Card's for every block | true |
| C15 | te: subject `key.component == "text_encoder"`; the tested stack is the text stack (its depth equals `text_config.num_hidden_layers`, or `num_hidden_layers`); the vision stack is listed in `attribution.other_stacks` with the reason `no probe basis`; probe tokenizer file `tokenizer/tokenizer.json` | true |
| C16 | disclaimers: the `scout attribute` text output and the view carry the three `DISCLAIMERS` plus `ATTRIBUTION_CAVEAT` | true |

**E2** (`preflight`) evaluates C0, C1, C2 and C7a. **E3** (`plan`) adds C3 and the plan-time half of C4. **E4** (`run`)
evaluates everything.

The `P4_EXIT` payload records, per target:
- plan_id, bytes_planned, bytes_cap, weight bytes, disk_peak, wall_s, compute_s and rss_peak
- `n_tested` and `n_attributed` per reference
- the null summary per reference (`n`, `mean`, `sd`, `max`, `mu0`, `s0`)
- the median and minimum `z_adj` for the base, and the maximum `z_adj` for the control
- the object id

It also records the worker hello, the backend string, `git_commit`, `git_dirty` and `prior_attempts`, whether or not a
check uses them.

### 2.4 Why these numbers (frozen in T320; exit numbers set by planner under the 2026-09-28 delegation)

- **α = 0.01 family-wise, Bonferroni over all (block, reference) tests of a job.** At a valid null, the chance of one or
  more false block attributions in a job is at most 1 %. With 80 blocks and 2 references, `n_tests = 160` and
  `z_crit = 3.836`. For the te job (28 blocks), `n_tests = 56` and `z_crit = 3.570`.
- **Attributed fraction ≥ 0.95 and median z_adj ≥ 10 for the base (C10).**
  - *Synthetic measurement* (planner, numpy scratch under /tmp/p4, `m = 256` test neurons, 1024 probes). A derived
    block with neuron permutation and noise equal to the weight standard deviation (ε = 1.0) gives z = 14.7–15.3 and
    96 % gate/up agreement. At ε = 2.0, the median z is 3.0.
  - *Scaling to the exit.* z scales with `√(m−1)`, and m = 2048 at the exit, so the same blocks would give z ≈ 42 and
    z ≈ 8.5. The ceiling is `√2047 = 45.2`, reached by an identical block.
  - *What the targets imply.* R1-Distill is SFT on about 800k samples, and the Qwen-Image text encoder is expected to
    equal Qwen2.5-VL-7B. Both sit far below ε = 1. The 0.95 allows up to 4 of 80 (or 1 of 28) blocks to fail for a
    reason nobody anticipated; that failure is reported, not hidden. The median bound of 10 is 2.6 × `z_crit`, so the
    base must be attributed by a wide margin, not just above the line.
- **Control n_attributed == 0 and null_ok (C11).** The control shares width (70b) or architecture family (te) with the
  subject, but not training. At α = 0.01, a correct test attributes nothing. `null_ok` must hold, so the control is
  *tested*: a control that abstains because its in-run null is inflated FAILs C11. That would mean independent models
  share neuron structure the test cannot separate, and it goes to the human.
- **Resources (C9).** These are bounds, not targets.
  - *Wall clock.* 179 GB at ≥ 10 MB/s is ≤ 5 h, plus ≤ 1 h of compute (§2.5), for a 6 h cap on the 70b job. The te job
    reads 26 GB, for a 1 h cap.
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
  four), and scout installed from a git checkout at the same commit as the client.
- **V** sshd is reachable from the client host, with key auth.
- **V** Outbound HTTPS to `huggingface.co`, `*.hf.co` (`cdn-lfs*.hf.co`, `cas-bridge.xethub.hf.co`).
- **V** `HF_TOKEN`, or `~/.cache/huggingface/token`, is present on the Spark.
- **V** Internet throughput is unknown. The C9 bound assumes ≥ 10 MB/s sustained.

| Step | Needs | This container |
|---|---|---|
| E0, all implementation | PyPI: numpy, scipy (new, T314), pyarrow, httpx, pyyaml, pytest, anthropic | available (scipy 1.17.1 installed in a scratch venv) |
| E1, E3 (client) | `huggingface.co` API + git; HF_TOKEN; about 60 MB meta+header (six tokenizers, 30 + 37 shard headers, one GGUF header of about 8 MB) | blocked (proxy 403 on huggingface.co) |
| E2–E4 (worker) | ssh to the Spark; Spark egress to the Hub; HF_TOKEN on the Spark | no Spark from this container |
| E4 bytes | about 205 GB of weight ranges (179.0 GB + 26.0 GB), the same on the Spark's disk, peak 179 GB at a time | — |

**Byte estimate at plan time (published configs; the E3 plans are the exact values).**

| job | model | tensors planned | bytes |
|---|---|---|---|
| 70b | subject GGUF Q4_K_M | gate+up Q4_K 80 × 2 × 28672×8192 (144 B per 256 weights) = 21,139,292,160; `token_embd` Q4_K 591,003,648 (**V** type); `ffn_norm` F32 2,621,440 | 21,732,917,248 |
| 70b | base Llama-3.3-70B-Instruct BF16 | gate+up 75,161,927,680; embed 2,101,346,304; norms 1,310,720 | 77,264,584,704 |
| 70b | control Qwen2.5-72B BF16 | gate+up 77,510,737,920; embed 2,491,416,576; norms 1,310,720 | 80,003,465,216 |
| 70b | **total** / slack (bytes_planned/256) / cap | | **179,000,967,168** / 699,222,528 / 179,700,189,696 |
| te | subject Qwen-Image text_encoder (**V** BF16) | text stack only: gate+up 28 × 2 × 18944×3584 × 2 = 7,604,273,152; embed 1,089,994,752; norms 200,704 | 8,694,468,608 |
| te | base Qwen2.5-VL-7B-Instruct BF16 | same shapes | 8,694,468,608 |
| te | control Llama-3.1-8B BF16 | gate+up 7,516,192,768; embed 1,050,673,152; norms 262,144 | 8,567,128,064 |
| te | **total** / slack (4 × 64 MiB) / cap | | **25,956,065,280** / 268,435,456 / 26,224,500,736 |

**Compute estimate (70b; measured on this 4-core x86 container, scaled conservatively).**
- *Neuron responses.* A reference block's response matrix `X·Wᵀ` (2048 × 8192 × 28672, float32) takes 2.4 s. There
  are 2 per block, per reference and per probe set, so 320 in total: about 13 min.
- *Correlations.* The correlation products (2048 × 2048 × 29568) number 960 at about 1 s: about 16 min.
- *Matching.* A rectangular linear assignment of 2048 × 28672 takes 0.4–0.5 s (scipy 1.17.1). There are 960: about 8 min.
- *Decoding.* Decoding and reading 179 GB from NVMe: about 5 min.
- *Total and memory.* COMPUTE is about 45 min. The memory estimate is 10.5 GiB (§3.2).

The te job is about 6 % of this.

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
  <SCOUT_SPARK_CMD or "scout jobs worker">`. It writes the job spec (`job.v1`, §4.4) on stdin and reads `jobproto.v1`
  messages (§4.5), one JSON object per stdout line. Remote stderr lines are passed to the client display, prefixed with
  the host.
- `LocalBackend` runs the same worker as a local subprocess (`sys.executable -m scout jobs worker`). It is used on the
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

Throughout, the worker sends a heartbeat every 30 s from a thread. Every message is written with flush. If stdout
breaks (`BrokenPipeError`: the client or ssh died), the worker aborts. The `finally` block of COMPUTE, or of
FULL_DOWNLOAD, purges on every exception path.

**Scratch, disk accounting and purge (T311).**
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
    documented cron line for the Spark (`*/15 * * * * scout jobs sweep --json >> ~/.cache/scout/sweep.log`).
- **Client side.** On disconnect, the client marks the job failed and immediately runs `backend.sweep()` (for ssh:
  `ssh host scout jobs sweep --json`), then prints its report. A still-running worker keeps its lock and is not
  touched; it aborts on its next write to the broken pipe.
- `scout jobs audit` lists every non-lock file and byte under the root, read-only. C7a and C7e use it.

### 3.4 Logs streamed back into the ByteLog (T301, T315)

- Each `event` message carries a worker `LogEvent.to_dict()`. The client calls `ByteLog.ingest(event, origin=<worker
  hostname>)`, which appends a `LogEvent` with:
  - a new client `seq`
  - the worker's stage name, event, path, url, range, status, bytes, `bytes_by_class`, attempt, elapsed and note
  - `origin` set (a new optional field; `to_dict` emits it only when set)
  - the updated client totals
- Ingest never calls `stage()`, so the client's stage order is unaffected.
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
- **Attribution cache.** `find_attribution(subject_key, reference_keys)` returns an entry with `has.attribution`,
  `attribution_refs == sorted(reference keys)` and the current `params_digest`.
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
6. **In-run null (addition, measured necessity).**
   - *The problem.* The paper's null assumes that independent models share no neuron structure. The planner simulated
     independent models whose neurons read a shared rank-16 semantic subspace, with within-neuron gate/up correlation
     γ:

     | setting | per-block null z |
     |---|---|
     | s = 0.3, a = 0.3, γ = 0.3 | mean 0.08, sd 1.10, max 2.74 |
     | s = 1.0, a = 2.0, γ = 0.9 | **mean 2.33**, sd 1.03, **max 4.78** |

     In the second setting the paper's parametric p-value alone would attribute independent blocks.
   - *The shift null.* For each tested position t in the ordered tested list T (n = |T| ≥ `min_blocks` = 8) and each
     k in `shifts = [1, 2]`, the subject block T[t] is tested against the reference block a(T[(t+k) mod n]). Different
     blocks of the same model have unrelated neurons, so these n·2 z-values measure the run's own null. In the inflated
     simulation they gave mean 2.36 and max 4.71, tracking the true null. In derived runs their max was ≤ 2.5 in every
     quantisation setting.
   - *Measured at the offline fixture scale* (T313 defaults: 16 blocks, hidden 256, 512 neurons, 1088 tokens, 1024
     probes, m = 512, z_crit(32 tests) = 3.421, 6 independent pairs each):

     | null setting | shift-null mean | raw parametric z | z_adj |
     |---|---|---|---|
     | inflated | 3.05–3.33 | attributes 6–8 blocks in **every** pair | attributes 0 (max 2.16) |
     | default | −0.06–0.32 | — | attributes 0 (max 2.39) |

     Derived pairs: z_adj ≥ 18.8 in all 16 blocks at ε ≤ 1.0 (ceiling √511 = 22.6), and 16 of 16 (min 6.8) at ε = 2.0.
     At half size (hidden 128, 256 neurons, m = 256, 10 pairs) the inflated raw test attributed in 9 of 10 pairs and
     z_adj in none.
   - *The adjustment.* `μ0 = max(0, mean)` and `s0 = max(1, sd(ddof = 1))`, then `z_adj = (z − μ0)/s0` and
     `p = ½·erfc(z_adj/√2)`. The adjustment can only make the test more conservative than the paper's. `null_ok` is
     `(max_null − μ0)/s0 < z_crit`, which means no mismatched pair would itself have been attributed. If `null_ok` is
     false, that reference is `abstain` ("in-run null exceeds the critical value") and attributes nothing.
7. **Multiple comparisons.** `n_tests` counts every (block, reference) test computed in the job, including those of a
   reference that later abstains. `z_crit = NormalDist().inv_cdf(1 − α/n_tests)`, with α = 0.01. A test is attributed
   iff `null_ok` and `z_adj ≥ z_crit`, which is Bonferroni: family-wise error ≤ α.

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
- Equal-depth pairs (both exit targets) get the identity.
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
 alpha: 0.01, n_tests: int, z_crit: float,
 references: [{card_key, title, content_digest: str|null, stack_prefix: str|null, depth: int|null,
               status: "tested"|"abstain"|"untestable", reason: str|null,
               alignment: {mode, kind, structure_informative}|null,
               probe: {n_common: int, n_probe: int, probe_sha256: str}|null,
               null: {n, mean, sd, max, mu0, s0}|null, null_ok: bool|null,
               n_tested: int, n_attributed: int, median_z_adj: float|null}],
 blocks: [{index: int, position: int, status: "attributed"|"not_attributed"|"untestable", reason: str|null,
           primary: int|null,
           tests: [{reference: int, reference_block: int|null, m: int|null, rho: float|null, z: float|null,
                    z_adj: float|null, p: float|null, agree_frac: float|null, attributed: bool, reason: str|null}]}],
 caveat: ATTRIBUTION_CAVEAT,
 job: {job_id, host, plan_id}|null, computed_at: ISO-8601 Z}
```
All floats are rounded to 6 decimals, and p is stored as a float (it underflows to 0.0 near z ≈ 38).

`ATTRIBUTION_CAVEAT` = "Block attribution tests whether a block's MLP neurons match a reference block's up to
permutation. 'Not attributed' is not evidence of independence: the base may be missing from the references, or the
block may have been retrained. Weights cannot show which model came first."

### 4.3 FullPlan (T310; served and printed, never persisted, except via GateDecision and JobRecord)
```
{plan_id: str (16 hex = sha256(json ["attrib.v1", [[repo, sha, component, [[path, start, end]...]] per model]])[:16]),
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
 references: [{target, repo, revision_sha, component}],
 plan: {plan_id, bytes_planned, bytes_cap, disk_bytes},
 confirmation: {plan_id, approved: true, via, bytes_confirmed: int|null, decided_at},
 endpoint: "https://huggingface.co", scratch_root: str|null, wire_audit: bool}
```
`target` is `repo@<40-hex>[#selector]` for Hub models, or an absolute local path that must exist on the worker host
(the offline rehearsal).

### 4.5 Job protocol `jobproto.v1` (T315; one JSON object per stdout line)
```
{"proto":"jobproto.v1","type":"hello","job_id"|null,"hostname","pid","machine","system","python","numpy","scipy",
 "scout_version","git_commit","git_dirty","params_digest","mem_total_bytes","disk_free_bytes","scratch_root","has_hf_token"}
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
              attribution_refs: [str] ("repo@sha/component" sorted) | [],
              params_digest: str, content_digest: str|null, created_at: ISO-8601 Z, scout_version}
```

### 4.7 ByteLog additions (T301)
- `LogEvent.origin: str | None = None`.
- `event` values gain `"job"`: 0-byte notes for job lifecycle (submitted, hello, finished, sweep).
- The new methods are listed in T301's interface.

### 4.8 View additions (T318)
```
depth_strips[].cells[].attribution: null | {status, primary_title: str|null, z_adj: float|null, p: float|null}
view.attribution: null | {stack_prefix, references: [{title, status, n_tested, n_attributed}], caveat: ATTRIBUTION_CAVEAT}
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
  "z": "rho*sqrt(m-1)", "null_adjust": "(z-max(0,mean))/max(1,sd)", "alpha": 0.01,
  "correction": "bonferroni-all-tests", "sided": "upper", "window": 0}
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
P4_PREFLIGHT {hello, checks: {name: ok}, git_commit, git_dirty}
P4_PLAN      {plans: {label: {plan_id, bytes_planned, bytes_cap, disk_bytes}}, git_commit, code_dirty}
P4_ATTEMPT   {plans: {label: {plan_id, bytes_cap}}, confirm_plans: {plan_id: cap}, git_commit, code_dirty, prior_attempts: int}
P4_EXIT      {attempt_seq, passed, failed: [check], error: str|null, targets: {label: {...§2.3 payload}}, hello, backend,
              git_commit, git_dirty}
```

## 5. Assumptions and open decisions (each with a recommended default)

Assumptions:
- **A5.** P1–P3 are implemented exactly as specified in T001–T218.
- **A6.** The DGX Spark facts in §2.5 (each marked V). C1 verifies the checkable ones at E2.
- **A7.** The §3.8 reading of Zhu et al. (V). The exit thresholds do not depend on the paper's details. They depend on
  the statistic as specified here, which is frozen.
- **A8.** The exit repos keep the layouts assumed in §2.1 (V items). A mismatch found at E1/E3 is a plan revision
  before `P4_PLAN`.

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
| D68 | Attribution statistic | Zhu et al. gate/up Spearman of rectangular LAP matchings on embedding-probe responses; in-run shift null; Bonferroni α = 0.01 | §3.8. Both deviations are measured and stated. |
| D69 | Block semantics | Attributed / not attributed (not evidence of independence) / untestable; MLP only; structure-only alignment, window 0; MoE blocks untestable | Honest about what the test sees. Windowed search and MoE experts are later.md. |
| D70 | Text encoders | Component selector; the tested stack is the GLU stack whose width equals the embedding; P2 tokenizer resolution; T5/CLIP untestable | §3.9. |
| D71 | Exit targets | §2.1: 70b = the R1-Distill-Llama-70B Q4_K_M GGUF (base Llama-3.3-70B-Instruct, control Qwen2.5-72B); te = the Qwen-Image text_encoder (base Qwen2.5-VL-7B-Instruct, control Llama-3.1-8B) | The GGUF subject exercises quantised input, full downloads and the 70B scale in one job, at 179 GB instead of 234 GB. Alternative: `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` (BF16) as the subject, +55 GB and no quantised path in the exit. |
| D72 | Spark assumptions | §2.5 (V) | Checked by C1 at E2. |
| D73 | Exit ledger | P3 line format and chain, kinds P4_*; `p3_ledger` gains a `kinds` argument | Reruns are visible, and the code freeze is anchored at `P4_PLAN`. |
| D74 | GPU | Not used in P4 | Compute is about 45 min on CPU (§2.5). CUDA wheels for aarch64 would add an environment dependency for no exit need (later.md). |
| D75 | New dependency | `scipy>=1.11` for `linear_sum_assignment` | A numpy Jonker–Volgenant would be about 150 lines of untested-in-the-wild code. scipy has aarch64 wheels. |

## 6. Architecture (files)

```
scout/bytelog.py, scout/errors.py                               T301  (P1/P2 files, extended)
scout/sources.py, scout/hub.py (lfs_sha256)                     T302
tests/helpers/gguf_fixtures.py                                  T303  GGUF writer + ggml block encoders (test only)
scout/dequant.py                                                T305  GGML_TYPES, decoders, FP8, QUANT_PARAMS, detect_quant
scout/gguf.py                                                   T304  header parser, adapter, config/vocab
scout/card.py (card.v3), P2 C8 / P3 C3 edits                    T306
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
| Independent reference | 0 attributed at α | T314 `test_null_family`, T321 |
| Partially re-initialised subject | Exactly the re-initialised blocks are not attributed | T314 `test_partial_reinit` |
| Neuron-permuted / hidden-rotated / norm-folded derivative | Attributed (invariance) | T314 `test_invariances` |
| Inflated shared-structure null | Adjusted away (z_adj); a shifted-copy subject makes the reference abstain (`null_ok` false) | T314 `test_inflated_null_adjusted`, `test_shifted_copy_abstains` |

## 8. Risks and how each is tested

| Risk | Mitigation / test |
|---|---|
| A weight byte is read without confirmation on the worker | Worker re-plans and requires the exact pair; grant in FULL_DOWNLOAD only; `Source.read_range` preflight. T316 `test_spec_cap_widened_refused` (a spec with a larger cap or a changed plan_id is refused with 0 weight bytes), T301 grant-stage tests, C4/C6 |
| GGUF header read touches weight bytes | Provable lower bound; T304 `test_safe_lookahead_never_past_infos` (spy on every requested range: all `< infos_end`) over fixtures with 1..3 KV arrays and 2..200 tensors; C14 live |
| Weights left on disk (invariant 1) | Purge in `finally`; PURGE check; sweeper; no mmap. T311 `test_write_read_purge`, `test_sweep_orphan_after_kill`, `test_sweep_active_skipped`; T316 `test_purge_on_compute_error`, T321 rehearsal crash test, C7 (live audit over the backend) |
| Crash between download and purge | flock-based orphan detection; sweep at worker start; cron line documented. T311 `test_sweep_orphan_after_kill` (subprocess killed with SIGKILL) |
| Remote log incomplete or bytes unlogged | Ingest completeness (seqs `0..n−1`, count equals `n_events`) and totals equal; worker wire audit. T315 `test_ingest_complete`, C5, C6 |
| The matching test's null is invalid on real independent models (shared neuron structure) | In-run shift null with conservative adjustment and `null_ok`. T314 `test_inflated_null_adjusted`: with s = 1, a = 2, γ = 0.9 (6 pairs), raw z attributes in ≥ 4 of 6 pairs (measured 6 of 6, 6–8 blocks each) and z_adj in ≤ 1 block in total (measured 0). T314 `test_shifted_copy_abstains`: a subject whose block t copies reference block t+1 gets `null_ok` false, so the reference abstains with 0 attributed. C11 requires the control tested and 0 attributed. A real FAIL goes to the human. |
| Embedding-probe deviation loses power on real models | Measured power on fixtures (ε up to 1.0); C10 needs median z_adj ≥ 10; a FAIL goes to the human with z per block in the evidence. Forward-pass activations are later.md. |
| Wrong neuron roles for an architecture | Roles only from `ATTRIB_PARAMS.roles`; missing → untestable with a reason; C3 (every block planned), C14/C15 |
| Dequantisation bugs | Byte-exact decode of hand-packed blocks for every type (T303 encoders, T305 tests); encode→decode error bounds; GGUF subject vs its own BF16 source in the rehearsal: z at the ceiling |
| Quantisation shifts P2/P3 statistics | Measured (§3.7); T308 `test_quantised_sigma_r` (fixture Q4_K copy: related median r ≥ 0.95); JEV caveat documented |
| Store corruption under concurrent writers | flock + atomic replace; T309 `test_concurrent_put` (8 processes put different Cards: all refs present, JSON valid) |
| Stale cache after a code change | `params_digest` in every entry; T309 `test_params_digest_invalidates` |
| P3 ledger hashes break | Byte-identical import/export; T309 `test_export_import_bytes` |
| Client/worker version skew | `params_digest` and `git_commit` in the hello and spec; `JobError` on mismatch (T316); C1/C2 |
| P3 C10 import rule broken (`scout.analysis` must not reach `scout.gate`/`scan`/`hub`/`sources`) | `scout.card` and `scout.view` never import `scout.attrib` or `scout.fullplan`; `ATTRIBUTION_CAVEAT` is duplicated by value in `view.py` and tested equal; `scout.cka` imports only `scout.dequant`. Tests: T306 `test_card_imports`, T308 `test_cka_imports`, T314 `test_no_forbidden_imports`, T318 `test_caveat_equal`, and P3 C10 itself in the P3 rehearsal (E0) |
| P1–P3 regressions (card.v3) | T306 changes P2 C8 and P3 C3 in the same change; T308 `test_p3_plan_unchanged`; every task runs the full `pytest -q` |
| Hub throughput too low for C9 | Bound documented (≥ 10 MB/s); the evidence records per-stage times; a FAIL goes to the human, and the bound is never raised after the fact |
| Memory on the unified 128 GB | Estimate in the plan display (≈ 10.5 GiB); `rss_peak_bytes` in the result; C9 ≤ 48 GiB |

## 9. Changes to earlier contracts

| Earlier contract | Change | Task |
|---|---|---|
| `scout/bytelog.py` (P1 T002, P2 T101) | `classify_range`: `.gguf` handled like `.safetensors` with a default header bound of 24 bytes (`GGUF_PROLOGUE`); `set_header_len` may be called repeatedly for `.gguf` with non-decreasing values (a decreasing value raises `ValueError`; safetensors unchanged); `grant(..., stage: Stage = Stage.SAMPLED_READ)`, and weight preflight allowed only in the grant's stage; `stage()` revokes the grant when leaving that stage; `LogEvent.origin` (optional; `to_dict` emits it only when set); `ingest()`, `totals_by_origin()`, `disk_add/disk_release/disk_resident/disk_peak`, `add_job_record/job_records`; `view(prefix)` returning a `LogView` that prefixes every path a Source logs (one job log holds several repos whose file names collide); `NOTE_EVENTS += {"job"}`. `to_card_dict()` is **unchanged** (card.py composes the v3 fields). Without a GGUF path, a FULL_DOWNLOAD grant or ingest, behaviour is byte-identical to P2; no P1/P2 test file is edited | T301 |
| `scout/errors.py` | new `JobError` (exit 7), `PlanMismatch(JobError)`, `BackendError(JobError)`, `PurgeError` (exit 8), `DiskSpaceError` (exit 9), `StoreError` (exit 1), `QuantError` (exit 1) | T301 |
| `scout/sources.py` `RepoFile` (P1 T004) | new field `lfs_sha256: str \| None = None` (last, defaulted; P1 constructors unchanged) | T302 |
| `tests/helpers/fakehub.py` (P1 T003) | `add_repo(..., lfs_sha256: bool = False)`; default responses byte-identical | T302 |
| `scout/hub.py` (P1 T005) | `resolve()` fills `lfs_sha256` from `siblings[].lfs.sha256` when present | T302 |
| `scout/card.py` (P1 T009, P2 T102, P3 T202) | `card.v3` (§4.1); `component_slug` maps characters outside `[A-Za-z0-9._-]` to `_` (P1 component names are unaffected; GGUF components `gguf:<path>`); `SUPPORTED_SCHEMA_VERSIONS` += v3; `PARQUET_SCHEMA` += `source_name`; `build_card` kwargs `weights_format`, `quant`, `content_digest`, `gguf_files`, `source_names`; new `with_attribution()`, `object_id()`; the P1–P3 card tests are updated to v3 | T306 |
| `scripts/exit_expectations_p2.py` C8 (P2 T115, P3 T202) | accepts `card.v1`, `card.v2` or `card.v3`; not a threshold | T306 |
| `scripts/exit_expectations_p3.py` C3 (P3 T216) | `card.v2` becomes `card.v2` or `card.v3`, every other C3 condition unchanged; not a threshold | T306 |
| `scout/scan.py` (P1 T010, P2 T108, P3 T207) | selector (`split_selector`), GGUF intake (P1 D10 extended), GGUF config/tokenizer, Card v3 fill, `inspect_all()`; `parse_target` signature unchanged | T307 |
| `scout/gate.py` (P2 T106, P3 T206) | σ-role and anchor eligibility also accept `QUANT_PARAMS["sample_dtypes"]`; anchor row byte size via `dequant.row_nbytes`. Plans of float dtypes are byte-identical | T308 |
| `scout/cka.py` (P3 T205) | `find_embedding` also accepts dtypes in `QUANT_PARAMS["sample_dtypes"]`; `CKA_PARAMS` unchanged | T308 |
| `scout/sigma.py` (P2 T107, P3 T204) | `decode_tensor` dispatches `sample_dtypes` to `dequant.decode(..., out="float64")`; `SIGMA_PARAMS`, `SUPPORTED_DTYPES` and `compute_all` unchanged | T308 |
| `scout/view.py` (P1 T011) | attribution per cell and `view.attribution` (§4.8); `DISCLAIMERS` unchanged | T318 |
| `scout/server.py` (P1 T013, P2 T113) | `make_server(..., store: CardStore \| None = None)`; `GET /api/cards` lists store entries when a store is set; new `GET /api/view?repo=&revision_sha=&component=[&object=]` | T318 |
| `scout/web/index.html` (P1 T014, P2 T114) | adds `#stored`, `#stored-list`, `#stored-refresh`; P1/P2 ids kept | T318 |
| `scout/web/app.js` (P1 T014, P2 T114) | stored-Card list and view; strip cells coloured by attribution; legend with caveat | T318 |
| `scout/cli.py` (P1 T012, P2 T112, P3 T215) | `scout attribute`, `scout jobs {worker,sweep,audit}`, `scout store {import,ls}`, `scout scan --store DIR`, `#selector` targets, `scout serve --store` | T319 |
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
| T306 | Card v3 schema, `with_attribution`, `object_id`; P2 C8 / P3 C3 accept v3 | opus | T202, T216, T301 |
| T311 | Scratch dir, disk accounting, purge, orphan sweeper, audit, worker lock | opus | T301 |
| T313 | Attribution fixtures: GLU families, derivations, GGUF/FP8/TE writers (test only) | sonnet | T103, T208, T303 |
| T304 | GGUF header parser (safe lookahead), adapter, config and vocab | opus | T203, T301, T303, T305 |
| T308 | Quantised σ-sample and anchor reads (gate/sigma/cka) | opus | T204, T205, T206, T303, T305, T313 |
| T310 | Full plan + full gate (`plan_attribution`, `format_full_plan`, `run_full_gate`) | opus | T106, T205, T305, T306, T311, T313 |
| T307 | Scan: selector, GGUF intake, Card v3 fill, `inspect_all` | opus | T207, T302, T304, T305, T306, T308 |
| T312 | Chunked FULL_DOWNLOAD to scratch | opus | T301, T310, T311 |
| T314 | Attribution statistic (probe, responses, LAP, Spearman, shift null, Bonferroni) + scipy | opus | T203, T205, T209, T305, T310, T313 |
| T315 | Job spec, protocol, backends (inproc/local/ssh), client ingest | opus | T301, T310, T311 |
| T309 | Card store + `cached_scan` | opus | T304, T306, T307, T314 |
| T316 | Worker `run_job` + `scout.jobs.worker` entrypoint | opus | T307, T310, T311, T312, T314, T315 |
| T317 | Attribution client (`attribute()`): cache, plan, gate, submit, ingest, Card v3, store | opus | T309, T315, T316 |
| T318 | View + server + frontend: attribution strip, stored Cards | sonnet | T011, T013, T014, T113, T114, T306, T309 |
| T319 | CLI: attribute, jobs, store, `scan --store`, selectors | sonnet | T215, T309, T311, T316, T317, T318 |
| T320 | P4 exit expectations C0–C16 (pure) + frozen literals | opus | T016, T115, T216, T306, T309, T310, T314, T317 |
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
| `scout/card.py`, `tests/test_card*.py`, `scripts/exit_expectations_p2.py`, `scripts/exit_expectations_p3.py` and their tests | T306 |
| `scout/scan.py` | T307 |
| `scout/gate.py`, `scout/sigma.py`, `scout/cka.py` | T308 |
| `pyproject.toml` | T314 |
| `scout/view.py`, `scout/server.py`, `scout/web/index.html`, `scout/web/app.js` | T318 |
| `scout/cli.py` | T319 |
| `scripts/pin_exit.py`, `scripts/p3_ledger.py`, `.gitignore` | T321 |

**Suites stay green.** Every task's acceptance runs the full `pytest -q` (P1–P4, including the P1–P3 offline exit
rehearsals). T306 flips `card.v3` and amends P2 C8 and P3 C3 in the same change.

**Routing.**
- Opus: tasks that touch download gating (T301, T304, T310, T312, T315, T316, T317), Card schema (T306, T309), similarity
  or dequantisation math (T305, T308, T314), invariant 1 (T311), or exit evidence (T320, T321).
- Sonnet, each with every signature and output string given:
  - T302 (a defaulted field and one parse)
  - T303 and T313 (test fixtures with every constant and bit layout given)
  - T318 (view fields, one route, rendering over a finished schema)
  - T319 (CLI plumbing over finished functions)

## 11. Items deferred to P4 by earlier phases, and their disposition

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
| P3 later | Per-block attribution evaluated on the P3 labeled set | **Re-deferred** (later.md). About 60–80 GB of full MLP reads on the Spark; valuable as a null calibration beyond the two exit controls, but not needed for the exit |
| FUTURE_IMPROVEMENTS P1–P3 | Minor items tied to P1–P3 files | Not P4 scope; unchanged |

## 12. Out of scope for Phase 4
- the retrain-plan loop, report export and UX polish (P5)
- GPU compute, a job queue beyond one serial worker, multi-node execution
- GPTQ/AWQ/bnb dequantisation, `.bin` pickles, T5/CLIP/MoE attribution, windowed alignment search
- JEV retraining or recalibration on quantised pairs

See `later.md`.
