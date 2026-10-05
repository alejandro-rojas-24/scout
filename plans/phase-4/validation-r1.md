# Phase 4 validation (validator, fresh context), plan r0

verdict: REVISE

Scope read: project-context.md, phases.md §Phase 4, phase-4 plan.md, tasks T301-T321, later.md; P1-P3 plan.md and task
YAMLs used only as contracts. Simulations were run under /tmp/val4 with numpy 2.4.6 and scipy 1.17.1. The
dependency/wave script found no cycle, no order violation and no file owned twice. Environment limits (no Hub, no
Spark, gated repos) are not findings.

Checked and sound, so not listed below:
- The byte arithmetic of §2.5 and the T310 slack cases.
- GGUF bit layouts in T303/T305 against ggml's dequantize_row_q4_0/q8_0/q4_K/q5_K/q6_K and get_scale_min_k4, plus the
  GGML type-id and block-size table.
- The GGUF lower-bound argument: 13 B per KV, 24 B per tensor info, 8 B per string element. Every NeedMore bound is
  <= infos_end <= data_offset, so header bytes end exactly at infos_end.
- Bonferroni: z_crit 3.836 at 160 tests and 3.570 at 56.
- Rectangular LAP cost: 2048 x 28672 took 0.7-0.8 s on null-like costs.
- The rotation/permutation/norm-folding invariance of x̂Wᵀ.
- References to the P1-P3 contracts: structure_alignment, find_embedding, VocabIds, GateDecision fields, p3_ledger
  KINDS, P2 C8, P3 C3, CountingTransport.orig_path, DEFAULT_THRESHOLD_BYTES. All are accurate except the ones flagged
  below.

findings:

- severity: major
  where: plan §3.8 steps 6-7 ("Different blocks of the same model have unrelated neurons, so these n·2 z-values measure the run's own null"; "family-wise error ≤ α"), §2.4, T313 glu_weights, T314 test_null_family / test_inflated_null_adjusted
  problem: >
    The shift null is valid only if the z of a misaligned pair (t vs a(t+k)) is exchangeable with the null z of the
    aligned pair (t vs a(t)). The planner's generator draws each layer's neuron directions independently per model and
    per layer. Its inflation is therefore depth-independent, and the shift null tracks it by construction. Real
    independent LLMs trained on similar data plausibly share depth-local features ("universal neurons" at similar
    depths). In that case aligned independent blocks are more alike than shifted ones.
    Simulation (/tmp/val4/depthlocal.py): a per-layer shared feature dictionary, drift 1.0 between layers, s=1, a=2,
    γ=0.9. Aligned null z mean was 2.81 against a shift-null mean of 1.24. There were 2 false block attributions in 64
    at α=0.01 and no abstention: null_ok stayed true, so the guard did not fire.
    The gap grows with m (/tmp/val4/scale.py, same setting, 12 blocks): shift-to-aligned gap 0.08 at m=256 and 1.54 at
    m=1024. At the exit's m=2048 it would be larger still. The offline fixtures run at m <= 512, so they cannot reveal
    the effect.
    The exit's C11 would catch this as a FAIL, not a false PASS. But the attribution.v1 p-values and the stated FWER
    ≤ α are not justified in this regime, and every non-exit use of `scout attribute` inherits the overclaim.
  fix: >
    Add to T313 a frozen "depth-local universality" null family: a shared per-layer feature dictionary with drift.
    Add to T314 a test at m >= 1024 that asserts a bound on false attributions for it. If the bound cannot be met,
    change the null so it preserves depth alignment. State the exchangeability assumption in §3.8 and in the
    attribution.v1 method/caveat text, and describe p as "in-run adjusted, assumes a depth-exchangeable null", not as
    an unconditional FWER. Record the null-scaling-with-m argument in §2.4 next to the √(m−1) power argument.

- severity: major
  where: T316 behavior 2 and 8 (on_event -> emit from 4 download threads), P1 T002 behavior 5 ("call on_event outside the lock"), T315 behavior 4f ("worker_seqs is consecutive (each seq = previous + 1)"), T320 C5
  problem: >
    Worker events are appended under the ByteLog lock, but on_event (the stdout emit) runs outside it. With 4
    concurrent download threads, fetch events can reach stdout out of seq order. T315 then marks the job "failed"
    because the seqs are not consecutive in arrival order. T317 raises JobError and writes no Card, and C5 FAILs.
    A healthy 70b job, about 2,800 chunk events over hours, can therefore be thrown away after 179 GB. The offline
    tests can become flaky for the same reason.
  fix: >
    Serialise emission in seq order on the worker, for example by emitting under a dedicated lock that also covers
    the append, or by a sequencer thread that releases events strictly by seq. Alternatively, require T315/C5 only
    that the set of ingested seqs is exactly {first..first+n−1}. Add a test with many concurrent chunk events.

- severity: major
  where: T316 behavior 3 (log.note("job", sweep) before any stage) vs T301 behavior 5 (ingest requires a Stage name) and P1 T002 behavior 5 (stage "NONE" when no stage yet); T315 behavior 4e
  problem: >
    The worker's first log event, the sweep note, is recorded with stage "NONE". inspect_all only calls
    stage(RESOLVE) later. The client's ByteLog.ingest rejects "NONE" with ValueError, so every real job fails on its
    first event. T315 does not say how run_job_via handles an ingest ValueError: kill, fail and sweep, or an uncaught
    exception. The specs of T301, T315 and T316 contradict each other, and the fix crosses task files.
  fix: >
    Let ingest accept "NONE" (T301), or have the worker log the sweep after stage(RESOLVE), or report it only in the
    result. Specify in T315 that an ingest error kills the worker, marks the job failed and sweeps.

- severity: major
  where: plan §4.3/T310 interface `FullPlan.reads(self) -> list` (method) used with P2 `gate.flag_confirmer` (T106: decision() sets n_reads = len(plan.reads)); T310 test_run_full_gate; T317 behavior 6
  problem: >
    T310 and T317 pass a FullPlan to P2's flag_confirmer. That calls P2 decision(plan, ...), which evaluates
    len(plan.reads). On FullPlan, reads is a bound method, so the call raises TypeError, and run_full_gate turns that
    into "declined via error". As specified, T310's own acceptance test ("flag_confirmer(plan_id, cap) approves via
    cli-flag") cannot pass. Every confirmed attribution would be declined. The plan's claim that the P2 contract is
    reused unchanged is inaccurate.
  fix: >
    Make FullPlan.reads a list field and update T312/T316/T320 accordingly. Alternatively, define
    fullplan.full_flag_confirmer built on full_decision and use it in T310/T317 instead of gate.flag_confirmer.

- severity: major
  where: plan §2.2 E3 ("-> commit exit/ledger_p4.jsonl") and E4; §2.3 C1 ("worker git_commit == client git_commit"); T320 check_env; T321 behavior 2/4
  problem: >
    E3 tells the human to commit the ledger, which moves the client's HEAD. C2 explicitly allows that, since it only
    diffs CODE_PATHS since the P4_PLAN commit. C1 compares the exact worker and client commits, and nothing in §2.2
    syncs the Spark checkout. The new commit would also have to be pushed somewhere the Spark can fetch. Following
    the E-steps literally therefore FAILs C1 at E4. The same applies to any pins commit after E2.
  fix: >
    Compare a code identity instead of the commit: the tree hashes of CODE_PATHS (`git rev-parse HEAD:scout`,
    `HEAD:scripts`, `HEAD:pyproject.toml`) or a hash of the tracked files, reported in the hello. Otherwise add an
    explicit documented "sync the Spark checkout to the client HEAD" step before E2 and E4, and state that it is
    setup, not intervention.

- severity: major
  where: plan §2.3 C15 ("probe tokenizer file tokenizer/tokenizer.json"), §3.9, T320 behavior 16; the Qwen-Image layout is not V-marked in §2.1
  problem: >
    C15 hard-codes `tokenizer/tokenizer.json` for Qwen/Qwen-Image. Its tokenizer/ folder very likely ships vocab.json
    and merges.txt without tokenizer.json. P2's resolution (T104) then falls back to `tokenizer/vocab.json`, and P3's
    vocab_ids accepts that, so attribution itself would work. But C15 is evaluated only at E4, after about 205 GB,
    and correcting it is a scripts/ change after P4_PLAN, so a new plan and rerun are needed.
  fix: >
    Mark the file with V in §2.1. Evaluate C15's tokenizer clause at E3 from the stored header-scan Card
    (stats.tokenizer_minhash.source_file) so that a mismatch surfaces before P4_PLAN. Accept `tokenizer/vocab.json`
    as well.

- severity: minor
  where: plan §4.4 job.v1 vs T315 make_spec / T316 behavior 6 (spec["plan"]["threshold_bytes"]); T310 behavior 12
  problem: >
    §4.4 omits plan.threshold_bytes, which T315 writes and T316 reads. The worker also uses the spec's threshold for
    its own below-threshold auto-approval, so the worker gate is only as strong as the client-supplied threshold. A
    spec with a huge threshold is approved on the worker without a human pair. The ranges are still pinned, so the
    worker never reads more than the displayed plan.
  fix: >
    Add threshold_bytes to §4.4. Either cap the worker's threshold at DEFAULT_GATE_THRESHOLD_BYTES, or require
    via "cli-flag" when bytes_cap exceeds the worker's own default.

- severity: minor
  where: T311 behavior 1 (mkdir, then open and flock .lock) and 6 (a directory without .lock is an orphan); `scout jobs sweep` via cron or the client after a failure runs without WorkerLock
  problem: >
    A sweep that lists a job directory between its mkdir and the flock treats it as an orphan and deletes its
    contents, including a .lock the job has just locked. After that the job's lock no longer protects it, and the
    next sweep deletes the live job's tensors. This causes job failure, not a weight leak, but it is a correctness race.
  fix: >
    Create the job directory under a temporary name with .lock already flocked, then rename it into place.
    Alternatively, have the sweep skip directories younger than N seconds, or delete only when it can take WorkerLock
    non-blocking.

- severity: minor
  where: plan §3.3 "Crash mid-job", D59
  problem: >
    If the worker crashes and the client is also gone, up to 179 GB of weights stay on the Spark until the next
    worker start or cron run. The cron is only documented; neither E2 nor C1 verifies it. The claim "The finally
    block ... purges on every exception path" also does not cover the heartbeat thread's BrokenPipeError during
    COMPUTE: the main thread keeps computing for up to about 45 min after the client dies.
  fix: >
    Report whether the sweep cron is installed in the hello, and check it in C1 or state the bound explicitly. Have a
    heartbeat BrokenPipe set an abort flag that attrib checks per block.

- severity: minor
  where: T317 behavior 7, T315 behavior 4g
  problem: >
    For a failed job, the client displays the error and the sweep report, but not the bytes the worker fetched, and
    no Card is written. Up to about 179 GB fetched on the Spark are then visible only in the in-memory ByteLog, and
    only as stage lines.
  fix: >
    On failure, display totals_by_origin and the JobRecord (bytes by class, disk peak), and include the totals in the
    JobError message.

- severity: minor
  where: plan §2.3 C6, T320 behavior 7 ("every request to a path ending .safetensors/.gguf")
  problem: >
    Data requests go to CDN/Xet URLs whose paths do not end in .safetensors or .gguf. Read literally, the range
    clause is vacuous for every byte-carrying request. It also needs the repo to identify "that model", since
    orig_path is repo-relative.
  fix: >
    Specify that C6 uses CountingTransport.orig_path, following redirects, and the repo parsed from the
    /{owner}/{name}/resolve URL, for every request that carries a Range header.

- severity: minor
  where: T316 behavior 4 (wire_audit imports scripts.exit_check on the worker), §2.5
  problem: >
    `scripts` is not a packaged module (T001: packages = ["scout"]). The import works only when the worker runs from
    an editable install that puts the repo root on sys.path. E2 does not exercise it, so a non-editable install on
    the Spark fails only at E4, although still before any weight byte is read.
  fix: >
    Report `wire_audit_available` in the hello and check it in C1 at E2, or move CountingTransport into scout.

- severity: minor
  where: plan §2.1 targets, §2.3 C10/C11, T321 rehearsal
  problem: >
    Neither live target exercises block localisation. Both are expected to be about 100 % attributed, and the te
    subject is most likely a byte-identical copy of Qwen2.5-VL-7B (equal content_digest). A bug that makes per-block
    verdicts non-local, such as a result shared across blocks or an alignment that ignores the block, would pass
    C10/C11. Only T314 test_partial_reinit covers this, offline.
  fix: >
    Record in the evidence whether subject and base content_digest are equal. Add a partially re-initialised subject
    to the T321 rehearsal with per-block expected statuses.

- severity: minor
  where: plan §2.1 "Reruns are allowed, because the statistic is deterministic", §2.4
  problem: >
    With correct code, the control can falsely attribute a block (up to α = 1 % per job), and each reference can
    abstain on null_ok by chance (about 1 %). A correct implementation therefore FAILs C10/C11 with a probability of
    a few %. Because the statistic is deterministic at fixed pins, a rerun reproduces the FAIL.
  fix: >
    State in §2.4 that such a FAIL can only be resolved by a logged human decision, and give its probability.

- severity: minor
  where: T306 depends_on (uses RepoFile.lfs_sha256 from T302), T320 depends_on (C12/C16 use T318 build_view attribution fields and view.ATTRIBUTION_CAVEAT)
  problem: Missing edges T306→T302 and T320→T318. The waves happen to order them, so nothing breaks today.
  fix: Add the edges.

- severity: minor
  where: plan §3.8 "Alignment" ("Equal-depth pairs (both exit targets) get the identity"); §2.5 te byte table
  problem: >
    The te control Llama-3.1-8B has 32 blocks against the subject's 28, so its alignment is reldepth, not the
    identity. The plan downloads all 32 control blocks, about 1 GB, although only 28 are ever tested (window 0).
  fix: Correct the statement. Optionally plan only the reference blocks that the alignment and the shifts use.

- severity: minor
  where: plan §3.8 item 7 vs T314 behavior 9c
  problem: >
    The two specs define n_tests differently. §3.8 includes the tests of a reference that later abstains. T314 counts
    "references with status tested", which is evaluated before abstain is assigned. The two readings can be
    implemented differently.
  fix: >
    State that n_tests counts the k=0 tests of every reference that reached testing, abstaining or not, and assert
    it in test_bonferroni with an abstaining reference.

- severity: minor
  where: T319 behavior 1 (`scout attribute --json` -> {card_key, object, cached, attribution})
  problem: >
    The JSON report carries ATTRIBUTION_CAVEAT inside the attribution, but not the three standing disclaimers
    (invariant 5).
  fix: Add "disclaimers": DISCLAIMERS to the JSON output and test it.

- severity: minor
  where: plan §3.5 (refs/, by-weights/, STORE), §3.3 cron `>> ~/.cache/scout/sweep.log`
  problem: >
    Invariant 1 says the Card is the only persisted artifact. The store also persists ref and by-weights index
    files, and the documented cron appends to a log. Their status is not stated.
  fix: >
    State that refs/ and by-weights/ are derived indexes, and make them rebuildable from objects/, for example with a
    `store reindex` or by import_tree over the objects. Drop the sweep.log redirect or state its status.

- severity: minor
  where: T309 same_weights / by-weights index, T318 stored-Card routes and UI list
  problem: >
    Simplicity: the by-weights index and the UI stored-Card list are not needed for the exit or for the roadmap line
    ("Card store with caching"). They add code and tests.
  fix: Consider moving them to later.md, or keep them and accept the extra surface knowingly.
