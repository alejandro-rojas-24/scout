# Phase 5 validation (r1)

verdict: REVISE

Scope read: project-context.md, phases.md Phase 5, plans/phase-5/plan.md, tasks T401-T419, later.md (scope only),
and P1-P4 plans/tasks as contracts. Environment limits (no Hub, Spark, GPU or API key here) are not findings.
Mechanical checks (script under /tmp/p5val): every depends_on resolves, there are no cycles, every task sits in its
earliest wave, the plan's §10 table matches the YAML deps and routes, and no file belongs to two tasks.

findings:

- severity: blocker
  where: T409 behavior 6 (inputs.ledgers summary), T411 behavior 11 (L10b), T417 behaviors 6 and 9, plan §2.2 E5/E8, §3.7 L10
  problem: >
    The bundle embeds `inputs.ledgers` = {n_entries, last_seq, last_line_sha256} of every ledger it is given,
    and that includes p5. E5 renders and lints with the p5 ledger as it stands. E5 then appends P5_REPORTS, and
    E6/E7 append P5_READER. E8 "re-lints every file". L10b rebuilds the bundle from the now longer p5 ledger, so it
    differs from the embedded bundle. That makes L10b count 1 and C15 FAIL, so "EXIT CHECK (P5)" FAILs on a correct
    build every time. The same staleness hits every product report that cites a live ledger. T419 test_rehearsal_pass
    would hit this, but T419 cannot edit T409, T411 or T417, so no task owns the fix.
  fix: >
    Make the rebuild independent of later appends. Either drop the ledger summaries from the rebuilt comparison and
    verify only the cited entries (seq + line_sha256, chain intact up to that seq), or have L10b read each ledger
    truncated at the last_seq recorded in the bundle. State which one in §3.7, T409 and T411, and add a T411 test:
    append an entry to a cited ledger after rendering, and L10b stays 0.

- severity: major
  where: plan §2.4 points (3)-(4) (lines 250-259), §8 rows 1-2, C11/C12, §2.1 refreeze rule, T416 behavior 3 (C2), T417 behavior 2
  problem: >
    The exit requires outcome "success", but the plan states no pass probability for a correct build. (a) C12 is
    called "the main risk", and the toys "do not establish C12 at exit scale" (0.42 training tokens per trainable
    parameter, against 1.3-3.5 in the toys). (b) The C11 risk is understated. The plan says the CE drift (z_adj
    about 2 by step 1000) is "far below crit", but it compares toy z at m = 256 against the toy crit. z = rho*sqrt(m-1)
    grows with the probe count, and crit does not. Rescaled to the exit's m = 2048, the toy-1 CE values at steps
    1000-1500 (1.66-2.01) become about 4.7-5.7, above the exit crit of 4.12. So if quality needs a long budget (which
    is (a)), the flagged signal is likely back by then. The outcome is then "neither" or "signal_persists" on a
    correct build. With a fixed seed that is close to deterministic. The only way out is a refreeze with a reason,
    and C2 accepts that as a plain PASS. That allows exactly the post-hoc budget/threshold change that §2.4 says must
    not happen ("the 0.90 bound and the budget are not changed after a run"), and E8 never distinguishes it.
  fix: >
    Before P5_FREEZE: rescale the toy drift to m = 2048 in §2.4 and §8, and state an expected C11/C12 pass
    probability, or the evidence that bounds it. Pre-register any fallback (for example a second frozen budget) in
    RETRAIN_PARAMS. Make a refreeze after a P5_RETRAIN that changes RETRAIN_PARAMS show in P5_EXIT as "PASS after
    refreeze: <reason>" (a listed exception like a waiver), not a plain PASS. Consider splitting loop-correctness rows
    (the trace is complete and consistent, unflagged blocks stay attributed, the flagged signal is measured on the
    live tensors) from the empirical outcome rows.

- severity: major
  where: plan §3.1 step 5, D81, §4.2 outcome rule, T406 behavior 5, C11
  problem: >
    The stop rule takes the first of up to 8 looks at which "no flagged block is attributed" holds. That condition is
    a failure to reject at a familywise-corrected crit; it needs no margin and does not have to persist. Under the
    drift above, z can sit just below crit at one eval and cross it at the next. The loop then records "success" at a
    transient point and never looks again. RETRAIN_CAVEAT warns about this in words, but C11 cannot tell a stable drop
    from a lucky look, so "confirm the signal drops" is not established.
  fix: >
    Require the drop to persist, by either (a) the stop row and the next eval row (or all remaining rows to
    max_steps, with quality still checked at the stop row) both having signal_dropped, or (b) a margin (for example
    max flagged z_adj <= crit/2) frozen in RETRAIN_PARAMS. Record the post-stop maximum flagged z_adj in retrain.v1
    and the report.

- severity: major
  where: T417 behaviors 5, 8 and 9 (the last P5_READER and the last P5_RETRAIN win), T419 test_rehearsal_crash_then_sweep, plan §8 last row
  problem: >
    Every gate that should be single-shot can be retried until it passes. (1) `read --reader llm` can be re-run as
    often as wanted; E8 uses the last P5_READER for that reports_seq, and P5_READER has no attempt count. A
    nondeterministic LLM reader can therefore be re-sampled until C16 passes. (2) The human reader's `read` prints the
    per-question score table and can be re-submitted with edited answers; the last submission wins. (3) After a
    completed P5_RETRAIN with passed false, `run` can be repeated with the same plan without a refreeze. That
    contradicts §8 ("a FAIL is resolved only by a logged human decision and a refreeze"). prior_attempts is recorded
    but never affects the verdict.
  fix: >
    Count every P5_READER per reports_seq and reader. For the LLM, pre-register k-of-n (or first attempt only) in
    READER_PARAMS. For the human, allow one submission per reports_seq unless a recorded reason is given, and don't
    print per-question correctness before the answers are hashed into the ledger. Refuse a `run` after a non-crash
    P5_RETRAIN FAIL unless --rerun-reason is given (crash reruns stay allowed). Have E8 list every retry with its
    reason, as for waivers.

- severity: major
  where: plan §3.8 (Q6, Q7, Q8, Q10 keys), D88, C16/C17, T412 behavior 1, T419 test_rehearsal_reader_fail
  problem: >
    "Hand a report to someone without editing" rests on the readers, and the readers are not shown able to fail.
    Q6 ("yes"), Q7 ("no"), Q8 ("no") and Q10 have constant keys a reader can give from priors without reading the
    report. Two of the four critical questions (Q6, Q8) are of this kind, and Q1 is trivial. That leaves Q2 as the
    only content-critical question. "required_edits == 0" depends entirely on how lenient the LLM is. The only failing
    reader test uses a stub (T419), so the real reader's ability to discriminate is never measured. The false-fail rate
    is not stated either: Q3 requires an exact set of repo-plus-reason pairs over the whole P3 library, and Q4 is
    ambiguous in the retrain report (the attribution section's 24 of 24 against the trace's 21 of 21 unflagged). With
    only one miss allowed, a correct build can fail C16. Since E8 takes the last attempt, that pushes toward retries.
  fix: >
    Add a negative control to E6/E7. Take a canary copy of one exit report with frozen, planted defects: a
    contradicted number, a sentence that answers the licence question, and the distillation disclaimer removed. Score
    the canary with the same reader; it must FAIL (a required edit quoting a planted defect, or a wrong critical key).
    Record that in C16/C17. Replace the constant-key critical questions with content-specific ones (for example "Which
    blocks of which reference are attributed?", or "What licence does the report give for X?"). Specify Q3 matching on
    reason codes shown verbatim. Label the attribution section as the unmodified (baseline) subject. State the
    expected false-fail rate, or the k-of-n rule from the previous finding.

- severity: major
  where: T408 behaviors 2, 7 and 8; T401 behavior 3; P4 T309 behaviors 4 and 9 (cached_scan returns the newest entry with has ⊇ require)
  problem: >
    After a completed retrain, the Card v4 is the newest store entry with has.tokenizer for the subject key. A second
    `scout retrain` on the same subject therefore gets the v4 Card from cached_scan(tokenizer=True) as its
    "header-scan Card", and so does an E4 rerun after a failed P5_RETRAIN. Planning succeeds and the GPU job runs.
    Then with_retrain raises "with_retrain needs a card.v3 header-scan Card", so no Card is stored and C13 FAILs.
    parent_object would also point at a v4 object. No test runs a retrain twice.
  fix: >
    Specify the parent: the newest card.v3 entry of the key that has neither has.attribution nor stats.retrain
    (rescan when there is none). Alternatively, with_retrain derives its v3 parent from a v4. Add a T408 test that
    retrains twice on the same subject.

- severity: major
  where: T404 (route sonnet, RT_* frozen, acceptance), T406 test_loop_success, T419 test_rehearsal_pass
  problem: >
    No one has measured the rehearsal fixture. T404's acceptance never runs the P4 attribution. Yet T406 and T419
    need rt_family to give: base attributed in all 8 blocks with null_ok and control_ok, control 0 hits, redrawn block
    3 not attributed, and outcome "success" with unflagged 6/6. The §2.4 toys are different models (hidden 96/128 on
    Python source, not Markov text). If the fixture misses, T406 (opus) cannot change the frozen constants in a file
    T404 owns. That is a hidden design decision routed to sonnet. Also, on order-1 Markov text the argmax is mostly
    set by the previous token, so top-1 agreement after re-init is probably already >= 0.90. The rehearsal would then
    succeed at the first eval and never exercise quality recovery.
  fix: >
    Have the planner run rt_family once before freezing RT_*, and record baseline z, control z, reinit top-1 and the
    stop step in §2.4. Add a T404 acceptance test for attribution (derived vs base 8/8, control 0, redrawn block 3 not
    attributed). Require in T406 that reinit top-1 < min_top1_agreement or reinit ppl_ratio > max_ppl_ratio, so the
    rehearsal exercises recovery. Route T404 to opus, or give T406 a sanctioned path to adjust the fixture.

- severity: minor
  where: T410 behavior 9 vs T411 behavior 6 (L5b)
  problem: The per-reference attribution table renders no `.reason`, while L5b requires a reason on every abstain/untestable reference row. Any report with an abstaining reference fails lint (exit 11). T411 has no L5b test.
  fix: Render the reference reason in `<span class="reason">` and add T411 test_l5b.

- severity: minor
  where: T410 behavior 10 (trace table)
  problem: Trace rows do not render base_status, null_ok or control_ok, so an abstention row looks the same as a signal drop to a reader.
  fix: Render them, or signal_dropped, per row.

- severity: minor
  where: T412 behavior 1 (Q9 retrain key)
  problem: The key "yes iff outcome == success" is wrong for quality_not_recovered, where the signal did drop.
  fix: Key Q9's second part on the stop row's (or the last row's) signal_dropped.

- severity: minor
  where: plan §2.1 lines 80-84 vs T416 behavior 3
  problem: The plan says C2 asserts the freeze's pins, model and cited-ledger sha256 at every later step; T416's C2 does not check them.
  fix: Add those comparisons to check_frozen.

- severity: minor
  where: C7 (plan §2.3), §3.2, T407 behavior 5
  problem: The claim "no checkpoint or other file was ever written" rests on scout's scratch bytes_written and the scratch audit. Writes by torch, transformers or triton outside scratch (~/.cache, /tmp) are not audited, and the AST test covers scout code only.
  fix: During COMPUTE, point HOME, TMPDIR, HF_HOME, TORCH_HOME and TRITON_CACHE_DIR into the job dir, so the post-purge 0-file audit covers them.

- severity: minor
  where: T416 behavior 5 ("data files only" row), T417
  problem: The row is delegated to T417, and T417 never defines it. The worker also reads README.md (licence) and the dataset API, so a literal "every dataset request is a data file" rule would FAIL on a correct build.
  fix: Define the allowed dataset paths (the API, README.md, the two files) in T416.

- severity: minor
  where: plan §3.3 "Data files", §3.4, D84
  problem: The P1 threshold is cumulative per ByteLog (P1 plan line 255). data_max_bytes 32 MiB, plus three tokenizers, plus the 16 MiB unknown-size reservation, can exceed 64 MiB. That is fail-safe (ReadThresholdExceeded), but "so the job's meta stays under the threshold" is not established.
  fix: State the budget against the cumulative job log, or lower data_max_bytes.

- severity: minor
  where: D99, plan §3.6, later.md
  problem: The plan says "scout never ... reads one back", but report-lint, `read` and `exit` all read report files back. The invariant 1 acknowledgement is still open while E5 writes the files.
  fix: Correct the wording, and make the human acknowledgement a precondition of E5 (as P3 D36).

- severity: minor
  where: LINT_PARAMS license_deny, L2a
  problem: The checks are lexical. "allows", "permits", "can be redistributed" and numbers written as words pass. Risk is low because only templates and model-card strings reach the text.
  fix: Extend the deny-list, and assert the templates contain no number words.

- severity: minor
  where: T410-T413 acceptance ("T409 fixtures")
  problem: The fixtures live in T409's test module. No helper file owns them, and T413 needs them across modules.
  fix: Move them to tests/helpers/report_fixtures.py, owned by T409.

- severity: minor
  where: C11/C12, T416 behaviors 8-9
  problem: No exit row shows that re-initialisation changed the forward pass, or that training changed the weights. If reinit touched only the fingerprinted copies, C11 and C12 would both pass.
  fix: Add rows reinit.ppl_ratio > 1 (or reinit top1 < 1) and retrained_digest != the digest of the re-initialised tensors.

- severity: minor
  where: T409 behavior 6 (P4_EXIT `targets.<label>.object`)
  problem: The P4 contract says only "the object id" for the P4_EXIT payload. The key name `object` is assumed, not specified.
  fix: Pin the key name against P4 T321, or match any 64-hex object value under targets.<label>.

- severity: minor
  where: T409 behavior 4, T410 behavior 8, T411 L2c, T413 GET /api/report
  problem: The System 2 report section (with L2c) and the web report route are covered by no exit row. They could be deferred without failing the exit.
  fix: Cut them or defer them to later.md, or justify keeping them.
