verdict: REVISE

# Phase 5 validation (round 2, against plan r1)

Scope read: project-context.md, phases.md Phase 5, plans/phase-5/plan.md, tasks T401-T419, later.md (leakage only),
and P1-P4 plans/tasks as contracts. Simulations: /tmp/p5val/sim.py (stop rule vs. validator), planner scratch
/tmp/p5r1 (rehearsal fixture logs v6/v7, fam_400_4_1_1_0.7_128.pkl), scipy check of the stated t-values.

Verified and holding:
- Statistics: t.isf(0.01/48, 23) = 4.1207, t.sf(4.12, 23) = 2.09e-4 (6 looks: 1.25e-3), t.isf(0.01/16, 7) = 5.202;
  2-of-3 with p = 0.10 gives 0.028 per file, 0.082 for three files and 0.107 for four, as stated.
- Rehearsal fixture claims (plan 2.4, T404 behavior 7) match planner log v7.log and the pickled family: control
  intermediate 96, z_adj 10.6 x 8, redrawn block 3 at -0.31, re-init 1.324 / 0.9365, stop 100 / confirm 150, max
  flagged z_adj 1.96. The step-100 candidate has a narrow margin (1.0477 vs 1.05), but test_loop_success does not pin
  the stop step, and every later row is a candidate.
- The T406 stop rule gives the documented results (rehearsal: success, stop 100, confirm 150).
- No two tasks share a file. Every dependency points to an earlier wave, all P1-P4 dependency ids exist, and the
  graph has no cycle. The P4 interfaces P5 uses (p4.Env.max_rss_bytes, check_env/check_resources signatures,
  LocalBackend argv_prefix, worker_hello, run_job_via, code_identity, crash_worker) match the T315/T316/T320/T321 specs.
- store.import_tree stamps the current params_digest, so imported P3 Cards stay findable. DISCLAIMERS[0] is the
  distillation sentence (P1 T011), and no bundle key contains "cosine" (P3/P4 schemas).
- Invariant 1 (no weight export, no save calls, env and cache redirected into the purged job dir) and invariant 2
  (data as logged META reads, recipe_digest inside plan_id, exact-pair gate) hold as specified.
- Coverage of the Phase 5 roadmap line is complete. Nothing from a later phase leaks in.
- D100 (PASS / PASS with finding) still meets the roadmap exit. The exit sentence is about the report, and a negative
  retrain outcome is rendered and read like a positive one (T419 test_rehearsal_finding). The loop's ability to show a
  drop is proven by the deterministic rehearsal.

findings:

- severity: major
  where: T401 behavior 2f (first "success" clause and the non-success clause) vs T406 behavior 5 and T406
    test_transient_drop_not_a_stop; plan 4.2
  problem: |
    validate_retrain gives two contradictory definitions of "success". Its first clause says stop_step must be the
    FIRST row with signal_dropped and quality_ok. T401's own test_validate_retrain_rules enforces that clause ("stop_step
    pointing at the second qualifying row -> problem"). The persistence clause in the same item says stop_step is the
    first such row whose NEXT row is also dropped. The text also says "signal_persists/quality_not_recovered/neither ->
    no row has both flags", but under persistence an unconfirmed candidate can come before a non-success end.
    Simulated (/tmp/p5val/sim.py):
    - T406 test_transient_drop_not_a_stop yields success with stop 150 and confirm 200. The first clause rejects it,
      so T406's validate_retrain assertion (behavior 8) fires.
    - A transient candidate followed by a persisting signal yields signal_persists, which the non-success clause
      rejects.
    On the Spark that AssertionError kills COMPUTE with no result message. T417 records it as a crash, the crash
    reruns use the same seed and repeat it, and the exit ends in a forced refreeze on a correct build.
  fix: |
    Delete the "FIRST row with signal_dropped and quality_ok" sentence and the "no row has both flags" requirement
    from 2f. Keep only the persistence rule. Make test_validate_retrain_rules use "a stop_step that is not the first
    confirmed candidate", and add a validate case for "candidate, not confirmed, later signal_persists -> []".

- severity: major
  where: T412 behavior 2-3 (answer_key, normalise "repos exact", "repo@sha"); T410 behaviors 7 and 9 (reference and
    subject rendered as titles); P3 T211 / P1 T011 title = f"{repo}@{sha[:12]}" (+ " / component"); T418
    answers_from_key
  problem: |
    The keys of critical questions Q2 and Q4 (and of Q3 and Q5) are bare repos. The report prints every reference only
    as its title, "repo@sha12". A reader who copies the printed identifier ("Qwen/Qwen2.5-0.5B@0123456789ab") is
    scored wrong under "repos exact". For the multi-component `te` subject (invariant 6 input), Q1's printed title is
    "Qwen/Qwen-Image@<sha12> / text_encoder". The "repo@sha" normaliser has no rule for a component suffix or trailing
    text, so a correct answer can fail critical Q1.
    These failures come from formatting, so they are correlated across the three samples, and 2-of-3 does not protect
    against them. The 0.08 false-fail estimate assumes independent samples. The gate is one-shot, and only a refreeze
    can fix a FAIL. The rehearsal cannot catch this, because answers_from_key produces bare repos.
  fix: |
    Specify the normaliser on what the report prints:
    - Accept "repo" or "repo@<hex prefix>" for every repo item, and ignore a trailing " / component", "#component" or
      a parenthetical.
    - Q1: take the first run of 7 or more hex characters after "@" as the sha.
    Make T418 answers_from_key answer with the printed titles (and the te subject title with its component), and add
    T412 tests for those forms.

- severity: major
  where: plan 3.8 canary rule and rates; T412 behavior 5 (D3) and behavior 6 (failed_as_required = not passed AND
    detected); T410 behavior 7 (VERDICT_SEMANTICS rendered in #system1); P3 T211 VERDICT_SEMANTICS
  problem: |
    In practice the canary rule can fail a careful reader rather than catch a careless one.
    - D1 does not make critical Q4 wrong, because Q4 is scoped to the #attribution table, which keeps the true count.
    - D2 and D3 are linked to the non-critical Q8 and Q9.
    - D3 often does not change Q9 either. The canary is the `retrain` report, where System 1 runs, and #system1 still
      shows VERDICT_SEMANTICS: "... (distillation and shared tokenizers remain possible)". From that, a careful reader
      still answers Q9 "no".
    A careful reader therefore answers Q8 "yes" (D2 detected) and everything else correctly: 9 of 10, all critical
    questions right, so `passed` is true. Unless they also file a required edit, failed_as_required is false and C16
    or C17 FAILs. T412 test (d) already shows that one wrong non-critical answer leaves the canary "passed".
    The stated c <= 0.10 rests on three effective defects that do not exist. Each C16 FAIL costs a refreeze and a full
    GPU retrain rerun (reports require a P5_RETRAIN after the last freeze).
  fix: |
    Define failed_as_required as len(detected) >= min_detected (drop "not passed"). A prior-answering reader still
    detects nothing and fails the canary row.
    Alternatively, also strip the distillation clause from VERDICT_SEMANTICS in the canary (D3), or pick a canary
    source whose System 1 is not run.
    Restate c with the defects that actually work, and add a T412 test: "careful reader: Q8 yes, Q9 no, no edits ->
    failed_as_required True".

- severity: major
  where: T417 behavior 8 (read --reader llm appends P5_READER only after all 12 calls are scored); plan 2.1 "One-shot
    gates"; T416 check_one_shot (C18)
  problem: |
    The LLM gate is one-shot only if the invocation finishes. If it is killed, raises after some responses arrived, or
    is interrupted after a per-sample line is printed (print order is not specified for llm, unlike the human path),
    nothing reaches the ledger. A rerun then draws fresh samples, and C18 cannot see the discarded draw. `run`
    closes this gap with P5_ATTEMPT before the job; `read` has no equivalent.
  fix: |
    - Append a P5_READ_ATTEMPT {reports_seq, reader} before the first call. Treat an attempt without its P5_READER as a
      consumed reading in C18 and in `read` (exit 5 unless refrozen), except for the existing "every call failed before
      any response" case, which is recorded as unreachable.
    - Append P5_READER before printing any score, as the human path does.
    - Add T417 tests for a kill after k responses, followed by a second read.

- severity: major
  where: T417 interface and behavior 3 (populate_store imports only the report subjects' P4 entries); T409 behaviors 5
    and 7 (licences of "every attribution reference", data.cards for "references"); T411 L6 with require_ledger; P4
    P4_EXIT records only targets.<label>.object
  problem: |
    The `te` report's attribution references are Qwen/Qwen2.5-VL-7B-Instruct, the independent candidate OLMo-2 and
    the control Llama-3.1-8B. Their header-scan Cards exist only in cards/p4-store, and populate_store does not import
    them. No ledger entry cites them: P4_EXIT names only the subject's attribution object, and Qwen2.5-VL and OLMo-2-7B
    are not in P3.
    T409 needs those Cards for the licence table (invariant 5: the licence question names the claimed licences) and
    lists "references" among the Cards in inputs.cards, but says nothing about what happens when a reference Card is
    absent. The implementer must therefore either crash, silently print "unknown", or include uncited Cards, which
    makes L6 (require_ledger) FAIL C15 for `te`, a gating row. The rehearsal world has the same gap: TE_BASE and
    CONTROL are not in cards_p3.
  fix: |
    - Have populate_store also import the P4 attribution reference and control entries of each P4 report target,
      byte-identically.
    - In T409, state the role and citation rule for attribution-reference Cards: cited through the attribution Card's
      own citation, or exempt from per-Card L6 citation with an explicit rule.
    - Define the licence fallback when a reference Card is absent, and add a T409/T411 test on a te-style fixture.

- severity: minor
  where: T410 behavior 10 (trace cells base_status rendered as plain "str" data_el); T411 L4b; LINT_PARAMS
    verdict_containers
  problem: |
    A trace row with base_status "abstain" puts the word "abstain" in visible text outside every verdict container.
    L4b then counts it, and a correct retrain report fails lint (C15, gating) whenever any eval row abstains. This is
    rare on a success, plausible on a FINDING. T410 test_trace_flags_rendered exercises exactly this row, but
    test_no_stray_verdict_words does not.
  fix: Render base_status cells (and the baseline reference status) with cls "attrib-status", and add that bundle to
    test_no_stray_verdict_words.

- severity: minor
  where: T417 behavior 5b (a crash rerun needs no refreeze); T416 exit_status (crash reruns do not qualify the status);
    T407 behavior 4 (client death -> Aborted, no result)
  problem: |
    The live "retrain eval step" notes stream to the operator. Killing the client after watching them is recorded as a
    crash, and up to 2 reruns are allowed with a plain "PASS" status (the reruns appear only in the notes). The fixed
    seeds limit what this gains, but the status line can hide a discarded draw.
  fix: Qualify the status ("PASS after crash rerun (...)") whenever a crash rerun happened after COMPUTE started, or
    classify a client-side abort during COMPUTE as non-crash.

- severity: minor
  where: T417 interface (--s2-model on read llm); T412 llm_read ("unless overridden"); T416 check_reader (C16 checks
    served_model non-null and the host only)
  problem: The frozen READER_PARAMS.llm_model can be replaced at E6 by a flag, and C16 does not compare the requested
    or served model with the frozen value. The measuring instrument is then chosen after the freeze.
  fix: Remove --s2-model from `read`, or make C16 require requested_model == READER_PARAMS["llm_model"] (a mismatch
    is a FAIL that needs a refreeze).

- severity: minor
  where: T412 make_canary in scout/report/reader.py; T417 behaviors 6 and 7 (reports/p5/canary.html, reader/ copies);
    invariant 5
  problem: |
    Product code (the scout package) produces a scout-branded report that deliberately drops the distillation
    disclaimer and asserts a licence permission. The file looks like a real report: it carries an intact bundle and a
    regenerate command, it persists under reports/, and nothing marks it.
  fix: Move make_canary to scripts/ (exit-only), and have `exit` delete reports/p5/canary.html and reports/p5/reader/
    after recording their sha256 values. Or at least add a non-visible marker in a meta tag that the reader text
    extraction drops.

- severity: minor
  where: plan 3.8 Readers (human); T412 neutral_names
  problem: The human gets two near-identical files for the same subject (retrain and canary). Comparing them reveals
    the canary, so C17 measures diffing rather than reading.
  fix: State this limit in plan 3.8, or give the human the canary in a separate session after the three reports are
    locked in the ledger.

- severity: minor
  where: T416 behavior 3 (C2 requires a refreeze_reason only after P5_ATTEMPT or P5_REPORTS) vs T417 behavior 2 and
    T416 behavior 15 (also after P5_READER)
  problem: The rule is inconsistent: a refreeze after a P5_READER with no reason would pass C2.
  fix: Add P5_READER to the C2 clause.

- severity: minor
  where: T408 behavior 3 (data_license reads README.md via source.read_file on the client) vs behavior 10 and
    test_plan_only ("no data file body read (Source.read_file spy)")
  problem: The spy as written would trip on the README read.
  fix: Scope the spy to the two data file paths.

- severity: minor
  where: T409, used by the T410-T413 acceptance tests ("fixtures from the T409 test fixtures"); plan 11 deferred minor
  problem: The trigger condition of the deferred minor is already met: four later tasks import fixtures from another
    task's test module, which none of them owns.
  fix: Add tests/helpers/report_fixtures.py to T409's files now.

- severity: minor
  where: T404 test_runtime (<= 180 s; planner measured 176.2 s in v6.log)
  problem: |
    A 2 % margin on a wall-clock test is flaky on loaded CI. The test also rebuilds the family a second time on a fresh
    directory.
  fix: Bound it at 2x the measurement or mark it slow, and reuse the session fixture's timing.

- severity: minor
  where: plan 2.1 report-targets table ("System 1 analysis vs the P3 library ... P3 Cards (imported)"); T409
    select_cards ("analysis = newest with has.sigma or has.anchors")
  problem: |
    After E4, the newest sigma-bearing Card of the subject is the card.v4. It copies the P3 stats, but its fetch_log
    is the retrain client's log, not the σ-sample reads. So the System 1 input is not the P3 Card the table names.
  fix: State that analysis uses the v4 Card (stats identical to its parent), or prefer non-v4 Cards for the analysis
    role.

- severity: minor
  where: T412 behavior 5 D1 ("the first <data> in #summary ending /n_attributed")
  problem: The plan says the base's count, but the first summary slot is the base only if the summary lists the role
    "reference" first.
  fix: Select the attrib_ref summary item whose role is "reference".

- severity: minor
  where: plan 3.6 / T409 behavior 4, T410 behavior 8, T411 L2c, T413 GET /api/report (simplicity)
  problem: The System 2 report section, L2c and the report route have no exit row, so they could be cut without
    failing the exit.
  fix: Decide at CHECKPOINT, as noted in plan 11. Cutting S2 from reports removes L2c and the s2/s2in source kinds.

- severity: minor
  where: later.md "Black-box behavioural signal" ("reader Q6")
  problem: The reference is stale. The distillation question is Q9.
  fix: Change it to Q9.

summary: 0 blockers, 5 majors, 13 minors.
