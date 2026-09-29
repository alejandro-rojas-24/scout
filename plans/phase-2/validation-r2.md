# Phase 2 validation (round 2, fresh context)

verdict: REVISE

Scope read: project-context.md, phases.md §Phase 2, plans/phase-2/plan.md (r1), tasks T101–T116, later.md, and the
Phase 1 plan and tasks as the contract. The environment limits (no huggingface.co here, gated Llama) are not findings.

Checks that came out clean (so the planner does not re-open them):
- Byte arithmetic in §2.2 and §3.3: 205,520,896 / 222,298,112 (Qwen), 536,870,912 / 553,648,128 (Llama), total cap
  998,244,352, memory peak 570,425,344 B (544 MiB). All correct. C2's kv_dim formula gives 512 and 1024.
- MinHash overflow: (2^32−1)^2 + 2^32−1 < 2^64. There is no overflow, and SE ≤ 0.031 holds.
- The fixture null replicates. My independent numpy reimplementation (/tmp/p2val/sim.py) of the T103 generator and
  the T109 statistic at exit geometry (28 qwen2-style vs 32 llama-style, reldepth, 200 pairs) gives median mean −0.005,
  SD 0.118, q95 0.171, max 0.351, z_shift max 2.95 (0/200 > 3), top-1 max 0.12, PR 3.77. That matches §2.3.
- Upscaled and pruned r on the true mapping after per-side cubic detrend (10 seeds): median 0.96–0.99. The
  T111 `test_upscaled`/`test_pruned` expectations are reachable.
- Gate against invariant 2: the plan is logged and displayed before any decision on every path. Approval needs the
  exact (plan_id, bytes_cap) pair in the CLI, API and runner. ByteLog enforces the grant (stage, pure-weight range,
  inside the grant, cap incl. reservations), and retries are rechecked. The wire is checked independently (C1, C5,
  C6). Plan ids are deterministic, so E2's display binds E3's approval.
- Purge against invariant 1: bytes live only in `SampleBuffers`. Each buffer is popped before its SVD, and every
  exception path clears the buffers. The COMPUTE purge event precedes REPORT. PURGE re-checks the live object and rolls
  back Cards if anything is resident. Only derived σ reaches the Card.
- Phase 1 references are accurate: ByteLog/`WeightReadRefused`/`note` set (T002), `_get` retry loop (T005),
  `TensorInfo`/`SafetensorsHeader` fields (T006), `Stack.indices`/sort order (T008), `card_paths`/`_model` slug and
  `stat_sigma_curve list<float64>` (T009), scan steps and stage order (T010), `DISCLAIMERS` (T011), `make_server` kwargs
  (T013), weight counter classes (T014), `CountingTransport`/`_client_for`/`orig_path` via Location (T015), and
  `Check`/`HF_ENDPOINT_URL`/`ALLOWED_HOSTS`/`host_allowed` (T016). `moe_tensors` accepts `kv_heads`/`head_dim`.

findings:

- severity: major
  where: plan §2.3 "Unrelated pair, absolute (Y = 0.40)" and "relative (z_shift ≤ 3.0)"; §3.4 step 3; D20; T103
    family_weights; T109 test_independent_null / test_null_geometry_rehearsal; exit C14
  problem: >
    The null evidence behind C14 comes from a fixture in which every layer's spectral parameters are iid around a
    smooth depth trend. Real decoder LLMs are not like that. Their first and last blocks are strongly anomalous, and
    in all of them the anomaly points the same way. That is shared, depth-localised structure with no lineage in it.
    A cubic least-squares detrend does not remove an endpoint spike. Through the endpoint leverage (h_00 ≈ 0.57 for
    a cubic at L = 28), it spreads the spike into the residual of every layer as the same depth pattern p(t) in both
    models. Under reldepth pairing, that pattern adds a positive term to every aligned R[i, a(i)]. It adds a
    mixed-sign term to the cyclically shifted pairings. So both the absolute median and z_shift rise.
    I measured this with the same reimplementation (exit geometry, 100 pairs each, anomaly = the same Δα added to
    the power-law exponent of the k/v spectra of the listed layers in both models):
      - baseline: median > 0.40 in 0 %, z > 3 in 0 %
      - layer 0 only, Δα = 1.0: 2 % and 4 %
      - first and last layer, Δα = 0.6: 5 % and 9 %
      - first and last layer, Δα = 1.0: 27 % and 12 % (median mean 0.33; PR falls to 1.8)
    The same pairs with blocks 0 and L−1 excluded before the detrend go back to baseline: 0 %, 0 %, max 0.33.
    The offline suite and the rehearsal cannot reveal this, because the fixture has no endpoint anomaly. The plan
    freezes the statistic before the real run and forbids changing it afterwards (§2.1). A C14 FAIL from this
    mechanism would therefore stall the phase, and it has nothing to do with lineage. §2.3 names a generic "PR near
    2" risk but not this mechanism, which is cheap to address now. The d_eff derivation also treats layers as
    independent (SD/√n). Real residuals are autocorrelated over depth, so the effective n is smaller than L. That
    makes the "3.2 formula-SD" margin optimistic too.
  fix: >
    Before freezing: (1) Add to T103 a frozen fixture option for shared depth-localised anomalies, for example a
    fixed Δα on blocks 0 and L−1 applied to every family. Add T109 null tests that use it at L = 16 and at exit
    geometry. (2) Make the statistic robust to it, and record the choice in D20 and SIGMA_PARAMS. Options are to
    exclude the first and last block from the detrend fit and from the median and in-run nulls (e.g.
    `"edge_blocks_excluded": 1`), or to use a robust (IRLS/Huber) depth detrend. Re-measure and restate the §2.3
    table. (3) State in §2.3 that the √n in the median-SD formula assumes depth-independent residuals.

- severity: minor
  where: plan §2.3 "Related pair, median (X = 0.80)" and "lower tail (q10 ≥ 0.50)"
  problem: >
    The fine-tune model is base = x and instruct = x + e, so only one side is perturbed. Its correlation is
    r = s/√(s² + ε²), not s²/(s² + ε²). (The latter holds when both sides are independent noisy copies.) X = 0.80
    therefore tolerates ε ≤ 0.75 s, not 0.5 s, and q10 = 0.50 tolerates ε ≤ 1.73 s, not ε = s. The thresholds are
    more lenient than the text claims. Also, with numpy's linear quantile over 28 values, q10 sits between the 3rd
    and 4th smallest, so "at most 2 of 28" is really about 3.
  fix: Correct the formula and the implied tolerances in §2.3. Either keep X and q10 with the corrected rationale, or
    restate them. Decide this before the freeze.

- severity: minor
  where: plan §3.2 flow step 2; T106 run_gate step 2; T101 (separate weight and non-weight accounting)
  problem: >
    Invariant 2 speaks of "the configured threshold". The plan has two independent budgets: the P1 non-weight budget
    (64 MiB, refuse-only) and the weight gate threshold (64 MiB, auto-approve below it). One scan can therefore
    fetch about 128 MiB with no human confirmation. That may be intended, but the plan never states it.
  fix: >
    Either compare `bytes_cap + totals.meta + totals.header` against the gate threshold, or state explicitly in §3.2
    and D19 that there are two thresholds and why each alone satisfies invariant 2.

- severity: minor
  where: T108 behavior C ("submit, per read, lambda: buffers.put(r.tensor, source.read_range(r.path, ...))")
  problem: If this lambda is written literally inside a loop, it late-binds `r`, so every task would read the last
    planned range. The acceptance tests would catch it, but the spec text is wrong.
  fix: Specify `pool.submit(fetch_one, r)` or a default-argument binding.

- severity: minor
  where: T109 test_null_geometry_rehearsal ("z_shift <= 3.0 for all 20"); T116 test_rehearsal_pass (one frozen pair)
  problem: >
    The measured maxima of z_shift over 200 null pairs are 2.85 (planner) and 2.95 (mine), right at the bound. The
    rng stream of T103 differs from both reimplementations, and the seeds are frozen. An "all 20" requirement
    therefore has a small but real chance to fail by construction and stall on "goes to the human". The L = 16 test
    already uses 49/50.
  fix: Use "≥ 19 of 20" (or the q95 form) for the geometry test, so it matches the L = 16 test.

- severity: minor
  where: T111 behavior 4 (dp-structure+weights), T110 combine_cost, DiffView weight.median_r
  problem: >
    For structurally compatible pairs, the alignment is chosen by minimising a cost that includes (1 − R)/2. The same
    R is then used to report median_r on that alignment, which biases r upward (selection on the outcome). It does
    not affect the exit: the related pair is identity (C11), and the unrelated pair uses reldepth. It does matter for
    same-architecture unrelated models, which are the P3 hard negatives.
  fix: Also report, or note in the DiffView, the median on the structure-only alignment when the two differ. Or list
    the bias in later.md for P3 calibration.

- severity: minor
  where: plan §3.3 role patterns / SIGMA_PARAMS.role_patterns; §7 multi-component row; later.md
  problem: >
    T5-style encoders (`encoder.block.#.layer.0.SelfAttention.{k,v}.weight`) are the common second text encoder in
    diffusers pipelines (invariant 6, multi-component). They match no role and are skipped without an architecture
    hint. later.md does not list them either.
  fix: Add a "T5 attention" skip hint and a later.md entry, or add the `SelfAttention.k.weight` / `.v.weight`
    patterns before the freeze.

- severity: minor
  where: T115 depends_on [T111]; T115 imports scripts.exit_expectations (P1 T016)
  problem: T016 is not in T115's transitive dependencies (T111 → T011/T105/T108/T109/T110 do not reach T016). This is
    harmless only because Phase 1 is assumed complete.
  fix: Add T016 to T115.depends_on.

- severity: minor
  where: exit C9; T115 behavior 3 C9; P1 plan §4.2 ("fetch_log holds the whole scan's log up to the REPORT
    stage_start event")
  problem: C9 compares the COMPUTE purge seq with the seq of the REPORT `stage_start` inside `card.fetch_log.events`.
    The P1 wording "up to" does not say whether that event is included. If it is not, C9 cannot find it and FAILs
    on a correct scan.
  fix: State in T102 (or T115) that the REPORT stage_start is included, or have C9 read the REPORT seq from the live
    log instead.

- severity: minor
  where: T112 `scout suggest` subcommand and `format_suggest` (SIMPLICITY)
  problem: The exit (C16/C17) uses the API route, and E5 uses the UI. The CLI suggest path is a third renderer that has
    to keep the disclaimers in sync, and no check needs it.
  fix: Cut `scout suggest` from P2, or keep it with the justification recorded.

summary: 0 blockers, 1 major, 9 minors.
