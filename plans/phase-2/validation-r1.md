# Phase 2 validation (validator round 1, 2026-09-28)

verdict: REVISE

Scope read: project-context.md, phases.md "Phase 2", plans/phase-2/{plan.md, tasks/T101-T116.yaml, later.md}, and plans/phase-1 plan + tasks, read only as the contract. The Phase 1 interfaces that Phase 2 references all exist with the stated signatures: ByteLog.preflight/record/release/set_header_len/note, the Stage enum incl. SAMPLED_READ/COMPUTE/PURGE, Source.read_file/read_range(path, start, length), HubSource._get attempt loop, CountingTransport records {orig_path, range, status, body_bytes}, _client_for, make_server(client_factory, endpoint), Check/HF_ENDPOINT_URL/ALLOWED_HOSTS/host_allowed, build_fixture_card, card_paths naming, DISCLAIMERS.
Environment limits (no huggingface.co, gated Llama) are not treated as findings. The arithmetic in §2.2 checks out: 205,520,896 / 536,870,912 / N = 998,244,352.

Numeric check behind F2/F3: I ran the T103 family generator (k/v spectra) through the T109 statistic in a scratch numpy venv, with 200 independent seed pairs, L=16, K=64 and a shared trend. Results: participation ratio of the detrended residuals ≈ 3.3. The null median r has mean -0.02, SD 0.18 and max 0.39. Without the detrend the mean is 0.92.

findings:

- severity: blocker
  where: plan §3.2 (Confirmers table, "CLI flag"), D23; T106 flag_confirmer; T112 behavior 1-2; T116 behavior 2 ("confirm" mode); plan §2.2 E3
  problem: >
    Invariant 2 requires explicit human confirmation "with bytes, disk and reason shown". `scout scan --sample
    --confirm-bytes N` approves any plan whose cap is <= N without printing the plan: format_plan is printed only on
    decline (T112 b.2), and flag_confirmer writes nothing (T106). So a blanket `--confirm-bytes 999999999999` reads
    weights from a plan nobody saw. The exit runner has the same gap at larger scale. E3 confirms each plan
    programmatically against a running byte sum from a different process (E2). Neither plan_id nor per-plan cap is
    bound, so a different plan with the same or a smaller total is approved without being shown.
  fix: >
    (a) run_gate always emits format_plan (to stderr for the CLI, and in the live log / plan field for the server)
    before the confirmer decides. (b) The flag approves only if bytes_confirmed == plan.bytes_cap exactly, so the
    number can only come from a displayed plan. Keep >= only for the interactive prompt/UI, where the plan is on
    screen. (c) E2 prints a plan digest (sha256 over sorted [plan_id, bytes_cap]). E3 takes `--confirm-plans
    <digest>` (or the list of plan_ids), and the runner confirms only plans whose plan_id is in it. C8 then asserts
    that the approved plan_id is in the E2 list. Add tests: a flag larger than the cap declines, a plan_id missing
    from the digest declines, and the plan text appears on stderr before any weight read.

- severity: major
  where: plan §2.3 C12/C14/C15 and "Why these thresholds"; later.md Phase 3 bullets "Hard negatives" and "An off-diagonal null in the DiffView"
  problem: >
    The exit cannot tell lineage from shared architecture. The related pair shares architecture (same shapes, same
    depth, identity alignment). The unrelated pair differs in shape (512x3584 vs 1024x4096), depth (28 vs 32) and
    block set, and is paired by the fixed reldepth mapping, which is the easiest possible negative. A statistic whose
    residual is driven by shape- or architecture-common structure, or by a depth profile that the cubic misses, would
    give C12 >= 0.80 and C14 <= 0.40 for reasons unrelated to derivation. The in-run control that would expose this
    is R off the diagonal for the related pair. That R is computed anyway, costs 0 bytes, and is already tested
    offline (T109 test_layer_specific). The plan explicitly defers it to later.md, and DiffView does not expose it.
  fix: >
    Bring the off-diagonal null into P2. DiffView.weight gains `offdiag_median_r` (the median of R[i,j] with
    j != a(i)) and `diag_top1_frac` (the fraction of subject layers whose argmax_j R[i,j] == a(i)). Add exit C18,
    frozen before any run: related `median_r - offdiag_median_r >= 0.40` and `diag_top1_frac >= 0.80`. If shape or
    architecture inflated r, every R[i,j] would rise and C18 would fail. Record both values in the EXIT evidence
    line. Keep the true hard negative (same architecture, independent training) in later.md for P3.

- severity: major
  where: plan §2.3 "Unrelated pair" (d_eff >= 5, null SD 0.118, "3.3 SD"); D20/D28; T109 test_independent_null; T116 rehearsal thresholds
  problem: >
    The Y = 0.40 justification rests on "the residual curves are smooth along rank, so take a conservative
    effective dimension d_eff >= 5". Smoothness implies low d_eff, not high. After per-layer centring and the
    column-wise depth detrend, the residual is dominated by one or two modes: each layer's deviation of spectral
    slope from its depth trend, times the shared log-rank shape. So the per-layer null r is close to ±cos(angle) of a
    low-dimensional vector, not approximately N(0, 1/(d-1)). The plan's own fixture confirms this. The participation
    ratio is ≈ 3.3, and the null-median SD is 0.18 at L=16, with the worst of 200 seed pairs at 0.39. T109's
    "median <= 0.40 for EVERY seed pair" therefore sits on the edge. When it fails, the easy fix is to change fixture
    seeds, which is hand-tuning by another route. The null also assumes that residuals of independent models are
    independent at matched relative depth. No real unrelated pair is measured before the exit, so C14 is the first
    and only test of that assumption.
  fix: >
    Redo the derivation with a measured d_eff: report the participation ratio of the residuals in DiffView.weight
    and the evidence line. Either (a) also remove the dominant rank mode, i.e. detrend each layer's curve against
    log-rank (linear) before Pearson, and re-derive the null SD; or (b) make the unrelated criterion relative to the
    in-run null from the major finding above. For example, require the unrelated median_r to lie within the
    empirical off-diagonal spread (|median_r - offdiag_median_r| <= 2 x its bootstrap SD) in addition to <= 0.40.
    Freeze the fixture seeds and seed lists in T103/T109/T116 as part of "no hand-tuning" (C10-style literal test),
    and give T109 a margin that the fixtures demonstrably meet: <= 0.40 over >= 50 seed pairs, 95th percentile.

- severity: major
  where: T109 acceptance "test_sigma_set_from_card" vs plan §10 waves (T109 in wave 4, T108 in wave 5)
  problem: >
    T109's acceptance needs "a card+table produced by T108's scan", but T109 depends only on T107 and runs a wave
    before T108. The task cannot meet its acceptance when it is scheduled. It also reads Card v1 fields (T102)
    without that dependency edge.
  fix: >
    Build the card in that test with scout.card.build_card (T102), with sigma_curves and sigma_by_tensor from
    sigma_values of fixture arrays, and add T102 to T109.depends_on. Alternatively, move the test to T111, which
    already depends on T108.

- severity: minor
  where: T101 behavior 9 / files
  problem: T101 says it may edit tests/test_bytelog.py and tests/test_hub_source.py (the ": no grant" message suffix), but neither file is in its `files` list.
  fix: Add both files to T101.files, or keep the P1 message text unchanged and put the reason in a separate field.

- severity: minor
  where: plan §3.1 PURGE, T108 step G, C9
  problem: The PURGE event "resident=0 verified" checks nothing. It is a second copy of a note already made. The only real evidence is SampleBuffers.resident_bytes at the end of COMPUTE.
  fix: Either assert that resident_bytes == 0 and that the buffers object was released (for example with a weakref) before emitting it, or drop the "verified" wording.

- severity: minor
  where: plan §2.3 C12 vs phases.md "high per-layer correlation"
  problem: Only the median is checked, so up to half the layers (for example an entire broken role or a broken depth band) could be low and C12 would still pass.
  fix: Add a frozen lower-quantile condition to C12, for example the 10th-percentile r >= 0.50, or at most 2 layers below 0.50.

- severity: minor
  where: plan §3.2 / D19 RETRY_SLACK_BYTES = 16 MiB per plan
  problem: On about 1 GB of 8 MiB whole-tensor reads, two mid-tensor resets exhaust the slack and fail the whole target (WeightReadRefused, no Card). That makes E3 flaky under network faults.
  fix: Scale the slack with the plan (for example max(16 MiB, 2 x the largest read)), or accept the flakiness and state that E3 is rerun as a whole with a new confirmation.

- severity: minor
  where: plan §7 inputs table, §3.3 role patterns
  problem: MLA attention (DeepSeek-style kv_a_proj_with_mqa / kv_b_proj) and fused QKV silently fall out as "role not found". Fused QKV is listed in later.md, but MLA is not.
  fix: Add MLA to later.md, and make sure the note text names the architecture reason.

- severity: minor
  where: T105 suggest output, T112 format_suggest, T113 GET /api/suggest, T114 suggestions list
  problem: The suggestion ranking shows tokenizer J and config score, but carries only 1 of the 3 disclaimers (CLI) or none (API/UI). Invariant 5 applies to reports, and a ranked "base candidate" list reads like one.
  fix: Add `disclaimers: DISCLAIMERS` to the suggest() result and render it wherever suggestions are shown.

- severity: minor
  where: plan §10 wave 2 {T102, T110}
  problem: T110 test_block_sets_from_parquet builds a card through build_fixture_card, which goes through scout/card.py while T102 is changing that file in parallel. There is no file conflict, but the test results depend on T102's state at that moment.
  fix: Build the table in T110's tests directly with pyarrow, or add T102 to T110.depends_on.

- severity: minor
  where: simplicity: T106 prompt_confirmer / T112 TTY path; T105 unscanned_claims; T110 kind "rearranged"
  problem: None of these is needed by the exit check or the roadmap line. The TTY prompt is a third confirmation path that must stay consistent with the other two.
  fix: Consider cutting the TTY prompt (the flag with exact cap, as in the blocker fix, plus the UI is enough) and unscanned_claims. Keep "rearranged" only if it costs no extra test surface.
