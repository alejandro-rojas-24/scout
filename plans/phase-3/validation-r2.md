# Phase 3 validation (round 2)

verdict: REVISE

Scope read: project-context.md, phases.md "Phase 3", plans/phase-3/plan.md, tasks T201-T217, later.md; P1/P2 plan +
tasks only as contracts. Environment limits (no huggingface.co, no ANTHROPIC_API_KEY, gated repos) are not counted as
findings. The exit numbers were delegated, so they are not findings either.

Verified independently:
- Debiased CKA (numpy, /tmp): 20 independent 256x2048 pairs give max |CKA| 0.008; plain CKA min 0.888; a random
  orthogonal Q gives 1.0; 3.7x scale gives 1.0; a 2 % fine-tune gives 0.9996; eps 1.0 gives 0.496; the T208 semantic
  fixture (seeds 11-19, hidden 128) gives 0.41-0.44 between independent families. A per-dimension rescale gives 0.46, which
  matches the limitation the plan documents. The T205 and T208 thresholds are consistent with these numbers.
- Anthropic SDK (`pip download`): 1.9.0 is the latest. Every 1.x requires `httpx2<3,>=2.0.0`. `claude-opus-5-5` first
  appears in 1.8.0 (absent in 1.7.0). `server-side-fallback-2026-07-01`, beta `fallbacks: Union[Iterable[...],
  Literal["default"]]`, `ThinkingConfigAdaptiveParam` and `OutputConfigParam{effort, format}` are present from 1.0.0. The
  non-beta `messages.create` has no `fallbacks`. `APIResponse.http_request` returns `http_response.request`. Passing an
  `httpx.Client` is rejected by an explicit check. `stream` defaults to `omit`, so it is not added to the body. §3.4.1 is
  accurate.
- OOF feasibility (§3.3): verified by hand. Only L15 and L17 are skipped, with or without L10. Sweep: 6·5+6·4+6·2+5·4+5·2+4·2
  = 104. Held-out counts: 10/9 with 7 hard. There are 40 labeled repos and 46 targets.

findings:

- severity: major
  where: plan §2.1 step 6 ("Held-out evaluation, once"), §2.3 C4, T214 behavior 2, T216 behavior 6, T217 behavior 1
  problem: >
    Nothing enforces a single held-out evaluation. Neither p3_freeze nor exit_check_p3 writes to log.jsonl: "the operator
    appends" every FREEZE and EXIT line. C4 uses the LAST FREEZE-model line, and it checks code only against that line's
    commit. The following loop therefore passes every check: E5 → E6 FAIL (nothing is appended) → edit feature, alignment
    or JEV code, which C5 literals do not cover → commit → re-train → new FREEZE-model line → E6 PASS. All held-out Cards
    exist from E4 on, and `scout analyze` or exit_check_p3 can score them at any time. So the held-out set can act as a
    tuning set. This undermines the claim that thresholds and code are fixed independently of held-out data. The
    orchestrator's REVIEW/EXIT loop is exactly the actor that would "fix and re-run" after a failure.
  fix: >
    Make exit_check_p3 append its own EXIT (or ATTEMPT) line to log.jsonl before it prints the verdict, with the model sha
    and the git commit. Add C4 rows that FAIL when (a) there is more than one FREEZE-model line after the last FREEZE-labeled
    line, unless each extra line carries a recorded human reason; (b) any earlier EXIT/ATTEMPT line names a different model
    sha; (c) CODE_PATHS differ between the FREEZE-labeled git_commit and the FREEZE-model git_commit. At minimum, report
    (c) and make it a CHECKPOINT item.

- severity: major
  where: T204 behavior 3 vs T207 behavior D' (also plan §9 row scout/sigma.py)
  problem: >
    compute_with_stats copies P2 compute_all's post-loop guard ("resident_bytes != 0 -> SigmaError('buffers not empty
    after COMPUTE')", P2 T107 behavior 4). T207 D' calls compute_with_stats(buffers, sigma_specs) FIRST, while the anchor
    reads are still resident in the same SampleBuffers, and pops the anchor buffers only afterwards. As specified, every
    scan with anchors=True raises SigmaError and clears the buffers. That includes anchor-only plans (deepseek-llm-7b-base,
    DeepSeek-V3-Base), where sigma_specs is empty. T207 test_anchors_approved_hub and the rehearsal cannot pass as written,
    and both tasks are marked spec_complete.
  fix: >
    Choose one and state it in both tasks. (a) T207 decodes and releases the anchor reads before calling
    compute_with_stats. (b) compute_with_stats checks only that every name in specs was popped (or takes `expect_empty:
    bool`), and T207 runs the final resident==0 check after the anchor loop. Add a T207 test that covers sigma+anchor and
    anchor-only plans.

- severity: major
  where: T202 (wave 1) vs T216 (wave 9); plan §9 row exit_expectations_p2 C8; §10 waves
  problem: >
    T202 switches SCHEMA_VERSION to "card.v2", so every Card built from wave 1 on is v2. P2 T115 C8 requires
    `schema_version == "card.v1"`, and P2 T116's offline rehearsal (tests/test_exit_check_p2_offline.py
    test_rehearsal_pass) builds real Cards through scan. It therefore FAILs from T202 until T216 changes C8 in wave 9. The
    `pytest -q` acceptance command of T202-T215 cannot be green. Each implementer must either edit
    scripts/exit_expectations_p2.py, which is outside their file list (and T216 owns it), or leave the suite red.
    tests/test_scan_p2.py::test_plain_unchanged ("options both False") has the same problem between T202 and T207, and
    T207 owns that file.
  fix: >
    Move the P2 C8 amendment, and its tests/test_exit_expectations_p2.py case, into T202's files (or into a wave-1 task that
    T202 depends on). Have T202 also make the test_plain_unchanged option assertion version-tolerant, or give T202 that
    edit.

- severity: major
  where: plan §3.3 "Width shortcut", §3.5 sizes, §5 train/calibration negatives, T201 hard flag, C2
  problem: >
    "Hard" means equal hidden_size, but the shortcut-prone features (log_fro_dlog, log_topk_dist, and σ-curve length) are
    per-tensor and depend on k/v SHAPE, not on hidden_size. Every train and calibration hard negative except L10 has
    different k/v shapes: TinyLlama k 256x2048 vs StableLM-2 2048x2048 (L11, L12, L16, L18), and Qwen3-1.7B 1024x2048 vs
    OLMo-2-1B 2048x2048 (L23-L25). They also differ in depth and norm type, so they are not "same architecture trained
    independently" (phases.md). The only identical-shape train negative is L10, which the plan itself expects may be
    dropped. The held-out set, by contrast, has 5 identical-shape independent pairs (L37, L38, L39, L40, L41). A model can
    therefore learn "identical k/v shapes ⇒ derived", and the pre-held-out stop (pool_hard, exit 4) cannot see it, because
    the pool contains no such negative. The claim that the three splits "have the same negative mix" is false on the axis
    that matters. The likely outcome is a C6 failure that burns the one-shot held-out set.
  fix: >
    Define hard as identical k/v shapes and equal depth (computed from Cards in C2), or add that stricter flag. Require at
    least 2 such negatives in train and at least 1 in calibration. For example, move the smollm2+dscoder contrast (or
    another identical-shape Llama-architecture pair; the planner verifies it) into calibration and re-balance the held-out
    set. Report n_pool_neg_sameshape in pool_hard.

- severity: minor
  where: plan §2.4 ("room to drop up to 3 entries"), §5 held-out families
  problem: >
    The not_derived margin is 1 (9 vs ≥ 8), not 3. Dropping the V-marked deepseek-coder-1.3b-base removes L38 and L44
    (not_derived 7, so C2 FAILs). Every mistral7b negative also depends on a verbatim "from-scratch" quote. The Mistral 7B
    paper (2310.06825) may not contain an explicit from-scratch statement, and losing that family leaves 4 negatives.
  fix: Correct the claim. Name pre-verified fallback held-out negatives in §5, or lower the dependence on single families.

- severity: minor
  where: plan §5 (L30), phases.md "including merges"
  problem: >
    The set has exactly one merge pair (L30). It is held-out only and V-marked on both repos. Train and calibration have no
    merge. If the Slerp or OpenHermes safetensors check fails, the labeled set contains no merge and the roadmap item is
    uncovered.
  fix: >
    Add a documented mergekit merge to train or calibration, and name a fallback merge for the held-out set.

- severity: minor
  where: T213 behavior 3 (SCAN line), C3, C4
  problem: >
    Card contents are not hashed. Held-out Cards live in a gitignored directory and can be re-scanned or edited after
    FREEZE-model. C1 accepts any later scanned_at, and C3 checks only structure.
  fix: >
    Record the sha256 of every Card JSON and Parquet file in the SCAN-confirm line, and check the files at EXIT.

- severity: minor
  where: T209 behavior 1d; plan §3.3 "Alignment"
  problem: >
    Identical block structure with unequal depth (SOLAR vs Mistral, fc/up, fg/up) gives an all-0 S, which stays on Viterbi
    over a flat cost. The alignment is therefore a tie-break unrelated to the true layer mapping, so the σ features of
    depth-upscaled pairs carry little information by construction. No test checks that an upscaled related pair's r_med
    differs from the null.
  fix: >
    State this in §3.3 and accept that these pairs rely on CKA (which the abstention budget covers), or add a T209
    assertion on the upscale pair's features.

- severity: minor
  where: plan §3.3 Calibration steps 1-4; T210 behavior 6
  problem: >
    The Platt pool mixes OOF logits from fold models trained on 7-16 examples with final-model logits. Their scales differ
    under λ = 1, so t_hi and t_lo are fitted on a mixture that does not match the final model's held-out logits. This is
    fail-safe, but it adds miscalibration risk on the one-shot run.
  fix: >
    Report the OOF and calibration pool separation separately in threshold_inputs, or fit Platt on the calibration split
    logits plus final-model refits only. At minimum, state the caveat.

- severity: minor
  where: T212 test_sdk_contract
  problem: >
    The only real-SDK test does not run assert_request_no_weights on the captured body. If the SDK adds a body key, that
    first shows up as a C12 FAIL at E7, after the cost has been paid.
  fix: >
    Add `assert_request_no_weights(handler_body)` to test_sdk_contract, for both routes.

- severity: minor
  where: plan §2.1 "Phase status rule"
  problem: >
    The plan says scans without anchors are "byte-identical to P2 apart from the Card version string". T207 D' also fills
    tensor_stats, spectral_topk and the stat_mean/std/fro/topk columns on every sample scan. Network bytes are identical;
    Card content is not.
  fix: >
    Reword the claim to "network reads and plan_id identical", and keep the T207 test.

- severity: minor
  where: plan §2.5 gate paragraph
  problem: >
    The plan says "Qwen2.5-1.5B-sized repos auto-approve below the threshold". They plan about 42.8 MiB, plus 16 MiB retry
    slack, plus about 7 MB of tokenizer meta, which exceeds 64 MiB, so they need a cli-flag confirmation. This is harmless,
    because the CONFIRM WITH line lists them.
  fix: Correct the sentence.

- severity: minor
  where: T216 static_import_violations (C10)
  problem: >
    The AST check covers direct imports only. A transitive import of scout.hub through scout.suggest, align or layercorr
    would pass. The runtime guard covers S1 traffic, so the risk is small.
  fix: >
    Also walk the scout.* modules those four modules import, one level deep or transitively.

summary: 0 blockers, 4 majors, 9 minors.
