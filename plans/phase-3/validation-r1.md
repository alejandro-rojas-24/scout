# Phase 3 validation (round 1)

verdict: REVISE

Scope read: project-context.md, phases.md §Phase 3, plans/phase-3/{plan.md, tasks/T201–T217.yaml, later.md}, and the
P1/P2 plan.md + task specs as contracts only. No validation*.md or log.jsonl was opened. Environment limits (no Hub
access here, gated repos, no API key) are not findings.

## Verified (no finding)
- Anchor list (T203): 256 distinct words, "the" … "expect"; the sha256 of `"\n".join(words)+"\n"` is
  `466fcb0770645851483e61a7c7a72041d80c7155a2f0e24530c2a1099c421516`, which matches §4.8, T203, T205 and T216.
- Debiased CKA (§3.2, T205), re-implemented in numpy from the T205 formula:
  - Independent 256×2048 over the T205 seeds: |CKA| max 0.008 over 20 pairs. Plain CKA min is 0.888, mean 0.889.
  - Plain CKA at d=128 is 0.334.
  - Rotation, permutation with sign flips, and 3.7× scale all give 1.000000.
  - eps 0.02 gives 0.9996. eps 1.0 gives 0.496 (d=128) and 0.502 (d=2048).
  - The estimator is also invariant to per-model mean offsets (anisotropy): an independent pair gives −0.0024 both with and without large offsets.
  - T208 semantic baseline: 0.41–0.45 at hidden 128 and 0.46–0.50 at 160. The T208 band [0.30, 0.60] holds.
- Invariant 4: every verdict feature is orthogonally invariant.
  - σ-curve r, spectral top-k/‖W‖_F and ‖W‖_F come from singular values.
  - CKA uses a Gram over token-matched rows.
  - mean/std are descriptive only, and there is no raw cosine anywhere.
- Counts: 40 labeled + 6 library-only = 46 targets. 45 pairs (train 9/9, cal 4/4, held-out 10/9, 7 hard).
  The held-out sweep has 6·5+6·4+6·2+5·4+5·2+4·2 = 104 pairs. Every hard flag agrees with the published `hidden_size`. The §2.5 byte table matches the P2 σ-sample rule for every group I recomputed.
- SDK surface (checked against the bundled claude-api reference, 2026-09):
  - `claude-opus-5-5` and `claude-sonnet-5` exist.
  - `thinking={"type":"adaptive"}` is valid, and on Opus 5.5 it is required (thinking cannot be disabled).
  - `output_config={"effort":..., "format":{"type":"json_schema",...}}` is valid.
  - `client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default")` is the documented scalar form.
  - `response.model` names the model that served the response. `stop_reason == "refusal"` is HTTP 200.
  - The plan sets effort "high" explicitly. That is correct, because the Opus 5.5 default is "medium".
- Labels checked against my knowledge of the org documentation. Correct: L03, L04, L05, L08, L27–L29, L32, L34–L36, the Qwen2.5-Math/Coder continued pretraining, and SOLAR from Mistral weights. No derived/not_derived label looks factually inverted. See B1 and m1 for L10.
- Leakage controls: family-disjoint splits; pairs confined to their split; T214 never loads held-out Cards; C4 string scan of the model file; every constant is frozen before data. No threshold is fit on held-out data.

## findings

```yaml
findings:
  - severity: blocker
    where: plan.md §5 families table (line 604) + L10 (line 645); T201 behavior 2g/5 and test_draft_file_valid
    problem: >
      TinyLlama/TinyLlama_v1.1 is in family `tinyllama`, the same family as TinyLlama-1431k-3T. L10 is a not_derived pair
      inside one family. T201 rule g ("label not_derived -> families differ") rejects it, so the draft labeled_v1.yaml
      cannot pass validate(). As a result, test_draft_file_valid fails, E3 FREEZE can never run, and the phase cannot
      close. The T201 note "evidence = independence evidence of both families" is also undefined for L10. An implementer
      would have to redesign the family structure on their own, which changes the OOF folds (D38).
    fix: >
      Put TinyLlama_v1.1 in its own train family (e.g. `tinyllama11`, independence evidence = the v1.1 model card).
      Update the "10 families" count in T201 and plan §5. Alternatively, drop L10 and replace it with another hard
      train negative. Re-derive the OOF fold feasibility for the new family layout.

  - severity: major
    where: plan.md §2.3 C12; T216 behavior 12; T217 run_s2 and test_rehearsal_s2_unavailable; T212 behavior 5b-c
    problem: >
      E7 can PASS while System 2 never produced an answer. Any exception, refusal, malformed output, or JSON truncated
      at max_tokens marks the call "unavailable". Final verdicts then equal S1, so the S2 metrics equal the S1 metrics,
      and C12 only requires s2_hosts to be non-empty "unless every out is unavailable". A missing or invalid key, a wrong
      model id, or an SDK incompatibility (see the next finding) therefore yields "EXIT CHECK (S2): PASS". The roadmap
      item "Reasoning model as System 2" is not falsified by its own exit.
    fix: >
      At E7, C12 must require available == true and a non-null served_model for every held-out pair, or state an
      explicit maximum number of unavailable pairs. An unavailable pair counts as FAIL, not as a pass-through. Add a
      distinct unavailable_reason for stop_reason "max_tokens". Change test_rehearsal_s2_unavailable to expect a C12
      FAIL. The offline `analyze --s2` degradation (exit 0) can stay as product behaviour.

  - severity: major
    where: T212 interface pyproject ("anthropic>=<installed version>") + acceptance test_anthropic_request_offline; T217 behavior 3 (run_s2 client); plan A4
    problem: >
      PyPI currently serves anthropic 1.9.0, and 1.x moved its HTTP layer from `httpx` to the fork `httpx2`. Passing an
      old `httpx.Client` as `http_client=` raises TypeError at construction. This affects both
      `httpx.Client(transport=httpx.MockTransport(...))` in T212's offline request test and
      `httpx.Client(event_hooks=...)` in T217's host recorder. Transports and event-hook request objects are `httpx2`
      types. With the lower bound set to "the installed version", the T212 acceptance test fails and E7 crashes, or,
      combined with the previous finding, falls into "unavailable". A4 covers only the `fallbacks` keyword.
    fix: >
      Specify the SDK major version. For 1.x: build the test client with `httpx2.Client(transport=httpx2.MockTransport(h))`
      (or `anthropic.DefaultHttpxClient`), put the host/body recorder on an httpx2 client, and declare `httpx2>=2.0`.
      Alternatively pin `anthropic<1` with a stated reason. Keep the S1 network guard on `httpx` (the Hub path), and note
      that it does not see SDK traffic.

  - severity: major
    where: T214 interface (p3_common.load_pair_cards -> dict[str, LoadedCard]); plan §10 depends_on and wave 7
    problem: >
      `LoadedCard` is defined in scout/analysis.py (T211). T214 depends only on T210 and T213, and it is scheduled in the
      same parallel wave as T211 ({T211, T214}). The T214 implementer cannot import the type it returns.
    fix: >
      Add T211 to T214.depends_on and move T214 to wave 8, together with T212 (no shared files). Alternatively, move
      LoadedCard into a module that T214 already depends on.

  - severity: major
    where: plan.md §5 train/calibration not_derived tables (lines 641–671), §3.3 thresholds, §2.4
    problem: >
      The training distribution differs from the evaluation distribution on exactly the axis the exit tests.
      - Only 2/9 train negatives and 2/4 calibration negatives are same-width (hard). Held-out has 7/9 hard, and C9
        sweeps the hard contrasts (Mistral/SOLAR vs Llama, SmolLM2 vs deepseek-coder).
      - Every derived train pair is same-width. log_fro_dlog, log_topk_dist and r_gap partly encode "same shapes", so
        λ=1 logistic regression on 18 pairs can learn width-equality as a lineage shortcut.
      - t_hi is set from a pool dominated by easy cross-width negatives, so it will likely sit at the 0.80 floor.
      The likely outcome is a false "derived" on a hard held-out negative. That fails C6/C9 and burns the one-shot
      held-out set (§2.1 step 6), so the phase misses its goal for a foreseeable reason.
    fix: >
      Before FREEZE, add same-hidden negatives to train and calibration so that at least half of each split's negatives
      are hard. Train has 2048-wide TinyLlama×3 and stablelm-2×2 (up to 6 cross pairs). Calibration has Qwen3-1.7B(-Base)
      × OLMo-2-1B variants (up to 6). Report pool false-derived on the hard subset in the FREEZE-model line.

  - severity: minor
    where: plan.md §5 L10; §3.1 verdict semantics
    problem: >
      "Independent re-run" within one org may reuse the same initialisation seed. Under §3.1 ("both from a common
      checkpoint"), a shared random init is arguably shared lineage, and the documentation is unlikely to state the seed.
    fix: The L10 verify step should require evidence that the init differs. If there is none, drop the pair as undocumented (§3.1 rule).

  - severity: minor
    where: plan.md §2.1 step 5–6; T216 C4; §4.9 EXIT line
    problem: >
      Held-out Cards exist before training (E4 precedes E5). The "no peeking" guarantee therefore rests on the frozen
      model sha only. Feature or JEV code could change between FREEZE-model and EXIT: git_commit and git_dirty are
      recorded but never compared.
    fix: Make C4 require EXIT git_commit == FREEZE-model git_commit and git_dirty == false. Alternatively, scan the held-out repos after FREEZE-model and check their scanned_at.

  - severity: minor
    where: T216 C3; plan §3.6 (DeepSeek-V3-Base "verify")
    problem: >
      C3 requires exactly one approved gate decision per repo. A repo whose plan has zero reads (P2 skips the gate) has
      none, so the exit hard-fails with no documented remedy. Example: MLA plus a non-BF16 embedding, which leaves no σ
      and no anchors.
    fix: E2 must flag zero-read plans and force a replacement decision before FREEZE. Alternatively, C3 accepts "no decision" when plan.reads is empty and the anchor-skip note is present.

  - severity: minor
    where: T209 behavior 1d
    problem: >
      Only an all-1 S triggers reldepth. A constant all-0.5 S (same role names, different widths or depths, e.g. Mistral
      vs SmolLM2) goes to Viterbi on a flat cost. The result is tie-break jumps, and the kind "pruned"/"upscaled" is
      reported to System 2 for unrelated models.
    fix: Use reldepth whenever S is constant, or label such alignments as structure-uninformative in the report and in s2in.

  - severity: minor
    where: plan.md §5 evidence tags for L06, L07, L19, L20, L31, L33 ([MC])
    problem: >
      These model-card bodies may not state the base checkpoint in text. The base_model field is hub_metadata, which is
      not allowed alone. The papers are the stronger source.
    fix: Cite Gemma 2/3, Qwen3 (2505.09388) and Llama 3 Herd as the pair evidence. Also mark the safetensors/tokenizer.json availability of teknium/OpenHermes-2.5-Mistral-7B and the stablelm-2-1_6b tokenizer.json as V.

  - severity: minor
    where: T215 behavior 1 (load_library), route sonnet
    problem: >
      "Newest directory = lexicographically largest name" picks the largest commit sha, not the pinned or newest
      revision. This is a hidden design decision in a sonnet task.
    fix: Resolve library Cards through the pins file, or fail when more than one revision directory exists.

  - severity: minor
    where: T212 CHEAP_MODEL; plan §2.5 E7 cost
    problem: >
      `claude-sonnet-5` is the previous generation, and the current cheaper Sonnet is `claude-sonnet-5-5`. The E7 cost
      estimate of "under $3" is optimistic: effort high allows up to 16k output per call at $20/MTok, so 19 calls can
      reach about $6.
    fix: Offer claude-sonnet-5-5 as the cheaper option (it also accepts fallbacks "default"). Restate the cost bound.

  - severity: minor
    where: phases.md Phase 3 exit ("exact numbers set by human") vs plan §2.4 / D45
    problem: The planner sets the exit numbers "under the human's delegation", but nothing in the plan records that delegation.
    fix: Add a human sign-off line for D45 values before FREEZE-labeled, and have C1 or C5 check it.

  - severity: minor
    where: T216 C12 / T217 run_s2
    problem: assert_no_weights is checked on a recomputed build_input(), not on the request body actually sent.
    fix: Capture the request body in the httpx2 event hook and run assert_no_weights on it.

  - severity: minor
    where: plan §3.4 vs T212 behavior 3
    problem: The §3.4 forbidden-key list omits "token_ids", which T212 includes.
    fix: Align the plan text with T212.

  - severity: minor
    where: T206/T207 acceptance (invariant 6)
    problem: >
      There is no anchor test for a multi-component repo (diffusers text_encoder/ + tokenizer/, the project's headline
      Qwen-text-encoder case) or for a sharded MoE embedding. The mapping from tokenizer to component embedding is
      untested.
    fix: Add one T207 fixture with a text_encoder component (Qwen-style embed name) plus tokenizer/tokenizer.json.

  - severity: minor
    where: plan §2.4 (C7 rationale)
    problem: >
      The correlation caveat is stated for C9 but not for C7. The 10 derived held-out pairs come from 4 families, and
      L27–L29 share one reference.
    fix: State the effective-n caveat for C7 as well.

  - severity: minor
    where: T217 test_rehearsal_leak
    problem: "Relabel a held-out pair as train" is undefined, because a pair's split derives from its families.
    fix: Specify moving one held-out family to train, then expect C2/C4 FAIL.

  - severity: minor
    where: T214 diagnostics (D49), EXIT "nulls"
    problem: Simplicity. No check consumes them.
    fix: They can be cut or deferred without affecting the exit.

  - severity: minor
    where: plan §4.3 / D36 (models/jev_v1.json committed)
    problem: Invariant 1 names the Card as the only persisted artifact. The model file is argued to be "configuration".
    fix: Have the human acknowledge D36 at the checkpoint, or amend the invariant wording.
```

Blockers: 1. Majors: 4. Minors: 15.
