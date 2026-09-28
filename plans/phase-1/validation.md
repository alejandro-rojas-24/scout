# Phase 1 validation

verdict: REVISE

Reviewed: project-context.md, phases.md (Phase 1), plans/phase-1/plan.md, tasks T001–T015, later.md (only for scope leakage).
Environment given: this container cannot reach huggingface.co (proxy 403). PyPI works.

Summary: the plan is detailed and mostly sound. Coverage of the roadmap line is complete and nothing from later phases leaked in. I re-derived the hard-coded Qwen3 constants and they are correct: 8B gives 8,190,735,360 params and 399 tensors; 30B-A3B gives 30,532,122,624 params, 18,867 tensors and 4,718,592 params per expert. The handling of the unreachable Hub keeps the exit check falsifiable: UNPINNED makes the script exit 2, and without network the script fails rather than skips.

Five majors remain:
- One exit threshold (C8) will very likely fail on the plan's own diffusion target.
- Invariant 5 is not met.
- "Renders correctly" is verified for only 1 of the 3 targets.
- The live "zero weight bytes" evidence comes entirely from the component under test.
- The read threshold can be exceeded by concurrent header reads.

## findings

### Majors

- severity: major
  where: plan.md §2 C8 (line 60); T011 behavior 2–3; T015 behavior 3 (C8); T001 behavior 8 (vae fixture)
  problem: |
    C8 requires `len(view.nodes) <= 100` for every Card. T011 creates a node for every path prefix, and that includes
    tensor leaves. T008 collapses only the first numeric segment, so nested indices stay literal.

    Qwen-Image's `vae` is AutoencoderKLQwenImage, a Wan-style 3D VAE:
    - encoder.down_blocks.N with norm1.gamma, conv1, conv2, conv_shortcut, resample.1, time_conv
    - mid_block.resnets.N and mid_block.attentions.0 in both encoder and decoder
    - decoder.up_blocks.N.resnets.{0,1,2}.*, where the inner index stays literal, so there are 3 subtrees per node
    - decoder.up_blocks.N.upsamplers.0.resample.1.*
    - quant_conv and post_quant_conv

    Counting by the T011 rules gives about 140 nodes (encoder ≈ 52, decoder ≈ 84, root and quant convs ≈ 7). The
    transformer comes out at ≈ 85, which is close to the limit.

    So a correct implementation most likely FAILs E3/C8, and the phase blocks on a human decision. The offline tests
    cannot catch this because the T001 pipeline vae fixture is tiny (5 tensors).
  fix: |
    Choose one of these before implementation:
    (a) Define C8 over non-`tensor` nodes. Tensor nodes are hidden by default in T014, so this matches what the user sees.
    (b) Keep leaves but raise or remove the limit for pipeline components.
    (c) Collapse repeated nested literal indices in the view.
    Then add a T001/T011 fixture that mirrors the Wan-VAE naming (up_blocks.N.resnets.M) and assert the chosen bound.

- severity: major
  where: invariant 5; plan.md §4.4 View; T014 behavior 3a (invariants_touched lists 5, but no step implements it); T012 scan output
  problem: |
    Invariant 5 says reports always state three things:
    - distillation is invisible to weight forensics
    - tokenizer reuse alone is not proof
    - licensing is a human decision

    The P1 page is the report surface. It shows the claimed `license` and `base_model` and has a REPORT stage. No task
    specifies these statements, and no test checks for them. T014 renders only a "claimed by model card" box.
  fix: |
    Add a fixed `disclaimers: [str, str, str]` constant to the view in §4.4 and T011, and render it in every view
    section in T014. Assert its presence in test_view and test_web_static. Optionally, also print the disclaimers to
    stderr from `scout scan`.

- severity: major
  where: plan.md §2 E5 (lines 115–121); C1 (line 53); T014 acceptance
  problem: |
    The exit requires that "Qwen3-8B, a Qwen3 MoE, and one diffusion pipeline each render correctly." C1 stops the clock
    when the API returns `views` JSON, so the browser is never in the loop. The only automated frontend tests are
    string greps on the static files.

    E5 visually confirms only Qwen3-30B-A3B. For Qwen3-8B and Qwen-Image, rendering is never checked, and Qwen-Image is
    the only multi-view (3 sections) case. A broken app.js for multiple views or for the pipeline case would still pass
    the exit.
  fix: |
    Extend E5 to all three pinned targets with a per-target checklist:
    - 8B: `layers ×36`, one strip of 36 cells
    - Qwen-Image: 3 sections, strips for text_encoder, transformer and vae; the vision `blocks` strip is present
    - all: disclaimers, and `weight 0 B` in green

    Record each confirmation in log.jsonl. The alternative is a headless check, which is heavier; the manual check is
    enough for a self-use tool.

- severity: major
  where: plan.md §2 C2/C3 (lines 54–55), §8 first risk row; T015 behavior 2–3
  problem: |
    On the live run, the "zero weight bytes" verdict and the header-byte equality come only from ByteLog. That is the
    instrumentation inside the code under test.

    Suppose a code path fetches without going through `_get` and `record()`. Examples: a stray `client.get`, or a
    future helper in scan.py. Those bytes are never counted, and C2 and C3 still PASS. The independent FakeHub
    `cdn_reads` check exists only offline.

    `make_server` already accepts `client_factory`, so the exit script can observe the wire at no extra cost.
  fix: |
    In T015 `run()`, pass a `client_factory` that wraps `httpx.HTTPTransport` in a counting transport. For every request
    it records the URL, the Range header and the response body bytes yielded. Add these checks:
    - C10: every safetensors request has a Range, and its end is ≤ 8+N for that file.
    - C11: the independently counted body bytes == ByteLog meta+header totals.
    Add an offline test in which a deliberately unlogged extra read makes C11 FAIL.

- severity: major
  where: invariant 2; plan.md §3.1 (lines 157–160) and D5; T002 behavior 4; T010 behavior 4 (8-thread header pool)
  problem: |
    `preflight()` checks `totals + request <= threshold` but reserves nothing. The actual bytes are added later in
    `record()`. With 8 concurrent header reads, all 8 preflights can pass against the same total.

    The non-weight read can then exceed the threshold by up to about 7 × (threshold − total). That is roughly 450 MiB
    at the 64 MiB default, with no confirmation or refusal. This is exactly the "pathological headers" case that D5 says
    the threshold stops.
  fix: |
    Have `preflight()` atomically reserve `end-start` under the lock. Release or adjust the reservation in
    `record()`, and on failure. Base the threshold test on `totals + reserved`. Add a T002 test in which 8 threads each
    preflight threshold/4 and at most 4 succeed.

### Minors

- severity: minor
  where: plan.md §2 lines 14–19, D3; T015 commands; later.md "Unscheduled"
  problem: |
    The handling of the unreachable Hub keeps the check falsifiable, because UNPINNED makes the script exit 2 and a
    missing network gives a non-zero exit (C1 fails), never a skip. Two things are left implicit:
    (1) Nothing says the phase stays open, not passed, when only E0 can run here.
    (2) Nothing requires the full exit_check table (stdout) and the host it ran on to be recorded as evidence. Only E5 is
    recorded. "The orchestrator runs..." cannot happen in this container, so in practice a human runs it elsewhere.
  fix: |
    State explicitly: "Phase 1 is not complete until E1–E5 have passed; E0 alone never closes the phase." Require that
    the exit_check stdout, the pins and the hostname be appended to log.jsonl by whoever runs it.

- severity: minor
  where: T015 acceptance (offline test uses only a small custom expectation)
  problem: |
    The default E1–E3 expectation callables are exercised for the first time on the network host. A KeyError or a
    vacuous predicate would surface only at EXIT. Two examples:
    - `config.raw["vision_config"]`
    - "some stack with prefix ending `layers`" matching an unintended stack
  fix: |
    Add offline unit tests that feed hand-built card, view and table dicts to each default expectation function, with
    one passing input and one failing input per check.

- severity: minor
  where: T005 behavior 2 and 4; T004 behavior 3
  problem: |
    Two reads bypass the threshold:
    - The API revision call has no preflight.
    - `read_file` with `size None` preflights a length of 0.
    So these reads are unbounded.
  fix: |
    Cap streamed bytes at `threshold - totals` inside `_get`, and refuse once the cap is hit. Alternatively, preflight
    the API call with a fixed bound (for example 16 MiB).

- severity: minor
  where: T005 behavior 5 (Range ignored)
  problem: |
    "Classified by position" conflicts with `record(start=start, ...)`. When Range is ignored, the body starts at file
    offset 0, not at `start`. Recording with `start=start` misclassifies the bytes, which can undercount weight.
  fix: |
    On a 200 response to a ranged request, specify `record(start=0, nbytes=received)`, and assert this in
    test_range_ignored.

- severity: minor
  where: plan.md §3.1 line 143; invariant 2 ("every byte fetched is logged")
  problem: |
    Redirect bodies and response headers are excluded from the count. This is declared openly (`counting` field), so it
    is acceptable. It is still a narrowing of the invariant.
  fix: |
    Log each 3xx as a 0-class `fetch` event with its body length in `note`, or accept the narrowing explicitly in D-table.

- severity: minor
  where: plan.md §2 C6, C9
  problem: |
    Both checks are nearly tautological:
    - C9: the root params are the sum of numel by construction.
    - C6: `revision_sha == pin` is guaranteed by RevisionMismatch.
  fix: |
    Keep both as cheap sanity checks, but do not count them as evidence. Alternatively, replace C9 with
    `params_total == Σ numel from the Parquet`.

- severity: minor
  where: plan.md §3.3 local key; T004 behavior 2
  problem: |
    `revision_kind = "local-content-hash"` is a stat hash (size and mtime), not a content hash. It is also not a commit
    SHA as invariant 1 describes the key. An HF cache snapshot folder already has the commit SHA as its directory name.
  fix: |
    Rename it to `local-stat-hash`. Optionally, detect `.../snapshots/<40hex>` and use that SHA as the key.

- severity: minor
  where: T012 acceptance test_scan_hub_fake
  problem: The test leaves the injection mechanism open ("monkeypatch make_source or ... httpx.Client"). That is a small design choice inside a sonnet task.
  fix: Add `client` passthrough or `client_factory` to `cli.main` for tests, or name the single monkeypatch target.

- severity: minor
  where: T015 tests/test_exit_network.py
  problem: The subprocess uses the relative paths `scripts/exit_check.py` and `exit/pins.json`, so it breaks when pytest runs from another cwd.
  fix: Resolve both paths relative to `Path(__file__).parents[1]`.

- severity: minor
  where: T005 behavior 6 (gzip decode path); T014 behavior 3d (collapse/expand, show-tensors toggle)
  problem: |
    Simplicity. The client sends `Accept-Encoding: identity`, so the gzip-decode branch is dead code, and a NetworkError
    would be enough. The subtree collapse/expand interaction is not needed for any exit check.
  fix: Drop the gzip branch. Treat collapse/expand as optional polish.

## checks with no finding
- COVERAGE: every item of the Phase 1 roadmap line is covered by a task: intake (hub@pin, local, sharded, model_index
  components), header-only scan plus model card, Card v0 with empty stats slots, graph plus depth strip (MoE-aware),
  live staged log and byte counter, and the frontend decision (D1). No later-phase work leaked in: the gate UI,
  tokenizer, sigma-curves, Card store and GGUF are all in later.md.
- DEPENDENCIES: no cycles. Transitive edges cover every helper import (FakeHub via T005, cards.py via T009). Tasks
  that run in parallel share no files.
- INPUTS: sharded, multi-component, MoE, missing card, lying base_model (recorded as a claim), gated (401/403),
  mid-read network failure, Range ignored and `.bin`-only are each handled and tested offline.
- ROUTING: no other sonnet task hides a material design decision. T014 is opus with spec_complete false, which is
  appropriate.
