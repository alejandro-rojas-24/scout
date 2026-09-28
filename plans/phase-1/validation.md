verdict: REVISE

findings:
  - severity: major
    where: "plan.md §2 (pins, E1-E3), §5 D2 fallback; T015 behavior 2-3 ('an unknown repo gets only the common checks'; main only validates 40-hex); T016 DEFAULT_EXPECTATIONS"
    problem: >
      The exit check does not enforce which targets it runs. scripts/exit_check.py accepts any pins.json whose
      values are 40-hex. It runs whatever repos are listed and gives unknown repos only C1-C11. So three things can
      each produce "EXIT CHECK: PASS" while the phase goal is unmet: (a) dropping a target from pins.json
      (e.g. Qwen-Image after an E3 FAIL); (b) swapping in the D2 fallback stabilityai/stable-diffusion-xl-base-1.0,
      for which no expectation function exists, so none of the E3-equivalent checks (components, down_blocks depth 3)
      run; (c) renaming a key. D2 invites (b) with no rule for who may trigger it or when. The plan's own
      "a FAIL goes to a human, never tuned away" rule can therefore be bypassed by editing the pins file, and the
      orchestrator is sanctioned to edit that file at EXIT.
    fix: >
      In main()/run(), FAIL (non-zero, with a named Check) unless set(pins) == set(expectations) exactly. Default:
      the three E1-E3 repos, and every repo must have an expectation function. If the D2 fallback is kept, add
      expect_sdxl_base (T016, with pass/fail tests) and a DEFAULT_EXPECTATIONS variant selected by an explicit flag.
      State in §2 that switching to the fallback is a human decision logged in log.jsonl before the run, never after
      a Qwen-Image FAIL without that record. Add an offline T015 test: pins missing one required repo -> non-zero;
      pins with an extra/unknown repo -> non-zero.

  - severity: minor
    where: "plan.md §2 pins paragraph; T015 command 'at EXIT only'"
    problem: >
      The pins are produced by the code under test (`scout resolve`), and C6 then compares Card SHAs to those same
      pins. That makes C6 self-referential: a resolve bug that returns a wrong but valid-looking SHA would go
      unnoticed. The E1-E3 constants still anchor correctness, so impact is low.
    fix: "When pinning, cross-check each SHA independently, e.g. `git ls-remote https://huggingface.co/<repo> refs/heads/main` or a plain curl of /api/models/<repo>/revision/main. Record both values in the EXIT log line."

  - severity: minor
    where: "T003 behavior 3; T005 behavior 7; plan.md §8 'Xet/CDN redirect' risk row"
    problem: >
      The Hub cannot be reached from the build container, so every real-Hub behavior must be mirrored by FakeHub.
      FakeHub serves non-LFS files (config.json, README.md, *.index.json, model_index.json) directly with 200 and
      only emits absolute Locations. The current Hub often answers non-LFS resolves with a 307 carrying a relative
      Location (/api/resolve-cache/...). The relative-Location path in T005 and the Location-to-orig_path mapping in
      CountingTransport (T015) are therefore never exercised offline, and E1-E3 depend on them.
    fix: "Add a FakeHub option (e.g. meta_redirect='relative307') that answers non-safetensors resolves with 307 and a relative Location to a second HUB path. Use it in one T005 test and in T015 test_offline_pass."

  - severity: minor
    where: "T005 behavior 2 (resolve error mapping)"
    problem: >
      RepoNotFoundError is selected by a case-sensitive substring "Repository Not Found" in the body. A 401 whose
      body differs (the real Hub may use different casing, and also sends the X-Error-Code header) falls through to
      `body["sha"]`. That raises KeyError, not a ScoutError, so the CLI shows a traceback instead of exit code 3
      (D8 / invariant 6 intake robustness).
    fix: "Map on the X-Error-Code header (RepositoryNotFound / GatedRepo / RevisionNotFound) first, then on a case-insensitive body match. Any other non-2xx from the API becomes a ScoutError subclass. Add a FakeHub test in which the body text differs."

  - severity: minor
    where: "T005 behavior 7 (redirect bodies)"
    problem: >
      3xx bodies are recorded but never preflighted or reserved. Up to MAX_REDIRECTS × META_MAX_BYTES (80 MiB) per
      request can be read outside the reservation budget, which exceeds the 64 MiB threshold. That is a small hole
      in the threshold side of invariant 2. Real redirect bodies are tiny.
    fix: "Cap redirect bodies at a small constant (e.g. 64 KiB) and preflight/reserve that amount per hop under path REDIRECT_PATH."

  - severity: minor
    where: "plan.md §2 E5; phases.md exit 'render correctly in under 10 seconds'"
    problem: >
      C1 times the API only (POST -> done with views), and the E5 checklist does not record browser render time or
      model-card tags. For a 100-node SVG, render time is almost certainly negligible, but the phase line is about
      render, and T014's static test only greps strings, so a broken app.js is caught only by E5.
    fix: "Add to E5 per target: 'page shows sections within 10 s of clicking Scan' and 'tags shown in the claimed box', both recorded in the log.jsonl line."

  - severity: minor
    where: "plan.md §8 risk '10 s budget on MoE'; later.md 'speculative header prefetch'"
    problem: >
      Each header costs 4 sequential round trips (resolve 302 + CDN, twice). On a high-latency host, C1 may fail on
      Qwen-Image (14 shards) or Qwen3-30B-A3B (16 shards) for reasons unrelated to correctness. The plan accepts
      that a FAIL goes to a human, which is fine, but there is a cheap mitigation that does not over-read.
    fix: "Optional: reuse the CDN Location from the 8-byte read for the [8, 8+N) read, re-resolving on 403/expiry. This keeps the exact ranges (C10 still holds via the Location mapping) and saves one hop per file."

notes:
  - "INVARIANTS 1-6: no violations found. The P1 'refuse instead of confirm' subset of invariant 2 is explicit (§3.1, D5) and the gate is deferred to P2. Disclaimers (invariant 5) are covered by T011, T012 and T014. Invariants 3 and 4 are not exercised in P1."
  - "COVERAGE: every item on the Phase 1 roadmap line maps to a task (intake T004/T005/T010, header scan + card parsing T006/T007, Card v0 T009, graph and strips T011/T014, live log T002/T013/T014, frontend decision D1, exit T015/T016). later.md holds only P2+ items; no leakage found in the tasks."
  - "EXIT CONSTANTS spot-checked by hand: Qwen3-8B 399 tensors / 8,190,735,360 params; Qwen3-30B-A3B 18,867 tensors / 30,532,122,624 params / 4,718,592 per expert. All are consistent with the stated configs."
  - "UNREACHABLE HUB: the handling keeps the check falsifiable. UNPINNED makes main() return 2, and the phase-status rule forbids closing on E0 alone. The one hole is the target-set enforcement described in the major finding."
  - "DEPENDENCIES: no cycles, the parallel groups share no files, and transitive edges cover every helper import."
