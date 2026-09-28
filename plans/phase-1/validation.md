# Phase 1 plan validation

Inputs reviewed: project-context.md, phases.md (Phase 1), plans/phase-1/plan.md,
tasks/T001–T016.yaml, later.md (scope-leak check only).

verdict: REVISE

findings:
  - severity: major
    where: "plan.md §2 L66-76, L52-58, L437; T015 interface L21-28 (run endpoint=None, main flag --budget-s), behavior 2/3/3a/4 (L52-75); T005 L22 (endpoint default = env HF_ENDPOINT)"
    problem: >
      The exit run does not fix where its data comes from or what C1's threshold is, and the evidence does not
      record either. exit_check.main passes endpoint=None, so HubSource uses env HF_ENDPOINT. `--budget-s` lets the
      operator raise the 10 s limit. The last-line summary records only {hostname, pins, passed, failed}, and no
      check asserts that the wire actually went to huggingface.co. pin_exit (with git ls-remote) can run on a host
      that has network, while exit_check runs anywhere else. Nothing ties the two hostnames together or ties the run
      to a code commit. In a container that cannot reach the Hub, E1-E3 can therefore PASS against a mirror or a
      local synthetic Hub that echoes the committed pins (constants are public), or with a relaxed budget, and the
      log.jsonl line looks the same as a genuine run. This is where the network block can make the exit check
      unfalsifiable. The rest of the D3 handling is sound: E0 cannot close the phase, there is no fallback, and pins
      are cross-checked with git.
    fix: >
      (1) exit_check.main always uses endpoint="https://huggingface.co" and exits non-zero if HF_ENDPOINT is set to
      anything else. pin_exit runs `scout resolve` with HF_ENDPOINT removed from the subprocess env.
      (2) Remove the --budget-s CLI flag and keep run(budget_s=...) for offline tests only.
      (3) Add common check C12: every CountingTransport record's host is in an allowed set (default
      {huggingface.co, *.hf.co}), and every card.source.endpoint == "https://huggingface.co". run() takes
      allowed_hosts so the FakeHub offline test can pass {hub.test, cdn.test}. Add an offline test in which a disallowed
      host makes C12 FAIL.
      (4) The final JSON summary line adds endpoint, budget_s, the sorted distinct wire hosts, and `git rev-parse
      HEAD` plus a dirty flag. The PIN line and the EXIT line must share the hostname, or the plan must say why not.
      (5) §2 and the §8 risk row state that a C1 FAIL keeps the phase open. The remedy is performance work and a
      rerun, never a waiver.

  - severity: minor
    where: "T005 behavior 2b (L47-48) and 6 (L63-77); T002 behavior 5-6 (L87-94); plan §3.1 L191-194 (invariant 2)"
    problem: >
      The API error mapping reads non-2xx bodies (case-insensitive body match), but no step records those bytes.
      The ByteLog schema also cannot record them: record() accepts only "fetch"/"retry", and note() is 0-byte. This
      contradicts "every byte that scout receives is recorded" and is untested. The real Hub usually sends
      X-Error-Code, so the path is rare, but it is an invariant-2 gap.
    fix: >
      Any body byte read from any response is recorded. For example, allow record(event="error", ...) with bytes,
      logged under a meta pseudo-path such as "@error" so that a CDN error body on a .safetensors URL is not
      classified as header/weight. test_error_mapping asserts that totals.meta grew by the body length.

  - severity: minor
    where: "plan L106 (C10); T016 behavior 2 C10 (L45-47); T015 behavior 1 (L47-51)"
    problem: >
      orig_path is mapped by exact equality between a Location string and a later request URL. With signed Xet URLs,
      httpx URL normalisation can break that equality. Unmapped CDN records are then silently skipped by C10. The
      ">= 1 record" guard is satisfied by the first-hop huggingface.co requests alone, so the hop that actually carries
      the bytes can go unchecked. C3 and C11 still bound the damage.
    fix: >
      C10 also FAILs if any 2xx record with body_bytes > 0, other than the API call, has orig_path None. Compare
      normalised httpx.URL objects instead of strings.

  - severity: minor
    where: "plan L107 (C11: 'so no unlogged fetch path exists'); T015 behavior 2; T015 test_unlogged_read_fails_c11"
    problem: >
      C11 only sees traffic through the injected client. A fetch through another httpx.Client, httpx.get or urllib
      bypasses both ByteLog and CountingTransport, so C11 still passes. The offline test only covers the bypass through
      self.client.
    fix: >
      In run(), patch httpx.HTTPTransport.handle_request at class level, or guard socket.connect, so that any
      in-process HTTP is counted or refused. Add an offline test in which read_file performs an unlogged fetch
      through a fresh httpx.Client and C11 (or the guard) FAILs. Soften the wording in L107.

  - severity: minor
    where: "plan §2 E1 (L109-120), E3 (L135-145); T016 behavior 3; invariant 5"
    problem: >
      "Render correctly" is checked automatically only through C8 and one E2 view row. Several properties rely
      entirely on the manual E5: E1/E3 view nodes and strips, the three disclaimers in every served view (invariant
      5), and whether the log is live rather than dumped at the end.
    fix: >
      Add per-target view checks:
      (a) every view.disclaimers == DISCLAIMERS;
      (b) every structure stack has a depth_strip with len(cells) == depth and a stack node with count == depth
          (E1 'layers ×36', E3 transformer 60);
      (c) at least one poll with status == "running" returned >= 1 new event.

  - severity: minor
    where: "plan L444 (risk 'cross-checked against config.json'); E1/E2 tables L113-133; D3 L378"
    problem: >
      The n_tensors and params_total constants were written without Hub access and are compared only with
      hard-coded values. The claimed config.json cross-check exists only for depth and n_experts. When a constant
      FAILs, the human has nothing independent to judge a correction against, so correcting it risks tuning to the
      observed output.
    fix: >
      Add checks that derive the expected values from the Card's own config.raw, using the formula already written
      in L147-149, and a check that params_total * 2 == index_total_size for all-BF16 cards. A constant can then be
      revised only when two independent sources agree.

  - severity: minor
    where: "plan L437 (10 s estimate); T010 steps 2, 3, 5 (L29-44)"
    problem: >
      The RTT estimate covers only the MoE. For Qwen-Image, reads run sequentially in two places:
      - RESOLVE: model_index.json plus 2 index files
      - META: README plus 3 configs
      Each of these reads takes 2 hops because of the relative 307. That is about 14 serial RTTs on top of 2 header
      waves × 4 hops and the TLS setup to two hosts. Qwen-Image, not the MoE, is the likely C1 risk.
    fix: >
      Read the index files and the META files through the existing thread pool, and state the Qwen-Image critical
      path in §8.

  - severity: minor
    where: "T008 behavior 2b/3 (L42-46); plan E5 L183-185"
    problem: >
      Any integer-indexed container with >= 2 indices becomes a Card stack. nn.Sequential children such as
      visual.merger.mlp.0/.2 and the VAE's encoder.head.0/.2 and decoder.head.0/.2 therefore show up as depth-2
      "stacks" with depth strips. That misrepresents the architecture in a forensics view, and E5's "a strip for each
      of its stacks" makes the check tautological.
    fix: >
      Mark stacks in which no block signature repeats and depth <= 3 as kind "indexed" rather than "repeated". Draw
      strips only for repeated stacks. Add a test using mlp.0/mlp.2.

  - severity: minor
    where: "D12 L387; T008 out_of_scope; later.md 'Unscheduled' (fused 3-D experts); invariant 6"
    problem: >
      Fused-expert MoE (gpt-oss or Llama-4 style, e.g. experts.gate_up_proj with a leading expert dimension) renders
      silently as dense: no expert group, moe=false, and no warning. The item is also unscheduled, although invariant
      6 requires MoE intake.
    fix: >
      T008 adds a structure warning when a tensor path contains an "experts" segment that is not followed by an
      integer and has ndim >= 3. later.md assigns fused-expert support to a numbered phase.

  - severity: minor
    where: "T004 behavior 4-5 (L45-49)"
    problem: >
      read_range uses a default buffered open(). BufferedReader reads ahead (>= 8 KiB), so the 8-byte and header reads
      pull tensor data into the process. That contradicts "never read a byte that preflight did not approve".
    fix: >
      Use open(path, "rb", buffering=0) and loop readinto until `length` is filled. Add a spy test asserting that the
      raw reads never exceed the requested range.

  - severity: minor
    where: "T004 behavior 2 (L36-40)"
    problem: >
      Stat handling for symlinked files is unspecified. In an HF cache snapshot, which is a common local target, every
      file is a symlink into blobs/. With lstat or DirEntry.stat(follow_symlinks=False), sizes become link lengths and
      read_header fails with a size mismatch.
    fix: >
      Specify os.stat that follows symlinks for files (directories are still not followed). Add test_symlinked_files
      using an HF-cache-shaped fixture.

  - severity: minor
    where: "T015 behavior 2 (L52-54)"
    problem: >
      One server with one client_factory is combined with "one CountingTransport per target", but the plan does not
      say how the factory switches between them. A lambda closing over the loop variable, or a factory created once,
      would attribute every target's wire traffic to one transport and corrupt C10/C11.
    fix: >
      Specify a mutable holder (current["transport"] set before each POST, with scans strictly sequential), or one
      make_server per target.

  - severity: minor
    where: "T009 behavior 4 (L57-60); T013 behavior 2"
    problem: >
      The temp names f"{json_path}.tmp-{pid}" collide when two server threads (same pid) scan the same target at the
      same time, for example after a double-click, which gives a corrupt or missing Card.
    fix: >
      Use tempfile.mkstemp(dir=json_path.parent), or pid + thread id + uuid.

  - severity: minor
    where: "T010 behavior 2 (L29-35)"
    problem: >
      When model_index.json exists, weight files at the repo root (e.g. SDXL's single-file sd_xl_base_1.0.safetensors)
      are skipped without any trace in the Card.
    fix: >
      Add a pipeline warning listing the root-level weight-extension files that were skipped.

  - severity: minor
    where: "plan E4 L154-159; D3 L378"
    problem: >
      E4 is only an "expect" comment with no scripted assertion. D3 says a human must "run E1–E4 locally", but E5
      also needs Hub access.
    fix: >
      Make E4 a network-marked test that asserts exit 0, totals.weight == 0, 1 card path, and the order of the stage
      lines. Change D3 to "E1–E5".

  - severity: minor
    where: "T014 behavior 3d (L34-35)"
    problem: >
      "OPTIONAL polish: collapse/expand subtree" is UX polish (roadmap P5) and is not needed for the exit check.
    fix: >
      Cut it and note it in later.md.

notes: >
  No blockers. Coverage of the Phase 1 scope is complete, and later.md shows no scope leakage. There are no dependency
  cycles or missing edges, and the tasks declared parallel have disjoint file sets. No sonnet-routed task hides a design
  decision. The E1/E2 constants were recomputed independently and match: Qwen3-8B 399 / 8,190,735,360 and
  Qwen3-30B-A3B 18,867 / 30,532,122,624.
  On the unreachable Hub, the approach is sound:
  - offline E0 against a FakeHub that models both the relative-307 and the absolute CDN redirects
  - E0 alone cannot close the phase
  - pins are cross-checked with git ls-remote
  - there is no fallback target
  The one gap is the major finding above: the exit run can be pointed elsewhere or given a relaxed threshold without
  leaving a trace in the evidence.
