verdict: REVISE

Validator, Phase 4, round 2 (plan "DRAFT r1"). Read: project-context.md, phases.md Phase 4, plan.md, tasks T301-T321,
later.md, and the P1-P3 task specs as contracts. Environment, credential and hardware limits were not counted as findings.

Checks run:
- Dependencies and waves: /tmp/val4r2/deps.py. No missing dependency, no cycle, no wave-order violation, and no file
  with two P4 owners. Every edited P1-P3 file's original owner is a transitive dependency of the editing task.
- Statistics: /tmp/val4r2/fwer.py, numpy + scipy 1.17.1. It reimplements T314 behaviour 8/9c exactly: the pooled
  control null, Student-t prediction, the control leave-one-out, the shift guard and Bonferroni over 2L tests. It runs
  20,000 jobs per row.
- Critical values: every value in §2.4 and §3.8 was recomputed and matches (4.036/4.038, 4.081/4.103, 4.305,
  4.548/4.706).
- Byte table: §2.5 was recomputed and matches (179,000,967,168 / 699,222,528; 25,956,065,280 / 268,435,456).

findings:

- severity: major
  where: plan.md §3.8 steps 6-7 (lines 706-709, 749-757), §2.4 (lines 158-161, 173-180), A9; T314 behavior 9c
  problem: |
    The family-wise error (FWER) bound does not hold in the depth-heterogeneous case that r1 was written for. The
    control leave-one-out self-test does not "catch a depth where the shared structure spikes".

    At a spike depth, the candidate's and the control's aligned z are exchangeable draws. Whenever the candidate test
    fires, the control test at that depth fires with roughly the same probability. So P(candidate false attribution
    AND control_ok AND null_ok) can reach about 0.25.

    Measured with T314's exact rule, N(mu_l, 1) nulls shared by candidate and control, and shift N(0, 1):

    | case | FWER, L = 80 | FWER, L = 28 | control LOO fired |
    |---|---|---|---|
    | one block with mu = 4 | 0.196 | 0.104 | 38 % / 29 % |
    | one block with mu = 5 | 0.168 | 0.123 | 75 % / 65 % |
    | blocks 0-2 with mu = 4 | 0.157 | 0.018 | |
    | homogeneous (the plan's MC case) | <= 0.005 | <= 0.005 | |

    mu = 4 is only rho0 ~ 0.09 at m = 2048. Block 0 is the most plausible place for such a spike, because the
    embedding probe is literally block 0's real input.

    The plan's A-null, "at every depth distributed like the pooled control", is therefore depth homogeneity. That
    contradicts the text claiming protection against depth spikes. The Monte Carlo figure (<= 0.0111) covers only
    per-job homogeneous means.

    The same mechanism also means a correct build FAILs C10/C11 far more often than the stated 1-2 % whenever the real
    null has a spike. With a block-0 spike at mu >= 5, C11 fails with p >= 0.65.
  fix: |
    Preferred: make candidate attribution also require a per-depth paired margin against the control at the same
    position: `(z_l - z_ctrl,l) / (s0 * sqrt(2)) >= crit`, in addition to `z_adj >= crit`.
    - In the same simulation this gives FWER <= 5e-5 in every spike and bump case above.
    - Derived blocks (z ~ 40 against a control near 0-5) still clear it by a wide margin.
    - C10's median z_adj criterion is unchanged.

    Otherwise, restate A-null explicitly as depth-homogeneous, delete the "catches a spike" claims (§3.8, §2.4, D76),
    and add a single-depth-spike row to the Monte Carlo.

    Either way:
    - add a T314 test with a one-depth spike fixture, expecting <= 1 false attribution over the triples
    - restate the false-FAIL probability in §2.4 as conditional on homogeneity

- severity: major
  where: plan.md §2.1 target table, §2.3 C10/C11; T320 check_positive/check_control; T321 run
  problem: |
    The live exit contains no independent candidate. Specificity on real weights (not attributing an unrelated model)
    is therefore never tested at E4.
    - C11 only tests the control against its own pooled null, which is a homogeneity self-check.
    - E4 passes even if real independent models are falsely attributed. Two ways this happens: a probe-inflation
      pattern the control does not share, or the spike case of the previous finding.
    - Synthetic E0 tests cover implementation bugs but not A-null on real models. A-null is the assumption the whole
      claim rests on, and it is deferred to later.md.
    - The te positive leg may be a byte-identical copy (the plan records this), so it adds little discriminating power.
  fix: |
    Add one ungated, documented-independent candidate to the te job (role "reference", not control), for example
    allenai/OLMo-2-1124-7B or mistralai/Mistral-7B-v0.1. That costs about 8 GB of MLP + embedding reads.

    Add a check C10b: that candidate is `tested` with 0 attributed blocks (and `null_ok`).

    Update P4_TARGETS/P4_REPOS (7 repos), the byte table, n_tests/crit, C0 and C3, and state the added false-FAIL
    probability (<= 0.01 under A-null).

- severity: major
  where: T306 (files, behavior 8), plan.md §2.1 phase status rule (lines 65-69), §10 "Suites stay green"
  problem: |
    T306 switches `SCHEMA_VERSION` to card.v3. Several P3 files hard-code card.v2, and no P4 task owns them:
    - `tests/test_scan_p3.py`: the T207 test `test_anchors_approved_hub` asserts "card.v2".
    - `scripts/p3_set.py` (T213 behavior 2): it skips an existing Card only if it "loads with schema card.v2". With v3
      Cards it re-reads weights, which breaks "Never re-read weights for an existing Card".
    - `tests/test_p3_set.py`: `test_confirm_mode` asserts card.v2, and `test_existing_card_skipped` expects 0 CDN
      weight requests on a rerun.

    Consequences:
    - T306's acceptance (`pytest -q` green) cannot be met without editing files outside its `files` list.
    - T307's statement that the scan tests are "already updated by T306" is false.
    - The P3 offline suite in E0 goes red.
  fix: |
    Add `tests/test_scan_p3.py`, `scripts/p3_set.py` and `tests/test_p3_set.py` to T306:
    - the p3_set skip rule accepts card.v2 | card.v3
    - the literals are updated, and no other assertion is weakened

    Then grep every P1-P3 task spec for schema-version literals (the validator found only these three besides the files
    already listed) and add them to §9.

- severity: minor
  where: plan.md §3.8 step 6 shift guard; T314 behavior 8
  problem: |
    One shift pair out of 2L with z >= about 4.06 (rho ~ 0.09 at m = 2048) makes the base abstain, so C10 FAILs.

    The shift pairs of a derived subject compare adjacent layers of the same lineage, in the same residual basis.
    Under the plan's own premise (shared features inflate aligned z by about 2 between independent models), these
    same-basis pairs could exceed that.

    The synthetic fixtures draw each layer's neurons independently given the model basis. Only the drift-0.5 rows add
    sharing across depths, at 12 blocks, which is far fewer than the 160 shift values of the 70b job.
  fix: |
    Make the guard relative: flag only a shift z that exceeds the aligned z of the same subject block. Alternatively,
    make abstention per block. Or add a fixture with cross-layer neuron inheritance and measure the guard's false-veto
    rate at 80 blocks.

- severity: minor
  where: T317 behavior 3; T309 find_attribution
  problem: |
    The attribution cache key is the sorted set of reference and control keys, without their roles. Calling with the
    base and control swapped hits a stored attribution that was computed with the opposite roles. plan_id, by contrast,
    includes roles (T310 behavior 7).
  fix: Include the role in `attribution_refs` (for example "control:repo@sha/comp"), and add a test in T309 and T317.

- severity: minor
  where: T317 behavior 7
  problem: |
    The `make_spec(...)` call omits `control_targets`, which T315 requires. validate_spec would then reject every spec.
    Implementers will probably notice, but the spec is incomplete.
  fix: Pass `control_targets=` in behavior 7.

- severity: minor
  where: T315 SshBackend argv / worker_hello/sweep/audit; T319 behavior 2 (route sonnet)
  problem: |
    The ssh path runs `scout jobs worker hello`, `scout jobs worker sweep --json` and `scout jobs worker audit --json`.
    T319, however, defines `scout jobs worker [--scratch-root] [--wire-audit]` and separate
    `scout jobs hello|sweep|audit`. Whether `jobs worker` passes a trailing subcommand through is an open argv design
    decision inside a sonnet task.

    The offline rehearsal uses `python -m scout.jobs.worker`, so the real ssh command line is never exercised before
    E2.
  fix: |
    Specify that `scout jobs worker` forwards its remaining arguments (argparse REMAINDER) to worker.main. Add T319
    tests for `scout jobs worker hello` and `scout jobs worker sweep --json`, or make SshBackend's default remote
    command `python -m scout.jobs.worker`.

- severity: minor
  where: plan.md §2.3 C1; T320 check_env; T316 hello
  problem: |
    - C1 does not check the hello's `has_hf_token`. A Spark without a token passes E2/E3 and fails only inside the E4
      job, on the gated Llama repos.
    - `wire_audit_available` needs `import scripts.exit_check` from the ssh working directory. A PEP 660 editable
      install does not put the repo root on sys.path, so this can fail even with "pip install -e".
  fix: |
    Add `has_hf_token is True` to C1. State how `scripts` becomes importable on the Spark (for example
    SCOUT_SPARK_CMD="cd <checkout> && python -m scout.jobs.worker") and add that to the §2.5 V list.

- severity: minor
  where: plan.md §2.4 Resources, §2.3 C9 (70b max_wall_s 21600)
  problem: |
    At the assumed 10 MB/s, 179 GB takes 4.97 h, plus about 0.75 h of compute: 5.7 h against the 6 h cap, a margin
    under 6 %. A correct build FAILs C9 if sustained throughput is below about 9.4 MB/s, and nothing measures
    throughput before E4.
  fix: State the implied minimum throughput explicitly. Optionally measure a timed ranged read at E2 and record it, so
    a C9 risk is visible before the 6 h run.

- severity: minor
  where: plan.md §3.8 step 7 ("makes the bound exact"), §2.4 line 180 ("deterministic ... a rerun reproduces")
  problem: |
    - With the max(0, ...) and max(1, ...) floors and Bonferroni, the t bound is conservative, not exact. It also
      assumes the candidate's z is independent of the control pool, which shared depth structure breaks (see the first
      major).
    - Bit-exact reproducibility of float32 BLAS results across thread counts is not guaranteed, and near-ties in the
      linear assignment can flip.
  fix: Change the wording to "conservative under A-null (independent draws)" and "reproducible up to float
    nondeterminism". The rerun policy is unchanged.

- severity: minor
  where: plan.md §2.2 E2 / T321 behavior 2
  problem: |
    At E2 there is no P4_PLAN entry yet, so C1 uses disk_needed = 0 and only the 50 GiB reserve is checked. §2.2 and
    §2.3 describe C1 as checking max(disk_bytes) + reserve at E2.
  fix: Also evaluate the C1 disk clause at E3 with the plans' disk_bytes (a preflight row in `plan`), or reword §2.2.

- severity: minor
  where: plan.md §3.5 by-weights / T309 same_weights, reindex, `scout store import`; T318 server store routes and app.js
    stored list
  problem: |
    None of these is needed for the exit: C12 uses build_view directly, and the digest equality is read from the Cards.
    The plan already accepts this surface knowingly (§11 deferred minors).
  fix: Optionally cut them, or keep them as accepted. Listed for completeness.

counts: blockers 0, majors 3, minors 9

Contract references checked and found accurate:
- P1 T015: CountingTransport records with orig_path and location
- P2 T104: tokenizer resolution incl. vocab.json
- P2 T110: viterbi continuation tie-break, which gives identity on an all-zero cost
- P2 T116: parse_confirm_plans
- P3 T203: VocabIds / vocab_ids (vocab.json byte-level)
- P3 T205: find_embedding
- P3 T209: structure_alignment signature and reldepth on constant non-zero cost, so the te control 28 vs 32 is reldepth
- P3 T214 / T216: CODE_PATHS
- P3 T218: p3_ledger

GGUF header-only safety (T304 behaviours 1 and 7, T301 behaviour 1): sound. The minimum-encoding bound (13 B per KV,
24 B per tensor info, the per-element minimums) is a valid lower bound on infos_end. Each round reads only
[have, bound), and have == infos_end at the end, so the C14 equality holds. Every byte is registered as header before
it is read, and a read past the bound would be classified as weight and refused without a grant.

Wave file-disjointness: holds as claimed.
