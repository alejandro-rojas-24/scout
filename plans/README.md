# Planning waves — index

This directory holds the plan-only output of the orchestrator wave (see
`docs/harness/orchestrator.md`). Implementation happens in a later, local wave.

## Branches (stacked; each contains everything before it)

| Phase | Branch | Status |
|---|---|---|
| harness | `claude/sleepy-fermat-dmjyu9` | harness + Phase 1 through validation r3 |
| 1 | `plan/phase-1-architecture-view` | PLANNED; IMPLEMENTED on `impl/phase-1-architecture-view` (16/16 tasks reviewed, E0 pass; E1–E5 pending Hub access) |
| 2 | `plan/phase-2-diff-view` | PLANNED (2 revise rounds, approved under directive) |
| 3 | `plan/phase-3-analysis-backend` | PLANNED (2 revise rounds, approved under directive; D36 needs human ack) |
| 4 | `plan/phase-4-infra-attribution` | PLANNED (2 revise rounds, approved under directive) |
| 5 | `plan/phase-5-product-refinement` | PLANNED (2 revise rounds, approved under directive; D99 needs human ack) |

Use the last branch in the table as the implementation base.

## Per-phase files
- `plan.md`: goal, exit check, decisions, schemas, risks, validation responses
- `tasks/T###.yaml`: task specs. Ids are unique across phases: P1 = T001.., P2 = T101.., P3 = T201.., P4 = T301.., P5 = T401..
- `validation-rN.md`: each validator round's findings, frozen once the round ends
- `validation.md`: the latest validator round, if it has not been frozen yet
- `later.md`: ideas that belong to a later phase
- `log.jsonl`: one event per line (`ts, agent, model, step, task_id, status, note`)

Non-blocking minor findings from every phase are collected in `/FUTURE_IMPROVEMENTS.md`.

## Wave policy (human directive, 2026-09-28)
- Human review is disabled, and the orchestrator approves at CHECKPOINT.
- Environment limits are documented, not treated as blockers: no Hugging Face access from the cloud container, gated repos needing HF_TOKEN, the DGX Spark.
- The planner gets at most 2 revise rounds. After that, remaining majors go into plan.md "Open issues" and do not block. Minors go to FUTURE_IMPROVEMENTS.md.
- Every validator runs in a fresh context and sees only the context, plan and tasks, never earlier validation files.

## Incidents worth studying (see each phase's log.jsonl)
- **P1: the auto-mode safety classifier had an outage.** It returned no verdict, so every write or spawn was blocked for several turns. The human had to re-prompt twice.
- **P1: a queued message caused two planners to edit T015 at once.** A SendMessage to an idle subagent was queued and delivered late. A second, fresh planner had been spawned for the same fix in the meantime, so both edited T015 concurrently. The merged result was verified by the orchestrator.
  - Lesson: after an agent finishes, don't message it again; spawn a new agent instead.
- **P1: the validators kept finding new majors while their count shrank** (r1 5 → r2 1 → r3 1). Each fresh validator found a different class of hole. The ceiling on revise rounds is what stopped the loop.
- **P2–P5: the classifier went down again several times.** In P2 one validator ran while the classifier was down, so the orchestrator checked that it had only written `validation.md`. In P3 the planner could not run its own task check; the orchestrator read the script, then ran it once the classifier recovered.
- **P5: API session limits (429) cut off two planners partway through.** For the initial draft, a fresh planner finished the consistency pass. For revision round 1, the same agent was resumed with SendMessage; it picked up its own partial edits and completed them. Resuming a *finished* agent worked here, which contrasts with the queued-message problem in P1.
- **Validators ran numbers, not just reviewed text.** From P2 on, validators re-simulated the statistics in numpy and found real defects:
  - P2: d_eff and the edge-block null
  - P4: a depth-local and a depth-spiked null
  - P3: SDK incompatibility, confirmed by downloading the wheels

  The planners then re-measured before changing thresholds. This was the main source of value in each round.

## Wave summary

| Phase | Tasks | Validator rounds | Blockers / majors per round | Open human acknowledgements |
|---|---|---|---|---|
| 1 | 16 (T001–T016) | 3 | r1 0/5 · r2 0/1 · r3 0/1 | — |
| 2 | 16 (T101–T116) | 2 | r1 1/3 · r2 0/1 | — |
| 3 | 18 (T201–T218) | 2 | r1 1/4 · r2 0/4 | D36 (anchor Gram is not weights) |
| 4 | 21 (T301–T321) | 2 | r1 0/6 · r2 0/3 | — |
| 5 | 19 (T401–T419) | 2 | r1 1/6 · r2 0/5 | D99 (report files are user exports) |

No phase reached APPROVE from a validator. Every phase was closed by the orchestrator under the directive once the revision cap was reached; the last round's majors were fixed but not re-validated. The implementation wave should therefore treat the reviewer step as the first independent check of each final revision.
