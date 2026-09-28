# scout — model lineage forensics tool

Development runs through the phase harness in `docs/harness/`:
- `project-context.md` — invariants and roadmap (binding for every agent)
- `phases.md` — per-phase scope and exit criteria
- `orchestrator.md` — the loop (PLAN > VALIDATE > REVISE > CHECKPOINT > ROUTE > REVIEW > EXIT)
- Subagents live in `.claude/agents/`.

Per-phase workspace: `plans/phase-N/` (plan.md, validation.md, tasks/, log.jsonl, later.md).
