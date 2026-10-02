# Phase 5: later / out-of-phase ideas

P5 is the last roadmap phase, so everything here is unscheduled. Each item says why it is not in P5.

## Decisions for the human (not work items until decided)
- **Export of retrained weights (D83).** P5 never writes retrained weights anywhere. If an export path is wanted, it
  needs an explicit amendment of invariant 1 first. A design sketch, so the decision can be made on something concrete:
  `scout retrain --export-weights DIR` as a second gate after the job (bytes, disk, destination and the licence
  question shown; exact confirmation as for downloads); the worker writes safetensors to DIR on the Spark (never to
  scratch), logs an `export` event, and the Card records `{dest_host, dest_path, bytes, sha256}`. From then on the file
  is the user's, not scout's; scout never reads it back. Open questions: whether any export is acceptable while the
  licence question is unanswered, and whether exported files should carry a lineage note in their metadata.
- **Report files and invariant 1 (D99).** Accepted provisionally as user-requested exports; the human should confirm
  (like P3 D36).
- **Human blind reader availability (D89).** Default: required; a logged waiver is possible.

## Retrain loop
- **Re-attribution with longer training.** On the planner's toy (§2.4), retrained gate/up neurons drift back towards the
  base's as training continues (z rising from null to about 2 at m = 256 by step 1000–1500), because the frozen rest of
  the network expects the original block function. P5 stops at the first eval step where both criteria hold and says so
  in `RETRAIN_CAVEAT`. Measure the drift on real models across budgets; if the signal reliably returns, the loop's
  answer "retraining these blocks removes the signal" is budget-dependent and the panel should show the whole curve.
- **Re-initialising more than the MLP.** Attention (q/k/v/o), norms, or whole blocks. The P4 test looks at MLP neurons
  only, so these would change quality cost but not the measured signal; they matter once attention attribution exists.
- **Quantised, MoE, pipeline-component and GGUF subjects.** Retraining needs a dequantised trainable copy (weights in
  memory are fine; the question is the plan and the recipe), per-expert re-initialisation, and a diffusers-aware model
  build.
- **Larger corpora.** Above 32 MiB, data needs its own byte class and gate (invariant 2) and a streaming reader.
- **Self-distillation data** (sampled from the subject) as an alternative data source without licence questions; it
  costs GPU time for sampling and pulls the new neurons towards the old function (§2.4 KD variant).
- **Multi-GPU / multi-job retraining**, resumable training across worker restarts (would need persisted optimizer
  state: an invariant 1 question).
- **Attribution COMPUTE abort on client death.** Retrain jobs abort within one step (D96); the P4 attribution runner
  keeps its stated 45-minute bound (P4 FUTURE_IMPROVEMENTS).

## Report
- **System 2 in the exit reports.** Excluded because an LLM answer cannot be rebuilt for L10b; a recorded-answer mode
  (the S2 output hashed into the ledger, L10b comparing against it) would allow it.
- **Release dates for direction hints** (P3 later.md): needs the Hub `createdAt` field in the Card (a Card field, one
  more API byte class decision). Seeded as a usage note.
- **Native PDF / Markdown export.** The print stylesheet covers PDF; add a generator only if the usage notes ask.
- **Per-family accuracy and calibration curves in the report**, once the labeled set grows (P3 later.md).

## Black-box behavioural signal (D90)
- Constraints for a future phase: a separate report section headed as behavioural (not weight) evidence; never an
  input to JEV or to System 2's verdict; its own labelled calibration set of documented distillation pairs (teacher,
  student, method) and documented negatives; generation through a gated, logged path (API calls or GPU jobs) with the
  prompt set pinned; and it must be able to abstain. Until then the report states the blind spot (DISCLAIMERS 1, reader
  Q9).

## UI (usage-notes queue; seeded in `docs/usage-notes.md`)
- Subtree collapse/expand in the module graph (P1).
- Starting attribution and retrain jobs from the UI with a gate panel and a live job panel (P4). This is a second
  confirmation surface for invariant 2 and needs its own design review.
- System 2 commentary from the report view (P3).

## Environment
- This container reaches PyPI (torch 2.14.0 installed from PyPI in a scratch venv; download.pytorch.org is blocked by
  the proxy) but not huggingface.co, the DGX Spark or a GPU. Allowlisting `huggingface.co` and `*.hf.co`, an ssh route
  to the Spark and an `ANTHROPIC_API_KEY` would let E1 and E3–E6 run from here.
