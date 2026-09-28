# Phase 1: later / out-of-phase ideas

Phase 2
- Download-gate confirmation UI and CLI prompt. P1 hard-refuses weight reads and
  over-threshold reads instead of asking.
- Tokenizer reading plus MinHash for reference suggestion. The `stats.tokenizer_minhash`
  slot is reserved.
- Sampled ranged weight reads and sigma-curves. The `stat_sigma_curve` column and the
  `sigma_curves` slot are reserved.
- Structural diff view; the frontend may need a real framework then (revisit D1).

Phase 3
- Claimed-vs-detected check of `base_model` (P1 records claims verbatim, `claimed: true`).
- Config vs header consistency as a formal tool (P1 only uses it in the exit check).

Phase 4
- Card store with content addressing: reopen or list existing Cards in the UI, and skip
  rescans when `(repo, sha)` already has a Card.
- GGUF / quantized / `.bin` inputs. A `.bin` pickle cannot be read header-only, so it
  needs the gate.
- Per-block attribution filling the depth strip (`stats.attribution`).

Unscheduled
- Fused 3-D expert tensors (e.g. gpt-oss `mlp.experts.gate_up_proj` with a leading
  expert dim), `shared_expert(s)` naming, and DeepSeek-style mixed dense/MoE layer
  labelling beyond signatures.
- Dataflow graph (needs model code or configs mapped to architectures); P1's graph is a
  module hierarchy.
- SSE/websocket live log instead of 250 ms polling.
- A pipeline-level Card for diffusers repos (P1 writes component Cards only).
- Speculative header prefetch (one ranged request of 8+guess bytes) to halve round
  trips. It is rejected for now because over-reading past the header would count as
  weight bytes.
- Pinning exit repos in CI once the build container can reach huggingface.co.
- Environment: the egress proxy in the planning container blocks huggingface.co,
  cdn-lfs*.hf.co and cas-bridge.xethub.hf.co. Allowlisting them would let EXIT run here.
- Local folder inside an HF cache `snapshots/<sha>/`: optionally use that SHA as the key after verifying blob hashes (validation round 1, rejected for P1).
- Reuse the CDN/Xet Location from the 8-byte read for the [8, 8+N) read, re-resolving on 403/expiry, to save one hop per file (validation round 2, deferred: not needed for the 10 s budget).
