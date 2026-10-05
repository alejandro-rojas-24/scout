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
T006: >1 non-canonical index with exactly one single file returns the single file; decide whether that should be AmbiguousWeightsError.
- LocalSource FIFO race (T004 review nit 4): a path swapped for a FIFO between resolve() and a read blocks in open() before the fstat check; fix with os.open(O_RDONLY|O_NONBLOCK) + S_ISREG check on the fd.
- LocalSource alias limitation (T004 review): an extensionless weight blob linked under a non-weight name, with no weight-named sibling on the same inode, is classified as meta (ByteLog classifies by name; content sniffing deferred).
- LocalSource symlink-after-resolve (T004 review r2): a new symlink (hidden or not) pointing at a listed file, created after resolve(), does not change the file's stat, so reads keep the resolve-time classification; re-resolve before reads if the tree may change.
- T004 (r3 review): hardlink aliases that name checks cannot see still count weight bytes as meta: a weight file inside a hidden directory (e.g. `.cache/model.bin`) hardlinked to a visible non-weight name, and a weight file outside root hardlinked into root. Suggested fix: for non-weight files, refuse (or warn) when `st_nlink` > number of names found for that inode; costs false positives on hardlink-dedup trees.
- T008 (review): nn.Sequential indices (e.g. visual.merger.mlp.{0,2}, VAE head.{0,2}) become depth-2 stacks with all-different signatures; the view/depth strip may want to suppress such stacks.
- T007/T009: model_card and pipeline subkey order differs from plan 4.2 (claimed last; model_index_path after components). Key sets match; reorder in metadata.to_dict if strict order matters.
- T005 (r3 review, PRIORITY): a caller-supplied client with a request event hook or a client-level `Authorization` header could still put the HF token on CDN requests. Fix: in HubSource.__init__, refuse request hooks and client-level Authorization too.
- T005 (r3 review): if a caller drains `resp.stream` directly (not read()/iter_*), num_bytes_downloaded stays 0; HubSource still fails closed but logs 0 bytes. Undetectable externally; document in code.
- Process: reviewer/implementer probe scripts in the shared scratchpad were overwritten by other agents; future evidence should be self-contained probes with task-specific names, or committed tests.
- T009 (r3 review): concurrent first-writer window can leave a JSON pointing at a missing Parquet until the next rescan (load fails cleanly). Fix in P4 with a per-key lock or content-addressed store.
- T010 (review): budget threshold can only trip mid-scan (HEADERS/META) at the HubSource level; the 16 MiB unknown-length reservation means small fixtures trip on the API preflight.
- T016/C12 (review): older HF revisions may redirect LFS to cdn-lfs*.huggingface.co, which ALLOWED_HOSTS (*.hf.co) does not cover; if E4 hits this, it is a human decision to widen the allowed set.
