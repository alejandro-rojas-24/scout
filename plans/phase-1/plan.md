# Phase 1 plan: Single architecture view

## 1. Goal

Given a Hub repo at a pinned revision or a local folder, scout reads only the
manifests, model card, configs and safetensors headers, and never weight bytes. From
these it writes a Card v0 (JSON + Parquet) per model or per pipeline component.
A local web page shows the scan's staged byte log live, a collapsed module graph
(MoE-aware) and a depth strip for each Card, all in under 10 s with zero weight
bytes read.

## 2. Exit check (runnable, falsifiable)

Pins: `exit/pins.json` maps each exit repo to a 40-hex commit SHA. It is committed
with the literal value `"UNPINNED"` in T015. At EXIT, on a machine that can reach
huggingface.co, the orchestrator runs the commands below once, writes the printed
SHAs into `exit/pins.json`, logs the action to `plans/phase-1/log.jsonl` and commits
the file. After that the SHAs never change. Every scan then requests `repo@<sha>`,
and `HubSource.resolve()` raises `RevisionMismatch` if the API returns a different SHA.

```bash
scout resolve Qwen/Qwen3-8B          # prints one 40-hex sha to stdout, 0 weight bytes
scout resolve Qwen/Qwen3-30B-A3B
scout resolve Qwen/Qwen-Image
```

When this plan was written, huggingface.co, cdn-lfs.hf.co and
cas-bridge.xethub.hf.co were rejected by this container's egress proxy (CONNECT
403). See decision D3.

### E0: offline suite (no network)
```bash
pip install -e '.[dev]'
pytest -q                 # expect: exit 0, 0 failures; network tests deselected
```

### E1–E3: exit script (network, pinned)
```bash
python scripts/exit_check.py --pins exit/pins.json     # expect exit code 0
# equivalent: pytest -m network -q tests/test_exit_network.py
```
The script starts the real server in-process (`make_server(port=0)`) with a temp
cards dir. For each target it POSTs `/api/scans`, polls `/api/scans/{id}` every
100 ms and stops the clock when `status == "done"` and `views` are present. That is
the same path the browser uses. It then loads the written Card JSON + Parquet from
disk. It prints one row per check (`target | check | expected | actual | PASS/FAIL`)
and exits non-zero on any FAIL or if any pin is not 40-hex.

**Common checks for every target (C1–C9):**

| # | Check | Threshold |
|---|-------|-----------|
| C1 | wall time POST → done with views | `< 10.0 s` |
| C2 | `totals.weight` from the server, from every Card's `fetch_log.totals.weight`, and the sum of `bytes_by_class.weight` over all events | all `== 0` |
| C3 | header bytes in successful `fetch` events | `== Σ over weight files (8 + header_len)`, which means exactly the headers were read and nothing more |
| C4 | per weight file: `8 + header_len + data_bytes` | `== size_bytes` (size from the Hub API; `data_bytes` from the header) |
| C5 | if `index_path` is set: `tensor_bytes_total` | `== index_total_size` (the value the repo publishes in its `*.index.json`) |
| C6 | Card files exist at `<out>/<repo_slug>/<sha>/<component_slug>.card.json` and `.tensors.parquet`; `key.revision_sha == pin` | true |
| C7 | Parquet row count | `== weights.n_tensors`; every `stat_*` column is 100 % null; every value in JSON `stats` is `null` |
| C8 | `view.nodes` count (collapsed graph) | `<= 100` |
| C9 | `view.nodes` root node params | `== weights.params_total` |

**E1 Qwen/Qwen3-8B @pin: 1 Card, component null**

| Check | Expected |
|---|---|
| config.model_type / architectures | `qwen3` / `["Qwen3ForCausalLM"]` |
| n_tensors | `399` (36 × 11 + 3) |
| params_total | `8190735360` |
| params_by_dtype | `{"BF16": 8190735360}` |
| stacks[0].prefix / depth | `model.layers` / `36` (== config.num_hidden_layers) |
| distinct block_signatures in model.layers | `1` |
| expert_groups | `[]` |
| model_card.present / license / base_model | `true` / `apache-2.0` / `["Qwen/Qwen3-8B-Base"]` |

**E2 Qwen/Qwen3-30B-A3B @pin: 1 Card**

| Check | Expected |
|---|---|
| config.model_type | `qwen3_moe` |
| n_tensors | `18867` (48 × (8 + 1 + 128 × 3) + 3) |
| params_total | `30532122624` |
| stacks[0] | prefix `model.layers`, depth `48` (== config.num_hidden_layers) |
| expert_groups | exactly 1: template `model.layers.#.mlp.experts.*`, `n_experts == 128 == config.num_experts`, `n_instances == 48`, `tensors_per_expert == 3`, `params_per_expert == 4718592 == 3 × hidden_size × moe_intermediate_size`, `homogeneous == true` |
| depth strip | 48 cells, all `moe == true`, all `n_experts == 128`, 1 distinct signature |
| view | exactly 1 node with `kind == "expert_group"` and `count == 128` |
| model_card.license / base_model | `apache-2.0` / `["Qwen/Qwen3-30B-A3B-Base"]` |

**E3 Qwen/Qwen-Image @pin (diffusers multi-component; text_encoder is a Qwen2.5-VL)**

| Check | Expected |
|---|---|
| number of Cards | `3`, components exactly `{text_encoder, transformer, vae}` |
| pipeline.pipeline_class | `QwenImagePipeline` |
| pipeline.components | contains `scheduler` and `tokenizer` with `has_weights == false`, and the 3 above with `has_weights == true` |
| text_encoder | `"Qwen2_5_VLForConditionalGeneration" in config.architectures`; a stack whose prefix ends with `layers` has depth `== config.raw["num_hidden_layers"]` (or `config.raw["text_config"]["num_hidden_layers"]` if that key exists; 28 at plan time); a stack whose prefix ends with `blocks` has depth `== config.raw["vision_config"]["depth"]` (32 at plan time); `expert_groups == []` |
| transformer | `config.class_name == "QwenImageTransformer2DModel"`; stack `transformer_blocks` depth `== config.raw["num_layers"]` (60 at plan time) |
| vae | `config.class_name == "AutoencoderKLQwenImage"`; exactly 1 weight file; `index_path == null` |
| all 3 Cards | same `key.revision_sha`; `model_card.license == "apache-2.0"` |

The hard-coded Qwen3 constants come from the published configs (hidden 4096 / 36
layers / 32 q-heads / 8 kv-heads / head_dim 128 / inter 12288 / vocab 151936,
untied; MoE: hidden 2048 / 48 layers / 32 q / 4 kv / 128 experts / moe_inter 768).
Qwen3 has no attention biases and does have q_norm/k_norm. If one of these
constants turns out wrong against the pinned revision, the check FAILs and a human
decides. Code is never tuned to make it pass.

### E4: CLI path (network, pinned)
```bash
scout scan "Qwen/Qwen3-8B@$(jq -r '."Qwen/Qwen3-8B"' exit/pins.json)" --out /tmp/cards
# expect: exit 0; stdout JSON has totals.weight == 0 and 1 card path; stderr shows
# live stage lines RESOLVE → HEADERS → META → REPORT in that order
```

### E5: human visual confirmation (non-automated, recorded in log.jsonl)
`scout serve`, open http://127.0.0.1:8765 and scan Qwen3-30B-A3B@pin. The page
must show all of the following:
- a byte counter with `weight 0` while the log streams
- the graph with a `layers ×48` node and an `experts ×128` node
- one depth strip of 48 cells
- the claimed `base_model` / `license` labelled "claimed by model card"

## 3. Definitions

### 3.1 Byte classes and "header-only"
Every byte that scout receives (HTTP body bytes yielded by `httpx.Response.iter_raw()`,
or bytes returned by a local `file.read()`) is recorded in one `fetch` or `retry`
event of the ByteLog. Each byte falls into exactly one class, computed by
`classify_range(path, start, end_exclusive, header_len)` from the file path and
byte offsets. The caller never supplies the class.

- **weight**
  - a byte at offset `>= 8 + N` of a `*.safetensors` file (where N is the u64 LE
    header length)
  - a byte at offset `>= 8` of a `*.safetensors` file whose N is not yet registered
  - any byte of a file with a weight extension other than `.safetensors`
    (`.bin .pt .pth .ckpt .gguf .h5 .msgpack .onnx .pb .npz .pkl`, case-insensitive)
- **header**: bytes `[0, 8)` of a `*.safetensors` file, plus `[8, 8 + N)` once N is
  registered with `ByteLog.set_header_len(path, N)`.
- **meta**: every other byte. This covers Hub API JSON responses (logged path
  `@api/revision`), `model_index.json`, `*.index.json`, `config.json` and
  `README.md`.
- HTTP response headers, TLS and redirect (3xx) bodies are not counted. Bytes the
  OS or TCP buffered but the client never yielded are also not counted. This is
  stated in the Card (`fetch_log.counting = "http-body-bytes-yielded"`).

**Zero weight bytes** means all three of the following for a scan:
- `ByteLog.totals["weight"] == 0`
- every event has `bytes_by_class["weight"] == 0`
- the persisted Card's `fetch_log.totals.weight == 0`

Enforcement in Phase 1 is stricter than a gate. `ByteLog.preflight()` raises
`WeightReadRefused` before any request whose range classifies as weight, so a
weight read is impossible by construction. Suppose a server ignores `Range` and
returns 200. The source then counts what it actually received (possibly weight
bytes, reported honestly), closes the stream and raises `RangeNotSupported`, and
no Card is written. A non-weight read is also refused (`ReadThresholdExceeded`)
when the running total plus the request length would exceed `threshold_bytes`
(default 64 MiB). The error reports the bytes requested, the running total, disk
use (0; nothing is written) and the reason. The interactive confirmation UI for
large reads is Phase 2 (download gate). Refusing is the safe P1 subset of
invariant 2.

### 3.2 Stages
The `Stage` enum follows the invariant order: RESOLVE=1, HEADERS=2, META=3,
SAMPLED_READ=4, FULL_DOWNLOAD=5, COMPUTE=6, COMPARE=7, REPORT=8, PURGE=9.
`ByteLog.stage()` raises `StageOrderError` if a scan tries to go backwards. P1
uses only these stages:
- **RESOLVE**: API revision call, `model_index.json`, `*.safetensors.index.json`.
  These are the manifests needed to locate headers.
- **HEADERS**: safetensors headers.
- **META**: `README.md` and each unit's `config.json`.
- **REPORT**: Card build and write.

PURGE is not emitted because nothing weight-bearing exists to purge. Headers live
in memory only.

### 3.3 Revision pinning and keys
- Hub target `owner/name[@rev]`. The source calls
  `GET {endpoint}/api/models/{repo}/revision/{rev or "main"}?blobs=true` and takes
  `sha`. Every file is then fetched from `/{repo}/resolve/{sha}/{path}`, so a scan
  can never mix revisions. If `rev` is 40-hex and differs from the returned sha,
  the source raises `RevisionMismatch`.
- Local folder target (an existing directory). The key is
  `repo = "local:" + abs_path`, and
  `revision_sha = sha256("\n".join(f"{relpath}\t{size}\t{mtime_ns}" sorted by relpath))`
  (64 hex), with `revision_kind = "local-content-hash"`. It is computed from
  `stat()` only, with no reads.
- A Card is keyed by `(key.repo, key.revision_sha, key.component)` and stored at
  `<out>/<repo_slug>/<revision_sha>/<component_slug>.card.json` and `.tensors.parquet`.
  - `repo_slug`: for Hub repos, `repo.replace("/", "__")`. For local folders,
    `"local__" + re.sub(r"[^A-Za-z0-9._-]", "_", abs_path.strip("/"))`.
  - `component_slug`: the component name, or `_model` when there is no component.

## 4. Data structures (exact field names)

### 4.1 LogEvent (ByteLog event, also embedded in Card and served to UI)
```
seq:int  ts:float(unix)  stage:str(Stage name)  event:str  source:"hub"|"local"|null
path:str|null  url:str|null  range:[start:int, end_exclusive:int]|null  status:int|null
bytes:int  bytes_by_class:{"meta":int,"header":int,"weight":int}  attempt:int
elapsed_ms:float  note:str|null  totals:{"meta":int,"header":int,"weight":int}  (running, after this event)
```
`event` is one of `stage_start`, `fetch` (successful read), `retry` (failed attempt;
its received bytes are still counted), `refused`, `error`, `card_written`.

### 4.2 Card v0 JSON (`*.card.json`)
```
schema_version: "card.v0"
key:        {repo:str, revision_sha:str, component:str|null}
source:     {kind:"hub"|"local", requested_revision:str|null, revision_kind:"git"|"local-content-hash",
             local_path:str|null, endpoint:str|null}
scan:       {scanned_at:str(ISO-8601 UTC, "Z"), scout_version:str, elapsed_s:float}
model_card: {present:bool, claimed:true, base_model:[str], base_model_relation:str|null,
             license:str|null, license_name:str|null, tags:[str], pipeline_tag:str|null,
             library_name:str|null, parse_error:str|null}
config:     {present:bool, path:str, model_type:str|null, architectures:[str],
             class_name:str|null, raw:object|null, parse_error:str|null}
pipeline:   null | {pipeline_class:str|null, model_index_path:"model_index.json",
             components:[{name:str, library:str|null, class_name:str|null, has_weights:bool}]}
weights:    {format:"safetensors", index_path:str|null, index_total_size:int|null,
             files:[{path:str, size_bytes:int|null, header_len:int, data_bytes:int, n_tensors:int,
                     metadata:object}],
             n_tensors:int, params_total:int, tensor_bytes_total:int, params_by_dtype:{str:int}}
structure:  {stacks:[{prefix:str, depth:int, indices:[int], block_params:[int],
                      block_signatures:[str], block_moe:[bool], block_n_experts:[int|null]}],
             expert_groups:[{template:str, n_experts:int, n_instances:int, params_per_expert:int,
                             tensors_per_expert:int, homogeneous:bool}],
             warnings:[str]}
stats:      {tensor_stats:null, spectral_topk:null, sigma_curves:null, tokenizer_minhash:null,
             attribution:null}                      # slots present, empty in v0
fetch_log:  {counting:"http-body-bytes-yielded", threshold_bytes:int,
             totals:{meta:int, header:int, weight:int}, events:[LogEvent]}
tensors_parquet: str                                 # sibling file name
```
- `fetch_log` holds the whole scan's log up to the REPORT `stage_start` event. It
  is the same for every component Card of one scan.
- `model_card` for a pipeline component comes from the repo-root `README.md`.
- `config` for a component comes from `<component>/config.json`.

### 4.3 Card v0 Parquet (`*.tensors.parquet`, one row per tensor)
| column | arrow type | notes |
|---|---|---|
| repo | string | |
| revision_sha | string | |
| component | string | `""` for root |
| name | string | original tensor name |
| collapsed_name | string | stack index → `#`, expert index → `*` |
| file | string | repo-relative path |
| dtype | string | safetensors dtype |
| shape | list<int64> | |
| numel | int64 | |
| nbytes | int64 | `data_end - data_begin` |
| data_begin | int64 | offset within data section |
| data_end | int64 | |
| stack_prefix | string (nullable) | |
| block_index | int32 (nullable) | |
| expert_index | int32 (nullable) | |
| stat_mean | float64 (nullable) | null in v0 |
| stat_std | float64 (nullable) | null in v0 |
| stat_fro_norm | float64 (nullable) | null in v0 |
| stat_spectral_topk | list<float64> (nullable) | null in v0 |
| stat_sigma_curve | list<float64> (nullable) | null in v0 |

Parquet file key-value metadata:
`{"schema_version":"card.v0","repo":…,"revision_sha":…,"component":…}`.
Rows are sorted by `(file, data_begin)`.

### 4.4 View (served, never persisted; built from the persisted Card by `build_view`)
```
{card_key:{repo,revision_sha,component},
 title:str,
 model_card:{…same as Card…},
 nodes:[{id:str, label:str, kind:"root"|"module"|"stack"|"expert_group"|"tensor",
         parent:str|null, params:int, count:int, shape:[int]|null, dtype:str|null}],
 depth_strips:[{prefix:str, depth:int,
                cells:[{index:int, params:int, signature:str, moe:bool, n_experts:int|null}]}],
 summary:{params_total:int, n_tensors:int, n_stacks:int, n_expert_groups:int,
          weight_bytes_read:int, header_bytes_read:int, meta_bytes_read:int}}
```

## 5. Assumptions and open decisions (each with a recommended default)

| # | Decision | Default (recommended) | Rationale |
|---|---|---|---|
| D1 | **Frontend stack** | No-build vanilla stack: a single `index.html` + `app.js` (ES2020, inline SVG, no npm, no CDN) served by a Python stdlib `http.server.ThreadingHTTPServer`. The page polls the server every 250 ms for the live log. | This is a self-use tool for one person on localhost. The graph has ≤100 nodes and the strips ≤ ~100 cells, so SVG by hand is enough. No toolchain means nothing to install or keep in sync, and implementers and reviewers stay in Python. Revisit in P2 only if the diff view needs real interaction (then: Svelte + Vite). Rejected: React/Vite (toolchain), Streamlit/Gradio (weak control over live log and custom SVG), FastAPI (an extra dependency for 3 routes). |
| D2 | Diffusion exit repo | `Qwen/Qwen-Image` (diffusers; `model_index.json`; sharded `text_encoder` = Qwen2.5-VL, sharded `transformer`, single-file `vae`; Apache-2.0, not gated) | It exercises sharded + multi-component together and matches the project's motivating case (a Qwen-derived text encoder). Fallback if the repo layout breaks the rules: `stabilityai/stable-diffusion-xl-base-1.0`, with expected components `{text_encoder, text_encoder_2, unet, vae}` and `stack down_blocks` depth 3. |
| D3 | Pinning when the Hub is unreachable here | Pins are resolved at EXIT with `scout resolve` on a host that can reach huggingface.co and its CDNs (`huggingface.co`, `cdn-lfs*.hf.co`, `cas-bridge.xethub.hf.co`), then committed. Implementation and the offline suite need no network. | The planning container's egress proxy rejected those hosts. A human must either allowlist them or run E1–E4 locally. |
| D4 | HTTP client | Own thin `httpx` client, not `huggingface_hub` | Every body byte must be counted and classified at the source. `huggingface_hub` downloads through its own cache and would write files (which violates invariant 1) and hide byte counts. |
| D5 | Read threshold | `64 MiB` of non-weight bytes per scan (`--max-read-bytes`). Weight reads are always refused in P1. | Qwen3-30B-A3B headers are ≈2–3 MB in total. 64 MiB leaves 20× headroom and still stops pathological headers. |
| D6 | Log persistence | The log is embedded in each Card (`fetch_log`) and streamed live to stderr (CLI) or `/api/scans/{id}` (UI). No separate log file. | Invariant 1: the Card is the only persisted artifact. |
| D7 | Meaning of "graph" in P1 | A collapsed module-hierarchy graph (tree from tensor names: stacks → `×depth`, experts → `×n`). It is not a dataflow graph. | Headers give names, shapes and dtypes, not dataflow. A dataflow graph would need model code (later.md). |
| D8 | Gated / private repo | Fail with `GatedRepoError` (exit code 3); the message includes the repo URL and whether `HF_TOKEN` was set. No partial Card. | A Card without headers would be an incomplete artifact. The token comes from the `HF_TOKEN` env variable only. |
| D9 | Component Cards | One Card per `model_index.json` component that has safetensors in its folder. Components without weights (scheduler, tokenizer) are listed in `pipeline.components` with `has_weights:false`. No pipeline-level Card. | This is the minimum that satisfies "one Card per component". |
| D10 | Weight file selection per unit | In this order: `model.safetensors.index.json`, `diffusion_pytorch_model.safetensors.index.json`, `model.safetensors`, `diffusion_pytorch_model.safetensors`; then exactly one other `*.safetensors.index.json`; then exactly one other `*.safetensors`. Anything else is `AmbiguousWeightsError`. No safetensors at all is `NoSafetensorsError` (`.bin`-only repos are unsupported in P1 because they cannot be read header-only). | Deterministic. Variant files (`*.fp16.safetensors`) are ignored when a canonical file exists. |
| D11 | Network failure policy | 3 attempts per request with backoff 0.5 s and 1.0 s, 10 s timeout. A failed attempt's received bytes are logged as a `retry` event. Once attempts run out: `NetworkError` (exit code 4), and no Card is written (all Cards are built in memory and written only after every unit succeeds). | The simplest option that is honest and atomic. |
| D12 | MoE detection | The segment `experts` followed by an integer segment. Fused 3-D expert tensors (gpt-oss style) and `shared_expert` naming are later work. | Covers Qwen3-MoE, Qwen2-MoE and Mixtral (`block_sparse_moe.experts.N`). |

## 6. Architecture (files)

```
pyproject.toml                         T001  deps: httpx, pyarrow, pyyaml; dev: pytest
scout/__init__.py                      T001  __version__ = "0.1.0"
scout/errors.py, scout/bytelog.py      T002
scout/sources.py (protocol + local)    T004
scout/hub.py                           T005
scout/safetensors_header.py            T006
scout/metadata.py                      T007  model card, config, model_index parsing
scout/structure.py                     T008  stacks, MoE collapsing, signatures
scout/card.py                          T009  Card v0 build / write / load
scout/scan.py                          T010  intake orchestration
scout/view.py                          T011
scout/cli.py                           T012
scout/server.py                        T013
scout/web/index.html, scout/web/app.js T014
scripts/exit_check.py, exit/pins.json, tests/test_exit_network.py  T015
tests/helpers/st_fixtures.py           T001  synthetic safetensors repos
tests/helpers/fakehub.py               T003  httpx.MockTransport fake Hub + CDN
```

## 7. Inputs handled and where each is tested (all offline)

| Input | Handling | Test |
|---|---|---|
| Sharded (`*.index.json`) | Read the index in RESOLVE; headers of all shards in parallel; check `total_size` | T010 `test_scan_dense_sharded_hub` |
| Local folder | `LocalSource`, stat-based key, same classification | T004 tests, T010 `test_scan_local_folder` |
| Multi-component (`model_index.json`) | One Card per weight-bearing component; shared README | T010 `test_scan_pipeline` |
| MoE | `experts.N` → `experts.*`, one expert group, strips flag MoE | T008, T010 `test_scan_moe` |
| Missing model card | `model_card.present=false`, scan succeeds | T007, T010 `test_scan_no_readme` |
| Gated repo | 401 → `GatedRepoError`, with token → success | T005, T010 |
| Network failure mid-read | reset after k bytes → retry counted → success; persistent → `NetworkError`, no files written | T005, T010 |
| Range ignored by server | bytes counted honestly, `RangeNotSupported`, no Card | T005 |
| `.bin`-only repo | `NoSafetensorsError`, 0 weight bytes | T010 |
| Weight read attempt | `WeightReadRefused` before any request is sent | T002, T005 |
| Oversized header / threshold | `ReadThresholdExceeded`, no Card | T002, T010 |
| Lying `base_model` | Recorded verbatim as `claimed`; no verification in P1 (P3) | T007 |

## 8. Risks and how each is tested

| Risk | Mitigation / test |
|---|---|
| A weight byte slips through (the invariant fails) | Preflight refusal (T002 unit test). FakeHub counts CDN request ranges, and T010 asserts that every served range ends at or before `8+N`. Exit C2 and C3 (header bytes exactly Σ(8+N)). |
| Server ignores Range or CDN returns 200 | T005 `test_range_ignored` checks honest counting plus an error. C3 would catch it live. |
| Xet/CDN redirect drops Range or needs auth | httpx follows redirects and resends Range; auth is dropped cross-origin by httpx. FakeHub redirects every LFS read to a separate host (T003/T005). Live: E1–E3. |
| 10 s budget on MoE (16 shards × 2 requests + redirect) | Headers are fetched with a thread pool (8 workers) and one shared `httpx.Client` (keep-alive). Budget ≈ 1 API call + 16×2 ranged requests ≈ 3–5 s. C1 measures it. |
| Hub unreachable from build container | Offline suite with FakeHub; exit run on a host that has access (D3). |
| Wrong hard-coded constants | Cross-checked against `config.json` in the same Card; a mismatch FAILs and goes to a human, never gets tuned away. |
| Partial Cards on failure | Cards are built in memory and written only after all units succeed, each by atomic temp+`os.replace`. T010 asserts the out dir stays empty on every error path. |
| Weights persisted to disk | No `huggingface_hub`, no cache. T010 `test_no_extra_files` asserts only `*.card.json`/`*.tensors.parquet` exist under out dir and that `HOME`/tmp (monkeypatched) gain no files. |
| Collapsing wrong (e.g. nested indices in UNet) | T008 fixtures: nested `down_blocks.0.resnets.1`, non-contiguous indices, heterogeneous experts produce a warning. |
| Pin drift | Requests use a 40-hex sha; `RevisionMismatch` test in T005. |

## 9. Task list

| id | title | route | depends_on |
|---|---|---|---|
| T001 | Scaffold + synthetic safetensors fixture writer | sonnet | – |
| T002 | ByteLog, byte classification, errors | opus | T001 |
| T003 | FakeHub test transport | sonnet | T001 |
| T004 | Source protocol + LocalSource | opus | T002 |
| T005 | HubSource (resolve, ranged reads, retries, gated) | opus | T002, T003, T004 |
| T006 | Safetensors header reader + weight-file selection | sonnet | T004 |
| T007 | Model card / config / model_index parsing | sonnet | T001 |
| T008 | Structure analysis (stacks, MoE collapse, signatures) | opus | T006 |
| T009 | Card v0 build / write / load | opus | T002, T006, T007, T008 |
| T010 | Scan orchestration (intake) | opus | T005, T009 |
| T011 | View model | sonnet | T009 |
| T012 | CLI (`scan`, `resolve`, `serve`) | sonnet | T010, T013 |
| T013 | HTTP server (stdlib) | sonnet | T010, T011 |
| T014 | Frontend (index.html + app.js) | opus | T013 |
| T015 | Exit check script, pins, network test | sonnet | T012, T013 |

Tasks that can run in parallel (no shared files): {T002, T003, T007}, then
{T004 ∥ T007}, and {T011 ∥ T010}.

## 10. Out of scope for Phase 1
Everything listed in `later.md`, notably:
- download gate UI and confirmation
- tokenizer reading
- sigma-curves or any weight statistics
- Card cache/store
- GGUF, `.bin` and quantized formats
- dataflow graphs
