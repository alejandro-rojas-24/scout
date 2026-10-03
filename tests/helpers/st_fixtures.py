"""Synthetic safetensors fixture writer (no real weights)."""
from __future__ import annotations

import json
import math
import pathlib

DTYPE_BYTES: dict[str, int] = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
FILL_BYTE: int = 0xAB
TensorSpec = tuple[str, tuple[int, ...]]  # (dtype, shape)

DEFAULT_README: str = "---\nlicense: apache-2.0\nbase_model:\n- Org/Base-Model\ntags:\n- test\npipeline_tag: text-generation\nlibrary_name: transformers\n---\n# Test model\n"


def _nbytes(spec: TensorSpec) -> int:
    dtype, shape = spec
    return math.prod(shape) * DTYPE_BYTES[dtype]


def safetensors_bytes(tensors: dict[str, TensorSpec], metadata: dict[str, str] | None = None) -> bytes:
    header: dict = {}
    cursor = 0
    for name, spec in tensors.items():
        nbytes = _nbytes(spec)
        header[name] = {"dtype": spec[0], "shape": list(spec[1]), "data_offsets": [cursor, cursor + nbytes]}
        cursor += nbytes
    if metadata is not None:
        header["__metadata__"] = metadata
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    while len(hb) % 8 != 0:
        hb += b" "
    return len(hb).to_bytes(8, "little") + hb + bytes([FILL_BYTE]) * cursor


def write_safetensors(path: pathlib.Path, tensors: dict[str, TensorSpec], metadata: dict[str, str] | None = None) -> int:
    data = safetensors_bytes(tensors, metadata)
    pathlib.Path(path).write_bytes(data)
    return int.from_bytes(data[:8], "little")


def params_of(tensors: dict[str, TensorSpec]) -> int:
    return sum(math.prod(shape) for _, shape in tensors.values())


def _attn(t: dict, p: str, dtype: str, hidden: int, heads: int, kv_heads: int, head_dim: int) -> None:
    t[f"{p}.self_attn.q_proj.weight"] = (dtype, (heads * head_dim, hidden))
    t[f"{p}.self_attn.k_proj.weight"] = (dtype, (kv_heads * head_dim, hidden))
    t[f"{p}.self_attn.v_proj.weight"] = (dtype, (kv_heads * head_dim, hidden))
    t[f"{p}.self_attn.o_proj.weight"] = (dtype, (hidden, heads * head_dim))
    t[f"{p}.self_attn.q_norm.weight"] = (dtype, (head_dim,))
    t[f"{p}.self_attn.k_norm.weight"] = (dtype, (head_dim,))


def _norms(t: dict, p: str, dtype: str, hidden: int) -> None:
    t[f"{p}.input_layernorm.weight"] = (dtype, (hidden,))
    t[f"{p}.post_attention_layernorm.weight"] = (dtype, (hidden,))


def dense_tensors(n_layers: int = 2, hidden: int = 8, inter: int = 16, vocab: int = 32,
                  heads: int = 2, kv_heads: int = 1, head_dim: int = 4, tie: bool = False,
                  dtype: str = "BF16", prefix: str = "model") -> dict[str, TensorSpec]:
    t: dict[str, TensorSpec] = {f"{prefix}.embed_tokens.weight": (dtype, (vocab, hidden))}
    for i in range(n_layers):
        p = f"{prefix}.layers.{i}"
        _attn(t, p, dtype, hidden, heads, kv_heads, head_dim)
        t[f"{p}.mlp.gate_proj.weight"] = (dtype, (inter, hidden))
        t[f"{p}.mlp.up_proj.weight"] = (dtype, (inter, hidden))
        t[f"{p}.mlp.down_proj.weight"] = (dtype, (hidden, inter))
        _norms(t, p, dtype, hidden)
    t[f"{prefix}.norm.weight"] = (dtype, (hidden,))
    if not tie:
        t["lm_head.weight"] = (dtype, (vocab, hidden))
    return t


def moe_tensors(n_layers: int = 2, n_experts: int = 4, hidden: int = 8, moe_inter: int = 4,
                vocab: int = 32, heads: int = 2, kv_heads: int = 1, head_dim: int = 4,
                dtype: str = "BF16") -> dict[str, TensorSpec]:
    t: dict[str, TensorSpec] = {"model.embed_tokens.weight": (dtype, (vocab, hidden))}
    for i in range(n_layers):
        p = f"model.layers.{i}"
        _attn(t, p, dtype, hidden, heads, kv_heads, head_dim)
        t[f"{p}.mlp.gate.weight"] = (dtype, (n_experts, hidden))
        for j in range(n_experts):
            t[f"{p}.mlp.experts.{j}.gate_proj.weight"] = (dtype, (moe_inter, hidden))
            t[f"{p}.mlp.experts.{j}.up_proj.weight"] = (dtype, (moe_inter, hidden))
            t[f"{p}.mlp.experts.{j}.down_proj.weight"] = (dtype, (hidden, moe_inter))
        _norms(t, p, dtype, hidden)
    t["model.norm.weight"] = (dtype, (hidden,))
    t["lm_head.weight"] = (dtype, (vocab, hidden))
    return t


def write_sharded(root: pathlib.Path, tensors: dict[str, TensorSpec], n_shards: int,
                  stem: str = "model") -> dict:
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    items = list(tensors.items())
    total_size = sum(_nbytes(s) for s in tensors.values())
    if n_shards == 1:
        name = f"{stem}.safetensors"
        n = write_safetensors(root / name, dict(items))
        return {"files": [name], "header_lens": {name: n}, "total_size": total_size}
    base, extra = divmod(len(items), n_shards)
    files: list[str] = []
    header_lens: dict[str, int] = {}
    weight_map: dict[str, str] = {}
    pos = 0
    for k in range(n_shards):
        size = base + (1 if k < extra else 0)
        chunk = dict(items[pos : pos + size])
        pos += size
        name = f"{stem}-{k + 1:05d}-of-{n_shards:05d}.safetensors"
        header_lens[name] = write_safetensors(root / name, chunk)
        files.append(name)
        for tn in chunk:
            weight_map[tn] = name
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (root / f"{stem}.safetensors.index.json").write_text(json.dumps(index, indent=2))
    return {"files": files, "header_lens": header_lens, "total_size": total_size}


def _finish_repo(root: pathlib.Path, config: dict, tensors: dict[str, TensorSpec],
                 n_shards: int, readme: str | None) -> dict:
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config))
    if readme is not None:
        (root / "README.md").write_text(readme)
    info = write_sharded(root, tensors, n_shards, stem="model")
    info.update({"params_total": params_of(tensors), "n_tensors": len(tensors), "tensors": tensors})
    return info


def write_dense_repo(root: pathlib.Path, n_shards: int = 2, readme: str | None = DEFAULT_README,
                     **dense_kwargs) -> dict:
    tensors = dense_tensors(**dense_kwargs)
    kw = dense_kwargs
    config = {
        "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
        "num_hidden_layers": kw.get("n_layers", 2), "hidden_size": kw.get("hidden", 8),
        "intermediate_size": kw.get("inter", 16), "vocab_size": kw.get("vocab", 32),
        "num_attention_heads": kw.get("heads", 2), "num_key_value_heads": kw.get("kv_heads", 1),
        "head_dim": kw.get("head_dim", 4), "tie_word_embeddings": kw.get("tie", False),
    }
    return _finish_repo(root, config, tensors, n_shards, readme)


def write_moe_repo(root: pathlib.Path, n_shards: int = 2, readme: str | None = DEFAULT_README,
                   **moe_kwargs) -> dict:
    tensors = moe_tensors(**moe_kwargs)
    kw = moe_kwargs
    config = {
        "model_type": "qwen3_moe", "architectures": ["Qwen3MoeForCausalLM"],
        "num_hidden_layers": kw.get("n_layers", 2), "hidden_size": kw.get("hidden", 8),
        "moe_intermediate_size": kw.get("moe_inter", 4), "num_experts": kw.get("n_experts", 4),
        "num_experts_per_tok": 2, "vocab_size": kw.get("vocab", 32),
        "num_attention_heads": kw.get("heads", 2), "num_key_value_heads": kw.get("kv_heads", 1),
        "head_dim": kw.get("head_dim", 4),
    }
    return _finish_repo(root, config, tensors, n_shards, readme)


def _vae_tensors() -> dict[str, TensorSpec]:
    W, G = ("F32", (2, 2)), ("F32", (2,))
    t: dict[str, TensorSpec] = {"encoder.conv_in.weight": W, "encoder.conv_in.bias": G}
    for i in range(4):
        p = f"encoder.down_blocks.{i}"
        t[f"{p}.norm1.gamma"] = G
        t[f"{p}.conv1.weight"] = W
        t[f"{p}.conv1.bias"] = G
        t[f"{p}.norm2.gamma"] = G
        t[f"{p}.conv2.weight"] = W
        t[f"{p}.conv2.bias"] = G
        if i in (0, 1, 2):
            t[f"{p}.resample.1.weight"] = W
    for r in range(2):
        t[f"encoder.mid_block.resnets.{r}.conv1.weight"] = W
        t[f"encoder.mid_block.resnets.{r}.conv2.weight"] = W
    t["encoder.mid_block.attentions.0.to_qkv.weight"] = W
    t["encoder.mid_block.attentions.0.proj.weight"] = W
    for r in range(2):
        t[f"decoder.mid_block.resnets.{r}.conv1.weight"] = W
    for i in range(4):
        for r in range(3):
            p = f"decoder.up_blocks.{i}.resnets.{r}"
            t[f"{p}.norm1.gamma"] = G
            t[f"{p}.conv1.weight"] = W
            t[f"{p}.conv1.bias"] = G
            t[f"{p}.conv2.weight"] = W
        if i in (0, 1, 2):
            t[f"decoder.up_blocks.{i}.upsamplers.0.resample.1.weight"] = W
            t[f"decoder.up_blocks.{i}.upsamplers.0.time_conv.weight"] = W
    t["decoder.conv_out.weight"] = W
    t["quant_conv.weight"] = W
    t["post_quant_conv.weight"] = W
    return t


def write_pipeline_repo(root: pathlib.Path, readme: str | None = DEFAULT_README) -> dict:
    root = pathlib.Path(root)
    root.mkdir(parents=True, exist_ok=True)
    model_index = {
        "_class_name": "TestPipeline", "_diffusers_version": "0.35.0",
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "Qwen3ForCausalLM"],
        "tokenizer": ["transformers", "Qwen2Tokenizer"],
        "transformer": ["diffusers", "TestTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKL"],
        "safety_checker": [None, None],
    }
    (root / "model_index.json").write_text(json.dumps(model_index))
    (root / "scheduler").mkdir(exist_ok=True)
    (root / "scheduler" / "scheduler_config.json").write_text(
        json.dumps({"_class_name": "FlowMatchEulerDiscreteScheduler"}))
    (root / "tokenizer").mkdir(exist_ok=True)
    (root / "tokenizer" / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 16}))

    te = write_dense_repo(root / "text_encoder", n_shards=2, readme=None)

    (root / "transformer").mkdir(exist_ok=True)
    (root / "transformer" / "config.json").write_text(
        json.dumps({"_class_name": "TestTransformer2DModel", "num_layers": 3}))
    tt: dict[str, TensorSpec] = {}
    for i in range(3):
        p = f"transformer_blocks.{i}"
        tt[f"{p}.attn.to_q.weight"] = ("BF16", (8, 8))
        tt[f"{p}.attn.to_k.weight"] = ("BF16", (8, 8))
        tt[f"{p}.ff.net.0.proj.weight"] = ("BF16", (16, 8))
        tt[f"{p}.ff.net.2.weight"] = ("BF16", (8, 16))
    tt["proj_out.weight"] = ("BF16", (4, 8))
    tr = write_sharded(root / "transformer", tt, 1, stem="diffusion_pytorch_model")
    tr.update({"params_total": params_of(tt), "n_tensors": len(tt), "tensors": tt})

    (root / "vae").mkdir(exist_ok=True)
    (root / "vae" / "config.json").write_text(json.dumps({"_class_name": "AutoencoderKLQwenImage"}))
    vt = _vae_tensors()
    vae = write_sharded(root / "vae", vt, 1, stem="diffusion_pytorch_model")
    vae.update({"params_total": params_of(vt), "n_tensors": len(vt), "tensors": vt})

    if readme is not None:
        (root / "README.md").write_text(readme)
    return {"components": {"text_encoder": te, "transformer": tr, "vae": vae}}
