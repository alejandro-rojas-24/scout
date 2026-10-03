import json

from tests.helpers.st_fixtures import (
    DTYPE_BYTES,
    FILL_BYTE,
    dense_tensors,
    moe_tensors,
    params_of,
    safetensors_bytes,
    write_dense_repo,
    write_pipeline_repo,
)


def test_roundtrip_header():
    tensors = {"a": ("F32", (2, 3)), "b": ("BF16", (4,)), "c": ("I8", ())}
    data = safetensors_bytes(tensors, {"format": "pt"})
    n = int.from_bytes(data[:8], "little")
    assert n % 8 == 0
    header = json.loads(data[8 : 8 + n])
    assert header["__metadata__"] == {"format": "pt"}
    assert list(k for k in header if k != "__metadata__") == ["a", "b", "c"]
    cursor = 0
    for name, (dtype, shape) in tensors.items():
        s, e = header[name]["data_offsets"]
        assert s == cursor
        size = DTYPE_BYTES[dtype]
        for d in shape:
            size *= d
        assert e - s == size
        cursor = e
    assert len(data) == 8 + n + cursor
    assert data[8 + n :] == bytes([FILL_BYTE]) * cursor


def test_dense_counts():
    assert len(dense_tensors(n_layers=2)) == 2 * 11 + 3
    assert len(dense_tensors(n_layers=2, tie=True)) == 24


def test_moe_counts():
    assert len(moe_tensors(n_layers=2, n_experts=4)) == 2 * (8 + 1 + 12) + 3


def test_sharded_index(tmp_path):
    info = write_dense_repo(tmp_path, n_shards=2)
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    tensors = info["tensors"]
    assert set(index["weight_map"]) == set(tensors)
    assert set(index["weight_map"].values()) == set(info["files"])
    total = sum(
        __import__("math").prod(s) * DTYPE_BYTES[d] for d, s in tensors.values()
    )
    assert index["metadata"]["total_size"] == total == info["total_size"]
    assert info["params_total"] == params_of(tensors)
    assert info["n_tensors"] == len(tensors)


def test_pipeline_layout(tmp_path):
    info = write_pipeline_repo(tmp_path)
    for rel in [
        "model_index.json",
        "README.md",
        "scheduler/scheduler_config.json",
        "tokenizer/tokenizer_config.json",
        "text_encoder/config.json",
        "text_encoder/model.safetensors.index.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ]:
        assert (tmp_path / rel).exists(), rel
    assert json.loads((tmp_path / "model_index.json").read_text())["_class_name"] == "TestPipeline"
    assert set(info["components"]) == {"text_encoder", "transformer", "vae"}
