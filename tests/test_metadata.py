import json

import pytest

from scout.metadata import parse_config, parse_model_card, parse_model_index
from tests.helpers.st_fixtures import DEFAULT_README, write_pipeline_repo


def test_card_full():
    c = parse_model_card(DEFAULT_README.encode())
    assert c.present and c.parse_error is None
    assert c.license == "apache-2.0"
    assert c.base_model == ["Org/Base-Model"]
    assert c.tags == ["test"]
    assert c.pipeline_tag == "text-generation"
    assert c.library_name == "transformers"


def test_card_base_model_str():
    c = parse_model_card(b"---\nbase_model: Org/X\n---\n")
    assert c.base_model == ["Org/X"]


def test_card_missing():
    c = parse_model_card(None)
    assert c.present is False
    assert c.base_model == [] and c.tags == [] and c.license is None


def test_card_no_front_matter():
    c = parse_model_card(b"# hi")
    assert c.present is True and c.parse_error is None
    assert c.base_model == [] and c.tags == [] and c.license is None


def test_card_bad_yaml():
    c = parse_model_card(b"---\n: [\n---")
    assert c.present is True
    assert c.parse_error


def test_card_unterminated_and_non_mapping():
    assert parse_model_card(b"---\nlicense: mit\n").parse_error == "unterminated front matter"
    assert parse_model_card(b"---\n- a\n- b\n---\n").parse_error == "front matter is not a mapping"


def test_card_license_list_and_relation():
    c = parse_model_card(b"---\nlicense: [a, b]\nlicense_name: N\nbase_model_relation: finetune\n---\n")
    assert c.license == "a,b" and c.license_name == "N" and c.base_model_relation == "finetune"


def test_card_lying_base_model():
    c = parse_model_card(b"---\nbase_model: meta-llama/Llama-3-8B\n---\n")
    assert c.base_model == ["meta-llama/Llama-3-8B"]
    d = c.to_dict()
    assert d["claimed"] is True
    assert d["base_model"] == ["meta-llama/Llama-3-8B"]


def test_config():
    q = parse_config(json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}).encode(),
                     "config.json")
    assert q.present and q.model_type == "qwen3" and q.architectures == ["Qwen3ForCausalLM"]
    assert q.class_name is None and q.raw["model_type"] == "qwen3" and q.path == "config.json"
    d = parse_config(b'{"_class_name": "AutoencoderKL"}', "vae/config.json")
    assert d.class_name == "AutoencoderKL" and d.model_type is None and d.architectures == []
    bad = parse_config(b"{nope", "config.json")
    assert bad.present and bad.parse_error and bad.raw is None
    assert parse_config(b"[1]", "config.json").parse_error
    none = parse_config(None, "config.json")
    assert none.present is False and none.path == "config.json"
    assert isinstance(q.to_dict(), dict)


def test_model_index(tmp_path):
    write_pipeline_repo(tmp_path)
    info = parse_model_index((tmp_path / "model_index.json").read_bytes())
    assert info.pipeline_class == "TestPipeline"
    assert [c.name for c in info.components] == ["scheduler", "text_encoder", "tokenizer", "transformer", "vae"]
    assert info.components[1].library == "transformers"
    assert info.components[1].class_name == "Qwen3ForCausalLM"
    assert info.to_dict()["model_index_path"] == "model_index.json"


def test_model_index_invalid():
    with pytest.raises(ValueError, match="model_index.json"):
        parse_model_index(b"{bad")
    with pytest.raises(ValueError):
        parse_model_index(b"[]")
