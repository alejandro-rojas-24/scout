"""Parsing of model card front matter, config.json and model_index.json.

All card values are claims, recorded verbatim; nothing is verified here.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import yaml


@dataclass(frozen=True)
class ModelCardInfo:
    present: bool
    base_model: list[str]
    base_model_relation: str | None
    license: str | None
    license_name: str | None
    tags: list[str]
    pipeline_tag: str | None
    library_name: str | None
    parse_error: str | None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["claimed"] = True
        return d


@dataclass(frozen=True)
class ConfigInfo:
    present: bool
    path: str
    model_type: str | None
    architectures: list[str]
    class_name: str | None
    raw: dict | None
    parse_error: str | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PipelineComponent:
    name: str
    library: str | None
    class_name: str | None
    has_weights: bool = False


@dataclass
class PipelineInfo:
    pipeline_class: str | None
    components: list[PipelineComponent] = field(default_factory=list)
    model_index_path: str = "model_index.json"

    def to_dict(self) -> dict:
        return asdict(self)


def _empty_card(present: bool, parse_error: str | None = None) -> ModelCardInfo:
    return ModelCardInfo(present, [], None, None, None, [], None, None, parse_error)


def _opt_str(v) -> str | None:
    return str(v) if v is not None else None


def parse_model_card(raw: bytes | None) -> ModelCardInfo:
    if raw is None:
        return _empty_card(False)
    text = raw.decode("utf-8", errors="replace").lstrip("﻿")
    lines = text.splitlines()
    first = next((i for i, ln in enumerate(lines) if ln.strip()), None)
    if first is None or lines[first].strip() != "---":
        return _empty_card(True)
    end = next((i for i in range(first + 1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return _empty_card(True, "unterminated front matter")
    try:
        data = yaml.safe_load("\n".join(lines[first + 1:end]))
    except yaml.YAMLError as e:
        return _empty_card(True, f"{type(e).__name__}: {e}")
    if not isinstance(data, dict):
        return _empty_card(True, "front matter is not a mapping")

    bm = data.get("base_model")
    if isinstance(bm, str):
        base_model = [bm.strip()]
    elif isinstance(bm, list):
        base_model = [str(x).strip() for x in bm if x is not None]
    else:
        base_model = []
    tg = data.get("tags")
    if isinstance(tg, list):
        tags = [str(x) for x in tg]
    elif isinstance(tg, str):
        tags = [tg]
    else:
        tags = []
    lic = data.get("license")
    license_ = ",".join(str(x) for x in lic) if isinstance(lic, list) else _opt_str(lic)
    return ModelCardInfo(
        present=True,
        base_model=base_model,
        base_model_relation=_opt_str(data.get("base_model_relation")),
        license=license_,
        license_name=_opt_str(data.get("license_name")),
        tags=tags,
        pipeline_tag=_opt_str(data.get("pipeline_tag")),
        library_name=_opt_str(data.get("library_name")),
        parse_error=None,
    )


def parse_config(raw: bytes | None, path: str) -> ConfigInfo:
    if raw is None:
        return ConfigInfo(False, path, None, [], None, None, None)
    try:
        j = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as e:
        return ConfigInfo(True, path, None, [], None, None, f"{type(e).__name__}: {e}")
    if not isinstance(j, dict):
        return ConfigInfo(True, path, None, [], None, None, "config is not a JSON object")
    arch = j.get("architectures")
    architectures = [str(x) for x in arch] if isinstance(arch, list) else []
    return ConfigInfo(True, path, _opt_str(j.get("model_type")), architectures,
                      _opt_str(j.get("_class_name")), j, None)


def parse_model_index(raw: bytes) -> PipelineInfo:
    try:
        j = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as e:
        raise ValueError(f"model_index.json: {e}") from e
    if not isinstance(j, dict):
        raise ValueError("model_index.json: not a JSON object")
    components: list[PipelineComponent] = []
    for key, value in j.items():
        if key.startswith("_"):
            continue
        if isinstance(value, list) and len(value) == 2:
            if value[0] is None and value[1] is None:
                continue
            components.append(PipelineComponent(key, value[0], value[1]))
    return PipelineInfo(pipeline_class=j.get("_class_name"), components=components)
