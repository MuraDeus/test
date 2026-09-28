"""
config_manager.py
영구 설정 파일 관리 모듈 (config.json / presets.json / characters.json)
Termux 환경에서 앱 재시작 후에도 데이터 보존
"""
import json
import os
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).parent.resolve()
CONFIG_FILE = BASE_DIR / "config.json"
PRESETS_FILE = BASE_DIR / "presets.json"
CHARACTERS_FILE = BASE_DIR / "characters.json"
OUTPUTS_DIR = BASE_DIR / "outputs"

DEFAULT_CONFIG: dict[str, Any] = {
    "llm": {
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "api_key": "",
        "model": "gpt-4o-mini",
        "temperature": 0.7
    },
    "nai": {
        "api_key": "",
        "model": "nai-diffusion-5-full"
    }
}

DEFAULT_PRESET: dict[str, Any] = {
    "name": "기본 애니",
    "positive_prompt": "best quality, amazing quality, very aesthetic, absurdres",
    "negative_prompt": (
        "lowres, {bad}, error, fewer, extra, missing, worst quality, jpeg artifacts, "
        "bad quality, watermark, unfinished, displeasing, chromatic aberration, signature, "
        "extra digits, artistic error, username, scan, [abstract]"
    ),
    "sampler": "k_euler_ancestral",
    "steps": 28,
    "scale": 6.0,
    "width": 832,
    "height": 1216,
    "noise_schedule": "karras",
    "cfg_rescale": 0.0,
    "smea": False,
    "smea_dyn": False
}


def _ensure_dirs() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        _save_json(path, default)
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        _save_json(path, default)
        return default


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── Config ────────────────────────────────────────────────────────────────────

def load_config() -> dict[str, Any]:
    cfg = _load_json(CONFIG_FILE, DEFAULT_CONFIG)
    # 누락된 키 보완
    for section, defaults in DEFAULT_CONFIG.items():
        cfg.setdefault(section, {})
        for k, v in defaults.items():
            cfg[section].setdefault(k, v)
    return cfg


def save_config(config: dict[str, Any]) -> None:
    _save_json(CONFIG_FILE, config)


def update_config_section(section: str, **kwargs) -> dict[str, Any]:
    cfg = load_config()
    cfg.setdefault(section, {}).update(kwargs)
    save_config(cfg)
    return cfg


# ── Style Presets ─────────────────────────────────────────────────────────────

def load_presets() -> list[dict[str, Any]]:
    data = _load_json(PRESETS_FILE, [DEFAULT_PRESET])
    return data if isinstance(data, list) else [DEFAULT_PRESET]


def save_presets(presets: list[dict[str, Any]]) -> None:
    _save_json(PRESETS_FILE, presets)


def get_preset_by_name(name: str) -> dict[str, Any] | None:
    for p in load_presets():
        if p.get("name") == name:
            return p
    return None


def upsert_preset(preset: dict[str, Any]) -> list[dict[str, Any]]:
    presets = load_presets()
    for i, p in enumerate(presets):
        if p.get("name") == preset["name"]:
            presets[i] = preset
            save_presets(presets)
            return presets
    presets.append(preset)
    save_presets(presets)
    return presets


def delete_preset(name: str) -> list[dict[str, Any]]:
    presets = [p for p in load_presets() if p.get("name") != name]
    save_presets(presets)
    return presets


# ── Character Presets ─────────────────────────────────────────────────────────

def load_characters() -> list[dict[str, Any]]:
    data = _load_json(CHARACTERS_FILE, [])
    return data if isinstance(data, list) else []


def save_characters(characters: list[dict[str, Any]]) -> None:
    _save_json(CHARACTERS_FILE, characters)


def get_character_by_name(name: str) -> dict[str, Any] | None:
    for c in load_characters():
        if c.get("name") == name:
            return c
    return None


def upsert_character(character: dict[str, Any]) -> list[dict[str, Any]]:
    characters = load_characters()
    for i, c in enumerate(characters):
        if c.get("name") == character["name"]:
            characters[i] = character
            save_characters(characters)
            return characters
    characters.append(character)
    save_characters(characters)
    return characters


def delete_character(name: str) -> list[dict[str, Any]]:
    characters = [c for c in load_characters() if c.get("name") != name]
    save_characters(characters)
    return characters


# ── Output Groups ─────────────────────────────────────────────────────────────

def list_output_groups() -> list[str]:
    _ensure_dirs()
    return sorted([d.name for d in OUTPUTS_DIR.iterdir() if d.is_dir()])


def get_group_dir(group_name: str) -> Path:
    group_dir = OUTPUTS_DIR / group_name
    group_dir.mkdir(parents=True, exist_ok=True)
    return group_dir


def list_images_in_group(group_name: str) -> list[str]:
    group_dir = OUTPUTS_DIR / group_name
    if not group_dir.exists():
        return []
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(
        str(p) for p in group_dir.iterdir()
        if p.suffix.lower() in exts
    )


_ensure_dirs()
