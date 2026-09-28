"""
llm_parser.py  (v3)
LLM API 통합 모듈:
  · parse_situation()          — 사용자 입력 → NAI V5 프롬프트 JSON
  · auto_generate_situation()  — 빈 상황 입력 → 그룹 문맥 기반 자동 장면 생성
  · auto_generate_dialogue()   — 빈 대사 입력 → 문맥 기반 한국어 대사 자동 창작
  · update_context()           — 컷 완성 후 context.json 누적 업데이트

추론 수준(reasoning_effort):
  "none"  → config temperature 그대로 사용
  "low"   → temperature 1.0  (표준 모델) / reasoning_effort "low"  (o-series)
  "medium"→ temperature 0.6  / reasoning_effort "medium"
  "high"  → temperature 0.2  / reasoning_effort "high"
"""
import json
import re
import time
import urllib.request
import urllib.error
from typing import Any

from config_manager import load_config, load_group_context, append_cut_to_context

# ── 상수 ──────────────────────────────────────────────────────────────────────

REASONING_EFFORT_OPTIONS: list[str] = ["none", "low", "medium", "high"]

_EFFORT_TO_TEMP: dict[str, float] = {
    "low": 1.0,
    "medium": 0.6,
    "high": 0.2,
}

# 고정 기술 규칙 — JSON 포맷·예시 포함. 항상 system prompt 끝에 결합.
_TECHNICAL_RULES = """
Rules:
1. Extract ONLY scene/background, composition, camera angle, lighting, and character actions/poses.
2. Do NOT include character appearance (hair, eye color, clothing) — those come from separate character presets.
3. Output concise, comma-separated English tags ordered by importance.
4. For negative_prompt, include technical/quality issues specific to the scene. Leave empty string if none.
5. Return ONLY valid JSON, no markdown fences, no explanation.

Output format (strict JSON):
{"scene_prompt": "tags here", "negative_prompt": "tags or empty string"}

Examples:
Input: "교실에서 창문 밖을 바라보는 장면, 오후 햇살"
Output: {"scene_prompt": "classroom, window, looking outside, afternoon sunlight, golden hour, side view, detailed background, school desk, bokeh", "negative_prompt": ""}

Input: "비 오는 골목길에서 달리는 장면, 위에서 내려다보는 구도"
Output: {"scene_prompt": "rainy alley, running, bird's eye view, rain drops, puddles, night, wet pavement, motion, dramatic lighting", "negative_prompt": "motion blur"}
"""


# ── 내부 유틸 ──────────────────────────────────────────────────────────────────

def _is_reasoning_model(model: str) -> bool:
    """OpenAI o1/o3 계열 — reasoning_effort 파라미터를 지원하는 모델."""
    return bool(re.match(r"^o\d", model.lower().strip()))


def _normalize_endpoint(url: str) -> str:
    """
    /v숫자 로 끝나는 베이스 URL에 /chat/completions 자동 보완.
    예: https://api.openai.com/v1 → .../v1/chat/completions
    """
    url = url.rstrip("/")
    if re.search(r"/v\d+$", url):
        url += "/chat/completions"
    return url


def _build_system_prompt(cfg: dict[str, Any]) -> str:
    """사용자 설정 베이스 프롬프트 + 고정 기술 규칙 결합."""
    user_base = cfg["llm"].get("base_prompt", "").strip()
    if user_base:
        return user_base + "\n\n" + _TECHNICAL_RULES.strip()
    return _TECHNICAL_RULES.strip()


def _extract_json(raw: str) -> dict:
    """LLM 응답 문자열에서 JSON 객체 추출 (마크다운 코드블록 자동 제거)."""
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"LLM이 유효한 JSON을 반환하지 않았습니다:\n{raw[:400]}")
    return json.loads(match.group())


def _format_recent_history(history: list, max_cuts: int = 5) -> str:
    """최근 컷 이력 → LLM 컨텍스트 텍스트."""
    if not history:
        return "(이전 컷 없음 — 첫 번째 컷)"
    lines = []
    for cut in history[-max_cuts:]:
        idx = cut.get("index", "?")
        summary = cut.get("cut_summary") or cut.get("situation", "")[:80]
        line = f"[컷 {idx}] {summary}"
        if cut.get("dialogue"):
            line += f'  / 대사: "{cut["dialogue"]}"'
        lines.append(line)
    return "\n".join(lines)


def _format_recent_dialogues(history: list, max_cuts: int = 3) -> str:
    """최근 대사 목록 → LLM 컨텍스트 텍스트."""
    dialogues = [c.get("dialogue", "") for c in history[-max_cuts:] if c.get("dialogue")]
    return "\n".join(f'"{d}"' for d in dialogues) if dialogues else "(이전 대사 없음)"


# ── 핵심 LLM 호출 ─────────────────────────────────────────────────────────────

def _call_llm(
    user_message: str,
    cfg: dict[str, Any],
    system_prompt: str | None = None,
    reasoning_effort: str = "none",
) -> str:
    """
    OpenAI-compatible LLM API 단일 호출.
    system_prompt=None 이면 config 기반 기본 시스템 프롬프트 사용.
    """
    llm_cfg = cfg["llm"]
    endpoint = _normalize_endpoint(llm_cfg["endpoint"])
    api_key = llm_cfg["api_key"]
    model = llm_cfg["model"]

    effective_system = system_prompt if system_prompt is not None else _build_system_prompt(cfg)

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": effective_system},
            {"role": "user", "content": user_message},
        ],
    }

    effort = (reasoning_effort or "none").lower()
    if effort != "none" and effort in _EFFORT_TO_TEMP:
        if _is_reasoning_model(model):
            payload["reasoning_effort"] = effort
        else:
            payload["temperature"] = _EFFORT_TO_TEMP[effort]
    else:
        payload["temperature"] = float(llm_cfg.get("temperature", 0.7))

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        if e.code == 404:
            raise RuntimeError(
                f"엔드포인트를 찾을 수 없습니다 (HTTP 404).\n"
                f"현재 요청 URL: {endpoint}\n"
                f"설정 탭에서 엔드포인트를 확인해주세요.\n"
                f"  · OpenAI:     https://api.openai.com/v1/chat/completions\n"
                f"  · LM Studio:  http://localhost:1234/v1/chat/completions\n"
                f"  · Ollama:     http://localhost:11434/v1/chat/completions\n"
                f"서버 응답: {error_body}"
            ) from e
        if e.code == 401:
            raise RuntimeError(
                f"API 인증 실패 (HTTP 401). 설정 탭에서 API Key를 확인해주세요.\n"
                f"서버 응답: {error_body}"
            ) from e
        raise RuntimeError(f"LLM API HTTP {e.code}: {error_body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"LLM 서버에 연결할 수 없습니다: {e.reason}\n"
            f"요청 URL: {endpoint}\n"
            f"서버가 실행 중인지, URL이 올바른지 확인해주세요."
        ) from e

    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"LLM 응답 파싱 실패: {body}") from e


# ── 공개 API ──────────────────────────────────────────────────────────────────

def parse_situation(situation: str, reasoning_effort: str = "none") -> dict[str, str]:
    """
    사용자 상황 설명 → {"scene_prompt": str, "negative_prompt": str}
    LLM 미설정 시 입력값을 그대로 scene_prompt로 사용.
    """
    cfg = load_config()
    if not cfg["llm"].get("api_key", "").strip():
        return {"scene_prompt": situation.strip(), "negative_prompt": ""}

    raw = _call_llm(situation, cfg, reasoning_effort=reasoning_effort)
    parsed = _extract_json(raw)
    return {
        "scene_prompt": str(parsed.get("scene_prompt", "")).strip(),
        "negative_prompt": str(parsed.get("negative_prompt", "")).strip(),
    }


def auto_generate_situation(
    group_name: str,
    character_names: list[str],
    reasoning_effort: str = "none",
) -> dict[str, str]:
    """
    상황 입력이 비어있을 때 그룹 문맥 기반으로 다음 장면을 자동 생성.

    Returns: {"situation": str, "scene_prompt": str, "negative_prompt": str}
    LLM 미설정 시 RuntimeError 발생.
    """
    cfg = load_config()
    if not cfg["llm"].get("api_key", "").strip():
        raise RuntimeError(
            "상황 자동 생성은 LLM API Key가 필요합니다.\n"
            "설정 탭에서 API Key를 입력하거나, 상황 설명을 직접 입력해주세요."
        )

    ctx = load_group_context(group_name)
    summary = ctx.get("summary", "")
    history = ctx.get("history", [])
    chars_state = ctx.get("characters_state", {})

    system_prompt = (
        "You are a manga story AI. Given a story context, generate the next panel's scene.\n"
        "Return ONLY valid JSON with no markdown fences:\n"
        '{"situation": "Korean 2-3 sentence scene description", '
        '"scene_prompt": "English Danbooru-style NovelAI tags for background/composition/lighting/action", '
        '"negative_prompt": ""}'
    )

    char_list = ", ".join(character_names) if character_names else "미지정"
    char_states_txt = (
        "\n".join(f"  · {k}: {v}" for k, v in chars_state.items())
        if chars_state else "  (초기 상태 또는 미지정)"
    )

    user_msg = (
        f"[줄거리 요약]\n{summary or '(첫 번째 컷 — 자유롭게 시작해주세요)'}\n\n"
        f"[최근 컷 흐름]\n{_format_recent_history(history)}\n\n"
        f"[캐릭터 현재 상태]\n{char_states_txt}\n\n"
        f"[이번 컷 등장 인물]\n{char_list}\n\n"
        "위 맥락에서 자연스럽게 이어지는 다음 컷의 장면을 생성하세요."
    )

    raw = _call_llm(user_msg, cfg, system_prompt=system_prompt, reasoning_effort=reasoning_effort)
    parsed = _extract_json(raw)
    return {
        "situation": str(parsed.get("situation", "")).strip(),
        "scene_prompt": str(parsed.get("scene_prompt", "")).strip(),
        "negative_prompt": str(parsed.get("negative_prompt", "")).strip(),
    }


def auto_generate_dialogue(
    scene_prompt: str,
    situation: str,
    group_name: str,
    character_names: list[str],
    reasoning_effort: str = "none",
) -> str:
    """
    대사 포함 체크됐지만 입력이 비어있을 때 문맥 기반 한국어 대사 자동 창작.
    LLM 실패 시 빈 문자열 반환 (이미지 생성은 계속).
    """
    cfg = load_config()
    if not cfg["llm"].get("api_key", "").strip():
        return ""

    ctx = load_group_context(group_name)
    summary = ctx.get("summary", "")
    history = ctx.get("history", [])

    system_prompt = (
        "You are a manga dialogue writer. Write one natural Korean dialogue line for the given scene.\n"
        "The dialogue should feel authentic to manga/webtoon style — emotional and concise.\n"
        "Return ONLY valid JSON with no markdown fences:\n"
        '{"dialogue": "한국어 대사"}'
    )

    char_list = ", ".join(character_names) if character_names else "미지정"

    user_msg = (
        f"[줄거리 요약]\n{summary or '(없음)'}\n\n"
        f"[현재 상황]\n{situation}\n\n"
        f"[현재 장면 태그]\n{scene_prompt}\n\n"
        f"[등장 인물]\n{char_list}\n\n"
        f"[최근 대사들]\n{_format_recent_dialogues(history)}\n\n"
        "위 상황에서 자연스럽게 이어지는 한국어 대사 한 줄을 생성하세요."
    )

    try:
        raw = _call_llm(user_msg, cfg, system_prompt=system_prompt, reasoning_effort=reasoning_effort)
        parsed = _extract_json(raw)
        return str(parsed.get("dialogue", "")).strip()
    except Exception:
        return ""


def update_context(
    group_name: str,
    situation: str,
    scene_prompt: str,
    dialogue: str,
    character_names: list[str],
    image_filename: str = "",
    reasoning_effort: str = "none",
) -> None:
    """
    컷 생성 완료 후 그룹 context.json 자동 업데이트.
    LLM이 장면을 요약하고 줄거리·캐릭터 상태를 누적 저장.
    LLM 미설정 또는 오류 시 기본 정보만 저장 (비치명적).
    """
    cfg = load_config()
    api_key = cfg["llm"].get("api_key", "").strip()

    ctx = load_group_context(group_name)
    history = ctx.get("history", [])
    summary = ctx.get("summary", "")
    cut_index = len(history) + 1

    cut_summary = ""
    updated_summary = summary
    char_updates: dict[str, str] = {}

    if api_key:
        system_prompt = (
            "You are a manga story tracker. Summarize the new cut and update the story context.\n"
            "Keep the summary concise (2-3 Korean sentences max).\n"
            "Return ONLY valid JSON with no markdown fences:\n"
            '{"updated_summary": "업데이트된 줄거리 요약", '
            '"cut_summary": "이 컷의 한 줄 요약", '
            '"character_updates": {"캐릭터명": "현재 상태"}}'
        )
        char_list = ", ".join(character_names) if character_names else "미지정"
        user_msg = (
            f"[이전 줄거리 요약]\n{summary or '(없음)'}\n\n"
            f"[새로운 컷 #{cut_index}]\n"
            f"상황 설명: {situation}\n"
            f"장면 태그: {scene_prompt}\n"
            f"대사: {dialogue or '(없음)'}\n"
            f"등장 인물: {char_list}"
        )
        try:
            raw = _call_llm(
                user_msg, cfg,
                system_prompt=system_prompt,
                reasoning_effort=reasoning_effort,
            )
            parsed = _extract_json(raw)
            cut_summary = str(parsed.get("cut_summary", "")).strip()
            updated_summary = str(parsed.get("updated_summary", summary)).strip()
            raw_updates = parsed.get("character_updates", {})
            char_updates = raw_updates if isinstance(raw_updates, dict) else {}
        except Exception:
            cut_summary = (situation or scene_prompt)[:80]

    cut_entry: dict[str, Any] = {
        "index": cut_index,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "situation": situation,
        "scene_prompt": scene_prompt,
        "dialogue": dialogue,
        "characters": character_names,
        "cut_summary": cut_summary or (situation or scene_prompt)[:80],
        "image": image_filename,
    }
    append_cut_to_context(group_name, cut_entry, updated_summary, char_updates)
