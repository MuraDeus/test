"""
llm_parser.py
LLM API를 통해 사용자 상황 설명 → NAI V5 장면 프롬프트 JSON 변환
반환: {"scene_prompt": "...", "negative_prompt": "..."}
"""
import json
import re
import urllib.request
import urllib.error
from typing import Any

from config_manager import load_config

# 고정 기술 규칙 파트 — JSON 포맷, 예시 등 파싱에 필수적인 내용
_TECHNICAL_RULES = """
Rules:
1. Extract ONLY scene/background, composition, camera angle, lighting, and character actions/poses.
2. Do NOT include character appearance (hair, eye color, clothing) — those come from separate character presets.
3. Output concise, comma-separated English tags ordered by importance.
4. For negative_prompt, include technical/quality issues specific to the scene (motion blur, lens flare, etc). Leave empty string if nothing specific.
5. Return ONLY valid JSON, no markdown fences, no explanation.

Output format (strict JSON):
{"scene_prompt": "tags here", "negative_prompt": "tags here or empty string"}

Examples:
Input: "교실에서 창문 밖을 바라보는 장면, 오후 햇살"
Output: {"scene_prompt": "classroom, window, looking outside, afternoon sunlight, golden hour, side view, detailed background, school desk, bokeh", "negative_prompt": ""}

Input: "비 오는 골목길에서 달리는 장면, 위에서 내려다보는 구도"
Output: {"scene_prompt": "rainy alley, running, bird's eye view, rain drops, puddles, night, wet pavement, motion, dramatic lighting", "negative_prompt": "motion blur"}
"""


def _build_system_prompt(cfg: dict[str, Any]) -> str:
    """
    config.json의 llm.base_prompt(사용자 설정) + 고정 기술 규칙을 결합하여
    최종 System Prompt를 생성한다.
    base_prompt가 비어 있으면 기술 규칙만 사용.
    """
    user_base = cfg["llm"].get("base_prompt", "").strip()
    if user_base:
        return user_base + "\n\n" + _TECHNICAL_RULES.strip()
    return _TECHNICAL_RULES.strip()


def _call_llm(situation: str, cfg: dict[str, Any]) -> str:
    llm_cfg = cfg["llm"]
    endpoint = llm_cfg["endpoint"].rstrip("/")
    api_key = llm_cfg["api_key"]
    model = llm_cfg["model"]
    temperature = float(llm_cfg.get("temperature", 0.7))

    system_prompt = _build_system_prompt(cfg)

    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": situation}
        ]
    }

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
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM API HTTP {e.code}: {error_body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"LLM API 연결 오류: {e.reason}") from e

    # OpenAI-compatible response
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"LLM 응답 파싱 실패: {body}") from e


def _extract_json(raw: str) -> dict[str, str]:
    # 마크다운 코드블록 제거
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    # JSON 객체 추출
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"LLM이 유효한 JSON을 반환하지 않았습니다: {raw}")
    parsed = json.loads(match.group())
    return {
        "scene_prompt": str(parsed.get("scene_prompt", "")).strip(),
        "negative_prompt": str(parsed.get("negative_prompt", "")).strip()
    }


def parse_situation(situation: str) -> dict[str, str]:
    """
    사용자 상황 설명 → {"scene_prompt": str, "negative_prompt": str}
    LLM 미설정 시 빈 프롬프트 반환(에러 없이 진행 가능하도록)
    """
    cfg = load_config()
    api_key = cfg["llm"].get("api_key", "").strip()

    if not api_key:
        # LLM 미설정: 사용자 입력을 그대로 scene_prompt로 사용
        return {"scene_prompt": situation.strip(), "negative_prompt": ""}

    raw = _call_llm(situation, cfg)
    return _extract_json(raw)
