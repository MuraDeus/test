"""
nai_client.py
NAI V5 API 호출 및 이미지 저장 모듈

NAI V5(nai-diffusion-5-full) API 사양:
  POST https://image.novelai.net/ai/generate-image
  Authorization: Bearer {token}
  Content-Type: application/json

V5 멀티 캐릭터: parameters.v4_prompt / v4_negative_prompt 구조 사용
"""
import io
import json
import time
import urllib.request
import urllib.error
import zipfile
from pathlib import Path
from typing import Any

from config_manager import get_group_dir, load_config, get_preset_by_name, get_character_by_name

NAI_ENDPOINT = "https://image.novelai.net/ai/generate-image"

# NAI V5가 지원하는 Sampler 목록
VALID_SAMPLERS = {
    "k_euler", "k_euler_ancestral", "k_dpmpp_2s_ancestral",
    "k_dpmpp_2m", "k_dpmpp_sde", "k_dpmpp_2m_sde",
    "ddim_v3"
}

# 모델별 지원 해상도 (width, height)
SUPPORTED_RESOLUTIONS = [
    (832, 1216), (1216, 832), (1024, 1024),
    (1024, 1536), (1536, 1024), (640, 1536), (1536, 640)
]


def _clamp_resolution(width: int, height: int) -> tuple[int, int]:
    """가장 가까운 지원 해상도로 보정"""
    best = min(SUPPORTED_RESOLUTIONS, key=lambda r: abs(r[0] - width) + abs(r[1] - height))
    return best


def _build_char_centers(count: int) -> list[dict[str, float]]:
    """캐릭터 수에 따라 화면 배치 중심좌표 자동 계산 (0~1 범위)"""
    if count <= 0:
        return []
    if count == 1:
        return [{"x": 0.5, "y": 0.5}]
    step = 1.0 / (count + 1)
    return [{"x": round(step * (i + 1), 3), "y": 0.5} for i in range(count)]


def _build_payload(
    scene_prompt: str,
    scene_negative: str,
    style_preset: dict[str, Any],
    characters: list[dict[str, Any]],
    dialogue_text: str | None,
    model: str
) -> dict[str, Any]:
    """NAI V5 API 요청 페이로드 조립"""

    # 1. Base Prompt = 장면 태그 + 그림체 긍정 프롬프트
    style_positive = style_preset.get("positive_prompt", "")
    base_caption_parts = [p for p in [scene_prompt, style_positive] if p.strip()]
    base_caption = ", ".join(base_caption_parts)

    # 2. 대사 처리: 체크박스 활성 + 텍스트 있을 때만 추가
    if dialogue_text and dialogue_text.strip():
        base_caption += f', speech bubble, text, korean text, Text: "{dialogue_text.strip()}"'

    # 3. Base Negative = 장면 부정 + 그림체 부정
    style_negative = style_preset.get("negative_prompt", "")
    neg_parts = [p for p in [scene_negative, style_negative] if p.strip()]
    base_negative = ", ".join(neg_parts)

    # 4. 캐릭터 배치 좌표
    centers = _build_char_centers(len(characters))

    # 5. v4_prompt / v4_negative_prompt (NAI V4/V5 멀티 캐릭터 구조)
    char_captions = []
    char_neg_captions = []
    for i, char in enumerate(characters):
        char_captions.append({
            "char_caption": char.get("prompt", ""),
            "centers": [centers[i]] if i < len(centers) else [{"x": 0.5, "y": 0.5}]
        })
        char_neg_captions.append({
            "char_caption": char.get("negative_prompt", ""),
            "centers": [centers[i]] if i < len(centers) else [{"x": 0.5, "y": 0.5}]
        })

    v4_prompt = {
        "caption": {
            "base_caption": base_caption,
            "char_captions": char_captions
        },
        "use_coords": False,
        "use_order": True
    }
    v4_negative_prompt = {
        "caption": {
            "base_caption": base_negative,
            "char_captions": char_neg_captions
        },
        "use_coords": False,
        "use_order": True
    }

    # 6. 해상도 보정
    raw_w = int(style_preset.get("width", 832))
    raw_h = int(style_preset.get("height", 1216))
    width, height = _clamp_resolution(raw_w, raw_h)

    # 7. Sampler 검증
    sampler = style_preset.get("sampler", "k_euler_ancestral")
    if sampler not in VALID_SAMPLERS:
        sampler = "k_euler_ancestral"

    parameters: dict[str, Any] = {
        "params_version": 3,
        "width": width,
        "height": height,
        "scale": float(style_preset.get("scale", 6.0)),
        "sampler": sampler,
        "steps": int(style_preset.get("steps", 28)),
        "n_samples": 1,
        "ucPreset": 0,
        "qualityToggle": False,
        "autoSmea": bool(style_preset.get("smea", False)),
        "dynamic_thresholding": False,
        "controlnet_strength": 1.0,
        "legacy": False,
        "add_original_image": False,
        "cfg_rescale": float(style_preset.get("cfg_rescale", 0.0)),
        "noise_schedule": style_preset.get("noise_schedule", "karras"),
        "legacy_v3_extend": False,
        "seed": 0,  # 0 = 랜덤
        "negative_prompt": base_negative,
        "v4_prompt": v4_prompt,
        "v4_negative_prompt": v4_negative_prompt
    }

    return {
        "input": base_caption,
        "model": model,
        "action": "generate",
        "parameters": parameters
    }


def _call_nai_api(payload: dict[str, Any], api_key: str) -> bytes:
    """NAI API 호출 → ZIP 바이트 반환"""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        NAI_ENDPOINT,
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
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"NAI API HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"NAI API 연결 오류: {e.reason}") from e


def _extract_image_from_zip(zip_bytes: bytes) -> bytes:
    """NAI 응답 ZIP에서 첫 번째 PNG 추출"""
    buf = io.BytesIO(zip_bytes)
    with zipfile.ZipFile(buf) as zf:
        for name in zf.namelist():
            if name.lower().endswith((".png", ".jpg", ".webp")):
                return zf.read(name)
        # 파일명 무관하게 첫 항목 반환
        if zf.namelist():
            return zf.read(zf.namelist()[0])
    raise RuntimeError("NAI 응답 ZIP에서 이미지를 찾을 수 없습니다.")


def generate_image(
    situation: str,
    scene_prompt: str,
    scene_negative: str,
    style_preset_name: str,
    character_names: list[str],
    dialogue_enabled: bool,
    dialogue_text: str,
    group_name: str
) -> tuple[str, str, str]:
    """
    NAI V5 이미지 생성 메인 함수

    Returns:
        (saved_image_path, final_prompt_display, error_message)
        에러 없으면 error_message=""
    """
    cfg = load_config()
    api_key = cfg["nai"].get("api_key", "").strip()
    model = cfg["nai"].get("model", "nai-diffusion-5-full").strip()

    if not api_key:
        return "", "", "NAI API 키가 설정되지 않았습니다. [탭 5: 설정]에서 입력해주세요."
    if not group_name.strip():
        return "", "", "생성 그룹(폴더)명을 입력해주세요."

    # 그림체 프리셋
    style_preset = get_preset_by_name(style_preset_name)
    if style_preset is None:
        from config_manager import load_presets
        presets = load_presets()
        style_preset = presets[0] if presets else {}

    # 캐릭터 프리셋
    chars: list[dict[str, Any]] = []
    for cname in character_names:
        c = get_character_by_name(cname)
        if c:
            chars.append({
                "prompt": c.get("prompt", ""),
                "negative_prompt": c.get("negative_prompt", "")
            })

    # 대사 텍스트 조건 처리
    effective_dialogue = dialogue_text if dialogue_enabled else ""

    # 페이로드 조립
    payload = _build_payload(
        scene_prompt=scene_prompt,
        scene_negative=scene_negative,
        style_preset=style_preset,
        characters=chars,
        dialogue_text=effective_dialogue,
        model=model
    )

    # 디버그용 프롬프트 표시 문자열
    prompt_display = (
        f"[Base Prompt]\n{payload['input']}\n\n"
        f"[Negative]\n{payload['parameters']['negative_prompt']}\n\n"
        f"[Characters]\n"
    )
    for i, cc in enumerate(payload["parameters"]["v4_prompt"]["caption"]["char_captions"]):
        prompt_display += f"  [{i+1}] {cc['char_caption']}\n"
    if not chars:
        prompt_display += "  (없음)\n"

    # API 호출
    try:
        zip_bytes = _call_nai_api(payload, api_key)
        image_bytes = _extract_image_from_zip(zip_bytes)
    except RuntimeError as e:
        return "", prompt_display, str(e)

    # 저장
    safe_group = "".join(c for c in group_name.strip() if c.isalnum() or c in "-_ ")
    if not safe_group:
        safe_group = "default"
    group_dir = get_group_dir(safe_group)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = group_dir / f"{timestamp}.png"
    save_path.write_bytes(image_bytes)

    return str(save_path), prompt_display, ""
