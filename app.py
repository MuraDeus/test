"""
app.py  (v5)
Gradio 기반 모바일 최적화 NAI V5 만화 생성기
Galaxy S26 Ultra Termux 환경 최적화

변경 사항:
  - Tab 1: reasoning_effort 드롭다운 / 그룹 문맥 Accordion / AI 자동 생성 파이프라인
  - Tab 2: Radio 목록 (클릭→즉시 편집) + 3버튼 CRUD
  - Tab 3: Radio 목록 (클릭→즉시 편집) + 3버튼 CRUD
  - Tab 4: 이미지/그룹 관리 (기존 유지)
  - Tab 5: reasoning_effort 기본값 설정 추가
"""
import gradio as gr
from pathlib import Path

from config_manager import (
    load_config, save_config,
    load_presets, upsert_preset, delete_preset, get_preset_by_name,
    load_characters, upsert_character, delete_character, get_character_by_name,
    list_output_groups, list_images_in_group,
    delete_image_file, delete_group_dir, rename_group_dir,
    load_group_context,
    BASE_DIR,
)
from llm_parser import (
    parse_situation,
    auto_generate_situation,
    auto_generate_dialogue,
    update_context,
    REASONING_EFFORT_OPTIONS,
)
from nai_client import generate_image, VALID_SAMPLERS


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

def _preset_names() -> list[str]:
    return [p["name"] for p in load_presets()]


def _char_names() -> list[str]:
    return [c["name"] for c in load_characters()]


def _group_choices() -> list[str]:
    return list_output_groups()


def _safe_name(raw: str) -> str:
    """파일시스템 안전 그룹명 (유니코드 알파뉴메릭 + -_공백 허용)."""
    return "".join(c for c in raw.strip() if c.isalnum() or c in "-_ ")


def _format_context_display(ctx: dict) -> str:
    """context.json → 사람이 읽기 좋은 요약 문자열."""
    summary = ctx.get("summary", "")
    history = ctx.get("history", [])
    chars = ctx.get("characters_state", {})
    lines = []
    if summary:
        lines.append(f"📖 줄거리: {summary}")
    if history:
        lines.append(f"🎬 누적 컷: {len(history)}컷")
        last = history[-1]
        last_text = last.get("cut_summary") or last.get("situation", "")[:60]
        lines.append(f"   └ 마지막: {last_text}")
    if chars:
        lines.append("👤 캐릭터 상태:")
        for name, state in list(chars.items())[:4]:
            lines.append(f"   · {name}: {state}")
    if not lines:
        lines.append("(문맥 없음 — 컷 생성 후 AI가 자동으로 누적합니다)")
    return "\n".join(lines)


def _default_reasoning() -> str:
    return load_config()["llm"].get("reasoning_effort", "none")


# ── 탭 1: 만화 생성 ───────────────────────────────────────────────────────────

def on_generate(
    situation, dialogue_enabled, dialogue_text,
    style_preset_name, character_names, group_name,
    reasoning_effort,
    progress=gr.Progress(),
):
    """
    통합 생성 파이프라인.
    · situation 비어있음 → LLM이 그룹 문맥에서 자동으로 다음 장면 생성
    · dialogue_enabled=True + dialogue 비어있음 → LLM이 대사 자동 창작
    · 생성 성공 후 context.json 자동 업데이트
    """
    safe_group = _safe_name(group_name or "")
    if not safe_group:
        return None, "", "생성 그룹(폴더)명을 입력해주세요.", gr.update(), gr.update(), gr.update()

    final_situation = (situation or "").strip()
    scene_prompt = ""
    scene_negative = ""

    # ── 1. 상황 설명 처리 ────────────────────────────────────────────────────
    if not final_situation:
        progress(0.05, desc="AI가 다음 장면을 자동 생성 중...")
        try:
            auto_result = auto_generate_situation(
                group_name=safe_group,
                character_names=character_names or [],
                reasoning_effort=reasoning_effort,
            )
            final_situation = auto_result["situation"]
            scene_prompt = auto_result["scene_prompt"]
            scene_negative = auto_result.get("negative_prompt", "")
        except Exception as e:
            return None, "", f"장면 자동 생성 오류: {e}", gr.update(), gr.update(), gr.update()
    else:
        progress(0.1, desc="LLM으로 프롬프트 변환 중...")
        try:
            parsed = parse_situation(final_situation, reasoning_effort=reasoning_effort)
            scene_prompt = parsed["scene_prompt"]
            scene_negative = parsed["negative_prompt"]
        except Exception as e:
            return None, "", f"LLM 오류: {e}", gr.update(), gr.update(), gr.update()

    # ── 2. 대사 처리 ────────────────────────────────────────────────────────
    effective_dialogue = ""
    generated_dialogue = ""
    if dialogue_enabled:
        raw_dlg = (dialogue_text or "").strip()
        if raw_dlg:
            effective_dialogue = raw_dlg
        else:
            progress(0.35, desc="AI가 대사를 자동 창작 중...")
            generated_dialogue = auto_generate_dialogue(
                scene_prompt=scene_prompt,
                situation=final_situation,
                group_name=safe_group,
                character_names=character_names or [],
                reasoning_effort=reasoning_effort,
            )
            effective_dialogue = generated_dialogue

    # ── 3. NAI V5 이미지 생성 ────────────────────────────────────────────────
    progress(0.5, desc="NAI V5 이미지 생성 중...")
    img_path, prompt_display, error = generate_image(
        situation=final_situation,
        scene_prompt=scene_prompt,
        scene_negative=scene_negative,
        style_preset_name=style_preset_name,
        character_names=character_names if character_names else [],
        dialogue_enabled=dialogue_enabled,
        dialogue_text=effective_dialogue,
        group_name=safe_group,
    )

    if error:
        return None, prompt_display, f"오류: {error}", gr.update(), gr.update(), gr.update()

    # ── 4. 문맥 업데이트 ────────────────────────────────────────────────────
    progress(0.9, desc="문맥 저장 중...")
    image_filename = Path(img_path).name if img_path else ""
    try:
        update_context(
            group_name=safe_group,
            situation=final_situation,
            scene_prompt=scene_prompt,
            dialogue=effective_dialogue,
            character_names=character_names or [],
            image_filename=image_filename,
            reasoning_effort=reasoning_effort,
        )
    except Exception:
        pass  # 문맥 업데이트 실패는 비치명적

    # ── 5. UI 업데이트 값 결정 ───────────────────────────────────────────────
    ctx_text = _format_context_display(load_group_context(safe_group))
    progress(1.0, desc="완료")

    # 자동 생성된 항목만 UI 반영 (사용자 입력이 있었으면 gr.update()로 그대로)
    situation_out = final_situation if not (situation or "").strip() else gr.update()
    dialogue_out = generated_dialogue if generated_dialogue else gr.update()

    return img_path, prompt_display, "생성 완료!", situation_out, dialogue_out, ctx_text


def build_tab_generate():
    """Returns (style_preset_dd, char_cbg) — 인물/그림체 탭 크로스 갱신용."""
    cfg_now = load_config()

    with gr.Column():
        gr.Markdown("## 만화 컷 생성")

        situation_input = gr.Textbox(
            label="대본 / 상황 설명  (비우면 AI가 이전 문맥에서 자동 생성)",
            placeholder="예: 비 오는 밤, 교실 창가에 홀로 앉아 창밖을 바라보는 장면.\n비워두면 AI가 스토리를 이어서 자동으로 만들어줍니다.",
            lines=4,
        )

        dialogue_enabled = gr.Checkbox(label="대사 포함", value=False)
        dialogue_text = gr.Textbox(
            label="대사  (비우면 AI가 자동 창작 / 대사 포함 체크 시에만 적용)",
            placeholder="예: 오늘도 비가 오네...\n비워두면 AI가 장면에 맞는 대사를 자동으로 만들어줍니다.",
            lines=2,
            visible=False,
        )
        dialogue_enabled.change(
            fn=lambda v: gr.update(visible=v),
            inputs=dialogue_enabled,
            outputs=dialogue_text,
        )

        style_preset_dd = gr.Dropdown(
            label="그림체 프리셋",
            choices=_preset_names(),
            value=_preset_names()[0] if _preset_names() else None,
        )
        char_cbg = gr.CheckboxGroup(
            label="등장 인물 (다중 선택)",
            choices=_char_names(),
        )

        # 그룹 선택 — 텍스트 직접 입력 또는 기존 그룹 드롭다운
        gr.Markdown("**생성 그룹 (저장 폴더)**")
        with gr.Row():
            group_input = gr.Textbox(
                label="폴더명 직접 입력",
                placeholder="예: 작품A_1화",
                scale=3,
            )
            group_select_dd = gr.Dropdown(
                label="기존 그룹 선택",
                choices=_group_choices(),
                value=None,
                allow_custom_value=False,
                scale=2,
            )

        # 그룹 문맥 표시 Accordion
        with gr.Accordion("그룹 문맥 (AI 자동 생성 참조)", open=False):
            context_summary_display = gr.Textbox(
                label="현재 스토리 맥락",
                interactive=False,
                lines=5,
                value="그룹을 선택하거나 입력하면 문맥이 표시됩니다.",
            )
            ctx_refresh_btn = gr.Button("문맥 새로고침", size="sm")

        # 추론 수준 드롭다운
        reasoning_dd = gr.Dropdown(
            label="LLM 추론 수준",
            choices=REASONING_EFFORT_OPTIONS,
            value=cfg_now["llm"].get("reasoning_effort", "none"),
            info="none=설정 temperature 사용 / low=창의적(1.0) / medium=균형(0.6) / high=정밀(0.2)",
        )

        with gr.Row():
            refresh_btn = gr.Button("목록 새로고침", size="sm")
            generate_btn = gr.Button("생성", variant="primary", size="lg")

        status_text = gr.Textbox(label="상태", interactive=False, lines=1)
        result_image = gr.Image(label="생성된 이미지", type="filepath", height=500)
        prompt_display = gr.Textbox(
            label="최종 프롬프트 확인", interactive=False, lines=8
        )

    # ── 이벤트 연결 ────────────────────────────────────────────────────────────
    def _ctx_from_group(g):
        g = (g or "").strip()
        if not g:
            return "그룹을 입력하거나 선택해주세요."
        return _format_context_display(load_group_context(g))

    group_select_dd.change(
        fn=lambda v: (v or "", _ctx_from_group(v)),
        inputs=group_select_dd,
        outputs=[group_input, context_summary_display],
    )
    group_input.change(
        fn=_ctx_from_group,
        inputs=group_input,
        outputs=context_summary_display,
    )
    ctx_refresh_btn.click(
        fn=_ctx_from_group,
        inputs=group_input,
        outputs=context_summary_display,
    )
    refresh_btn.click(
        fn=lambda: (
            gr.update(choices=_preset_names()),
            gr.update(choices=_char_names()),
            gr.update(choices=_group_choices()),
        ),
        outputs=[style_preset_dd, char_cbg, group_select_dd],
    )
    generate_btn.click(
        fn=on_generate,
        inputs=[
            situation_input, dialogue_enabled, dialogue_text,
            style_preset_dd, char_cbg, group_input,
            reasoning_dd,
        ],
        outputs=[result_image, prompt_display, status_text,
                 situation_input, dialogue_text, context_summary_display],
    )

    return style_preset_dd, char_cbg


# ── 탭 2: 인물 프리셋 관리 ───────────────────────────────────────────────────

def on_char_select(name):
    """Radio 클릭 → 폼 자동 채우기"""
    if not name:
        return "", "", ""
    char = get_character_by_name(name)
    if char is None:
        return "", "", ""
    return char["name"], char.get("prompt", ""), char.get("negative_prompt", "")


def on_char_new(name, prompt, negative_prompt):
    name = name.strip()
    if not name:
        return "인물 이름을 입력해주세요.", gr.update(), gr.update()
    if get_character_by_name(name):
        return (
            f"'{name}'은 이미 존재합니다. 수정하려면 목록에서 선택 후 '수정 사항 반영'을 눌러주세요.",
            gr.update(), gr.update(),
        )
    upsert_character({"name": name, "prompt": prompt.strip(), "negative_prompt": negative_prompt.strip()})
    names = _char_names()
    return f"'{name}' 새로 등록 완료.", gr.update(choices=names, value=name), gr.update(choices=names)


def on_char_update(selected_name, name, prompt, negative_prompt):
    if not selected_name:
        return "수정할 인물을 목록에서 먼저 선택해주세요.", gr.update(), gr.update()
    name = name.strip()
    if not name:
        return "인물 이름을 입력해주세요.", gr.update(), gr.update()
    if selected_name != name and get_character_by_name(name):
        return f"'{name}'은 이미 다른 인물로 등록되어 있습니다.", gr.update(), gr.update()
    if selected_name != name:
        delete_character(selected_name)
    upsert_character({"name": name, "prompt": prompt.strip(), "negative_prompt": negative_prompt.strip()})
    names = _char_names()
    return f"'{name}' 수정 완료.", gr.update(choices=names, value=name), gr.update(choices=names)


def on_char_delete(selected_name):
    if not selected_name:
        return "삭제할 인물을 목록에서 선택해주세요.", gr.update(), gr.update()
    delete_character(selected_name)
    names = _char_names()
    return f"'{selected_name}' 삭제 완료.", gr.update(choices=names, value=None), gr.update(choices=names)


def build_tab_characters(char_cbg_ref: gr.CheckboxGroup):
    """char_cbg_ref: 탭1 CheckboxGroup — 인물 CRUD 시 함께 갱신."""
    with gr.Column():
        gr.Markdown("## 인물(Character) 프리셋 관리")
        gr.Markdown("*목록에서 클릭하면 아래 폼에 자동으로 불러옵니다.*")

        # ── Radio 목록 (클릭→편집) ────────────────────────────────────────────
        char_list = gr.Radio(
            label="저장된 인물 목록 (클릭하여 선택·편집)",
            choices=_char_names(),
            value=None,
        )

        gr.Markdown("---")
        char_name = gr.Textbox(label="인물 이름", placeholder="예: 주인공_민수")
        char_prompt = gr.Textbox(
            label="외형 프롬프트 (Positive)",
            placeholder="예: 1boy, black hair, short hair, brown eyes, school uniform",
            lines=4,
        )
        char_neg = gr.Textbox(
            label="부정 프롬프트 (Undesired Content)",
            placeholder="예: glasses",
            lines=2,
        )

        with gr.Row():
            char_new_btn = gr.Button("새로 등록", variant="primary")
            char_update_btn = gr.Button("수정 사항 반영", variant="secondary")
            char_del_btn = gr.Button("삭제", variant="stop")

        char_clear_btn = gr.Button("폼 초기화 (새 입력 시작)", size="sm")
        char_status = gr.Textbox(label="상태", interactive=False, lines=1)

    char_list.change(fn=on_char_select, inputs=char_list, outputs=[char_name, char_prompt, char_neg])
    char_new_btn.click(
        fn=on_char_new,
        inputs=[char_name, char_prompt, char_neg],
        outputs=[char_status, char_list, char_cbg_ref],
    )
    char_update_btn.click(
        fn=on_char_update,
        inputs=[char_list, char_name, char_prompt, char_neg],
        outputs=[char_status, char_list, char_cbg_ref],
    )
    char_del_btn.click(
        fn=on_char_delete,
        inputs=char_list,
        outputs=[char_status, char_list, char_cbg_ref],
    )
    char_clear_btn.click(
        fn=lambda: (gr.update(value=None), "", "", ""),
        outputs=[char_list, char_name, char_prompt, char_neg],
    )


# ── 탭 3: 그림체 프리셋 관리 ─────────────────────────────────────────────────

SAMPLER_LIST = sorted(VALID_SAMPLERS)
_STYLE_DEFAULTS = ("", "", "", "k_euler_ancestral", 28, 6.0, 832, 1216, "karras", 0.0, False)


def on_style_select(name):
    """Radio 클릭 → 모든 입력 필드 자동 채우기."""
    if not name:
        return _STYLE_DEFAULTS
    p = get_preset_by_name(name)
    if p is None:
        return _STYLE_DEFAULTS
    return (
        p["name"],
        p.get("positive_prompt", ""),
        p.get("negative_prompt", ""),
        p.get("sampler", "k_euler_ancestral"),
        p.get("steps", 28),
        p.get("scale", 6.0),
        p.get("width", 832),
        p.get("height", 1216),
        p.get("noise_schedule", "karras"),
        p.get("cfg_rescale", 0.0),
        p.get("smea", False),
    )


def _build_preset_dict(name, pos, neg, sampler, steps, scale, width, height, noise, cfg_rescale, smea):
    return {
        "name": name.strip(),
        "positive_prompt": pos.strip(),
        "negative_prompt": neg.strip(),
        "sampler": sampler,
        "steps": int(steps),
        "scale": float(scale),
        "width": int(width),
        "height": int(height),
        "noise_schedule": noise,
        "cfg_rescale": float(cfg_rescale),
        "smea": bool(smea),
    }


def on_style_new(name, pos, neg, sampler, steps, scale, width, height, noise, cfg_rescale, smea):
    name = name.strip()
    if not name:
        return "프리셋 이름을 입력해주세요.", gr.update(), gr.update()
    if get_preset_by_name(name):
        return (
            f"'{name}'은 이미 존재합니다. 수정하려면 목록에서 선택 후 '수정 사항 반영'을 눌러주세요.",
            gr.update(), gr.update(),
        )
    upsert_preset(_build_preset_dict(name, pos, neg, sampler, steps, scale, width, height, noise, cfg_rescale, smea))
    names = _preset_names()
    return f"'{name}' 새로 등록 완료.", gr.update(choices=names, value=name), gr.update(choices=names)


def on_style_update(selected_name, name, pos, neg, sampler, steps, scale, width, height, noise, cfg_rescale, smea):
    if not selected_name:
        return "수정할 프리셋을 목록에서 먼저 선택해주세요.", gr.update(), gr.update()
    name = name.strip()
    if not name:
        return "프리셋 이름을 입력해주세요.", gr.update(), gr.update()
    if selected_name != name and get_preset_by_name(name):
        return f"'{name}'은 이미 다른 프리셋으로 등록되어 있습니다.", gr.update(), gr.update()
    if selected_name != name:
        delete_preset(selected_name)
    upsert_preset(_build_preset_dict(name, pos, neg, sampler, steps, scale, width, height, noise, cfg_rescale, smea))
    names = _preset_names()
    return f"'{name}' 수정 완료.", gr.update(choices=names, value=name), gr.update(choices=names)


def on_style_delete(selected_name):
    if not selected_name:
        return "삭제할 프리셋을 목록에서 선택해주세요.", gr.update(), gr.update()
    delete_preset(selected_name)
    names = _preset_names()
    return f"'{selected_name}' 삭제 완료.", gr.update(choices=names, value=None), gr.update(choices=names)


def build_tab_styles(style_dd_ref: gr.Dropdown):
    """style_dd_ref: 탭1 Dropdown — 프리셋 CRUD 시 함께 갱신."""
    with gr.Column():
        gr.Markdown("## NAI 그림체 프리셋 관리")
        gr.Markdown("*목록에서 클릭하면 아래 폼에 자동으로 불러옵니다.*")

        # ── Radio 목록 (클릭→편집) ────────────────────────────────────────────
        style_list = gr.Radio(
            label="저장된 프리셋 목록 (클릭하여 선택·편집)",
            choices=_preset_names(),
            value=None,
        )

        gr.Markdown("---")
        s_name = gr.Textbox(label="프리셋 이름", placeholder="예: 기본 애니")
        s_pos = gr.Textbox(
            label="긍정 프롬프트", lines=4,
            placeholder="best quality, amazing quality, very aesthetic, absurdres",
        )
        s_neg = gr.Textbox(
            label="부정 프롬프트", lines=3,
            placeholder="lowres, bad quality, ...",
        )

        with gr.Row():
            s_sampler = gr.Dropdown(label="Sampler", choices=SAMPLER_LIST, value="k_euler_ancestral")
            s_steps = gr.Slider(label="Steps", minimum=1, maximum=50, step=1, value=28)

        with gr.Row():
            s_scale = gr.Slider(label="CFG Scale", minimum=1.0, maximum=10.0, step=0.5, value=6.0)
            s_cfg_rescale = gr.Slider(label="CFG Rescale", minimum=0.0, maximum=1.0, step=0.01, value=0.0)

        with gr.Row():
            s_width = gr.Dropdown(label="Width", choices=[640, 832, 1024, 1216, 1536], value=832)
            s_height = gr.Dropdown(label="Height", choices=[640, 832, 1024, 1216, 1536], value=1216)

        with gr.Row():
            s_noise = gr.Dropdown(
                label="Noise Schedule",
                choices=["karras", "exponential", "polyexponential", "native"],
                value="karras",
            )
            s_smea = gr.Checkbox(label="SMEA", value=False)

        with gr.Row():
            style_new_btn = gr.Button("새로 등록", variant="primary")
            style_update_btn = gr.Button("수정 사항 반영", variant="secondary")
            style_del_btn = gr.Button("삭제", variant="stop")

        style_clear_btn = gr.Button("폼 초기화 (새 입력 시작)", size="sm")
        style_status = gr.Textbox(label="상태", interactive=False, lines=1)

    _field_inputs = [s_name, s_pos, s_neg, s_sampler, s_steps, s_scale,
                     s_width, s_height, s_noise, s_cfg_rescale, s_smea]

    style_list.change(fn=on_style_select, inputs=style_list, outputs=_field_inputs)
    style_new_btn.click(
        fn=on_style_new, inputs=_field_inputs,
        outputs=[style_status, style_list, style_dd_ref],
    )
    style_update_btn.click(
        fn=on_style_update, inputs=[style_list] + _field_inputs,
        outputs=[style_status, style_list, style_dd_ref],
    )
    style_del_btn.click(
        fn=on_style_delete, inputs=style_list,
        outputs=[style_status, style_list, style_dd_ref],
    )
    style_clear_btn.click(
        fn=lambda: (gr.update(value=None),) + _STYLE_DEFAULTS,
        outputs=[style_list] + _field_inputs,
    )


# ── 탭 4: 갤러리 ─────────────────────────────────────────────────────────────

def on_group_change(group_name):
    images = list_images_in_group(group_name) if group_name else []
    return images, images, None, "(이미지를 클릭하여 선택)"


def on_gallery_select(evt: gr.SelectData, images: list):
    if images and 0 <= evt.index < len(images):
        path = images[evt.index]
        return path, f"선택됨: {Path(path).name}"
    return None, "선택 실패"


def on_delete_image(selected_path, group_name):
    if not selected_path:
        return "갤러리에서 이미지를 먼저 클릭하여 선택해주세요.", gr.update(), gr.update(), None, "(이미지를 클릭하여 선택)"
    success = delete_image_file(selected_path)
    if success:
        images = list_images_in_group(group_name) if group_name else []
        return "이미지 삭제 완료.", images, images, None, "(이미지를 클릭하여 선택)"
    return "이미지 파일을 찾을 수 없습니다.", gr.update(), gr.update(), selected_path, f"선택됨: {Path(selected_path).name}"


def on_rename_group(old_name, new_name):
    if not old_name:
        return "그룹을 선택해주세요.", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    safe = _safe_name(new_name or "")
    if not safe:
        return "유효한 그룹명을 입력해주세요.", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    ok, msg = rename_group_dir(old_name, safe)
    if ok:
        groups = _group_choices()
        images = list_images_in_group(safe)
        return (
            f"'{old_name}' → '{safe}' 이름 변경 완료.",
            gr.update(choices=groups, value=safe),
            images, images, None, "(이미지를 클릭하여 선택)",
        )
    return f"변경 실패: {msg}", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()


def on_delete_group(group_name):
    if not group_name:
        return "삭제할 그룹을 선택해주세요.", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    ok = delete_group_dir(group_name)
    if ok:
        groups = _group_choices()
        return (
            f"'{group_name}' 그룹 전체 삭제 완료.",
            gr.update(choices=groups, value=None),
            [], [], None, "(이미지를 클릭하여 선택)",
        )
    return "그룹 삭제 실패.", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()


def build_tab_gallery():
    with gr.Column():
        gr.Markdown("## 갤러리 / 그룹별 모아보기")

        with gr.Row():
            gallery_group_dd = gr.Dropdown(
                label="그룹 선택",
                choices=_group_choices(),
                value=None,
                allow_custom_value=False,
            )
            gallery_refresh_btn = gr.Button("목록 새로고침", size="sm")

        with gr.Accordion("그룹 관리 (이름 변경 / 전체 삭제)", open=False):
            gr.Markdown("*위에서 그룹을 먼저 선택한 후 작업하세요.*")
            with gr.Row():
                new_group_name_input = gr.Textbox(label="새 그룹명", placeholder="변경할 이름 입력", scale=3)
                rename_group_btn = gr.Button("그룹명 변경", size="sm", scale=1)
            delete_group_btn = gr.Button("그룹 전체 삭제 (이미지 포함)", variant="stop")
            group_status = gr.Textbox(label="그룹 관리 상태", interactive=False, lines=1)

        gallery_images_state = gr.State(value=[])
        selected_img_path = gr.State(value=None)

        gallery_output = gr.Gallery(
            label="이미지 목록 (이미지를 클릭하여 선택)",
            columns=2,
            object_fit="contain",
            height="auto",
        )

        selected_info = gr.Textbox(label="선택된 이미지", interactive=False, lines=1, value="(이미지를 클릭하여 선택)")
        with gr.Row():
            delete_img_btn = gr.Button("선택 이미지 삭제", variant="stop")
            img_status = gr.Textbox(label="삭제 상태", interactive=False, lines=1, scale=3)

    _gallery_st = [gallery_output, gallery_images_state, selected_img_path, selected_info]
    _group_mgmt = [group_status, gallery_group_dd] + _gallery_st

    gallery_group_dd.change(fn=on_group_change, inputs=gallery_group_dd, outputs=_gallery_st)
    gallery_refresh_btn.click(fn=lambda: gr.update(choices=_group_choices()), outputs=gallery_group_dd)
    gallery_output.select(fn=on_gallery_select, inputs=gallery_images_state, outputs=[selected_img_path, selected_info])
    delete_img_btn.click(
        fn=on_delete_image,
        inputs=[selected_img_path, gallery_group_dd],
        outputs=[img_status] + _gallery_st,
    )
    rename_group_btn.click(fn=on_rename_group, inputs=[gallery_group_dd, new_group_name_input], outputs=_group_mgmt)
    delete_group_btn.click(fn=on_delete_group, inputs=gallery_group_dd, outputs=_group_mgmt)


# ── 탭 5: 설정 ────────────────────────────────────────────────────────────────

def _load_settings_values():
    cfg = load_config()
    return (
        cfg["llm"]["endpoint"],                         # vals[0]
        cfg["llm"]["api_key"],                          # vals[1]
        cfg["llm"]["model"],                            # vals[2]
        cfg["llm"]["temperature"],                      # vals[3]
        cfg["llm"].get("base_prompt", ""),              # vals[4]
        cfg["llm"].get("reasoning_effort", "none"),     # vals[5]
        cfg["nai"]["api_key"],                          # vals[6]
        cfg["nai"]["model"],                            # vals[7]
    )


def on_settings_save(
    llm_endpoint, llm_key, llm_model, llm_temp,
    llm_base_prompt, reasoning_effort,
    nai_key, nai_model,
):
    cfg = load_config()
    cfg["llm"]["endpoint"] = llm_endpoint.strip()
    cfg["llm"]["api_key"] = llm_key.strip()
    cfg["llm"]["model"] = llm_model.strip()
    cfg["llm"]["temperature"] = float(llm_temp)
    cfg["llm"]["base_prompt"] = llm_base_prompt
    cfg["llm"]["reasoning_effort"] = reasoning_effort
    cfg["nai"]["api_key"] = nai_key.strip()
    cfg["nai"]["model"] = nai_model.strip()
    save_config(cfg)
    return "설정 저장 완료."


def build_tab_settings():
    vals = _load_settings_values()

    with gr.Column():
        gr.Markdown("## 모델 및 API 설정")

        gr.Markdown("### LLM 설정")
        llm_endpoint = gr.Textbox(
            label="LLM 엔드포인트 (OpenAI-compatible)",
            value=vals[0],
            placeholder="https://api.openai.com/v1/chat/completions",
        )
        llm_key = gr.Textbox(label="LLM API Key", value=vals[1], type="password", placeholder="sk-...")
        llm_model = gr.Textbox(label="LLM 모델명", value=vals[2], placeholder="gpt-4o-mini")
        llm_temp = gr.Slider(label="Temperature (reasoning_effort=none 일 때 적용)",
                             minimum=0.0, maximum=2.0, step=0.05, value=float(vals[3]))
        llm_base_prompt = gr.Textbox(
            label="LLM 베이스 프롬프트 (System Prompt 앞부분)",
            value=vals[4],
            lines=5,
            placeholder=(
                "You are an AI assistant that converts user story descriptions into NovelAI V5 prompts.\n"
                "Convert background, composition, and actions into concise Danbooru-style English tags.\n"
                "Always keep character features separated and return result in structured JSON."
            ),
            info="비워두면 기본 기술 규칙만 사용됩니다.",
        )
        reasoning_effort_dd = gr.Dropdown(
            label="기본 LLM 추론 수준 (생성 탭에서 개별 오버라이드 가능)",
            choices=REASONING_EFFORT_OPTIONS,
            value=vals[5],
            info=(
                "none=설정 temperature 사용 | "
                "low=창의적(temp 1.0) | medium=균형(0.6) | high=정밀(0.2)\n"
                "o1/o3 계열 모델: temperature 대신 reasoning_effort 파라미터로 전송"
            ),
        )

        gr.Markdown("### NAI 설정")
        nai_key = gr.Textbox(label="NAI API Key", value=vals[6], type="password", placeholder="pst-...")
        nai_model = gr.Dropdown(
            label="NAI 기본 모델",
            choices=[
                "nai-diffusion-5-full",
                "nai-diffusion-5-curated",
                "nai-diffusion-4-5-full",
                "nai-diffusion-4-5-curated",
                "nai-diffusion-4-full",
                "nai-diffusion-4-curated-preview",
            ],
            value=vals[7],
            allow_custom_value=True,
        )

        settings_save_btn = gr.Button("설정 저장", variant="primary")
        settings_status = gr.Textbox(label="상태", interactive=False, lines=1)

    settings_save_btn.click(
        fn=on_settings_save,
        inputs=[llm_endpoint, llm_key, llm_model, llm_temp,
                llm_base_prompt, reasoning_effort_dd,
                nai_key, nai_model],
        outputs=settings_status,
    )


# ── 앱 조립 ───────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* 모바일(Termux 브라우저) 최적화 */
body, .gradio-container { max-width: 100% !important; padding: 8px !important; font-size: 15px; }
button.gr-button, .gr-button { min-height: 44px !important; font-size: 15px !important; }
textarea, select, input[type="text"] { font-size: 15px !important; }
.tab-nav button { padding: 10px 6px !important; font-size: 13px !important; }
.gr-row > button { flex: 1 !important; }
/* Radio 목록 스타일 — 모바일 터치 영역 */
.gr-radio label { padding: 8px 4px !important; min-height: 36px; display: flex; align-items: center; }
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="NAI V5 만화 생성기",
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    ) as app:
        gr.Markdown("# NAI V5 만화 생성기")

        with gr.Tabs():
            with gr.Tab("생성"):
                style_dd_ref, char_cbg_ref = build_tab_generate()
            with gr.Tab("인물"):
                build_tab_characters(char_cbg_ref)
            with gr.Tab("그림체"):
                build_tab_styles(style_dd_ref)
            with gr.Tab("갤러리"):
                build_tab_gallery()
            with gr.Tab("설정"):
                build_tab_settings()

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False,
    )
