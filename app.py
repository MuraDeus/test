"""
app.py
Gradio 기반 모바일 최적화 NAI V5 만화 생성기
Galaxy S26 Ultra Termux 환경 최적화 (세로 레이아웃, 터치 친화적 UI)
"""
import gradio as gr
from pathlib import Path

from config_manager import (
    load_config, save_config,
    load_presets, upsert_preset, delete_preset,
    load_characters, upsert_character, delete_character,
    list_output_groups, list_images_in_group,
    DEFAULT_PRESET, BASE_DIR
)
from llm_parser import parse_situation
from nai_client import generate_image, VALID_SAMPLERS


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

def _preset_names() -> list[str]:
    return [p["name"] for p in load_presets()]


def _char_names() -> list[str]:
    return [c["name"] for c in load_characters()]


def _group_choices() -> list[str]:
    return list_output_groups()


# ── 탭 1: 만화 생성 ────────────────────────────────────────────────────────────

def refresh_presets_dropdown():
    return gr.update(choices=_preset_names())


def refresh_chars_checkboxgroup():
    return gr.update(choices=_char_names())


def refresh_groups_dropdown():
    return gr.update(choices=_group_choices())


def on_generate(
    situation, dialogue_enabled, dialogue_text,
    style_preset_name, character_names,
    group_name, progress=gr.Progress()
):
    if not situation.strip():
        return None, "", "상황 설명을 입력해주세요."

    progress(0.1, desc="LLM으로 프롬프트 변환 중...")
    try:
        parsed = parse_situation(situation)
    except Exception as e:
        return None, "", f"LLM 오류: {e}"

    scene_prompt = parsed["scene_prompt"]
    scene_negative = parsed["negative_prompt"]

    progress(0.4, desc="NAI V5 이미지 생성 중...")
    img_path, prompt_display, error = generate_image(
        situation=situation,
        scene_prompt=scene_prompt,
        scene_negative=scene_negative,
        style_preset_name=style_preset_name,
        character_names=character_names if character_names else [],
        dialogue_enabled=dialogue_enabled,
        dialogue_text=dialogue_text,
        group_name=group_name
    )

    progress(1.0, desc="완료")
    if error:
        return None, prompt_display, f"오류: {error}"
    return img_path, prompt_display, "생성 완료!"


def build_tab_generate():
    with gr.Column():
        gr.Markdown("## 만화 컷 생성")

        situation_input = gr.Textbox(
            label="대본 / 상황 설명",
            placeholder="예: 비 오는 밤, 교실 창가에 홀로 앉아 창밖을 바라보는 장면. 우울한 분위기.",
            lines=4
        )

        with gr.Row():
            dialogue_enabled = gr.Checkbox(label="대사 포함", value=False)

        dialogue_text = gr.Textbox(
            label="대사 입력 (대사 포함 체크 시 적용)",
            placeholder="예: 오늘도 비가 오네...",
            lines=2,
            visible=False
        )
        dialogue_enabled.change(
            fn=lambda v: gr.update(visible=v),
            inputs=dialogue_enabled,
            outputs=dialogue_text
        )

        style_preset_dd = gr.Dropdown(
            label="그림체 프리셋",
            choices=_preset_names(),
            value=_preset_names()[0] if _preset_names() else None
        )

        char_cbg = gr.CheckboxGroup(
            label="등장 인물 (다중 선택)",
            choices=_char_names()
        )

        group_input = gr.Textbox(
            label="생성 그룹 (저장 폴더명)",
            placeholder="예: 작품A_1화",
            value=""
        )

        with gr.Row():
            refresh_btn = gr.Button("목록 새로고침", size="sm")
            generate_btn = gr.Button("생성", variant="primary", size="lg")

        status_text = gr.Textbox(label="상태", interactive=False, lines=1)
        result_image = gr.Image(label="생성된 이미지", type="filepath", height=500)
        prompt_display = gr.Textbox(
            label="최종 프롬프트 확인",
            interactive=False,
            lines=8
        )

    refresh_btn.click(
        fn=lambda: (
            gr.update(choices=_preset_names()),
            gr.update(choices=_char_names())
        ),
        outputs=[style_preset_dd, char_cbg]
    )

    generate_btn.click(
        fn=on_generate,
        inputs=[
            situation_input, dialogue_enabled, dialogue_text,
            style_preset_dd, char_cbg, group_input
        ],
        outputs=[result_image, prompt_display, status_text]
    )


# ── 탭 2: 인물 프리셋 관리 ────────────────────────────────────────────────────

def on_char_select(name):
    char = next((c for c in load_characters() if c["name"] == name), None)
    if char is None:
        return "", "", ""
    return char["name"], char.get("prompt", ""), char.get("negative_prompt", "")


def on_char_save(name, prompt, negative_prompt):
    if not name.strip():
        return "인물 이름을 입력해주세요.", gr.update(), gr.update()
    upsert_character({"name": name.strip(), "prompt": prompt.strip(), "negative_prompt": negative_prompt.strip()})
    names = _char_names()
    return f"'{name}' 저장 완료.", gr.update(choices=names, value=name.strip()), gr.update(choices=names)


def on_char_delete(name):
    if not name:
        return "삭제할 인물을 선택해주세요.", gr.update(), gr.update()
    delete_character(name)
    names = _char_names()
    return f"'{name}' 삭제 완료.", gr.update(choices=names, value=None), gr.update(choices=names)


def build_tab_characters():
    with gr.Column():
        gr.Markdown("## 인물(Character) 프리셋 관리")

        char_list = gr.Dropdown(
            label="저장된 인물 목록 (선택하여 편집)",
            choices=_char_names(),
            value=None,
            allow_custom_value=False
        )

        char_name = gr.Textbox(label="인물 이름", placeholder="예: 주인공_민수")
        char_prompt = gr.Textbox(
            label="외형 프롬프트 (Positive)",
            placeholder="예: 1boy, black hair, short hair, brown eyes, school uniform",
            lines=4
        )
        char_neg = gr.Textbox(
            label="부정 프롬프트 (Undesired Content)",
            placeholder="예: glasses",
            lines=2
        )

        with gr.Row():
            char_save_btn = gr.Button("저장", variant="primary")
            char_del_btn = gr.Button("삭제", variant="stop")

        char_status = gr.Textbox(label="상태", interactive=False, lines=1)

    char_list.change(fn=on_char_select, inputs=char_list, outputs=[char_name, char_prompt, char_neg])
    char_save_btn.click(
        fn=on_char_save,
        inputs=[char_name, char_prompt, char_neg],
        outputs=[char_status, char_list, char_list]
    )
    char_del_btn.click(
        fn=on_char_delete,
        inputs=char_list,
        outputs=[char_status, char_list, char_list]
    )


# ── 탭 3: 그림체 프리셋 관리 ──────────────────────────────────────────────────

SAMPLER_LIST = sorted(VALID_SAMPLERS)


def on_style_select(name):
    p = next((x for x in load_presets() if x["name"] == name), None)
    if p is None:
        return ("", "", "", "k_euler_ancestral", 28, 6.0, 832, 1216, "karras", 0.0, False)
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
        p.get("smea", False)
    )


def on_style_save(name, pos, neg, sampler, steps, scale, width, height, noise, cfg_rescale, smea):
    if not name.strip():
        return "프리셋 이름을 입력해주세요.", gr.update(), gr.update()
    preset = {
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
        "smea": bool(smea)
    }
    upsert_preset(preset)
    names = _preset_names()
    return f"'{name}' 저장 완료.", gr.update(choices=names, value=name.strip()), gr.update(choices=names)


def on_style_delete(name):
    if not name:
        return "삭제할 프리셋을 선택해주세요.", gr.update(), gr.update()
    delete_preset(name)
    names = _preset_names()
    return f"'{name}' 삭제 완료.", gr.update(choices=names, value=None), gr.update(choices=names)


def build_tab_styles():
    with gr.Column():
        gr.Markdown("## NAI 그림체 프리셋 관리")

        style_list = gr.Dropdown(
            label="저장된 프리셋 목록 (선택하여 편집)",
            choices=_preset_names(),
            value=None
        )

        s_name = gr.Textbox(label="프리셋 이름", placeholder="예: 기본 애니")
        s_pos = gr.Textbox(label="긍정 프롬프트", lines=4,
                           placeholder="best quality, amazing quality, very aesthetic, absurdres")
        s_neg = gr.Textbox(label="부정 프롬프트", lines=3,
                           placeholder="lowres, bad quality, ...")

        with gr.Row():
            s_sampler = gr.Dropdown(label="Sampler", choices=SAMPLER_LIST, value="k_euler_ancestral")
            s_steps = gr.Slider(label="Steps", minimum=1, maximum=50, step=1, value=28)

        with gr.Row():
            s_scale = gr.Slider(label="CFG Scale", minimum=1.0, maximum=10.0, step=0.5, value=6.0)
            s_cfg_rescale = gr.Slider(label="CFG Rescale", minimum=0.0, maximum=1.0, step=0.01, value=0.0)

        with gr.Row():
            s_width = gr.Dropdown(
                label="Width",
                choices=[640, 832, 1024, 1216, 1536],
                value=832
            )
            s_height = gr.Dropdown(
                label="Height",
                choices=[640, 832, 1024, 1216, 1536],
                value=1216
            )

        with gr.Row():
            s_noise = gr.Dropdown(
                label="Noise Schedule",
                choices=["karras", "exponential", "polyexponential", "native"],
                value="karras"
            )
            s_smea = gr.Checkbox(label="SMEA", value=False)

        with gr.Row():
            style_save_btn = gr.Button("저장", variant="primary")
            style_del_btn = gr.Button("삭제", variant="stop")

        style_status = gr.Textbox(label="상태", interactive=False, lines=1)

    style_list.change(
        fn=on_style_select,
        inputs=style_list,
        outputs=[s_name, s_pos, s_neg, s_sampler, s_steps, s_scale,
                 s_width, s_height, s_noise, s_cfg_rescale, s_smea]
    )
    style_save_btn.click(
        fn=on_style_save,
        inputs=[s_name, s_pos, s_neg, s_sampler, s_steps, s_scale,
                s_width, s_height, s_noise, s_cfg_rescale, s_smea],
        outputs=[style_status, style_list, style_list]
    )
    style_del_btn.click(
        fn=on_style_delete,
        inputs=style_list,
        outputs=[style_status, style_list, style_list]
    )


# ── 탭 4: 갤러리 ──────────────────────────────────────────────────────────────

def on_gallery_refresh(group_name):
    if not group_name:
        return []
    images = list_images_in_group(group_name)
    return images


def build_tab_gallery():
    with gr.Column():
        gr.Markdown("## 갤러리 / 그룹별 모아보기")

        with gr.Row():
            gallery_group_dd = gr.Dropdown(
                label="그룹 선택",
                choices=_group_choices(),
                value=None,
                allow_custom_value=False
            )
            gallery_refresh_btn = gr.Button("새로고침", size="sm")

        gallery_output = gr.Gallery(
            label="이미지 목록",
            columns=2,
            object_fit="contain",
            height="auto"
        )

    gallery_group_dd.change(fn=on_gallery_refresh, inputs=gallery_group_dd, outputs=gallery_output)
    gallery_refresh_btn.click(
        fn=lambda: gr.update(choices=_group_choices()),
        outputs=gallery_group_dd
    )


# ── 탭 5: 설정 ────────────────────────────────────────────────────────────────

def load_settings_values():
    cfg = load_config()
    return (
        cfg["llm"]["endpoint"],
        cfg["llm"]["api_key"],
        cfg["llm"]["model"],
        cfg["llm"]["temperature"],
        cfg["nai"]["api_key"],
        cfg["nai"]["model"]
    )


def on_settings_save(
    llm_endpoint, llm_key, llm_model, llm_temp,
    nai_key, nai_model
):
    cfg = load_config()
    cfg["llm"]["endpoint"] = llm_endpoint.strip()
    cfg["llm"]["api_key"] = llm_key.strip()
    cfg["llm"]["model"] = llm_model.strip()
    cfg["llm"]["temperature"] = float(llm_temp)
    cfg["nai"]["api_key"] = nai_key.strip()
    cfg["nai"]["model"] = nai_model.strip()
    save_config(cfg)
    return "설정 저장 완료."


def build_tab_settings():
    cfg_vals = load_settings_values()

    with gr.Column():
        gr.Markdown("## 모델 및 API 설정")

        gr.Markdown("### LLM 설정")
        llm_endpoint = gr.Textbox(
            label="LLM 엔드포인트 (OpenAI-compatible)",
            value=cfg_vals[0],
            placeholder="https://api.openai.com/v1/chat/completions"
        )
        llm_key = gr.Textbox(
            label="LLM API Key",
            value=cfg_vals[1],
            type="password",
            placeholder="sk-..."
        )
        llm_model = gr.Textbox(
            label="LLM 모델명",
            value=cfg_vals[2],
            placeholder="gpt-4o-mini"
        )
        llm_temp = gr.Slider(
            label="Temperature",
            minimum=0.0, maximum=2.0, step=0.05,
            value=float(cfg_vals[3])
        )

        gr.Markdown("### NAI 설정")
        nai_key = gr.Textbox(
            label="NAI API Key",
            value=cfg_vals[4],
            type="password",
            placeholder="pst-..."
        )
        nai_model = gr.Dropdown(
            label="NAI 기본 모델",
            choices=[
                "nai-diffusion-5-full",
                "nai-diffusion-5-curated",
                "nai-diffusion-4-5-full",
                "nai-diffusion-4-5-curated",
                "nai-diffusion-4-full",
                "nai-diffusion-4-curated-preview"
            ],
            value=cfg_vals[5],
            allow_custom_value=True
        )

        settings_save_btn = gr.Button("설정 저장", variant="primary")
        settings_status = gr.Textbox(label="상태", interactive=False, lines=1)

    settings_save_btn.click(
        fn=on_settings_save,
        inputs=[llm_endpoint, llm_key, llm_model, llm_temp, nai_key, nai_model],
        outputs=settings_status
    )


# ── 앱 조립 ───────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* 모바일(Termux 브라우저) 최적화 */
body, .gradio-container {
    max-width: 100% !important;
    padding: 8px !important;
    font-size: 15px;
}
.gr-button {
    min-height: 44px !important;
    font-size: 15px !important;
}
.gr-textbox textarea, .gr-dropdown select {
    font-size: 15px !important;
}
/* 탭 버튼 크게 */
.tab-nav button {
    padding: 10px 8px !important;
    font-size: 13px !important;
}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="NAI V5 만화 생성기",
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS
    ) as app:
        gr.Markdown("# NAI V5 만화 생성기")

        with gr.Tabs():
            with gr.Tab("생성"):
                build_tab_generate()
            with gr.Tab("인물"):
                build_tab_characters()
            with gr.Tab("그림체"):
                build_tab_styles()
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
        quiet=False
    )
