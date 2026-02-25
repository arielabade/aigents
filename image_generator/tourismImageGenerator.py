"""AI tourism poster generator with a portfolio-ready Gradio interface."""

from __future__ import annotations

import base64
import os
from io import BytesIO
from typing import Tuple

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from openai import OpenAIError
from PIL import Image

load_dotenv(override=True)

MODEL_NAME = "gpt-image-1"


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise OpenAIError(
            "OPENAI_API_KEY is missing. The UI can run without it, but image generation requires it."
        )
    return OpenAI(api_key=api_key)

PALETTES = {
    "AI Clean": {
        "primary": "#6FA8FF",
        "secondary": "#7DE3D1",
        "background": "#FBFDFF",
        "surface": "#E8EEF4",
        "text": "#1F2A37",
        "cta": "#2563EB",
    },
    "AI Premium Pastel": {
        "primary": "#8FAADC",
        "secondary": "#A8E6CF",
        "background": "#F7F9FB",
        "surface": "#EEF2F7",
        "text": "#3A4756",
        "cta": "#10B981",
    },
    "AI Human": {
        "primary": "#9EC5FF",
        "secondary": "#BEEAD9",
        "background": "#F6F4EF",
        "surface": "#EEF0EB",
        "text": "#5B6573",
        "cta": "#2563EB",
    },
}


CSS = """
:root {
  --ink-950: #07070C;
  --ink-900: #0B0C12;
  --ink-800: #111325;
  --line: rgba(255, 255, 255, 0.14);
  --line-strong: rgba(255, 255, 255, 0.24);
  --text: #F2F7FF;
  --muted: rgba(242, 247, 255, 0.72);
  --neon-1: #3ADAD5;
  --neon-2: #3AE0CA;
  --neon-3: #39C3F2;
}
.gradio-container {
  font-family: 'Inter', 'Segoe UI', sans-serif;
  background:
    radial-gradient(900px 420px at 20% 15%, rgba(58, 218, 213, 0.14), transparent 55%),
    radial-gradient(700px 360px at 82% 28%, rgba(57, 195, 242, 0.12), transparent 55%),
    radial-gradient(900px 520px at 55% 78%, rgba(58, 224, 202, 0.10), transparent 55%),
    linear-gradient(180deg, var(--ink-950) 0%, var(--ink-900) 45%, var(--ink-950) 100%);
  color: var(--text);
}
.gradio-container .prose,
.gradio-container label,
.gradio-container .block-title,
.gradio-container .gr-markdown {
  color: var(--text) !important;
}
.gradio-container .gr-box,
.gradio-container .gr-form,
.gradio-container .gr-panel,
.gradio-container .gr-group {
  background: rgba(15, 18, 32, 0.58) !important;
  border: 1px solid var(--line) !important;
  border-radius: 16px !important;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  background: rgba(11, 12, 18, 0.85) !important;
  color: var(--text) !important;
  border: 1px solid var(--line-strong) !important;
}
.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
  color: var(--muted) !important;
}
#app-card {
  border: 1px solid var(--line);
  border-radius: 16px;
  background: rgba(15, 18, 32, 0.58);
  backdrop-filter: blur(12px);
}
button.primary {
  background: linear-gradient(90deg, var(--neon-3), var(--neon-2)) !important;
  color: #0B0C12 !important;
  border: 1px solid rgba(58, 218, 213, 0.35) !important;
  font-weight: 800 !important;
}
.gradio-container a {
  color: var(--neon-1) !important;
}
"""


def build_prompt(city: str, visual_style: str, palette_name: str) -> str:
    palette = PALETTES[palette_name]
    return (
        f"Create a premium tourism poster for {city}. "
        f"Use a {visual_style} visual style, clean SaaS marketing composition, "
        f"dominant colors close to {palette['primary']} and {palette['secondary']}, "
        "soft pastel atmosphere, high detail landmarks, and strong depth."
    )


def generate_image(city: str, visual_style: str, palette_name: str) -> Tuple[Image.Image, str]:
    client = get_openai_client()
    response = client.images.generate(
        model=MODEL_NAME,
        prompt=build_prompt(city, visual_style, palette_name),
        size="1024x1024",
    )
    image_base64 = response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    image = Image.open(BytesIO(image_data))
    caption = (
        "### Portfolio Render\n"
        f"**City:** {city}  \n"
        f"**Style:** {visual_style}  \n"
        f"**Palette:** {palette_name}"
    )
    return image, caption


def build_interface() -> gr.Blocks:
    with gr.Blocks(
        title="AI Tourism Poster Studio",
        theme=gr.themes.Soft(),
        css=CSS,
    ) as demo:
        gr.Markdown(
            "# AI Tourism Poster Studio\n"
            "Create portfolio-ready travel visuals with a modern AI SaaS look."
        )
        with gr.Column(elem_id="app-card"):
            palette = gr.Dropdown(
                choices=list(PALETTES.keys()),
                value="AI Clean",
                label="Color Combo",
            )
            city = gr.Textbox(label="City", placeholder="Tokyo, Rio de Janeiro, Lisbon")
            visual_style = gr.Dropdown(
                choices=["Pop-art", "Renaissance", "Old-school", "Cartoon", "Photorealistic"],
                value="Photorealistic",
                label="Visual Style",
            )
            render_btn = gr.Button("Generate Poster", variant="primary")
        image = gr.Image(label="Generated Poster", type="pil")
        caption = gr.Markdown()

        render_btn.click(generate_image, inputs=[city, visual_style, palette], outputs=[image, caption])

    return demo


def artist(city: str) -> Image.Image:
    """Backward-compatible helper used by previous notebooks."""
    image, _ = generate_image(city=city, visual_style="Photorealistic", palette_name="AI Clean")
    return image


if __name__ == "__main__":
    build_interface().launch()
