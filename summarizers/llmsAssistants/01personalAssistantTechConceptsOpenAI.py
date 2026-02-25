"""OpenAI technology concept assistant with polished Gradio UI."""

from __future__ import annotations

import os
from datetime import datetime

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

MODEL_NAME = "gpt-4o-mini"
SYSTEM_PROMPT = (
    "You are a practical technical assistant for SaaS AI teams. "
    "Use markdown sections: What it is, How it was created, and Practical SaaS use."
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CSS = """
:root {
  --ink-950: #07070C;
  --ink-900: #0B0C12;
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
    linear-gradient(180deg, var(--ink-950) 0%, var(--ink-900) 45%, var(--ink-950) 100%);
  color: var(--text);
}
.gradio-container .prose,
.gradio-container label,
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
button.primary {
  background: linear-gradient(90deg, var(--neon-3), var(--neon-2)) !important;
  color: #0B0C12 !important;
  border: 1px solid rgba(58, 218, 213, 0.35) !important;
  font-weight: 800 !important;
}
"""


def generate_answer(question: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    content = response.choices[0].message.content or "No response generated."
    return (
        "# OpenAI Technical Brief\n"
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"## Query\n{question}\n\n"
        f"## Response\n{content}"
    )


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="OpenAI Tech Assistant", theme=gr.themes.Soft(), css=CSS) as demo:
        gr.Markdown("# OpenAI Tech Assistant\nHuman-friendly AI explanations for product portfolios.")
        question = gr.Textbox(
            label="Question",
            value="Explain vector database storage for AI applications.",
            lines=3,
        )
        ask_button = gr.Button("Generate Answer", variant="primary")
        output = gr.Markdown()
        ask_button.click(generate_answer, inputs=question, outputs=output)
    return demo


def user_prompt_for(_: str) -> str:
    """Backward-compatible helper used by old script."""
    return "Explain vector database storage for AI applications."


def generateAnswer(question: str) -> None:
    """Backward-compatible helper used by old script."""
    print(generate_answer(question))


if __name__ == "__main__":
    build_interface().launch()
