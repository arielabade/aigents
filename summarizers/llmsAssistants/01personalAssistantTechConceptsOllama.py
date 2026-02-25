"""Ollama technology concept assistant with SaaS-style Gradio UI."""

from __future__ import annotations

from datetime import datetime

import gradio as gr
import requests

OLLAMA_API = "http://localhost:11434/api/generate"
OLLAMA_VERSION = "http://localhost:11434/api/version"
OLLAMA_HEADERS = {"Content-Type": "application/json"}
MODEL_NAME = "llama3.2"

SYSTEM_PROMPT = (
    "You are a technical AI assistant. Respond in markdown using this format: "
    "1) What it is, 2) How it was created, 3) Practical use in SaaS products."
)

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


def ollama_is_ready() -> bool:
    try:
        response = requests.get(OLLAMA_VERSION, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def generate_answer(question: str) -> str:
    if not ollama_is_ready():
        return "# Ollama Offline\nStart Ollama locally and try again."

    payload = {
        "model": MODEL_NAME,
        "prompt": f"### System: {SYSTEM_PROMPT}\n\n### User: {question}\n\n### Assistant:",
        "stream": False,
    }
    response = requests.post(OLLAMA_API, headers=OLLAMA_HEADERS, json=payload, timeout=60)
    response.raise_for_status()
    body = response.json().get("response", "No response generated.")

    return (
        "# Technology Concept Explanation\n"
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"## Query\n{question}\n\n"
        f"## Response\n{body}\n\n"
        f"---\nGenerated using {MODEL_NAME}"
    )


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Ollama Tech Assistant", theme=gr.themes.Soft(), css=CSS) as demo:
        gr.Markdown("# Ollama Tech Assistant\nCalm technical responses for AI SaaS portfolio demos.")
        question = gr.Textbox(
            label="Question",
            value="Explain how vector database storage for AI works.",
            lines=3,
        )
        ask_button = gr.Button("Generate Answer", variant="primary")
        output = gr.Markdown()
        ask_button.click(generate_answer, inputs=question, outputs=output)
    return demo


def main() -> None:
    print(generate_answer("Explain how vector database storage for AI works."))


if __name__ == "__main__":
    build_interface().launch()
