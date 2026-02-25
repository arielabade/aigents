"""Dual-model chatbot debate with a polished Gradio portfolio interface."""

from __future__ import annotations

import os
from typing import List, Tuple

import gradio as gr
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

OPENAI_MODEL = "gpt-4o-mini"
OLLAMA_MODEL = "llama3.2"
OLLAMA_API = "http://localhost:11434/api/generate"
OLLAMA_VERSION = "http://localhost:11434/api/version"
OLLAMA_HEADERS = {"Content-Type": "application/json"}

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GPT_SYSTEM = (
    "You are a highly argumentative assistant. "
    "Disagree, challenge assumptions, and use concise snarky tone."
)
OLLAMA_SYSTEM = (
    "You are a calm and diplomatic assistant. "
    "Seek common ground and de-escalate conflicts while staying practical."
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
#chat-shell {
  border-radius: 16px;
  border: 1px solid var(--line);
  background: rgba(15, 18, 32, 0.52);
  backdrop-filter: blur(12px);
}
button.primary {
  background: linear-gradient(90deg, var(--neon-3), var(--neon-2)) !important;
  color: #0B0C12 !important;
  border: 1px solid rgba(58, 218, 213, 0.35) !important;
  font-weight: 800 !important;
}
"""


def ollama_available() -> bool:
    try:
        response = requests.get(OLLAMA_VERSION, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def call_gpt(assistant_history: List[str], user_history: List[str], temperature: float) -> str:
    messages = [{"role": "system", "content": GPT_SYSTEM}]
    for assistant_text, user_text in zip(assistant_history, user_history):
        messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": user_text})

    completion = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=temperature,
        messages=messages,
    )
    return completion.choices[0].message.content or ""


def call_ollama(last_gpt_message: str, temperature: float) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"### System: {OLLAMA_SYSTEM}\n\n### User: {last_gpt_message}\n\n### Assistant:",
        "stream": False,
        "options": {"temperature": temperature},
    }
    response = requests.post(OLLAMA_API, json=payload, headers=OLLAMA_HEADERS, timeout=45)
    response.raise_for_status()
    return response.json().get("response", "")


def run_debate(topic: str, turns: int, temperature: float) -> str:
    if not ollama_available():
        return "## Ollama Offline\nPlease start Ollama locally and retry."

    gpt_messages: List[str] = [f"My position on '{topic}' is absolute, and you are wrong."]
    ollama_messages: List[str] = [f"Let's discuss '{topic}' calmly and find practical common ground."]

    transcript: List[str] = ["# AI Debate Transcript"]
    transcript.append(f"**Topic:** {topic}\n")
    transcript.append(f"### GPT\n{gpt_messages[0]}\n")
    transcript.append(f"### Ollama\n{ollama_messages[0]}\n")

    for turn in range(1, turns + 1):
        gpt_reply = call_gpt(gpt_messages, ollama_messages, temperature)
        ollama_reply = call_ollama(gpt_reply, temperature)

        gpt_messages.append(gpt_reply)
        ollama_messages.append(ollama_reply)

        transcript.append(f"## Round {turn}")
        transcript.append(f"### GPT\n{gpt_reply}\n")
        transcript.append(f"### Ollama\n{ollama_reply}\n")

    return "\n".join(transcript)


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="AI Chat Duel", theme=gr.themes.Soft(), css=CSS) as demo:
        gr.Markdown(
            "# AI Chat Duel\n"
            "Portfolio-style SaaS demo: argumentative GPT versus diplomatic Ollama."
        )
        with gr.Row(elem_id="chat-shell"):
            with gr.Column(scale=1):
                topic = gr.Textbox(label="Debate Topic", value="Should AI agents replace most support teams?")
                turns = gr.Slider(label="Rounds", minimum=1, maximum=8, value=4, step=1)
                temperature = gr.Slider(label="Creativity", minimum=0.1, maximum=1.2, value=0.7, step=0.1)
                run_button = gr.Button("Run Debate", variant="primary")
            with gr.Column(scale=2):
                transcript = gr.Markdown(label="Transcript")

        run_button.click(run_debate, inputs=[topic, turns, temperature], outputs=transcript)

    return demo


def chat_loop() -> None:
    """Backward-compatible CLI runner."""
    print(run_debate("Should AI automate legal workflows?", turns=3, temperature=0.7))


if __name__ == "__main__":
    build_interface().launch()
