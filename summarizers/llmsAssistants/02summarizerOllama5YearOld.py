"""Ollama website summarizer for simple explanations with portfolio UI."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime

import gradio as gr
import requests
from bs4 import BeautifulSoup

MODEL_NAME = "llama3.2:1b"
OLLAMA_API = "http://localhost:11434/api/generate"
OLLAMA_VERSION = "http://localhost:11434/api/version"
HEADERS = {"Content-Type": "application/json"}

SYSTEM_PROMPT = (
    "You analyze websites and explain the content in clear markdown. "
    "Focus on numbers, key events, and simple language for children."
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


@dataclass
class ScrapedPage:
    url: str
    title: str
    content: str


def scrape_page(url: str) -> ScrapedPage:
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        timeout=20,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else "No title found"

    if soup.body:
        for tag in soup.body.find_all(["script", "style", "img", "input", "noscript"]):
            tag.decompose()
        content = soup.body.get_text(separator="\n", strip=True)
    else:
        content = ""

    return ScrapedPage(url=url, title=title, content=content)


def ollama_up() -> bool:
    try:
        return requests.get(OLLAMA_VERSION, timeout=5).status_code == 200
    except requests.RequestException:
        return False


def summarize_url(url: str) -> str:
    if not ollama_up():
        return "# Ollama Offline\nPlease start Ollama locally and try again."

    start = time.time()
    page = scrape_page(url)

    prompt = (
        f"### System: {SYSTEM_PROMPT}\n\n"
        f"### User: Analyze this page titled '{page.title}'. "
        "Summarize in child-friendly language and preserve key numbers.\n\n"
        f"Content:\n{page.content[:15000]}\n\n"
        "### Assistant:"
    )

    payload = {"model": MODEL_NAME, "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_API, headers=HEADERS, json=payload, timeout=90)
    response.raise_for_status()

    summary = response.json().get("response", "No summary generated.")
    elapsed = time.time() - start

    return (
        "# Website Analysis Report\n\n"
        f"- **URL:** {page.url}\n"
        f"- **Title:** {page.title}\n"
        f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- **Processing Time:** {elapsed:.2f}s\n"
        f"- **Model:** {MODEL_NAME}\n\n"
        "## Summary\n"
        f"{summary}"
    )


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Ollama Kid-Friendly Summarizer", theme=gr.themes.Soft(), css=CSS) as demo:
        gr.Markdown("# Ollama Kid-Friendly Summarizer\nAccessible summaries with an AI Clean visual style.")
        url = gr.Textbox(label="Website URL", value="https://en.wikipedia.org/wiki/World_War_II")
        run_button = gr.Button("Analyze Website", variant="primary")
        output = gr.Markdown()
        run_button.click(summarize_url, inputs=url, outputs=output)
    return demo


def main() -> None:
    print(summarize_url("https://en.wikipedia.org/wiki/World_War_II"))


if __name__ == "__main__":
    build_interface().launch()
