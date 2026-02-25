"""OpenAI web summarizer with a portfolio-friendly Gradio interface."""

from __future__ import annotations

import os
from dataclasses import dataclass

import gradio as gr
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

MODEL_NAME = "gpt-4o-mini"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
}

SYSTEM_PROMPT = (
    "You summarize websites with focus on important numbers and business signals. "
    "Explain clearly and keep markdown structure easy to scan."
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
class Website:
    url: str
    title: str
    text: str

    @classmethod
    def from_url(cls, url: str) -> "Website":
        response = requests.get(url, headers=HEADERS, timeout=20)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else "No title"

        if soup.body:
            for tag in soup.body.find_all(["script", "style", "img", "input", "noscript"]):
                tag.decompose()
            text = soup.body.get_text(separator="\n", strip=True)
        else:
            text = ""

        return cls(url=url, title=title, text=text)


def summarize_website(url: str, explain_like_child: bool) -> str:
    website = Website.from_url(url)
    user_prompt = (
        f"Website title: {website.title}\n"
        "Summarize the key ideas, highlight numeric data, and include business implications.\n"
        f"Explain for a 5-year-old: {'yes' if explain_like_child else 'no'}\n\n"
        f"Website content:\n{website.text[:12000]}"
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    summary = response.choices[0].message.content or "No summary generated."
    return f"# Website Summary\n\n**Source:** {url}\n\n{summary}"


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="OpenAI Website Summarizer", theme=gr.themes.Soft(), css=CSS) as demo:
        gr.Markdown("# OpenAI Website Summarizer\nPortfolio-ready analysis with AI Clean palette.")
        url = gr.Textbox(label="Website URL", value="https://www.bbc.com/news")
        explain_like_child = gr.Checkbox(label="Explain like I am 5 years old", value=False)
        run_button = gr.Button("Generate Summary", variant="primary")
        output = gr.Markdown(label="Summary")
        run_button.click(summarize_website, inputs=[url, explain_like_child], outputs=output)
    return demo


def summarize(url: str) -> str:
    """Backward-compatible helper used by prior examples."""
    return summarize_website(url=url, explain_like_child=True)


if __name__ == "__main__":
    build_interface().launch()
