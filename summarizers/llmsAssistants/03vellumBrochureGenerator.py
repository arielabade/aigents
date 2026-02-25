"""Streaming brochure generator with portfolio-grade Gradio interface."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List
from urllib.parse import urljoin

import gradio as gr
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from openai import OpenAIError

load_dotenv(override=True)

MODEL_NAME = "gemini-2.0-flash"


def get_openai_client() -> OpenAI:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise OpenAIError(
            "GEMINI_API_KEY is missing. The UI can run without it, but brochure generation requires it."
        )
    return OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

SCRAPE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
}

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
#brochure-shell {
  border: 1px solid var(--line);
  border-radius: 16px;
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


@dataclass
class Website:
    url: str
    title: str
    text: str
    links: List[str]

    @classmethod
    def from_url(cls, url: str) -> "Website":
        response = requests.get(url, headers=SCRAPE_HEADERS, timeout=20)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else "No title found"

        links = []
        for tag in soup.find_all("a"):
            href = tag.get("href")
            if href:
                links.append(urljoin(url, href))

        if soup.body:
            for noisy in soup.body.find_all(["script", "style", "img", "input", "noscript"]):
                noisy.decompose()
            text = soup.body.get_text(separator="\n", strip=True)
        else:
            text = ""

        return cls(url=url, title=title, text=text, links=links)


class BrochureCreator:
    def __init__(self, model: str = MODEL_NAME):
        self.model = model

    def pick_relevant_links(self, website: Website) -> Dict[str, List[Dict[str, str]]]:
        client = get_openai_client()
        system_prompt = (
            "Select brochure-relevant links from a company website. "
            "Respond in JSON with key 'links', each item containing 'type' and 'url'."
        )
        user_prompt = (
            f"Website: {website.url}\n"
            "Ignore privacy, terms, and social links. Prioritize about, product, pricing, docs, careers.\n\n"
            + "\n".join(website.links[:80])
        )

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        user_prompt
                        + "\n\nReturn ONLY valid JSON in this schema: "
                        + '{"links":[{"type":"about page","url":"https://example.com/about"}]}'
                    ),
                },
            ],
        )

        content = response.choices[0].message.content or '{"links": []}'
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(content[start : end + 1])
                except json.JSONDecodeError:
                    pass
            return {"links": []}

    def gather_context(self, company_name: str, website_url: str) -> str:
        landing = Website.from_url(website_url)
        link_map = self.pick_relevant_links(landing)

        blocks = [f"## Company\n{company_name}", f"## Landing Page\n{landing.text[:5000]}"]

        for item in link_map.get("links", [])[:3]:
            link_type = item.get("type", "Additional Page")
            link_url = item.get("url", "")
            if not link_url:
                continue
            page = Website.from_url(link_url)
            blocks.append(f"## {link_type}\n{page.text[:3500]}")

        return "\n\n".join(blocks)

    def stream_brochure(self, company_name: str, website_url: str, extra_requirements: str):
        client = get_openai_client()
        context = self.gather_context(company_name, website_url)
        system_prompt = (
            "You create high-converting B2B AI SaaS brochures in markdown. "
            "Include: Overview, Product Value, Why It Wins, Social Proof, CTA."
        )
        user_prompt = (
            f"Company: {company_name}\n"
            f"Extra requirements: {extra_requirements}\n\n"
            f"Website context:\n{context[:14000]}"
        )

        stream = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
        )

        partial = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            partial += delta
            yield partial


def build_interface() -> gr.Blocks:
    creator = BrochureCreator()

    with gr.Blocks(title="AI Brochure Studio", theme=gr.themes.Soft(), css=CSS) as demo:
        gr.Markdown("# AI Brochure Studio\nStreaming brochure generation for SaaS portfolio projects.")
        with gr.Row(elem_id="brochure-shell"):
            with gr.Column(scale=1):
                company_name = gr.Textbox(label="Company Name", value="Vellum")
                website_url = gr.Textbox(label="Website URL", value="https://www.vellum.ai")
                extra = gr.Textbox(
                    label="Extra Requirements",
                    value="Highlight enterprise reliability and AI workflow governance.",
                    lines=4,
                )
                generate = gr.Button("Generate Brochure", variant="primary")
            with gr.Column(scale=2):
                brochure = gr.Markdown(label="Brochure")

        generate.click(
            creator.stream_brochure,
            inputs=[company_name, website_url, extra],
            outputs=brochure,
        )

    return demo


def generate_brochure() -> None:
    creator = BrochureCreator()
    output = ""
    for chunk in creator.stream_brochure(
        "Vellum",
        "https://www.vellum.ai",
        "Highlight enterprise reliability and workflow orchestration.",
    ):
        output = chunk
    print(output)


if __name__ == "__main__":
    build_interface().launch()
