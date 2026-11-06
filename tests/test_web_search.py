from core.web_search import build_web_prompt, sources_markdown

RESULTS = [
    {"title": "Ollama", "url": "https://ollama.com", "snippet": "Run models locally.", "text": ""},
    {"title": "Docs", "url": "https://docs.ollama.com", "snippet": "short", "text": "Full page text here."},
]


def test_prompt_numbers_results_and_has_question():
    prompt = build_web_prompt("What is Ollama?", RESULTS)
    assert "[1] Ollama" in prompt and "[2] Docs" in prompt and "What is Ollama?" in prompt


def test_prompt_prefers_full_page_text_over_snippet():
    prompt = build_web_prompt("q", RESULTS)
    assert "Full page text here." in prompt and "Run models locally." in prompt


def test_sources_markdown_lists_numbered_links():
    md = sources_markdown(RESULTS)
    assert "1. [Ollama](https://ollama.com)" in md and "2. [Docs]" in md