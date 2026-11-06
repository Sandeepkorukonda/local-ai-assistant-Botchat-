from core.rag import build_rag_prompt, sources_text


def test_prompt_contains_question_and_pages():
    hits = [(3, "Python is a language."), (1, "It was created in 1991.")]
    prompt = build_rag_prompt("Who made it?", hits)
    assert "Who made it?" in prompt and "[Page 3]" in prompt
    assert "pages 1, 3" in sources_text(hits)