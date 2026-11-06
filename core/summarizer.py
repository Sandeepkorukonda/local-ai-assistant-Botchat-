"""Summarize short PDFs in one call, long ones with map-reduce."""
import config
from core.llm import chat_once


def summarize(model: str, full_text: str, style: str) -> str:
    instr = config.STYLES[style]
    if len(full_text) <= 10000:
        return chat_once(model, f"{instr}\n\nDocument:\n{full_text}")
    parts = [full_text[i:i + 8000] for i in range(0, len(full_text), 8000)]
    partials = [chat_once(model, f"Summarize this section in 5-6 sentences:\n\n{p}") for p in parts]
    combined = "\n\n".join(partials)
    return chat_once(model, f"{instr}\n\nThese are section summaries of one document:\n{combined}")