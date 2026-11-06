"""Every call to Ollama chat lives here (local Ollama). Hides Qwen3's <think> text."""
import re

import ollama

import config

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def installed_models() -> list[str]:
    try:
        return [m.model for m in ollama.list().models]
    except Exception:
        return []


def _options() -> dict:
    return {"num_ctx": config.NUM_CTX, "temperature": config.TEMPERATURE}


def _no_think(model: str, messages: list[dict]) -> list[dict]:
    """Qwen3 skips its long 'thinking' step if the last user message ends with /no_think."""
    if not (config.DISABLE_THINKING and model.startswith("qwen2")):
        return messages
    msgs = [dict(m) for m in messages]
    for m in reversed(msgs):
        if m["role"] == "user":
            m["content"] += "\n/no_think"
            break
    return msgs


def hide_think(chunks):
    """Remove <think>...</think> blocks from a stream of text chunks (tags may be split)."""
    buf, in_think, started = "", False, False
    for chunk in chunks:
        buf += chunk
        while True:
            if in_think:
                end = buf.find(THINK_CLOSE)
                if end == -1:
                    buf = buf[-len(THINK_CLOSE):]  # keep a possible partial closing tag
                    break
                buf, in_think = buf[end + len(THINK_CLOSE):], False
            else:
                start = buf.find(THINK_OPEN)
                if start == -1:
                    keep = next((k for k in range(min(len(buf), len(THINK_OPEN) - 1), 0, -1)
                                 if THINK_OPEN.startswith(buf[-k:])), 0)
                    out, buf = buf[:len(buf) - keep], buf[len(buf) - keep:]
                    if not started:
                        out = out.lstrip()
                    if out:
                        started = True
                        yield out
                    break
                out = buf[:start]
                if not started:
                    out = out.lstrip()
                if out:
                    started = True
                    yield out
                buf, in_think = buf[start + len(THINK_OPEN):], True
    if not in_think and buf:
        yield buf if started else buf.lstrip()


def stream_chat(model: str, messages: list[dict]):
    stream = ollama.chat(model=model, messages=_no_think(model, messages),
                         stream=True, options=_options())
    texts = (part["message"]["content"] for part in stream)
    yield from (hide_think(texts) if config.HIDE_THINKING else texts)


def chat_once(model: str, prompt: str) -> str:
    r = ollama.chat(model=model, messages=_no_think(model, [{"role": "user", "content": prompt}]),
                    options=_options())
    text = r["message"]["content"]
    if config.HIDE_THINKING:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.strip()