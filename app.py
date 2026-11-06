"""Streamlit UI. All AI logic lives in core/. Run with: streamlit run app.py"""

import hashlib
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load environment variables before importing modules that may use them.
load_dotenv()

import config
from core import llm, pdf_utils, rag, summarizer, voice, web_search


# ---------- Page configuration ----------
BASE_DIR = Path(__file__).resolve().parent
LOGO_PATH = BASE_DIR / "chatbot_logo.png"

st.set_page_config(
    page_title="Botchat AI",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🤖",
    layout="wide",
)


# ---------- Header ----------
header_col1, header_col2 = st.columns([1, 7], vertical_alignment="center")

with header_col1:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=85)
    else:
        st.markdown("# 🤖")

with header_col2:
    st.title("Botchat AI")
    st.caption("Your personal AI assistant powered by Ollama")


# ---------- State ----------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("doc", None)
st.session_state.setdefault("last_audio", None)


# ---------- Sidebar ----------
with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=110)

    st.title("Botchat AI")
    st.caption("Your AI-powered workspace")
    st.divider()

    models = llm.installed_models()

    if not models:
        st.error(
            "Ollama isn't running or no models are installed.\n\n"
            f"Run: `ollama pull {config.CHAT_MODEL_DEFAULT}`"
        )
        st.stop()

    default_idx = next(
        (
            i
            for i, installed_model in enumerate(models)
            if installed_model.startswith(config.CHAT_MODEL_DEFAULT)
        ),
        0,
    )
    model = st.selectbox("Model", models, index=default_idx)

    st.header("PDF")
    pdf = st.file_uploader("Upload a PDF", type="pdf")

    if pdf and (
        st.session_state.doc is None
        or st.session_state.doc["name"] != pdf.name
    ):
        with st.spinner("Reading and indexing PDF..."):
            pages = pdf_utils.read_pdf(pdf)

            if not pages:
                st.error(
                    "No text found. This may be a scanned PDF (needs OCR)."
                )
            else:
                chunks = pdf_utils.chunk_pages(pages)

                try:
                    vectors = rag.embed([chunk[1] for chunk in chunks])
                except Exception as e:
                    st.error(
                        f"Embedding failed. Run `ollama pull "
                        f"{config.EMBED_MODEL}`.\n\n{e}"
                    )
                    st.stop()

                st.session_state.doc = {
                    "name": pdf.name,
                    "text": "\n\n".join(text for _, text in pages),
                    "chunks": chunks,
                    "vectors": vectors,
                }

    use_pdf = False

    if st.session_state.doc:
        st.success(f"Loaded: {st.session_state.doc['name']}")

        style = st.selectbox("Summary style", list(config.STYLES))

        if st.button("Summarize PDF"):
            with st.spinner("Summarizing (long PDFs take a while)..."):
                summary = summarizer.summarize(
                    model,
                    st.session_state.doc["text"],
                    style,
                )

            st.session_state.messages.append(
                {"role": "assistant", "content": summary}
            )

        use_pdf = st.checkbox(
            "Answer questions from the PDF",
            value=True,
        )

    st.header("Web")
    use_web = st.checkbox("🌐 Search the web")

    if use_web:
        if not os.getenv("OLLAMA_API_KEY"):
            st.warning(
                "Add OLLAMA_API_KEY to your .env file to use web search."
            )
        elif use_pdf:
            st.caption(
                "PDF answers are on, so web search is skipped. "
                "Untick the PDF box to search the web."
            )
        else:
            st.caption("Your question is sent to a search service.")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------- Chat history ----------
if not st.session_state.messages:
    st.markdown("### 👋 Hello! How can I help you today?")
    st.caption(
        "Ask a question, upload a PDF, or enable web search from the sidebar."
    )

for message in st.session_state.messages:
    if message["role"] == "assistant":
        with st.chat_message(
            "assistant",
            avatar=str(LOGO_PATH) if LOGO_PATH.exists() else "🤖",
        ):
            st.markdown(message["content"])
    else:
        with st.chat_message("user", avatar="🧑"):
            st.markdown(message["content"])


# ---------- Input: typed or spoken ----------
typed = st.chat_input("Message Botchat AI...")
audio = st.audio_input("🎙️ Or speak your question")

prompt = typed

if audio is not None:
    audio_data = audio.getvalue()
    audio_digest = hashlib.md5(audio_data).hexdigest()

    if audio_digest != st.session_state.last_audio:
        st.session_state.last_audio = audio_digest

        with st.spinner("Transcribing..."):
            prompt = voice.transcribe(audio_data)


# ---------- Answer ----------
if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    history = [
        {"role": "system", "content": config.SYSTEM_PROMPT}
    ] + st.session_state.messages[-10:]

    sources = ""

    if use_pdf and st.session_state.doc:
        document = st.session_state.doc
        hits = rag.retrieve(
            prompt,
            document["chunks"],
            document["vectors"],
        )

        history[-1] = {
            "role": "user",
            "content": rag.build_rag_prompt(prompt, hits),
        }
        sources = rag.sources_text(hits)

    elif use_web:
        try:
            with st.spinner("Searching the web..."):
                query = web_search.make_query(model, prompt)
                results = web_search.search_web(query)

            if results:
                history[-1] = {
                    "role": "user",
                    "content": web_search.build_web_prompt(prompt, results),
                }
                sources = web_search.sources_markdown(results)

                st.caption(f"🔎 Searched for: {query}")

                with st.expander("Raw search results (for checking)"):
                    for result in results:
                        st.markdown(
                            f"**{result['title']}**  \n{result['url']}"
                        )
                        st.write(
                            (result["text"] or result["snippet"])[:500]
                        )
            else:
                st.warning(
                    "No web results found. Answering without search."
                )

        except Exception as e:
            st.warning(
                f"Web search failed, answering without it. ({e})"
            )

    with st.chat_message(
        "assistant",
        avatar=str(LOGO_PATH) if LOGO_PATH.exists() else "🤖",
    ):
        try:
            answer = st.write_stream(
                llm.stream_chat(model, history)
            )

            if sources:
                st.markdown(sources)

        except Exception as e:
            answer = f"Error talking to Ollama: {e}"
            st.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer + sources,
        }
    )
