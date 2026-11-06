# Local Chatbot with Ollama

A private chatbot that runs fully on your computer. It can chat, summarize PDFs, answer questions from a PDF (with page sources), and take voice input.

## Features
- Streamed chat with conversation memory
- PDF summary (short paragraph / bullets / key takeaways) using map-reduce for long files
- PDF question answering (RAG) with page references
- Voice input using faster-whisper
- Optional web search for fresh information (Ollama web search API) with source links
- Works offline after models are downloaded

## Tech stack
Python, Ollama, Streamlit, pypdf, faster-whisper, NumPy, python-dotenv

## Setup
1. Install Ollama from https://ollama.com
2. `ollama pull llama3.2` and `ollama pull nomic-embed-text`
3. `python -m venv venv` and activate it
4. `pip install -r requirements.txt`
5. (Optional, for web search) copy `.env.example` to `.env` and add your `OLLAMA_API_KEY`
6. `streamlit run app.py`

## Project structure
- `app.py` – Streamlit UI
- `config.py` – settings
- `core/` – llm, pdf_utils, rag, summarizer, voice, web_search
- `tests/` – unit tests (`pytest`)

## Limitations
Scanned PDFs need OCR (not included). Small local models are less accurate than large cloud models, and speed depends on your hardware.

## Future work
OCR, multiple PDFs, saved chat history, text-to-speech, satellite image analysis (EuroSAT)