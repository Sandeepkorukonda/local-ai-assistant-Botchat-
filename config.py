"""All settings in one place."""

# ---------- Models ----------
CHAT_MODEL_DEFAULT = "qwen2.5"     # matched by the start of the name (picks qwen2.5:3b)
EMBED_MODEL = "nomic-embed-text"
NUM_CTX = 6144                     # context window; lower = faster, higher = reads more text
TEMPERATURE = 0                    # 0 = consistent answers
QWEN_OPTIONS = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}   # safe settings for Qwen3
HIDE_THINKING = True               # remove <think>...</think> text (Qwen3)
DISABLE_THINKING = True            # ask Qwen3 to skip thinking (faster on CPU)

# ---------- PDF ----------
CHUNK_SIZE = 1200                  # characters per chunk
CHUNK_OVERLAP = 150
TOP_K = 3                          # chunks retrieved per question

# ---------- Voice ----------
WHISPER_SIZE = "base"              # tiny / base / small / medium

# ---------- Chat ----------
SYSTEM_PROMPT = "You are a helpful assistant. Be clear and concise."

STYLES = {
    "Short paragraph": "Write a concise summary in one short paragraph.",
    "Bullet points": "Summarize the document as clear bullet points.",
    "Key takeaways": "List the 5 most important takeaways, each with one line of explanation.",
}

# ---------- Web search ----------
WEB_PROVIDER = "ddgs"              # "ddgs" (free, no key) or "tavily" (needs TAVILY_API_KEY)
WEB_MAX_RESULTS = 4                # search results to use
WEB_FETCH_PAGES = 1                # read the full text of the top N pages (0 = snippets only, fastest)
WEB_FETCH_TIMEOUT = 6              # seconds to wait for pages; slow pages are skipped
WEB_PAGE_CHARS = 3000              # max characters kept from each fetched page
WEB_MAX_CHARS = 5000               # total characters of web text sent to the model
WEB_REWRITE_QUERY = False          # True = model improves your search query (better, but slower)