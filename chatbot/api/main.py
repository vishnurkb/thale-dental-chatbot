"""FastAPI backend exposing POST /chat, serving the injectable widget script,
and serving the mirrored site itself at "/" (with the chat bubble already
injected into every page - see scripts/inject_widget.py). Single process,
single port - the widget's API_URL is same-origin relative, so whichever port
this ends up bound to (see run.py's free-port picker), the widget just works
with no separate static server or hardcoded port to keep in sync."""
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chatbot.config import SITE_DIR, CHROMA_DIR
from chatbot.retrieval.retriever import retrieve
from chatbot.llm.factory import get_llm_provider
from chatbot.prompt import SYSTEM_PROMPT, build_context, postprocess_answer

WIDGET_DIR = Path(__file__).resolve().parent.parent / "widget"
EMBED_JS_PATH = WIDGET_DIR / "embed.js"

_sessions: dict[str, list[dict]] = defaultdict(list)
MAX_HISTORY_TURNS = 6

_llm_provider = None
_llm_provider_error = None


def get_provider():
    """Lazy singleton - a bad/missing OPENROUTER_API_KEY (or other provider
    misconfig) must not crash the whole app at import time (that would fail
    the deploy outright on hosts like Render). Instead /chat reports a clear
    error to the caller and the app stays up."""
    global _llm_provider, _llm_provider_error
    if _llm_provider is None and _llm_provider_error is None:
        try:
            _llm_provider = get_llm_provider()
        except Exception as e:
            _llm_provider_error = str(e)
            print(f"[startup] LLM provider init failed: {e}")
    return _llm_provider


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_provider()  # surface config errors immediately in the deploy logs
    if SITE_DIR.exists() and not (CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())):
        print("[startup] No vector index found - building it now...")
        from chatbot.ingest.build_index import build_index
        count = build_index()
        print(f"[startup] Indexed {count} chunks.")
    yield


app = FastAPI(title="Thale Dental Chatbot", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    sources: list[str]


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    message = req.message.strip()
    if not message:
        return ChatResponse(reply="Please type a question.", sources=[])

    provider = get_provider()
    if provider is None:
        print(f"[chat] LLM provider unavailable: {_llm_provider_error}")
        return ChatResponse(
            reply="Sorry, the chatbot is temporarily unavailable. Please try again shortly.",
            sources=[],
        )

    try:
        hits = retrieve(message)
    except Exception as e:
        print(f"[chat] Retrieval error (index may be missing): {e}")
        return ChatResponse(
            reply="Sorry, the chatbot is temporarily unavailable. Please try again shortly.",
            sources=[],
        )
    context = build_context(hits)

    history = _sessions[req.session_id]
    history.append({"role": "user", "content": message})
    history = history[-MAX_HISTORY_TURNS * 2:]

    try:
        answer = provider.generate(SYSTEM_PROMPT, history, context)
    except RuntimeError as e:
        print(f"[chat] LLM provider error: {e}")
        return ChatResponse(
            reply="Sorry, the chatbot is temporarily unavailable. Please try again shortly.",
            sources=[],
        )

    answer = postprocess_answer(answer, hits)
    history.append({"role": "assistant", "content": answer})
    _sessions[req.session_id] = history

    sources = sorted({h["metadata"]["url"] for h in hits})
    return ChatResponse(reply=answer, sources=sources)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/chatbot-widget.js")
def widget_script():
    return FileResponse(EMBED_JS_PATH, media_type="application/javascript")


@app.get("/widget")
def standalone_widget():
    """Bare widget page (no site content) - useful for isolated testing."""
    return FileResponse(WIDGET_DIR / "widget.html")


# Registered last: everything not matched above (i.e. the real site pages)
# falls through to the mirrored site, which already has the widget injected.
if SITE_DIR.exists():
    app.mount("/", StaticFiles(directory=str(SITE_DIR), html=True), name="site")
