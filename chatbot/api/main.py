"""FastAPI backend exposing POST /chat, serving the injectable widget script,
and serving the mirrored site itself at "/" (with the chat bubble already
injected into every page - see scripts/inject_widget.py). Single process,
single port - the widget's API_URL is same-origin relative, so whichever port
this ends up bound to (see run.py's free-port picker), the widget just works
with no separate static server or hardcoded port to keep in sync."""
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chatbot.config import SITE_DIR
from chatbot.retrieval.retriever import retrieve
from chatbot.llm.factory import get_llm_provider
from chatbot.prompt import SYSTEM_PROMPT, build_context, postprocess_answer

WIDGET_DIR = Path(__file__).resolve().parent.parent / "widget"
EMBED_JS_PATH = WIDGET_DIR / "embed.js"

app = FastAPI(title="Thale Dental Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

llm_provider = get_llm_provider()
_sessions: dict[str, list[dict]] = defaultdict(list)
MAX_HISTORY_TURNS = 6


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

    hits = retrieve(message)
    context = build_context(hits)

    history = _sessions[req.session_id]
    history.append({"role": "user", "content": message})
    history = history[-MAX_HISTORY_TURNS * 2:]

    try:
        answer = llm_provider.generate(SYSTEM_PROMPT, history, context)
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
