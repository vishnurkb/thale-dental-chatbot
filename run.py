"""One-command launcher: makes sure Ollama is up, makes sure the vector index
exists, picks a free port, starts the backend (which also serves the widget at
"/"), and opens it in the default browser. Used by start.bat."""
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from chatbot.config import CHROMA_DIR, OLLAMA_HOST, SITE_DIR, LLM_PROVIDER  # noqa: E402

PREFERRED_PORT = 8001
OLLAMA_EXE_CANDIDATES = [
    Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe",
]


def find_free_port(preferred: int) -> int:
    """Try the preferred port first; if taken, ask the OS for any free port."""
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("Could not find a free port")


def ollama_is_up() -> bool:
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/version", timeout=2)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def start_ollama_if_needed():
    if ollama_is_up():
        print("[run] Ollama already running.")
        return
    exe = next((p for p in OLLAMA_EXE_CANDIDATES if p.exists()), None)
    if exe is None:
        print("[run] WARNING: Ollama not found and not running. Install from "
              "https://ollama.com/download, then re-run this script.")
        return
    print(f"[run] Starting Ollama from {exe} ...")
    subprocess.Popen([str(exe), "serve"], creationflags=subprocess.CREATE_NO_WINDOW
                      if hasattr(subprocess, "CREATE_NO_WINDOW") else 0)
    for _ in range(20):
        if ollama_is_up():
            print("[run] Ollama is up.")
            return
        time.sleep(1)
    print("[run] WARNING: Ollama did not come up in time; chat answers may fail.")


def ensure_index_built():
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        print(f"[run] Vector index already exists at {CHROMA_DIR}.")
        return
    if not SITE_DIR.exists():
        print(f"[run] WARNING: site mirror not found at {SITE_DIR}; chatbot will "
              "have no knowledge until it exists and the index is built.")
        return
    print("[run] No index found - building it now (one-time, may take a minute)...")
    from chatbot.ingest.build_index import build_index
    count = build_index()
    print(f"[run] Indexed {count} chunks.")


def open_browser_when_ready(url: str):
    for _ in range(30):
        try:
            requests.get(url, timeout=1)
            webbrowser.open(url)
            return
        except requests.exceptions.RequestException:
            time.sleep(0.5)
    print(f"[run] Server did not come up in time; open {url} manually.")


def main():
    if LLM_PROVIDER == "ollama":
        start_ollama_if_needed()
    else:
        print(f"[run] LLM_PROVIDER={LLM_PROVIDER} - skipping local Ollama startup.")
    ensure_index_built()

    port = find_free_port(PREFERRED_PORT)
    url = f"http://127.0.0.1:{port}/"
    print(f"[run] Starting chatbot on {url}")

    threading.Thread(target=open_browser_when_ready, args=(url,), daemon=True).start()

    import uvicorn
    from chatbot.api.main import app
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    main()
