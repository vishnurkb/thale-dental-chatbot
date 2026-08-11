"""One-time (idempotent) post-process: insert the chatbot widget <script> tag
into every mirrored site page so the whole site shows the chat bubble when
served through the backend. Safe to re-run - skips pages that already have it."""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SITE_DIR = BASE_DIR / "site"
TAG = '<script src="/chatbot-widget.js"></script>'


def inject():
    if not SITE_DIR.exists():
        print(f"No site/ directory at {SITE_DIR}")
        return 0

    changed = 0
    for path in SITE_DIR.rglob("*.html"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if TAG in text:
            continue
        if "</body>" in text:
            text = text.replace("</body>", f"{TAG}\n</body>", 1)
        else:
            text += f"\n{TAG}\n"
        path.write_text(text, encoding="utf-8", errors="ignore")
        changed += 1

    print(f"Injected widget script into {changed} page(s).")
    return changed


if __name__ == "__main__":
    sys.exit(0 if inject() >= 0 else 1)
