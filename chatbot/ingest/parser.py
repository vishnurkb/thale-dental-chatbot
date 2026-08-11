"""Parse mirrored HTML pages into (text, metadata) chunks, structure-aware per page type."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re

from bs4 import BeautifulSoup


@dataclass
class Chunk:
    text: str
    page_type: str
    title: str
    url: str
    chunk_id: str


def classify_page_type(rel_path: str) -> str:
    p = rel_path.replace("\\", "/").lower()
    if p.startswith("faq/"):
        return "faq"
    if p.startswith("our-fees/"):
        return "fee"
    if p.startswith("doctor/"):
        return "doctor"
    if p.startswith("contact/"):
        return "contact"
    if p.startswith(("service/", "treatment/", "careplan/", "denplan/",
                      "childrens-check-ups/", "testimonials/", "about/")):
        return "service"
    return "blog"


# This WordPress theme repeats the same nav header and cookie-consent/footer/
# WhatsApp-widget boilerplate on every one of the 109 mirrored pages. Left in,
# it dilutes every chunk's embedding with identical noise (and wastes tokens
# in the LLM context) - strip it before chunking.
_NAV_RE = re.compile(r"Skip to content Phone:.*?Blog Contact", re.IGNORECASE)
_FOOTER_MARKERS = [
    "Necessary Cookies Always Active",
    "Development by AIG Solution",
    "Reject All Accept All Powered by",
]


def _strip_boilerplate(text: str) -> str:
    text = _NAV_RE.sub("", text)
    cut_positions = [p for p in (text.find(m) for m in _FOOTER_MARKERS) if p != -1]
    if cut_positions:
        text = text[: min(cut_positions)]
    return text.strip()


def _clean_text(el) -> str:
    text = el.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return _strip_boilerplate(text)


def _page_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    return _clean_text(h1) if h1 else "Untitled"


def _main_content(soup: BeautifulSoup):
    for sel in ["main", "article", "#content", ".entry-content", "body"]:
        node = soup.select_one(sel)
        if node:
            return node
    return soup


def parse_faq(soup: BeautifulSoup, title: str, url: str) -> list[Chunk]:
    content = _main_content(soup)
    chunks = []
    headers = content.find_all(["h2", "h3"])
    for i, h in enumerate(headers):
        question = _clean_text(h)
        if not question:
            continue
        parts = []
        for sib in h.find_next_siblings():
            if sib.name in ("h2", "h3"):
                break
            txt = _clean_text(sib)
            if txt:
                parts.append(txt)
        answer = " ".join(parts).strip()
        if not answer:
            continue
        chunks.append(Chunk(
            text=f"Q: {question}\nA: {answer}",
            page_type="faq",
            title=question,
            url=url,
            chunk_id=f"{url}#faq-{i}",
        ))
    return chunks


def parse_fees(soup: BeautifulSoup, title: str, url: str) -> list[Chunk]:
    content = _main_content(soup)
    chunks = []
    rows = content.find_all("tr")
    for i, row in enumerate(rows):
        cells = [_clean_text(c) for c in row.find_all(["td", "th"])]
        cells = [c for c in cells if c]
        if len(cells) < 2:
            continue
        treatment, price = cells[0], cells[-1]
        chunks.append(Chunk(
            text=f"{treatment}: {price}",
            page_type="fee",
            title=treatment,
            url=url,
            chunk_id=f"{url}#fee-{i}",
        ))
    if not chunks:
        text = _clean_text(content)
        if text:
            chunks.append(Chunk(text=text, page_type="fee", title=title, url=url, chunk_id=f"{url}#fee-0"))
    return chunks


def parse_doctor(soup: BeautifulSoup, title: str, url: str) -> list[Chunk]:
    content = _main_content(soup)
    text = _clean_text(content)
    if not text:
        return []
    return [Chunk(text=f"{title}: {text}", page_type="doctor", title=title, url=url, chunk_id=f"{url}#doctor-0")]


def parse_contact(soup: BeautifulSoup, title: str, url: str) -> list[Chunk]:
    content = _main_content(soup)
    text = _clean_text(content)
    if not text:
        return []
    return [Chunk(text=text, page_type="contact", title=title, url=url, chunk_id=f"{url}#contact-0")]


def chunk_generic_text(text: str, page_type: str, title: str, url: str,
                        chunk_size_words: int = 220, overlap_words: int = 30) -> list[Chunk]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    idx = 0
    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        piece = " ".join(words[start:end])
        chunks.append(Chunk(text=piece, page_type=page_type, title=title, url=url, chunk_id=f"{url}#chunk-{idx}"))
        idx += 1
        if end == len(words):
            break
        start = end - overlap_words
    return chunks


def parse_page(html_path: Path, rel_path: str, base_url: str) -> list[Chunk]:
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    title = _page_title(soup)

    clean_rel = rel_path.replace("\\", "/").removesuffix("index.html").rstrip("/")
    url = f"{base_url.rstrip('/')}/{clean_rel}".rstrip("/") + "/"

    page_type = classify_page_type(rel_path)

    if page_type == "faq":
        return parse_faq(soup, title, url)
    if page_type == "fee":
        return parse_fees(soup, title, url)
    if page_type == "doctor":
        return parse_doctor(soup, title, url)
    if page_type == "contact":
        return parse_contact(soup, title, url)

    content = _main_content(soup)
    text = _clean_text(content)
    return chunk_generic_text(text, page_type, title, url)
