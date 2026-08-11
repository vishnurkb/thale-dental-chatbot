#!/usr/bin/env python3
"""
Static mirror of thealedental.co.uk
Crawls all internal pages + downloads all assets (css/js/img/fonts),
rewrites links to relative local paths so site is browsable offline
and ready to embed a chatbot widget into.
"""
import os
import re
import sys
import time
import queue
import urllib.parse as up

import requests
from bs4 import BeautifulSoup

BASE = "https://thealedental.co.uk"
BASE_HOST = up.urlparse(BASE).netloc
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 SiteMirrorBot/1.0"
}

session = requests.Session()
session.headers.update(HEADERS)

visited_pages = set()
downloaded_assets = set()
page_queue = queue.Queue()
page_queue.put(BASE + "/")

ASSET_EXT = (
    ".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico",
    ".woff", ".woff2", ".ttf", ".eot", ".otf", ".mp4", ".webm", ".pdf", ".json",
    ".avif",
)

MAX_PAGES = 500  # safety cap


def url_to_local_path(url):
    """Map a URL to a local file path under OUT_DIR, mirroring URL structure."""
    parsed = up.urlparse(url)
    path = parsed.path
    if path == "" or path == "/":
        path = "/index.html"
    elif path.endswith("/"):
        path = path + "index.html"
    else:
        # if no extension, treat as a page -> save as .html
        base, ext = os.path.splitext(path)
        if not ext:
            path = path + "/index.html" if False else path + ".html"
    local = os.path.join(OUT_DIR, path.lstrip("/").replace("/", os.sep))
    return local


def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def is_internal(url):
    p = up.urlparse(url)
    return (p.netloc == "" or p.netloc == BASE_HOST)


def normalize(url, base_url):
    if not url:
        return None
    url = url.strip()
    if url.startswith(("mailto:", "tel:", "javascript:", "#", "data:")):
        return None
    full = up.urljoin(base_url, url)
    full, _frag = up.urldefrag(full)
    return full


def download_asset(url):
    if url in downloaded_assets:
        return
    downloaded_assets.add(url)
    local_path = url_to_local_path(url)
    if os.path.exists(local_path):
        return
    try:
        r = session.get(url, timeout=30)
        if r.status_code != 200:
            print(f"  [asset {r.status_code}] {url}")
            return
        ensure_dir(local_path)
        with open(local_path, "wb") as f:
            f.write(r.content)
        print(f"  [asset OK] {url}")
    except Exception as e:
        print(f"  [asset ERR] {url} -> {e}")


def rewrite_and_collect(soup, page_url):
    """Rewrite asset/link URLs to local relative paths, queue internal pages,
    download assets. Returns modified soup."""
    page_local = url_to_local_path(page_url)

    def rel_from_page(target_local):
        rel = os.path.relpath(target_local, os.path.dirname(page_local))
        return rel.replace(os.sep, "/")

    tag_attrs = [
        ("img", "src"), ("img", "data-src"), ("script", "src"),
        ("link", "href"), ("a", "href"), ("source", "src"),
        ("video", "src"), ("audio", "src"), ("iframe", "src"),
    ]

    for tag_name, attr in tag_attrs:
        for tag in soup.find_all(tag_name):
            val = tag.get(attr)
            if not val:
                continue
            full = normalize(val, page_url)
            if not full or not is_internal(full):
                continue
            if full.lower().endswith(ASSET_EXT):
                download_asset(full)
                tag[attr] = rel_from_page(url_to_local_path(full))
            else:
                # internal page link
                if tag_name == "a":
                    clean, _ = up.urldefrag(full)
                    if clean.rstrip("/") not in [p.rstrip("/") for p in visited_pages]:
                        page_queue.put(clean)
                    tag[attr] = rel_from_page(url_to_local_path(clean))

    # srcset handling (responsive images)
    for tag in soup.find_all(["img", "source"]):
        srcset = tag.get("srcset")
        if not srcset:
            continue
        parts = []
        for entry in srcset.split(","):
            entry = entry.strip()
            if not entry:
                continue
            bits = entry.split()
            u = bits[0]
            descriptor = " " + " ".join(bits[1:]) if len(bits) > 1 else ""
            full = normalize(u, page_url)
            if full and is_internal(full) and full.lower().endswith(ASSET_EXT):
                download_asset(full)
                parts.append(rel_from_page(url_to_local_path(full)) + descriptor)
            else:
                parts.append(entry)
        tag["srcset"] = ", ".join(parts)

    # inline <style> url(...) references
    for style_tag in soup.find_all("style"):
        if style_tag.string:
            style_tag.string = rewrite_css_urls(style_tag.string, page_url, rel_from_page)

    return soup


def rewrite_css_urls(css_text, base_url, rel_from_page):
    def repl(m):
        quote = m.group(1) or ""
        url = m.group(2)
        full = normalize(url, base_url)
        if full and is_internal(full):
            download_asset(full)
            return f"url({quote}{rel_from_page(url_to_local_path(full))}{quote})"
        return m.group(0)
    return re.sub(r'url\((["\']?)([^"\')]+)\1\)', repl, css_text)


def process_css_file(local_path, css_url):
    """After downloading a .css file, rewrite its internal url(...) refs too."""
    try:
        with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception:
        return

    def rel_from_css(target_local):
        rel = os.path.relpath(target_local, os.path.dirname(local_path))
        return rel.replace(os.sep, "/")

    new_text = rewrite_css_urls(text, css_url, rel_from_css)
    if new_text != text:
        with open(local_path, "w", encoding="utf-8", errors="ignore") as f:
            f.write(new_text)


def crawl():
    count = 0
    css_urls_downloaded_this_run = set()
    while not page_queue.empty() and count < MAX_PAGES:
        url = page_queue.get()
        clean_url = url.rstrip("/") if url.rstrip("/") != BASE.rstrip("/") else url
        if clean_url in visited_pages or url in visited_pages:
            continue
        visited_pages.add(url)
        visited_pages.add(clean_url)

        try:
            r = session.get(url, timeout=30)
        except Exception as e:
            print(f"[page ERR] {url} -> {e}")
            continue

        ctype = r.headers.get("Content-Type", "")
        if r.status_code != 200:
            print(f"[page {r.status_code}] {url}")
            continue
        if "text/html" not in ctype:
            continue

        count += 1
        print(f"[{count}] page OK: {url}")

        soup = BeautifulSoup(r.text, "html.parser")
        # snapshot asset set before, to know which css files are new
        before_assets = set(downloaded_assets)
        soup = rewrite_and_collect(soup, url)
        new_css = [a for a in downloaded_assets - before_assets if a.lower().endswith(".css")]

        local_path = url_to_local_path(url)
        ensure_dir(local_path)
        with open(local_path, "w", encoding="utf-8", errors="ignore") as f:
            f.write(str(soup))

        for css_url in new_css:
            process_css_file(url_to_local_path(css_url), css_url)

        time.sleep(0.15)  # be polite

    print(f"\nDone. Pages: {count}, Assets: {len(downloaded_assets)}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    crawl()
