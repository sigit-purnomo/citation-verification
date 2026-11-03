"""Lapis 2: membaca halaman artikel di situs penerbit (tanpa API).

DOI → halaman artikel → metadata standar yang disisipkan penerbit untuk mesin pengindeks
(Highwire Press `citation_*`, Dublin Core `DC.*`, Open Graph). Dari situ diambil judul,
abstrak, dan tautan PDF. PDF hanya terunduh bila artikelnya memang terbuka.

Aturan kesopanan:
- menghormati robots.txt setiap situs;
- jeda minimal per domain;
- User-Agent jujur (tidak menyamar sebagai browser);
- tidak mencoba melewati login, paywall, atau CAPTCHA.
"""
from __future__ import annotations

import re
import threading
import time
from html import unescape
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests

from .parsing import norm_ws

BOT_NAME = "CiteCheck"
UA = f"Mozilla/5.0 (compatible; {BOT_NAME}/1.0; citation verification for academic proposals)"
MIN_INTERVAL = 2.0  # detik antar-permintaan ke domain yang sama

_lock = threading.Lock()
_last_hit: dict[str, float] = {}
_robots: dict[str, RobotFileParser | None] = {}


# ---------------------------------------------------------------- kesopanan
def _throttle(url: str):
    host = urlparse(url).netloc.lower()
    with _lock:
        wait = _last_hit.get(host, 0) + MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_hit[host] = time.monotonic()


def allowed(url: str) -> bool:
    """Periksa robots.txt (di-cache per domain). Bila robots.txt tak terbaca, dianggap boleh."""
    p = urlparse(url)
    base = f"{p.scheme}://{p.netloc}"
    if base not in _robots:
        rp = None
        try:
            r = requests.get(base + "/robots.txt", timeout=10, headers={"User-Agent": UA})
            if r.status_code == 200 and "text" in r.headers.get("Content-Type", "text"):
                rp = RobotFileParser()
                rp.parse(r.text.splitlines())
        except requests.RequestException:
            pass
        _robots[base] = rp
    rp = _robots[base]
    return True if rp is None else rp.can_fetch(BOT_NAME, url)


def polite_get(url: str, accept: str = "text/html", timeout: int = 25, stream: bool = False):
    if not allowed(url):
        return None, "diblokir robots.txt"
    _throttle(url)
    try:
        r = requests.get(url, timeout=timeout, stream=stream, allow_redirects=True,
                         headers={"User-Agent": UA, "Accept": accept,
                                  "Accept-Language": "en,id;q=0.8"})
    except requests.RequestException as e:
        return None, f"galat jaringan: {type(e).__name__}"
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"
    # pengalihan bisa menuju domain lain yang punya robots.txt berbeda
    if r.url != url and not allowed(r.url):
        return None, "diblokir robots.txt"
    return r, "ok"


# ---------------------------------------------------------------- parsing HTML
class _MetaParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, list[str]] = {}
        self.links: list[tuple[str, str]] = []  # (href, teks/judul)

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "meta":
            key = (a.get("name") or a.get("property") or "").lower()
            if key and a.get("content"):
                self.meta.setdefault(key, []).append(a["content"].strip())
        elif tag == "a" and a.get("href"):
            self.links.append((a["href"], (a.get("title") or a.get("class") or "").lower()))


def _first(meta, *keys) -> str:
    for k in keys:
        for v in meta.get(k, []):
            if v.strip():
                return unescape(v.strip())
    return ""


_ABS_BLOCK = re.compile(
    r"<(section|div)[^>]*(?:class|id)\s*=\s*\"[^\"]*abstract[^\"]*\"[^>]*>(.*?)</\1>", re.I | re.S)


def _abstract_from_body(html: str) -> str:
    for m in _ABS_BLOCK.finditer(html):
        text = norm_ws(unescape(re.sub(r"<[^>]+>", " ", m.group(2))))
        text = re.sub(r"^(abstract|abstrak)\s*[:.]?\s*", "", text, flags=re.I)
        if len(text) > 200:
            return text[:5000]
    return ""


def parse_landing(html: str, page_url: str) -> dict:
    p = _MetaParser()
    try:
        p.feed(html)
    except Exception:  # noqa: BLE001 - HTML rusak
        pass
    m = p.meta
    title = _first(m, "citation_title", "dc.title", "og:title")
    abstract = _first(m, "citation_abstract", "dc.description", "dcterms.abstract", "description")
    abstract = norm_ws(re.sub(r"<[^>]+>", " ", abstract))
    if len(abstract) < 200:  # deskripsi meta sering terpotong → cari blok abstrak di isi halaman
        abstract = _abstract_from_body(html) or abstract
    abstract = re.sub(r"^(abstract|abstrak)\s*[:.]?\s*", "", abstract, flags=re.I)

    pdfs = [urljoin(page_url, u) for u in m.get("citation_pdf_url", [])]
    for href, hint in p.links:  # tautan PDF umum (OJS: kelas "obj_galley_link pdf")
        if href.lower().endswith(".pdf") or "pdf" in hint or "/article/download/" in href:
            u = urljoin(page_url, href)
            if u not in pdfs:
                pdfs.append(u)
    return dict(title=title, abstract=abstract if len(abstract) >= 120 else "",
                pdf_urls=pdfs[:4], url=page_url)


def _pdf_variants(u: str) -> list[str]:
    """OJS: /article/view/A/B menampilkan viewer HTML; berkasnya ada di /article/download/A/B."""
    out = [u]
    if "/article/view/" in u:
        out.insert(0, u.replace("/article/view/", "/article/download/"))
    return out


def download_pdf(url: str, limit: int = 30_000_000) -> tuple[bytes | None, str]:
    for u in _pdf_variants(url):
        r, status = polite_get(u, accept="application/pdf,*/*;q=0.8", timeout=40, stream=True)
        if r is None:
            continue
        data = b""
        for chunk in r.iter_content(1 << 16):
            data += chunk
            if len(data) > limit:
                return None, "berkas terlalu besar"
        r.close()
        if data[:5] == b"%PDF-":
            return data, "ok"
        status = "bukan PDF (kemungkinan berbayar/perlu login)"
    return None, status


def read_landing(doi: str = "", url: str = "") -> tuple[dict | None, str]:
    """Buka halaman artikel dari DOI (atau URL di daftar pustaka) dan baca metadatanya."""
    target = f"https://doi.org/{doi}" if doi else url
    if not target:
        return None, "tidak ada DOI/URL"
    r, status = polite_get(target)
    if r is None:
        return None, status
    if "html" not in r.headers.get("Content-Type", "html"):
        return None, "bukan halaman HTML"
    return parse_landing(r.text, r.url), "ok"
