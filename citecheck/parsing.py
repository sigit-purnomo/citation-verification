"""Ekstraksi teks dokumen (PDF/DOCX/TXT) dan parsing daftar pustaka."""
from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass

DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"<>]+", re.I)
URL_RE = re.compile(r"https?://\S+")
UC = "A-ZÀ-ÖØ-Þ"


def fold(s: str) -> str:
    """Huruf kecil + buang diakritik, untuk pencocokan nama."""
    s = unicodedata.normalize("NFKD", s or "")
    return s.encode("ascii", "ignore").decode().lower().strip()


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


# ---------------------------------------------------------------- teks
def extract_text(data: bytes, filename: str) -> str:
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        try:
            import pymupdf as fitz
        except ImportError:  # versi lama
            import fitz
        with fitz.open(stream=data, filetype="pdf") as doc:
            text = "\n".join(page.get_text("text") for page in doc)
        return clean_pdf_text(text)
    if name.endswith(".docx"):
        import docx
        d = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in d.paragraphs)
    return data.decode("utf-8", errors="ignore")


def clean_pdf_text(text: str) -> str:
    text = (text.replace("­", "").replace("ﬁ", "fi")
            .replace("ﬂ", "fl").replace(" ", " "))
    text = re.sub(r"(\w)-\n([a-z])", r"\1\2", text)  # pemenggalan kata akhir baris
    text = re.sub(r"[ \t]+", " ", text)
    return text


_REF_HEAD = re.compile(
    r"^[ \t]*(?:\d+\.?[ \t]*|[IVX]+\.[ \t]*|BAB[ \t]+\w+[ \t]*)?"
    r"(daftar[ \t]+pustaka|daftar[ \t]+referensi|referensi|references|reference[ \t]+list|"
    r"bibliography|bibliografi|kepustakaan|literature[ \t]+cited|works[ \t]+cited)[ \t]*:?[ \t]*$",
    re.I | re.M,
)
_APPX_HEAD = re.compile(r"^[ \t]*(lampiran|appendix|appendices)\b[^\n]{0,60}$", re.I | re.M)


def split_body_refs(text: str) -> tuple[str, str]:
    """Pisahkan isi proposal dan bagian daftar pustaka (judul terakhir yang cocok)."""
    heads = list(_REF_HEAD.finditer(text))
    if not heads:
        return text, ""
    h = heads[-1]
    body, refs = text[: h.start()], text[h.end():]
    ap = _APPX_HEAD.search(refs)
    if ap:
        refs = refs[: ap.start()]
    return body, refs


# ---------------------------------------------------------------- daftar pustaka
@dataclass
class Reference:
    idx: int            # urutan dalam daftar pustaka (1-based)
    raw: str
    num: int = 0        # nomor gaya numerik [n], jika ada
    first_author: str = ""
    authors: str = ""
    year: str = ""
    title: str = ""
    doi: str = ""
    url: str = ""

    @property
    def label(self) -> str:
        names = re.findall(rf"([{UC}][\w'’\-]+),\s+[{UC}](?:\.|\b)", self.authors)
        name = self.first_author or "?"
        if len(names) == 2:
            name = f"{names[0]} & {names[1]}"
        elif len(names) > 2 or (not names and re.search(r"&| dan | and ", self.authors)):
            name += " et al."
        return f"{name} ({self.year or 'n.d.'})"


_NUM_START = re.compile(r"^\s*(?:\[(\d{1,3})\]|(\d{1,3})\.\s)")
_APA_START = re.compile(rf"^\s*[{UC}][\w'’\-]+(?:[ \-][A-Za-zÀ-ÿ'’\-]+){{0,3}},\s+[{UC}]\.")
_ORG_START = re.compile(rf"^\s*[{UC}][^,()\n]{{2,90}}\.?\s*\((?:(?:19|20)\d{{2}}|n\.d\.)")


def _complete(buf: str) -> bool:
    return len(buf) > 40 and re.search(r"(19|20)\d{2}|n\.d\.", buf) is not None


def split_references(ref_text: str) -> list[str]:
    lines = [ln.strip() for ln in ref_text.splitlines() if ln.strip()]
    entries, buf = [], ""
    for line in lines:
        starts = _NUM_START.match(line) or _APA_START.match(line) or _ORG_START.match(line)
        if starts and buf and _complete(buf):
            entries.append(buf)
            buf = line
        else:
            buf = f"{buf} {line}".strip()
    if buf:
        entries.append(buf)
    # fallback: semua entri tergabung dalam satu paragraf panjang
    if len(entries) <= 2 and len(ref_text) > 1500:
        joined = norm_ws(ref_text)
        parts = re.split(rf"(?<=[.)])\s+(?=[{UC}][\w'’\-]+,\s+[{UC}]\.)", joined)
        if len(parts) > len(entries):
            entries = parts
    return [norm_ws(e) for e in entries if len(norm_ws(e)) > 25]


def parse_reference(raw: str, idx: int) -> Reference:
    r = Reference(idx=idx, raw=raw)
    nm = _NUM_START.match(raw)
    r.num = int(nm.group(1) or nm.group(2)) if nm else idx
    s = _NUM_START.sub("", raw, count=1).strip()

    m = DOI_RE.search(raw)
    if m:
        r.doi = m.group(0).rstrip(".,;)]")
    u = URL_RE.search(raw)
    if u:
        r.url = u.group(0).rstrip(".,;)")

    ym = re.search(r"\(\s*((?:19|20)\d{2}[a-z]?)[^)]*\)|\((n\.d\.)\)", s)
    if ym:  # APA / Harvard
        r.year = ym.group(1) or "n.d."
        r.authors = s[: ym.start()].strip(" .,")
        rest = s[ym.end():].lstrip(" .")
        tm = re.match(r"(.+?[.?!])(?:\s|$)", rest)
        r.title = (tm.group(1) if tm else rest[:250]).rstrip(".").strip()
    else:   # IEEE / Vancouver / lainnya
        yy = re.findall(r"\b((?:19|20)\d{2})[a-z]?\b", DOI_RE.sub("", s))
        r.year = yy[-1] if yy else ""
        qm = re.search(r"[“\"](.+?)[,.]?[”\"]", s)
        if qm:
            r.title = qm.group(1).strip()
            r.authors = s[: qm.start()].strip(" ,")
        else:
            parts = s.split(". ")
            r.authors = parts[0]
            r.title = parts[1] if len(parts) > 1 else ""

    seg = r.authors.split(",")[0].strip()
    if re.match(rf"^(?:[{UC}]\.\s*-?)+\s*[{UC}]", seg):  # "A. B. Utami" (IEEE)
        seg = seg.split()[-1]
    r.first_author = seg.strip(" .&")
    return r


def parse_references(ref_text: str) -> list[Reference]:
    return [parse_reference(e, i + 1) for i, e in enumerate(split_references(ref_text))]
