"""Menyiapkan teks artikel sumber: PDF unggahan atau pengambilan daring
(OpenAlex → PDF open access, Crossref, Semantic Scholar), lalu memecahnya
menjadi kalimat berlabel bagian (ABSTRACT, INTRODUCTION, RESULTS, ...)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import requests
from rapidfuzz import fuzz

from .citations import split_sentences
from urllib.parse import urlparse

from . import landing
from .parsing import DOI_RE, Reference, extract_text, fold, norm_ws

UA = "CiteCheck/1.0 (citation verification tool)"

# urutan penting: kunci yang lebih spesifik di depan
SECTION_KEYS = [
    ("articleinfo", "SKIP"), ("keywords", "SKIP"), ("katakunci", "SKIP"),
    ("abstract", "ABSTRACT"), ("abstrak", "ABSTRACT"), ("highlights", "ABSTRACT"),
    ("introduction", "INTRODUCTION"), ("pendahuluan", "INTRODUCTION"),
    ("background", "INTRODUCTION"), ("latarbelakang", "INTRODUCTION"),
    ("theoreticalimplication", "IMPLICATIONS"), ("theoreticalcontribution", "IMPLICATIONS"),
    ("managerialimplication", "IMPLICATIONS"), ("practicalimplication", "IMPLICATIONS"),
    ("implication", "IMPLICATIONS"), ("implikasi", "IMPLICATIONS"),
    ("literaturereview", "LITERATURE REVIEW"), ("theoretical", "LITERATURE REVIEW"),
    ("theory", "LITERATURE REVIEW"), ("conceptual", "LITERATURE REVIEW"),
    ("relatedwork", "LITERATURE REVIEW"), ("tinjauanpustaka", "LITERATURE REVIEW"),
    ("kajianpustaka", "LITERATURE REVIEW"), ("landasanteori", "LITERATURE REVIEW"),
    ("hypothes", "HYPOTHESES"), ("researchmodel", "HYPOTHESES"),
    ("materialsandmethods", "METHODS"), ("method", "METHODS"), ("researchdesign", "METHODS"),
    ("datacollection", "METHODS"), ("metode", "METHODS"), ("metodologi", "METHODS"),
    ("resultsanddiscussion", "RESULTS & DISCUSSION"), ("hasildanpembahasan", "RESULTS & DISCUSSION"),
    ("result", "RESULTS"), ("finding", "RESULTS"), ("dataanalysis", "RESULTS"), ("hasil", "RESULTS"),
    ("discussion", "DISCUSSION"), ("generaldiscussion", "DISCUSSION"), ("pembahasan", "DISCUSSION"),
    ("conclusion", "CONCLUSION"), ("concludingremarks", "CONCLUSION"),
    ("kesimpulan", "CONCLUSION"), ("simpulan", "CONCLUSION"), ("penutup", "CONCLUSION"),
    ("limitation", "LIMITATIONS"), ("futureresearch", "LIMITATIONS"), ("keterbatasan", "LIMITATIONS"),
    ("acknowledg", "SKIP"), ("ucapanterimakasih", "SKIP"), ("funding", "SKIP"),
    ("declarationofcompeting", "SKIP"), ("conflictofinterest", "SKIP"), ("dataavailability", "SKIP"),
    ("creditauthorship", "SKIP"),
]
STOP_KEYS = ["references", "referencelist", "bibliography", "daftarpustaka",
             "literaturecited", "workscited", "appendix", "lampiran"]
PRIMARY_SECTIONS = {"ABSTRACT", "RESULTS", "RESULTS & DISCUSSION", "DISCUSSION", "CONCLUSION", "IMPLICATIONS"}

INNER_CITE = re.compile(
    r"\[\d{1,3}(?:\s*[-–,]\s*\d{1,3})*\]"
    r"|\([^()]*[A-Z][^()]*(?:19|20)\d{2}[a-z]?[^()]*\)"
    r"|\bet al\.,?\s*\(?(?:19|20)\d{2}"
)
_JUNK = re.compile(r"©|all rights reserved|creative ?commons|https?://|doi\.org|e-mail|"
                   r"corresponding author|received \d|accepted \d|available online|licen[cs]e", re.I)


@dataclass
class SourceDoc:
    ref_idx: int
    origin: str               # asal teks, mis. "PDF unggahan: x.pdf"
    coverage: str             # "teks lengkap" | "abstrak saja"
    title: str = ""
    url: str = ""
    sentences: list[str] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)
    secondary: list[bool] = field(default_factory=list)  # kalimat yang mengutip penelitian lain


def _heading(line: str) -> str | None:
    raw = line.strip()
    if not raw or len(raw) > 80 or raw.endswith((".", ",", ";")) and len(raw) > 25:
        return None
    s = re.sub(r"^\s*(?:\d+(?:\.\d+)*\.?|[IVX]+\.)\s*", "", raw)
    compact = re.sub(r"[^a-z]", "", fold(s))
    if not compact or len(compact) > 45:
        return None
    if any(compact.startswith(k) for k in STOP_KEYS):
        return "STOP"
    for key, label in SECTION_KEYS:
        if compact.startswith(key):
            return label
    return None


def _junk(s: str) -> bool:
    if len(s) < 30:
        return True
    letters = sum(c.isalpha() for c in s)
    return letters / max(len(s), 1) < 0.55 or bool(_JUNK.search(s))


def _add(doc: SourceDoc, label: str, text: str):
    for s in split_sentences(text):
        if _junk(s):
            continue
        doc.sentences.append(s)
        doc.sections.append(label)
        doc.secondary.append(label not in {"ABSTRACT"} and bool(INNER_CITE.search(s)))


def build_from_fulltext(text: str, ref_idx: int, origin: str, title: str = "", url: str = "") -> SourceDoc:
    doc = SourceDoc(ref_idx, origin, "teks lengkap", title, url)
    label, buf = "BAGIAN AWAL", []
    for line in text.splitlines():
        h = _heading(line)
        if h:
            _add(doc, label, " ".join(buf)) if label != "SKIP" else None
            buf = []
            if h == "STOP" and len(doc.sentences) > 20:
                break  # daftar pustaka artikel tidak dipakai sebagai evidence
            label = "SKIP" if h == "STOP" else h
            continue
        buf.append(line)
    else:
        if label != "SKIP":
            _add(doc, label, " ".join(buf))
    return doc


def build_from_abstract(abstract: str, ref_idx: int, origin: str, title: str = "", url: str = "") -> SourceDoc:
    doc = SourceDoc(ref_idx, origin, "abstrak saja", title, url)
    _add(doc, "ABSTRACT", abstract)
    return doc


# ---------------------------------------------------------------- PDF unggahan
def match_uploaded(pdf_texts: dict[str, str], refs: list[Reference]) -> dict[str, int | None]:
    """Petakan setiap PDF unggahan ke entri pustaka via DOI atau kemiripan judul."""
    out = {}
    for name, text in pdf_texts.items():
        head = text[:6000]
        best, best_score = None, 0
        dois = {d.rstrip(".,;)").lower() for d in DOI_RE.findall(head)}
        for r in refs:
            if r.doi and r.doi.lower() in dois:
                best, best_score = r.idx, 101
                break
            if len(r.title) > 20:
                sc = fuzz.partial_ratio(fold(r.title), fold(norm_ws(head)))
                if sc > best_score:
                    best, best_score = r.idx, sc
        out[name] = best if best_score >= 88 else None
    return out


# ---------------------------------------------------------------- daring
def _get(url, params=None, timeout=20):
    try:
        r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": UA})
        if r.status_code == 200:
            return r
    except requests.RequestException:
        pass
    return None


def _title_ok(ref: Reference, found: str) -> bool:
    if not ref.title or not found:
        return bool(ref.doi)
    return fuzz.token_set_ratio(fold(ref.title), fold(found)) >= 80


def _openalex(ref: Reference, mailto: str):
    params = {"mailto": mailto} if mailto else {}
    if ref.doi:
        r = _get(f"https://api.openalex.org/works/https://doi.org/{ref.doi}", params)
        if r:
            return r.json()
    if ref.title:
        r = _get("https://api.openalex.org/works", {**params, "search": ref.title[:250], "per_page": 5})
        if r:
            for w in r.json().get("results", []):
                if _title_ok(ref, w.get("title") or ""):
                    return w
    return None


def _inverted_to_text(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return " ".join(w for _, w in pos)


def _crossref_abstract(doi: str) -> str:
    r = _get(f"https://api.crossref.org/works/{doi}")
    if not r:
        return ""
    ab = r.json().get("message", {}).get("abstract", "")
    return norm_ws(re.sub(r"<[^>]+>", " ", ab)).removeprefix("Abstract ").strip()


def _s2(ref: Reference) -> tuple[str, str]:
    fields = "title,abstract,openAccessPdf"
    if ref.doi:
        r = _get(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{ref.doi}", {"fields": fields})
        papers = [r.json()] if r else []
    else:
        r = _get("https://api.semanticscholar.org/graph/v1/paper/search",
                 {"query": ref.title[:250], "limit": 3, "fields": fields})
        papers = r.json().get("data", []) if r else []
    for p in papers:
        if _title_ok(ref, p.get("title") or ""):
            pdf = (p.get("openAccessPdf") or {}).get("url") or ""
            return p.get("abstract") or "", pdf
    return "", ""


def fetch_online(ref: Reference, mailto: str = "", try_pdf: bool = True,
                 use_landing: bool = True) -> tuple[SourceDoc | None, list[str]]:
    """Cari teks artikel secara daring. Mengembalikan (dokumen, log langkah).

    Lapis 1: OpenAlex/Crossref/Semantic Scholar (metadata terbuka) → PDF open access.
    Lapis 2: halaman artikel di situs penerbit (via DOI/URL) → abstrak & PDF terbuka.
    Teks lengkap diutamakan; bila tidak ada, dipakai abstrak terbaik yang ditemukan.
    """
    log: list[str] = []
    work = _openalex(ref, mailto)
    log.append("OpenAlex: ditemukan" if work else "OpenAlex: tidak ditemukan")
    title = (work or {}).get("title") or ref.title
    doi = ref.doi or ((work or {}).get("doi") or "").replace("https://doi.org/", "")
    landing_url = f"https://doi.org/{doi}" if doi else ref.url
    abstracts: list[tuple[str, str]] = []  # (sumber, teks)
    ab = _inverted_to_text((work or {}).get("abstract_inverted_index"))
    if ab:
        abstracts.append(("abstrak OpenAlex", ab))

    pdf_urls = []
    if work:
        for loc in [work.get("best_oa_location")] + (work.get("locations") or []):
            if loc and loc.get("pdf_url") and loc["pdf_url"] not in pdf_urls:
                pdf_urls.append(loc["pdf_url"])
    if not abstracts and doi:
        ab = _crossref_abstract(doi)
        if ab:
            abstracts.append(("abstrak Crossref", ab))
    if not abstracts or not pdf_urls:
        ab2, pdf2 = _s2(ref)
        if ab2:
            abstracts.append(("abstrak Semantic Scholar", ab2))
        if pdf2:
            pdf_urls.append(pdf2)

    def try_pdfs(urls, origin) -> SourceDoc | None:
        for u in urls[:3]:
            data, status = landing.download_pdf(u)
            log.append(f"PDF {urlparse(u).netloc}: {status}")
            if data:
                doc = build_from_fulltext(extract_text(data, "a.pdf"), ref.idx, origin, title, u)
                if len(doc.sentences) > 40:
                    return doc
                log.append("PDF terbaca tetapi teksnya terlalu sedikit (mungkin hasil pindai)")
        return None

    if try_pdf and pdf_urls:
        doc = try_pdfs(pdf_urls, "PDF open access (OpenAlex/Semantic Scholar)")
        if doc:
            return doc, log

    # ---- lapis 2: halaman penerbit
    if use_landing and (landing_url or ref.url):
        info, status = landing.read_landing(doi, "" if doi else ref.url)
        log.append(f"Halaman penerbit: {status}")
        if info and (not ref.title or not info["title"] or _title_ok(ref, info["title"])):
            host = urlparse(info["url"]).netloc
            new_pdfs = [u for u in info["pdf_urls"] if u not in pdf_urls]
            if try_pdf and new_pdfs:
                doc = try_pdfs(new_pdfs, f"PDF dari halaman penerbit ({host})")
                if doc:
                    return doc, log
            if info["abstract"]:
                abstracts.append((f"abstrak halaman penerbit ({host})", info["abstract"]))
            landing_url = info["url"]
        elif info:
            log.append("Halaman penerbit: judul tidak cocok, diabaikan")

    if abstracts:
        src, text = max(abstracts, key=lambda x: len(x[1]))  # abstrak terlengkap
        return build_from_abstract(text, ref.idx, src, title, landing_url or ""), log
    return None, log
