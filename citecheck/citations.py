"""Deteksi sitasi dalam teks (APA/Harvard/numerik) dan pemetaannya ke daftar pustaka."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from .parsing import UC, Reference, fold, norm_ws

NAME = rf"[{UC}][\w'’\-]+"
YR = r"(?:19|20)\d{2}[a-z]?"

NARR_RE = re.compile(
    rf"\b(?P<auth>{NAME}(?:\s+(?:et\s+al\.?|dkk\.?|(?:&|dan|and)\s+{NAME}))?)\s*"
    rf"\((?P<years>{YR}(?:\s*[,;]\s*{YR})*)(?:\s*[,:][^)]*)?\)"
)
PAREN_RE = re.compile(rf"\(([^()]{{0,400}}?\b{YR}\b[^()]{{0,120}}?)\)")
NUM_RE = re.compile(r"\[(\d{1,3}(?:\s*[-–,]\s*\d{1,3})*)\]")
PREFIX_RE = re.compile(r"^(?:e\.g\.,?|misalnya,?|mis\.,?|lihat juga|lihat|see also|see|cf\.|dalam|in)\s+", re.I)
YEAR_ONLY = re.compile(rf"\b{YR}\b")
_NOT_NAMES = {"pada", "tahun", "sejak", "hingga", "sampai", "in", "since", "until", "during",
              "year", "sebelum", "after", "before", "from", "dari", "per", "bulan", "januari",
              "februari", "maret", "april", "mei", "juni", "juli", "agustus", "september",
              "oktober", "november", "desember", "the", "this", "dalam"}


@dataclass
class CitationInstance:
    cid: int
    sent_idx: int
    sentence: str
    context: str            # kalimat sebelumnya, untuk konteks
    marker: str             # penanda sitasi persis seperti di teks
    surname: str
    year: str
    ref_idx: int
    co_cited: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- kalimat
def split_sentences(text: str) -> list[str]:
    t = norm_ws(text)
    t = re.sub(r"\b(et al|dkk|e\.g|i\.e|hlm|vol|no|pp|cf|vs|dll|dsb|dr|prof|fig|eds?|jr)\.",
               lambda m: m.group(0)[:-1] + "<DOT>", t, flags=re.I)
    t = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", t)
    t = re.sub(rf"\b([{UC}])\.", r"\1<DOT>", t)
    parts = re.split(rf"(?<=[.!?])\s+(?=[{UC}0-9“\"(\[])", t)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


# ---------------------------------------------------------------- sitasi
def find_citations(sentence: str) -> list[dict]:
    out, taken = [], []
    for m in NARR_RE.finditer(sentence):
        first = re.match(NAME, m.group("auth")).group(0)
        if first.lower() in _NOT_NAMES:
            continue
        for y in re.findall(YR, m.group("years")):
            out.append(dict(kind="narr", marker=m.group(0), surname=first,
                            author_text=m.group("auth"), year=y, num=None))
        taken.append(m.span())

    for m in PAREN_RE.finditer(sentence):
        if any(a <= m.start() < b for a, b in taken):
            continue
        for item in m.group(1).split(";"):
            item = PREFIX_RE.sub("", item.strip())
            ys = list(YEAR_ONLY.finditer(item))
            if not ys:
                continue
            auth = item[: ys[0].start()].strip(" ,")
            nm = re.search(NAME, auth)
            if not nm or nm.group(0).lower() in _NOT_NAMES:
                continue
            for y in ys:
                out.append(dict(kind="paren", marker=m.group(0), surname=nm.group(0),
                                author_text=auth, year=y.group(0), num=None))

    for m in NUM_RE.finditer(sentence):
        nums: set[int] = set()
        for part in re.split(r"\s*,\s*", m.group(1)):
            if re.search(r"[-–]", part):
                a, b = (int(x) for x in re.split(r"\s*[-–]\s*", part))
                if 0 <= b - a < 30:
                    nums.update(range(a, b + 1))
            elif part.strip():
                nums.add(int(part))
        for n in sorted(nums):
            out.append(dict(kind="num", marker=m.group(0), surname="", author_text="", year="", num=n))
    return out


def strip_citations(sentence: str) -> str:
    """Kalimat klaim tanpa penanda sitasi (nama penulis naratif dipertahankan)."""
    s = NARR_RE.sub(lambda m: m.group("auth"), sentence)
    s = PAREN_RE.sub(lambda m: "" if re.search(NAME, m.group(1)) else m.group(0), s)
    s = NUM_RE.sub("", s)
    return norm_ws(re.sub(r"\s+([.,;:])", r"\1", s))


def match_reference(c: dict, refs: list[Reference]) -> Reference | None:
    fs, fa, y = fold(c["surname"]), fold(c["author_text"]), c["year"]

    def name_ok(r: Reference, fuzzy=False) -> bool:
        fr = fold(r.first_author)
        if not fr:
            return False
        if fr == fs or fa.startswith(fr) or fr.startswith(fs) or fr.split()[-1] == fs:
            return True
        return fuzzy and fuzz.ratio(fr, fs) >= 85

    for fuzzy in (False, True):
        cands = [r for r in refs if r.year[:4] == y[:4] and name_ok(r, fuzzy)]
        if cands:
            if len(cands) > 1 and len(y) == 5:
                exact = [r for r in cands if r.year == y]
                cands = exact or cands
            return cands[0]
    return None


def extract_instances(body: str, refs: list[Reference]):
    """Kembalikan (instances, orphans, uncited_refs)."""
    numeric_style = sum(1 for r in refs if re.match(r"^\s*(\[\d+\]|\d{1,3}\.\s)", r.raw)) >= max(2, len(refs) // 2)
    by_num = {r.num: r for r in refs}
    sents = split_sentences(body)
    instances, orphans, used = [], [], set()

    for i, s in enumerate(sents):
        cites = find_citations(s)
        if not cites:
            continue
        seen, found = set(), []
        for c in cites:
            if c["kind"] == "num":
                if not numeric_style:
                    continue
                ref = by_num.get(c["num"])
            else:
                ref = match_reference(c, refs)
                if ref is None and c["kind"] == "narr":
                    continue  # kemungkinan bukan sitasi
            key = ref.idx if ref else ("orphan", c["surname"], c["year"], c["num"])
            if key in seen:
                continue
            seen.add(key)
            found.append((c, ref))

        labels = [r.label for _, r in found if r]
        for c, ref in found:
            if ref is None:
                tag = f"[{c['num']}]" if c["kind"] == "num" else f"{c['author_text']} ({c['year']})"
                orphans.append(dict(sitasi=tag, kalimat=s))
                continue
            used.add(ref.idx)
            instances.append(CitationInstance(
                cid=len(instances) + 1, sent_idx=i, sentence=s,
                context=sents[i - 1] if i > 0 else "", marker=c["marker"],
                surname=c["surname"] or ref.first_author, year=c["year"] or ref.year,
                ref_idx=ref.idx, co_cited=[lb for lb in labels if lb != ref.label]))
    uncited = [r for r in refs if r.idx not in used]
    return instances, orphans, uncited
