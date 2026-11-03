"""Penemuan passage kandidat: BM25 + (opsional) embedding multibahasa, digabung RRF."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .parsing import fold
from .sources import PRIMARY_SECTIONS, SourceDoc

STOP = set("""the a an of and or to in on for with by is are was were be been this that these those
as at from it its their our we they which who also than then thus however into such can may not
yang dan di ke dari pada untuk dengan dalam ini itu adalah sebagai oleh atau juga akan telah tidak
lebih serta bahwa karena para merupakan menjadi dapat secara""".split())


def tok(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", fold(s)) if w not in STOP and len(w) > 1]


@dataclass
class Passage:
    pid: str
    section: str
    text: str
    secondary: bool
    center: int

    def as_dict(self):
        return dict(pid=self.pid, section=self.section, text=self.text, secondary=self.secondary)


class Retriever:
    def __init__(self, doc: SourceDoc, embedder=None):
        self.doc = doc
        self.bm25 = BM25Okapi([tok(s) or ["_"] for s in doc.sentences])
        self.embedder = embedder
        self.emb = (embedder.encode(doc.sentences, normalize_embeddings=True, batch_size=64)
                    if embedder is not None and doc.sentences else None)

    def search(self, queries: list[str], k: int = 8) -> list[int]:
        n = len(self.doc.sentences)
        if n == 0:
            return []
        fused = np.zeros(n)
        for q in [q for q in queries if q and q.strip()]:
            rankings = [np.argsort(-self.bm25.get_scores(tok(q)))]
            if self.emb is not None:
                qv = self.embedder.encode([q], normalize_embeddings=True)[0]
                rankings.append(np.argsort(-(self.emb @ qv)))
            for order in rankings:
                for rank, i in enumerate(order[:60]):
                    fused[i] += 1.0 / (60 + rank)
        for i in range(n):  # prior: temuan artikel > latar belakang yang mengutip orang lain
            if self.doc.sections[i] in PRIMARY_SECTIONS:
                fused[i] *= 1.15
            if self.doc.secondary[i]:
                fused[i] *= 0.9
        order = [int(i) for i in np.argsort(-fused) if fused[i] > 0]
        return order[:k]


def build_passages(doc: SourceDoc, idxs, window: int = 1) -> list[Passage]:
    """Perluas tiap kalimat hit dengan ±window kalimat di bagian yang sama."""
    out, covered = [], set()
    for i in idxs:
        if i in covered:
            continue
        sec = doc.sections[i]
        lo, hi = i, i
        while lo - 1 >= 0 and i - lo < window and doc.sections[lo - 1] == sec and lo - 1 not in covered:
            lo -= 1
        while hi + 1 < len(doc.sentences) and hi - i < window and doc.sections[hi + 1] == sec and hi + 1 not in covered:
            hi += 1
        covered.update(range(lo, hi + 1))
        out.append(Passage(f"#{len(out) + 1}", sec, " ".join(doc.sentences[lo:hi + 1]),
                           any(doc.secondary[lo:hi + 1]), i))
    return out


def lexical_overlap(claim: str, text: str) -> float:
    q, t = set(tok(claim)), set(tok(text))
    return len(q & t) / len(q) if q else 0.0
