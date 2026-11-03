"""Orkestrasi: proposal → sitasi → passage kandidat → verdict + evidence."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from .citations import CitationInstance, extract_instances, strip_citations
from .parsing import Reference, extract_text, parse_references, split_body_refs
from .retrieval import Retriever, build_passages
from .sources import SourceDoc
from . import nli, verify


def parse_proposal(data: bytes, filename: str):
    text = extract_text(data, filename)
    body, ref_text = split_body_refs(text)
    refs = parse_references(ref_text)
    instances, orphans, uncited = extract_instances(body, refs)
    return dict(refs=refs, instances=instances, orphans=orphans, uncited=uncited,
                n_chars=len(text), has_ref_section=bool(ref_text.strip()))


def retrieve_all(instances: list[CitationInstance], docs: dict[int, SourceDoc | None],
                 queries_en: list[str], top_k: int = 8, embedder=None, window: int = 1) -> dict[int, list]:
    retrievers = {i: Retriever(d, embedder) for i, d in docs.items() if d and d.sentences}
    out = {}
    for inst, q_en in zip(instances, queries_en):
        doc = docs.get(inst.ref_idx)
        if not doc or not doc.sentences:
            out[inst.cid] = []
            continue
        if doc.coverage == "abstrak saja" and len(doc.sentences) <= 30:
            idxs = list(range(len(doc.sentences)))  # abstrak pendek: berikan semuanya
            out[inst.cid] = build_passages(doc, idxs, window=0)
        else:
            idxs = retrievers[inst.ref_idx].search([strip_citations(inst.sentence), q_en], k=top_k)
            out[inst.cid] = build_passages(doc, idxs, window=window)
    return out


def run_verification(instances, refs: list[Reference], docs, passages_by_cid, llm=None,
                     workers: int = 4, on_progress=None, nli_scorer=None, thresholds=None) -> list[dict]:
    if nli_scorer is not None:
        workers = 1  # model NLI sudah memakai semua core CPU
    by_idx = {r.idx: r for r in refs}

    def one(inst: CitationInstance) -> dict:
        ref, doc = by_idx[inst.ref_idx], docs.get(inst.ref_idx)
        passages = passages_by_cid.get(inst.cid, [])
        claim = strip_citations(inst.sentence)
        base = dict(cid=inst.cid, ref_idx=ref.idx, ref_label=ref.label, ref_title=ref.title,
                    ref_raw=ref.raw, kalimat=inst.sentence, marker=inst.marker,
                    sumber=doc.origin if doc else "—", cakupan=doc.coverage if doc else "—",
                    url=doc.url if doc else "", kandidat=[p.as_dict() for p in passages])
        if not doc or not passages:
            base.update(verdict="TIDAK_DAPAT_DIVERIFIKASI", keyakinan=None, klaim_inti=claim, bukti=[],
                        penjelasan="Teks artikel sumber tidak tersedia. Unggah PDF artikel ini atau aktifkan pencarian daring.",
                        catatan=[])
        elif nli_scorer is not None:
            base.update(nli.judge(claim, passages, nli_scorer, doc.coverage, **(thresholds or {})))
        elif llm is None:
            base.update(verify.heuristic(claim, passages, doc.coverage))
        else:
            base.update(verify.verify_claim(llm, claim, inst.sentence, inst.context, inst.co_cited,
                                            ref.label, ref.title, doc.coverage, passages))
        return base

    results = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(one, inst) for inst in instances]
        for n, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            if on_progress:
                on_progress(n, len(futs))
    return sorted(results, key=lambda r: r["cid"])
