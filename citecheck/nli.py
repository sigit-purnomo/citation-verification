"""Penilaian tanpa LLM: Natural Language Inference (NLI) multibahasa.

Untuk setiap pasangan (kalimat artikel = premis, klaim proposal = hipotesis) model
memberi probabilitas entailment / neutral / contradiction. Evidence selalu berupa
kalimat asli artikel, jadi kutipannya dijamin verbatim.

Semua model diunduh gratis dari Hugging Face pada pemakaian pertama dan berjalan di CPU.
"""
from __future__ import annotations

import os
import re

import numpy as np

from .retrieval import Passage

NLI_MODELS = {
    "mDeBERTa-v3 base (akurat, ±1,1 GB)": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    "MiniLMv2 L6 (ringan, ±0,5 GB)": "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli",
}
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _quantize(model):
    """Kuantisasi dinamis int8 untuk layer Linear: hemat RAM & lebih cepat di CPU."""
    try:
        import torch
        q = getattr(torch.ao.quantization, "quantize_dynamic", None) or torch.quantization.quantize_dynamic
        return q(model, {torch.nn.Linear}, dtype=torch.qint8)
    except Exception:  # noqa: BLE001 - tetap jalan tanpa kuantisasi
        return model


def load_embedder(quantize: bool = True):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMBED_MODEL, device="cpu")
    return _quantize(m) if quantize else m


class NLIScorer:
    def __init__(self, model_name: str, quantize: bool = True, max_length: int = 256):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        labels = {i: str(l).lower() for i, l in model.config.id2label.items()}
        self.ix = [next(i for i, l in labels.items() if l.startswith(k)) for k in ("entail", "neutral", "contra")]
        self.model = _quantize(model) if quantize else model
        self.max_length = max_length

    def score(self, pairs: list[tuple[str, str]], batch_size: int = 16) -> np.ndarray:
        """Kembalikan array (n, 3): [entailment, neutral, contradiction]."""
        out = []
        with self.torch.inference_mode():
            for s in range(0, len(pairs), batch_size):
                chunk = pairs[s:s + batch_size]
                enc = self.tok([p for p, _ in chunk], [h for _, h in chunk], truncation="only_first",
                               padding=True, max_length=self.max_length, return_tensors="pt")
                probs = self.torch.softmax(self.model(**enc).logits, dim=-1).numpy()
                out.append(probs[:, self.ix])
        return np.vstack(out) if out else np.zeros((0, 3))


# ---------------------------------------------------------------- klaim → hipotesis
_REPORTING = re.compile(r"^(?:menurut|according to|berdasarkan(?: penelitian| temuan| studi| hasil)?)\s+[^,]{1,80},\s*", re.I)
_FINDING = re.compile(r"^[^,]{0,90}?\b(?:menemukan|menunjukkan|menyatakan|mengungkapkan|membuktikan|melaporkan|"
                      r"menegaskan|found|showed|reported|demonstrated|revealed|argued)\s+(?:bahwa|that)\s+", re.I)


def to_hypothesis(claim: str) -> str:
    """Buang frasa pelaporan ('Menurut X,', 'X menemukan bahwa') agar yang dinilai isi klaimnya."""
    h = _REPORTING.sub("", claim.strip())
    h = _FINDING.sub("", h)
    h = h[:1].upper() + h[1:] if h else claim
    return h.rstrip(" .") + "."


def judge(claim: str, passages: list[Passage], scorer, coverage: str,
          t_support: float = 0.70, t_partial: float = 0.40, t_contra: float = 0.70) -> dict:
    hyp = to_hypothesis(claim)
    if not passages:
        return dict(verdict="TIDAK_DAPAT_DIVERIFIKASI", keyakinan=None, klaim_inti=hyp, bukti=[], catatan=[],
                    penjelasan="Tidak ada passage kandidat.")
    probs = scorer.score([(p.text, hyp) for p in passages])
    ent, con = probs[:, 0], probs[:, 2]
    prim = np.array([not p.secondary for p in passages])
    best_e = float(ent.max())
    best_c_prim = float(con[prim].max()) if prim.any() else 0.0
    best_e_prim = float(ent[prim].max()) if prim.any() else 0.0

    mixed = best_c_prim >= t_contra and best_e >= t_partial
    if best_c_prim >= t_contra and best_c_prim > best_e:
        verdict, conf = "BERTENTANGAN", best_c_prim
    elif mixed:
        verdict, conf = "DIDUKUNG_SEBAGIAN", min(best_e, best_c_prim)
    elif best_e >= t_support:
        verdict, conf = "DIDUKUNG", best_e
    elif best_e >= t_partial:
        verdict, conf = "DIDUKUNG_SEBAGIAN", best_e
    else:
        verdict = "TIDAK_DAPAT_DIVERIFIKASI" if coverage == "abstrak saja" else "TIDAK_DIDUKUNG"
        conf = 1 - best_e

    bukti, used = [], set()
    for i in np.argsort(-ent)[:3]:
        if ent[i] >= 0.30:
            p = passages[i]
            bukti.append(dict(pid=p.pid, section=p.section, kutipan=p.text, status="persis", skor=100,
                              peran="latar" if p.secondary else "mendukung", nli=float(ent[i])))
            used.add(int(i))
    for i in np.argsort(-con)[:2]:
        if con[i] >= 0.50 and int(i) not in used:
            p = passages[i]
            bukti.append(dict(pid=p.pid, section=p.section, kutipan=p.text, status="persis", skor=100,
                              peran="bertentangan", nli=float(con[i])))
    if not bukti:  # tampilkan passage terdekat agar bisa ditinjau manual
        for p in passages[:2]:
            bukti.append(dict(pid=p.pid, section=p.section, kutipan=p.text, status="persis", skor=100,
                              peran="kandidat", nli=None))

    catatan = []
    if mixed and verdict != "BERTENTANGAN":
        catatan.append("Artikel memuat kalimat yang mendukung sekaligus temuan yang bertentangan dengan klaim — "
                       "kemungkinan klaim terlalu luas atau mengabaikan hasil tidak signifikan.")
    if verdict in ("DIDUKUNG", "DIDUKUNG_SEBAGIAN") and best_e_prim < t_partial:
        catatan.append("Dukungan hanya berasal dari kalimat latar belakang yang mengutip studi lain "
                       "(indikasi sitasi sekunder) — pertimbangkan mensitasi sumber primernya.")
    if len(hyp.split()) > 30:
        catatan.append("Kalimat proposal panjang/majemuk; NLI cenderung menilainya netral. "
                       "Periksa evidence secara manual atau pecah klaimnya.")
    if coverage == "abstrak saja":
        catatan.append("Hanya abstrak yang tersedia; unggah PDF artikel untuk penilaian lebih lengkap.")

    top = passages[int(np.argmax(ent))]
    penjelasan = (f"Skor entailment tertinggi {best_e:.2f} pada {top.section} ({top.pid}); "
                  f"skor kontradiksi tertinggi pada temuan artikel {best_c_prim:.2f}. "
                  f"Dinilai dari {len(passages)} kalimat kandidat ({coverage}). "
                  "Verdict otomatis dari model NLI — baca evidence sebelum menyimpulkan.")
    return dict(verdict=verdict, keyakinan=round(conf, 3), klaim_inti=hyp, penjelasan=penjelasan,
                bukti=bukti, catatan=catatan)
