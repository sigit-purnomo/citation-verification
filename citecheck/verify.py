"""Penilaian klaim vs evidence dengan LLM, plus pemeriksaan bahwa kutipan benar-benar
ada di artikel (verbatim grounding) — langkah yang membuat evidence dapat dipercaya."""
from __future__ import annotations

import json
import re
import time

from rapidfuzz import fuzz

from .parsing import fold, norm_ws
from .retrieval import Passage, lexical_overlap

VERDICTS = ["DIDUKUNG", "DIDUKUNG_SEBAGIAN", "TIDAK_DIDUKUNG", "BERTENTANGAN",
            "TIDAK_DAPAT_DIVERIFIKASI", "PERLU_TINJAUAN_MANUAL", "GALAT"]
ROLES = {"mendukung", "membatasi", "bertentangan", "latar", "kandidat"}


class LLM:
    def __init__(self, provider: str, api_key: str, model: str, base_url: str | None = None):
        self.provider, self.model = provider, model
        if provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        else:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key or "none", base_url=base_url or None)

    def complete(self, system: str, user: str, max_tokens: int = 1500) -> str:
        last = None
        for attempt in range(3):
            try:
                if self.provider == "anthropic":
                    r = self.client.messages.create(
                        model=self.model, max_tokens=max_tokens, system=system,
                        messages=[{"role": "user", "content": user}])
                    return "".join(getattr(b, "text", "") for b in r.content)
                r = self.client.chat.completions.create(
                    model=self.model, max_tokens=max_tokens, temperature=0,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
                return r.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 - jaringan/rate limit
                last = e
                time.sleep(2 * (attempt + 1))
        raise last


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M)
    a, b = text.find("{"), text.rfind("}")
    if a < 0 or b < 0:
        raise ValueError("Respons LLM bukan JSON")
    return json.loads(text[a:b + 1])


# ---------------------------------------------------------------- query lintas bahasa
QUERY_SYSTEM = (
    "You turn academic claims (often written in Indonesian) into short English search queries "
    "for finding the supporting passage inside the cited English-language paper. Keep key constructs, "
    "variables, population and direction of effect. Output JSON only: "
    '{"queries":[{"i":0,"q":"..."}]}'
)


def make_queries(llm: LLM, claims: list[str], batch: int = 25) -> list[str]:
    out = [""] * len(claims)
    for start in range(0, len(claims), batch):
        chunk = claims[start:start + batch]
        payload = json.dumps([{"i": start + j, "claim": c} for j, c in enumerate(chunk)], ensure_ascii=False)
        try:
            data = parse_json(llm.complete(QUERY_SYSTEM, payload, max_tokens=3000))
            for q in data.get("queries", []):
                if 0 <= int(q["i"]) < len(out):
                    out[int(q["i"])] = q.get("q", "")
        except Exception:  # noqa: BLE001 - query terjemahan hanya pelengkap
            pass
    return out


# ---------------------------------------------------------------- verifikasi
VERIFY_SYSTEM = """Anda adalah auditor sitasi akademik. Nilai apakah KLAIM dalam proposal secara wajar didukung oleh ARTIKEL yang disitasi, HANYA berdasarkan potongan teks artikel yang diberikan.

Aturan:
1. Kutipan bukti WAJIB disalin persis (verbatim) dari satu potongan, dalam bahasa aslinya. Jangan menerjemahkan, meringkas, mengubah kata, atau menggabungkan potongan berbeda. Maksimal 2 kalimat per kutipan, maksimal 4 kutipan.
2. Utamakan bukti dari temuan artikel itu sendiri (ABSTRACT, RESULTS, DISCUSSION, CONCLUSION). Potongan bertanda [SEKUNDER] adalah pernyataan latar belakang yang mengutip penelitian lain: beri peran "latar", dan bila hanya itu yang mendukung klaim, tulis di catatan bahwa ada indikasi sitasi sekunder (sebaiknya sitasi sumber primernya).
3. Sertakan juga bukti yang membatasi atau bertentangan (efek tidak signifikan, hanya berlaku pada kelompok tertentu, dimediasi variabel lain, dsb.) bila ada.
4. Waspadai: overgeneralisasi, klaim kausal dari desain survei/korelasional, beda populasi/konteks/negara, angka yang tidak cocok, arah hubungan terbalik.
5. Kalimat proposal dapat mensitasi beberapa sumber; nilai hanya bagian klaim yang wajar diatribusikan ke artikel ini.
6. Bila teks yang tersedia hanya abstrak dan tidak cukup untuk menilai, gunakan TIDAK_DAPAT_DIVERIFIKASI, bukan TIDAK_DIDUKUNG.

Verdict:
- DIDUKUNG: inti klaim dinyatakan jelas oleh artikel.
- DIDUKUNG_SEBAGIAN: sebagian klaim didukung, atau artikel lebih terbatas/bernuansa daripada klaim.
- TIDAK_DIDUKUNG: artikel relevan tetapi tidak menyatakan klaim tersebut.
- BERTENTANGAN: artikel menyatakan hal yang berlawanan.
- TIDAK_DAPAT_DIVERIFIKASI: teks yang tersedia tidak cukup.

Keluarkan HANYA JSON:
{"klaim_inti":"klaim yang diatribusikan ke artikel ini, 1 kalimat","verdict":"...","keyakinan":0.0,"penjelasan":"2-4 kalimat bahasa Indonesia","bukti":[{"id":"#1","kutipan":"...","peran":"mendukung|membatasi|bertentangan|latar"}],"catatan":["..."]}"""


def _n(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'").replace("–", "-")
    return norm_ws(fold(s))


def locate_quote(quote: str, passages: list[Passage], pid: str | None = None):
    """Cari kutipan di passage. Status: 'persis', 'mirip', atau 'tidak ditemukan'."""
    nq = _n(quote).strip(' ."')
    if not nq or not passages:
        return None, "tidak ditemukan", 0
    ordered = sorted(passages, key=lambda p: p.pid != pid)
    for p in ordered:
        if nq in _n(p.text):
            return p, "persis", 100
    best = max(ordered, key=lambda p: fuzz.partial_ratio(nq, _n(p.text)))
    score = fuzz.partial_ratio(nq, _n(best.text))
    return (best, "mirip", score) if score >= 90 else (None, "tidak ditemukan", score)


def _format_passages(passages: list[Passage]) -> str:
    return "\n\n".join(f"[{p.pid}] {p.section}{' [SEKUNDER]' if p.secondary else ''}\n{p.text}" for p in passages)


def verify_claim(llm: LLM, claim: str, sentence: str, context: str, co_cited: list[str],
                 ref_label: str, ref_title: str, coverage: str, passages: list[Passage]) -> dict:
    user = (
        f"ARTIKEL YANG DISITASI: {ref_label} — {ref_title}\n"
        f"Cakupan teks yang tersedia: {coverage}\n\n"
        f"KONTEKS (kalimat sebelumnya): {context or '-'}\n"
        f"KALIMAT PROPOSAL: {sentence}\n"
        f"KLAIM (tanpa penanda sitasi): {claim}\n"
        f"Sumber lain yang disitasi pada kalimat yang sama: {', '.join(co_cited) or '-'}\n\n"
        f"POTONGAN ARTIKEL:\n{_format_passages(passages)}"
    )
    try:
        data = parse_json(llm.complete(VERIFY_SYSTEM, user))
    except Exception as e:  # noqa: BLE001
        return dict(verdict="GALAT", keyakinan=None, klaim_inti="", bukti=[], catatan=[],
                    penjelasan=f"Gagal memanggil/mengurai respons LLM: {e}")

    verdict = str(data.get("verdict", "")).upper().replace(" ", "_")
    if verdict not in VERDICTS:
        verdict = "PERLU_TINJAUAN_MANUAL"
    bukti = []
    for b in data.get("bukti", [])[:4]:
        q = str(b.get("kutipan", "")).strip()
        p, status, score = locate_quote(q, passages, b.get("id"))
        role = str(b.get("peran", "mendukung")).lower()
        bukti.append(dict(pid=p.pid if p else b.get("id", "?"), section=p.section if p else "?",
                          kutipan=q, peran=role if role in ROLES else "mendukung",
                          status=status, skor=score))
    catatan = [str(c) for c in data.get("catatan", []) if str(c).strip()]
    if any(b["status"] == "tidak ditemukan" for b in bukti):
        catatan.append("Ada kutipan yang tidak ditemukan di teks artikel (kemungkinan parafrase/halusinasi LLM) — abaikan kutipan bertanda ✗.")
    try:
        conf = float(data.get("keyakinan"))
    except (TypeError, ValueError):
        conf = None
    return dict(verdict=verdict, keyakinan=conf, klaim_inti=data.get("klaim_inti", ""),
                penjelasan=data.get("penjelasan", ""), bukti=bukti, catatan=catatan)


def heuristic(claim: str, passages: list[Passage], coverage: str) -> dict:
    """Mode tanpa LLM: tampilkan kandidat evidence dan skor tumpang-tindih leksikal."""
    scored = sorted(passages, key=lambda p: -lexical_overlap(claim, p.text))[:3]
    best = lexical_overlap(claim, scored[0].text) if scored else 0.0
    return dict(
        verdict="PERLU_TINJAUAN_MANUAL", keyakinan=None, klaim_inti=claim,
        penjelasan=(f"Mode tanpa LLM: ditampilkan {len(scored)} passage paling relevan (cakupan: {coverage}). "
                    f"Tumpang-tindih kata kunci tertinggi {best:.0%}. Nilai kesesuaiannya secara manual."),
        bukti=[dict(pid=p.pid, section=p.section, kutipan=p.text, peran="kandidat",
                    status="persis", skor=100) for p in scored],
        catatan=["Klaim berbahasa Indonesia vs artikel berbahasa Inggris: aktifkan embedding multibahasa "
                 "atau LLM untuk pencarian lintas bahasa."] if best < 0.2 else [])
