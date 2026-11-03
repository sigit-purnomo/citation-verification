"""CiteCheck — verifikasi kesesuaian sitasi proposal dengan evidence dari artikel asli.

Jalankan:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import html
import io
import json
import os

import pandas as pd
import streamlit as st

from citecheck import pipeline, sources, verify
from citecheck.citations import strip_citations
from citecheck.parsing import Reference, extract_text

st.set_page_config(page_title="CiteCheck · Verifikasi Sitasi", page_icon="🔎", layout="wide")

VERDICT_STYLE = {
    "DIDUKUNG": ("Didukung", "#1a7f37", "#dafbe1"),
    "DIDUKUNG_SEBAGIAN": ("Didukung sebagian", "#9a6700", "#fff8c5"),
    "TIDAK_DIDUKUNG": ("Tidak didukung", "#cf222e", "#ffebe9"),
    "BERTENTANGAN": ("Bertentangan", "#a40e26", "#ffd8d3"),
    "TIDAK_DAPAT_DIVERIFIKASI": ("Tak dapat diverifikasi", "#57606a", "#eaeef2"),
    "PERLU_TINJAUAN_MANUAL": ("Perlu tinjauan manual", "#0969da", "#ddf4ff"),
    "GALAT": ("Galat", "#57606a", "#eaeef2"),
}
ROLE_STYLE = {"mendukung": "#1a7f37", "membatasi": "#9a6700", "bertentangan": "#cf222e",
              "latar": "#6e7781", "kandidat": "#0969da"}

st.markdown("""
<style>
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-weight:600;font-size:.82rem}
.ref{font-weight:600}
.claim{font-size:1.02rem;line-height:1.6;margin:.4rem 0 .6rem}
.claim mark{background:#fff3b0;padding:0 3px;border-radius:4px}
.ev{border-left:3px solid #d0d7de;padding:.35rem .8rem;margin:.45rem 0}
.ev-head{font-family:ui-monospace,Menlo,monospace;font-size:.78rem;color:#57606a;letter-spacing:.02em}
.ev-head .role{font-family:inherit;font-weight:700;margin-left:.5rem}
.ev-q{margin-top:.2rem;line-height:1.55}
.qv-ok{color:#1a7f37}.qv-mid{color:#9a6700}.qv-bad{color:#cf222e}
.muted{color:#6e7781;font-size:.85rem}
</style>""", unsafe_allow_html=True)


def secret(name: str) -> str:
    try:
        return st.secrets.get(name, "") or os.environ.get(name, "")
    except Exception:  # noqa: BLE001 - tidak ada secrets.toml
        return os.environ.get(name, "")


# ---------------------------------------------------------------- cache
@st.cache_data(show_spinner=False)
def cached_parse(data: bytes, name: str):
    return pipeline.parse_proposal(data, name)


@st.cache_data(show_spinner=False)
def cached_text(data: bytes, name: str) -> str:
    return extract_text(data, name)


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def cached_fetch(ref: Reference, mailto: str, try_pdf: bool):
    return sources.fetch_online(ref, mailto, try_pdf)


@st.cache_resource(show_spinner="Memuat model embedding multibahasa…")
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Pengaturan")
    mode = st.radio("Mesin penilai", ["Claude (Anthropic API)", "API kompatibel OpenAI", "Tanpa LLM (heuristik)"],
                    help="LLM dipakai untuk menerjemahkan klaim menjadi query dan menilai kesesuaian. "
                         "Tanpa LLM, aplikasi hanya menampilkan passage kandidat.")
    api_key, model, base_url = "", "", ""
    if mode.startswith("Claude"):
        api_key = st.text_input("ANTHROPIC_API_KEY", value=secret("ANTHROPIC_API_KEY"), type="password")
        model = st.text_input("Model", value=secret("CITECHECK_MODEL") or "claude-sonnet-5-5")
    elif mode.startswith("API"):
        base_url = st.text_input("Base URL", value=secret("OPENAI_BASE_URL"),
                                 placeholder="https://…/v1 (kosongkan untuk OpenAI)")
        api_key = st.text_input("API key", value=secret("OPENAI_API_KEY"), type="password")
        model = st.text_input("Model", value=secret("CITECHECK_MODEL") or "gpt-4.1-mini")

    st.divider()
    st.subheader("Sumber artikel")
    online = st.checkbox("Cari artikel secara daring", value=True,
                         help="OpenAlex → PDF open access; bila gagal, abstrak dari OpenAlex/Crossref/Semantic Scholar.")
    try_pdf = st.checkbox("Coba unduh PDF open access", value=True, disabled=not online)
    mailto = st.text_input("Email untuk OpenAlex (opsional)", value=secret("OPENALEX_MAILTO"),
                           help="Masuk ke 'polite pool' OpenAlex agar lebih stabil.")

    st.divider()
    st.subheader("Pencarian evidence")
    top_k = st.slider("Kalimat kandidat per sitasi", 4, 15, 8)
    use_emb = st.checkbox("Embedding multibahasa (ID↔EN)", value=False,
                          help="Butuh paket sentence-transformers (~500 MB model). Membantu pencarian lintas bahasa.")
    workers = st.slider("Panggilan paralel", 1, 8, 4)

# ---------------------------------------------------------------- header
st.title("🔎 CiteCheck")
st.caption("Unggah proposal → aplikasi mendeteksi setiap sitasi, mengambil teks artikel yang disitasi, "
           "lalu menampilkan kutipan evidence dari artikel asli beserta penilaian kesesuaiannya.")

with st.expander("Bagaimana cara kerjanya? (rekayasa balik alur evidence ala Consensus)"):
    st.markdown("""
1. **Pecah proposal menjadi kalimat dan deteksi sitasi** — APA/Harvard `(Utami et al., 2025)`, naratif `Utami (2025)`, dan numerik `[12]`. Setiap pasangan *kalimat × referensi* menjadi satu klaim yang diperiksa (sama seperti pil sitasi di Consensus).
2. **Petakan sitasi ke daftar pustaka** (nama penulis pertama + tahun, termasuk sufiks 2025a/b) dan ambil DOI/judul.
3. **Ambil teks artikel**: PDF yang Anda unggah (paling andal) atau daring — OpenAlex → PDF open access → abstrak (OpenAlex/Crossref/Semantic Scholar).
4. **Segmentasi artikel per bagian** (ABSTRACT, INTRODUCTION, RESULTS, DISCUSSION…) dan per kalimat; daftar pustaka artikel dibuang. Kalimat yang sendirinya mengutip studi lain ditandai **sekunder** — inilah mengapa Consensus menampilkan label seperti `INTRODUCTION (#2)` dengan rujukan `[44-46]`.
5. **Retrieval**: klaim (plus terjemahan query Inggris dari LLM) dicari dengan BM25 dan opsional embedding multibahasa, digabung *reciprocal rank fusion*; bagian temuan diberi bobot lebih.
6. **Penilaian + kutipan verbatim**: LLM memilih kutipan *persis* dari passage dan memberi peran (mendukung / membatasi / bertentangan / latar) serta verdict.
7. **Grounding check**: setiap kutipan dicocokkan ulang ke teks artikel. Kutipan yang tidak ditemukan diberi tanda ✗, sehingga evidence tidak bisa dikarang.
""")

c1, c2 = st.columns(2)
prop = c1.file_uploader("1 · Proposal (PDF / DOCX / TXT)", type=["pdf", "docx", "txt", "md"])
ref_files = c2.file_uploader("2 · PDF artikel yang disitasi (opsional, sangat disarankan)",
                             type=["pdf"], accept_multiple_files=True)

if not prop:
    st.info("Unggah proposal untuk mulai. Proposal harus memiliki bagian **Daftar Pustaka / References**.")
    st.stop()

pdata = prop.getvalue()
pkey = hashlib.md5(pdata).hexdigest()
if st.session_state.get("pkey") != pkey:
    st.session_state.update(pkey=pkey, docs=None, results=None)

with st.spinner("Membaca proposal…"):
    parsed = cached_parse(pdata, prop.name)
refs: list[Reference] = parsed["refs"]
instances = parsed["instances"]
refs_by_idx = {r.idx: r for r in refs}

if not refs:
    st.error("Bagian Daftar Pustaka/References tidak ditemukan atau tidak terbaca. Pastikan judulnya berdiri "
             "di baris sendiri (mis. 'DAFTAR PUSTAKA') dan setiap entri dimulai di baris baru.")
    st.stop()

# ---------------------------------------------------------------- ringkasan parsing
st.subheader("Hasil pembacaan proposal")
m = st.columns(5)
m[0].metric("Sitasi terdeteksi", len(instances))
m[1].metric("Kalimat bersitasi", len({i.sent_idx for i in instances}))
m[2].metric("Entri pustaka", len(refs))
m[3].metric("Sitasi tanpa entri pustaka", len(parsed["orphans"]))
m[4].metric("Pustaka tak pernah disitasi", len(parsed["uncited"]))

with st.expander("Daftar pustaka yang terbaca"):
    st.dataframe(pd.DataFrame([dict(No=r.idx, Label=r.label, Judul=r.title, DOI=r.doi,
                                    Disitasi=sum(i.ref_idx == r.idx for i in instances)) for r in refs]),
                 hide_index=True, width="stretch")
if parsed["orphans"] or parsed["uncited"]:
    with st.expander("⚠️ Inkonsistensi sitasi ↔ daftar pustaka"):
        if parsed["orphans"]:
            st.markdown("**Sitasi di teks tanpa entri di daftar pustaka**")
            st.dataframe(pd.DataFrame(parsed["orphans"]), hide_index=True, width="stretch")
        if parsed["uncited"]:
            st.markdown("**Entri daftar pustaka yang tidak pernah disitasi**")
            st.dataframe(pd.DataFrame([dict(Label=r.label, Entri=r.raw) for r in parsed["uncited"]]),
                         hide_index=True, width="stretch")
if not instances:
    st.warning("Tidak ada sitasi yang berhasil dipetakan ke daftar pustaka.")
    st.stop()

# ---------------------------------------------------------------- PDF unggahan
cited = sorted({i.ref_idx for i in instances})
pdf_texts = {f.name: cached_text(f.getvalue(), f.name) for f in (ref_files or [])}
mapping: dict[int, str] = {}
if pdf_texts:
    auto = sources.match_uploaded(pdf_texts, refs)
    opts = ["— tidak dipetakan —"] + [f"{r.idx}. {r.label} — {r.title[:70]}" for r in refs]
    with st.expander(f"Pemetaan PDF unggahan ({sum(v is not None for v in auto.values())}/{len(auto)} otomatis)",
                     expanded=any(v is None for v in auto.values())):
        for name in pdf_texts:
            default = auto[name] if auto[name] is not None else 0
            choice = st.selectbox(name, opts, index=default, key=f"map_{name}")
            if choice != opts[0]:
                mapping[int(choice.split(".")[0])] = name

# ---------------------------------------------------------------- langkah 1: sumber
st.subheader("Langkah 1 · Siapkan teks artikel sumber")
if st.button("📥 Siapkan artikel", type="primary" if st.session_state.docs is None else "secondary"):
    docs, bar = {}, st.progress(0.0, "Menyiapkan…")
    for n, idx in enumerate(cited, 1):
        ref = refs_by_idx[idx]
        bar.progress(n / len(cited), f"{n}/{len(cited)} · {ref.label}")
        doc = None
        if idx in mapping:
            doc = sources.build_from_fulltext(pdf_texts[mapping[idx]], idx,
                                              f"PDF unggahan: {mapping[idx]}", ref.title)
        elif online and (ref.doi or len(ref.title) > 15):
            doc = cached_fetch(ref, mailto, try_pdf)
        docs[idx] = doc
    bar.empty()
    st.session_state.update(docs=docs, results=None)

docs = st.session_state.docs
if docs is not None:
    rows = []
    for idx in cited:
        d, r = docs.get(idx), refs_by_idx[idx]
        rows.append(dict(Referensi=r.label, Judul=r.title[:90], Sumber=d.origin if d else "tidak tersedia",
                         Cakupan=d.coverage if d else "—", Kalimat=len(d.sentences) if d else 0,
                         Sitasi=sum(i.ref_idx == idx for i in instances)))
    df_src = pd.DataFrame(rows)
    full = (df_src.Cakupan == "teks lengkap").sum()
    st.caption(f"Teks lengkap: {full} · abstrak saja: {(df_src.Cakupan == 'abstrak saja').sum()} · "
               f"tidak tersedia: {(df_src.Kalimat == 0).sum()} dari {len(cited)} artikel. "
               "Klaim yang hanya dicek terhadap abstrak bisa berstatus 'tak dapat diverifikasi'.")
    st.dataframe(df_src, hide_index=True, width="stretch")

# ---------------------------------------------------------------- langkah 2: verifikasi
st.subheader("Langkah 2 · Verifikasi sitasi")
if docs is None:
    st.info("Jalankan Langkah 1 terlebih dahulu.")
    st.stop()

if st.button("🧪 Jalankan verifikasi", type="primary" if st.session_state.results is None else "secondary"):
    llm = None
    if not mode.startswith("Tanpa"):
        if not api_key and not base_url:
            st.error("Isi API key di sidebar, atau pilih mode 'Tanpa LLM'.")
            st.stop()
        llm = verify.LLM("anthropic" if mode.startswith("Claude") else "openai", api_key, model, base_url)
    embedder = None
    if use_emb:
        try:
            embedder = load_embedder()
        except ImportError:
            st.warning("sentence-transformers belum terpasang; lanjut tanpa embedding.")
    with st.spinner("Menyusun query pencarian lintas bahasa…"):
        claims = [strip_citations(i.sentence) for i in instances]
        q_en = verify.make_queries(llm, claims) if llm else [""] * len(claims)
    with st.spinner("Mencari passage kandidat di setiap artikel…"):
        passages = pipeline.retrieve_all(instances, docs, q_en, top_k, embedder)
    bar = st.progress(0.0, "Menilai…")
    results = pipeline.run_verification(
        instances, refs, docs, passages, llm, workers,
        on_progress=lambda n, t: bar.progress(n / t, f"Menilai sitasi {n}/{t}"))
    bar.empty()
    st.session_state.results = results

results = st.session_state.results
if not results:
    st.stop()


# ---------------------------------------------------------------- tampilan hasil
def badge(v: str) -> str:
    label, fg, bg = VERDICT_STYLE.get(v, (v, "#333", "#eee"))
    return f'<span class="badge" style="color:{fg};background:{bg}">{label}</span>'


def highlight(sentence: str, marker: str) -> str:
    s = html.escape(sentence)
    mk = html.escape(marker)
    return s.replace(mk, f"<mark>{mk}</mark>", 1) if mk else s


def evidence_html(b: dict) -> str:
    qv = {"persis": ('<span class="qv-ok">✓ kutipan terverifikasi</span>'),
          "mirip": (f'<span class="qv-mid">≈ mirip ({b.get("skor", 0):.0f}%)</span>'),
          "tidak ditemukan": '<span class="qv-bad">✗ tidak ditemukan di artikel</span>'}[b["status"]]
    color = ROLE_STYLE.get(b["peran"], "#6e7781")
    return (f'<div class="ev" style="border-left-color:{color}">'
            f'<div class="ev-head">❝ {html.escape(b["section"])} ({html.escape(str(b["pid"]))})'
            f'<span class="role" style="color:{color}">{html.escape(b["peran"])}</span> · {qv}</div>'
            f'<div class="ev-q">“{html.escape(b["kutipan"])}”</div></div>')


st.divider()
counts = pd.Series([r["verdict"] for r in results]).value_counts()
cols = st.columns(len(counts))
for col, (v, n) in zip(cols, counts.items()):
    col.markdown(f"{badge(v)}<br><span style='font-size:1.6rem;font-weight:700'>{n}</span>",
                 unsafe_allow_html=True)

f1, f2, f3 = st.columns([2, 2, 1])
sel_v = f1.multiselect("Filter verdict", list(counts.index), default=list(counts.index),
                       format_func=lambda v: VERDICT_STYLE.get(v, (v,))[0])
ref_opts = sorted({r["ref_label"] for r in results})
sel_r = f2.multiselect("Filter referensi", ref_opts)
order = f3.selectbox("Urutkan", ["Urutan dokumen", "Verdict", "Referensi"])

shown = [r for r in results if r["verdict"] in sel_v and (not sel_r or r["ref_label"] in sel_r)]
if order == "Verdict":
    rank = {v: i for i, v in enumerate(VERDICT_STYLE)}
    shown.sort(key=lambda r: (rank.get(r["verdict"], 99), r["cid"]))
elif order == "Referensi":
    shown.sort(key=lambda r: (r["ref_label"], r["cid"]))

for r in shown:
    with st.container(border=True):
        conf = f" · keyakinan {r['keyakinan']:.0%}" if isinstance(r.get("keyakinan"), (int, float)) else ""
        st.markdown(f"{badge(r['verdict'])} &nbsp; <span class='ref'>{html.escape(r['ref_label'])}</span>"
                    f" <span class='muted'>· {html.escape(r['cakupan'])} · {html.escape(r['sumber'])}{conf}</span>",
                    unsafe_allow_html=True)
        st.markdown(f"<div class='claim'>{highlight(r['kalimat'], r['marker'])}</div>", unsafe_allow_html=True)
        if r.get("klaim_inti") and r["verdict"] != "PERLU_TINJAUAN_MANUAL":
            st.markdown(f"**Klaim yang diatribusikan:** {r['klaim_inti']}")
        if r.get("penjelasan"):
            st.markdown(r["penjelasan"])
        if r["bukti"]:
            st.markdown("".join(evidence_html(b) for b in r["bukti"]), unsafe_allow_html=True)
        for c in r.get("catatan", []):
            st.markdown(f"⚠️ <span class='muted'>{html.escape(c)}</span>", unsafe_allow_html=True)
        with st.expander(f"Semua passage kandidat ({len(r['kandidat'])}) · entri pustaka"):
            st.caption(r["ref_raw"])
            if r.get("url"):
                st.markdown(f"[Buka artikel]({r['url']})")
            for p in r["kandidat"]:
                tag = " · sekunder" if p["secondary"] else ""
                st.markdown(f"`{p['section']} ({p['pid']}){tag}` {p['text']}")

# ---------------------------------------------------------------- ekspor
st.divider()
flat = [dict(No=r["cid"], Kalimat=r["kalimat"], Referensi=r["ref_label"], Judul=r["ref_title"],
             Verdict=VERDICT_STYLE.get(r["verdict"], (r["verdict"],))[0], Keyakinan=r.get("keyakinan"),
             Klaim=r.get("klaim_inti", ""), Penjelasan=r.get("penjelasan", ""),
             Evidence="\n".join(f'[{b["section"]} {b["pid"]} · {b["peran"]} · {b["status"]}] {b["kutipan"]}'
                                for b in r["bukti"]),
             Catatan="\n".join(r.get("catatan", [])), Sumber=r["sumber"], Cakupan=r["cakupan"])
        for r in results]
df = pd.DataFrame(flat)
xbuf = io.BytesIO()
df.to_excel(xbuf, index=False, sheet_name="CiteCheck")
e1, e2, e3 = st.columns(3)
e1.download_button("⬇️ Excel", xbuf.getvalue(), "citecheck.xlsx", width="stretch")
e2.download_button("⬇️ CSV", df.to_csv(index=False).encode("utf-8-sig"), "citecheck.csv", width="stretch")
e3.download_button("⬇️ JSON lengkap", json.dumps(results, ensure_ascii=False, indent=2).encode(),
                   "citecheck.json", width="stretch")
