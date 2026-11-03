# CiteCheck — Verifikasi Sitasi Proposal terhadap Artikel Asli

Aplikasi Streamlit untuk memeriksa apakah setiap sitasi dalam proposal benar-benar
didukung oleh artikel yang disitasi, dengan menampilkan **kutipan evidence verbatim**
dari artikel asli — meniru panel *Evidence* di Consensus.

## Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

API key bisa diisi di sidebar, atau di `.streamlit/secrets.toml`:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
CITECHECK_MODEL   = "claude-sonnet-5-5"
OPENALEX_MAILTO   = "nama@kampus.ac.id"
# atau untuk server kompatibel OpenAI (mis. LLM yang di-host sendiri):
# OPENAI_BASE_URL = "https://llm.kampus.ac.id/v1"
# OPENAI_API_KEY  = "..."
```

Coba dengan berkas di folder `contoh/` (proposal + 2 PDF artikel; 1 referensi sengaja
tidak punya PDF, 1 sitasi sengaja tidak ada di daftar pustaka).

## Rekayasa balik: bagaimana Consensus mengekstrak evidence

Dari tampilan panel Evidence (mis. `ABSTRACT (#1)`, `INTRODUCTION (#2)` dengan rujukan
`[44-46]`, `ABSTRACT (#3)` yang menyebut efek *tidak signifikan*), dapat disimpulkan alurnya:

| # | Tahap | Yang terlihat di Consensus | Implementasi di CiteCheck |
|---|-------|----------------------------|---------------------------|
| 1 | Unit klaim | Satu pil sitasi = satu kalimat jawaban × satu paper | `citations.py`: kalimat × referensi |
| 2 | Teks sumber | "Read Abstracts and PDFs" | `sources.py`: PDF unggahan → OpenAlex/OA PDF → abstrak |
| 3 | Segmentasi | Label bagian (ABSTRACT, INTRODUCTION) + nomor passage | `sources.py`: deteksi judul bagian, buang daftar pustaka |
| 4 | Retrieval | Hanya beberapa passage paling relevan | `retrieval.py`: BM25 + embedding (opsional), RRF, prior bagian temuan |
| 5 | Kutipan verbatim | Kutipan persis berbahasa asli, bukan parafrase | `verify.py`: LLM wajib menyalin persis |
| 6 | Bukti berimbang | Ikut menampilkan temuan yang membatasi | Peran: mendukung / membatasi / bertentangan / latar |
| 7 | Sitasi sekunder | Kalimat Introduction yang mengutip `[44-46]` | Kalimat ber-sitasi di artikel ditandai *sekunder* |
| 8 | Grounding | Kutipan dapat ditelusuri ke teks | Kutipan dicocokkan ulang; ✓ persis / ≈ mirip / ✗ tidak ditemukan |

## Struktur

```
app.py                  UI Streamlit
citecheck/parsing.py    ekstraksi teks PDF/DOCX, parsing daftar pustaka
citecheck/citations.py  pemecahan kalimat, deteksi & pemetaan sitasi
citecheck/sources.py    teks artikel (unggahan/daring), segmentasi bagian
citecheck/retrieval.py  pencarian passage kandidat
citecheck/verify.py     penilaian LLM + verifikasi kutipan
citecheck/pipeline.py   orkestrasi
```

## Batasan

- Hasil terbaik bila PDF artikel diunggah. Banyak penerbit memblokir unduhan otomatis,
  sehingga daring sering hanya mendapat abstrak → status "tak dapat diverifikasi" lebih sering.
- Parsing PDF dua kolom/hasil pindai dapat mengacaukan urutan kalimat atau deteksi bagian.
- Gaya sitasi yang didukung: APA/Harvard (parenthetical & naratif) dan numerik `[n]`.
  Footnote (gaya Chicago) belum didukung.
- Verdict adalah bantuan penilaian, bukan keputusan akhir; baca evidence-nya.
