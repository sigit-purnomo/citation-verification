# CiteCheck — Verifikasi Sitasi Proposal terhadap Artikel Asli

Aplikasi Streamlit untuk memeriksa apakah setiap sitasi dalam proposal benar-benar
didukung oleh artikel yang disitasi, dengan menampilkan **kutipan evidence verbatim**
dari artikel asli — meniru panel *Evidence* di Consensus.

## Mode penilaian (tanpa LLM)

- **NLI lokal** — model *Natural Language Inference* multibahasa dari Hugging Face menilai setiap
  pasangan (kalimat artikel, klaim proposal): entailment → *Didukung*, kontradiksi → *Bertentangan*.
  Evidence adalah kalimat asli artikel, jadi selalu verbatim.
- **Kandidat saja** — hanya menampilkan kalimat artikel paling relevan untuk ditinjau manual.

Model (gratis, diunduh otomatis pada pemakaian pertama, berjalan di CPU, dikuantisasi int8):

| Fungsi | Model |
|---|---|
| Pencarian lintas bahasa | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| NLI (default) | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` |
| NLI (hemat memori) | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |

## Menjalankan lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy ke Streamlit Community Cloud

1. Unggah isi folder ini ke repositori GitHub (boleh privat). Folder `contoh/` opsional.
2. Di https://share.streamlit.io → **Create app** → pilih repo, branch, dan `app.py`.
   Di *Advanced settings* pilih Python 3.11 atau 3.12.
3. (Opsional) *Secrets*: `OPENALEX_MAILTO = "nama@kampus.ac.id"`.
4. Proposal bersifat rahasia: setelah deploy, buka **Settings → Sharing** dan batasi siapa yang
   boleh melihat aplikasi.

Catatan kapasitas:
- Pemakaian pertama mengunduh model (±1,5 GB) sehingga butuh beberapa menit; setelah itu model
  di-cache dan dipakai bersama oleh semua pengguna.
- Batas memori Community Cloud terbatas (dokumentasi Streamlit menyebut maksimal ±2,7 GB).
  Bila muncul pesan *"over its resource limits"*, pilih model NLI **MiniLMv2** di sidebar,
  atau pindahkan aplikasi ke Hugging Face Spaces (Streamlit didukung, RAM lebih besar).
- CPU hanya beberapa core: untuk ±100 sitasi, perkirakan beberapa menit.

## Kalibrasi ambang

Ambang default (Didukung ≥ 0,70; Sebagian ≥ 0,40; Bertentangan ≥ 0,70) adalah titik awal.
Sebaiknya kalibrasi dengan 20–30 sitasi yang sudah dinilai manual oleh dosen, lalu sesuaikan
slider *Ambang verdict*.

## Rekayasa balik: bagaimana Consensus mengekstrak evidence

Dari tampilan panel Evidence (mis. `ABSTRACT (#1)`, `INTRODUCTION (#2)` dengan rujukan
`[44-46]`, `ABSTRACT (#3)` yang menyebut efek *tidak signifikan*), dapat disimpulkan alurnya:

| # | Tahap | Yang terlihat di Consensus | Implementasi di CiteCheck |
|---|-------|----------------------------|---------------------------|
| 1 | Unit klaim | Satu pil sitasi = satu kalimat jawaban × satu paper | `citations.py`: kalimat × referensi |
| 2 | Teks sumber | "Read Abstracts and PDFs" | `sources.py`: PDF unggahan → OpenAlex/OA PDF → abstrak |
| 3 | Segmentasi | Label bagian (ABSTRACT, INTRODUCTION) + nomor passage | `sources.py`: deteksi judul bagian, buang daftar pustaka |
| 4 | Retrieval | Hanya beberapa passage paling relevan | `retrieval.py`: BM25 + embedding (opsional), RRF, prior bagian temuan |
| 5 | Kutipan verbatim | Kutipan persis berbahasa asli, bukan parafrase | Evidence = kalimat asli artikel |
| 6 | Bukti berimbang | Ikut menampilkan temuan yang membatasi | Peran: mendukung / membatasi / bertentangan / latar |
| 7 | Sitasi sekunder | Kalimat Introduction yang mengutip `[44-46]` | Kalimat ber-sitasi di artikel ditandai *sekunder* |
| 8 | Penilaian | Klaim dinilai terhadap evidence | `nli.py`: skor entailment/kontradiksi per kalimat |

## Struktur

```
app.py                  UI Streamlit
citecheck/parsing.py    ekstraksi teks PDF/DOCX, parsing daftar pustaka
citecheck/citations.py  pemecahan kalimat, deteksi & pemetaan sitasi
citecheck/sources.py    teks artikel (unggahan/daring), segmentasi bagian
citecheck/retrieval.py  pencarian passage kandidat
citecheck/nli.py        penilaian tanpa LLM (NLI) + pemuatan model
citecheck/verify.py     (opsional) penilaian LLM + verifikasi kutipan, tidak dipakai di UI
citecheck/pipeline.py   orkestrasi
```

## Batasan

- Hasil terbaik bila PDF artikel diunggah. Banyak penerbit memblokir unduhan otomatis,
  sehingga daring sering hanya mendapat abstrak → status "tak dapat diverifikasi" lebih sering.
- Parsing PDF dua kolom/hasil pindai dapat mengacaukan urutan kalimat atau deteksi bagian.
- Gaya sitasi yang didukung: APA/Harvard (parenthetical & naratif) dan numerik `[n]`.
  Footnote (gaya Chicago) belum didukung.
- NLI menilai klaim tunggal dengan baik, tetapi kalimat proposal yang panjang/majemuk sering
  dinilai netral; aplikasi memberi catatan untuk kasus ini.
- Tanpa LLM tidak ada penjelasan naratif (overgeneralisasi, klaim kausal, dsb.).
- Verdict adalah bantuan penilaian, bukan keputusan akhir; baca evidence-nya.
