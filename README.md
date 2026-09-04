# Nimbus — Chatbot AI (Streamlit + Groq)

Chatbot dengan antarmuka Streamlit, jawaban streaming dari Groq
(`llama-3.3-70b-versatile`), riwayat di SQLite, dan RAG di atas dokumen
yang diunggah — embedding Gemini, vector store Qdrant local mode.

> Tahap 1 (chat dasar), tahap 2 (upload file), dan tahap 3 (RAG) sudah jalan.
> Ekstraksi isi gambar menyusul di tahap 4.

## Fitur

- Sidebar riwayat percakapan, urut dari yang paling baru
- Tombol **New Chat** untuk memulai percakapan baru
- Klik percakapan di sidebar untuk membukanya kembali
- Judul percakapan dibuat otomatis oleh Groq dari pesan pertama (maks. 6 kata)
- Tombol hapus percakapan dengan konfirmasi dua langkah
- Jawaban tampil streaming lewat `st.write_stream`
- Seluruh pesan tersimpan di SQLite sehingga tidak hilang saat refresh
- Tema gelap kustom, font Plus Jakarta Sans, chrome bawaan Streamlit disembunyikan

### Upload dokumen

- Tipe yang diterima: PDF, DOCX, XLSX, PPTX, PNG, JPG/JPEG
- Maksimal **5 MB per file** dan **5 file tersimpan**; pelanggaran ditolak
  dengan pesan yang jelas, bukan exception mentah
- Tahapan proses ditampilkan langsung lewat `st.status`
- Daftar dokumen dengan ikon per tipe, ukuran, jumlah karakter, dan jam unggah
- Indikator kuota "3 dari 5 file terpakai"
- Hapus per dokumen (dengan konfirmasi), teksnya ikut terhapus

**File asli tidak pernah disimpan permanen.** Alurnya:

1. File diterima, ditulis ke folder temp
2. Validasi tipe dan ukuran
3. Parsing menjadi teks (pypdf, python-docx, openpyxl, python-pptx)
4. Metadata dan teks disimpan ke SQLite
5. File asli dihapus dari disk — lewat `finally`, jadi tetap terhapus
   walaupun tahap sebelumnya gagal

Gambar hanya dicatat metadatanya (`status = pending`); ekstraksi isinya
dikerjakan di tahap 4.

### RAG

Vector store **Qdrant local mode** (on-disk di `./data/qdrant`, tanpa server
dan tanpa Docker), metric **COSINE**. Embedding memakai **Gemini Embedding
API** (`gemini-embedding-001`), `task_type` `RETRIEVAL_DOCUMENT` untuk isi
dokumen dan `RETRIEVAL_QUERY` untuk pertanyaan.

**Chunking** — target 800 karakter, overlap 150. Potongan diambil di batas
kalimat; kalimat yang kepanjangan dipecah di batas kata, tidak pernah di
tengah kata. Payload tiap chunk: `document_id`, `filename`, `chunk_index`,
`text`.

**Saat upload:** teks → chunking → embedding tiap chunk → simpan ke Qdrant.

**Saat bertanya:**

1. Pertanyaan di-embedding
2. Cari 5 chunk termirip di Qdrant
3. Kutipannya disusun jadi context dan dikirim ke Groq bersama pertanyaan
   aslinya — sebagai **teks biasa**; vector tidak pernah ikut dikirim
4. Model diinstruksikan menjawab hanya dari context, dan mengaku tidak tahu
   kalau informasinya memang tidak ada di dokumen

Di UI:

- Expander **Sumber** di bawah tiap jawaban: nama file, potongan teks chunk,
  dan skor kemiripannya. Tersimpan di SQLite, jadi tetap ada setelah refresh.
- Progres chunking dan embedding tampil di `st.status` saat upload
- Toggle **Jawab pakai dokumen** di sidebar untuk mematikan RAG dan
  membandingkan jawaban dengan/tanpa dokumen
- Menghapus dokumen ikut menghapus chunk-nya dari Qdrant

Ukuran collection Qdrant diambil dari panjang vector yang benar-benar
dikembalikan API, bukan dari angka di konfigurasi, supaya collection tidak
pernah dibuat dengan dimensi yang salah.

## Struktur

```
app.py                  Entry point Streamlit (UI, routing, styling)
core/config.py          Membaca environment variable
core/db.py              Koneksi dan skema SQLite
core/llm.py             Wrapper pemanggilan Groq
core/ingest.py          Pipeline upload: validasi, parsing, simpan, hapus
core/chunking.py        Pemecahan teks jadi chunk
core/embeddings.py      Pemanggilan Gemini Embedding API
core/vectorstore.py     Qdrant local mode
core/rag.py             Perekat: index saat upload, retrieve saat bertanya
.streamlit/config.toml  Tema aplikasi
```

## Skema database

| Tabel           | Kolom                                                                       |
| --------------- | --------------------------------------------------------------------------- |
| `conversations` | `id`, `title`, `created_at`, `updated_at`                                    |
| `messages`      | `id`, `conversation_id`, `role`, `content`, `created_at`                     |
| `documents`     | `id`, `filename`, `filetype`, `filesize`, `char_count`, `status`, `uploaded_at` |
| `document_text` | `id`, `document_id`, `content`                                               |
| `message_sources` | `id`, `message_id`, `document_id`, `filename`, `chunk_index`, `score`, `text` |

`messages.conversation_id`, `document_text.document_id`, dan
`message_sources.message_id` memakai foreign key dengan `ON DELETE CASCADE`,
jadi menghapus induknya ikut membersihkan turunannya.

`message_sources` menyimpan kutipan yang dipakai untuk menjawab tiap pesan —
tabel ini di luar spesifikasi tahap 3, ditambahkan supaya expander "Sumber"
tetap ada setelah halaman di-refresh.

Nilai `documents.status`:

| Status      | Arti                                                  |
| ----------- | ----------------------------------------------------- |
| `indexed`   | teks diekstrak dan chunk-nya sudah masuk Qdrant         |
| `processed` | teks diekstrak, tapi belum terindeks (embedding gagal) |
| `no_text`   | file terbaca tapi tidak ada teks (mis. PDF hasil scan) |
| `pending`   | gambar, ekstraksi isinya menyusul di tahap 4           |

## Menjalankan

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # lalu isi GROQ_API_KEY
streamlit run app.py
```

Ambil API key Groq di <https://console.groq.com/keys> dan API key Gemini di
<https://aistudio.google.com/apikey>. Tanpa `GEMINI_API_KEY` aplikasi tetap
jalan: dokumen masih diurai dan disimpan, hanya RAG-nya yang mati.

## Environment variable

| Variabel           | Wajib | Default                            |
| ------------------ | ----- | ---------------------------------- |
| `GROQ_API_KEY`     | ya    | —                                  |
| `GROQ_BASE_URL`    | tidak | `https://api.groq.com/openai/v1`   |
| `GROQ_MODEL`       | tidak | `llama-3.3-70b-versatile`          |
| `GROQ_TITLE_MODEL` | tidak | mengikuti `GROQ_MODEL`             |
| `APP_DB_PATH`      | tidak | `data/chat.db`                     |

API key hanya dibaca dari `os.environ` dan tidak pernah ditulis di dalam kode.
File `.env`, `*.db`, dan folder `data/` (termasuk `data/qdrant`) sudah masuk
`.gitignore`.

Catatan: `server.maxUploadSize` di `.streamlit/config.toml` sengaja diset 25 MB.
Batas sebenarnya tetap 5 MB dan divalidasi di `core/ingest.py` — plafon yang
lebih longgar dipakai supaya file yang melewati batas tetap sampai ke validator
dan ditolak dengan pesan sendiri, bukan error bawaan Streamlit.
