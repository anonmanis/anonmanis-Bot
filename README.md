# Nimbus — Chatbot AI (Streamlit + Groq)

Chatbot dengan antarmuka Streamlit, jawaban streaming dari Groq
(`llama-3.3-70b-versatile`), dan riwayat percakapan tersimpan di SQLite.

> Tahap 1 chat dasar dan tahap 2 upload file sudah jalan. Retrieval (RAG)
> menyusul di tahap berikutnya — dokumen yang diunggah belum dipakai untuk
> menjawab.

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

## Struktur

```
app.py                  Entry point Streamlit (UI, routing, styling)
core/config.py          Membaca environment variable
core/db.py              Koneksi dan skema SQLite
core/llm.py             Wrapper pemanggilan Groq
core/ingest.py          Pipeline upload: validasi, parsing, simpan, hapus
.streamlit/config.toml  Tema aplikasi
```

## Skema database

| Tabel           | Kolom                                                                       |
| --------------- | --------------------------------------------------------------------------- |
| `conversations` | `id`, `title`, `created_at`, `updated_at`                                    |
| `messages`      | `id`, `conversation_id`, `role`, `content`, `created_at`                     |
| `documents`     | `id`, `filename`, `filetype`, `filesize`, `char_count`, `status`, `uploaded_at` |
| `document_text` | `id`, `document_id`, `content`                                               |

`messages.conversation_id` dan `document_text.document_id` memakai foreign key
dengan `ON DELETE CASCADE`, jadi menghapus induknya ikut membersihkan turunannya.

Nilai `documents.status`:

| Status      | Arti                                                  |
| ----------- | ----------------------------------------------------- |
| `processed` | teks berhasil diekstrak                                |
| `no_text`   | file terbaca tapi tidak ada teks (mis. PDF hasil scan) |
| `pending`   | gambar, ekstraksi isinya menyusul di tahap 4           |

## Menjalankan

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # lalu isi GROQ_API_KEY
streamlit run app.py
```

Ambil API key di <https://console.groq.com/keys>.

## Environment variable

| Variabel           | Wajib | Default                            |
| ------------------ | ----- | ---------------------------------- |
| `GROQ_API_KEY`     | ya    | —                                  |
| `GROQ_BASE_URL`    | tidak | `https://api.groq.com/openai/v1`   |
| `GROQ_MODEL`       | tidak | `llama-3.3-70b-versatile`          |
| `GROQ_TITLE_MODEL` | tidak | mengikuti `GROQ_MODEL`             |
| `APP_DB_PATH`      | tidak | `data/chat.db`                     |

API key hanya dibaca dari `os.environ` dan tidak pernah ditulis di dalam kode.
File `.env`, `*.db`, dan folder `data/` sudah masuk `.gitignore`.

Catatan: `server.maxUploadSize` di `.streamlit/config.toml` sengaja diset 25 MB.
Batas sebenarnya tetap 5 MB dan divalidasi di `core/ingest.py` — plafon yang
lebih longgar dipakai supaya file yang melewati batas tetap sampai ke validator
dan ditolak dengan pesan sendiri, bukan error bawaan Streamlit.
