# Nimbus — Chatbot AI (Streamlit + Groq)

Chatbot dengan antarmuka Streamlit, jawaban streaming dari Groq
(`llama-3.3-70b-versatile`), dan riwayat percakapan tersimpan di SQLite.

> Tahap ini fokus ke **chat dasar**. Fitur upload dokumen dan RAG menyusul.

## Fitur

- Sidebar riwayat percakapan, urut dari yang paling baru
- Tombol **New Chat** untuk memulai percakapan baru
- Klik percakapan di sidebar untuk membukanya kembali
- Judul percakapan dibuat otomatis oleh Groq dari pesan pertama (maks. 6 kata)
- Tombol hapus percakapan dengan konfirmasi dua langkah
- Jawaban tampil streaming lewat `st.write_stream`
- Seluruh pesan tersimpan di SQLite sehingga tidak hilang saat refresh
- Tema gelap kustom, font Plus Jakarta Sans, chrome bawaan Streamlit disembunyikan

## Struktur

```
app.py                  Entry point Streamlit (UI, routing, styling)
core/config.py          Membaca environment variable
core/db.py              Koneksi dan skema SQLite
core/llm.py             Wrapper pemanggilan Groq
.streamlit/config.toml  Tema aplikasi
```

## Skema database

| Tabel           | Kolom                                                        |
| --------------- | ------------------------------------------------------------ |
| `conversations` | `id`, `title`, `created_at`, `updated_at`                     |
| `messages`      | `id`, `conversation_id`, `role`, `content`, `created_at`      |

`messages.conversation_id` memakai foreign key ke `conversations.id`
dengan `ON DELETE CASCADE`.

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
