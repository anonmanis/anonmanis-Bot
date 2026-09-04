"""Konfigurasi aplikasi.

Semua nilai sensitif dibaca dari environment variable (``os.environ``).
Tidak ada API key yang di-hardcode di dalam repository ini.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# ``python-dotenv`` bersifat opsional: kalau tersedia, isi file .env di root
# project akan dimuat ke dalam os.environ supaya nyaman saat development.
try:  # pragma: no cover - hanya jalur kenyamanan lokal
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except Exception:  # pragma: no cover - dotenv tidak terpasang, abaikan saja
    pass

APP_NAME = "Nimbus"
APP_TAGLINE = "Asisten AI pribadi"

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TITLE_MODEL = "llama-3.3-70b-versatile"
DEFAULT_DB_PATH = "data/chat.db"

# --- RAG ---------------------------------------------------------------- #
# gemini-embedding-001 adalah model embedding stabil di Gemini API. Dimensi
# default menurut halaman modelnya 768 (bisa 128–3072). Nilai di bawah hanya
# dipakai sebagai permintaan ke API; ukuran collection Qdrant selalu diambil
# dari panjang vector yang benar-benar dikembalikan API, jadi tidak akan
# meleset kalau dokumentasinya berubah.
DEFAULT_EMBED_MODEL = "gemini-embedding-001"
DEFAULT_EMBED_DIM = 768
DEFAULT_QDRANT_PATH = "data/qdrant"
DEFAULT_COLLECTION = "documents"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5


def _env(name: str, default: str = "") -> str:
    """Ambil environment variable dan bersihkan spasi di ujungnya."""
    return (os.environ.get(name) or "").strip() or default


def _resolve(raw_path: str) -> Path:
    """Path relatif dihitung dari root project, bukan dari cwd."""
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    return path


@dataclass(frozen=True)
class Settings:
    """Kumpulan konfigurasi runtime aplikasi."""

    groq_api_key: str
    base_url: str
    model: str
    title_model: str
    db_path: Path
    gemini_api_key: str
    embed_model: str
    embed_dim: int
    qdrant_path: Path
    collection: str

    @property
    def is_configured(self) -> bool:
        """True kalau API key tersedia sehingga aplikasi bisa memanggil Groq."""
        return bool(self.groq_api_key)

    @property
    def rag_available(self) -> bool:
        """True kalau embedding bisa dipanggil, jadi RAG bisa dinyalakan."""
        return bool(self.gemini_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Baca konfigurasi dari environment (hasilnya di-cache)."""
    try:
        embed_dim = int(_env("GEMINI_EMBED_DIM", str(DEFAULT_EMBED_DIM)))
    except ValueError:
        embed_dim = DEFAULT_EMBED_DIM

    return Settings(
        groq_api_key=_env("GROQ_API_KEY"),
        base_url=_env("GROQ_BASE_URL", DEFAULT_BASE_URL),
        model=_env("GROQ_MODEL", DEFAULT_MODEL),
        title_model=_env("GROQ_TITLE_MODEL", _env("GROQ_MODEL", DEFAULT_TITLE_MODEL)),
        db_path=_resolve(_env("APP_DB_PATH", DEFAULT_DB_PATH)),
        # GOOGLE_API_KEY diterima juga karena itu nama yang dipakai SDK google-genai.
        gemini_api_key=_env("GEMINI_API_KEY", _env("GOOGLE_API_KEY")),
        embed_model=_env("GEMINI_EMBED_MODEL", DEFAULT_EMBED_MODEL),
        embed_dim=embed_dim,
        qdrant_path=_resolve(_env("QDRANT_PATH", DEFAULT_QDRANT_PATH)),
        collection=_env("QDRANT_COLLECTION", DEFAULT_COLLECTION),
    )
