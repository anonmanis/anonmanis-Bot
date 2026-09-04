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


def _env(name: str, default: str = "") -> str:
    """Ambil environment variable dan bersihkan spasi di ujungnya."""
    return (os.environ.get(name) or "").strip() or default


@dataclass(frozen=True)
class Settings:
    """Kumpulan konfigurasi runtime aplikasi."""

    groq_api_key: str
    base_url: str
    model: str
    title_model: str
    db_path: Path

    @property
    def is_configured(self) -> bool:
        """True kalau API key tersedia sehingga aplikasi bisa memanggil Groq."""
        return bool(self.groq_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Baca konfigurasi dari environment (hasilnya di-cache)."""
    db_path = Path(_env("APP_DB_PATH", DEFAULT_DB_PATH)).expanduser()
    if not db_path.is_absolute():
        db_path = Path(__file__).resolve().parent.parent / db_path

    return Settings(
        groq_api_key=_env("GROQ_API_KEY"),
        base_url=_env("GROQ_BASE_URL", DEFAULT_BASE_URL),
        model=_env("GROQ_MODEL", DEFAULT_MODEL),
        title_model=_env("GROQ_TITLE_MODEL", _env("GROQ_MODEL", DEFAULT_TITLE_MODEL)),
        db_path=db_path,
    )
