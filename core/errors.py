"""Penerjemah error teknis menjadi pesan yang ramah dibaca pengguna.

Semua pemanggilan API keluar (Groq dan Gemini) melewati modul ini supaya
yang muncul di layar berupa kalimat yang bisa ditindaklanjuti, bukan
traceback atau pesan mentah dari SDK.
"""

from __future__ import annotations

import re

RATE_LIMIT = (
    "Batas pemakaian API tercapai (rate limit). "
    "Tunggu sebentar lalu coba lagi."
)
TIMEOUT = "Permintaan ke server terlalu lama dan dihentikan. Coba lagi sebentar lagi."
CONNECTION = "Tidak bisa terhubung ke server. Periksa koneksi internet lalu coba lagi."
AUTH = "API key ditolak. Periksa kembali isi file .env."
SERVER = "Server sedang bermasalah di sisi mereka. Coba lagi beberapa saat lagi."
UNKNOWN = "Terjadi kendala tak terduga. Coba lagi sebentar lagi."


def status_code(exc: BaseException) -> int | None:
    """Ambil kode HTTP dari exception SDK mana pun, kalau ada."""
    for attribute in ("status_code", "code", "http_status"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int) and 100 <= value <= 599:
            return value

    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int):
        return value

    # SDK yang hanya menaruh kodenya di dalam teks pesan.
    match = re.search(r"\b(4\d\d|5\d\d)\b", str(exc))
    return int(match.group(1)) if match else None


def is_rate_limit(exc: BaseException) -> bool:
    if status_code(exc) == 429:
        return True
    text = str(exc).lower()
    return "rate limit" in text or "too many requests" in text or "resource_exhausted" in text


def retry_after_seconds(exc: BaseException) -> float | None:
    """Baca header Retry-After kalau server menyebutkannya."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    match = re.search(r"try again in ([\d.]+)s", str(exc), re.IGNORECASE)
    return float(match.group(1)) if match else None


def is_retryable(exc: BaseException) -> bool:
    """True untuk kegagalan sementara yang masuk akal dicoba ulang."""
    if is_rate_limit(exc):
        return True
    code = status_code(exc)
    if code is not None and code >= 500:
        return True
    name = exc.__class__.__name__.lower()
    return "timeout" in name or "connection" in name


def friendly_message(exc: BaseException, prefix: str = "") -> str:
    """Ubah exception menjadi satu kalimat yang enak dibaca pengguna."""
    if is_rate_limit(exc):
        wait = retry_after_seconds(exc)
        message = RATE_LIMIT
        if wait:
            message = (
                "Batas pemakaian API tercapai (rate limit). "
                f"Coba lagi sekitar {max(1, round(wait))} detik lagi."
            )
    else:
        code = status_code(exc)
        name = exc.__class__.__name__.lower()
        if code in (401, 403):
            message = AUTH
        elif code is not None and code >= 500:
            message = SERVER
        elif "timeout" in name:
            message = TIMEOUT
        elif "connection" in name:
            message = CONNECTION
        else:
            message = UNKNOWN

    return f"{prefix} {message}".strip() if prefix else message
