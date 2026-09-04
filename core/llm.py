"""Wrapper tipis di atas Groq API (endpoint OpenAI-compatible).

Semua pemanggilan model lewat modul ini supaya UI tidak perlu tahu detail
transport-nya. API key diambil dari environment melalui ``core.config``.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Iterator, Sequence

from openai import OpenAI

from core.config import get_settings

SYSTEM_PROMPT = (
    "Kamu adalah Nimbus, asisten AI yang ramah, ringkas, dan akurat. "
    "Jawab dengan bahasa yang sama seperti bahasa pengguna. "
    "Gunakan markdown seperlunya: gunakan poin-poin untuk daftar dan blok kode "
    "dengan penanda bahasa untuk kode. Kalau tidak tahu sesuatu, katakan terus "
    "terang daripada mengarang."
)

# Context dikirim sebagai TEKS biasa di dalam system prompt; pertanyaan
# pengguna tetap dikirim apa adanya. Vector tidak pernah ikut dikirim.
RAG_PROMPT_TEMPLATE = """{base}

Untuk pertanyaan berikut, kamu diberi kutipan dari dokumen yang diunggah pengguna.

ATURAN MENJAWAB:
- Jawab HANYA berdasarkan kutipan di bawah.
- Kalau jawabannya tidak ada di kutipan, katakan terus terang bahwa informasinya
  tidak ada di dokumen yang diunggah. Jangan mengarang dan jangan menambal dari
  pengetahuan umum.
- Sebut nama file sumbernya saat menyampaikan fakta dari kutipan.
- Jawab dengan bahasa yang sama seperti pertanyaan pengguna.

KUTIPAN DOKUMEN:
{context}
"""

TITLE_PROMPT = (
    "Buat judul singkat untuk percakapan berdasarkan pesan pertama pengguna. "
    "Aturan: maksimal 6 kata, tanpa tanda kutip, tanpa tanda titik di akhir, "
    "gunakan bahasa yang sama dengan pesan pengguna. "
    "Balas HANYA dengan judulnya saja."
)

MAX_TITLE_WORDS = 6


class LLMConfigError(RuntimeError):
    """Dipakai saat konfigurasi (mis. API key) belum lengkap."""


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Client OpenAI yang diarahkan ke endpoint Groq."""
    settings = get_settings()
    if not settings.groq_api_key:
        raise LLMConfigError(
            "GROQ_API_KEY belum diset. Salin .env.example menjadi .env, "
            "isi API key dari https://console.groq.com/keys, lalu jalankan ulang aplikasi."
        )
    return OpenAI(
        api_key=settings.groq_api_key,
        base_url=settings.base_url,
        timeout=60.0,
        max_retries=2,
    )


def _payload(messages: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    """Ubah baris database menjadi payload chat completions."""
    return [
        {"role": str(m["role"]), "content": str(m["content"])}
        for m in messages
        if str(m.get("role")) in ("user", "assistant", "system") and str(m.get("content", "")).strip()
    ]


def build_system_prompt(context: str | None = None) -> str:
    """System prompt biasa, atau versi RAG kalau ada context dokumen."""
    if not context:
        return SYSTEM_PROMPT
    return RAG_PROMPT_TEMPLATE.format(base=SYSTEM_PROMPT, context=context)


def stream_chat(
    messages: Sequence[dict[str, Any]],
    *,
    context: str | None = None,
    temperature: float = 0.6,
    max_tokens: int = 2048,
) -> Iterator[str]:
    """Streaming jawaban model, potongan demi potongan.

    ``context`` berisi kutipan dokumen sebagai teks biasa. Generator ini
    dipakai langsung oleh ``st.write_stream`` di UI.
    """
    settings = get_settings()
    stream = get_client().chat.completions.create(
        model=settings.model,
        messages=[
            {"role": "system", "content": build_system_prompt(context)},
            *_payload(messages),
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        piece = getattr(delta, "content", None)
        if piece:
            yield piece


def fallback_title(text: str) -> str:
    """Judul cadangan dari teks pengguna kalau pemanggilan model gagal."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return "Percakapan baru"
    words = cleaned.split(" ")[:MAX_TITLE_WORDS]
    title = " ".join(words)
    return (title[:60] + "…") if len(title) > 60 else title


def generate_title(first_user_message: str) -> str:
    """Minta Groq meringkas pesan pertama menjadi judul maksimal 6 kata."""
    settings = get_settings()
    try:
        response = get_client().chat.completions.create(
            model=settings.title_model,
            messages=[
                {"role": "system", "content": TITLE_PROMPT},
                {"role": "user", "content": first_user_message[:2000]},
            ],
            temperature=0.2,
            max_tokens=32,
        )
        raw = (response.choices[0].message.content or "").strip()
    except Exception:
        return fallback_title(first_user_message)

    # Bersihkan tanda kutip, penomoran, dan tanda baca di ujung judul.
    title = raw.splitlines()[0].strip()
    title = title.strip("\"'“”‘’ ")
    title = re.sub(r"^(judul|title)\s*[:\-]\s*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+", " ", title).strip(" .,:;-–—")

    words = title.split(" ")
    if len(words) > MAX_TITLE_WORDS:
        title = " ".join(words[:MAX_TITLE_WORDS])

    return title or fallback_title(first_user_message)
