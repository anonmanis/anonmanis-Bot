"""Pembacaan gambar memakai model multimodal Groq.

Gambar dikirim ke model vision sebagai blok ``image_url`` berisi data URI
base64, lalu hasil deskripsinya diperlakukan persis seperti teks dokumen
lain: dipecah jadi chunk, di-embedding, dan masuk ke Qdrant. Dengan begitu
isi gambar ikut tercari lewat RAG, bukan hanya bisa ditanya sekali.
"""

from __future__ import annotations

import base64
from pathlib import Path

from core import errors
from core.config import get_settings
from core.llm import LLMConfigError, get_client, with_retry

MIME_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
}

VISION_PROMPT = (
    "Deskripsikan isi gambar ini selengkap mungkin dalam Bahasa Indonesia, "
    "lalu salin SEMUA teks yang terlihat di dalamnya.\n\n"
    "Susun jawabanmu seperti ini:\n"
    "1. Ringkasan satu kalimat tentang gambar ini secara keseluruhan.\n"
    "2. Deskripsi rinci: objek, orang, tata letak, warna, dan hubungan antar "
    "elemen. Kalau ini diagram, bagan, atau tangkapan layar, jelaskan "
    "strukturnya dan apa yang digambarkan.\n"
    "3. Teks yang terbaca: salin persis semua tulisan yang muncul, termasuk "
    "judul, label, angka, tabel, dan keterangan kecil. Tulis apa adanya, "
    "jangan diterjemahkan atau diringkas. Kalau tidak ada teks sama sekali, "
    "tulis 'Tidak ada teks pada gambar.'\n\n"
    "Tulis sebagai teks biasa yang mengalir supaya mudah dicari kembali. "
    "Jangan mengarang isi yang tidak terlihat."
)

MAX_TOKENS = 1536


class VisionError(RuntimeError):
    """Pembacaan gambar gagal."""


def is_supported(extension: str) -> bool:
    return extension.lower() in MIME_TYPES


def data_uri(image_bytes: bytes, extension: str) -> str:
    """Bungkus byte gambar menjadi data URI base64."""
    mime = MIME_TYPES.get(extension.lower(), "image/png")
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def describe_image(image_bytes: bytes, filename: str) -> str:
    """Minta model vision mendeskripsikan gambar dan membaca teks di dalamnya.

    Mengembalikan teks deskripsi. Melempar ``VisionError`` dengan pesan yang
    sudah ramah dibaca kalau gagal.
    """
    extension = Path(filename).suffix.lower().lstrip(".")
    if not is_supported(extension):
        raise VisionError(f"Tipe gambar .{extension} tidak didukung model vision.")

    settings = get_settings()
    try:
        client = get_client()
    except LLMConfigError as exc:
        raise VisionError(str(exc)) from exc

    def call():
        return client.chat.completions.create(
            model=settings.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri(image_bytes, extension)},
                        },
                    ],
                }
            ],
            temperature=0.2,
            max_tokens=MAX_TOKENS,
        )

    try:
        response = with_retry(call)
    except Exception as exc:  # noqa: BLE001 - diterjemahkan jadi pesan ramah
        raise VisionError(errors.friendly_message(exc, "Gambar gagal dibaca.")) from exc

    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise VisionError("Model vision tidak mengembalikan deskripsi apa pun.")
    return text
