"""Pemanggilan Gemini Embedding API.

Model default ``gemini-embedding-001`` (model embedding stabil di Gemini API).
Dokumentasi menyebut dimensi bisa diatur 128–3072; nilai yang dipakai diambil
dari ``core.config`` dan dikirim lewat ``output_dimensionality``.

Dua catatan yang mempengaruhi implementasi di bawah:

* Untuk retrieval, korpus dan pertanyaan harus memakai ``task_type`` yang
  berbeda: ``RETRIEVAL_DOCUMENT`` untuk isi dokumen, ``RETRIEVAL_QUERY``
  untuk pertanyaan pengguna.
* Sebagian model embedding (mis. seri multimodal) menggabungkan beberapa
  ``contents`` menjadi SATU vector, bukan satu vector per teks. Karena itu
  jumlah vector yang kembali selalu dicek; kalau tidak cocok, embedding
  diulang satu per satu supaya tiap chunk pasti punya vector sendiri.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Iterable, Sequence

from core import errors
from core.config import REQUEST_TIMEOUT, get_settings
from core.llm import with_retry

TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"

# Ukuran batch konservatif; kalau ditolak API, otomatis turun ke satu per satu.
BATCH_SIZE = 8


class EmbeddingError(RuntimeError):
    """Embedding gagal dibuat (konfigurasi kurang atau API bermasalah)."""


@lru_cache(maxsize=1)
def get_client():
    """Client google-genai (di-cache supaya tidak dibuat ulang tiap rerun)."""
    settings = get_settings()
    if not settings.gemini_api_key:
        raise EmbeddingError(
            "GEMINI_API_KEY belum diset, jadi dokumen tidak bisa diindeks untuk RAG."
        )
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover
        raise EmbeddingError("Paket google-genai belum terpasang.") from exc

    from google.genai import types

    return genai.Client(
        api_key=settings.gemini_api_key,
        # HttpOptions.timeout dihitung dalam milidetik.
        http_options=types.HttpOptions(timeout=int(REQUEST_TIMEOUT * 1000)),
    )


def _config(task_type: str, with_task: bool = True):
    from google.genai import types

    settings = get_settings()
    fields: dict[str, object] = {"output_dimensionality": settings.embed_dim}
    if with_task:
        fields["task_type"] = task_type
    return types.EmbedContentConfig(**fields)


def normalize(vector: Sequence[float]) -> list[float]:
    """Normalisasi L2.

    Dokumentasi Gemini menyebut vector perlu dinormalisasi manual saat
    ``output_dimensionality`` bukan nilai penuh. Menormalisasi vector yang
    sudah ternormalisasi tidak mengubah apa pun, jadi ini aman dilakukan
    untuk semua ukuran.
    """
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        return list(vector)
    return [value / norm for value in vector]


def _call(contents: list[str], task_type: str) -> list[list[float]]:
    """Satu panggilan API; mengembalikan vector mentah apa adanya."""
    client = get_client()
    model = get_settings().embed_model
    try:
        response = with_retry(
            lambda: client.models.embed_content(
                model=model, contents=contents, config=_config(task_type)
            )
        )
    except Exception as exc:
        # Model yang tidak mengenal task_type (mis. seri multimodal) dicoba lagi tanpa itu.
        if "task_type" not in str(exc).lower():
            raise
        response = with_retry(
            lambda: client.models.embed_content(
                model=model, contents=contents, config=_config(task_type, with_task=False)
            )
        )
    return [list(embedding.values or []) for embedding in (response.embeddings or [])]


def _batched(items: Sequence[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def embed_texts(texts: Sequence[str], task_type: str = TASK_DOCUMENT) -> list[list[float]]:
    """Embedding untuk banyak teks; hasilnya satu vector per teks, urut sama."""
    texts = [text for text in texts]
    if not texts:
        return []

    vectors: list[list[float]] = []
    for batch in _batched(texts, BATCH_SIZE):
        fatal: Exception | None = None
        try:
            batch_vectors = _call(batch, task_type)
        except Exception as exc:  # noqa: BLE001 - dicoba ulang per teks di bawah
            batch_vectors = []
            fatal = exc

        # Batch dianggap sah hanya kalau jumlah vector-nya sama dengan
        # jumlah teks. Kalau tidak, ulangi satu per satu.
        if len(batch_vectors) != len(batch) or not all(batch_vectors):
            if fatal is not None and not errors.is_retryable(fatal):
                # Rate limit atau gangguan sementara sudah habis jatah ulangnya,
                # atau errornya permanen: tidak ada gunanya mengulang per teks.
                raise EmbeddingError(
                    errors.friendly_message(fatal, "Embedding gagal dibuat.")
                ) from fatal
            batch_vectors = []
            for text in batch:
                try:
                    single = _call([text], task_type)
                except Exception as exc:
                    raise EmbeddingError(
                        errors.friendly_message(exc, "Embedding gagal dibuat.")
                    ) from exc
                if not single or not single[0]:
                    raise EmbeddingError("API embedding tidak mengembalikan vector.")
                batch_vectors.append(single[0])

        vectors.extend(normalize(vector) for vector in batch_vectors)

    return vectors


def embed_query(question: str) -> list[float]:
    """Embedding untuk pertanyaan pengguna (task type khusus pencarian)."""
    vectors = embed_texts([question], task_type=TASK_QUERY)
    if not vectors:
        raise EmbeddingError("Pertanyaan gagal di-embedding.")
    return vectors[0]
