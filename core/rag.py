"""Perekat antara chunking, embedding, dan vector store.

Dua alur yang dilayani modul ini:

* **Saat upload** — teks → chunking → embedding tiap chunk → simpan ke Qdrant.
* **Saat bertanya** — pertanyaan → embedding → cari chunk termirip → susun
  jadi context berbentuk teks biasa.

Yang dikirim ke model bahasa selalu **teks** chunk-nya, bukan vector.
Embedding hanya dipakai untuk pencarian.
"""

from __future__ import annotations

from typing import Callable

from core import embeddings, vectorstore
from core.chunking import chunk_text
from core.config import TOP_K, get_settings
from core.vectorstore import SearchHit

ProgressFn = Callable[[str], None]


def index_document(
    document_id: int,
    filename: str,
    text: str,
    *,
    progress: ProgressFn | None = None,
) -> int:
    """Pecah teks jadi chunk, embed satu per satu, lalu simpan ke Qdrant.

    Mengembalikan jumlah chunk yang terindeks. Melempar ``EmbeddingError``
    atau ``VectorStoreError`` kalau gagal — pemanggil yang memutuskan cara
    menyampaikannya ke pengguna.
    """

    def step(message: str) -> None:
        if progress is not None:
            progress(message)

    chunks = chunk_text(text)
    if not chunks:
        return 0

    step(f"Memecah teks jadi {len(chunks)} chunk…")
    step(f"Membuat embedding {len(chunks)} chunk…")
    vectors = embeddings.embed_texts(chunks, task_type=embeddings.TASK_DOCUMENT)

    step(f"Menyimpan {len(chunks)} vector ke Qdrant…")
    return vectorstore.upsert_chunks(document_id, filename, chunks, vectors)


def remove_document(document_id: int) -> None:
    """Hapus chunk milik dokumen dari Qdrant (aman dipanggil walau belum ada)."""
    try:
        vectorstore.delete_document(document_id)
    except Exception:
        # Indeks yang belum terbentuk bukan alasan untuk menggagalkan penghapusan.
        pass


def retrieve(question: str, top_k: int = TOP_K) -> list[SearchHit]:
    """Cari ``top_k`` chunk yang paling mirip dengan pertanyaan."""
    if not question.strip() or not vectorstore.has_index():
        return []
    vector = embeddings.embed_query(question)
    return vectorstore.search(vector, limit=top_k)


def build_context(hits: list[SearchHit]) -> str:
    """Susun chunk terpilih menjadi blok teks untuk dilampirkan ke prompt."""
    blocks = []
    for number, hit in enumerate(hits, start=1):
        blocks.append(
            f"[{number}] Sumber: {hit.filename} (bagian {hit.chunk_index + 1})\n{hit.text}"
        )
    return "\n\n".join(blocks)


def is_available() -> bool:
    """True kalau RAG bisa dipakai: API key embedding ada dan indeks terisi."""
    return get_settings().rag_available and vectorstore.has_index()
