"""Penyimpanan vector memakai Qdrant local mode (on-disk, tanpa server).

Client dibuat dengan ``QdrantClient(path=...)`` sehingga datanya tersimpan
sebagai file di ``./data/qdrant``. Mode ini mengunci folder penyimpanan untuk
satu proses, jadi client-nya dipakai bersama sebagai singleton — Streamlit
menjalankan ulang script-nya berkali-kali dan membuat client baru setiap
rerun akan langsung bentrok dengan lock tersebut.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models

from core.config import get_settings

# Namespace tetap supaya id point bisa dihitung ulang dari (document_id, chunk_index).
_POINT_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

_client: QdrantClient | None = None
_lock = threading.Lock()


class VectorStoreError(RuntimeError):
    """Operasi ke Qdrant gagal."""


@dataclass(frozen=True)
class SearchHit:
    """Satu chunk hasil pencarian."""

    document_id: int
    filename: str
    chunk_index: int
    text: str
    score: float


def get_client() -> QdrantClient:
    """Client Qdrant lokal (singleton, dibuat sekali per proses)."""
    global _client
    with _lock:
        if _client is None:
            settings = get_settings()
            settings.qdrant_path.parent.mkdir(parents=True, exist_ok=True)
            _client = QdrantClient(path=str(settings.qdrant_path))
        return _client


def ensure_collection(dimension: int) -> None:
    """Buat collection kalau belum ada, memakai dimensi vector sebenarnya.

    Ukuran diambil dari panjang vector yang benar-benar dikembalikan API
    embedding, bukan dari angka di konfigurasi, supaya collection tidak
    pernah dibuat dengan ukuran yang salah.
    """
    client = get_client()
    name = get_settings().collection

    if client.collection_exists(name):
        existing = client.get_collection(name).config.params.vectors.size
        if existing != dimension:
            raise VectorStoreError(
                f"Collection '{name}' sudah dibuat untuk vector {existing} dimensi, "
                f"sedangkan model sekarang menghasilkan {dimension} dimensi. "
                "Hapus folder data/qdrant lalu unggah ulang dokumennya."
            )
        return

    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
    )


def _point_id(document_id: int, chunk_index: int) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{document_id}:{chunk_index}"))


def upsert_chunks(
    document_id: int,
    filename: str,
    chunks: list[str],
    vectors: list[list[float]],
) -> int:
    """Simpan chunk beserta vector-nya. Mengembalikan jumlah yang tersimpan."""
    if not chunks:
        return 0
    if len(chunks) != len(vectors):
        raise VectorStoreError("Jumlah chunk dan vector tidak sama.")

    ensure_collection(len(vectors[0]))
    points = [
        models.PointStruct(
            id=_point_id(document_id, index),
            vector=vector,
            payload={
                "document_id": document_id,
                "filename": filename,
                "chunk_index": index,
                "text": chunk,
            },
        )
        for index, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    get_client().upsert(collection_name=get_settings().collection, points=points)
    return len(points)


def search(vector: list[float], limit: int = 5) -> list[SearchHit]:
    """Cari chunk termirip. Kembalikan daftar kosong kalau indeks belum ada."""
    client = get_client()
    name = get_settings().collection
    if not client.collection_exists(name):
        return []

    response = client.query_points(
        collection_name=name, query=vector, limit=limit, with_payload=True
    )
    hits: list[SearchHit] = []
    for point in response.points:
        payload: dict[str, Any] = point.payload or {}
        hits.append(
            SearchHit(
                document_id=int(payload.get("document_id", 0)),
                filename=str(payload.get("filename", "")),
                chunk_index=int(payload.get("chunk_index", 0)),
                text=str(payload.get("text", "")),
                score=float(point.score),
            )
        )
    return hits


def delete_document(document_id: int) -> None:
    """Hapus seluruh chunk milik satu dokumen."""
    client = get_client()
    name = get_settings().collection
    if not client.collection_exists(name):
        return
    client.delete(
        collection_name=name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=document_id)
                    )
                ]
            )
        ),
    )


def count_chunks(document_id: int | None = None) -> int:
    """Jumlah chunk seluruhnya, atau milik satu dokumen saja."""
    client = get_client()
    name = get_settings().collection
    if not client.collection_exists(name):
        return 0

    count_filter = None
    if document_id is not None:
        count_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id", match=models.MatchValue(value=document_id)
                )
            ]
        )
    return int(client.count(collection_name=name, count_filter=count_filter).count)


def has_index() -> bool:
    """True kalau ada chunk yang siap dicari."""
    try:
        return count_chunks() > 0
    except Exception:
        return False
