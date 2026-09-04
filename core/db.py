"""Lapisan penyimpanan SQLite untuk history chat dan metadata percakapan.

Memakai modul ``sqlite3`` bawaan Python. Setiap operasi membuka koneksi
singkat lewat context manager sehingga aman dipakai dari beberapa thread
yang dibuat Streamlit saat rerun.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from core.config import get_settings

DEFAULT_TITLE = "Percakapan baru"

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT    NOT NULL DEFAULT 'Percakapan baru',
    created_at  TEXT    NOT NULL,
    updated_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT    NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content         TEXT    NOT NULL,
    created_at      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages (conversation_id, id);

CREATE INDEX IF NOT EXISTS idx_conversations_updated
    ON conversations (updated_at DESC);

CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT    NOT NULL,
    filetype    TEXT    NOT NULL,
    filesize    INTEGER NOT NULL,
    char_count  INTEGER NOT NULL DEFAULT 0,
    status      TEXT    NOT NULL,
    uploaded_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS document_text (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content     TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_document_text_document
    ON document_text (document_id);

CREATE INDEX IF NOT EXISTS idx_documents_uploaded
    ON documents (uploaded_at DESC);
"""


def _now() -> str:
    """Timestamp UTC ISO-8601 (urut secara leksikografis)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def db_path() -> Path:
    return get_settings().db_path


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Buka koneksi SQLite dengan foreign key aktif dan commit otomatis."""
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Buat tabel dan index kalau belum ada."""
    with connect() as conn:
        conn.executescript(SCHEMA)


# --------------------------------------------------------------------------- #
# Conversations
# --------------------------------------------------------------------------- #
def create_conversation(title: str = DEFAULT_TITLE) -> int:
    """Buat percakapan baru dan kembalikan id-nya."""
    stamp = _now()
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO conversations (title, created_at, updated_at) VALUES (?, ?, ?)",
            (title.strip() or DEFAULT_TITLE, stamp, stamp),
        )
        return int(cur.lastrowid)


def list_conversations(limit: int = 200) -> list[dict[str, Any]]:
    """Daftar percakapan, yang paling baru diperbarui ada di atas."""
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT c.id,
                   c.title,
                   c.created_at,
                   c.updated_at,
                   (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id)
                       AS message_count
            FROM conversations c
            ORDER BY c.updated_at DESC, c.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_conversation(conversation_id: int) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
    return dict(row) if row else None


def rename_conversation(conversation_id: int, title: str) -> None:
    title = title.strip()
    if not title:
        return
    with connect() as conn:
        conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, _now(), conversation_id),
        )


def touch_conversation(conversation_id: int) -> None:
    """Tandai percakapan sebagai baru saja dipakai."""
    with connect() as conn:
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (_now(), conversation_id),
        )


def delete_conversation(conversation_id: int) -> None:
    """Hapus percakapan beserta seluruh pesannya."""
    with connect() as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))


def delete_empty_conversations(except_id: int | None = None) -> None:
    """Bersihkan percakapan kosong supaya sidebar tidak penuh 'Percakapan baru'."""
    with connect() as conn:
        conn.execute(
            """
            DELETE FROM conversations
            WHERE id != COALESCE(?, -1)
              AND id NOT IN (SELECT DISTINCT conversation_id FROM messages)
            """,
            (except_id,),
        )


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #
def add_message(conversation_id: int, role: str, content: str) -> int:
    """Simpan satu pesan dan perbarui ``updated_at`` percakapannya."""
    if role not in ("user", "assistant", "system"):
        raise ValueError(f"role tidak dikenal: {role!r}")

    stamp = _now()
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO messages (conversation_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, role, content, stamp),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (stamp, conversation_id),
        )
        return int(cur.lastrowid)


def list_messages(conversation_id: int) -> list[dict[str, Any]]:
    """Seluruh pesan pada satu percakapan, urut dari yang paling lama."""
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id, conversation_id, role, content, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def count_messages(conversation_id: int, role: str | None = None) -> int:
    sql = "SELECT COUNT(*) FROM messages WHERE conversation_id = ?"
    params: list[Any] = [conversation_id]
    if role:
        sql += " AND role = ?"
        params.append(role)
    with connect() as conn:
        return int(conn.execute(sql, params).fetchone()[0])


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #
def save_document(
    filename: str,
    filetype: str,
    filesize: int,
    status: str,
    text: str = "",
) -> int:
    """Simpan metadata dokumen beserta teks hasil ekstraksinya.

    Keduanya ditulis dalam satu transaksi supaya tidak pernah ada baris
    ``documents`` yang teksnya menggantung setengah jalan.
    """
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO documents (filename, filetype, filesize, char_count, status, uploaded_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (filename, filetype, filesize, len(text), status, _now()),
        )
        document_id = int(cur.lastrowid)
        if text:
            conn.execute(
                "INSERT INTO document_text (document_id, content) VALUES (?, ?)",
                (document_id, text),
            )
        return document_id


def list_documents() -> list[dict[str, Any]]:
    """Daftar dokumen, yang paling baru diunggah ada di atas."""
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id, filename, filetype, filesize, char_count, status, uploaded_at
            FROM documents
            ORDER BY uploaded_at DESC, id DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def count_documents() -> int:
    with connect() as conn:
        return int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])


def get_document_text(document_id: int) -> str:
    """Gabungan teks milik satu dokumen (string kosong kalau tidak ada)."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT content FROM document_text WHERE document_id = ? ORDER BY id",
            (document_id,),
        ).fetchall()
    return "\n".join(row["content"] for row in rows)


def delete_document(document_id: int) -> None:
    """Hapus dokumen beserta seluruh data turunannya."""
    with connect() as conn:
        conn.execute("DELETE FROM document_text WHERE document_id = ?", (document_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
