"""Pemecahan teks panjang menjadi chunk untuk diindeks.

Aturan: ukuran target ~800 karakter dengan overlap ~150 karakter, dipotong
di batas kalimat kalau memungkinkan, dan tidak pernah memotong di tengah kata.
"""

from __future__ import annotations

import re

from core.config import CHUNK_OVERLAP, CHUNK_SIZE

# Pisah setelah tanda akhir kalimat, atau di baris kosong / awal baris "# ".
_SENTENCE_BREAK = re.compile(r"(?<=[.!?…])\s+|\n{2,}|\n(?=#\s)")


def split_sentences(text: str) -> list[str]:
    """Pecah teks menjadi kalimat (atau baris) yang sudah dirapikan."""
    return [part.strip() for part in _SENTENCE_BREAK.split(text) if part and part.strip()]


def split_long_sentence(sentence: str, size: int) -> list[str]:
    """Pecah kalimat yang lebih panjang dari ``size`` di batas kata.

    Dipakai untuk kasus seperti baris tabel spreadsheet yang tidak punya
    tanda baca akhir kalimat sama sekali.
    """
    pieces: list[str] = []
    current = ""
    for word in sentence.split(" "):
        # Satu token yang lebih panjang dari size tidak punya batas kata yang
        # bisa dipakai (mis. base64 atau URL panjang). Ini satu-satunya kasus
        # yang terpaksa dipotong keras.
        while len(word) > size:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:size])
            word = word[size:]
        if not word:
            continue
        if current and len(current) + 1 + len(word) > size:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        pieces.append(current)
    return pieces


def chunk_text(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Pecah ``text`` menjadi daftar chunk yang saling menimpa sebagian.

    Chunk disusun dari kalimat utuh: kalimat ditumpuk sampai mendekati
    ``size``, lalu beberapa kalimat terakhir dibawa ke chunk berikutnya
    sebagai overlap supaya konteks di perbatasan tidak hilang.
    """
    text = (text or "").strip()
    if not text:
        return []
    if overlap >= size:
        raise ValueError("overlap harus lebih kecil dari size")

    # Kalimat yang kepanjangan dipecah pada ``size - overlap``, bukan ``size``,
    # supaya jatah overlap selalu muat dan chunk tidak pernah melewati target.
    room = size - overlap
    sentences: list[str] = []
    for sentence in split_sentences(text):
        if len(sentence) > room:
            sentences.extend(split_long_sentence(sentence, room))
        else:
            sentences.append(sentence)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        if current and current_len + 1 + len(sentence) > size:
            chunks.append(" ".join(current))
            tail, tail_len = _overlap_tail(current, overlap)
            # Overlap dilewati kalau justru membuat chunk berikutnya
            # melewati target ukuran.
            if tail_len + len(sentence) + 1 > size:
                tail, tail_len = [], 0
            current, current_len = tail, tail_len
        current.append(sentence)
        current_len += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))

    return [chunk for chunk in chunks if chunk.strip()]


def _overlap_tail(sentences: list[str], overlap: int) -> tuple[list[str], int]:
    """Ambil kalimat-kalimat terakhir yang totalnya masih di bawah ``overlap``."""
    tail: list[str] = []
    tail_len = 0
    for sentence in reversed(sentences):
        if tail and tail_len + len(sentence) + 1 > overlap:
            break
        if not tail and len(sentence) > overlap:
            # Kalimat terakhir sendirian sudah lebih panjang dari jatah
            # overlap: ambil ekornya saja, dipotong di batas kata terdekat.
            snippet = sentence[-overlap:]
            space = snippet.find(" ")
            snippet = snippet[space + 1 :] if space != -1 else snippet
            return ([snippet], len(snippet) + 1) if snippet else ([], 0)
        tail.insert(0, sentence)
        tail_len += len(sentence) + 1
    # Jangan bawa seluruh chunk sebelumnya sebagai overlap.
    if len(tail) == len(sentences) and len(sentences) > 1:
        tail = tail[1:]
        tail_len -= len(sentences[0]) + 1
    return tail, max(tail_len, 0)
