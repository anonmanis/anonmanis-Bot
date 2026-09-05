"""anonmanis-Chat, chatbot AI berbasis Streamlit + Groq.

Entry point aplikasi. Tahap ini fokus ke chat dasar: percakapan tersimpan
di SQLite, jawaban di-stream dari Groq, judul dibuat otomatis.
"""

from __future__ import annotations

from html import escape
from typing import Iterable, Iterator

import streamlit as st

from core import db, errors, ingest, llm, rag, vectorstore
from core.config import (
    APP_NAME,
    APP_TAGLINE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    TOP_K,
    get_settings,
)

USER_AVATAR = ":material/person:"
BOT_AVATAR = ":material/auto_awesome:"

# Ikon garis 16px untuk tiap tipe file; mewarisi warna aksen lewat currentColor.
_SVG = '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">{}</svg>'
_PAGE = '<path d="M9.2 1.8H4.4A1.4 1.4 0 0 0 3 3.2v9.6a1.4 1.4 0 0 0 1.4 1.4h7.2a1.4 1.4 0 0 0 1.4-1.4V5.6z"/><path d="M9.2 1.8v3.8H13"/>'

FILE_ICONS = {
    "pdf": _SVG.format(_PAGE + '<path d="M5.4 10.6h5.2"/>'),
    "docx": _SVG.format(_PAGE + '<path d="M5.4 8.6h5.2M5.4 11.2h3.4"/>'),
    "xlsx": _SVG.format(
        '<rect x="2.4" y="2.4" width="11.2" height="11.2" rx="1.6"/>'
        '<path d="M2.4 6.4h11.2M2.4 9.6h11.2M6.4 2.4v11.2"/>'
    ),
    "pptx": _SVG.format(
        '<rect x="2" y="2.8" width="12" height="8.4" rx="1.4"/>'
        '<path d="M8 11.2v2.4M5.6 13.6h4.8"/>'
    ),
    "image": _SVG.format(
        '<rect x="2.2" y="3" width="11.6" height="10" rx="1.8"/>'
        '<circle cx="6" cy="6.4" r="1.1"/>'
        '<path d="M3 11.6l3.1-3 2.4 2.2 2.1-2 2.4 2.4"/>'
    ),
}

# Label pendek untuk status dokumen yang bukan "processed"
STATUS_LABELS = {
    ingest.STATUS_PENDING_EXTRACTION: "menunggu tahap 4",
    ingest.STATUS_NO_TEXT: "tanpa teks",
    ingest.STATUS_FAILED: "gagal",
    ingest.STATUS_PROCESSED: "belum terindeks",
}

SUGGESTIONS = [
    ("Jelaskan konsep", "Jelaskan apa itu Retrieval Augmented Generation dengan analogi sederhana."),
    ("Bantu menulis", "Tuliskan email singkat dan sopan untuk menunda meeting ke minggu depan."),
    ("Tulis kode", "Buat fungsi Python untuk membaca file CSV besar secara batch, lengkap dengan contoh pemakaian."),
]

# --------------------------------------------------------------------------- #
# Tampilan: font Google Fonts + penyesuaian komponen bawaan Streamlit
# --------------------------------------------------------------------------- #
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --ac-bg:        #0B0D12;
    --ac-surface:   #12151E;
    --ac-elevated:  #171B27;
    --ac-border:    #242A3A;
    --ac-text:      #E9EBF2;
    --ac-muted:     #8A92A6;
    --ac-accent:    #7C5CFF;
    --ac-accent-deep: #4B31D6;
    --ac-accent-soft: #9B85FF;
    --ac-on-accent: #FFFFFF;
    --ac-danger:    #FF8B8B;

    /* Tiga tingkat transparansi aksen dipakai konsisten:
       dim untuk isian, line untuk garis tepi, edge untuk tepi aktif. */
    --ac-accent-dim:  rgba(124, 92, 255, 0.14);
    --ac-accent-line: rgba(124, 92, 255, 0.30);
    --ac-accent-edge: rgba(124, 92, 255, 0.55);

    /* Skala tipografi */
    --ac-fs-2xs:  0.68rem;
    --ac-fs-xs:   0.72rem;
    --ac-fs-sm:   0.78rem;
    --ac-fs-base: 0.83rem;
    --ac-fs-md:   0.95rem;
    --ac-fs-lg:   1.05rem;

    /* Skala jarak */
    --ac-gap-xs: 0.35rem;
    --ac-gap-sm: 0.55rem;
    --ac-gap-md: 0.85rem;
    --ac-gap-lg: 1.4rem;

    --ac-radius:    18px;
    --ac-radius-sm: 11px;
    --ac-chat-width: 46rem;
}

html, body, .stApp, [data-testid="stAppViewContainer"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    font-feature-settings: 'cv02', 'cv03', 'ss01';
}

/* --- Sembunyikan chrome bawaan: menu, footer, badge Deploy, status --- */
#MainMenu,
footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stAppDeployButton"],
[data-testid="stMainMenu"] {
    display: none !important;
}
[data-testid="stHeader"] {
    background: transparent !important;
    height: 0 !important;
}

/* --- Lebar area baca dibatasi walau layout wide --- */
.stMainBlockContainer,
[data-testid="stMainBlockContainer"] {
    max-width: var(--ac-chat-width) !important;
    padding-top: 3.25rem !important;
    padding-bottom: 4rem !important;
}
[data-testid="stBottomBlockContainer"] {
    max-width: calc(var(--ac-chat-width) + 2rem) !important;
    padding-bottom: 1.25rem !important;
}
[data-testid="stBottom"] > div {
    background: linear-gradient(to bottom, rgba(11, 13, 18, 0), var(--ac-bg) 22%);
}

/* --- Bubble chat --- */
[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 0 !important;
    margin-bottom: 1.5rem;
    gap: 0.85rem;
}
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
    background: var(--ac-surface);
    border: 1px solid var(--ac-border);
    border-radius: var(--ac-radius);
    padding: 0.95rem 1.2rem;
    line-height: 1.75;
    font-size: var(--ac-fs-md);
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.28);
    overflow-x: auto;
}
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] p:last-child {
    margin-bottom: 0;
}
/* Pesan pengguna diberi warna aksen yang lembut */
[class*="st-key-msg-user-"] [data-testid="stChatMessageContent"] {
    background: var(--ac-accent-dim);
    border-color: var(--ac-accent-line);
}
[class*="st-key-msg-"] { gap: 0 !important; }

/* --- Avatar --- */
[data-testid="stChatMessageAvatarCustom"] {
    width: 2.15rem !important;
    height: 2.15rem !important;
    border-radius: var(--ac-radius-sm) !important;
    border: 1px solid var(--ac-border);
    flex-shrink: 0;
}
[class*="st-key-msg-assistant-"] [data-testid="stChatMessageAvatarCustom"] {
    background: linear-gradient(140deg, var(--ac-accent), var(--ac-accent-deep));
    border-color: var(--ac-accent-edge);
    color: var(--ac-on-accent);
    box-shadow: 0 0 0 3px var(--ac-accent-dim);
}
[class*="st-key-msg-user-"] [data-testid="stChatMessageAvatarCustom"] {
    background: var(--ac-elevated);
    color: var(--ac-muted);
}
[data-testid="stChatMessageAvatarCustom"] [data-testid="stIconMaterial"] {
    font-size: 1.15rem;
}

/* --- Kotak input chat --- */
[data-testid="stChatInput"] {
    background: var(--ac-surface);
    border: 1px solid var(--ac-border);
    border-radius: 16px;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.4);
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--ac-accent-edge);
    box-shadow: 0 0 0 3px var(--ac-accent-dim), 0 8px 28px rgba(0, 0, 0, 0.4);
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--ac-muted); }

/* --- Sidebar --- */
/* Sidebar dilebarkan dari default 300px: sekarang menampung daftar dokumen
   dengan nama, tipe, ukuran, jumlah karakter, dan jam unggah dalam satu baris. */
[data-testid="stSidebar"] {
    border-right: 1px solid var(--ac-border);
    width: 21rem !important;
}
[data-testid="stSidebar"] > div { width: 21rem; }
[data-testid="stSidebarContent"] { padding: 1.4rem 0.9rem 1rem; }
[data-testid="stSidebar"] hr {
    margin: 0.9rem 0;
    border-color: var(--ac-border);
    opacity: 0.75;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.35rem; }

.ac-brand {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.1rem 0.35rem 0.35rem;
}
.ac-brand-mark {
    display: grid;
    place-items: center;
    width: 2.35rem;
    height: 2.35rem;
    border-radius: 13px;
    background: linear-gradient(140deg, var(--ac-accent), var(--ac-accent-deep));
    color: var(--ac-on-accent);
    font-size: 1.1rem;
    font-weight: 700;
    box-shadow: 0 4px 16px var(--ac-accent-line);
}
.ac-brand-name {
    font-size: var(--ac-fs-lg);
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.ac-brand-sub { font-size: var(--ac-fs-sm); color: var(--ac-muted); }

.ac-section {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.25rem 0.5rem 0.4rem;
    font-size: var(--ac-fs-2xs);
    font-weight: 700;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    color: var(--ac-muted);
}
.ac-section span.ac-count {
    letter-spacing: 0;
    font-weight: 600;
    color: var(--ac-muted);
    opacity: 0.75;
}
.ac-hint {
    padding: 0.7rem 0.6rem;
    font-size: var(--ac-fs-sm);
    line-height: 1.6;
    color: var(--ac-muted);
}
.ac-foot {
    padding: 0.55rem 0.6rem 0;
    font-size: var(--ac-fs-xs);
    color: var(--ac-muted);
    line-height: 1.7;
}
.ac-foot code {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: var(--ac-fs-xs);
    color: var(--ac-accent);
    background: var(--ac-accent-dim);
    padding: 0.1rem 0.35rem;
    border-radius: 6px;
}

/* Tombol daftar percakapan: rata kiri, sudut lembut, hover halus */
[data-testid="stSidebar"] [data-testid="stButton"] button {
    justify-content: flex-start !important;
    font-weight: 500;
    padding: 0.45rem 0.7rem;
    border-radius: var(--ac-radius-sm) !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] button [data-testid="stMarkdownContainer"] {
    width: 100%;
    text-align: left;
}
[data-testid="stSidebar"] [data-testid="stButton"] button p {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 100%;
}
[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"]:hover {
    background: var(--ac-surface);
    color: var(--ac-text);
}
[data-testid="stSidebar"] .st-key-new_chat { margin-top: 0.55rem; }
[data-testid="stSidebar"] .st-key-new_chat button {
    justify-content: center !important;
    border-radius: 999px !important;
    font-weight: 600;
    padding: 0.55rem 0.9rem;
    box-shadow: 0 4px 14px var(--ac-accent-line);
}
[data-testid="stSidebar"] [class*="st-key-del_"] button,
[data-testid="stSidebar"] [class*="st-key-confirm_"] button,
[data-testid="stSidebar"] [class*="st-key-cancel_"] button {
    justify-content: center !important;
    padding: 0.45rem !important;
    color: var(--ac-muted);
}
[data-testid="stSidebar"] [class*="st-key-del_"] button:hover,
[data-testid="stSidebar"] [class*="st-key-confirm_"] button:hover {
    color: var(--ac-danger);
    background: rgba(255, 139, 139, 0.12);
}

/* --- Empty state --- */
.st-key-empty_wrap {
    display: flex;
    flex-direction: column;
    justify-content: center;
    min-height: 66vh;
}
.ac-empty { padding: 0 0 1.75rem; text-align: center; }
.ac-empty-mark {
    display: inline-grid;
    place-items: center;
    width: 3.6rem;
    height: 3.6rem;
    margin-bottom: 1.4rem;
    border-radius: 20px;
    background: linear-gradient(140deg, var(--ac-accent), var(--ac-accent-deep));
    color: var(--ac-on-accent);
    font-size: 1.65rem;
    font-weight: 700;
    box-shadow: 0 10px 34px var(--ac-accent-line);
}
.ac-empty h1 {
    margin: 0 0 0.6rem;
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.035em;
}
.ac-empty p {
    max-width: 30rem;
    margin: 0 auto;
    color: var(--ac-muted);
    font-size: var(--ac-fs-md);
    line-height: 1.75;
}
.ac-empty-label {
    margin: 1.6rem 0 0.9rem;
    font-size: var(--ac-fs-2xs);
    font-weight: 700;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    color: var(--ac-muted);
    text-align: center;
}
.st-key-suggestions .stButton > button {
    width: 100%;
    min-height: 3.1rem;
    border-radius: 14px;
    background: var(--ac-surface);
    border: 1px solid var(--ac-border);
    color: var(--ac-text);
    font-size: var(--ac-fs-base);
    font-weight: 500;
    line-height: 1.4;
    transition: border-color 0.15s ease, transform 0.15s ease;
}
.st-key-suggestions .stButton > button:hover {
    border-color: var(--ac-accent-edge);
    background: var(--ac-elevated);
    color: var(--ac-text);
    transform: translateY(-1px);
}


/* --- Toggle RAG di sidebar --- */
.st-key-rag_toggle { padding: 0.5rem 0.6rem 0.1rem; }
.st-key-rag_toggle label p { font-size: var(--ac-fs-sm) !important; font-weight: 500; }

/* --- Expander "Sumber" di bawah jawaban --- */
[data-testid="stMainBlockContainer"] [data-testid="stExpander"] {
    /* 3rem = lebar avatar + jarak, supaya sejajar dengan bubble jawaban */
    margin: -0.6rem 0 0 3rem;
    width: calc(100% - 3rem) !important;
}
[data-testid="stMainBlockContainer"] [data-testid="stExpander"] details {
    background: transparent;
    border: 1px solid var(--ac-border);
    border-radius: 13px;
}
[data-testid="stMainBlockContainer"] [data-testid="stExpander"] summary {
    font-size: var(--ac-fs-sm);
    font-weight: 600;
    color: var(--ac-muted);
}
[data-testid="stMainBlockContainer"] [data-testid="stExpander"] summary:hover {
    color: var(--ac-accent);
}

.ac-src { padding: 0.55rem 0 0.7rem; border-top: 1px solid var(--ac-border); }
.ac-src:first-child { border-top: none; padding-top: 0.1rem; }
.ac-src-head {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.4rem;
}
.ac-src-rank {
    display: grid;
    place-items: center;
    flex-shrink: 0;
    width: 1.25rem;
    height: 1.25rem;
    border-radius: 6px;
    background: var(--ac-accent-dim);
    border: 1px solid var(--ac-accent-line);
    color: var(--ac-accent);
    font-size: var(--ac-fs-2xs);
    font-weight: 700;
}
.ac-src-file {
    font-size: var(--ac-fs-sm);
    font-weight: 600;
    color: var(--ac-text);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.ac-src-part { font-size: var(--ac-fs-xs); color: var(--ac-muted); }
.ac-src-score {
    margin-left: auto;
    flex-shrink: 0;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: var(--ac-fs-xs);
    color: var(--ac-accent);
    background: var(--ac-accent-dim);
    padding: 0.1rem 0.4rem;
    border-radius: 6px;
}
.ac-src-bar {
    height: 3px;
    margin-bottom: 0.5rem;
    border-radius: 99px;
    background: var(--ac-elevated);
    overflow: hidden;
}
.ac-src-bar > span {
    display: block;
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, var(--ac-accent), var(--ac-accent-soft));
}
.ac-src-text {
    max-height: 7rem;
    overflow-y: auto;
    padding: 0.55rem 0.7rem;
    border-radius: 10px;
    background: var(--ac-surface);
    border: 1px solid var(--ac-border);
    font-size: var(--ac-fs-sm);
    line-height: 1.65;
    color: var(--ac-muted);
    white-space: pre-wrap;
    word-break: break-word;
}

/* --- Pencarian dan paginasi daftar dokumen --- */
.st-key-doc_search { padding: 0.15rem 0.1rem 0.35rem; }
.st-key-doc_search [data-testid="stTextInput"] input {
    background: var(--ac-surface);
    border-radius: var(--ac-radius-sm);
    font-size: var(--ac-fs-sm);
    padding: 0.4rem 0.7rem;
}
.st-key-doc_search [data-testid="stTextInput"] input::placeholder {
    color: var(--ac-muted);
}
.st-key-doc_search [data-testid="stTextInput"] > div:focus-within {
    border-color: var(--ac-accent-edge);
}

.ac-pager {
    display: flex;
    flex-direction: column;
    align-items: center;
    line-height: 1.35;
    font-size: var(--ac-fs-xs);
    color: var(--ac-text);
}
.ac-pager span {
    font-size: var(--ac-fs-2xs);
    color: var(--ac-muted);
}
[data-testid="stSidebar"] [class*="st-key-doc_prev"] button,
[data-testid="stSidebar"] [class*="st-key-doc_next"] button {
    justify-content: center !important;
    padding: 0.35rem !important;
    color: var(--ac-muted);
}
[data-testid="stSidebar"] [class*="st-key-doc_prev"] button:hover:not(:disabled),
[data-testid="stSidebar"] [class*="st-key-doc_next"] button:hover:not(:disabled) {
    color: var(--ac-accent);
    background: var(--ac-accent-dim);
}

/* --- Status pemrosesan lampiran di area chat --- */
.st-key-chat-upload [data-testid="stExpander"] {
    margin: 0 0 var(--ac-gap-md) 3rem;
    width: calc(100% - 3rem) !important;
}
.st-key-chat-upload [data-testid="stExpander"] details {
    background: var(--ac-surface);
    border-color: var(--ac-border);
}
.st-key-chat-upload [data-testid="stMarkdown"] p {
    position: relative;
    font-size: var(--ac-fs-xs);
    line-height: 1.55;
    color: var(--ac-muted);
    margin: 0 0 0.3rem;
    padding-left: 0.85rem;
}
.st-key-chat-upload [data-testid="stMarkdown"] p::before {
    content: "";
    position: absolute;
    left: 0.15rem;
    top: 0.55em;
    width: 4px;
    height: 4px;
    border-radius: 99px;
    background: var(--ac-accent);
}

/* --- Panel "Tentang" --- */
.st-key-about [data-testid="stExpander"] { margin: 0; width: auto !important; }
.st-key-about [data-testid="stExpander"] details {
    background: var(--ac-surface);
    border: 1px solid var(--ac-border);
    border-radius: 12px;
}
.st-key-about [data-testid="stExpander"] summary {
    font-size: var(--ac-fs-sm);
    font-weight: 600;
    color: var(--ac-muted);
}
.ac-about p { margin: 0 0 var(--ac-gap-md); }
.ac-about-lead {
    font-size: var(--ac-fs-sm);
    line-height: 1.7;
    color: var(--ac-text);
}
.ac-about-flow {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.3rem;
    margin-bottom: var(--ac-gap-sm);
}
.ac-about-flow span {
    padding: 0.16rem 0.5rem;
    border-radius: 7px;
    background: var(--ac-accent-dim);
    border: 1px solid var(--ac-accent-line);
    color: var(--ac-text);
    font-size: var(--ac-fs-2xs);
    font-weight: 500;
    white-space: nowrap;
}
.ac-about-flow i {
    width: 0.55rem;
    height: 1px;
    background: var(--ac-border);
    flex-shrink: 0;
}
.ac-about-table {
    width: 100%;
    margin: var(--ac-gap-md) 0;
    border-collapse: collapse;
    font-size: var(--ac-fs-xs);
}
.ac-about-table td {
    padding: 0.28rem 0;
    border-bottom: 1px solid var(--ac-border);
    vertical-align: top;
}
.ac-about-table tr:last-child td { border-bottom: none; }
.ac-about-table td:first-child { color: var(--ac-muted); width: 42%; }
.ac-about-table code {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: var(--ac-fs-2xs);
    color: var(--ac-accent);
    background: var(--ac-accent-dim);
    padding: 0.05rem 0.3rem;
    border-radius: 5px;
    word-break: break-all;
}
.ac-about-foot {
    font-size: var(--ac-fs-xs) !important;
    line-height: 1.7;
    color: var(--ac-muted);
    margin-bottom: 0 !important;
}

/* --- Layar tablet: sidebar dipersempit, area chat ikut menyesuaikan --- */
@media (max-width: 1180px) {
    :root { --ac-chat-width: 40rem; }
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div { width: 18.5rem !important; }
}

@media (max-width: 900px) {
    :root {
        --ac-chat-width: 100%;
        --ac-radius: 15px;
    }
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div { width: 17.5rem !important; }
    .stMainBlockContainer,
    [data-testid="stMainBlockContainer"] {
        padding-left: 1.1rem !important;
        padding-right: 1.1rem !important;
        padding-top: 2.4rem !important;
    }
    [data-testid="stBottomBlockContainer"] {
        padding-left: 1.1rem !important;
        padding-right: 1.1rem !important;
    }
    /* Chip saran menumpuk dua baris, bukan tiga kolom sempit */
    .st-key-suggestions [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
    .st-key-suggestions [data-testid="stColumn"] {
        flex: 1 1 46% !important;
        min-width: 46% !important;
    }
    .ac-empty { padding-top: 1rem; }
    .ac-empty h1 { font-size: 1.55rem; }
    .ac-empty p { font-size: var(--ac-fs-md); }
    .st-key-empty_wrap { min-height: 52vh; }
    /* Expander sumber ikut melebar penuh di layar sempit */
    [data-testid="stMainBlockContainer"] [data-testid="stExpander"] {
        margin-left: 0;
        width: 100% !important;
    }
    .st-key-retry_wrap { margin-left: 0; }
}

@media (max-width: 640px) {
    [data-testid="stChatMessage"] { gap: 0.6rem; }
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
        padding: 0.8rem 0.95rem;
        font-size: var(--ac-fs-md);
    }
    .ac-src-head { flex-wrap: wrap; }
    .ac-src-score { margin-left: 0; }
}

/* --- Bagian dokumen di sidebar --- */
.st-key-uploader [data-testid="stFileUploaderDropzone"] {
    background: var(--ac-surface);
    border: 1px dashed var(--ac-border);
    border-radius: 14px;
    padding: 0.7rem 0.8rem;
    min-height: 0;
    transition: border-color 0.15s ease, background 0.15s ease;
}
.st-key-uploader [data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--ac-accent-edge);
    background: var(--ac-elevated);
}
/* Streamlit menulis plafon server (25MB) di dropzone. Batas sebenarnya 5 MB
   dan divalidasi di core/ingest.py, jadi teksnya diganti agar tidak
   menyesatkan: font-size 0 menyembunyikan teks asli, ::after menggantinya. */
.st-key-uploader [data-testid="stFileUploaderDropzoneInstructions"] span {
    font-size: 0;
    line-height: 1.5;
}
.st-key-uploader [data-testid="stFileUploaderDropzoneInstructions"] span::after {
    content: "PDF · DOCX · XLSX · PPTX · PNG · JPG";
    font-size: var(--ac-fs-xs);
    color: var(--ac-muted);
    white-space: normal;
}
.st-key-uploader [data-testid="stFileUploaderDropzone"] button {
    border-radius: 999px !important;
    font-size: var(--ac-fs-sm);
    padding: 0.25rem 0.75rem;
}
.st-key-uploader [data-testid="stFileUploaderFile"] { font-size: var(--ac-fs-sm); }

/* Baris info: jumlah dokumen tersimpan dan batas per sekali unggah */
.ac-quota { padding: 0.55rem 0.6rem 0.2rem; }
.ac-quota-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.3rem 0.5rem;
    font-size: var(--ac-fs-xs);
    color: var(--ac-muted);
    margin-bottom: 0.4rem;
}
.ac-quota-head strong { color: var(--ac-text); font-weight: 600; }
.ac-quota-head { margin-bottom: 0; }
.ac-quota-cap { font-size: var(--ac-fs-2xs); opacity: 0.8; }
/* Baris dokumen */
.ac-doc {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.38rem 0 0.38rem 0.45rem;
    min-width: 0;
}
.ac-doc-icon {
    display: grid;
    place-items: center;
    flex-shrink: 0;
    width: 1.7rem;
    height: 1.7rem;
    border-radius: 9px;
    background: var(--ac-accent-dim);
    border: 1px solid var(--ac-accent-line);
    color: var(--ac-accent);
}
.ac-doc-icon svg { width: 15px; height: 15px; display: block; }
.ac-doc-body { display: flex; flex-direction: column; min-width: 0; gap: 0.1rem; }
.ac-doc-name {
    font-size: var(--ac-fs-base);
    font-weight: 500;
    color: var(--ac-text);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.ac-doc-meta {
    font-size: var(--ac-fs-2xs);
    color: var(--ac-muted);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.ac-note {
    padding: 0.6rem 0.6rem 0;
    font-size: var(--ac-fs-2xs);
    line-height: 1.65;
    color: var(--ac-muted);
    opacity: 0.85;
}

/* Alert hasil unggah dibuat ringkas supaya muat di sidebar */
[data-testid="stSidebar"] [data-testid="stAlertContainer"] {
    padding: 0.55rem 0.7rem;
    font-size: var(--ac-fs-sm);
    line-height: 1.55;
    border-radius: 12px;
}

/* Chip file bawaan uploader disesuaikan dengan tema */
.st-key-uploader [data-testid="stFileChip"] {
    background: var(--ac-elevated) !important;
    border: 1px solid var(--ac-border) !important;
    border-radius: 10px;
    font-size: var(--ac-fs-sm);
}
.st-key-uploader [data-testid="stFileChip"] svg { color: var(--ac-accent); }

/* st.status: tiap tahap pemrosesan tampil sebagai langkah bertanda */
[data-testid="stSidebar"] [data-testid="stExpander"] details {
    border-radius: 12px;
    border-color: var(--ac-border);
    background: var(--ac-surface);
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    font-size: var(--ac-fs-sm);
    font-weight: 600;
}
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stMarkdown"] p {
    position: relative;
    font-size: var(--ac-fs-xs);
    line-height: 1.55;
    color: var(--ac-muted);
    margin: 0 0 0.3rem;
    padding-left: 0.85rem;
}
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stMarkdown"] p::before {
    content: "";
    position: absolute;
    left: 0.15rem;
    top: 0.55em;
    width: 4px;
    height: 4px;
    border-radius: 99px;
    background: var(--ac-accent);
}

/* --- Tombol "Coba lagi" saat jawaban gagal --- */
.st-key-retry_wrap { margin: 0.35rem 0 0 3rem; }
.st-key-retry_wrap button {
    border-radius: 999px !important;
    font-size: var(--ac-fs-base);
    padding: 0.3rem 0.9rem;
    color: var(--ac-muted);
}
.st-key-retry_wrap button:hover {
    color: var(--ac-text);
    border-color: var(--ac-accent-edge);
}

/* --- Header percakapan aktif --- */
.ac-thread-head {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    margin-bottom: 1.6rem;
    padding-bottom: 0.85rem;
    border-bottom: 1px solid var(--ac-border);
}
.ac-thread-title {
    font-size: var(--ac-fs-lg);
    font-weight: 700;
    letter-spacing: -0.02em;
}
.ac-thread-meta { font-size: var(--ac-fs-sm); color: var(--ac-muted); }

/* Scrollbar tipis biar konsisten dengan tema */
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: var(--ac-scroll);
    border-radius: 99px;
    border: 2px solid var(--ac-bg);
}
::-webkit-scrollbar-thumb:hover { background: var(--ac-scroll-hover); }
</style>
"""


# --------------------------------------------------------------------------- #
# Helper
# --------------------------------------------------------------------------- #
def tee(stream: Iterable[str], buffer: list[str]) -> Iterator[str]:
    """Teruskan potongan stream ke UI sambil menyimpannya ke ``buffer``.

    Berguna supaya jawaban parsial tetap tersimpan kalau koneksi terputus
    di tengah jalan.
    """
    for piece in stream:
        buffer.append(piece)
        yield piece


def active_conversation_id() -> int | None:
    return st.session_state.get("conversation_id")


def open_conversation(conversation_id: int | None) -> None:
    st.session_state.conversation_id = conversation_id
    st.session_state.pending_delete = None
    st.session_state.pending_doc_delete = None


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar(conversations: list[dict]) -> None:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="ac-brand">
                <div class="ac-brand-mark">aC</div>
                <div>
                    <div class="ac-brand-name">{APP_NAME}</div>
                    <div class="ac-brand-sub">{APP_TAGLINE}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(key="new_chat"):
            if st.button(
                "New Chat",
                type="primary",
                width="stretch",
                icon=":material/add:",
                help="Mulai percakapan baru",
            ):
                open_conversation(None)
                st.rerun()

        st.divider()
        render_documents_section()

        st.divider()
        st.markdown(
            f'<div class="ac-section">Riwayat'
            f'<span class="ac-count">{len(conversations)}</span></div>',
            unsafe_allow_html=True,
        )

        if not conversations:
            st.markdown(
                '<div class="ac-hint">Belum ada percakapan tersimpan. '
                "Kirim pesan pertamamu untuk memulai.</div>",
                unsafe_allow_html=True,
            )
        else:
            for conv in conversations:
                render_conversation_row(conv)

        st.divider()
        with st.container(key="about"):
            render_about()



# --------------------------------------------------------------------------- #
# Sidebar: dokumen
# --------------------------------------------------------------------------- #
def shorten(name: str, limit: int = 26) -> str:
    """Potong nama file yang terlalu panjang, ekstensinya tetap terlihat."""
    if len(name) <= limit:
        return name
    stem, dot, extension = name.rpartition(".")
    head = (stem or name)[: limit - len(extension) - 4]
    return f"{head}…{dot}{extension}" if dot else f"{name[: limit - 1]}…"


def process_uploads(files: list) -> tuple[list, str | None]:
    """Jalankan pipeline ingest untuk tiap file sambil menampilkan tahapannya.

    Mengembalikan ``(hasil, pesan_penolakan_batch)``. Kalau jumlah file dalam
    satu unggahan melebihi batas, tidak ada satu pun yang diproses supaya
    pengguna sendiri yang memilih file mana yang jadi dikirim.

    Pemanggil yang memutuskan kapan harus ``st.rerun()``, karena unggahan dari
    sidebar dan dari kotak chat punya kelanjutan yang berbeda.
    """
    batch_error = ingest.check_batch(len(files))
    if batch_error:
        return [], batch_error

    results = []
    for uploaded in files:
        label = shorten(uploaded.name, 22)
        with st.status(f"Memproses {label}", expanded=True) as status:
            result = ingest.ingest(uploaded.name, uploaded.getvalue(), progress=st.write)
            status.update(
                label=f"{label}, {'selesai' if result.ok else 'ditolak'}",
                state="complete" if result.ok else "error",
                expanded=False,
            )
        results.append(result)

    return results, None


def render_upload_results() -> None:
    """Hasil unggahan terakhir; hilang sendiri begitu ada interaksi lain."""
    batch_error = st.session_state.pop("upload_error", None)
    if batch_error:
        st.error(batch_error, icon=":material/block:")

    for result in st.session_state.pop("upload_results", []):
        name = shorten(result.filename, 22)
        if result.ok:
            st.success(f"**{name}** {result.message}", icon=":material/check_circle:")
        else:
            st.error(f"**{name}** ditolak. {result.message}", icon=":material/block:")


def render_limits(used: int) -> None:
    """Jumlah dokumen tersimpan beserta batas yang berlaku per sekali unggah."""
    label = f"<strong>{used}</strong> dokumen tersimpan" if used else "Belum ada dokumen"
    st.markdown(
        f"""
        <div class="ac-quota">
            <div class="ac-quota-head">
                <span>{label}</span>
                <span class="ac-quota-cap">maks {ingest.MAX_FILES_PER_BATCH} file/unggah
                    &middot; {ingest.MAX_FILE_SIZE_MB} MB/file</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_document_row(document: dict) -> None:
    """Satu baris dokumen: ikon, nama, metadata, dan tombol hapus berkonfirmasi."""
    document_id = document["id"]
    filetype = document["filetype"]
    icon = FILE_ICONS.get(filetype if filetype not in ingest.IMAGE_TYPES else "image", "")

    if st.session_state.get("pending_doc_delete") == document_id:
        label, confirm, cancel = st.columns(
            [0.6, 0.2, 0.2], gap="small", vertical_alignment="center"
        )
        label.markdown('<div class="ac-hint">Hapus dokumen ini?</div>', unsafe_allow_html=True)
        with confirm.container(key=f"confirmdoc_{document_id}"):
            if st.button(
                "", icon=":material/check:", key=f"confirmdoc_btn_{document_id}",
                width="stretch", help="Ya, hapus dokumen dan teksnya",
            ):
                with st.spinner("Menghapus dokumen dan chunk-nya…"):
                    ingest.delete_document(document_id)
                st.session_state.pending_doc_delete = None
                st.rerun()
        with cancel.container(key=f"canceldoc_{document_id}"):
            if st.button(
                "", icon=":material/close:", key=f"canceldoc_btn_{document_id}",
                width="stretch", help="Batal",
            ):
                st.session_state.pending_doc_delete = None
                st.rerun()
        return

    detail = ingest.type_label(filetype) + " · " + ingest.human_size(document["filesize"])
    if document["status"] == ingest.STATUS_INDEXED:
        chunks = vectorstore.count_chunks(document_id)
        detail += f" · {chunks} chunk"
    elif document["status"] in STATUS_LABELS:
        detail += " · " + STATUS_LABELS[document["status"]]
    else:
        detail += " · " + ingest.thousands(document["char_count"]) + " karakter"
    uploaded_at = document["uploaded_at"][11:16]

    info_col, delete_col = st.columns([0.87, 0.13], gap="small", vertical_alignment="center")
    with info_col:
        st.markdown(
            f"""
            <div class="ac-doc">
                <span class="ac-doc-icon">{icon}</span>
                <span class="ac-doc-body">
                    <span class="ac-doc-name">{escape(shorten(document["filename"]))}</span>
                    <span class="ac-doc-meta">{escape(detail)} · {uploaded_at}</span>
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with delete_col.container(key=f"deldoc_{document_id}"):
        if st.button(
            "", icon=":material/delete:", key=f"deldoc_btn_{document_id}",
            width="stretch", type="tertiary", help="Hapus dokumen",
        ):
            st.session_state.pending_doc_delete = document_id
            st.rerun()


def rag_enabled() -> bool:
    """Apakah pencarian dokumen sedang dinyalakan."""
    return bool(st.session_state.get("rag_on", True)) and get_settings().rag_available


def render_about() -> None:
    """Ringkasan arsitektur sistem, supaya alurnya bisa dibaca dari dalam app."""
    settings = get_settings()
    with st.expander("Tentang aplikasi ini", expanded=False):
        st.markdown(
            f"""
            <div class="ac-about">
              <p class="ac-about-lead">
                {APP_NAME} menjawab pertanyaan berdasarkan dokumen yang kamu unggah.
                Isi dokumen dicari lebih dulu, baru dikirim ke model bahasa sebagai
                kutipan, jadi jawabannya bisa ditelusuri ke sumbernya.
              </p>

              <div class="ac-about-flow">
                <span>Unggah</span><i></i>
                <span>Ekstraksi teks</span><i></i>
                <span>Chunking</span><i></i>
                <span>Embedding</span><i></i>
                <span>Qdrant</span>
              </div>
              <div class="ac-about-flow">
                <span>Pertanyaan</span><i></i>
                <span>Embedding</span><i></i>
                <span>Cari {TOP_K} chunk</span><i></i>
                <span>Kutipan + pertanyaan</span><i></i>
                <span>Groq</span><i></i>
                <span>Jawaban + Sumber</span>
              </div>

              <table class="ac-about-table">
                <tr><td>Antarmuka</td><td>Streamlit</td></tr>
                <tr><td>Model chat</td><td><code>{settings.model}</code></td></tr>
                <tr><td>Model vision</td><td><code>{settings.vision_model}</code></td></tr>
                <tr><td>Embedding</td><td><code>{settings.embed_model}</code></td></tr>
                <tr><td>Vector store</td><td>Qdrant local mode, metric COSINE</td></tr>
                <tr><td>Riwayat &amp; metadata</td><td>SQLite</td></tr>
                <tr><td>Chunk</td><td>{CHUNK_SIZE} karakter, overlap {CHUNK_OVERLAP}</td></tr>
                <tr><td>Top-K</td><td>{TOP_K} kutipan per pertanyaan</td></tr>
              </table>

              <p class="ac-about-foot">
                File bisa diunggah lewat sidebar atau dilampirkan langsung di kotak
                chat, keduanya melewati pipeline yang sama. Dokumen PDF, DOCX, XLSX,
                dan PPTX diurai jadi teks. Gambar dibaca model vision lebih dulu,
                lalu deskripsinya diperlakukan seperti teks dokumen biasa. Maksimal
                {ingest.MAX_FILES_PER_BATCH} file sekali unggah dan
                {ingest.MAX_FILE_SIZE_MB} MB per file, dan itu batas per unggah,
                bukan kuota permanen. File aslinya tidak pernah disimpan permanen,
                hanya hasil pemrosesannya.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_rag_controls() -> None:
    """Toggle RAG plus keterangan kenapa dimatikan kalau belum bisa dipakai.

    Pilihan toggle disimpan di ``rag_on`` — kunci biasa, bukan kunci widget.
    Streamlit membuang state milik widget yang tidak sempat dirender pada
    suatu run, dan tombol hapus/pindah percakapan di atas memang memanggil
    ``st.rerun()`` sebelum toggle ini sempat tampil. Kalau nilainya disimpan
    di kunci widget, RAG akan menyala sendiri setiap kali itu terjadi.
    """
    settings = get_settings()
    indexed_chunks = vectorstore.count_chunks()

    if not settings.rag_available:
        st.markdown(
            '<div class="ac-note">RAG mati: <code>GEMINI_API_KEY</code> belum diset, '
            "jadi dokumen tidak bisa di-embedding.</div>",
            unsafe_allow_html=True,
        )
        return

    with st.container(key="rag_toggle"):
        st.session_state.rag_on = st.toggle(
            "Jawab pakai dokumen",
            value=st.session_state.get("rag_on", True),
            key="rag_toggle_widget",
            disabled=indexed_chunks == 0,
            help=(
                "Matikan untuk membandingkan jawaban tanpa dokumen."
                if indexed_chunks
                else "Belum ada dokumen terindeks."
            ),
        )

    if indexed_chunks:
        state = "aktif" if st.session_state.rag_on else "nonaktif"
        st.markdown(
            f'<div class="ac-note">{indexed_chunks} chunk terindeks di Qdrant · '
            f"pencarian <strong>{state}</strong>.</div>",
            unsafe_allow_html=True,
        )


DOCS_PER_PAGE = 5


def paged_documents(total: int) -> tuple[list, int, int, int, str]:
    """Ambil satu halaman daftar dokumen, beserta kotak pencariannya.

    Pencarian dan tombol halaman hanya muncul kalau daftarnya sudah lebih
    panjang dari satu halaman, supaya sidebar tetap ringkas saat dokumennya
    masih sedikit. Kata kunci disimpan di kunci biasa (bukan kunci widget)
    karena Streamlit membuang state widget yang tidak sempat dirender pada
    suatu run, dan tombol unggah di atas memang memicu rerun lebih dulu.
    """
    query = ""
    if total > DOCS_PER_PAGE:
        with st.container(key="doc_search"):
            query = st.text_input(
                "Cari dokumen",
                value=st.session_state.get("doc_query", ""),
                key="doc_query_widget",
                placeholder="Cari nama file…",
                label_visibility="collapsed",
            ).strip()
        if query != st.session_state.get("doc_query", ""):
            st.session_state.doc_page = 1
        st.session_state.doc_query = query
    else:
        st.session_state.doc_query = ""

    matched = db.count_documents(query)
    pages = max(1, -(-matched // DOCS_PER_PAGE))
    page = min(max(1, int(st.session_state.get("doc_page", 1))), pages)
    st.session_state.doc_page = page

    documents = db.list_documents(
        search=query, limit=DOCS_PER_PAGE, offset=(page - 1) * DOCS_PER_PAGE
    )
    return documents, matched, page, pages, query


def render_pager(page: int, pages: int, matched: int) -> None:
    """Tombol pindah halaman untuk daftar dokumen."""
    prev_col, info_col, next_col = st.columns(
        [0.18, 0.64, 0.18], gap="small", vertical_alignment="center"
    )
    with prev_col.container(key="doc_prev"):
        if st.button(
            "", icon=":material/chevron_left:", key="doc_prev_btn",
            width="stretch", type="tertiary", disabled=page <= 1,
            help="Halaman sebelumnya",
        ):
            st.session_state.doc_page = page - 1
            st.rerun()
    info_col.markdown(
        f'<div class="ac-pager">Hal {page} dari {pages}'
        f'<span>{matched} dokumen</span></div>',
        unsafe_allow_html=True,
    )
    with next_col.container(key="doc_next"):
        if st.button(
            "", icon=":material/chevron_right:", key="doc_next_btn",
            width="stretch", type="tertiary", disabled=page >= pages,
            help="Halaman berikutnya",
        ):
            st.session_state.doc_page = page + 1
            st.rerun()


def render_documents_section() -> None:
    """Blok unggah dokumen: uploader, kuota, dan daftar dokumen tersimpan."""
    used = db.count_documents()

    st.markdown(
        f'<div class="ac-section">Dokumen<span class="ac-count">{used}</span></div>',
        unsafe_allow_html=True,
    )

    with st.container(key="uploader"):
        files = st.file_uploader(
            "Unggah dokumen",
            type=list(ingest.ALLOWED_TYPES),
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.get('upload_round', 0)}",
            label_visibility="collapsed",
        )
    if files:
        results, batch_error = process_uploads(files)
        st.session_state.upload_results = results
        st.session_state.upload_error = batch_error
        # Ganti key uploader supaya widget kosong lagi dan file tidak diproses dua kali.
        st.session_state.upload_round = st.session_state.get("upload_round", 0) + 1
        st.session_state.doc_page = 1
        st.rerun()

    render_upload_results()
    render_limits(used)

    documents, matched, page, pages, query = paged_documents(used)

    if not documents:
        pesan = (
            f"Tidak ada dokumen yang cocok dengan <strong>{escape(query)}</strong>."
            if query
            else "Belum ada dokumen. Unggah lewat kotak di atas atau lampirkan di chat."
        )
        st.markdown(f'<div class="ac-hint">{pesan}</div>', unsafe_allow_html=True)

    for document in documents:
        render_document_row(document)

    if pages > 1:
        render_pager(page, pages, matched)

    if documents:
        st.markdown(
            '<div class="ac-note">File asli tidak disimpan, hanya teks hasil '
            "ekstraksi dan vector-nya.</div>",
            unsafe_allow_html=True,
        )

    render_rag_controls()


def render_conversation_row(conv: dict) -> None:
    """Satu baris percakapan: tombol buka + tombol hapus (dengan konfirmasi)."""
    conv_id = conv["id"]
    is_active = conv_id == active_conversation_id()

    if st.session_state.get("pending_delete") == conv_id:
        label, confirm, cancel = st.columns([0.6, 0.2, 0.2], gap="small", vertical_alignment="center")
        label.markdown('<div class="ac-hint">Hapus chat ini?</div>', unsafe_allow_html=True)
        with confirm.container(key=f"confirm_{conv_id}"):
            if st.button(
                "", icon=":material/check:", key=f"confirm_btn_{conv_id}",
                width="stretch", help="Ya, hapus",
            ):
                db.delete_conversation(conv_id)
                if is_active:
                    open_conversation(None)
                st.session_state.pending_delete = None
                st.rerun()
        with cancel.container(key=f"cancel_{conv_id}"):
            if st.button(
                "", icon=":material/close:", key=f"cancel_btn_{conv_id}",
                width="stretch", help="Batal",
            ):
                st.session_state.pending_delete = None
                st.rerun()
        return

    open_col, delete_col = st.columns([0.82, 0.18], gap="small", vertical_alignment="center")
    with open_col:
        if st.button(
            conv["title"],
            key=f"open_{conv_id}",
            width="stretch",
            type="primary" if is_active else "tertiary",
            help=f"{conv['message_count']} pesan · diperbarui {conv['updated_at'][:16].replace('T', ' ')} UTC",
        ):
            open_conversation(conv_id)
            st.rerun()
    with delete_col.container(key=f"del_{conv_id}"):
        if st.button(
            "", icon=":material/delete:", key=f"del_btn_{conv_id}",
            width="stretch", type="tertiary", help="Hapus percakapan",
        ):
            st.session_state.pending_delete = conv_id
            st.rerun()


# --------------------------------------------------------------------------- #
# Area chat
# --------------------------------------------------------------------------- #
def render_empty_state() -> None:
    with st.container(key="empty_wrap"):
        _render_empty_state_body()


def _render_empty_state_body() -> None:
    st.markdown(
        f"""
        <div class="ac-empty">
            <div class="ac-empty-mark">✦</div>
            <h1>Halo, ada yang bisa dibantu?</h1>
            <p>Tanyakan apa saja ke {APP_NAME}. Setiap percakapan tersimpan otomatis
            dan bisa dibuka kembali kapan pun lewat sidebar.</p>
        </div>
        <div class="ac-empty-label">Coba mulai dari sini</div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(key="suggestions"):
        columns = st.columns(len(SUGGESTIONS), gap="small")
        for column, (label, prompt) in zip(columns, SUGGESTIONS):
            with column:
                if st.button(label, key=f"suggest_{label}", width="stretch"):
                    st.session_state.queued_prompt = prompt
                    st.rerun()


def render_thread_header(conversation: dict, message_count: int) -> None:
    # Judul berasal dari ringkasan model, jadi tetap di-escape sebelum masuk HTML.
    title = escape(conversation["title"])
    st.markdown(
        f"""
        <div class="ac-thread-head">
            <span class="ac-thread-title">{title}</span>
            <span class="ac-thread-meta">{message_count} pesan</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(sources: list[dict]) -> None:
    """Expander "Sumber": nama file, potongan teks, dan skor kemiripannya."""
    if not sources:
        return
    files = sorted({str(source["filename"]) for source in sources})
    summary = f"Sumber · {len(sources)} kutipan dari {len(files)} file"
    with st.expander(summary, expanded=False):
        for number, source in enumerate(sources, start=1):
            score = float(source["score"])
            st.markdown(
                f"""
                <div class="ac-src">
                    <div class="ac-src-head">
                        <span class="ac-src-rank">{number}</span>
                        <span class="ac-src-file">{escape(str(source["filename"]))}</span>
                        <span class="ac-src-part">bagian {int(source["chunk_index"]) + 1}</span>
                        <span class="ac-src-score">{score:.3f}</span>
                    </div>
                    <div class="ac-src-bar"><span style="width:{max(0.0, min(score, 1.0)) * 100:.1f}%"></span></div>
                    <div class="ac-src-text">{escape(str(source["text"]))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_message(role: str, content: str, key: str | int = "live") -> None:
    """Render satu bubble chat.

    Dibungkus container ber-key supaya CSS bisa membedakan bubble pengguna
    dan bubble asisten (Streamlit memakai test-id yang sama untuk keduanya
    ketika avatar-nya kustom).
    """
    avatar = USER_AVATAR if role == "user" else BOT_AVATAR
    with st.container(key=f"msg-{role}-{key}"):
        with st.chat_message(role, avatar=avatar):
            st.markdown(content)
        # Expander sumber sengaja di luar bubble: kalau di dalam, lebarnya
        # menembus batas bubble dan menutupi teks jawaban.
        if role == "assistant" and isinstance(key, int):
            render_sources(db.list_message_sources(key))


def read_submission(submission) -> tuple[str | None, list]:
    """Pisahkan teks dan lampiran dari hasil ``st.chat_input``.

    Widget mengembalikan string biasa kalau tidak menerima file, atau objek
    ``ChatInputValue`` dengan atribut ``text`` dan ``files`` kalau menerima.
    """
    if submission is None:
        return None, []
    if isinstance(submission, str):
        return submission.strip() or None, []
    text = (getattr(submission, "text", "") or "").strip()
    files = list(getattr(submission, "files", None) or [])
    return text or None, files


def attachment_note(results: list) -> str:
    """Catatan lampiran yang ikut disimpan di isi pesan pengguna."""
    names = [result.filename for result in results if result.ok]
    return f"\n\n_Lampiran: {', '.join(names)}_" if names else ""


def retrieve_sources(question: str) -> tuple[list[dict], str | None]:
    """Cari chunk pendukung di Qdrant. Kembalikan (sumber, pesan error)."""
    if not rag_enabled():
        return [], None
    try:
        hits = rag.retrieve(question)
    except Exception as exc:  # noqa: BLE001 - pencarian gagal tidak boleh membatalkan chat
        return [], errors.friendly_message(
            exc, "Pencarian dokumen gagal, jawaban dibuat tanpa dokumen."
        )
    return [
        {
            "document_id": hit.document_id,
            "filename": hit.filename,
            "chunk_index": hit.chunk_index,
            "score": hit.score,
            "text": hit.text,
        }
        for hit in hits
    ], None


def generate_reply(conversation_id: int, question: str | None = None) -> None:
    """Stream jawaban asisten untuk percakapan, lalu simpan hasilnya.

    Kalau RAG aktif, pertanyaan dicari dulu di Qdrant dan kutipan yang
    ketemu dikirim ke Groq sebagai teks biasa di dalam system prompt.

    Pesan error disimpan di ``session_state`` supaya tetap terlihat setelah
    ``st.rerun()`` — kalau ditampilkan langsung, alert-nya akan ikut terhapus.
    """
    messages = db.list_messages(conversation_id)
    if question is None:
        question = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )

    sources: list[dict] = []
    retrieval_error: str | None = None
    if question:
        with st.spinner(f"Mencari {TOP_K} kutipan paling relevan…"):
            sources, retrieval_error = retrieve_sources(question)

    context = rag.build_context(
        [vectorstore.SearchHit(**source) for source in sources]
    ) if sources else None

    buffer: list[str] = []
    error: str | None = None

    with st.container(key="msg-assistant-live"), st.chat_message("assistant", avatar=BOT_AVATAR):
        try:
            st.write_stream(tee(llm.stream_chat(messages, context=context), buffer))
        except llm.LLMConfigError as exc:
            error = str(exc)
        except Exception as exc:  # noqa: BLE001 - error apa pun tetap dilaporkan ke pengguna
            error = errors.friendly_message(exc, "Jawaban gagal diambil dari Groq.")

    answer = "".join(buffer).strip()
    if answer:
        message_id = db.add_message(conversation_id, "assistant", answer)
        db.save_message_sources(message_id, sources)
    if error:
        st.session_state.last_error = error
    elif retrieval_error:
        st.session_state.last_error = retrieval_error


def handle_prompt(prompt: str | None, attachments: list | None = None) -> None:
    """Proses lampiran (kalau ada), simpan pesan pengguna, lalu stream jawaban.

    Lampiran diproses lebih dulu supaya chunk-nya sudah ada di Qdrant saat
    pertanyaan pada pesan yang sama dicari jawabannya.
    """
    attachments = list(attachments or [])
    results: list = []

    if attachments:
        with st.container(key="chat-upload"):
            results, batch_error = process_uploads(attachments)
        st.session_state.upload_results = results
        st.session_state.upload_error = batch_error
        st.session_state.doc_page = 1
        if batch_error:
            # Seluruh batch ditolak, jadi pertanyaannya tidak ikut dikirim:
            # jawaban tanpa file yang dimaksud justru menyesatkan.
            st.rerun()
            return

    if not prompt:
        # Hanya melampirkan file tanpa bertanya: cukup diproses lalu segarkan.
        st.rerun()
        return

    conv_id = active_conversation_id()
    is_new_conversation = conv_id is None

    if is_new_conversation:
        conv_id = db.create_conversation()
        st.session_state.conversation_id = conv_id

    content = prompt + attachment_note(results)
    db.add_message(conv_id, "user", content)
    render_message("user", content)
    generate_reply(conv_id, question=prompt)

    if is_new_conversation:
        with st.spinner("Membuat judul percakapan…"):
            db.rename_conversation(conv_id, llm.generate_title(prompt))

    st.rerun()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(
        page_title=f"{APP_NAME} · {APP_TAGLINE}",
        page_icon="✦",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    db.init_db()
    settings = get_settings()
    st.session_state.setdefault("conversation_id", None)
    st.session_state.setdefault("pending_delete", None)
    st.session_state.setdefault("pending_doc_delete", None)
    st.session_state.setdefault("upload_round", 0)
    st.session_state.setdefault("rag_on", True)
    st.session_state.setdefault("doc_page", 1)
    st.session_state.setdefault("doc_query", "")

    conv_id = active_conversation_id()
    db.delete_empty_conversations(except_id=conv_id)

    # Percakapan bisa saja sudah dihapus dari tab lain.
    if conv_id is not None and db.get_conversation(conv_id) is None:
        open_conversation(None)
        conv_id = None

    render_sidebar(db.list_conversations())

    submission = st.chat_input(
        f"Kirim pesan ke {APP_NAME}…",
        accept_file="multiple",
        file_type=list(ingest.ALLOWED_TYPES),
        disabled=not settings.is_configured,
    )
    prompt, attachments = read_submission(submission)
    prompt = prompt or st.session_state.pop("queued_prompt", None)

    if not settings.is_configured:
        st.warning(
            "**GROQ_API_KEY belum diset.** Salin `.env.example` menjadi `.env`, "
            "isi API key dari [console.groq.com/keys](https://console.groq.com/keys), "
            "lalu jalankan ulang aplikasi.",
            icon=":material/key_off:",
        )

    messages = db.list_messages(conv_id) if conv_id else []
    if messages:
        conversation = db.get_conversation(conv_id)
        if conversation:
            render_thread_header(conversation, len(messages))
        for message in messages:
            render_message(message["role"], message["content"], key=message["id"])
    elif not prompt and not attachments:
        render_empty_state()

    if prompt or attachments:
        handle_prompt(prompt, attachments)
        return

    if st.session_state.pop("retry_pending", False) and conv_id:
        generate_reply(conv_id)
        st.rerun()

    error = st.session_state.pop("last_error", None)
    if error:
        st.error(error, icon=":material/error:")

    # Pesan terakhir dari pengguna berarti jawaban gagal dibuat — tawarkan ulang.
    if messages and messages[-1]["role"] == "user" and settings.is_configured:
        with st.container(key="retry_wrap"):
            if st.button("Coba lagi", icon=":material/refresh:", key="retry"):
                st.session_state.retry_pending = True
                st.rerun()


if __name__ == "__main__":
    main()
