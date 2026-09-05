"""Nimbus — chatbot AI berbasis Streamlit + Groq.

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
    --nb-bg:        #0B0D12;
    --nb-surface:   #12151E;
    --nb-elevated:  #171B27;
    --nb-border:    #242A3A;
    --nb-text:      #E9EBF2;
    --nb-muted:     #8A92A6;
    --nb-accent:    #7C5CFF;
    --nb-accent-deep: #4B31D6;
    --nb-accent-soft: #9B85FF;
    --nb-on-accent: #FFFFFF;
    --nb-danger:    #FF8B8B;

    /* Tiga tingkat transparansi aksen dipakai konsisten:
       dim untuk isian, line untuk garis tepi, edge untuk tepi aktif. */
    --nb-accent-dim:  rgba(124, 92, 255, 0.14);
    --nb-accent-line: rgba(124, 92, 255, 0.30);
    --nb-accent-edge: rgba(124, 92, 255, 0.55);

    /* Skala tipografi */
    --nb-fs-2xs:  0.68rem;
    --nb-fs-xs:   0.72rem;
    --nb-fs-sm:   0.78rem;
    --nb-fs-base: 0.83rem;
    --nb-fs-md:   0.95rem;
    --nb-fs-lg:   1.05rem;

    /* Skala jarak */
    --nb-gap-xs: 0.35rem;
    --nb-gap-sm: 0.55rem;
    --nb-gap-md: 0.85rem;
    --nb-gap-lg: 1.4rem;

    --nb-radius:    18px;
    --nb-radius-sm: 11px;
    --nb-chat-width: 46rem;
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
    max-width: var(--nb-chat-width) !important;
    padding-top: 3.25rem !important;
    padding-bottom: 4rem !important;
}
[data-testid="stBottomBlockContainer"] {
    max-width: calc(var(--nb-chat-width) + 2rem) !important;
    padding-bottom: 1.25rem !important;
}
[data-testid="stBottom"] > div {
    background: linear-gradient(to bottom, rgba(11, 13, 18, 0), var(--nb-bg) 22%);
}

/* --- Bubble chat --- */
[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 0 !important;
    margin-bottom: 1.5rem;
    gap: 0.85rem;
}
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
    background: var(--nb-surface);
    border: 1px solid var(--nb-border);
    border-radius: var(--nb-radius);
    padding: 0.95rem 1.2rem;
    line-height: 1.75;
    font-size: var(--nb-fs-md);
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.28);
    overflow-x: auto;
}
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] p:last-child {
    margin-bottom: 0;
}
/* Pesan pengguna diberi warna aksen yang lembut */
[class*="st-key-msg-user-"] [data-testid="stChatMessageContent"] {
    background: var(--nb-accent-dim);
    border-color: var(--nb-accent-line);
}
[class*="st-key-msg-"] { gap: 0 !important; }

/* --- Avatar --- */
[data-testid="stChatMessageAvatarCustom"] {
    width: 2.15rem !important;
    height: 2.15rem !important;
    border-radius: var(--nb-radius-sm) !important;
    border: 1px solid var(--nb-border);
    flex-shrink: 0;
}
[class*="st-key-msg-assistant-"] [data-testid="stChatMessageAvatarCustom"] {
    background: linear-gradient(140deg, var(--nb-accent), var(--nb-accent-deep));
    border-color: var(--nb-accent-edge);
    color: var(--nb-on-accent);
    box-shadow: 0 0 0 3px var(--nb-accent-dim);
}
[class*="st-key-msg-user-"] [data-testid="stChatMessageAvatarCustom"] {
    background: var(--nb-elevated);
    color: var(--nb-muted);
}
[data-testid="stChatMessageAvatarCustom"] [data-testid="stIconMaterial"] {
    font-size: 1.15rem;
}

/* --- Kotak input chat --- */
[data-testid="stChatInput"] {
    background: var(--nb-surface);
    border: 1px solid var(--nb-border);
    border-radius: 16px;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.4);
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--nb-accent-edge);
    box-shadow: 0 0 0 3px var(--nb-accent-dim), 0 8px 28px rgba(0, 0, 0, 0.4);
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--nb-muted); }

/* --- Sidebar --- */
/* Sidebar dilebarkan dari default 300px: sekarang menampung daftar dokumen
   dengan nama, tipe, ukuran, jumlah karakter, dan jam unggah dalam satu baris. */
[data-testid="stSidebar"] {
    border-right: 1px solid var(--nb-border);
    width: 21rem !important;
}
[data-testid="stSidebar"] > div { width: 21rem; }
[data-testid="stSidebarContent"] { padding: 1.4rem 0.9rem 1rem; }
[data-testid="stSidebar"] hr {
    margin: 0.9rem 0;
    border-color: var(--nb-border);
    opacity: 0.75;
}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.35rem; }

.nb-brand {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.1rem 0.35rem 0.35rem;
}
.nb-brand-mark {
    display: grid;
    place-items: center;
    width: 2.35rem;
    height: 2.35rem;
    border-radius: 13px;
    background: linear-gradient(140deg, var(--nb-accent), var(--nb-accent-deep));
    color: var(--nb-on-accent);
    font-size: 1.1rem;
    font-weight: 700;
    box-shadow: 0 4px 16px var(--nb-accent-line);
}
.nb-brand-name {
    font-size: var(--nb-fs-lg);
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.nb-brand-sub { font-size: var(--nb-fs-sm); color: var(--nb-muted); }

.nb-section {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.25rem 0.5rem 0.4rem;
    font-size: var(--nb-fs-2xs);
    font-weight: 700;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    color: var(--nb-muted);
}
.nb-section span.nb-count {
    letter-spacing: 0;
    font-weight: 600;
    color: var(--nb-muted);
    opacity: 0.75;
}
.nb-hint {
    padding: 0.7rem 0.6rem;
    font-size: var(--nb-fs-sm);
    line-height: 1.6;
    color: var(--nb-muted);
}
.nb-foot {
    padding: 0.55rem 0.6rem 0;
    font-size: var(--nb-fs-xs);
    color: var(--nb-muted);
    line-height: 1.7;
}
.nb-foot code {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: var(--nb-fs-xs);
    color: var(--nb-accent);
    background: var(--nb-accent-dim);
    padding: 0.1rem 0.35rem;
    border-radius: 6px;
}

/* Tombol daftar percakapan: rata kiri, sudut lembut, hover halus */
[data-testid="stSidebar"] [data-testid="stButton"] button {
    justify-content: flex-start !important;
    font-weight: 500;
    padding: 0.45rem 0.7rem;
    border-radius: var(--nb-radius-sm) !important;
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
    background: var(--nb-surface);
    color: var(--nb-text);
}
[data-testid="stSidebar"] .st-key-new_chat { margin-top: 0.55rem; }
[data-testid="stSidebar"] .st-key-new_chat button {
    justify-content: center !important;
    border-radius: 999px !important;
    font-weight: 600;
    padding: 0.55rem 0.9rem;
    box-shadow: 0 4px 14px var(--nb-accent-line);
}
[data-testid="stSidebar"] [class*="st-key-del_"] button,
[data-testid="stSidebar"] [class*="st-key-confirm_"] button,
[data-testid="stSidebar"] [class*="st-key-cancel_"] button {
    justify-content: center !important;
    padding: 0.45rem !important;
    color: var(--nb-muted);
}
[data-testid="stSidebar"] [class*="st-key-del_"] button:hover,
[data-testid="stSidebar"] [class*="st-key-confirm_"] button:hover {
    color: var(--nb-danger);
    background: rgba(255, 139, 139, 0.12);
}

/* --- Empty state --- */
.st-key-empty_wrap {
    display: flex;
    flex-direction: column;
    justify-content: center;
    min-height: 66vh;
}
.nb-empty { padding: 0 0 1.75rem; text-align: center; }
.nb-empty-mark {
    display: inline-grid;
    place-items: center;
    width: 3.6rem;
    height: 3.6rem;
    margin-bottom: 1.4rem;
    border-radius: 20px;
    background: linear-gradient(140deg, var(--nb-accent), var(--nb-accent-deep));
    color: var(--nb-on-accent);
    font-size: 1.65rem;
    font-weight: 700;
    box-shadow: 0 10px 34px var(--nb-accent-line);
}
.nb-empty h1 {
    margin: 0 0 0.6rem;
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.035em;
}
.nb-empty p {
    max-width: 30rem;
    margin: 0 auto;
    color: var(--nb-muted);
    font-size: var(--nb-fs-md);
    line-height: 1.75;
}
.nb-empty-label {
    margin: 1.6rem 0 0.9rem;
    font-size: var(--nb-fs-2xs);
    font-weight: 700;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    color: var(--nb-muted);
    text-align: center;
}
.st-key-suggestions .stButton > button {
    width: 100%;
    min-height: 3.1rem;
    border-radius: 14px;
    background: var(--nb-surface);
    border: 1px solid var(--nb-border);
    color: var(--nb-text);
    font-size: var(--nb-fs-base);
    font-weight: 500;
    line-height: 1.4;
    transition: border-color 0.15s ease, transform 0.15s ease;
}
.st-key-suggestions .stButton > button:hover {
    border-color: var(--nb-accent-edge);
    background: var(--nb-elevated);
    color: var(--nb-text);
    transform: translateY(-1px);
}


/* --- Toggle RAG di sidebar --- */
.st-key-rag_toggle { padding: 0.5rem 0.6rem 0.1rem; }
.st-key-rag_toggle label p { font-size: var(--nb-fs-sm) !important; font-weight: 500; }

/* --- Expander "Sumber" di bawah jawaban --- */
[data-testid="stMainBlockContainer"] [data-testid="stExpander"] {
    /* 3rem = lebar avatar + jarak, supaya sejajar dengan bubble jawaban */
    margin: -0.6rem 0 0 3rem;
    width: calc(100% - 3rem) !important;
}
[data-testid="stMainBlockContainer"] [data-testid="stExpander"] details {
    background: transparent;
    border: 1px solid var(--nb-border);
    border-radius: 13px;
}
[data-testid="stMainBlockContainer"] [data-testid="stExpander"] summary {
    font-size: var(--nb-fs-sm);
    font-weight: 600;
    color: var(--nb-muted);
}
[data-testid="stMainBlockContainer"] [data-testid="stExpander"] summary:hover {
    color: var(--nb-accent);
}

.nb-src { padding: 0.55rem 0 0.7rem; border-top: 1px solid var(--nb-border); }
.nb-src:first-child { border-top: none; padding-top: 0.1rem; }
.nb-src-head {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.4rem;
}
.nb-src-rank {
    display: grid;
    place-items: center;
    flex-shrink: 0;
    width: 1.25rem;
    height: 1.25rem;
    border-radius: 6px;
    background: var(--nb-accent-dim);
    border: 1px solid var(--nb-accent-line);
    color: var(--nb-accent);
    font-size: var(--nb-fs-2xs);
    font-weight: 700;
}
.nb-src-file {
    font-size: var(--nb-fs-sm);
    font-weight: 600;
    color: var(--nb-text);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.nb-src-part { font-size: var(--nb-fs-xs); color: var(--nb-muted); }
.nb-src-score {
    margin-left: auto;
    flex-shrink: 0;
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: var(--nb-fs-xs);
    color: var(--nb-accent);
    background: var(--nb-accent-dim);
    padding: 0.1rem 0.4rem;
    border-radius: 6px;
}
.nb-src-bar {
    height: 3px;
    margin-bottom: 0.5rem;
    border-radius: 99px;
    background: var(--nb-elevated);
    overflow: hidden;
}
.nb-src-bar > span {
    display: block;
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, var(--nb-accent), var(--nb-accent-soft));
}
.nb-src-text {
    max-height: 7rem;
    overflow-y: auto;
    padding: 0.55rem 0.7rem;
    border-radius: 10px;
    background: var(--nb-surface);
    border: 1px solid var(--nb-border);
    font-size: var(--nb-fs-sm);
    line-height: 1.65;
    color: var(--nb-muted);
    white-space: pre-wrap;
    word-break: break-word;
}

/* --- Panel "Tentang" --- */
.st-key-about [data-testid="stExpander"] { margin: 0; width: auto !important; }
.st-key-about [data-testid="stExpander"] details {
    background: var(--nb-surface);
    border: 1px solid var(--nb-border);
    border-radius: 12px;
}
.st-key-about [data-testid="stExpander"] summary {
    font-size: var(--nb-fs-sm);
    font-weight: 600;
    color: var(--nb-muted);
}
.nb-about p { margin: 0 0 var(--nb-gap-md); }
.nb-about-lead {
    font-size: var(--nb-fs-sm);
    line-height: 1.7;
    color: var(--nb-text);
}
.nb-about-flow {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.3rem;
    margin-bottom: var(--nb-gap-sm);
}
.nb-about-flow span {
    padding: 0.16rem 0.5rem;
    border-radius: 7px;
    background: var(--nb-accent-dim);
    border: 1px solid var(--nb-accent-line);
    color: var(--nb-text);
    font-size: var(--nb-fs-2xs);
    font-weight: 500;
    white-space: nowrap;
}
.nb-about-flow i {
    width: 0.55rem;
    height: 1px;
    background: var(--nb-border);
    flex-shrink: 0;
}
.nb-about-table {
    width: 100%;
    margin: var(--nb-gap-md) 0;
    border-collapse: collapse;
    font-size: var(--nb-fs-xs);
}
.nb-about-table td {
    padding: 0.28rem 0;
    border-bottom: 1px solid var(--nb-border);
    vertical-align: top;
}
.nb-about-table tr:last-child td { border-bottom: none; }
.nb-about-table td:first-child { color: var(--nb-muted); width: 42%; }
.nb-about-table code {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: var(--nb-fs-2xs);
    color: var(--nb-accent);
    background: var(--nb-accent-dim);
    padding: 0.05rem 0.3rem;
    border-radius: 5px;
    word-break: break-all;
}
.nb-about-foot {
    font-size: var(--nb-fs-xs) !important;
    line-height: 1.7;
    color: var(--nb-muted);
    margin-bottom: 0 !important;
}

/* --- Layar tablet: sidebar dipersempit, area chat ikut menyesuaikan --- */
@media (max-width: 1180px) {
    :root { --nb-chat-width: 40rem; }
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div { width: 18.5rem !important; }
}

@media (max-width: 900px) {
    :root {
        --nb-chat-width: 100%;
        --nb-radius: 15px;
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
    .nb-empty { padding-top: 1rem; }
    .nb-empty h1 { font-size: 1.55rem; }
    .nb-empty p { font-size: var(--nb-fs-md); }
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
        font-size: var(--nb-fs-md);
    }
    .nb-src-head { flex-wrap: wrap; }
    .nb-src-score { margin-left: 0; }
}

/* --- Bagian dokumen di sidebar --- */
.st-key-uploader [data-testid="stFileUploaderDropzone"] {
    background: var(--nb-surface);
    border: 1px dashed var(--nb-border);
    border-radius: 14px;
    padding: 0.7rem 0.8rem;
    min-height: 0;
    transition: border-color 0.15s ease, background 0.15s ease;
}
.st-key-uploader [data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--nb-accent-edge);
    background: var(--nb-elevated);
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
    font-size: var(--nb-fs-xs);
    color: var(--nb-muted);
    white-space: normal;
}
.st-key-uploader [data-testid="stFileUploaderDropzone"] button {
    border-radius: 999px !important;
    font-size: var(--nb-fs-sm);
    padding: 0.25rem 0.75rem;
}
.st-key-uploader [data-testid="stFileUploaderFile"] { font-size: var(--nb-fs-sm); }

/* Indikator kuota "3 dari 5 file terpakai" */
.nb-quota { padding: 0.55rem 0.6rem 0.15rem; }
.nb-quota-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 0.5rem;
    font-size: var(--nb-fs-xs);
    color: var(--nb-muted);
    margin-bottom: 0.4rem;
}
.nb-quota-head strong { color: var(--nb-text); font-weight: 600; }
.nb-quota-cap { font-size: var(--nb-fs-2xs); opacity: 0.8; }
.nb-quota-bar {
    height: 4px;
    border-radius: 99px;
    background: var(--nb-elevated);
    overflow: hidden;
}
.nb-quota-bar > span {
    display: block;
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, var(--nb-accent), var(--nb-accent-soft));
    transition: width 0.25s ease;
}
.nb-quota.is-full .nb-quota-bar > span { background: var(--nb-danger); }

/* Baris dokumen */
.nb-doc {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    padding: 0.38rem 0 0.38rem 0.45rem;
    min-width: 0;
}
.nb-doc-icon {
    display: grid;
    place-items: center;
    flex-shrink: 0;
    width: 1.7rem;
    height: 1.7rem;
    border-radius: 9px;
    background: var(--nb-accent-dim);
    border: 1px solid var(--nb-accent-line);
    color: var(--nb-accent);
}
.nb-doc-icon svg { width: 15px; height: 15px; display: block; }
.nb-doc-body { display: flex; flex-direction: column; min-width: 0; gap: 0.1rem; }
.nb-doc-name {
    font-size: var(--nb-fs-base);
    font-weight: 500;
    color: var(--nb-text);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.nb-doc-meta {
    font-size: var(--nb-fs-2xs);
    color: var(--nb-muted);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.nb-note {
    padding: 0.6rem 0.6rem 0;
    font-size: var(--nb-fs-2xs);
    line-height: 1.65;
    color: var(--nb-muted);
    opacity: 0.85;
}

/* Alert hasil unggah dibuat ringkas supaya muat di sidebar */
[data-testid="stSidebar"] [data-testid="stAlertContainer"] {
    padding: 0.55rem 0.7rem;
    font-size: var(--nb-fs-sm);
    line-height: 1.55;
    border-radius: 12px;
}

/* Chip file bawaan uploader disesuaikan dengan tema */
.st-key-uploader [data-testid="stFileChip"] {
    background: var(--nb-elevated) !important;
    border: 1px solid var(--nb-border) !important;
    border-radius: 10px;
    font-size: var(--nb-fs-sm);
}
.st-key-uploader [data-testid="stFileChip"] svg { color: var(--nb-accent); }

/* st.status: tiap tahap pemrosesan tampil sebagai langkah bertanda */
[data-testid="stSidebar"] [data-testid="stExpander"] details {
    border-radius: 12px;
    border-color: var(--nb-border);
    background: var(--nb-surface);
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    font-size: var(--nb-fs-sm);
    font-weight: 600;
}
[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stMarkdown"] p {
    position: relative;
    font-size: var(--nb-fs-xs);
    line-height: 1.55;
    color: var(--nb-muted);
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
    background: var(--nb-accent);
}

/* --- Tombol "Coba lagi" saat jawaban gagal --- */
.st-key-retry_wrap { margin: 0.35rem 0 0 3rem; }
.st-key-retry_wrap button {
    border-radius: 999px !important;
    font-size: var(--nb-fs-base);
    padding: 0.3rem 0.9rem;
    color: var(--nb-muted);
}
.st-key-retry_wrap button:hover {
    color: var(--nb-text);
    border-color: var(--nb-accent-edge);
}

/* --- Header percakapan aktif --- */
.nb-thread-head {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    margin-bottom: 1.6rem;
    padding-bottom: 0.85rem;
    border-bottom: 1px solid var(--nb-border);
}
.nb-thread-title {
    font-size: var(--nb-fs-lg);
    font-weight: 700;
    letter-spacing: -0.02em;
}
.nb-thread-meta { font-size: var(--nb-fs-sm); color: var(--nb-muted); }

/* Scrollbar tipis biar konsisten dengan tema */
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: var(--nb-scroll);
    border-radius: 99px;
    border: 2px solid var(--nb-bg);
}
::-webkit-scrollbar-thumb:hover { background: var(--nb-scroll-hover); }
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
            <div class="nb-brand">
                <div class="nb-brand-mark">N</div>
                <div>
                    <div class="nb-brand-name">{APP_NAME}</div>
                    <div class="nb-brand-sub">{APP_TAGLINE}</div>
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
            f'<div class="nb-section">Riwayat'
            f'<span class="nb-count">{len(conversations)}</span></div>',
            unsafe_allow_html=True,
        )

        if not conversations:
            st.markdown(
                '<div class="nb-hint">Belum ada percakapan tersimpan. '
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


def process_uploads(files: list) -> None:
    """Jalankan pipeline ingest untuk tiap file sambil menampilkan tahapannya."""
    results = []
    for uploaded in files:
        label = shorten(uploaded.name, 22)
        with st.status(f"Memproses {label}", expanded=True) as status:
            result = ingest.ingest(uploaded.name, uploaded.getvalue(), progress=st.write)
            status.update(
                label=f"{label} — {'selesai' if result.ok else 'ditolak'}",
                state="complete" if result.ok else "error",
                expanded=False,
            )
        results.append(result)

    st.session_state.upload_results = results
    # Ganti key uploader supaya widget-nya kosong lagi dan file tidak diproses dua kali.
    st.session_state.upload_round = st.session_state.get("upload_round", 0) + 1
    st.rerun()


def render_upload_results() -> None:
    """Hasil unggahan terakhir; hilang sendiri begitu ada interaksi lain."""
    for result in st.session_state.pop("upload_results", []):
        name = shorten(result.filename, 22)
        if result.ok:
            st.success(f"**{name}** {result.message}", icon=":material/check_circle:")
        else:
            st.error(f"**{name}** ditolak. {result.message}", icon=":material/block:")


def render_quota(used: int) -> None:
    """Indikator '3 dari 5 file terpakai' beserta bar-nya."""
    total = ingest.MAX_DOCUMENTS
    percent = round(used / total * 100)
    state = " is-full" if used >= total else ""
    st.markdown(
        f"""
        <div class="nb-quota{state}">
            <div class="nb-quota-head">
                <span><strong>{used}</strong> dari {total} file terpakai</span>
                <span class="nb-quota-cap">maks {ingest.MAX_FILE_SIZE_MB} MB/file</span>
            </div>
            <div class="nb-quota-bar"><span style="width:{percent}%"></span></div>
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
        label.markdown('<div class="nb-hint">Hapus dokumen ini?</div>', unsafe_allow_html=True)
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
            <div class="nb-doc">
                <span class="nb-doc-icon">{icon}</span>
                <span class="nb-doc-body">
                    <span class="nb-doc-name">{escape(shorten(document["filename"]))}</span>
                    <span class="nb-doc-meta">{escape(detail)} · {uploaded_at}</span>
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
            <div class="nb-about">
              <p class="nb-about-lead">
                {APP_NAME} menjawab pertanyaan berdasarkan dokumen yang kamu unggah.
                Isi dokumen dicari lebih dulu, baru dikirim ke model bahasa sebagai
                kutipan, jadi jawabannya bisa ditelusuri ke sumbernya.
              </p>

              <div class="nb-about-flow">
                <span>Unggah</span><i></i>
                <span>Ekstraksi teks</span><i></i>
                <span>Chunking</span><i></i>
                <span>Embedding</span><i></i>
                <span>Qdrant</span>
              </div>
              <div class="nb-about-flow">
                <span>Pertanyaan</span><i></i>
                <span>Embedding</span><i></i>
                <span>Cari {TOP_K} chunk</span><i></i>
                <span>Kutipan + pertanyaan</span><i></i>
                <span>Groq</span><i></i>
                <span>Jawaban + Sumber</span>
              </div>

              <table class="nb-about-table">
                <tr><td>Antarmuka</td><td>Streamlit</td></tr>
                <tr><td>Model chat</td><td><code>{settings.model}</code></td></tr>
                <tr><td>Model vision</td><td><code>{settings.vision_model}</code></td></tr>
                <tr><td>Embedding</td><td><code>{settings.embed_model}</code></td></tr>
                <tr><td>Vector store</td><td>Qdrant local mode, metric COSINE</td></tr>
                <tr><td>Riwayat &amp; metadata</td><td>SQLite</td></tr>
                <tr><td>Chunk</td><td>{CHUNK_SIZE} karakter, overlap {CHUNK_OVERLAP}</td></tr>
                <tr><td>Top-K</td><td>{TOP_K} kutipan per pertanyaan</td></tr>
              </table>

              <p class="nb-about-foot">
                Dokumen PDF, DOCX, XLSX, dan PPTX diurai jadi teks. Gambar dibaca
                model vision lebih dulu, lalu deskripsinya diperlakukan seperti teks
                dokumen biasa. File aslinya tidak pernah disimpan permanen, hanya
                hasil pemrosesannya.
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
            '<div class="nb-note">RAG mati: <code>GEMINI_API_KEY</code> belum diset, '
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
            f'<div class="nb-note">{indexed_chunks} chunk terindeks di Qdrant · '
            f"pencarian <strong>{state}</strong>.</div>",
            unsafe_allow_html=True,
        )


def render_documents_section() -> None:
    """Blok unggah dokumen: uploader, kuota, dan daftar dokumen tersimpan."""
    documents = db.list_documents()
    used = len(documents)

    st.markdown(
        f'<div class="nb-section">Dokumen'
        f'<span class="nb-count">{used}/{ingest.MAX_DOCUMENTS}</span></div>',
        unsafe_allow_html=True,
    )

    if used < ingest.MAX_DOCUMENTS:
        with st.container(key="uploader"):
            files = st.file_uploader(
                "Unggah dokumen",
                type=list(ingest.ALLOWED_TYPES),
                accept_multiple_files=True,
                key=f"uploader_{st.session_state.get('upload_round', 0)}",
                label_visibility="collapsed",
            )
        if files:
            process_uploads(files)
    else:
        st.markdown(
            '<div class="nb-hint">Penyimpanan penuh. Hapus salah satu dokumen '
            "dulu untuk bisa mengunggah lagi.</div>",
            unsafe_allow_html=True,
        )

    render_upload_results()
    render_quota(used)

    for document in documents:
        render_document_row(document)

    if documents:
        st.markdown(
            '<div class="nb-note">File asli tidak disimpan — hanya teks hasil '
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
        label.markdown('<div class="nb-hint">Hapus chat ini?</div>', unsafe_allow_html=True)
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
        <div class="nb-empty">
            <div class="nb-empty-mark">✦</div>
            <h1>Halo, ada yang bisa dibantu?</h1>
            <p>Tanyakan apa saja ke {APP_NAME}. Setiap percakapan tersimpan otomatis
            dan bisa dibuka kembali kapan pun lewat sidebar.</p>
        </div>
        <div class="nb-empty-label">Coba mulai dari sini</div>
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
        <div class="nb-thread-head">
            <span class="nb-thread-title">{title}</span>
            <span class="nb-thread-meta">{message_count} pesan</span>
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
                <div class="nb-src">
                    <div class="nb-src-head">
                        <span class="nb-src-rank">{number}</span>
                        <span class="nb-src-file">{escape(str(source["filename"]))}</span>
                        <span class="nb-src-part">bagian {int(source["chunk_index"]) + 1}</span>
                        <span class="nb-src-score">{score:.3f}</span>
                    </div>
                    <div class="nb-src-bar"><span style="width:{max(0.0, min(score, 1.0)) * 100:.1f}%"></span></div>
                    <div class="nb-src-text">{escape(str(source["text"]))}</div>
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


def handle_prompt(prompt: str) -> None:
    """Simpan pesan pengguna, stream jawaban, lalu simpan hasilnya."""
    conv_id = active_conversation_id()
    is_new_conversation = conv_id is None

    if is_new_conversation:
        conv_id = db.create_conversation()
        st.session_state.conversation_id = conv_id

    db.add_message(conv_id, "user", prompt)
    render_message("user", prompt)
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

    conv_id = active_conversation_id()
    db.delete_empty_conversations(except_id=conv_id)

    # Percakapan bisa saja sudah dihapus dari tab lain.
    if conv_id is not None and db.get_conversation(conv_id) is None:
        open_conversation(None)
        conv_id = None

    render_sidebar(db.list_conversations())

    prompt = st.chat_input(
        f"Kirim pesan ke {APP_NAME}…",
        disabled=not settings.is_configured,
    )
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
    elif not prompt:
        render_empty_state()

    if prompt:
        handle_prompt(prompt)
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
