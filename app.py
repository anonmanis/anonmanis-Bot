"""Nimbus — chatbot AI berbasis Streamlit + Groq.

Entry point aplikasi. Tahap ini fokus ke chat dasar: percakapan tersimpan
di SQLite, jawaban di-stream dari Groq, judul dibuat otomatis.
"""

from __future__ import annotations

from html import escape
from typing import Iterable, Iterator

import streamlit as st

from core import db, llm
from core.config import APP_NAME, APP_TAGLINE, get_settings

USER_AVATAR = ":material/person:"
BOT_AVATAR = ":material/auto_awesome:"

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
    --nb-accent-dim: rgba(124, 92, 255, 0.14);
    --nb-radius:    18px;
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
    font-size: 0.965rem;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.28);
    overflow-x: auto;
}
[data-testid="stChatMessage"] [data-testid="stChatMessageContent"] p:last-child {
    margin-bottom: 0;
}
/* Pesan pengguna diberi warna aksen yang lembut */
[class*="st-key-msg-user-"] [data-testid="stChatMessageContent"] {
    background: var(--nb-accent-dim);
    border-color: rgba(124, 92, 255, 0.34);
}
[class*="st-key-msg-"] { gap: 0 !important; }

/* --- Avatar --- */
[data-testid="stChatMessageAvatarCustom"] {
    width: 2.15rem !important;
    height: 2.15rem !important;
    border-radius: 11px !important;
    border: 1px solid var(--nb-border);
    flex-shrink: 0;
}
[class*="st-key-msg-assistant-"] [data-testid="stChatMessageAvatarCustom"] {
    background: linear-gradient(140deg, var(--nb-accent), #4B31D6);
    border-color: rgba(124, 92, 255, 0.55);
    color: #FFFFFF;
    box-shadow: 0 0 0 3px rgba(124, 92, 255, 0.12);
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
    border-color: rgba(124, 92, 255, 0.6);
    box-shadow: 0 0 0 3px rgba(124, 92, 255, 0.16), 0 8px 28px rgba(0, 0, 0, 0.4);
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--nb-muted); }

/* --- Sidebar --- */
[data-testid="stSidebar"] { border-right: 1px solid var(--nb-border); }
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
    background: linear-gradient(140deg, var(--nb-accent), #4B31D6);
    color: #FFFFFF;
    font-size: 1.1rem;
    font-weight: 700;
    box-shadow: 0 4px 16px rgba(124, 92, 255, 0.35);
}
.nb-brand-name {
    font-size: 1.06rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.nb-brand-sub { font-size: 0.75rem; color: var(--nb-muted); }

.nb-section {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.25rem 0.5rem 0.4rem;
    font-size: 0.68rem;
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
    font-size: 0.8rem;
    line-height: 1.6;
    color: var(--nb-muted);
}
.nb-foot {
    padding: 0.55rem 0.6rem 0;
    font-size: 0.72rem;
    color: var(--nb-muted);
    line-height: 1.7;
}
.nb-foot code {
    font-family: 'JetBrains Mono', ui-monospace, monospace;
    font-size: 0.7rem;
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
    border-radius: 11px !important;
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
    box-shadow: 0 4px 14px rgba(124, 92, 255, 0.28);
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
    color: #FF6B6B;
    background: rgba(255, 107, 107, 0.12);
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
    background: linear-gradient(140deg, var(--nb-accent), #4B31D6);
    color: #FFFFFF;
    font-size: 1.65rem;
    font-weight: 700;
    box-shadow: 0 10px 34px rgba(124, 92, 255, 0.38);
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
    font-size: 0.95rem;
    line-height: 1.75;
}
.nb-empty-label {
    margin: 1.6rem 0 0.9rem;
    font-size: 0.68rem;
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
    font-size: 0.83rem;
    font-weight: 500;
    line-height: 1.4;
    transition: border-color 0.15s ease, transform 0.15s ease;
}
.st-key-suggestions .stButton > button:hover {
    border-color: rgba(124, 92, 255, 0.55);
    background: var(--nb-elevated);
    color: var(--nb-text);
    transform: translateY(-1px);
}

/* --- Tombol "Coba lagi" saat jawaban gagal --- */
.st-key-retry_wrap { margin: 0.35rem 0 0 3rem; }
.st-key-retry_wrap button {
    border-radius: 999px !important;
    font-size: 0.83rem;
    padding: 0.3rem 0.9rem;
    color: var(--nb-muted);
}
.st-key-retry_wrap button:hover {
    color: var(--nb-text);
    border-color: rgba(124, 92, 255, 0.55);
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
    font-size: 1.02rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}
.nb-thread-meta { font-size: 0.75rem; color: var(--nb-muted); }

/* Scrollbar tipis biar konsisten dengan tema */
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: #262C3C;
    border-radius: 99px;
    border: 2px solid var(--nb-bg);
}
::-webkit-scrollbar-thumb:hover { background: #333B50; }
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
        settings = get_settings()
        st.markdown(
            f'<div class="nb-foot">Model <code>{settings.model}</code><br>'
            "Riwayat tersimpan lokal di SQLite.</div>",
            unsafe_allow_html=True,
        )


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


def generate_reply(conversation_id: int) -> None:
    """Stream jawaban asisten untuk percakapan, lalu simpan hasilnya.

    Pesan error disimpan di ``session_state`` supaya tetap terlihat setelah
    ``st.rerun()`` — kalau ditampilkan langsung, alert-nya akan ikut terhapus.
    """
    buffer: list[str] = []
    error: str | None = None

    with st.container(key="msg-assistant-live"), st.chat_message("assistant", avatar=BOT_AVATAR):
        try:
            st.write_stream(tee(llm.stream_chat(db.list_messages(conversation_id)), buffer))
        except llm.LLMConfigError as exc:
            error = str(exc)
        except Exception as exc:  # noqa: BLE001 - error apa pun tetap dilaporkan ke pengguna
            error = f"Gagal mengambil jawaban dari Groq: {exc}"

    answer = "".join(buffer).strip()
    if answer:
        db.add_message(conversation_id, "assistant", answer)
    if error:
        st.session_state.last_error = error


def handle_prompt(prompt: str) -> None:
    """Simpan pesan pengguna, stream jawaban, lalu simpan hasilnya."""
    conv_id = active_conversation_id()
    is_new_conversation = conv_id is None

    if is_new_conversation:
        conv_id = db.create_conversation()
        st.session_state.conversation_id = conv_id

    db.add_message(conv_id, "user", prompt)
    render_message("user", prompt)
    generate_reply(conv_id)

    if is_new_conversation:
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
