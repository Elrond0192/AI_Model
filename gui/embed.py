"""Basketball Performance AI – Brandable embed widget.

A minimal, white-labelable Streamlit chat page designed to be iframed into a
customer's website or served as a lightweight chat-only portal.

All visual elements (title, colours, logo, footer) are driven by environment
variables so that the same Docker image can be reused for multiple customers
without any code changes.

Environment variables
---------------------
BRAND_TITLE         Title shown in the header (default: "Basketball Performance AI")
BRAND_PRIMARY_COLOR Primary accent colour in CSS hex (default: "#1565C0")
BRAND_LOGO_URL      Optional URL to a PNG/SVG logo. Empty → no logo.
BRAND_API_URL       Base URL of the FastAPI server (default: "http://localhost:8000")
BRAND_FOOTER_TEXT   Optional disclaimer / tagline shown below the chat input.

Launch
------
    streamlit run gui/embed.py --server.port 8502

Or via docker-compose:
    docker compose up embed
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Branding – read from environment
# ---------------------------------------------------------------------------

_TITLE        = os.environ.get("BRAND_TITLE",         "Basketball Performance AI").strip()
_PRIMARY      = os.environ.get("BRAND_PRIMARY_COLOR", "#1565C0").strip()
_LOGO_URL     = os.environ.get("BRAND_LOGO_URL",      "").strip()
_API_URL      = os.environ.get("BRAND_API_URL",       "http://localhost:8000").rstrip("/")
_FOOTER       = os.environ.get("BRAND_FOOTER_TEXT",   "").strip()
_API_TOKEN    = os.environ.get("JWT_EMBED_TOKEN",     "").strip()  # optional pre-issued token

# ---------------------------------------------------------------------------
# Page config – must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=_TITLE,
    page_icon="🏀",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Inject brand CSS
# ---------------------------------------------------------------------------

st.markdown(
    f"""
<style>
:root {{
    --color-primary: {_PRIMARY};
}}
/* Remove default Streamlit chrome to give an embed-friendly look */
#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stToolbar"] {{ display: none !important; }}
.block-container {{ padding-top: 1rem !important; padding-bottom: 1rem !important; }}

/* Chat header */
.embed-header {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    background: var(--color-primary);
    border-radius: 10px;
    margin-bottom: 1rem;
    color: #fff;
}}
.embed-header img {{ height: 36px; border-radius: 4px; }}
.embed-header h1 {{
    font-size: 1.2rem;
    margin: 0;
    font-weight: 700;
    color: #fff;
}}

/* Chat bubbles */
.msg-user {{
    background: var(--color-primary);
    color: #fff;
    border-radius: 14px 14px 4px 14px;
    padding: 10px 14px;
    margin: 4px 0 4px auto;
    max-width: 80%;
    width: fit-content;
    font-size: 0.95rem;
}}
.msg-ai {{
    background: #f0f4f8;
    color: #1a1a2e;
    border-radius: 14px 14px 14px 4px;
    padding: 10px 14px;
    margin: 4px auto 4px 0;
    max-width: 80%;
    width: fit-content;
    font-size: 0.95rem;
}}
.embed-footer {{
    text-align: center;
    color: #888;
    font-size: 0.78rem;
    margin-top: 8px;
}}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

logo_html = f'<img src="{_LOGO_URL}" alt="logo" />' if _LOGO_URL else ""
st.markdown(
    f'<div class="embed-header">{logo_html}<h1>{_TITLE}</h1></div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "embed_messages" not in st.session_state:
    st.session_state.embed_messages = []
if "embed_session_id" not in st.session_state:
    st.session_state.embed_session_id = str(uuid.uuid4())

# ---------------------------------------------------------------------------
# Render conversation history
# ---------------------------------------------------------------------------

for msg in st.session_state.embed_messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="msg-ai">{msg["content"]}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

prompt = st.chat_input("Ask me about a player…")

if prompt:
    # Append user message immediately.
    st.session_state.embed_messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="msg-user">{prompt}</div>', unsafe_allow_html=True)

    # Call the FastAPI chat endpoint.
    headers: dict = {"Content-Type": "application/json"}
    if _API_TOKEN:
        headers["Authorization"] = f"Bearer {_API_TOKEN}"

    suggestions: list = []
    try:
        resp = requests.post(
            f"{_API_URL}/api/v1/chat",
            json={
                "message":    prompt,
                "session_id": st.session_state.embed_session_id,
            },
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        reply = data.get("reply", "")
        suggestions = data.get("suggestions", [])
        # Update session_id from server so context is preserved across turns.
        st.session_state.embed_session_id = data.get(
            "session_id", st.session_state.embed_session_id
        )
    except requests.exceptions.ConnectionError:
        reply = "⚠️ Cannot reach the AI server. Please try again later."
    except requests.exceptions.Timeout:
        reply = "⚠️ The AI server took too long to respond. Please try again."
    except Exception:  # noqa: BLE001
        reply = "⚠️ An unexpected error occurred. Please try again later."

    st.session_state.embed_messages.append({"role": "assistant", "content": reply})
    st.markdown(f'<div class="msg-ai">{reply}</div>', unsafe_allow_html=True)

    # Render suggestion chips if present.
    if suggestions:
        st.markdown("**Suggestions:**")
        cols = st.columns(min(len(suggestions), 3))
        for i, sug in enumerate(suggestions[:3]):
            with cols[i]:
                if st.button(sug, key=f"sug_{i}_{len(st.session_state.embed_messages)}"):
                    st.session_state.embed_messages.append({"role": "user", "content": sug})
                    st.rerun()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

if _FOOTER:
    st.markdown(f'<div class="embed-footer">{_FOOTER}</div>', unsafe_allow_html=True)
