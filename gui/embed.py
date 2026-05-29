"""Basketball Performance AI – Brandable Dash embed widget.

A minimal, white-labelable Dash chat page designed to be iframed into a
customer's website or served as a standalone lightweight chat portal.

All visual elements (title, colours, logo, footer) are driven by environment
variables so that the same Docker image can be reused for multiple customers
without any code changes.

Environment variables
---------------------
BRAND_TITLE         Title shown in the header (default: "Basketball Performance AI")
BRAND_PRIMARY_COLOR Primary accent colour in CSS hex (default: "#1565C0")
BRAND_LOGO_URL      Optional URL to a PNG/SVG logo. Empty → no logo shown.
BRAND_API_URL       Base URL of the FastAPI server (default: "http://localhost:8000")
BRAND_FOOTER_TEXT   Optional disclaimer / tagline shown below the chat input.
JWT_EMBED_TOKEN     Optional pre-issued JWT forwarded to the FastAPI chat endpoint.
EMBED_PORT          Port to listen on (default: 8502).
EMBED_DEBUG         Set to "true" to enable Dash hot-reload / debug mode.

Launch
------
    python gui/embed.py

Or via docker-compose:
    docker compose up embed
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import dash
from dash import Input, Output, State, callback_context, dcc, html, no_update

# ---------------------------------------------------------------------------
# Branding – read from environment
# ---------------------------------------------------------------------------

_TITLE     = os.environ.get("BRAND_TITLE",         "Basketball Performance AI").strip()
_PRIMARY   = os.environ.get("BRAND_PRIMARY_COLOR", "#1565C0").strip()
_LOGO_URL  = os.environ.get("BRAND_LOGO_URL",      "").strip()
_API_URL   = os.environ.get("BRAND_API_URL",       "http://localhost:8000").rstrip("/")
_FOOTER    = os.environ.get("BRAND_FOOTER_TEXT",   "").strip()
_API_TOKEN = os.environ.get("JWT_EMBED_TOKEN",     "").strip()

_PORT  = int(os.environ.get("EMBED_PORT",  "8502"))
_DEBUG = os.environ.get("EMBED_DEBUG", "").lower() in ("true", "1", "yes")

# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    title=_TITLE,
    update_title=None,           # suppress "Updating…" browser tab flicker
    suppress_callback_exceptions=True,
)
server = app.server  # expose Flask server for WSGI deployment (gunicorn)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

_logo_el = html.Img(src=_LOGO_URL, alt="logo", className="embed-logo") if _LOGO_URL else None

app.layout = html.Div(
    className="embed-root",
    children=[
        # Persistent client-side state (stored in the browser, never on server)
        dcc.Store(id="store-messages", data=[]),
        dcc.Store(id="store-session",  data={"session_id": str(uuid.uuid4())}),

        # Header
        html.Div(
            className="embed-header",
            children=[*([_logo_el] if _logo_el else []), html.H1(_TITLE, className="embed-title")],
        ),

        # Conversation window
        html.Div(id="chat-window", className="chat-window"),

        # Suggestion chips (empty until the API returns suggestions)
        html.Div(id="suggestion-chips", className="suggestion-chips"),

        # Input row
        html.Div(
            className="input-row",
            children=[
                dcc.Input(
                    id="chat-input",
                    type="text",
                    placeholder="Ask me about a player…",
                    debounce=False,
                    className="chat-input",
                    n_submit=0,
                    autoComplete="off",
                ),
                html.Button("➤", id="send-btn", className="send-btn", n_clicks=0),
            ],
        ),

        # Optional footer / disclaimer
        *([html.Div(_FOOTER, className="embed-footer")] if _FOOTER else []),
    ],
)

# ---------------------------------------------------------------------------
# CSS + page chrome injected via index_string.
# The CSS variables --primary and --primary-dark drive all brand colours so
# that changing BRAND_PRIMARY_COLOR is the only required customisation.
# ---------------------------------------------------------------------------

app.index_string = f"""<!DOCTYPE html>
<html lang="en">
<head>
  {{%metas%}}
  <title>{{%title%}}</title>
  {{%favicon%}}
  {{%css%}}
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {{
      --primary:      {_PRIMARY};
      --primary-dark: color-mix(in srgb, {_PRIMARY} 78%, #000);
    }}
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: #f5f7fa;
      height: 100dvh;
      display: flex;
      justify-content: center;
      align-items: stretch;
    }}
    .embed-root {{
      display: flex;
      flex-direction: column;
      width: 100%;
      max-width: 700px;
      height: 100dvh;
      background: #fff;
      box-shadow: 0 0 24px rgba(0,0,0,.08);
    }}
    /* Header */
    .embed-header {{
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 14px 20px;
      background: var(--primary);
      flex-shrink: 0;
    }}
    .embed-logo  {{ height: 36px; border-radius: 4px; }}
    .embed-title {{ font-size: 1.15rem; font-weight: 700; color: #fff; }}
    /* Chat window */
    .chat-window {{
      flex: 1;
      overflow-y: auto;
      padding: 16px 20px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      scroll-behavior: smooth;
    }}
    .msg-user {{
      background: var(--primary);
      color: #fff;
      border-radius: 18px 18px 4px 18px;
      padding: 10px 16px;
      margin-left: auto;
      max-width: 80%;
      font-size: .9rem;
      line-height: 1.5;
      word-break: break-word;
    }}
    .msg-ai {{
      background: #f0f4f8;
      color: #1a1a2e;
      border-radius: 18px 18px 18px 4px;
      padding: 10px 16px;
      margin-right: auto;
      max-width: 80%;
      font-size: .9rem;
      line-height: 1.5;
      word-break: break-word;
      white-space: pre-wrap;
    }}
    /* Suggestion chips */
    .suggestion-chips {{
      padding: 4px 20px 8px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      flex-shrink: 0;
    }}
    .suggestion-chip {{
      background: var(--primary);
      color: #fff;
      border: none;
      border-radius: 9999px;
      padding: 6px 14px;
      font-size: .82rem;
      cursor: pointer;
      transition: background .15s;
    }}
    .suggestion-chip:hover {{ background: var(--primary-dark); }}
    /* Input row */
    .input-row {{
      display: flex;
      gap: 8px;
      padding: 12px 20px;
      border-top: 1px solid #e2e8f0;
      flex-shrink: 0;
    }}
    .chat-input {{
      flex: 1;
      border: 1px solid #d1d9e6;
      border-radius: 9999px;
      padding: 10px 18px;
      font-size: .9rem;
      outline: none;
      transition: border-color .15s;
    }}
    .chat-input:focus {{ border-color: var(--primary); }}
    .send-btn {{
      background: var(--primary);
      color: #fff;
      border: none;
      border-radius: 9999px;
      width: 42px;
      height: 42px;
      font-size: 1.1rem;
      cursor: pointer;
      flex-shrink: 0;
      transition: background .15s;
    }}
    .send-btn:hover {{ background: var(--primary-dark); }}
    /* Footer */
    .embed-footer {{
      text-align: center;
      color: #888;
      font-size: .75rem;
      padding: 6px 20px 10px;
      flex-shrink: 0;
    }}
  </style>
  <script>
    /* Auto-scroll chat window to the latest message on every DOM update */
    const _scrollBottom = () => {{
      const w = document.getElementById('chat-window');
      if (w) w.scrollTop = w.scrollHeight;
    }};
    const _mo = new MutationObserver(_scrollBottom);
    document.addEventListener('DOMContentLoaded', () => {{
      const w = document.getElementById('chat-window');
      if (w) _mo.observe(w, {{ childList: true, subtree: true }});
    }});
  </script>
</head>
<body>
  {{%app_entry%}}
  <footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
</body>
</html>"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _call_api(message: str, session_id: str) -> tuple[str, list[str], str]:
    """POST to the FastAPI chat endpoint.

    Returns ``(reply, suggestions, session_id)`` on success, or a user-facing
    error string with empty suggestions when the request fails.
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if _API_TOKEN:
        headers["Authorization"] = f"Bearer {_API_TOKEN}"
    try:
        resp = requests.post(
            f"{_API_URL}/api/v1/chat",
            json={"message": message, "session_id": session_id},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return (
            data.get("reply", ""),
            data.get("suggestions", []),
            data.get("session_id", session_id),
        )
    except requests.exceptions.ConnectionError:
        return "⚠️ Cannot reach the AI server. Please try again later.", [], session_id
    except requests.exceptions.Timeout:
        return "⚠️ The AI server took too long to respond. Please try again.", [], session_id
    except Exception:  # noqa: BLE001
        return "⚠️ An unexpected error occurred. Please try again later.", [], session_id


def _render_messages(messages: list[dict]) -> list:
    """Convert a list of ``{role, content}`` dicts into Dash html elements."""
    return [
        html.Div(m["content"], className="msg-user" if m["role"] == "user" else "msg-ai")
        for m in messages
    ]


def _make_chips(suggestions: list[str]) -> list:
    """Build suggestion-chip button elements from a list of suggestion strings."""
    return [
        html.Button(
            s,
            id={"type": "chip", "index": i},
            className="suggestion-chip",
            n_clicks=0,
        )
        for i, s in enumerate(suggestions[:4])
    ]


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@app.callback(
    Output("store-messages",   "data"),
    Output("store-session",    "data"),
    Output("chat-window",      "children"),
    Output("suggestion-chips", "children"),
    Output("chat-input",       "value"),
    Input("send-btn",          "n_clicks"),
    Input("chat-input",        "n_submit"),
    State("chat-input",        "value"),
    State("store-messages",    "data"),
    State("store-session",     "data"),
    prevent_initial_call=True,
)
def handle_send(
    _n_clicks: int,
    _n_submit: int,
    user_text: str | None,
    messages: list,
    session_state: dict,
) -> tuple:
    """Handle the Send button or Enter-key submission."""
    if not user_text or not user_text.strip():
        return no_update, no_update, no_update, no_update, no_update

    text       = user_text.strip()
    session_id = (session_state or {}).get("session_id") or str(uuid.uuid4())
    messages   = list(messages or [])

    messages.append({"role": "user", "content": text})
    reply, suggestions, new_sid = _call_api(text, session_id)
    messages.append({"role": "assistant", "content": reply})

    return (
        messages,
        {"session_id": new_sid},
        _render_messages(messages),
        _make_chips(suggestions),
        "",   # clear the input field
    )


@app.callback(
    Output("store-messages",   "data",     allow_duplicate=True),
    Output("store-session",    "data",     allow_duplicate=True),
    Output("chat-window",      "children", allow_duplicate=True),
    Output("suggestion-chips", "children", allow_duplicate=True),
    Input({"type": "chip", "index": dash.ALL}, "n_clicks"),
    State({"type": "chip", "index": dash.ALL}, "children"),
    State("store-messages",    "data"),
    State("store-session",     "data"),
    prevent_initial_call=True,
)
def handle_chip(
    n_clicks_list: list[int],
    chip_labels: list[str],
    messages: list,
    session_state: dict,
) -> tuple:
    """Send the text of a tapped suggestion chip as a user message."""
    ctx = callback_context
    if not ctx.triggered or not any(n_clicks_list):
        return no_update, no_update, no_update, no_update

    # Identify which chip fired from the triggered prop_id JSON
    triggered_prop = ctx.triggered[0]["prop_id"]  # e.g. '{"index":2,"type":"chip"}.n_clicks'
    try:
        idx = json.loads(triggered_prop.split(".")[0])["index"]
    except (json.JSONDecodeError, KeyError, IndexError, ValueError):
        return no_update, no_update, no_update, no_update

    if idx >= len(chip_labels):
        return no_update, no_update, no_update, no_update

    text       = chip_labels[idx]
    session_id = (session_state or {}).get("session_id") or str(uuid.uuid4())
    messages   = list(messages or [])

    messages.append({"role": "user", "content": text})
    reply, suggestions, new_sid = _call_api(text, session_id)
    messages.append({"role": "assistant", "content": reply})

    return (
        messages,
        {"session_id": new_sid},
        _render_messages(messages),
        _make_chips(suggestions),
    )


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=_PORT, debug=_DEBUG)
