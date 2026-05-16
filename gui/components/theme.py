"""U2 – Minimal design system: CSS token constants and helpers.

Use inject_css() once per Streamlit page to apply the design system.
Dark mode is detected from DARK_MODE env var ("1"/"true") or Streamlit session_state.
"""
from __future__ import annotations
import os
import streamlit as st

TOKENS = {
    "light": {
        "primary": "#1565C0", "primary_hover": "#0D47A1",
        "secondary": "#E3F2FD", "surface": "#FFFFFF", "surface_alt": "#F0F4F8",
        "text": "#1A1A2E", "text_muted": "#5A6475", "border": "#D1D9E6",
        "success": "#2E7D32", "warning": "#E65100", "error": "#C62828", "info": "#1565C0",
        "rating_high": "#2E7D32", "rating_mid": "#F57F17", "rating_low": "#C62828",
    },
    "dark": {
        "primary": "#42A5F5", "primary_hover": "#90CAF9",
        "secondary": "#1A2744", "surface": "#0E1117", "surface_alt": "#1A1F2E",
        "text": "#E8EAF6", "text_muted": "#9E9E9E", "border": "#2C3349",
        "success": "#66BB6A", "warning": "#FFA726", "error": "#EF5350", "info": "#42A5F5",
        "rating_high": "#66BB6A", "rating_mid": "#FFA726", "rating_low": "#EF5350",
    },
}
SPACING = {"xs": 4, "sm": 8, "md": 16, "lg": 24, "xl": 40}
FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
RADIUS = {"sm": "4px", "md": "8px", "lg": "12px", "pill": "9999px"}

def _is_dark_mode() -> bool:
    env_flag = os.environ.get("DARK_MODE", "").strip().lower()
    if env_flag in ("1", "true", "yes"):
        return True
    # Streamlit exposes the active theme in session_state under a private key.
    # This relies on an undocumented key that may change across Streamlit versions;
    # it is used only as a best-effort hint and falls back to light mode gracefully.
    theme = st.session_state.get("_theme", {})
    return str(theme.get("base", "")).lower() == "dark"

def get_tokens(dark: bool | None = None) -> dict:
    if dark is None:
        dark = _is_dark_mode()
    return TOKENS["dark" if dark else "light"]

def inject_css(dark: bool | None = None) -> None:
    """Inject design-system CSS into the Streamlit page. Idempotent."""
    t = get_tokens(dark)
    css = f"""
<style>
:root {{
    --color-primary: {t['primary']}; --color-primary-hover: {t['primary_hover']};
    --color-surface: {t['surface']}; --color-surface-alt: {t['surface_alt']};
    --color-text: {t['text']}; --color-text-muted: {t['text_muted']};
    --color-border: {t['border']}; --color-success: {t['success']};
    --color-warning: {t['warning']}; --color-error: {t['error']};
}}
.rating-high {{ color: {t['rating_high']}; font-weight: 700; }}
.rating-mid  {{ color: {t['rating_mid']};  font-weight: 700; }}
.rating-low  {{ color: {t['rating_low']};  font-weight: 700; }}
.bai-card {{ background: var(--color-surface-alt); border: 1px solid var(--color-border);
             border-radius: 8px; padding: 16px; margin-bottom: 8px; }}
.text-muted {{ color: var(--color-text-muted); font-size: 0.875rem; }}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)
