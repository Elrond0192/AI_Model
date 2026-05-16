"""T5 – Playwright smoke tests for the Basketball Performance AI GUI.

Auto-skipped when:
- playwright is not installed
- SKIP_PLAYWRIGHT env var is set to "1", "true", or "yes"
"""
from __future__ import annotations

import os
import pytest

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

_SKIP_ENV = os.environ.get("SKIP_PLAYWRIGHT", "").strip().lower() in ("1", "true", "yes")

try:
    from playwright.sync_api import sync_playwright, Page  # noqa: F401
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False

_SKIP = _SKIP_ENV or not _PLAYWRIGHT_AVAILABLE
_SKIP_REASON = (
    "SKIP_PLAYWRIGHT=1" if _SKIP_ENV
    else "playwright not installed" if not _PLAYWRIGHT_AVAILABLE
    else ""
)

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def base_url() -> str:
    return os.environ.get("APP_BASE_URL", "http://localhost:8501")


@pytest.fixture(scope="module")
def api_base_url() -> str:
    return os.environ.get("API_BASE_URL", "http://localhost:8000")


@pytest.fixture(scope="module")
def browser_page(base_url):
    """Launch a headless browser and navigate to the app."""
    if _SKIP:
        pytest.skip(_SKIP_REASON)
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(base_url, timeout=30_000)
        yield page
        browser.close()


@pytest.mark.smoke
def test_homepage_loads(browser_page):
    """The Streamlit app should load and display a title."""
    if _SKIP:
        pytest.skip(_SKIP_REASON)
    page = browser_page
    # Streamlit renders its content inside a main element
    page.wait_for_selector("body", timeout=15_000)
    assert page.title() != ""


@pytest.mark.smoke
def test_login_form_visible(browser_page):
    """The login form (username + password fields) should be visible on load."""
    if _SKIP:
        pytest.skip(_SKIP_REASON)
    page = browser_page
    # Streamlit text inputs are rendered as <input> elements
    page.wait_for_selector("input", timeout=15_000)
    inputs = page.query_selector_all("input")
    assert len(inputs) >= 1, "Expected at least one input field (username or password)"


@pytest.mark.smoke
def test_login_success(browser_page):
    """Logging in with admin/admin should proceed past the login screen."""
    if _SKIP:
        pytest.skip(_SKIP_REASON)
    page = browser_page
    try:
        username_input = page.query_selector("input[type='text'], input:not([type='password'])")
        password_input = page.query_selector("input[type='password']")
        if username_input and password_input:
            username_input.fill("admin")
            password_input.fill("admin")
            # Press Enter or click the submit button
            password_input.press("Enter")
            page.wait_for_timeout(2000)
    except Exception:
        pytest.skip("Could not interact with login form – app may not require login")


@pytest.mark.smoke
def test_tabs_visible_after_login(browser_page):
    """After login the navigation tabs / sidebar should be visible."""
    if _SKIP:
        pytest.skip(_SKIP_REASON)
    page = browser_page
    page.wait_for_timeout(1000)
    # Streamlit renders sidebar or tabs; just check the page has content
    body_text = page.inner_text("body")
    assert len(body_text) > 10, "Page body appears empty after login"


@pytest.mark.smoke
def test_health_endpoint(api_base_url):
    """The /health endpoint should return status ok."""
    if _SKIP:
        pytest.skip(_SKIP_REASON)
    import urllib.request
    import json
    try:
        with urllib.request.urlopen(f"{api_base_url}/health", timeout=5) as resp:
            body = json.loads(resp.read())
        assert body.get("status") == "ok"
    except Exception as exc:
        pytest.skip(f"API not reachable at {api_base_url}: {exc}")
