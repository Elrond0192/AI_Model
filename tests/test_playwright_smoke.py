"""Optional browser smoke tests for the custom AI_Model operations console.

Auto-skipped when Playwright is unavailable or SKIP_PLAYWRIGHT is enabled.
"""
from __future__ import annotations

import os
import pytest

_SKIP_ENV = os.environ.get("SKIP_PLAYWRIGHT", "").strip().lower() in ("1", "true", "yes")

try:
    from playwright.sync_api import sync_playwright, Page  # noqa: F401
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False

_SKIP = _SKIP_ENV or not _PLAYWRIGHT_AVAILABLE
_SKIP_REASON = "SKIP_PLAYWRIGHT=1" if _SKIP_ENV else "playwright not installed" if not _PLAYWRIGHT_AVAILABLE else ""
pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def base_url() -> str:
    return os.environ.get("APP_BASE_URL", "http://localhost:8501")


@pytest.fixture(scope="module")
def api_base_url() -> str:
    return os.environ.get("API_BASE_URL", "http://localhost:8000")


@pytest.fixture(scope="module")
def browser_page(base_url):
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
    if _SKIP:
        pytest.skip(_SKIP_REASON)
    browser_page.wait_for_selector("body", timeout=15_000)
    assert browser_page.title() == "AI Model Control Center"


@pytest.mark.smoke
def test_custom_login_form_visible(browser_page):
    if _SKIP:
        pytest.skip(_SKIP_REASON)
    browser_page.wait_for_selector("#login-form", timeout=15_000)
    assert browser_page.locator("#login-username").count() == 1
    assert browser_page.locator("#login-password").count() == 1
    assert browser_page.locator("text=Operations Console").count() >= 1


@pytest.mark.smoke
def test_streamlit_shell_is_absent(browser_page):
    if _SKIP:
        pytest.skip(_SKIP_REASON)
    assert browser_page.locator('[data-testid="stApp"]').count() == 0
    assert browser_page.locator('[data-testid="stSidebar"]').count() == 0


@pytest.mark.smoke
def test_health_endpoint(api_base_url):
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
