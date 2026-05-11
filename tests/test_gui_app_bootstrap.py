"""Tests for the Streamlit app import-path bootstrap."""
from __future__ import annotations

import ast
from importlib.machinery import PathFinder
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "gui" / "app.py"


def test_gui_app_bootstrap_adds_repo_root_to_sys_path():
    module = ast.parse(APP_PATH.read_text(encoding="utf-8"), filename=str(APP_PATH))
    bootstrap = ast.Module(body=module.body[:5], type_ignores=[])

    original_sys_path = sys.path[:]
    sys.path[:] = [str(APP_PATH.parent)]
    namespace = {"__file__": str(APP_PATH)}

    try:
        assert PathFinder.find_spec("basketball_ai", sys.path) is None

        exec(compile(bootstrap, filename=str(APP_PATH), mode="exec"), namespace)

        assert sys.path[0] == str(REPO_ROOT)
        assert PathFinder.find_spec("basketball_ai", sys.path) is not None
    finally:
        sys.path[:] = original_sys_path
