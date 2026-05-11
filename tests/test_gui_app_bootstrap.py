"""Tests for the Streamlit app import-path bootstrap."""
from __future__ import annotations

import ast
from importlib.machinery import PathFinder
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "gui" / "app.py"


def _is_sys_path_insert(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
        and isinstance(statement.value.func.value, ast.Attribute)
        and isinstance(statement.value.func.value.value, ast.Name)
        and statement.value.func.value.value.id == "sys"
        and statement.value.func.value.attr == "path"
        and statement.value.func.attr == "insert"
    )


def test_gui_app_bootstrap_adds_repo_root_to_sys_path():
    module = ast.parse(APP_PATH.read_text(encoding="utf-8"), filename=str(APP_PATH))
    bootstrap_body = [module.body[0]]

    index = 1
    if isinstance(module.body[index], ast.ImportFrom) and module.body[index].module == "__future__":
        bootstrap_body.append(module.body[index])
        index += 1

    for statement in module.body[index:]:
        bootstrap_body.append(statement)
        if _is_sys_path_insert(statement):
            break

    assert _is_sys_path_insert(bootstrap_body[-1])
    bootstrap = ast.Module(body=bootstrap_body, type_ignores=[])

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
