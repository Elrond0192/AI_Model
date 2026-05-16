"""U3 – Minimal i18n helper. Usage: from i18n import t; t("key") or t("key", lang="en")"""
from __future__ import annotations
import json, logging, os
from functools import lru_cache
from pathlib import Path
from typing import Any
logger = logging.getLogger(__name__)
_I18N_DIR = Path(__file__).parent
_DEFAULT_LANG = os.environ.get("APP_LANG", "it")

@lru_cache(maxsize=8)
def _load(lang: str) -> dict[str, str]:
    path = _I18N_DIR / f"{lang}.json"
    if not path.exists():
        logger.warning("[i18n] Missing translation file: %s – falling back to 'it'", path)
        path = _I18N_DIR / "it.json"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)

def t(key: str, lang: str | None = None, **kwargs: Any) -> str:
    lang = lang or _DEFAULT_LANG
    value = _load(lang).get(key, key)
    if kwargs:
        try:
            value = value.format(**kwargs)
        except (KeyError, ValueError):
            pass
    return value
