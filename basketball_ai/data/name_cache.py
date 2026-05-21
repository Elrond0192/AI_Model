"""Application-level name resolution cache.

Maps player/team display names to their IdGlobal values and back.
Persisted to JSON for cross-restart survival; evicted entries are
re-fetched from the data layer.

No DB dependencies – replaces any hypothetical Lookup table.

Usage::

    cache = NameResolutionCache()
    id_ = cache.resolve_player("LeBron James")  # returns IdGlobal or None
    cache.set_player("LeBron James", 42)
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

_logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(os.environ.get("NAME_CACHE_FILE", "data/name_resolution_cache.json"))
_DEFAULT_TTL  = int(os.environ.get("NAME_CACHE_TTL_SECONDS", str(24 * 3600)))  # 24h


class NameResolutionCache:
    """JSON-backed name → ID cache with per-entry TTL.

    Keys are lowercased, stripped names. Values are ``(id, expiry_ts)`` pairs.
    The file is rewritten atomically on every mutation.
    """

    def __init__(
        self,
        path: Path = _DEFAULT_PATH,
        ttl: int = _DEFAULT_TTL,
    ) -> None:
        self.path = Path(path)
        self.ttl  = ttl
        self._players: Dict[str, Tuple[int, float]] = {}
        self._teams:   Dict[str, Tuple[int, float]] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve_player(self, name: str) -> Optional[int]:
        """Return IdGlobal for *name*, or None if missing / expired."""
        return self._get(self._players, name)

    def resolve_team(self, name: str) -> Optional[int]:
        """Return IdGlobal for *name*, or None if missing / expired."""
        return self._get(self._teams, name)

    def set_player(self, name: str, id_: int) -> None:
        self._set(self._players, name, id_)

    def set_team(self, name: str, id_: int) -> None:
        self._set(self._teams, name, id_)

    def invalidate_player(self, name: str) -> None:
        self._players.pop(self._key(name), None)
        self._persist()

    def invalidate_team(self, name: str) -> None:
        self._teams.pop(self._key(name), None)
        self._persist()

    def stats(self) -> dict:
        now = time.time()
        valid_p = sum(1 for _, (_, exp) in self._players.items() if exp > now)
        valid_t = sum(1 for _, (_, exp) in self._teams.items() if exp > now)
        return {
            "players": {"total": len(self._players), "valid": valid_p},
            "teams":   {"total": len(self._teams),   "valid": valid_t},
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(name: str) -> str:
        return name.strip().lower()

    def _get(self, store: Dict[str, Tuple[int, float]], name: str) -> Optional[int]:
        key = self._key(name)
        entry = store.get(key)
        if entry is None:
            return None
        id_, expiry = entry
        if time.time() > expiry:
            del store[key]
            self._persist()
            return None
        return id_

    def _set(self, store: Dict[str, Tuple[int, float]], name: str, id_: int) -> None:
        key = self._key(name)
        store[key] = (id_, time.time() + self.ttl)
        self._persist()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._players = {
                k: tuple(v) for k, v in raw.get("players", {}).items()
                if isinstance(v, (list, tuple)) and len(v) == 2
            }
            self._teams = {
                k: tuple(v) for k, v in raw.get("teams", {}).items()
                if isinstance(v, (list, tuple)) and len(v) == 2
            }
            _logger.debug("[NameCache] Loaded from %s", self.path)
        except Exception as exc:
            _logger.warning("[NameCache] Could not load cache file: %s", exc)

    def _persist(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {"players": {k: list(v) for k, v in self._players.items()},
                     "teams":   {k: list(v) for k, v in self._teams.items()}},
                    indent=2,
                ),
                encoding="utf-8",
            )
            tmp.replace(self.path)
        except Exception as exc:
            _logger.warning("[NameCache] Could not persist cache: %s", exc)
