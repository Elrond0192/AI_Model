"""Serving helper for the validated BB-Rating uncertainty layer."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class BBRatingUncertainty:
    """Load and serve the final Calibration 1.17 empirical tables."""

    def __init__(self, artifact: dict[str, Any]) -> None:
        if artifact.get("status") != "fitted":
            raise ValueError("BB-Rating uncertainty artifact is not fitted")
        self.artifact = artifact
        self.calibration_version = str(artifact.get("calibration_version", "1.17"))
        self.bb_rating_version = str(artifact.get("bb_rating_version", "1.9"))
        self.min_samples = int(artifact.get("min_samples", 50))
        tables = artifact.get("tables") or {}
        self.global_table = tables.get("global")
        self.league_tables = {
            str(row["league_key"]): dict(row)
            for row in tables.get("league", [])
            if isinstance(row, dict) and row.get("league_key") is not None
        }
        self.exposure_tables = {
            str(int(row["exposure_band"])): dict(row)
            for row in tables.get("exposure", [])
            if isinstance(row, dict) and row.get("exposure_band") is not None
        }
        self.league_exposure_tables = {
            (str(row["league_key"]), str(int(row["exposure_band"]))): dict(row)
            for row in tables.get("league_exposure", [])
            if (
                isinstance(row, dict)
                and row.get("league_key") is not None
                and row.get("exposure_band") is not None
            )
        }
        definition = artifact.get("exposure_band_definition") or {}
        self.global_edges = self._edges(definition.get("global_edges_minutes"))
        self.league_edges = {
            str(league): self._edges(edges)
            for league, edges in (definition.get("league_edges_minutes") or {}).items()
        }

    @staticmethod
    def _edges(value: Any) -> np.ndarray | None:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return None
        try:
            edges = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            return None
        return edges if np.isfinite(edges).all() else None

    @classmethod
    def from_file(cls, path: str | Path) -> "BBRatingUncertainty":
        artifact_path = Path(path)
        if not artifact_path.exists():
            raise FileNotFoundError(str(artifact_path))
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("BB-Rating uncertainty artifact must be a JSON object")
        return cls(payload)

    @staticmethod
    def _band(value: Any, edges: np.ndarray | None) -> int | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric) or edges is None:
            return None
        return int(np.clip(np.searchsorted(edges, numeric, side="right") + 1, 1, 4))

    @staticmethod
    def _exposure_minutes(row: pd.Series) -> float | None:
        if "minutes_total" in row.index:
            value = pd.to_numeric(row.get("minutes_total"), errors="coerce")
            if pd.notna(value) and np.isfinite(float(value)):
                return float(value)

        games = pd.to_numeric(row.get("games_played"), errors="coerce")
        mpg = pd.to_numeric(row.get("minutes_per_game"), errors="coerce")
        if pd.notna(games) and pd.notna(mpg):
            value = float(games) * float(mpg)
            if np.isfinite(value):
                return value
        return None

    @staticmethod
    def _compact_table(table: dict[str, Any], source: str) -> dict[str, Any]:
        return {
            "source": source,
            "sample_size": int(table["n"]),
            "p50": float(table["p50"]),
            "p75": float(table["p75"]),
            "p90": float(table["p90"]),
        }

    def for_player(
        self,
        frame: pd.DataFrame,
        *,
        player_global_id: str,
        league: str,
        season: int,
        phase: str,
        score: int,
    ) -> dict[str, Any]:
        league_key = str(league).strip().upper()
        phase_key = str(phase).strip().upper()
        season_year = int(str(season).split("-")[0])

        if frame.empty:
            return {
                "available": False,
                "reason": "player_frame_unavailable",
                "calibration_version": self.calibration_version,
                "bb_rating_version": self.bb_rating_version,
            }

        matches = frame.loc[
            frame["league_key"].astype(str).str.upper().eq(league_key)
            & pd.to_numeric(frame["season"], errors="coerce").eq(season_year)
            & frame["competition"].astype(str).str.upper().eq(phase_key)
            & frame["player_global_id"].astype(str).eq(str(player_global_id))
        ]
        if matches.empty:
            return {
                "available": False,
                "reason": "player_exposure_unavailable",
                "calibration_version": self.calibration_version,
                "bb_rating_version": self.bb_rating_version,
            }

        exposure_minutes = self._exposure_minutes(matches.iloc[0])
        league_band = self._band(
            exposure_minutes,
            self.league_edges.get(league_key),
        )
        global_band = self._band(exposure_minutes, self.global_edges)

        table: dict[str, Any] | None = None
        source = "unavailable"

        if league_band is not None:
            table = self.league_exposure_tables.get(
                (league_key, str(league_band))
            )
            if table is not None:
                source = "league+exposure"

        if table is None:
            table = self.league_tables.get(league_key)
            if table is not None:
                source = "league"

        if table is None and global_band is not None:
            table = self.exposure_tables.get(str(global_band))
            if table is not None:
                source = "exposure"

        if table is None and self.global_table is not None:
            table = self.global_table
            source = "global"

        if table is None:
            return {
                "available": False,
                "reason": "no_supported_uncertainty_cell",
                "calibration_version": self.calibration_version,
                "bb_rating_version": self.bb_rating_version,
                "exposure_minutes": exposure_minutes,
                "exposure_band": league_band,
                "exposure_band_scope": "league" if league_band is not None else "global",
            }

        p50 = float(table["p50"])
        p75 = float(table["p75"])
        p90 = float(table["p90"])
        lower = max(1, int(np.floor(float(score) - p90)))
        upper = min(100, int(np.ceil(float(score) + p90)))

        return {
            "available": True,
            "calibration_version": self.calibration_version,
            "bb_rating_version": self.bb_rating_version,
            "source": source,
            "sample_size": int(table["n"]),
            "exposure_minutes": (
                round(exposure_minutes, 1) if exposure_minutes is not None else None
            ),
            "exposure_band": league_band if league_band is not None else global_band,
            "exposure_band_scope": "league" if league_band is not None else "global",
            "absolute_change": {
                "p50": p50,
                "p75": p75,
                "p90": p90,
            },
            "p90_score_range": {
                "lower": lower,
                "upper": upper,
            },
            "interpretation": (
                "P50/P75/P90 indicano soglie empiriche sulla variazione assoluta "
                "del BB-Rating osservata tra stagioni consecutive. Il range P90 "
                "è l'intervallo simmetrico attorno al rating corrente coerente "
                "con tale soglia."
            ),
        }


__all__ = ["BBRatingUncertainty"]
