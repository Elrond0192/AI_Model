"""FASE D — ingest raw physical-table CSV exports into the canonical POC dump.

The user exports the physical HoopmetricsEngine tables to CSV (plain export,
no psql scripts, no DDL, no complex SQL) and this module normalizes them into
the canonical ``player_stats.csv`` consumed by
``basketball_ai.metric_rating.poc``.

Expected files in ``--raw`` (names are case-insensitive):

    advancedstats_player_<LEAGUE>.csv   <- Analisi.AdvancedStats_Player_*
    anagrafiche_<LEAGUE>.csv            <- Anagrafiche.<LEAGUE>

Example::

    python -m basketball_ai.metric_rating.ingest \
        --raw data/metric_poc/raw \
        --out data/metric_poc/player_stats.csv

The normalization mirrors the ``"AI_Source"`` adapter: column names are
lowercased and looked up by candidate keys, so the physical column casing does
not matter. NULL metric values are preserved (they become empty cells in the
canonical CSV and are dropped by the POC population builder).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

_STATS_PREFIX = "advancedstats_player_"
_REGISTRY_PREFIX = "anagrafiche_"

_NUM_RE = re.compile(r"^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$")
_COMPETITION_MAP = {
    "": "RS", "REGULAR": "RS", "REGULAR SEASON": "RS", "REG SEASON": "RS", "RS": "RS",
    "PLAYOFF": "PO", "PLAYOFFS": "PO", "POSTSEASON": "PO", "PO": "PO",
    "TOTAL": "TOT", "ALL": "TOT", "TOT": "TOT",
    "SUPER CUP": "SUPERCUP", "SUPERCUP": "SUPERCUP", "CUP": "CUP",
}

CANONICAL_COLUMNS = (
    "league_key", "season", "competition", "player_id", "player_global_id",
    "player_name", "position", "games_played", "minutes_per_game",
    "raptor_total", "lebron_total", "vorp", "games_started", "starter_pct",
)


# ---------------------------------------------------------------------------
# Pure normalization helpers (mirror "AI_Source"."TextValue"/"NumericValue"/…)
# ---------------------------------------------------------------------------

def lower_keys(row: dict[str, Any]) -> dict[str, Any]:
    """Lowercase column names (physical tables use mixed case)."""
    return {str(key).lower(): value for key, value in row.items()}


def text_value(doc: dict[str, Any], *candidates: str) -> Optional[str]:
    for candidate in candidates:
        value = doc.get(candidate.lower())
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def numeric_value(doc: dict[str, Any], *candidates: str) -> Optional[float]:
    value = text_value(doc, *candidates)
    if value is None:
        return None
    value = value.replace("%", "").replace(",", ".")
    if _NUM_RE.match(value):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def int_value(doc: dict[str, Any], *candidates: str) -> Optional[int]:
    value = numeric_value(doc, *candidates)
    return None if value is None else int(value)  # truncates toward zero


def competition_code(value: Optional[str]) -> str:
    text = (value or "").strip().upper()
    if text in _COMPETITION_MAP:
        return _COMPETITION_MAP[text]
    return re.sub(r"[^A-Z0-9]", "", text)


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------

def normalize_player_row(doc: dict[str, Any], league_key: str) -> Optional[dict[str, Any]]:
    """One physical AdvancedStats row -> canonical dict (None if unusable)."""
    season = int_value(doc, "season")
    player_id = text_value(doc, "id", "playerid", "idplayer")
    if season is None or player_id is None:
        return None
    return {
        "league_key": league_key,
        "season": season,
        "competition": competition_code(text_value(doc, "competition")),
        "source_player_id": player_id,
        "games": numeric_value(doc, "games", "gamesplayed"),
        "minutes_total": numeric_value(doc, "min", "minutes"),
        "raptor_total": numeric_value(doc, "raptortotal", "raptor_total"),
        "lebron_total": numeric_value(doc, "lebrontotal", "lebron_total"),
        "vorp": numeric_value(doc, "vorp"),
        "games_started": numeric_value(doc, "gamesstarted", "games_started"),
        "starter_pct": numeric_value(doc, "starterpct", "starter_pct"),
    }


def normalize_registry_row(doc: dict[str, Any], league_key: str) -> Optional[dict[str, Any]]:
    """One physical Anagrafiche row -> identity dict (None if unusable)."""
    season = int_value(doc, "season")
    player_id = text_value(doc, "id")
    if season is None or player_id is None:
        return None
    global_id = text_value(doc, "idglobal", "globalid") or f"{league_key}:{player_id}"
    return {
        "league_key": league_key,
        "season": season,
        "source_player_id": player_id,
        "player_global_id": global_id,
        "player_name": text_value(doc, "normalizedplayername", "playername", "name")
        or player_id,
        "position": text_value(doc, "pos", "position") or "PG",
    }


def dedupe_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the best row per (player, league, season, competition): most games,
    then most minutes — mirrors the adapter's row_number() ordering. Registry
    rows (no competition column) are deduped per (player, league, season)."""
    best: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            row["source_player_id"],
            row["league_key"],
            row["season"],
            row.get("competition"),
        )
        current = best.get(key)
        if current is None:
            best[key] = row
            continue
        games = row.get("games") or 0.0
        minutes = row.get("minutes_total") or 0.0
        cur_games = current.get("games") or 0.0
        cur_minutes = current.get("minutes_total") or 0.0
        if (games, minutes) > (cur_games, cur_minutes):
            best[key] = row
    return list(best.values())


def build_canonical_frame(
    player_rows: Iterable[dict[str, Any]],
    registry_rows: Iterable[dict[str, Any]],
) -> pd.DataFrame:
    """Join stats + identity, compute minutes_per_game, canonical columns."""
    stats = pd.DataFrame(dedupe_rows(player_rows))
    registry = pd.DataFrame(dedupe_rows(registry_rows))
    if stats.empty:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    merged = stats.merge(
        registry,
        on=["league_key", "season", "source_player_id"],
        how="left",
        suffixes=("", "_reg"),
    ) if not registry.empty else stats.copy()
    merged["minutes_per_game"] = merged.apply(
        lambda r: (r["minutes_total"] / r["games"]) if (r.get("games") or 0) > 0 else None,
        axis=1,
    )
    merged["player_id"] = merged["source_player_id"]
    merged["games_played"] = merged["games"]
    # Fill any canonical column that the source rows do not carry (robustness).
    for column in CANONICAL_COLUMNS:
        if column not in merged.columns:
            merged[column] = None
    # pandas treats a tuple as a single label; column selection needs a list.
    out = merged[list(CANONICAL_COLUMNS)]
    return out.sort_values(
        ["league_key", "season", "competition", "player_id"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Raw directory reading
# ---------------------------------------------------------------------------

def _league_token(name: str, marker: str) -> Optional[str]:
    """Extract the league key right after *marker* (case-insensitive), e.g.
    ``Analisi_AdvancedStats_Player_ITA1_export_...`` -> ``ITA1``."""
    index = name.lower().find(marker)
    if index < 0:
        return None
    rest = name[index + len(marker):]
    match = re.match(r"([A-Za-z0-9]+)", rest)
    return match.group(1).upper() if match else None


def _discover(raw_dir: Path) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]]]:
    """Classify the exported CSVs by marker in the file name (case-insensitive):
    ``advancedstats_player_<LEAGUE>`` for stats, ``anagrafiche_<LEAGUE>`` for
    registries. Accepts plain names (``advancedstats_player_ITA1.csv``) as well
    as tool-style names (``Analisi_AdvancedStats_Player_ITA1_export_...csv``)."""
    stats: list[tuple[str, Path]] = []
    registries: list[tuple[str, Path]] = []
    for path in sorted(raw_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".csv":
            continue
        league = _league_token(path.name, _STATS_PREFIX)
        if league:
            stats.append((league, path))
            continue
        league = _league_token(path.name, _REGISTRY_PREFIX)
        if league:
            registries.append((league, path))
    return stats, registries


def ingest_raw(
    raw_dir: str | Path,
    out_path: str | Path,
) -> dict[str, Any]:
    """Normalize raw physical-table exports into the canonical POC dump."""
    raw = Path(raw_dir)
    stats_files, registry_files = _discover(raw)
    if not stats_files:
        raise ValueError(
            f"No {_STATS_PREFIX}*.csv files found in {raw}; export "
            "Analisi.AdvancedStats_Player_* tables as CSV."
        )

    player_rows: list[dict[str, Any]] = []
    for league_key, path in stats_files:
        frame = pd.read_csv(path)
        for _, row in frame.iterrows():
            normalized = normalize_player_row(lower_keys(row.to_dict()), league_key)
            if normalized is not None:
                player_rows.append(normalized)

    registry_rows: list[dict[str, Any]] = []
    for league_key, path in registry_files:
        frame = pd.read_csv(path)
        for _, row in frame.iterrows():
            normalized = normalize_registry_row(lower_keys(row.to_dict()), league_key)
            if normalized is not None:
                registry_rows.append(normalized)

    canonical = build_canonical_frame(player_rows, registry_rows)
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(output, index=False)

    return {
        "raw_dir": str(raw),
        "stats_files": [str(path) for _, path in stats_files],
        "registry_files": [str(path) for _, path in registry_files],
        "rows_exported": len(canonical),
        "leagues": sorted(canonical["league_key"].dropna().unique().tolist())
        if not canonical.empty
        else [],
        "out": str(output),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m basketball_ai.metric_rating.ingest",
        description="Normalize raw physical-table CSV exports into the POC dump.",
    )
    parser.add_argument("--raw", required=True, help="Directory with the exported CSVs")
    parser.add_argument("--out", default="data/metric_poc/player_stats.csv")
    args = parser.parse_args(argv)
    summary = ingest_raw(args.raw, args.out)
    print(
        f"OK: {summary['rows_exported']} rows, leagues={summary['leagues']} "
        f"-> {summary['out']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
