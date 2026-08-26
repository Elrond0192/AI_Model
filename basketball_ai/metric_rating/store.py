"""FASE I — access to the stored Metric Rating distributions.

The API serves ratings from the **precomputed, versioned** distributions
(``AI.MetricDistribution``) — never recomputed per request.

- ``load_metric_distributions`` reads them from PostgreSQL (schema ``AI``);
- ``load_metric_distributions_csv`` reads the backfill CSV (POC / local runs).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

_SCHEMA_RE = __import__("re").compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_metric_distributions(
    url: Optional[str] = None,
    schema: str = "AI",
) -> list[dict[str, Any]]:
    """Read ``"<schema>"."MetricDistribution"`` rows as a list of dicts."""
    if not _SCHEMA_RE.fullmatch(schema):
        raise ValueError(f"Invalid schema {schema!r}")
    from sqlalchemy import text

    from basketball_ai.data.postgres_loader import get_engine

    engine = get_engine(url)
    try:
        with engine.connect() as connection:
            statement = text(
                f'SELECT * FROM "{schema}"."MetricDistribution" ORDER BY metric, league_key, season, competition'
            )
            return [dict(row) for row in connection.execute(statement).mappings().all()]
    finally:
        engine.dispose()


def load_metric_distributions_csv(path: str | Path) -> list[dict[str, Any]]:
    """Read a metric_distribution CSV (backfill export) as a list of dicts."""
    import pandas as pd

    frame = pd.read_csv(path)
    return frame.to_dict("records")
