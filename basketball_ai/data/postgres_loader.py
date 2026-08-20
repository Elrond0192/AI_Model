"""PostgreSQL data access for the production AI service.

The application only knows a SQLAlchemy URL.  Docker resolves the host through
``host.docker.internal``; SSH tunnels and credentials are deliberately outside
the application boundary.
"""
from __future__ import annotations

import os
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


def get_engine(url: str | None = None) -> Engine:
    database_url = url or os.environ.get("DATABASE_URL", "")
    if not database_url.startswith("postgresql+"):
        raise RuntimeError("DATABASE_URL must be a postgresql+psycopg SQLAlchemy URL")
    return create_engine(database_url, pool_pre_ping=True, connect_args={"connect_timeout": int(os.getenv("POSTGRES_CONNECT_TIMEOUT", "10"))})


def load_all_data(url: str | None = None) -> dict[str, Any]:
    """Load canonical views. Deployments provide these stable views, not table names.

    The views prevent the model from depending on WordPress or legacy physical
    partition naming and are intentionally read-only for the AI database role.
    """
    engine = get_engine(url)
    views = ("leagues", "teams", "players", "player_stats", "team_player_relations")
    data = {name: pd.read_sql(text(f"SELECT * FROM ai_source.{name}"), engine) for name in views}
    data.update({
        "league_dict": data["leagues"].set_index("id").to_dict("index") if "id" in data["leagues"] else {},
        "team_dict": data["teams"].set_index("id").to_dict("index") if "id" in data["teams"] else {},
        "player_dict": data["players"].set_index("id").to_dict("index") if "id" in data["players"] else {},
        "league_teams": {},
    })
    return data
