"""Tenant isolation utilities.

Provides a decorator ``@tenant_scoped`` that injects the caller's ``tenant_id``
into route function kwargs, and a helper ``filter_by_tenant`` that filters
a DataFrame by tenant_id column.

Usage::

    @router.get("/players")
    @tenant_scoped
    def list_players(request: Request, ..., tenant_id: str = "default"):
        df = filter_by_tenant(data["players"], tenant_id)
        ...
"""
from __future__ import annotations

import logging

import pandas as pd

_logger = logging.getLogger(__name__)


def get_tenant_id(request) -> str:
    """Extract tenant_id from request.state, defaulting to 'default'."""
    return getattr(request.state, "tenant_id", "default") or "default"


def filter_by_tenant(
    df: pd.DataFrame,
    tenant_id: str,
    tenant_col: str = "tenant_id",
) -> pd.DataFrame:
    """Filter *df* by *tenant_id* if the column exists; otherwise return *df* unchanged.

    Args:
        df:         DataFrame to filter.
        tenant_id:  Caller's tenant identifier.
        tenant_col: Column name to filter on.

    Returns:
        Filtered DataFrame.
    """
    if df.empty or tenant_col not in df.columns:
        return df
    if tenant_id == "default":
        return df
    return df[df[tenant_col] == tenant_id]
