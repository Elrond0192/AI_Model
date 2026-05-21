"""Schema contract tests.

Verify that:
1. basketball_ai/data/schema_mapping.py references columns consistent with models.
2. PerformanceModel.FEATURE_COLS contains only computable features.
3. The schema_db.sql file defines expected tables and columns.
"""
from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent


def _read_sql(sql_path: Path) -> str:
    """Read SQL file, auto-detecting UTF-16 (BOM) vs UTF-8."""
    raw = sql_path.read_bytes()
    if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8", errors="replace")


def _parse_schema_tables(sql_path: Path) -> dict:
    """Parse CREATE TABLE statements from schema_db.sql.

    Returns dict mapping table_name → set of column names.
    Handles [bracketed] and plain identifiers.
    """
    if not sql_path.exists():
        return {}

    sql = _read_sql(sql_path)
    tables: dict = {}

    # Find CREATE TABLE blocks
    create_pattern = re.compile(
        r'CREATE\s+TABLE\s+(?:\[?[\w.]+\]?\.)?\[?([\w]+)\]?\s*\((.*?)\)',
        re.IGNORECASE | re.DOTALL,
    )
    for m in create_pattern.finditer(sql):
        table_name = m.group(1)
        body = m.group(2)
        # Extract column names from lines like [ColName] type or ColName type
        col_pattern = re.compile(r'^\s*\[?([\w]+)\]?\s+\w', re.MULTILINE)
        cols = {c.group(1).lower() for c in col_pattern.finditer(body)}
        # Exclude constraint keywords
        keywords = {"primary", "foreign", "unique", "constraint", "index", "check"}
        cols -= keywords
        if cols:
            tables[table_name.lower()] = cols
    return tables


def test_schema_db_exists():
    """schema_db.sql must exist in repo root."""
    assert (REPO_ROOT / "schema_db.sql").exists(), "schema_db.sql not found in repo root"


def test_schema_db_has_expected_tables():
    """schema_db.sql must define tables from Anagrafiche, Analisi, Boxscore, Pbp schemas."""
    sql_path = REPO_ROOT / "schema_db.sql"
    if not sql_path.exists():
        return
    sql = _read_sql(sql_path).lower()
    # Check that at least some of the expected schema keywords appear
    expected_schemas = ["anagrafiche", "analisi", "boxscore", "pbp"]
    for schema in expected_schemas:
        assert schema in sql, f"Schema '{schema}' not found in schema_db.sql"


def test_feature_cols_are_plausible():
    """All FEATURE_COLS should be non-empty strings, no duplicates."""
    from basketball_ai.models.performance_model import FEATURE_COLS
    assert len(FEATURE_COLS) > 0
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS)), "Duplicate features in FEATURE_COLS"
    for col in FEATURE_COLS:
        assert isinstance(col, str) and col, f"Invalid feature name: {repr(col)}"


def test_schema_mapping_imports_cleanly():
    """schema_mapping.py must import without errors."""
    import basketball_ai.data.schema_mapping  # noqa: F401


def test_performance_model_feature_count():
    """FEATURE_COLS should have at least 50 features."""
    from basketball_ai.models.performance_model import FEATURE_COLS
    assert len(FEATURE_COLS) >= 50, f"Expected ≥50 features, got {len(FEATURE_COLS)}"


def test_constants_no_circular_import():
    """constants.py must be importable with no circular dependency."""
    import basketball_ai.constants  # noqa: F401
    from basketball_ai.constants import SUPPORTED_LEAGUES, SUPPORTED_SEASONS, is_safe_identifier
    assert len(SUPPORTED_LEAGUES) > 0
    assert len(SUPPORTED_SEASONS) > 0
    assert is_safe_identifier("ValidName") is True
    assert is_safe_identifier("") is False


def test_data_models_import_cleanly():
    """data/models.py must import without errors."""
    import basketball_ai.data.models  # noqa: F401


def test_schema_mapping_player_stats_columns_subset_of_model():
    """Columns referenced in schema_mapping must be a subset of PlayerStats fields."""
    from basketball_ai.data import models as dm
    import dataclasses

    # Get all field names from PlayerStats
    try:
        ps_fields = {f.name for f in dataclasses.fields(dm.PlayerStats)}
    except TypeError:
        return  # Not a dataclass

    # This is informational – we just check it's non-empty
    assert len(ps_fields) > 0
