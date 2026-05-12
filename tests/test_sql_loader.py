"""Tests for src.data.sql_loader auto-discovery logic.

Uses SQLite in-memory databases so no real SQL Server is required.
"""
from __future__ import annotations

import os

import pytest

from basketball_ai.data.sql_loader import (
    _MIN_OVERLAP,
    _REQUIRED_COLUMNS,
    _normalize_identifier,
    _normalize_connection_string,
    _score_table,
    _is_transient_connection_error,
)


# ---------------------------------------------------------------------------
# _score_table
# ---------------------------------------------------------------------------


def test_normalize_identifier_handles_decimal_hex_and_text():
    assert _normalize_identifier("42") == 42
    assert _normalize_identifier("0000009B") == 155
    assert _normalize_identifier("0x9B") == 155
    assert _normalize_identifier(" player-001 ") == "player-001"
    assert _normalize_identifier(7.0) == 7

def test_score_table_full_match():
    actual = [
        "id", "name", "country", "tier",
        "competitiveness_score", "avg_pace", "avg_offensive_rating",
    ]
    assert _score_table(actual, _REQUIRED_COLUMNS["leagues"]) == 1.0


def test_score_table_case_insensitive():
    actual = [
        "ID", "Name", "Country", "Tier",
        "Competitiveness_Score", "Avg_Pace", "Avg_Offensive_Rating",
    ]
    assert _score_table(actual, _REQUIRED_COLUMNS["leagues"]) == 1.0


def test_score_table_partial():
    actual = ["id", "name", "country"]
    score = _score_table(actual, _REQUIRED_COLUMNS["leagues"])
    assert score == pytest.approx(3 / 7)


def test_score_table_no_match():
    score = _score_table(["foo", "bar", "baz"], _REQUIRED_COLUMNS["leagues"])
    assert score == 0.0


def test_score_table_empty_expected():
    assert _score_table(["id", "name"], []) == 0.0


def test_score_table_empty_actual():
    score = _score_table([], _REQUIRED_COLUMNS["leagues"])
    assert score == 0.0


def test_score_above_min_overlap_threshold():
    """A table matching all player_stats columns must beat _MIN_OVERLAP."""
    actual = _REQUIRED_COLUMNS["player_stats"]
    assert _score_table(actual, _REQUIRED_COLUMNS["player_stats"]) >= _MIN_OVERLAP


# ---------------------------------------------------------------------------
# _discover_table_mapping  (requires sqlalchemy; skip if absent)
# ---------------------------------------------------------------------------

sqlalchemy = pytest.importorskip("sqlalchemy", reason="sqlalchemy not installed")


def _make_engine():
    from sqlalchemy import create_engine
    return create_engine("sqlite:///:memory:")


def _create_standard_tables(conn):
    """Create tables with the canonical names and column sets."""
    from sqlalchemy import text
    statements = [
        """CREATE TABLE leagues (
               id INTEGER, name TEXT, country TEXT, tier INTEGER,
               competitiveness_score REAL, avg_pace REAL,
               avg_offensive_rating REAL
           )""",
        """CREATE TABLE teams (
               id INTEGER, name TEXT, league_id INTEGER,
               playing_style TEXT, pace REAL, offensive_rating REAL,
               defensive_rating REAL, three_point_attempt_rate REAL,
               assists_per_game REAL, star_player_usage REAL,
               league_tier INTEGER, formation TEXT
           )""",
        """CREATE TABLE players (
               id INTEGER, name TEXT, age INTEGER, position TEXT,
               nationality TEXT, height_cm REAL, weight_kg REAL,
               dominant_hand TEXT, current_team_id INTEGER,
               current_league_id INTEGER, draft_year INTEGER,
               draft_pick INTEGER
           )""",
        """CREATE TABLE player_stats (
               player_id INTEGER, season TEXT, team_id INTEGER,
               league_id INTEGER, games_played INTEGER,
               minutes_per_game REAL, points REAL, rebounds REAL,
               assists REAL, fg_pct REAL, rating REAL,
               steals REAL, blocks REAL, turnovers REAL,
               per REAL, ts_pct REAL, usg_pct REAL, bpm REAL,
               vorp REAL, win_shares REAL
           )""",
        """CREATE TABLE team_player_relations (
               team_id INTEGER, player_id INTEGER, season TEXT,
               role TEXT, jersey_number INTEGER
           )""",
    ]
    for sql in statements:
        conn.execute(text(sql))
    conn.commit()


def test_discover_standard_names():
    """Exact-name matching (step 2) resolves all five tables."""
    from basketball_ai.data.sql_loader import _discover_table_mapping

    engine = _make_engine()
    with engine.connect() as conn:
        _create_standard_tables(conn)

    mapping = _discover_table_mapping(engine)
    assert mapping["leagues"]               == "leagues"
    assert mapping["teams"]                 == "teams"
    assert mapping["players"]               == "players"
    assert mapping["player_stats"]          == "player_stats"
    assert mapping["team_player_relations"] == "team_player_relations"


def test_discover_alternate_names():
    """Scored fallback (step 3) picks the right table when names differ."""
    from sqlalchemy import text
    from basketball_ai.data.sql_loader import _discover_table_mapping

    engine = _make_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE lgs (
                id INTEGER, name TEXT, country TEXT, tier INTEGER,
                competitiveness_score REAL, avg_pace REAL,
                avg_offensive_rating REAL
            )"""))
        conn.execute(text("""
            CREATE TABLE tms (
                id INTEGER, name TEXT, league_id INTEGER,
                playing_style TEXT, pace REAL, offensive_rating REAL,
                defensive_rating REAL, three_point_attempt_rate REAL
            )"""))
        conn.execute(text("""
            CREATE TABLE plrs (
                id INTEGER, name TEXT, age INTEGER, position TEXT,
                nationality TEXT, height_cm REAL, weight_kg REAL,
                dominant_hand TEXT
            )"""))
        conn.execute(text("""
            CREATE TABLE stats (
                player_id INTEGER, season TEXT, team_id INTEGER,
                league_id INTEGER, games_played INTEGER,
                minutes_per_game REAL, points REAL, rebounds REAL,
                assists REAL, fg_pct REAL, rating REAL
            )"""))
        conn.execute(text("""
            CREATE TABLE rels (
                team_id INTEGER, player_id INTEGER, season TEXT, role TEXT
            )"""))
        conn.commit()

    mapping = _discover_table_mapping(engine)
    assert mapping["leagues"]               == "lgs"
    assert mapping["teams"]                 == "tms"
    assert mapping["players"]               == "plrs"
    assert mapping["player_stats"]          == "stats"
    assert mapping["team_player_relations"] == "rels"


def test_discover_env_var_override(monkeypatch):
    """AZURE_SQL_TABLE_<NAME> env-var takes precedence over auto-discovery."""
    from sqlalchemy import text
    from basketball_ai.data.sql_loader import _discover_table_mapping

    engine = _make_engine()
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE custom_leagues (
                id INTEGER, name TEXT, country TEXT, tier INTEGER,
                competitiveness_score REAL, avg_pace REAL,
                avg_offensive_rating REAL
            )"""))
        # Remaining tables with canonical names
        conn.execute(text("""
            CREATE TABLE teams (
                id INTEGER, name TEXT, league_id INTEGER,
                playing_style TEXT, pace REAL, offensive_rating REAL,
                defensive_rating REAL, three_point_attempt_rate REAL
            )"""))
        conn.execute(text("""
            CREATE TABLE players (
                id INTEGER, name TEXT, age INTEGER, position TEXT,
                nationality TEXT, height_cm REAL, weight_kg REAL,
                dominant_hand TEXT
            )"""))
        conn.execute(text("""
            CREATE TABLE player_stats (
                player_id INTEGER, season TEXT, team_id INTEGER,
                league_id INTEGER, games_played INTEGER,
                minutes_per_game REAL, points REAL, rebounds REAL,
                assists REAL, fg_pct REAL, rating REAL
            )"""))
        conn.execute(text("""
            CREATE TABLE team_player_relations (
                team_id INTEGER, player_id INTEGER, season TEXT, role TEXT
            )"""))
        conn.commit()

    monkeypatch.setenv("AZURE_SQL_TABLE_LEAGUES", "custom_leagues")
    mapping = _discover_table_mapping(engine)
    assert mapping["leagues"] == "custom_leagues"
    assert mapping["teams"]   == "teams"     # canonical resolution unaffected


def test_discover_missing_table_raises():
    """ValueError is raised when no table meets the overlap threshold."""
    from sqlalchemy import text
    from basketball_ai.data.sql_loader import _discover_table_mapping

    engine = _make_engine()
    with engine.connect() as conn:
        # Only one completely unrelated table
        conn.execute(text("CREATE TABLE unrelated (foo INTEGER, bar TEXT)"))
        conn.commit()

    with pytest.raises(ValueError, match="leagues"):
        _discover_table_mapping(engine)


def test_discover_env_var_nonexistent_table_raises(monkeypatch):
    """ValueError is raised when an env-var override points to a missing table."""
    from sqlalchemy import text
    from basketball_ai.data.sql_loader import _discover_table_mapping

    engine = _make_engine()
    with engine.connect() as conn:
        _create_standard_tables(conn)

    monkeypatch.setenv("AZURE_SQL_TABLE_LEAGUES", "does_not_exist")
    with pytest.raises(ValueError, match="does_not_exist"):
        _discover_table_mapping(engine)


# ---------------------------------------------------------------------------
# _normalize_connection_string
# ---------------------------------------------------------------------------

from urllib.parse import unquote_plus as _unquote_plus


class TestNormalizeConnectionString:
    """Tests for _normalize_connection_string."""

    @staticmethod
    def _decode_odbc_connect(url: str) -> str:
        """Extract and URL-decode the odbc_connect value from a SQLAlchemy URL."""
        return _unquote_plus(url.split("odbc_connect=", 1)[1])

    def test_sqlalchemy_url_returned_unchanged(self):
        """A proper SQLAlchemy URL must pass through without modification."""
        url = (
            "mssql+pyodbc://user:pass@server.database.windows.net/db"
            "?driver=ODBC+Driver+18+for+SQL+Server"
        )
        assert _normalize_connection_string(url) == url

    def test_sqlalchemy_url_with_odbc_connect_unchanged(self):
        """A SQLAlchemy URL that already uses odbc_connect is left intact."""
        url = "mssql+pyodbc:///?odbc_connect=Driver%3D%7BODBC+Driver+18%7D%3B"
        assert _normalize_connection_string(url) == url

    def test_odbc_string_with_driver_wrapped(self):
        """An ODBC string (with Driver=) is wrapped via odbc_connect."""
        odbc = (
            "Driver={ODBC Driver 18 for SQL Server};"
            "Server=tcp:myserver.database.windows.net,1433;"
            "Database=mydb;Uid=myuser;Pwd=mypass;"
            "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
        )
        result = _normalize_connection_string(odbc)
        assert result.startswith("mssql+pyodbc:///?odbc_connect=")
        decoded = self._decode_odbc_connect(result)
        assert "Driver={ODBC Driver 18 for SQL Server}" in decoded
        assert "Server=tcp:myserver.database.windows.net,1433" in decoded

    def test_ado_net_string_converted(self):
        """An ADO.NET connection string is parsed and converted to a SQLAlchemy URL."""
        ado = (
            "Server=tcp:myserver.database.windows.net,1433;"
            "Initial Catalog=mydb;"
            "User ID=myuser;"
            "Password=mypass;"
            "Encrypt=True;"
            "TrustServerCertificate=False;"
            "Connection Timeout=30;"
        )
        result = _normalize_connection_string(ado)
        assert result.startswith("mssql+pyodbc:///?odbc_connect=")
        decoded = self._decode_odbc_connect(result)
        assert "SERVER=myserver.database.windows.net,1433" in decoded
        assert "DATABASE=mydb" in decoded
        assert "UID=myuser" in decoded
        assert "PWD=mypass" in decoded

    def test_ado_net_tcp_prefix_stripped(self):
        """The 'tcp:' prefix added by Azure Portal is removed from SERVER=."""
        ado = (
            "Server=tcp:host.database.windows.net,1433;"
            "Initial Catalog=db;"
            "User ID=u;Password=p;"
        )
        result = _normalize_connection_string(ado)
        decoded = self._decode_odbc_connect(result)
        assert "SERVER=host.database.windows.net,1433" in decoded
        assert "tcp:" not in decoded

    def test_ado_net_default_driver_injected(self, monkeypatch):
        """When AZURE_SQL_DRIVER is not set a default driver is used."""
        monkeypatch.delenv("AZURE_SQL_DRIVER", raising=False)
        ado = "Server=host;Initial Catalog=db;User ID=u;Password=p;"
        result = _normalize_connection_string(ado)
        decoded = self._decode_odbc_connect(result)
        assert "DRIVER={ODBC Driver 18 for SQL Server}" in decoded

    def test_ado_net_custom_driver_from_env(self, monkeypatch):
        """AZURE_SQL_DRIVER env-var is used as the driver name for ADO.NET strings."""
        monkeypatch.setenv("AZURE_SQL_DRIVER", "ODBC Driver 17 for SQL Server")
        ado = "Server=host;Initial Catalog=db;User ID=u;Password=p;"
        result = _normalize_connection_string(ado)
        decoded = self._decode_odbc_connect(result)
        assert "DRIVER={ODBC Driver 17 for SQL Server}" in decoded

    def test_ado_net_default_timeout(self, monkeypatch):
        """Default Connection Timeout is 60 s when not specified in the string."""
        monkeypatch.delenv("AZURE_SQL_CONNECT_TIMEOUT", raising=False)
        ado = "Server=host;Initial Catalog=db;User ID=u;Password=p;"
        result = _normalize_connection_string(ado)
        decoded = self._decode_odbc_connect(result)
        assert "Connection Timeout=60" in decoded

    def test_ado_net_custom_timeout_from_env(self, monkeypatch):
        """AZURE_SQL_CONNECT_TIMEOUT env-var overrides the default timeout."""
        monkeypatch.setenv("AZURE_SQL_CONNECT_TIMEOUT", "90")
        ado = "Server=host;Initial Catalog=db;User ID=u;Password=p;"
        result = _normalize_connection_string(ado)
        decoded = self._decode_odbc_connect(result)
        assert "Connection Timeout=90" in decoded

    def test_ado_net_timeout_from_string_takes_precedence(self, monkeypatch):
        """Connection Timeout in the ADO.NET string overrides the env-var default."""
        monkeypatch.setenv("AZURE_SQL_CONNECT_TIMEOUT", "90")
        ado = "Server=host;Initial Catalog=db;User ID=u;Password=p;Connection Timeout=45;"
        result = _normalize_connection_string(ado)
        decoded = self._decode_odbc_connect(result)
        assert "Connection Timeout=45" in decoded


# ---------------------------------------------------------------------------
# _is_transient_connection_error
# ---------------------------------------------------------------------------

class TestIsTransientConnectionError:
    """Tests for _is_transient_connection_error."""

    def test_08001_sqlstate_recognised(self):
        assert _is_transient_connection_error(Exception("('08001', 'TCP timeout')"))

    def test_timeout_keyword_recognised(self):
        assert _is_transient_connection_error(Exception("Connection timeout expired"))

    def test_case_insensitive_timeout(self):
        assert _is_transient_connection_error(Exception("Login Timeout"))

    def test_unrelated_error_not_transient(self):
        assert not _is_transient_connection_error(Exception("42000: syntax error"))

    def test_permission_error_not_transient(self):
        assert not _is_transient_connection_error(Exception("28000: login failed"))


# ---------------------------------------------------------------------------
# get_engine retry logic
# ---------------------------------------------------------------------------

class TestGetEngineRetry:
    """Tests for the retry loop in get_engine.

    These tests use a real SQLite in-memory engine and patch the connect()
    method so that no real SQL Server is required.
    """

    @staticmethod
    def _make_engine(fail_times: int = 0, error_msg: str = "('08001', 'TCP timeout')"):
        """Return a SQLite engine whose first *fail_times* connect() calls raise *error_msg*.

        Failures are raised as ``sqlalchemy.exc.OperationalError`` to match
        what get_engine() now catches.
        """
        from sqlalchemy import create_engine
        from sqlalchemy.exc import OperationalError
        engine = create_engine("sqlite:///:memory:")
        _original_connect = engine.connect
        calls: list = []

        class _FakeConn:
            def __enter__(self):
                calls.append(1)
                if len(calls) <= fail_times:
                    raise OperationalError(error_msg, params=None, orig=Exception(error_msg))
                return _original_connect().__enter__()

            def __exit__(self, *args):
                return False

        engine.connect = lambda: _FakeConn()  # type: ignore[method-assign]
        return engine

    def test_returns_engine_on_first_success(self, monkeypatch):
        """get_engine returns immediately when the first connection succeeds."""
        monkeypatch.setattr("basketball_ai.data.sql_loader._build_connection_url",
                            lambda: "sqlite:///:memory:")
        monkeypatch.setattr("time.sleep", lambda _: None)

        engine = self._make_engine(fail_times=0)

        import sqlalchemy as _sa
        monkeypatch.setattr(_sa, "create_engine", lambda *a, **kw: engine)

        from basketball_ai.data.sql_loader import get_engine
        result = get_engine(max_retries=3, retry_delay=0)
        assert result is engine

    def test_retries_on_transient_error_then_succeeds(self, monkeypatch):
        """Engine is returned after retrying a transient 08001 error."""
        monkeypatch.setattr("basketball_ai.data.sql_loader._build_connection_url",
                            lambda: "sqlite:///:memory:")
        slept: list = []
        monkeypatch.setattr("time.sleep", lambda d: slept.append(d))

        engine = self._make_engine(fail_times=1, error_msg="('08001', 'TCP timeout')")

        import sqlalchemy as _sa
        monkeypatch.setattr(_sa, "create_engine", lambda *a, **kw: engine)

        from basketball_ai.data.sql_loader import get_engine
        result = get_engine(max_retries=3, retry_delay=5.0)
        assert result is engine
        assert len(slept) == 1, "Expected exactly one sleep() between retry attempts"

    def test_no_retry_on_non_transient_error(self, monkeypatch):
        """Non-transient errors are raised immediately without retry."""
        monkeypatch.setattr("basketball_ai.data.sql_loader._build_connection_url",
                            lambda: "sqlite:///:memory:")
        slept: list = []
        monkeypatch.setattr("time.sleep", lambda d: slept.append(d))

        engine = self._make_engine(fail_times=99, error_msg="28000: login failed")

        import sqlalchemy as _sa
        monkeypatch.setattr(_sa, "create_engine", lambda *a, **kw: engine)

        from basketball_ai.data.sql_loader import get_engine
        with pytest.raises(Exception, match="28000"):
            get_engine(max_retries=3, retry_delay=5.0)
        assert len(slept) == 0, "Non-transient errors should not trigger a sleep/retry"

    def test_all_retries_exhausted_raises(self, monkeypatch):
        """Exception is re-raised after all retry attempts are exhausted."""
        monkeypatch.setattr("basketball_ai.data.sql_loader._build_connection_url",
                            lambda: "sqlite:///:memory:")
        monkeypatch.setattr("time.sleep", lambda _: None)

        engine = self._make_engine(fail_times=99, error_msg="('08001', 'persistent timeout')")

        import sqlalchemy as _sa
        monkeypatch.setattr(_sa, "create_engine", lambda *a, **kw: engine)

        from basketball_ai.data.sql_loader import get_engine
        with pytest.raises(Exception, match="08001"):
            get_engine(max_retries=2, retry_delay=0)

    def test_invalid_max_retries_raises(self):
        """A max_retries value less than 1 raises ValueError immediately."""
        from basketball_ai.data.sql_loader import get_engine
        with pytest.raises(ValueError, match="max_retries must be >= 1"):
            get_engine(max_retries=0)
