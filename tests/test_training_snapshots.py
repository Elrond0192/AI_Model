from __future__ import annotations

import pandas as pd
import pytest

from basketball_ai.data.training_snapshots import (
    create_training_snapshot,
    list_training_snapshots,
    load_training_snapshot,
)


def test_snapshot_round_trip_is_versioned_and_verified(tmp_path):
    data = {
        "player_stats": pd.DataFrame(
            [{"player_id": 1, "season": 2024}, {"player_id": 1, "season": 2025}]
        ),
        "source_contract": "competition-v1",
    }
    manifest = create_training_snapshot(data, profile="production", root=tmp_path)
    loaded = load_training_snapshot(manifest["snapshot_id"], tmp_path)

    pd.testing.assert_frame_equal(loaded["player_stats"], data["player_stats"])
    assert manifest["sha256"]
    assert list_training_snapshots(tmp_path)[0]["snapshot_id"] == manifest["snapshot_id"]


def test_snapshot_checksum_blocks_corrupt_payload(tmp_path):
    manifest = create_training_snapshot(
        {"player_stats": pd.DataFrame([{"season": 2025}])},
        profile="production",
        root=tmp_path,
    )
    (tmp_path / manifest["file"]).write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="checksum"):
        load_training_snapshot(manifest["snapshot_id"], tmp_path)


def test_snapshot_id_cannot_escape_root(tmp_path):
    with pytest.raises(ValueError):
        load_training_snapshot("../secret", tmp_path)
