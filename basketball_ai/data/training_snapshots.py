"""Immutable, versioned training snapshots for reproducible model runs."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
import tempfile
from typing import Any
import uuid

import joblib


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_root(root: str | Path | None = None) -> Path:
    return Path(root or os.getenv("TRAINING_SNAPSHOT_DIR", "models_saved/snapshots"))


def create_training_snapshot(
    data: dict[str, Any],
    *,
    profile: str,
    root: str | Path | None = None,
) -> dict[str, Any]:
    target_root = snapshot_root(root)
    target_root.mkdir(parents=True, exist_ok=True)
    created = datetime.now(UTC)
    snapshot_id = f"{created:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    target = target_root / f"{snapshot_id}.joblib"
    with tempfile.NamedTemporaryFile(dir=target_root, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        joblib.dump(data, temporary, compress=3)
        digest = _sha256_file(temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    stats = data.get("player_stats")
    seasons = [] if stats is None or stats.empty else sorted(int(v) for v in stats["season"].dropna().unique())
    manifest = {
        "snapshot_id": snapshot_id,
        "created_at": created.isoformat(),
        "profile": profile,
        "source_contract": data.get("source_contract", "competition-v1"),
        "sha256": digest,
        "player_rows": int(len(stats)) if stats is not None else 0,
        "seasons": seasons,
        "file": target.name,
    }
    manifest_target = target_root / f"{snapshot_id}.json"
    manifest_target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_training_snapshot(snapshot_id: str, root: str | Path | None = None) -> dict[str, Any]:
    if not snapshot_id or any(token in snapshot_id for token in ("/", "\\", "..")):
        raise ValueError("Invalid snapshot id")
    target_root = snapshot_root(root)
    manifest = json.loads((target_root / f"{snapshot_id}.json").read_text(encoding="utf-8"))
    payload = target_root / manifest["file"]
    digest = _sha256_file(payload)
    if digest != manifest["sha256"]:
        raise RuntimeError("Training snapshot checksum mismatch")
    data = joblib.load(payload)
    data["training_snapshot_manifest"] = manifest
    return data


def list_training_snapshots(root: str | Path | None = None) -> list[dict[str, Any]]:
    target_root = snapshot_root(root)
    if not target_root.exists():
        return []
    manifests = []
    for path in target_root.glob("*.json"):
        try:
            manifests.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(manifests, key=lambda item: item.get("created_at", ""), reverse=True)
