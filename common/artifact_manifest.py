"""Small stage-manifest helpers for DiagUQ pipeline artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from common.pair_context import DiagUQPairContext, validate_stage_manifest_provenance


MANIFEST_SCHEMA_VERSION = 1


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_stage_manifest(
    output_dir: str | Path,
    *,
    stage: str,
    status: str,
    dataset: Optional[str] = None,
    model: Optional[str] = None,
    artifacts: Optional[Mapping[str, Any]] = None,
    sanity: Optional[Mapping[str, Any]] = None,
    error: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
    pair_context: Optional[DiagUQPairContext] = None,
    filename: str = "manifest.json",
) -> Path:
    """Write a compact JSON manifest for one pipeline stage."""
    if status not in {"success", "failed", "skipped", "incomplete"}:
        raise ValueError(f"unsupported manifest status: {status!r}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "stage": stage,
        "status": status,
        "dataset": dataset,
        "model": model,
        "created_at": _now_utc(),
        "artifacts": dict(artifacts or {}),
        "sanity": dict(sanity or {}),
    }
    if error is not None:
        payload["error"] = error
    if pair_context is not None:
        payload.update(pair_context.manifest_provenance(out_dir))
    if extra:
        payload.update(dict(extra))

    path = out_dir / filename
    if pair_context is not None:
        validate_stage_manifest_provenance(payload, manifest_path=path, ctx=pair_context)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_stage_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_success_manifest(path: str | Path, *, stage: Optional[str] = None) -> dict[str, Any]:
    payload = load_stage_manifest(path)
    if stage is not None and payload.get("stage") != stage:
        raise RuntimeError(
            f"manifest stage mismatch at {path}: {payload.get('stage')!r} != {stage!r}"
        )
    if payload.get("status") != "success":
        raise RuntimeError(
            f"manifest is not successful at {path}: status={payload.get('status')!r}"
        )
    return payload