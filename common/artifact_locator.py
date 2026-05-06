"""Runtime artifact locator for dataset variants.

The CLI and registries speak in base dataset names such as ``triviaqa``.
Runtime artifacts may live under split-qualified variants such as
``triviaqa__train``. This module is the shared boundary between those two
identities so downstream stages do not guess file names independently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from common.artifact_paths import (
    ask4conf_jsonl_path,
    ask4conf_metadata_path,
    ask4conf_success_marker,
    extend_path,
    extend_samples_path,
    mextend_bleu_path,
    mextend_path,
    mextend_rouge_path,
    mextend_samples_path,
    normalize_split_tag,
    response_answer_audit_csv_path,
    response_answer_audit_json_path,
    semantic_entropy_path,
    split_artifact_dir,
    split_dataset_and_raw,
)
from common.pair_context import (
    DEFAULT_DATASET_SPLIT,
    DiagUQPairContext,
    resolve_dataset_variant,
    resolve_pair_context,
    validate_stage_manifest_provenance,
)
from common.runtime_paths import get_test_output_dir


MANIFEST_NAME = "response_cache_manifest.json"
MANIFEST_VERSION = 2

DIAGUQ_SUBDIR = "diaguq"
LEGACY_DIAGUQ_SUBDIR = "mduq"


class ArtifactResolutionError(FileNotFoundError):
    """Raised when a required artifact cannot be resolved."""


@dataclass(frozen=True)
class ResponseCacheArtifacts:
    dataset_name: str
    dataset_variant: str
    artifact_prefix: str
    model_name: str
    runtime_root: Path
    response_cache_dir: Path
    manifest_path: Optional[Path]
    paths: Dict[str, Path]
    manifest_artifacts: Dict[str, Dict[str, Any]]
    existing: Dict[str, bool]
    checked_paths: List[Path]
    resolution_strategy: str

    @property
    def missing(self) -> List[str]:
        return [name for name, exists in self.existing.items() if not exists]

    def require(self, key: str) -> Path:
        path = self.paths.get(key)
        if path is not None and path.is_file():
            return path
        checked = "\n  ".join(str(p) for p in self.checked_paths)
        discovered = sorted(
            str(p.parent.name)
            for p in self.runtime_root.glob(f"{self.dataset_name}__*/{self.model_name}")
            if p.is_dir()
        )
        discovered_msg = ", ".join(discovered) if discovered else "none"
        raise ArtifactResolutionError(
            "[artifact-locator] missing required response-cache artifact "
            f"{key!r} for dataset={self.dataset_name!r} "
            f"variant={self.dataset_variant!r} model={self.model_name!r}.\n"
            f"Resolved response_cache_dir: {self.response_cache_dir}\n"
            f"Manifest: {self.manifest_path or 'not found'}\n"
            f"Candidate locations checked:\n  {checked}\n"
            "Split-qualified dataset directories discovered for this pair: "
            f"{discovered_msg}.\n"
            "Run `python run.py build-response-cache` for this dataset/model, "
            "or check whether artifacts were generated under a different "
            "dataset variant."
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_variant": self.dataset_variant,
            "artifact_prefix": self.artifact_prefix,
            "model_name": self.model_name,
            "runtime_root": str(self.runtime_root),
            "response_cache_dir": str(self.response_cache_dir),
            "manifest_path": str(self.manifest_path) if self.manifest_path else None,
            "paths": {k: str(v) for k, v in self.paths.items()},
            "manifest_artifacts": self.manifest_artifacts,
            "existing": dict(self.existing),
            "checked_paths": [str(p) for p in self.checked_paths],
            "resolution_strategy": self.resolution_strategy,
        }


def _runtime_root(runtime_root: Optional[str | Path] = None) -> Path:
    return Path(get_test_output_dir()) if runtime_root is None else Path(runtime_root)


def _base_dataset(dataset_name: str) -> str:
    return split_dataset_and_raw(dataset_name)[0]


def _default_variant(dataset_name: str, split: Optional[str] = None) -> str:
    return resolve_dataset_variant(dataset_name, split)[0]


def _dedupe(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _variant_candidates(
    dataset_name: str,
    model_name: Optional[str],
    root: Path,
    split: Optional[str] = None,
    available_paths: Optional[Sequence[str | Path]] = None,
) -> List[str]:
    base = _base_dataset(dataset_name)
    explicit = dataset_name if "__" in dataset_name else ""
    candidates: List[str] = [explicit, _default_variant(dataset_name, split), dataset_name]

    if explicit and not available_paths:
        return _dedupe(candidates)

    if available_paths:
        for raw_path in available_paths:
            path = Path(raw_path)
            for part in (path.name, path.parent.name, path.parent.parent.name):
                if part == base or part.startswith(f"{base}__"):
                    candidates.append(part)

    if model_name:
        for model_dir in sorted(root.glob(f"{base}__*/{model_name}")):
            if model_dir.is_dir():
                candidates.append(model_dir.parent.name)
        legacy_dir = root / base / model_name
        if legacy_dir.is_dir():
            candidates.append(base)

    return _dedupe(candidates)


def resolve_runtime_dataset_variant(
    dataset_name: str,
    split: Optional[str] = None,
    available_paths: Optional[Sequence[str | Path]] = None,
    *,
    model_name: Optional[str] = None,
    runtime_root: Optional[str | Path] = None,
) -> str:
    """Resolve the runtime dataset variant for a base CLI dataset name."""
    root = _runtime_root(runtime_root)
    for variant in _variant_candidates(
        dataset_name, model_name, root, split, available_paths
    ):
        if model_name is None:
            return variant
        pair_dir = split_artifact_dir(variant, model_name, root)
        if pair_dir.is_dir() or mextend_path(variant, model_name, root).is_file():
            return variant
    return _default_variant(dataset_name, split)


def _paths_for_variant(
    dataset_variant: str, model_name: str, root: Path
) -> Dict[str, Path]:
    pair_dir = split_artifact_dir(dataset_variant, model_name, root)
    mextend = mextend_path(dataset_variant, model_name, root)
    return {
        "response_cache_dir": pair_dir,
        "manifest": pair_dir / MANIFEST_NAME,
        "mextend": mextend,
        "mextend_samples": mextend_samples_path(dataset_variant, model_name, root),
        "mextend_rouge": mextend_rouge_path(dataset_variant, model_name, root),
        "mextend_bleu": mextend_bleu_path(dataset_variant, model_name, root),
        "extend": extend_path(dataset_variant, model_name, root),
        "extend_samples": extend_samples_path(dataset_variant, model_name, root),
        "semantic_entropy": semantic_entropy_path(dataset_variant, model_name, root),
        "response_answer_audit_csv": response_answer_audit_csv_path(dataset_variant, model_name, root),
        "response_answer_audit_json": response_answer_audit_json_path(dataset_variant, model_name, root),
        "ask4conf_jsonl": ask4conf_jsonl_path(dataset_variant, model_name, root),
        "ask4conf_metadata": ask4conf_metadata_path(dataset_variant, model_name, root),
        "ask4conf_success_marker": ask4conf_success_marker(dataset_variant, model_name, root),
        "ask4conf_source_errors": ask4conf_jsonl_path(
            dataset_variant, model_name, root
        ).with_name(f"{dataset_variant}_source_errors.jsonl"),
        "ask4conf_source_error_audit_json": ask4conf_jsonl_path(
            dataset_variant, model_name, root
        ).with_name(f"{dataset_variant}_source_error_audit.json"),
        "ask4conf_source_error_audit_csv": ask4conf_jsonl_path(
            dataset_variant, model_name, root
        ).with_name(f"{dataset_variant}_source_error_audit.csv"),
        "mextend_validation": mextend.with_name(
            f"{dataset_variant}_mextend_validation.json"
        ),
    }


def _response_artifact_keys() -> List[str]:
    return [
        "mextend",
        "mextend_rouge",
        "mextend_bleu",
        "extend",
        "semantic_entropy",
        "response_answer_audit_csv",
        "response_answer_audit_json",
        "ask4conf_jsonl",
        "ask4conf_metadata",
        "ask4conf_source_errors",
        "ask4conf_source_error_audit_json",
        "ask4conf_source_error_audit_csv",
        "mextend_validation",
    ]


def _ask4conf_manifest_summary(meta_path: Path) -> Dict[str, Any]:
    if not meta_path.is_file():
        return {}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(meta, Mapping):
        return {}
    summary = {
        "loaded_rows": meta.get("loaded_rows", meta.get("source_count", meta.get("expected_count"))),
        "valid_answer_rows": meta.get("valid_answer_rows", meta.get("valid_answer_count")),
        "placeholder_or_missing_rows": meta.get("placeholder_or_missing_rows", meta.get("placeholder_or_missing_count")),
        "source_error_rate": meta.get("source_error_rate"),
        "source_error_policy": meta.get("source_error_policy"),
        "source_error_threshold": meta.get("source_error_threshold", meta.get("source_validation_threshold_used")),
        "skipped_ask4conf_rows": meta.get("skipped_ask4conf_rows", meta.get("source_failed_count")),
        "retried_rows": meta.get("retried_rows", 0),
        "retry_success_rows": meta.get("retry_success_rows", 0),
    }
    return {key: value for key, value in summary.items() if value is not None}


def _existing(paths: Mapping[str, Path]) -> Dict[str, bool]:
    return {key: path.exists() for key, path in paths.items()}


def _manifest_candidates(
    dataset_name: str,
    model_name: str,
    root: Path,
    split: Optional[str] = None,
) -> List[Path]:
    variants = _variant_candidates(dataset_name, model_name, root, split)
    paths = [split_artifact_dir(v, model_name, root) / MANIFEST_NAME for v in variants]
    if "__" in dataset_name:
        return _dedupe_paths(paths)
    base = _base_dataset(dataset_name)
    paths.extend(sorted(root.glob(f"{base}__*/{model_name}/{MANIFEST_NAME}")))
    return _dedupe_paths(paths)


def _dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def resolve_manifest_artifact_path(
    value: Any,
    pair_root: Path,
    test_output_root: Path,
) -> Optional[Path]:
    """Resolve a manifest artifact path without double-joining pair roots.

    Supported forms:
    - absolute string path
    - legacy string relative to the test-output root
    - string relative to the pair root
    - dict with ``path`` and/or ``relative_path`` plus optional ``relative_to``
    """
    if value is None:
        return None
    if isinstance(value, Mapping):
        raw_path = value.get("path")
        relative_path = value.get("relative_path")
        test_output_relative = value.get("test_output_relative_path")
        relative_to = value.get("relative_to")
        if raw_path:
            raw = Path(str(raw_path))
            if raw.is_absolute():
                return raw
        if relative_path:
            rel = Path(str(relative_path))
            if rel.is_absolute():
                return rel
            if relative_to == "test_output_root":
                return test_output_root / rel
            return pair_root / rel
        if test_output_relative:
            rel = Path(str(test_output_relative))
            if rel.is_absolute():
                return rel
            return test_output_root / rel
        value = raw_path
        if value is None:
            return None

    path = Path(str(value))
    if path.is_absolute():
        return path

    pair_rel = pair_root.relative_to(test_output_root) if pair_root.is_relative_to(test_output_root) else None
    parts = path.parts
    if pair_rel is not None:
        pair_parts = pair_rel.parts
        if len(parts) >= len(pair_parts) and parts[: len(pair_parts)] == pair_parts:
            return test_output_root / path

    root_candidate = test_output_root / path
    pair_candidate = pair_root / path
    if root_candidate.exists() and not pair_candidate.exists():
        return root_candidate
    if pair_candidate.exists() and not root_candidate.exists():
        return pair_candidate
    if str(value).startswith((".", "..")):
        return pair_candidate
    return pair_candidate


def _coerce_manifest_path(value: Any, manifest_dir: Path, root: Path) -> Optional[Path]:
    return resolve_manifest_artifact_path(value, manifest_dir, root)


def _manifest_path_value(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _artifact_manifest_entry(
    key: str,
    path: Path,
    response_cache_dir: Path,
    root: Path,
    *,
    status: Optional[str] = None,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        relative_path = path.relative_to(response_cache_dir).as_posix()
    except ValueError:
        relative_path = None
    try:
        test_output_relative_path = path.relative_to(root).as_posix()
    except ValueError:
        test_output_relative_path = None
    resolved_status = status or ("generated" if path.exists() else "missing")
    out: Dict[str, Any] = {
        "status": resolved_status,
        "path": str(path.resolve()),
        "relative_path": relative_path,
        "test_output_relative_path": test_output_relative_path,
    }
    if reason:
        out["reason"] = reason
    elif resolved_status == "missing":
        out["reason"] = "artifact file does not exist"
    return out


def _from_manifest(
    manifest_path: Path,
    dataset_name: str,
    model_name: str,
    root: Path,
    checked: List[Path],
) -> Optional[ResponseCacheArtifacts]:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("model_name") not in (None, model_name):
        return None
    manifest_dataset = str(payload.get("dataset_name") or dataset_name)
    if _base_dataset(manifest_dataset) != _base_dataset(dataset_name):
        return None
    variant = str(payload.get("dataset_variant") or manifest_path.parent.parent.name)
    paths = _paths_for_variant(variant, model_name, root)
    manifest_artifacts: Dict[str, Dict[str, Any]] = {}
    for key, entry in (payload.get("artifacts") or {}).items():
        if not isinstance(entry, Mapping):
            continue
        manifest_artifacts[str(key)] = dict(entry)
        coerced = resolve_manifest_artifact_path(entry, manifest_path.parent, root)
        if coerced is not None:
            paths[str(key)] = coerced
    for key, value in (payload.get("paths") or {}).items():
        coerced = _coerce_manifest_path(value, manifest_path.parent, root)
        if coerced is not None:
            paths[key] = coerced
            manifest_artifacts.setdefault(
                key,
                {
                    "status": "generated" if coerced.exists() else "missing",
                    "path": str(coerced),
                    "relative_path": (
                        coerced.relative_to(manifest_path.parent).as_posix()
                        if coerced.is_relative_to(manifest_path.parent)
                        else None
                    ),
                },
            )
    paths["manifest"] = manifest_path
    checked.extend(paths[k] for k in _response_artifact_keys() if k in paths)
    return ResponseCacheArtifacts(
        dataset_name=_base_dataset(dataset_name),
        dataset_variant=variant,
        artifact_prefix=str(payload.get("artifact_prefix") or variant),
        model_name=model_name,
        runtime_root=root,
        response_cache_dir=paths["response_cache_dir"],
        manifest_path=manifest_path,
        paths=paths,
        manifest_artifacts=manifest_artifacts,
        existing=_existing(paths),
        checked_paths=list(checked),
        resolution_strategy="manifest",
    )


def _result_for_variant(
    dataset_name: str,
    dataset_variant: str,
    model_name: str,
    root: Path,
    checked: List[Path],
    strategy: str,
) -> ResponseCacheArtifacts:
    paths = _paths_for_variant(dataset_variant, model_name, root)
    checked.extend(paths[k] for k in _response_artifact_keys() if k in paths)
    manifest = paths["manifest"] if paths["manifest"].is_file() else None
    return ResponseCacheArtifacts(
        dataset_name=_base_dataset(dataset_name),
        dataset_variant=dataset_variant,
        artifact_prefix=dataset_variant,
        model_name=model_name,
        runtime_root=root,
        response_cache_dir=paths["response_cache_dir"],
        manifest_path=manifest,
        paths=paths,
        manifest_artifacts={},
        existing=_existing(paths),
        checked_paths=list(checked),
        resolution_strategy=strategy,
    )


def locate_response_cache_artifacts(
    dataset_name: str,
    model_name: str,
    runtime_root: Optional[str | Path] = None,
    split: Optional[str] = None,
) -> ResponseCacheArtifacts:
    """Locate response-cache artifacts for a dataset/model pair.

    Resolution order: manifest, canonical variant path, split-qualified
    directories, then legacy base-name paths.
    """
    root = _runtime_root(runtime_root)
    checked: List[Path] = []

    stale_manifest: Optional[ResponseCacheArtifacts] = None
    for manifest_path in _manifest_candidates(dataset_name, model_name, root, split):
        checked.append(manifest_path)
        if not manifest_path.is_file():
            continue
        resolved = _from_manifest(manifest_path, dataset_name, model_name, root, checked)
        if resolved is None:
            continue
        if resolved.paths.get("mextend") and resolved.paths["mextend"].is_file():
            return resolved
        stale_manifest = resolved

    variants = _variant_candidates(dataset_name, model_name, root, split)
    for variant in variants:
        resolved = _result_for_variant(
            dataset_name, variant, model_name, root, checked, "candidate"
        )
        if resolved.paths["mextend"].is_file():
            return resolved

    base = _base_dataset(dataset_name)
    if "__" not in dataset_name:
        for pair_dir in sorted(root.glob(f"{base}__*/{model_name}")):
            variant = pair_dir.parent.name
            resolved = _result_for_variant(
                dataset_name, variant, model_name, root, checked, "split_discovery"
            )
            if resolved.paths["mextend"].is_file():
                return resolved

    if "__" not in dataset_name:
        legacy = _result_for_variant(
            dataset_name, base, model_name, root, checked, "legacy_base_name"
        )
        if legacy.paths["mextend"].is_file():
            return legacy
    return stale_manifest or _result_for_variant(
        dataset_name,
        _default_variant(dataset_name, split),
        model_name,
        root,
        checked,
        "unresolved_default",
    )


def get_response_cache_dir(
    dataset_name: str, model_name: str, runtime_root: Optional[str | Path] = None
) -> Path:
    return locate_response_cache_artifacts(
        dataset_name, model_name, runtime_root
    ).response_cache_dir


def get_diaguq_output_dir(
    dataset_name: str, model_name: str, runtime_root: Optional[str | Path] = None
) -> Path:
    return resolve_pair_context(
        dataset_name, model_name, runtime_root=_runtime_root(runtime_root)
    ).diaguq_root


def _variant_dirs_for_outputs(
    dataset_name: str, model_name: str, root: Path
) -> List[str]:
    base = _base_dataset(dataset_name)
    variants = _variant_candidates(dataset_name, model_name, root)
    variants.extend(
        path.parent.name for path in sorted(root.glob(f"{base}__*/{model_name}"))
    )
    variants.append(base)
    return _dedupe(variants)


def locate_hidden_bank_dir(
    dataset_name: str, model_name: str, runtime_root: Optional[str | Path] = None
) -> Path:
    return resolve_pair_context(
        dataset_name, model_name, runtime_root=_runtime_root(runtime_root)
    ).hidden_bank_dir


def locate_dimension_target_dir(
    dataset_name: str, model_name: str, runtime_root: Optional[str | Path] = None
) -> Path:
    return resolve_pair_context(
        dataset_name, model_name, runtime_root=_runtime_root(runtime_root)
    ).dimension_targets_dir


def write_response_cache_manifest(
    dataset_name: str,
    dataset_variant: str,
    model_name: str,
    runtime_root: Optional[str | Path] = None,
    *,
    artifact_status: Optional[Mapping[str, str]] = None,
    artifact_reasons: Optional[Mapping[str, str]] = None,
    pair_context: Optional[DiagUQPairContext] = None,
) -> Path:
    root = _runtime_root(runtime_root)
    ctx = pair_context or resolve_pair_context(dataset_variant, model_name, runtime_root=root)
    dataset_variant = ctx.resolved_variant
    paths = _paths_for_variant(dataset_variant, model_name, root)
    response_cache_dir = paths["response_cache_dir"]
    response_cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = response_cache_dir / MANIFEST_NAME
    artifact_status = dict(artifact_status or {})
    artifact_reasons = dict(artifact_reasons or {})
    artifacts = {
        key: _artifact_manifest_entry(
            key,
            path,
            response_cache_dir,
            root,
            status=artifact_status.get(key),
            reason=artifact_reasons.get(key),
        )
        for key, path in paths.items()
        if key != "manifest"
    }
    payload = {
        "schema_version": MANIFEST_VERSION,
        "dataset_name": _base_dataset(dataset_name),
        "requested_dataset": ctx.requested_dataset,
        "dataset_variant": dataset_variant,
        "resolved_variant": ctx.resolved_variant,
        "split": ctx.split,
        "artifact_prefix": dataset_variant,
        "model_name": model_name,
        "model": model_name,
        "runtime_root": str(root),
        "pair_root": str(ctx.pair_root),
        "diaguq_root": str(ctx.diaguq_root),
        "stage_output_dir": str(ctx.response_cache_root),
        "response_cache_dir": str(response_cache_dir.resolve() if response_cache_dir.exists() else response_cache_dir),
        "response_cache_relative_path": _manifest_path_value(response_cache_dir, root),
        "artifacts": artifacts,
        "paths": {
            key: _manifest_path_value(path, root)
            for key, path in paths.items()
            if key != "manifest"
        },
        "existing": {
            key: path.exists() for key, path in paths.items() if key != "manifest"
        },
    }
    ask4conf_summary = _ask4conf_manifest_summary(paths["ask4conf_metadata"])
    if ask4conf_summary:
        payload["ask4conf_summary"] = ask4conf_summary
        payload.update(ask4conf_summary)
    payload.update(ctx.split_metadata)
    validate_stage_manifest_provenance(payload, manifest_path=manifest_path, ctx=ctx)
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest_path
