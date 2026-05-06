"""Discover existing DiagUQ downstream artifact roots.

This module intentionally does not define or migrate the artifact layout. It
only finds the root that is already on disk, such as
``test_output/triviaqa__train/<model>/diaguq``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from common.runtime_paths import get_test_output_dir


KNOWN_SPLITS: Tuple[str, ...] = ("train", "validation", "dev", "test")
DIAGUQ_METHOD_SUBDIR = "diaguq"
LEGACY_METHOD_SUBDIR = "mduq"


@dataclass(frozen=True)
class ExistingDiagUQArtifactRoot:
    requested_dataset_name: str
    requested_model_name: str
    output_root: Path
    found: bool
    artifact_root: Optional[Path]
    dataset_root_name: Optional[str]
    split_label: str
    method_subdir: Optional[str]
    model_root_name: Optional[str]
    checked_paths: Tuple[Path, ...]
    reason: str

    def missing_subdirs(self, names: Sequence[str]) -> List[Tuple[str, Path]]:
        if self.artifact_root is None:
            return []
        return [
            (name, self.artifact_root / name)
            for name in names
            if not (self.artifact_root / name).is_dir()
        ]

    def describe(self) -> str:
        if self.found and self.artifact_root is not None:
            return (
                f"requested dataset={self.requested_dataset_name!r}, "
                f"model={self.requested_model_name!r}, "
                f"resolved artifact root={self.dataset_root_name!r}, "
                f"split={self.split_label!r}, method={self.method_subdir!r}, "
                f"path={self.artifact_root}"
            )
        checked = "\n  ".join(str(path) for path in self.checked_paths)
        return (
            f"no existing DiagUQ artifact root for "
            f"dataset={self.requested_dataset_name!r}, "
            f"model={self.requested_model_name!r}. {self.reason}\n"
            f"Checked paths:\n  {checked}"
        )


def _runtime_root(output_root: Optional[str | Path]) -> Path:
    return Path(get_test_output_dir()) if output_root is None else Path(output_root)


def _base_dataset_name(dataset_name: str) -> str:
    return dataset_name.split("__", 1)[0]


def _split_label(dataset_root_name: str, base_dataset_name: str) -> str:
    prefix = f"{base_dataset_name}__"
    if dataset_root_name.startswith(prefix):
        raw = dataset_root_name[len(prefix):]
        return raw if raw in KNOWN_SPLITS else "unknown"
    return "unknown"


def _dedupe(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _dataset_candidates(
    dataset_name: str,
    preferred_split: Optional[str],
    artifact_root_name: Optional[str],
    allow_train_fallback: bool,
) -> List[str]:
    if artifact_root_name:
        return [artifact_root_name]

    base = _base_dataset_name(dataset_name)
    candidates: List[str] = []
    if "__" in dataset_name:
        candidates.append(dataset_name)
        if not allow_train_fallback:
            return _dedupe(candidates)
    if preferred_split:
        split = preferred_split.split("__", 1)[-1]
        candidates.append(f"{base}__{split}")
        if not allow_train_fallback:
            return _dedupe(candidates)
    candidates.extend(f"{base}__{split}" for split in KNOWN_SPLITS)
    candidates.append(base)
    return _dedupe(candidates)


def _legacy_dataset_candidates(
    dataset_name: str,
    artifact_root_name: Optional[str],
    allow_train_fallback: bool,
) -> List[str]:
    if artifact_root_name:
        return [artifact_root_name]
    base = _base_dataset_name(dataset_name)
    candidates = []
    if "__" in dataset_name:
        candidates.append(dataset_name)
        if not allow_train_fallback:
            return _dedupe(candidates)
    candidates.extend([f"{base}__train", base])
    return _dedupe(candidates)


def _model_candidates(model_name: str) -> List[str]:
    candidates = [model_name]
    try:
        from registry.model_registry import get_canonical_name, get_local_dir_name

        canonical = get_canonical_name(model_name)
        candidates.append(canonical)
        candidates.append(get_local_dir_name(canonical))
    except Exception:
        pass
    return _dedupe(candidates)


def resolve_existing_diaguq_artifact_root(
    dataset_name: str,
    model_name: str,
    output_root: Optional[str | Path] = None,
    preferred_split: Optional[str] = None,
    artifact_root_name: Optional[str] = None,
    allow_train_fallback: bool = False,
) -> ExistingDiagUQArtifactRoot:
    """Find the first existing DiagUQ artifact root for a pair.

    ``artifact_root_name`` is an explicit advanced override such as
    ``"triviaqa__train"``. When omitted, split-qualified roots are searched
    before the base dataset root, with legacy ``mduq`` roots checked last.
    """
    root = _runtime_root(output_root)
    base = _base_dataset_name(dataset_name)
    checked: List[Path] = []
    model_roots = _model_candidates(model_name)

    search_plan: List[Tuple[str, str]] = []
    for dataset_root in _dataset_candidates(
        dataset_name, preferred_split, artifact_root_name, allow_train_fallback
    ):
        search_plan.append((dataset_root, DIAGUQ_METHOD_SUBDIR))
    for dataset_root in _legacy_dataset_candidates(dataset_name, artifact_root_name, allow_train_fallback):
        search_plan.append((dataset_root, LEGACY_METHOD_SUBDIR))

    for dataset_root, method_subdir in search_plan:
        for model_root in model_roots:
            artifact_root = root / dataset_root / model_root / method_subdir
            checked.append(artifact_root)
            if artifact_root.is_dir():
                return ExistingDiagUQArtifactRoot(
                    requested_dataset_name=dataset_name,
                    requested_model_name=model_name,
                    output_root=root,
                    found=True,
                    artifact_root=artifact_root,
                    dataset_root_name=dataset_root,
                    split_label=_split_label(dataset_root, base),
                    method_subdir=method_subdir,
                    model_root_name=model_root,
                    checked_paths=tuple(checked),
                    reason="found",
                )

    reason = (
        "explicit artifact root not found"
        if artifact_root_name
        else "no candidate root exists"
    )
    return ExistingDiagUQArtifactRoot(
        requested_dataset_name=dataset_name,
        requested_model_name=model_name,
        output_root=root,
        found=False,
        artifact_root=None,
        dataset_root_name=None,
        split_label="unknown",
        method_subdir=None,
        model_root_name=None,
        checked_paths=tuple(checked),
        reason=reason,
    )


def require_existing_diaguq_artifact_root(
    dataset_name: str,
    model_name: str,
    output_root: Optional[str | Path] = None,
    preferred_split: Optional[str] = None,
    artifact_root_name: Optional[str] = None,
    allow_train_fallback: bool = False,
) -> ExistingDiagUQArtifactRoot:
    resolved = resolve_existing_diaguq_artifact_root(
        dataset_name,
        model_name,
        output_root,
        preferred_split=preferred_split,
        artifact_root_name=artifact_root_name,
        allow_train_fallback=allow_train_fallback,
    )
    if not resolved.found:
        raise FileNotFoundError(resolved.describe())
    return resolved
