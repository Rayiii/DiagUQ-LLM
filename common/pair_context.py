"""Canonical pair-scoped artifact context for DiagUQ stages."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from common.artifact_paths import normalize_split_tag, split_dataset_and_raw
from common.runtime_paths import get_test_output_dir
from common.single_split_policy import split_metadata_for_variant


DEFAULT_DATASET_SPLIT: Mapping[str, str] = {
    "coqa": "train",
    "triviaqa": "train",
    "ambigqa": "train",
    "truthfulqa": "validation",
    "mmlu": "train",
    "wmt": "test",
}


INVALID_CHECKPOINT_VARIANT_MESSAGE = (
    "Invalid checkpoint resolved_variant contains comma; use per-pair train context instead of global train_split."
)


@dataclass(frozen=True)
class DiagUQPairContext:
    requested_dataset: str
    resolved_variant: str
    split: str
    model: str
    test_output_root: Path
    pair_root: Path
    diaguq_root: Path
    response_cache_root: Path
    hidden_bank_dir: Path
    dimension_targets_dir: Path
    checkpoint_dir: Path
    eval_dir: Path
    analysis_dir: Path

    @property
    def split_metadata(self) -> dict[str, Any]:
        return split_metadata_for_variant(self.resolved_variant)

    @property
    def identity(self) -> tuple[str, str, str, str]:
        return (self.requested_dataset, self.resolved_variant, self.split, self.model)

    def row_provenance(self, sample_id: Any) -> dict[str, Any]:
        payload = {
            "dataset": self.resolved_variant,
            "resolved_variant": self.resolved_variant,
            "split": self.split,
            "model": self.model,
            "sample_id": sample_id,
            "source_sample_id": sample_id,
        }
        metadata = self.split_metadata
        if metadata.get("source_dataset") is not None:
            payload["source_dataset"] = metadata.get("source_dataset")
        if metadata.get("virtual_split") is not None:
            payload["virtual_split"] = metadata.get("virtual_split")
        return payload

    def manifest_provenance(self, stage_output_dir: str | Path) -> dict[str, Any]:
        payload = {
            "requested_dataset": self.requested_dataset,
            "resolved_variant": self.resolved_variant,
            "split": self.split,
            "model": self.model,
            "pair_root": str(self.pair_root),
            "diaguq_root": str(self.diaguq_root),
            "stage_output_dir": str(Path(stage_output_dir)),
        }
        payload.update(self.split_metadata)
        return payload

    def stage_output_dir(self, stage: str) -> Path:
        key = _normalize_stage(stage)
        if key == "response_cache":
            return self.response_cache_root
        if key == "hidden_bank":
            return self.hidden_bank_dir
        if key in {"diagnostic_targets", "dimension_targets"}:
            return self.dimension_targets_dir
        if key in {"training", "checkpoints"}:
            return self.checkpoint_dir
        if key == "evaluation":
            return self.eval_dir
        if key == "export_analysis":
            return self.analysis_dir
        raise ValueError(f"unknown DiagUQ stage: {stage!r}")


def resolve_dataset_variant(dataset: str, split: Optional[str] = None) -> tuple[str, str]:
    """Resolve a requested dataset and optional split to a canonical variant."""
    base_dataset, embedded_split = split_dataset_and_raw(dataset)
    raw_split = split or embedded_split or DEFAULT_DATASET_SPLIT.get(base_dataset, "")
    if raw_split:
        variant = normalize_split_tag(base_dataset, raw_split)
        _, resolved_split = split_dataset_and_raw(variant)
        return variant, resolved_split or raw_split
    return dataset, embedded_split or "unknown"


def resolve_pair_context(
    dataset: str,
    model: str,
    split: Optional[str] = None,
    runtime_root: Optional[str | Path] = None,
) -> DiagUQPairContext:
    """Return the single authoritative artifact context for one pair."""
    root = Path(get_test_output_dir()) if runtime_root is None else Path(runtime_root)
    resolved_variant, resolved_split = resolve_dataset_variant(dataset, split)
    pair_root = root / resolved_variant / model
    diaguq_root = pair_root / "diaguq"
    return DiagUQPairContext(
        requested_dataset=dataset,
        resolved_variant=resolved_variant,
        split=resolved_split,
        model=model,
        test_output_root=root,
        pair_root=pair_root,
        diaguq_root=diaguq_root,
        response_cache_root=pair_root,
        hidden_bank_dir=diaguq_root / "hidden_bank",
        dimension_targets_dir=diaguq_root / "dimension_targets",
        checkpoint_dir=diaguq_root / "checkpoints",
        eval_dir=diaguq_root / "eval",
        analysis_dir=diaguq_root / "analysis",
    )


def _assert_valid_checkpoint_context(ctx: DiagUQPairContext) -> DiagUQPairContext:
    if "," in ctx.resolved_variant:
        raise ValueError(INVALID_CHECKPOINT_VARIANT_MESSAGE)
    return ctx


def resolve_checkpoint_context_for_eval(
    eval_ctx: DiagUQPairContext,
    train_pairs: Iterable[tuple[str, str]],
    checkpoint_dataset: Optional[str] = None,
) -> DiagUQPairContext:
    """Resolve the checkpoint root that corresponds to one evaluation pair.

    ``train_pairs`` must contain already resolved train artifact roots from the
    run plan. Display-only split strings such as ``"train,validation_train"``
    are rejected so they cannot become runtime artifact variants.
    """
    if checkpoint_dataset:
        return _assert_valid_checkpoint_context(
            resolve_pair_context(
                checkpoint_dataset,
                eval_ctx.model,
                runtime_root=eval_ctx.test_output_root,
            )
        )

    eval_base, _ = split_dataset_and_raw(eval_ctx.resolved_variant)
    matches: list[DiagUQPairContext] = []
    for train_dataset, train_model in train_pairs:
        if train_model != eval_ctx.model:
            continue
        train_ctx = _assert_valid_checkpoint_context(
            resolve_pair_context(
                train_dataset,
                train_model,
                runtime_root=eval_ctx.test_output_root,
            )
        )
        train_base, _ = split_dataset_and_raw(train_ctx.resolved_variant)
        if train_base == eval_base:
            matches.append(train_ctx)

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(
            "No matching train checkpoint context for eval pair: "
            f"eval_resolved_variant={eval_ctx.resolved_variant!r} model={eval_ctx.model!r}. "
            "Pass checkpoint_dataset explicitly for cross-dataset evaluation."
        )
    raise ValueError(
        "Multiple matching train checkpoint contexts for eval pair: "
        f"eval_resolved_variant={eval_ctx.resolved_variant!r} model={eval_ctx.model!r} "
        f"matches={[ctx.resolved_variant for ctx in matches]}"
    )


def assert_pair_output_path(ctx: DiagUQPairContext, path: str | Path) -> Path:
    """Fail if ``path`` is not owned by ``ctx``'s pair root."""
    candidate = Path(path)
    if not _is_relative_to(candidate, ctx.pair_root):
        raise AssertionError(f"output path is outside pair_root: {candidate} not under {ctx.pair_root}")
    text = _norm_text(candidate)
    if ctx.resolved_variant not in text or ctx.model not in text:
        raise AssertionError(
            "output path does not contain resolved_variant/model: "
            f"path={candidate} resolved_variant={ctx.resolved_variant} model={ctx.model}"
        )
    return candidate


def assert_diaguq_output_path(
    ctx: DiagUQPairContext,
    path: str | Path,
    *,
    stage_token: Optional[str] = None,
) -> Path:
    """Fail if a stage artifact path is not owned by ``ctx``'s DiagUQ root."""
    candidate = assert_pair_output_path(ctx, path)
    if not _is_relative_to(candidate, ctx.diaguq_root):
        raise AssertionError(f"output path is outside diaguq_root: {candidate} not under {ctx.diaguq_root}")
    if stage_token and stage_token not in _norm_text(candidate):
        raise AssertionError(f"output path missing stage token {stage_token!r}: {candidate}")
    return candidate


def assert_no_duplicate_output_dirs(
    contexts: Sequence[DiagUQPairContext],
    stage: str,
) -> None:
    """Fail when distinct pair contexts would write the same stage dir."""
    seen: dict[str, DiagUQPairContext] = {}
    for ctx in contexts:
        path = ctx.stage_output_dir(stage)
        key = _path_key(path)
        previous = seen.get(key)
        if previous is not None and previous.identity != ctx.identity:
            raise ValueError(
                "duplicate output directory for distinct pair contexts: "
                f"{path} maps {previous.identity} and {ctx.identity}"
            )
        seen[key] = ctx


def validate_stage_manifest_provenance(
    payload: Mapping[str, Any],
    *,
    manifest_path: Optional[str | Path] = None,
    ctx: Optional[DiagUQPairContext] = None,
) -> None:
    """Validate manifest-level pair provenance against paths and split names."""
    resolved_variant = str(payload.get("resolved_variant") or "")
    split = str(payload.get("split") or "")
    model = str(payload.get("model") or "")
    pair_root = Path(str(payload.get("pair_root") or ""))
    diaguq_root = Path(str(payload.get("diaguq_root") or ""))
    stage_output_dir = Path(str(payload.get("stage_output_dir") or ""))

    if not resolved_variant:
        raise ValueError("manifest missing resolved_variant")
    if not split:
        raise ValueError("manifest missing split")
    if not model:
        raise ValueError("manifest missing model")
    expected_variant, expected_split = split_dataset_and_raw(resolved_variant)
    del expected_variant
    if expected_split and split != expected_split:
        raise ValueError(
            f"manifest split {split!r} does not match resolved_variant {resolved_variant!r}"
        )
    pair_text = _norm_text(pair_root)
    if resolved_variant not in pair_text or model not in pair_text:
        raise ValueError(
            "manifest pair_root does not contain resolved_variant/model: "
            f"pair_root={pair_root} resolved_variant={resolved_variant} model={model}"
        )
    if diaguq_root and not _is_relative_to(diaguq_root, pair_root):
        raise ValueError(f"manifest diaguq_root is outside pair_root: {diaguq_root}")
    if stage_output_dir and not (
        _is_relative_to(stage_output_dir, diaguq_root)
        or stage_output_dir == pair_root
    ):
        raise ValueError(f"manifest stage_output_dir is outside owned roots: {stage_output_dir}")
    if manifest_path is not None:
        manifest = Path(manifest_path)
        if not _is_relative_to(manifest, stage_output_dir):
            raise ValueError(f"manifest file is outside stage_output_dir: {manifest}")
    if ctx is not None:
        if resolved_variant != ctx.resolved_variant or model != ctx.model or split != ctx.split:
            raise ValueError(
                "manifest provenance does not match pair context: "
                f"manifest={(resolved_variant, split, model)} ctx={(ctx.resolved_variant, ctx.split, ctx.model)}"
            )


def _normalize_stage(stage: str) -> str:
    return stage.replace("-", "_")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _norm_text(path: Path) -> str:
    return path.as_posix().replace("\\", "/")


def _path_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def contexts_for_pairs(
    pairs: Iterable[tuple[str, str]],
    *,
    runtime_root: Optional[str | Path] = None,
) -> list[DiagUQPairContext]:
    return [resolve_pair_context(dataset, model, runtime_root=runtime_root) for dataset, model in pairs]
