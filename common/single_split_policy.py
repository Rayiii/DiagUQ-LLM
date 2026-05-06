"""Single-split dataset policy helpers for DiagUQ."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from common.artifact_paths import normalize_split_tag, split_dataset_and_raw


DEFAULT_INTERNAL_TRAIN_RATIO = 0.7
DEFAULT_INTERNAL_SPLIT_SEED = 42
INTERNAL_SPLIT_POLICY = "internal_split"
SAME_SPLIT_DEBUG_POLICY = "same_split_debug"
TRUTHFULQA_SINGLE_SPLIT_MESSAGE = (
    "TruthfulQA has only validation split. Use --single-split-policy internal_split, "
    "--cross-dataset-eval, or --allow-single-split-same-eval for debug."
)


@dataclass(frozen=True)
class InternalSplitInfo:
    base_dataset: str
    source_split: str
    virtual_split: str
    source_variant: str
    virtual_variant: str


def internal_split_name(source_split: str, virtual_split: str) -> str:
    if virtual_split not in {"train", "eval"}:
        raise ValueError(f"virtual_split must be 'train' or 'eval', got {virtual_split!r}")
    return f"{source_split}_{virtual_split}"


def internal_split_variant(base_dataset: str, source_split: str, virtual_split: str) -> str:
    return normalize_split_tag(base_dataset, internal_split_name(source_split, virtual_split))


def parse_internal_split_variant(dataset_name: str) -> Optional[InternalSplitInfo]:
    base_dataset, raw_split = split_dataset_and_raw(dataset_name)
    if not raw_split:
        return None
    for role in ("train", "eval"):
        suffix = f"_{role}"
        if raw_split.endswith(suffix):
            source_split = raw_split[: -len(suffix)]
            if not source_split:
                return None
            return InternalSplitInfo(
                base_dataset=base_dataset,
                source_split=source_split,
                virtual_split=role,
                source_variant=normalize_split_tag(base_dataset, source_split),
                virtual_variant=normalize_split_tag(base_dataset, raw_split),
            )
    return None


def is_internal_split_variant(dataset_name: str) -> bool:
    return parse_internal_split_variant(dataset_name) is not None


def sample_goes_to_internal_train(
    sample_id: Any,
    *,
    seed: int = DEFAULT_INTERNAL_SPLIT_SEED,
    train_ratio: float = DEFAULT_INTERNAL_TRAIN_RATIO,
) -> bool:
    if not 0.0 < float(train_ratio) < 1.0:
        raise ValueError(f"internal train ratio must be in (0, 1), got {train_ratio!r}")
    payload = f"{int(seed)}:{sample_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / float(2**64)
    return value < float(train_ratio)


def row_belongs_to_internal_split(
    row: Mapping[str, Any],
    virtual_split: str,
    *,
    seed: int = DEFAULT_INTERNAL_SPLIT_SEED,
    train_ratio: float = DEFAULT_INTERNAL_TRAIN_RATIO,
) -> bool:
    sample_id = row.get("sample_id")
    if sample_id in (None, ""):
        raise ValueError("internal split row is missing sample_id")
    in_train = sample_goes_to_internal_train(sample_id, seed=seed, train_ratio=train_ratio)
    return in_train if virtual_split == "train" else not in_train


def split_metadata_for_variant(
    dataset_name: str,
    *,
    policy: Optional[str] = None,
    seed: int = DEFAULT_INTERNAL_SPLIT_SEED,
    train_ratio: float = DEFAULT_INTERNAL_TRAIN_RATIO,
    same_split_debug: bool = False,
    checkpoint_dataset: Optional[str] = None,
) -> dict[str, Any]:
    info = parse_internal_split_variant(dataset_name)
    if info is not None:
        return {
            "source_dataset": info.source_variant,
            "virtual_split": info.virtual_split,
            "internal_split_seed": int(seed),
            "internal_train_ratio": float(train_ratio),
            "split_policy": INTERNAL_SPLIT_POLICY,
            "held_out_evaluation": True,
            "same_split_evaluation": False,
        }
    if same_split_debug:
        base_dataset, raw_split = split_dataset_and_raw(dataset_name)
        source_variant = normalize_split_tag(base_dataset, raw_split) if raw_split else dataset_name
        return {
            "source_dataset": source_variant,
            "virtual_split": None,
            "internal_split_seed": int(seed),
            "internal_train_ratio": float(train_ratio),
            "split_policy": SAME_SPLIT_DEBUG_POLICY,
            "held_out_evaluation": False,
            "same_split_evaluation": True,
            "warning": "same-split run; not a main held-out result",
        }
    out: dict[str, Any] = {}
    if policy:
        out["split_policy"] = policy
    if checkpoint_dataset is not None:
        out["checkpoint_dataset"] = checkpoint_dataset
        out["eval_dataset"] = dataset_name
        out["cross_dataset_evaluation"] = checkpoint_dataset != dataset_name
        out["trained_on_truthfulqa"] = not checkpoint_dataset.startswith("truthfulqa")
    return out
