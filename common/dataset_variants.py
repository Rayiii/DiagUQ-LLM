"""Dataset variant parsing helpers for split-qualified DiagUQ names."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from common.artifact_paths import normalize_split_tag
from registry.dataset_registry import get_split_names


@dataclass(frozen=True)
class DatasetVariant:
    requested_dataset: str
    base_dataset: str
    split: str
    split_tag: str
    sub_variant: Optional[str] = None
    available_splits: Tuple[str, ...] = ()


def parse_dataset_variant(
    dataset_name: str,
    *,
    split: Optional[str] = None,
    prefer: str = "mduq",
) -> DatasetVariant:
    """Parse ``<dataset>__<split>`` names without confusing split tags.

    MMLU formatter keys include the task name, e.g.
    ``mmlu__abstract_algebra__validation``. In that case the final token is
    the split and the middle token(s) are carried as ``sub_variant``.
    """
    requested = str(dataset_name or "").strip()
    if not requested:
        raise ValueError("dataset_name must be a non-empty string")

    parts = requested.split("__")
    base_dataset = parts[0]
    try:
        available_splits = tuple(get_split_names(base_dataset, prefer=prefer))
    except Exception:
        available_splits = ()

    explicit_split = str(split).strip() if split is not None else ""
    sub_variant: Optional[str] = None
    if explicit_split:
        raw_split = explicit_split
    elif len(parts) == 1:
        raw_split = ""
    elif len(parts) > 2 and available_splits and parts[-1] in available_splits:
        raw_split = parts[-1]
        sub_variant = "__".join(parts[1:-1]) or None
    else:
        raw_split = "__".join(parts[1:])

    if raw_split:
        if sub_variant:
            split_tag = f"{base_dataset}__{sub_variant}__{raw_split}"
        else:
            split_tag = normalize_split_tag(base_dataset, raw_split)
    else:
        split_tag = requested

    return DatasetVariant(
        requested_dataset=requested,
        base_dataset=base_dataset,
        split=raw_split,
        split_tag=split_tag,
        sub_variant=sub_variant,
        available_splits=available_splits,
    )
