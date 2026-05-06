"""Split-qualified dataset variant loading for hidden-bank generation."""

from __future__ import annotations

from typing import Any, Callable, Optional

from common.artifact_paths import normalize_split_tag
from common.dataset_variants import DatasetVariant, parse_dataset_variant
from common.single_split_policy import internal_split_variant
from registry.dataset_registry import get_dataset_spec, get_split_names, list_diaguq_datasets


class UnsupportedDatasetVariantError(ValueError):
    """Raised when hidden-bank generation cannot load a dataset variant."""


SUPPORTED_HIDDEN_BANK_DATASETS = frozenset(
    {"triviaqa", "ambigqa", "truthfulqa", "mmlu", "wmt"}
)


def available_hidden_bank_variants() -> tuple[str, ...]:
    variants: list[str] = []
    for dataset in list_diaguq_datasets():
        if dataset not in SUPPORTED_HIDDEN_BANK_DATASETS:
            continue
        try:
            splits = get_split_names(dataset, prefer="mduq")
        except Exception:  # noqa: BLE001
            continue
        variants.extend(normalize_split_tag(dataset, split) for split in splits)
        spec = get_dataset_spec(dataset, prefer="mduq")
        if spec.get("default_single_split_policy") == "internal_split":
            source_splits = tuple(spec.get("available_splits") or splits)
            for source_split in source_splits:
                variants.append(internal_split_variant(dataset, source_split, "train"))
                variants.append(internal_split_variant(dataset, source_split, "eval"))
    return tuple(dict.fromkeys(variants))


def resolve_supported_dataset_variant(resolved_variant: str) -> DatasetVariant:
    variant = parse_dataset_variant(resolved_variant, prefer="mduq")
    available = available_hidden_bank_variants()
    if (
        variant.base_dataset not in SUPPORTED_HIDDEN_BANK_DATASETS
        or not variant.split
        or variant.split_tag not in available
    ):
        raise UnsupportedDatasetVariantError(
            "generate_X does not support "
            f"resolved_variant={resolved_variant!r}; available variants={available}"
        )
    return variant


def load_dataset_for_variant(
    resolved_variant: str,
    limit: Optional[int] = None,
    *,
    tokenizer: Any = None,
    cache: bool = True,
    allow_full_formatting: bool = False,
    internal_train_ratio: float = 0.7,
    internal_split_seed: int = 42,
    formatter_loader: Optional[Callable[..., list[dict[str, Any]]]] = None,
) -> list[dict[str, Any]]:
    """Load formatter rows for a split-qualified DiagUQ dataset variant."""
    variant = resolve_supported_dataset_variant(resolved_variant)
    if tokenizer is None:
        raise TypeError("load_dataset_for_variant requires tokenizer=<model tokenizer>")
    if formatter_loader is None:
        from data.formatters import load_formatted_dataset as formatter_loader

    return formatter_loader(
        variant.base_dataset,
        variant.split,
        tokenizer,
        limit=limit,
        cache=cache,
        dataset_variant=variant.split_tag,
        allow_full_formatting=allow_full_formatting,
        internal_train_ratio=internal_train_ratio,
        internal_split_seed=internal_split_seed,
    )
