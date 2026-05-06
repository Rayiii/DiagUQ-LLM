"""Canonical filesystem layout for the DiagUQ response cache.

This module is the single source of truth for two things:

1. **Split-tag normalization** -- turning a (``dataset``, ``raw_split``) pair
   into the canonical ``<dataset>__<raw_split>`` token used throughout the
   pipeline. ``raw_split`` is *control flow* state (``"train"`` / ``"test"``);
   ``split_tag`` is *naming* state (``"triviaqa__train"`` /
   ``"wmt__test"``). The two MUST never be confused, because that confusion
   is exactly what produced bugs like ``triviaqa__triviaqa__train.jsonl``
   and ``triviaqa_mextend.json`` (missing ``__train``).

2. **Response-cache artifact paths** -- every reader / writer of the
   ``*_mextend.json`` family of files must call the helpers below instead
   of concatenating strings inline.

The on-disk layout produced by these helpers is::

    <test_output_root>/<split_tag>/<model>/<split_tag>_mextend.json
    <test_output_root>/<split_tag>/<model>/<split_tag>_mextend_samples.json
    <test_output_root>/<split_tag>/<model>/<split_tag>_mextend_rouge.json   # QA
    <test_output_root>/<split_tag>/<model>/<split_tag>_mextend_bleu.json    # WMT
    <test_output_root>/<split_tag>/<model>/<split_tag>_extend.json          # samples
    <test_output_root>/<split_tag>/<model>/<split_tag>_extend_samples.json
    <test_output_root>/<split_tag>/<model>/<split_tag>_semantic_entropy.json
    <test_output_root>/<split_tag>/<model>/response_answer_audit.csv
    <test_output_root>/<split_tag>/<model>/response_answer_audit.json
    <test_output_root>/ask4conf/<model>/<split_tag>.jsonl
    <test_output_root>/ask4conf/<model>/SUCCESSFUL__<split_tag>
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from common.runtime_paths import get_test_output_dir


__all__ = [
    "KNOWN_DATASETS",
    "normalize_split_tag",
    "split_dataset_and_raw",
    "test_output_root",
    "split_artifact_dir",
    "mextend_path",
    "mextend_samples_path",
    "mextend_rouge_path",
    "mextend_bleu_path",
    "mextend_metric_path",
    "extend_path",
    "extend_samples_path",
    "semantic_entropy_path",
    "response_answer_audit_csv_path",
    "response_answer_audit_json_path",
    "ask4conf_dir",
    "ask4conf_jsonl_path",
    "ask4conf_success_marker",
    "ask4conf_metadata_path",
    "ask4conf_aggregate_json_path",
]


# Datasets the response-cache pipeline knows how to address.
KNOWN_DATASETS = ("triviaqa", "coqa", "wmt", "mmlu", "ambigqa", "truthfulqa")


# ---------------------------------------------------------------------------
# Split-tag normalization
# ---------------------------------------------------------------------------


def normalize_split_tag(dataset: str, raw_split: str) -> str:
    """Return the canonical ``<dataset>__<raw_split>`` token.

    Idempotent: if ``raw_split`` already starts with ``"<dataset>__"`` (or
    equals ``dataset``) it is returned unchanged. ``raw_split`` may also be
    given pre-prefixed with leading ``__`` (e.g. ``"__train"``) -- those
    leading underscores are stripped before re-joining.

    Examples
    --------
    >>> normalize_split_tag("triviaqa", "train")
    'triviaqa__train'
    >>> normalize_split_tag("wmt", "test")
    'wmt__test'
    >>> normalize_split_tag("triviaqa", "triviaqa__train")
    'triviaqa__train'
    >>> normalize_split_tag("triviaqa", "__train")
    'triviaqa__train'
    """
    if not dataset:
        raise ValueError("dataset must be a non-empty string")
    raw = (raw_split or "").strip()
    if raw == "" or raw == dataset:
        return dataset
    if raw.startswith(f"{dataset}__"):
        return raw
    raw = raw.lstrip("_")
    if not raw:
        return dataset
    return f"{dataset}__{raw}"


def split_dataset_and_raw(split_tag: str) -> tuple[str, str]:
    """Split a canonical ``split_tag`` back into ``(dataset, raw_split)``.

    Returns ``(split_tag, "")`` if the tag has no ``__`` separator
    (e.g. ``"mmlu"`` itself).
    """
    if "__" in split_tag:
        ds, raw = split_tag.split("__", 1)
        return ds, raw
    return split_tag, ""


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


PathLike = Union[str, Path, None]


def test_output_root(output_root: PathLike = None) -> Path:
    if output_root is None:
        return Path(get_test_output_dir())
    return Path(output_root)


def split_artifact_dir(
    split_tag: str, model_name: str, output_root: PathLike = None
) -> Path:
    """``<root>/<split_tag>/<model>/`` -- the per-pair response-cache dir."""
    return test_output_root(output_root) / split_tag / model_name


def _per_pair_path(
    split_tag: str,
    model_name: str,
    suffix: str,
    output_root: PathLike = None,
) -> Path:
    return (
        split_artifact_dir(split_tag, model_name, output_root)
        / f"{split_tag}{suffix}"
    )


def mextend_path(split_tag, model_name, output_root=None) -> Path:
    """``<split_tag>_mextend.json`` -- main response cache."""
    return _per_pair_path(split_tag, model_name, "_mextend.json", output_root)


def mextend_samples_path(split_tag, model_name, output_root=None) -> Path:
    """``<split_tag>_mextend_samples.json`` -- early-progress snapshot."""
    return _per_pair_path(
        split_tag, model_name, "_mextend_samples.json", output_root
    )


def mextend_rouge_path(split_tag, model_name, output_root=None) -> Path:
    return _per_pair_path(
        split_tag, model_name, "_mextend_rouge.json", output_root
    )


def mextend_bleu_path(split_tag, model_name, output_root=None) -> Path:
    return _per_pair_path(
        split_tag, model_name, "_mextend_bleu.json", output_root
    )


def mextend_metric_path(split_tag, model_name, output_root=None) -> Path:
    """Pick rouge or bleu based on the dataset embedded in ``split_tag``."""
    dataset, _ = split_dataset_and_raw(split_tag)
    if dataset == "wmt":
        return mextend_bleu_path(split_tag, model_name, output_root)
    return mextend_rouge_path(split_tag, model_name, output_root)


def extend_path(split_tag, model_name, output_root=None) -> Path:
    """``<split_tag>_extend.json`` -- sampled-answers cache."""
    return _per_pair_path(split_tag, model_name, "_extend.json", output_root)


def extend_samples_path(split_tag, model_name, output_root=None) -> Path:
    return _per_pair_path(
        split_tag, model_name, "_extend_samples.json", output_root
    )


def semantic_entropy_path(split_tag, model_name, output_root=None) -> Path:
    return _per_pair_path(
        split_tag, model_name, "_semantic_entropy.json", output_root
    )


def response_answer_audit_csv_path(split_tag, model_name, output_root=None) -> Path:
    return split_artifact_dir(split_tag, model_name, output_root) / "response_answer_audit.csv"


def response_answer_audit_json_path(split_tag, model_name, output_root=None) -> Path:
    return split_artifact_dir(split_tag, model_name, output_root) / "response_answer_audit.json"


def ask4conf_dir(model_name: str, output_root: PathLike = None) -> Path:
    return test_output_root(output_root) / "ask4conf" / model_name


def ask4conf_jsonl_path(
    split_tag: str, model_name: str, output_root: PathLike = None
) -> Path:
    """``<root>/ask4conf/<model>/<split_tag>.jsonl``.

    Note: historically this was written as
    ``<ds>__<split_tag>.jsonl`` (e.g. ``triviaqa__triviaqa__train.jsonl``)
    -- a duplicated-name bug. The canonical form below uses ``split_tag``
    directly.
    """
    return ask4conf_dir(model_name, output_root) / f"{split_tag}.jsonl"


def ask4conf_success_marker(
    split_tag: str, model_name: str, output_root: PathLike = None
) -> Path:
    return (
        ask4conf_dir(model_name, output_root) / f"SUCCESSFUL__{split_tag}"
    )


def ask4conf_metadata_path(
    split_tag: str, model_name: str, output_root: PathLike = None
) -> Path:
    """``<root>/ask4conf/<model>/<split_tag>.meta.json``.

    A small JSON sidecar describing the ask4conf run for one
    ``(model, split_tag)``: expected/written/skipped/failed counts,
    parse-failure count, line count of the jsonl, and protocol version.
    Used by :func:`build-response-cache` to decide whether the stage
    needs to be re-run.
    """
    return ask4conf_dir(model_name, output_root) / f"{split_tag}.meta.json"


def ask4conf_aggregate_json_path(
    dataset: str, model_name: str, output_root: PathLike = None
) -> Path:
    """``<root>/ask4conf/<model>/<dataset>.json`` -- the aggregate JSON
    sometimes produced by the legacy llama_3_8b path."""
    return ask4conf_dir(model_name, output_root) / f"{dataset}.json"
