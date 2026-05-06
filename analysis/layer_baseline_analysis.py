"""Paper-ready analysis utilities for fixed-layer DiagUQ baselines.

This module is intentionally read-only with respect to model artifacts: it
consumes existing ``layer_baseline_metrics.csv``,
``layer_baseline_predictions.csv`` and DiagUQ ``per_sample.csv`` files. It
does not train models, rebuild hidden banks, or run LLM generation.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from loguru import logger

from common.pair_context import DiagUQPairContext, resolve_pair_context


HEADLINE_METRICS = ("AUROC", "AUPRC", "AUARC", "ECE", "Brier")
HIGHER_IS_BETTER = ("AUROC", "AUPRC", "AUARC")
LOWER_IS_BETTER = ("ECE", "Brier")
DEFAULT_FEATURE_MODE = "query_answer_relation_concat"
LAYER_SCAN_PREFIX = "fixed_layer_scan"
BASELINE_METHODS = (
    "best_fixed_layer_mlp",
    "fixed_middle_layer_mlp",
    "last_layer_mlp",
    "uniform_multilayer_mean_mlp",
    "fixed_multilayer_concat_mlp",
    "oracle_best_fixed_layer",
)
BASELINE_PREFIX = {
    "best_fixed_layer_mlp": "best_fixed_layer",
    "fixed_middle_layer_mlp": "fixed_middle_layer",
    "last_layer_mlp": "last_layer",
    "uniform_multilayer_mean_mlp": "uniform_multilayer_mean",
    "fixed_multilayer_concat_mlp": "fixed_multilayer_concat",
    "oracle_best_fixed_layer": "oracle_best_fixed_layer",
}
DELTA_BASELINE_ALIASES = {
    "best_fixed_layer_mlp": "best_fixed_layer",
    "uniform_multilayer_mean_mlp": "uniform_mean",
    "fixed_multilayer_concat_mlp": "concat",
    "last_layer_mlp": "last_layer",
    "fixed_middle_layer_mlp": "middle_layer",
}
DIMENSION_METRICS = (
    "ambiguity_metric",
    "knowledge_gap_metric",
    "predictive_variability_metric",
    "mean_diagonal_margin",
    "max_prediction_prediction_corr",
)
DIMENSION_FOR_METRIC = {
    "ambiguity_metric": "ambiguity",
    "knowledge_gap_metric": "knowledge_gap",
    "predictive_variability_metric": "predictive_variability",
    "mean_diagonal_margin": "all_dimensions",
    "max_prediction_prediction_corr": "all_dimensions",
}
DIMENSIONS = ("ambiguity", "knowledge_gap", "predictive_variability")


class LayerAnalysisError(RuntimeError):
    """Raised for missing or inconsistent layer-analysis artifacts."""


def _np():
    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        raise LayerAnalysisError("NumPy is required for layer-baseline analysis") from exc
    return np


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fr:
        return [dict(row) for row in csv.DictReader(fr)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fw:
        writer = csv.DictWriter(fw, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(val) for val in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _csv_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _int_or_none(value: Any) -> Optional[int]:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _finite_mean(values: Sequence[Any]) -> float:
    finite = [_float(value) for value in values if math.isfinite(_float(value))]
    if not finite:
        return float("nan")
    return float(sum(finite) / len(finite))


def _metric_rows_for_feature_mode(rows: Sequence[Mapping[str, Any]], feature_mode: str = DEFAULT_FEATURE_MODE) -> list[dict[str, Any]]:
    materialized = [dict(row) for row in rows]
    matching = [row for row in materialized if row.get("feature_mode") == feature_mode]
    return matching or materialized


def _row_for_method(rows: Sequence[Mapping[str, Any]], method: str) -> Optional[Mapping[str, Any]]:
    for row in rows:
        if row.get("method") == method:
            return row
    return None


def _metric_from_row(row: Optional[Mapping[str, Any]], metric: str) -> float:
    if not row:
        return float("nan")
    return _float(row.get(metric))


def _load_diaguq_metrics(eval_ctx: DiagUQPairContext) -> dict[str, float]:
    metrics_path = eval_ctx.eval_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"missing DiagUQ metrics for comparison: {metrics_path}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    for row in payload:
        if isinstance(row, Mapping) and row.get("method") == "mduq":
            return {metric: _float(row.get(metric)) for metric in HEADLINE_METRICS}
    raise KeyError(f"metrics.json has no method='mduq' row: {metrics_path}")


def build_layer_vs_diaguq_summary_row(
    eval_ctx: DiagUQPairContext,
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_mode: str = DEFAULT_FEATURE_MODE,
) -> dict[str, Any]:
    rows = _metric_rows_for_feature_mode(rows, feature_mode)
    diaguq = _load_diaguq_metrics(eval_ctx)
    out: dict[str, Any] = {
        "dataset": eval_ctx.resolved_variant,
        "model": eval_ctx.model,
        "feature_mode": feature_mode,
        "n_eval": None,
    }
    mduq_n = None
    for metric in HEADLINE_METRICS:
        out[f"diaguq_{metric}"] = diaguq.get(metric, float("nan"))
    for method in BASELINE_METHODS:
        row = _row_for_method(rows, method)
        prefix = BASELINE_PREFIX[method]
        if row and mduq_n is None:
            mduq_n = _int_or_none(row.get("n_eval")) or _int_or_none(row.get("n_total"))
        for metric in HEADLINE_METRICS:
            out[f"{prefix}_{metric}"] = _metric_from_row(row, metric)
        out[f"{prefix}_method"] = method
    out["n_eval"] = mduq_n

    best_row = _row_for_method(rows, "best_fixed_layer_mlp")
    oracle_row = _row_for_method(rows, "oracle_best_fixed_layer")
    out["selected_best_fixed_layer"] = _int_or_none((best_row or {}).get("selected_layer") or (best_row or {}).get("layer_id"))
    out["oracle_best_layer"] = _int_or_none((oracle_row or {}).get("selected_layer") or (oracle_row or {}).get("layer_id"))
    out["selected_best_fixed_layer_score"] = _float((best_row or {}).get("selected_layer_score"))

    for metric in HIGHER_IS_BETTER:
        diaguq_value = _float(out.get(f"diaguq_{metric}"))
        for method, alias in DELTA_BASELINE_ALIASES.items():
            prefix = BASELINE_PREFIX[method]
            baseline_value = _float(out.get(f"{prefix}_{metric}"))
            out[f"diaguq_minus_{alias}_{metric}"] = (
                diaguq_value - baseline_value
                if math.isfinite(diaguq_value) and math.isfinite(baseline_value)
                else float("nan")
            )
    for metric in LOWER_IS_BETTER:
        diaguq_value = _float(out.get(f"diaguq_{metric}"))
        for method, alias in DELTA_BASELINE_ALIASES.items():
            prefix = BASELINE_PREFIX[method]
            baseline_value = _float(out.get(f"{prefix}_{metric}"))
            out[f"{alias}_minus_diaguq_{metric}"] = (
                baseline_value - diaguq_value
                if math.isfinite(diaguq_value) and math.isfinite(baseline_value)
                else float("nan")
            )

    # Legacy aliases kept for older scripts and notebooks.
    out["diaguq_minus_best_fixed_layer"] = out.get("diaguq_minus_best_fixed_layer_AUROC")
    out["diaguq_minus_uniform_mean"] = out.get("diaguq_minus_uniform_mean_AUROC")
    out["diaguq_minus_concat"] = out.get("diaguq_minus_concat_AUROC")

    warnings: list[str] = []
    if _float(out.get("diaguq_minus_best_fixed_layer_AUROC")) <= 0:
        warnings.append("DiagUQ does not outperform best_fixed_layer_mlp on AUROC")
    if _float(out.get("diaguq_minus_concat_AUROC")) < 0:
        warnings.append("fixed_multilayer_concat_mlp outperforms DiagUQ on AUROC")
    out["warnings"] = "; ".join(warnings)
    return out


def write_cross_dataset_layer_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_root: str | Path,
    model: str,
) -> dict[str, Path]:
    summary_rows = [dict(row) for row in rows]
    cross_dir = Path(output_root) / "_cross_dataset_summary" / model
    average = _average_summary_row(summary_rows, model=model)
    rows_with_average = [*summary_rows, average]
    csv_path = cross_dir / "all_layer_vs_diaguq_summary.csv"
    json_path = cross_dir / "all_layer_vs_diaguq_summary.json"
    _write_csv(csv_path, rows_with_average)
    _write_json(json_path, rows_with_average)
    write_paper_tables(cross_dir)
    return {"csv": csv_path, "json": json_path}


def _average_summary_row(rows: Sequence[Mapping[str, Any]], *, model: str) -> dict[str, Any]:
    metric_columns = [
        key
        for row in rows
        for key in row.keys()
        if any(key.endswith(f"_{metric}") for metric in HEADLINE_METRICS)
        or key.startswith("diaguq_minus_")
        or key.endswith("_minus_diaguq_ECE")
        or key.endswith("_minus_diaguq_Brier")
    ]
    metric_columns = list(dict.fromkeys(metric_columns))
    out: dict[str, Any] = {
        "dataset": "average",
        "model": model,
        "feature_mode": rows[0].get("feature_mode") if rows else DEFAULT_FEATURE_MODE,
        "n_datasets": len({str(row.get("dataset")) for row in rows if row.get("dataset")}),
    }
    for key in metric_columns:
        out[key] = _finite_mean([row.get(key) for row in rows])
    for method, alias in DELTA_BASELINE_ALIASES.items():
        auroc_key = f"diaguq_minus_{alias}_AUROC"
        ece_key = f"{alias}_minus_diaguq_ECE"
        brier_key = f"{alias}_minus_diaguq_Brier"
        out[f"n_datasets_diaguq_beats_{alias}_AUROC"] = sum(1 for row in rows if _float(row.get(auroc_key)) > 0)
        out[f"n_datasets_diaguq_beats_{alias}_ECE"] = sum(1 for row in rows if _float(row.get(ece_key)) > 0)
        out[f"n_datasets_diaguq_beats_{alias}_Brier"] = sum(1 for row in rows if _float(row.get(brier_key)) > 0)
    return out


def aggregate_existing_layer_summaries(
    eval_pairs: Iterable[tuple[str, str]],
    *,
    output_root: str | Path,
) -> list[Path]:
    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    for dataset, model in eval_pairs:
        ctx = resolve_pair_context(dataset, model, runtime_root=output_root)
        path = ctx.analysis_dir / "layer_vs_diaguq_summary.csv"
        rows = _read_csv(path)
        if not rows:
            continue
        rows_by_model.setdefault(model, []).extend(rows)
    written: list[Path] = []
    for model, rows in rows_by_model.items():
        paths = write_cross_dataset_layer_summary(rows, output_root=output_root, model=model)
        written.extend(paths.values())
    return written


def _as_array(values: Sequence[Any]):
    np = _np()
    return np.asarray([_float(value) for value in values], dtype=np.float64)


def _binary_mask(score, label):
    np = _np()
    mask = np.isfinite(score) & np.isfinite(label)
    return score[mask], label[mask]


def _trapz(y, x) -> float:
    np = _np()
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _auroc(score, label) -> float:
    np = _np()
    score, label = _binary_mask(score, label)
    if score.size == 0 or len(np.unique(label)) < 2:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    label = (label[order] >= 0.5).astype(np.float64)
    pos = float(label.sum())
    neg = float(label.size - pos)
    if pos <= 0 or neg <= 0:
        return float("nan")
    tpr = np.concatenate([[0.0], np.cumsum(label) / pos, [1.0]])
    fpr = np.concatenate([[0.0], np.cumsum(1.0 - label) / neg, [1.0]])
    return _trapz(tpr, fpr)


def _auprc(score, label) -> float:
    np = _np()
    score, label = _binary_mask(score, label)
    if score.size == 0 or label.sum() == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    label = (label[order] >= 0.5).astype(np.float64)
    tp = np.cumsum(label)
    fp = np.cumsum(1.0 - label)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / max(label.sum(), 1e-12)
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def _auarc(confidence, correct) -> float:
    np = _np()
    confidence, correct = _binary_mask(confidence, correct)
    n = confidence.size
    if n == 0:
        return float("nan")
    order = np.argsort(-confidence, kind="mergesort")
    correct = (correct[order] >= 0.5).astype(np.float64)
    acc = np.cumsum(correct) / np.arange(1, n + 1)
    coverage = np.arange(1, n + 1) / n
    return _trapz(acc, coverage)


def _ece(confidence, correct, n_bins: int = 15) -> float:
    np = _np()
    confidence, correct = _binary_mask(confidence, correct)
    if confidence.size == 0:
        return float("nan")
    confidence = np.clip(confidence, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence >= lo) & (confidence <= hi) if hi == 1.0 else (confidence >= lo) & (confidence < hi)
        if not mask.any():
            continue
        out += (mask.sum() / confidence.size) * abs(float(confidence[mask].mean()) - float(correct[mask].mean()))
    return float(out)


def _brier(confidence, correct) -> float:
    np = _np()
    confidence, correct = _binary_mask(confidence, correct)
    if confidence.size == 0:
        return float("nan")
    return float(np.mean((confidence - (correct >= 0.5).astype(np.float64)) ** 2))


def _method_metrics(confidence, correct) -> dict[str, float]:
    return {
        "AUROC": _auroc(confidence, correct),
        "AUPRC": _auprc(confidence, correct),
        "AUARC": _auarc(confidence, correct),
        "ECE": _ece(confidence, correct),
        "Brier": _brier(confidence, correct),
    }


def _delta_metric(metric: str, diaguq: float, baseline: float) -> float:
    if not (math.isfinite(diaguq) and math.isfinite(baseline)):
        return float("nan")
    if metric in LOWER_IS_BETTER:
        return baseline - diaguq
    return diaguq - baseline


def _is_missing_sample_id(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == "" or value.strip().lower() in {"none", "nan"}
    return False


def _canonical_sample_id(value: Any) -> str:
    return str(value).strip()


def _row_dataset_identity(row: Mapping[str, Any]) -> Optional[str]:
    for key in ("resolved_variant", "dataset"):
        value = row.get(key)
        if not _is_missing_sample_id(value):
            return str(value).strip()
    return None


def _sample_id_dtype_summary(rows: Sequence[Mapping[str, Any]]) -> Any:
    counts = Counter(type(row.get("sample_id")).__name__ for row in rows if "sample_id" in row)
    if not counts:
        return None
    if len(counts) == 1:
        return next(iter(counts))
    return dict(sorted(counts.items()))


def _unique_sample_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for row in rows:
        value = row.get("sample_id")
        if _is_missing_sample_id(value):
            continue
        sample_id = _canonical_sample_id(value)
        if sample_id not in seen:
            seen.add(sample_id)
            out.append(sample_id)
    return out


def _require_sample_id_column(rows: Sequence[Mapping[str, Any]], *, label: str, path: Path) -> None:
    if not rows:
        return
    if not any("sample_id" in row for row in rows):
        raise LayerAnalysisError(f"bootstrap-layer-comparison requires sample_id in {label}: {path}")
    missing = [idx for idx, row in enumerate(rows) if _is_missing_sample_id(row.get("sample_id"))]
    if missing:
        raise LayerAnalysisError(
            f"bootstrap-layer-comparison requires non-empty sample_id in {label}: "
            f"{path}; missing_count={len(missing)} examples={missing[:5]}"
        )


def _join_key(row: Mapping[str, Any], *, use_dataset_identity: bool) -> tuple[str, Optional[str]]:
    sample_id = _canonical_sample_id(row.get("sample_id"))
    dataset_identity = _row_dataset_identity(row) if use_dataset_identity else None
    return sample_id, dataset_identity


def _selected_method_rows(rows: Sequence[Mapping[str, Any]], method: str) -> list[dict[str, Any]]:
    method_rows = [dict(row) for row in rows if row.get("method") == method]
    feature_mode_rows = [row for row in method_rows if row.get("feature_mode") == DEFAULT_FEATURE_MODE]
    return feature_mode_rows or method_rows


def _match_rows(
    diaguq_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[Mapping[str, Any], Mapping[str, Any]]], dict[str, Any]]:
    use_dataset_identity = any(_row_dataset_identity(row) for row in diaguq_rows) and any(
        _row_dataset_identity(row) for row in baseline_rows
    )
    by_key: dict[tuple[str, Optional[str]], Mapping[str, Any]] = {}
    duplicate_diaguq: list[tuple[str, Optional[str]]] = []
    for row in diaguq_rows:
        key = _join_key(row, use_dataset_identity=use_dataset_identity)
        if key in by_key:
            duplicate_diaguq.append(key)
            continue
        by_key[key] = row
    if duplicate_diaguq:
        raise LayerAnalysisError(f"duplicate DiagUQ sample_id keys in per_sample.csv: examples={duplicate_diaguq[:5]}")

    seen_layer: set[tuple[str, Optional[str]]] = set()
    duplicate_layer: list[tuple[str, Optional[str]]] = []
    paired: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    matched_keys: set[tuple[str, Optional[str]]] = set()
    unmatched_layer: list[str] = []
    for row in baseline_rows:
        key = _join_key(row, use_dataset_identity=use_dataset_identity)
        if key in seen_layer:
            duplicate_layer.append(key)
            continue
        seen_layer.add(key)
        left = by_key.get(key)
        if left is None:
            unmatched_layer.append(key[0])
            continue
        paired.append((left, row))
        matched_keys.add(key)
    if duplicate_layer:
        raise LayerAnalysisError(f"duplicate layer baseline sample_id keys for method: examples={duplicate_layer[:5]}")
    unmatched_diaguq = [key[0] for key in by_key if key not in matched_keys]
    sanity = {
        "n_matched": len(paired),
        "used_dataset_identity_in_join": use_dataset_identity,
        "unmatched_diaguq_sample_examples": unmatched_diaguq[:5],
        "unmatched_layer_sample_examples": unmatched_layer[:5],
    }
    return paired, sanity


def _paired_rows(diaguq_rows: Sequence[Mapping[str, Any]], baseline_rows: Sequence[Mapping[str, Any]]) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    paired, _ = _match_rows(diaguq_rows, baseline_rows)
    return paired


def _used_fallback_index_as_sample_id(analysis_dir: Path) -> bool:
    path = Path(analysis_dir) / "sample_id_alignment_sanity.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(payload.get("used_fallback_index_as_sample_id"))


def _bootstrap_one(
    *,
    dataset: str,
    model: str,
    method: str,
    paired: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    n_bootstrap: int,
    seed: int,
    ci_level: float,
) -> list[dict[str, Any]]:
    np = _np()
    n = len(paired)
    base = {
        "dataset": dataset,
        "model": model,
        "baseline_method": method,
        "n_matched": n,
        "n_bootstrap": int(n_bootstrap),
    }
    if n < 30:
        logger.warning("[bootstrap-layer] dataset={} method={} matched samples < 30; skipping CI", dataset, method)
        return [
            {
                **base,
                "metric": metric,
                "delta_mean": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "p_positive": float("nan"),
                "significant_positive": False,
                "significant_negative": False,
                "skip_reason": "matched_sample_count_lt_30",
            }
            for metric in HEADLINE_METRICS
        ]

    diaguq_conf = _as_array([left.get("mduq_confidence") for left, _ in paired])
    baseline_conf = _as_array([right.get("predicted_confidence") for _, right in paired])
    correct = _as_array([left.get("correct") for left, _ in paired])
    observed_d = _method_metrics(diaguq_conf, correct)
    observed_b = _method_metrics(baseline_conf, correct)
    rng = np.random.default_rng(int(seed))
    deltas = {metric: [] for metric in HEADLINE_METRICS}
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        boot_d = _method_metrics(diaguq_conf[idx], correct[idx])
        boot_b = _method_metrics(baseline_conf[idx], correct[idx])
        for metric in HEADLINE_METRICS:
            value = _delta_metric(metric, boot_d[metric], boot_b[metric])
            if math.isfinite(value):
                deltas[metric].append(value)
    alpha = (1.0 - float(ci_level)) / 2.0
    out = []
    for metric in HEADLINE_METRICS:
        values = np.asarray(deltas[metric], dtype=np.float64)
        observed = _delta_metric(metric, observed_d[metric], observed_b[metric])
        if values.size == 0:
            ci_low = ci_high = p_positive = float("nan")
        else:
            ci_low = float(np.quantile(values, alpha))
            ci_high = float(np.quantile(values, 1.0 - alpha))
            p_positive = float((values > 0).mean())
        out.append(
            {
                **base,
                "metric": metric,
                "delta_mean": observed,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_positive": p_positive,
                "significant_positive": bool(math.isfinite(ci_low) and ci_low > 0),
                "significant_negative": bool(math.isfinite(ci_high) and ci_high < 0),
                "skip_reason": None,
            }
        )
    return out


def run_bootstrap_layer_comparison(
    eval_pairs: Iterable[tuple[str, str]],
    *,
    output_root: str | Path,
    n_bootstrap: int = 1000,
    seed: int = 42,
    ci_level: float = 0.95,
    force: bool = False,
) -> list[dict[str, Any]]:
    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    all_dataset_rows: list[dict[str, Any]] = []
    for dataset, model in eval_pairs:
        ctx = resolve_pair_context(dataset, model, runtime_root=output_root)
        diaguq_path = ctx.analysis_dir / "per_sample.csv"
        baseline_path = ctx.eval_dir / "layer_baseline_predictions.csv"
        if not baseline_path.is_file():
            raise LayerAnalysisError(
                f"missing {baseline_path}. Run ablate-layers with prediction export enabled first."
            )
        if not diaguq_path.is_file():
            raise LayerAnalysisError(f"missing DiagUQ per-sample predictions: {diaguq_path}")
        diaguq_rows = _read_csv(diaguq_path)
        baseline_rows = _read_csv(baseline_path)
        methods = [method for method in BASELINE_METHODS if any(row.get("method") == method for row in baseline_rows)]
        method_rows_by_method = {method: _selected_method_rows(baseline_rows, method) for method in methods}
        match_sanity: dict[str, Any] = {
            "dataset": ctx.resolved_variant,
            "model": model,
            "diaguq_per_sample_csv": str(diaguq_path),
            "layer_baseline_predictions_csv": str(baseline_path),
            "n_diaguq_rows": len(diaguq_rows),
            "n_diaguq_unique_sample_ids": len(_unique_sample_ids(diaguq_rows)),
            "n_layer_rows": len(baseline_rows),
            "n_layer_unique_sample_ids_by_method": {
                method: len(_unique_sample_ids(method_rows)) for method, method_rows in method_rows_by_method.items()
            },
            "n_matched_by_method": {},
            "sample_id_dtype_diaguq": _sample_id_dtype_summary(diaguq_rows),
            "sample_id_dtype_layer": _sample_id_dtype_summary(baseline_rows),
            "unmatched_diaguq_sample_examples": {},
            "unmatched_layer_sample_examples": {},
            "used_fallback_index_as_sample_id": _used_fallback_index_as_sample_id(ctx.analysis_dir),
        }
        match_sanity_path = ctx.analysis_dir / "layer_bootstrap_match_sanity.json"
        try:
            _require_sample_id_column(diaguq_rows, label="DiagUQ analysis/per_sample.csv", path=diaguq_path)
            _require_sample_id_column(baseline_rows, label="layer_baseline_predictions.csv", path=baseline_path)
        except LayerAnalysisError as exc:
            match_sanity["error"] = str(exc)
            _write_json(match_sanity_path, match_sanity)
            raise
        dataset_rows: list[dict[str, Any]] = []
        for offset, method in enumerate(methods):
            method_rows = method_rows_by_method[method]
            try:
                paired, method_sanity = _match_rows(diaguq_rows, method_rows)
            except LayerAnalysisError as exc:
                match_sanity["error"] = str(exc)
                _write_json(match_sanity_path, match_sanity)
                raise
            match_sanity["n_matched_by_method"][method] = method_sanity["n_matched"]
            match_sanity["unmatched_diaguq_sample_examples"][method] = method_sanity["unmatched_diaguq_sample_examples"]
            match_sanity["unmatched_layer_sample_examples"][method] = method_sanity["unmatched_layer_sample_examples"]
            match_sanity.setdefault("used_dataset_identity_in_join_by_method", {})[method] = method_sanity[
                "used_dataset_identity_in_join"
            ]
            dataset_rows.extend(
                _bootstrap_one(
                    dataset=ctx.resolved_variant,
                    model=model,
                    method=method,
                    paired=paired,
                    n_bootstrap=n_bootstrap,
                    seed=int(seed) + offset,
                    ci_level=ci_level,
                )
            )
        ctx.analysis_dir.mkdir(parents=True, exist_ok=True)
        _write_json(match_sanity_path, match_sanity)
        csv_path = ctx.analysis_dir / "layer_bootstrap_ci.csv"
        json_path = ctx.analysis_dir / "layer_bootstrap_ci.json"
        if force or not csv_path.exists():
            _write_csv(csv_path, dataset_rows)
            _write_json(json_path, dataset_rows)
        rows_by_model.setdefault(model, []).extend(dataset_rows)
        all_dataset_rows.extend(dataset_rows)

    for model, rows in rows_by_model.items():
        cross_dir = Path(output_root) / "_cross_dataset_summary" / model
        _write_csv(cross_dir / "all_layer_bootstrap_ci.csv", rows)
        _write_json(cross_dir / "all_layer_bootstrap_ci.json", rows)
        write_paper_tables(cross_dir)
    return all_dataset_rows


def _layer_scan_rows(rows: Sequence[Mapping[str, Any]], feature_mode: str = DEFAULT_FEATURE_MODE) -> list[dict[str, Any]]:
    filtered = [
        dict(row)
        for row in rows
        if str(row.get("method") or "").startswith(LAYER_SCAN_PREFIX)
        and _int_or_none(row.get("layer_id")) is not None
    ]
    matching = [row for row in filtered if row.get("feature_mode") == feature_mode]
    return matching or filtered


def _matrix_rows_by_layer(
    rows: Sequence[Mapping[str, Any]],
    *,
    row_key: str,
    value_key: str,
    layer_key: str = "layer_id",
) -> tuple[list[dict[str, Any]], list[int]]:
    layers = sorted({int(_int_or_none(row.get(layer_key))) for row in rows if _int_or_none(row.get(layer_key)) is not None})
    grouped: dict[str, dict[int, float]] = {}
    meta: dict[str, dict[str, Any]] = {}
    for row in rows:
        layer = _int_or_none(row.get(layer_key))
        if layer is None:
            continue
        key = str(row.get(row_key))
        grouped.setdefault(key, {})[layer] = _float(row.get(value_key))
        meta.setdefault(key, {k: row.get(k) for k in ("dataset", "dimension", "metric", "target_status", "target_source", "target_reliability") if k in row})
    matrix_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        out = dict(meta.get(key) or {})
        out[row_key] = key
        for layer in layers:
            out[f"layer_{layer}"] = grouped[key].get(layer, float("nan"))
        matrix_rows.append(out)
    return matrix_rows, layers


def _plot_heatmap(
    matrix_rows: Sequence[Mapping[str, Any]],
    *,
    row_label_key: str,
    layers: Sequence[int],
    title: str,
    output_base: Path,
    output_format: str,
    best_by_row: Optional[Mapping[str, int]] = None,
) -> list[Path]:
    if output_format == "none":
        return []
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        raise LayerAnalysisError("matplotlib and NumPy are required for heatmap plotting") from exc
    labels = [str(row.get(row_label_key)) for row in matrix_rows]
    values = np.asarray([[ _float(row.get(f"layer_{layer}")) for layer in layers] for row in matrix_rows], dtype=np.float64)
    fig_width = max(6.0, 0.55 * len(layers) + 2.0)
    fig_height = max(3.0, 0.45 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    masked = np.ma.masked_invalid(values)
    im = ax.imshow(masked, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(layers)), [str(layer) for layer in layers], rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Candidate layer")
    ax.set_ylabel(row_label_key.replace("_", " ").title())
    ax.set_title(title)
    for row_idx, row_label in enumerate(labels):
        best_layer = (best_by_row or {}).get(row_label)
        for col_idx, layer in enumerate(layers):
            value = values[row_idx, col_idx]
            if math.isfinite(float(value)):
                marker = "★\n" if best_layer == layer else ""
                ax.text(col_idx, row_idx, f"{marker}{value:.3f}", ha="center", va="center", fontsize=8, color="white")
            if best_layer == layer:
                ax.add_patch(plt.Rectangle((col_idx - 0.5, row_idx - 0.5), 1, 1, fill=False, edgecolor="white", linewidth=2.0))
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    written: list[Path] = []
    formats = ("png", "pdf") if output_format == "both" else (output_format,)
    for fmt in formats:
        path = output_base.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=300 if fmt == "png" else None)
        written.append(path)
    plt.close(fig)
    return written


def _best_layer_rows(dataset_rows: Sequence[Mapping[str, Any]], metric: str) -> list[dict[str, Any]]:
    out = []
    by_dataset: dict[str, list[Mapping[str, Any]]] = {}
    for row in dataset_rows:
        by_dataset.setdefault(str(row.get("dataset")), []).append(row)
    for dataset, rows in sorted(by_dataset.items()):
        ranked = sorted(
            [row for row in rows if math.isfinite(_float(row.get(metric)))],
            key=lambda row: _float(row.get(metric)),
            reverse=True,
        )
        if not ranked:
            continue
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else {}
        worst = ranked[-1]
        out.append(
            {
                "dataset": dataset,
                "best_layer_by_AUROC": _int_or_none(best.get("layer_id")),
                "best_layer_AUROC": _float(best.get(metric)),
                "second_best_layer": _int_or_none(second.get("layer_id")),
                "second_best_layer_AUROC": _float(second.get(metric)),
                "layer_spread": _float(best.get(metric)) - _float(worst.get(metric)),
            }
        )
    return out


def _dimension_semantics(ctx: DiagUQPairContext, dimension: str) -> dict[str, Any]:
    if dimension == "all_dimensions":
        return {"target_status": "available", "target_source": "multiple_dimensions", "target_reliability": float("nan"), "available": True}
    path = ctx.dimension_targets_dir / "dimension_targets.json"
    if not path.is_file():
        return {"target_status": "unknown", "target_source": "unknown", "target_reliability": float("nan"), "available": True}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"target_status": "unknown", "target_source": "unknown", "target_reliability": float("nan"), "available": True}
    status_key = f"dim_{dimension}_target_status"
    source_key = f"dim_{dimension}_target_source"
    reliability_key = f"dim_{dimension}_target_reliability"
    statuses = [str(row.get(status_key) or "") for row in rows if isinstance(row, Mapping)]
    sources = [str(row.get(source_key) or "") for row in rows if isinstance(row, Mapping)]
    reliabilities = [_float(row.get(reliability_key)) for row in rows if isinstance(row, Mapping)]
    available = any(status not in {"", "missing", "unavailable"} for status in statuses)
    status = max(set(statuses), key=statuses.count) if statuses else "unknown"
    source = max(set(sources), key=sources.count) if sources else "unknown"
    return {
        "target_status": status,
        "target_source": source,
        "target_reliability": _finite_mean(reliabilities),
        "available": available,
    }


def run_layer_heatmap_analysis(
    eval_pairs: Iterable[tuple[str, str]],
    *,
    output_root: str | Path,
    metric: str = "AUROC",
    output_format: str = "both",
    include_dimension_heatmaps: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if output_format not in {"png", "pdf", "both", "none"}:
        raise ValueError("output_format must be one of png, pdf, both, none")
    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    ctxs: list[DiagUQPairContext] = []
    for dataset, model in eval_pairs:
        ctx = resolve_pair_context(dataset, model, runtime_root=output_root)
        ctxs.append(ctx)
        metric_rows = _layer_scan_rows(_read_csv(ctx.eval_dir / "layer_baseline_metrics.csv"))
        for row in metric_rows:
            row["dataset"] = ctx.resolved_variant
            row["model"] = model
        rows_by_model.setdefault(model, []).extend(metric_rows)
    if dry_run:
        return {"planned_models": sorted(rows_by_model), "planned_pairs": [(ctx.resolved_variant, ctx.model) for ctx in ctxs]}

    result: dict[str, Any] = {"dataset_layer_matrices": [], "best_layer_tables": [], "dimension_matrices": []}
    for model, rows in rows_by_model.items():
        cross_dir = Path(output_root) / "_cross_dataset_summary" / model
        matrix_rows, layers = _matrix_rows_by_layer(rows, row_key="dataset", value_key=metric)
        matrix_csv = cross_dir / "dataset_layer_auroc_matrix.csv"
        _write_csv(matrix_csv, matrix_rows)
        best_rows = _best_layer_rows(rows, metric)
        best_csv = cross_dir / "best_layer_by_dataset.csv"
        _write_csv(best_csv, best_rows)
        best_by_dataset = {str(row["dataset"]): int(row["best_layer_by_AUROC"]) for row in best_rows if row.get("best_layer_by_AUROC") not in (None, "")}
        _plot_heatmap(
            matrix_rows,
            row_label_key="dataset",
            layers=layers,
            title=f"Dataset x Layer {metric}",
            output_base=cross_dir / "dataset_layer_auroc_heatmap",
            output_format=output_format,
            best_by_row=best_by_dataset,
        )
        result["dataset_layer_matrices"].append(str(matrix_csv))
        result["best_layer_tables"].append(str(best_csv))

        if include_dimension_heatmaps:
            dimension_rows = _run_dimension_heatmaps_for_model(ctxs, model, output_root=output_root, output_format=output_format)
            result["dimension_matrices"].extend(dimension_rows)
        write_paper_tables(cross_dir)
    return result


def _run_dimension_heatmaps_for_model(
    ctxs: Sequence[DiagUQPairContext],
    model: str,
    *,
    output_root: str | Path,
    output_format: str,
) -> list[str]:
    cross_dir = Path(output_root) / "_cross_dataset_summary" / model
    all_matrix_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    written: list[str] = []
    for ctx in ctxs:
        if ctx.model != model:
            continue
        scan_rows = _layer_scan_rows(_read_csv(ctx.eval_dir / "layer_baseline_metrics.csv"))
        dataset_matrix_rows: list[dict[str, Any]] = []
        for metric in DIMENSION_METRICS:
            dimension = DIMENSION_FOR_METRIC[metric]
            semantics = _dimension_semantics(ctx, dimension)
            metric_rows = []
            if semantics.get("available"):
                for row in scan_rows:
                    value = _float(row.get(metric))
                    metric_rows.append(
                        {
                            "dataset": ctx.resolved_variant,
                            "dimension": dimension,
                            "metric": metric,
                            "layer_id": _int_or_none(row.get("layer_id")),
                            "value": value,
                            "target_status": semantics.get("target_status"),
                            "target_source": semantics.get("target_source"),
                            "target_reliability": semantics.get("target_reliability"),
                        }
                    )
            matrix_rows, layers = _matrix_rows_by_layer(metric_rows, row_key="dimension", value_key="value")
            for row in matrix_rows:
                row["dataset"] = ctx.resolved_variant
                row["metric"] = metric
                row["target_status"] = semantics.get("target_status")
                row["target_source"] = semantics.get("target_source")
                row["target_reliability"] = semantics.get("target_reliability")
            dataset_matrix_rows.extend(matrix_rows)
            all_matrix_rows.extend(matrix_rows)
            ranked = sorted(metric_rows, key=lambda row: _float(row.get("value")), reverse=True)
            ranked = [row for row in ranked if math.isfinite(_float(row.get("value")))]
            if ranked:
                best = ranked[0]
                second = ranked[1] if len(ranked) > 1 else {}
                worst = ranked[-1]
                best_rows.append(
                    {
                        "dataset": ctx.resolved_variant,
                        "dimension": dimension,
                        "metric": metric,
                        "best_layer": best.get("layer_id"),
                        "best_layer_score": best.get("value"),
                        "second_best_layer": second.get("layer_id"),
                        "second_best_layer_score": second.get("value"),
                        "layer_spread": _float(best.get("value")) - _float(worst.get("value")),
                        "target_status": semantics.get("target_status"),
                        "target_source": semantics.get("target_source"),
                        "target_reliability": semantics.get("target_reliability"),
                    }
                )
            if matrix_rows:
                best_by_dimension = {str(row.get("dimension")): _best_layer_from_matrix(row, layers) for row in matrix_rows}
                _plot_heatmap(
                    matrix_rows,
                    row_label_key="dimension",
                    layers=layers,
                    title=f"{ctx.resolved_variant} Layer x Dimension {metric}",
                    output_base=ctx.analysis_dir / f"layer_dimension_heatmap_{metric}",
                    output_format=output_format,
                    best_by_row=best_by_dimension,
                )
        dataset_best_layers = {row["best_layer"] for row in best_rows if row.get("dataset") == ctx.resolved_variant and row.get("best_layer") not in (None, "")}
        note = "dimension-specific layer preference observed."
        if len(dataset_best_layers) <= 1 and dataset_best_layers:
            note = "dimension-specific layer preference is weak for this dataset."
        for row in best_rows:
            if row.get("dataset") == ctx.resolved_variant:
                row["preference_note"] = note
        matrix_csv = ctx.analysis_dir / "layer_dimension_metric_matrix.csv"
        _write_csv(matrix_csv, dataset_matrix_rows)
        written.append(str(matrix_csv))
    _write_csv(cross_dir / "all_layer_dimension_metric_matrix.csv", all_matrix_rows)
    _write_csv(cross_dir / "layer_dimension_best_layer_summary.csv", best_rows)
    written.extend([
        str(cross_dir / "all_layer_dimension_metric_matrix.csv"),
        str(cross_dir / "layer_dimension_best_layer_summary.csv"),
    ])
    return written


def _best_layer_from_matrix(row: Mapping[str, Any], layers: Sequence[int]) -> Optional[int]:
    best_layer = None
    best_value = -float("inf")
    for layer in layers:
        value = _float(row.get(f"layer_{layer}"))
        if math.isfinite(value) and value > best_value:
            best_layer = int(layer)
            best_value = value
    return best_layer


def _format_float(value: Any, digits: int = 3) -> str:
    number = _float(value)
    return f"{number:.{digits}f}" if math.isfinite(number) else "NA"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _latex_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    cols = "l" * len(headers)
    row_end = r" \\"
    lines = [f"\\begin{{tabular}}{{{cols}}}", "\\toprule", " & ".join(headers) + row_end, "\\midrule"]
    for row in rows:
        lines.append(" & ".join(str(value) for value in row) + row_end)
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def write_paper_tables(cross_dir: str | Path) -> dict[str, Path]:
    cross = Path(cross_dir)
    summary_rows = [row for row in _read_csv(cross / "all_layer_vs_diaguq_summary.csv") if row.get("dataset") != "average"]
    bootstrap_rows = _read_csv(cross / "all_layer_bootstrap_ci.csv")
    best_layer_rows = _read_csv(cross / "best_layer_by_dataset.csv")
    written: dict[str, Path] = {}
    if summary_rows:
        main_headers = ["Dataset", "Last", "Middle", "Best Fixed", "Uniform", "Concat", "DiagUQ", "Δ vs Best Fixed", "Δ vs Uniform"]
        main_rows = [
            [
                row.get("dataset"),
                _format_float(row.get("last_layer_AUROC")),
                _format_float(row.get("fixed_middle_layer_AUROC")),
                _format_float(row.get("best_fixed_layer_AUROC")),
                _format_float(row.get("uniform_multilayer_mean_AUROC")),
                _format_float(row.get("fixed_multilayer_concat_AUROC")),
                _format_float(row.get("diaguq_AUROC")),
                _format_float(row.get("diaguq_minus_best_fixed_layer_AUROC")),
                _format_float(row.get("diaguq_minus_uniform_mean_AUROC")),
            ]
            for row in summary_rows
        ]
        cal_headers = ["Dataset", "Best Fixed ECE", "Uniform ECE", "Concat ECE", "DiagUQ ECE", "Best Fixed Brier", "DiagUQ Brier"]
        cal_rows = [
            [
                row.get("dataset"),
                _format_float(row.get("best_fixed_layer_ECE")),
                _format_float(row.get("uniform_multilayer_mean_ECE")),
                _format_float(row.get("fixed_multilayer_concat_ECE")),
                _format_float(row.get("diaguq_ECE")),
                _format_float(row.get("best_fixed_layer_Brier")),
                _format_float(row.get("diaguq_Brier")),
            ]
            for row in summary_rows
        ]
        md = "# Main Layer Baseline Table\n\n" + _markdown_table(main_headers, main_rows) + "\n\n# Calibration Table\n\n" + _markdown_table(cal_headers, cal_rows) + "\n"
        tex = _latex_table(main_headers, main_rows) + "\n\n" + _latex_table(cal_headers, cal_rows) + "\n"
        md_path = cross / "paper_table_layer_baselines.md"
        tex_path = cross / "paper_table_layer_baselines.tex"
        md_path.write_text(md, encoding="utf-8")
        tex_path.write_text(tex, encoding="utf-8")
        written["layer_baselines_md"] = md_path
        written["layer_baselines_tex"] = tex_path
    if bootstrap_rows:
        rows = [row for row in bootstrap_rows if row.get("metric") == "AUROC"]
        headers = ["Dataset", "Baseline", "Δ AUROC", "95% CI", "significant?"]
        table_rows = [
            [
                row.get("dataset"),
                row.get("baseline_method"),
                _format_float(row.get("delta_mean")),
                f"[{_format_float(row.get('ci_low'))}, {_format_float(row.get('ci_high'))}]",
                "yes" if str(row.get("significant_positive")).lower() == "true" else "no",
            ]
            for row in rows
        ]
        path = cross / "paper_table_bootstrap_ci.md"
        path.write_text(_markdown_table(headers, table_rows) + "\n", encoding="utf-8")
        written["bootstrap_md"] = path
    if best_layer_rows:
        headers = ["Dataset", "Validation-selected layer", "Oracle layer", "Layer spread"]
        summary_by_dataset = {row.get("dataset"): row for row in summary_rows}
        table_rows = [
            [
                row.get("dataset"),
                (summary_by_dataset.get(row.get("dataset")) or {}).get("selected_best_fixed_layer", "NA"),
                (summary_by_dataset.get(row.get("dataset")) or {}).get("oracle_best_layer", "NA"),
                _format_float(row.get("layer_spread")),
            ]
            for row in best_layer_rows
        ]
        path = cross / "paper_table_best_layer_by_dataset.md"
        path.write_text(_markdown_table(headers, table_rows) + "\n", encoding="utf-8")
        written["best_layer_md"] = path
    return written