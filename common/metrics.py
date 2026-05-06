"""Metrics for diagnostic-dimension validity and homogeneity checks."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from common.diagnostic_target_specs import AVAILABLE_TARGET_STATUSES
from common.numpy_compat import safe_trapezoid


DEFAULT_DIAGNOSTIC_DIMENSIONS: Tuple[str, ...] = (
    "ambiguity",
    "knowledge_gap",
    "predictive_variability",
)


def _float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _json_float(value: float) -> Optional[float]:
    return float(value) if math.isfinite(float(value)) else None


def _column(rows: Sequence[Mapping[str, Any]], column: str) -> np.ndarray:
    return np.asarray([_float_or_nan(row.get(column)) for row in rows], dtype=np.float64)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    x = x[mask].astype(np.float64)
    y = y[mask].astype(np.float64)
    x = x - float(x.mean())
    y = y - float(y.mean())
    denom = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(x, y) / denom)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    idx = 0
    while idx < values.shape[0]:
        end = idx + 1
        while end < values.shape[0] and values[order[end]] == values[order[idx]]:
            end += 1
        avg_rank = 0.5 * (idx + end - 1) + 1.0
        ranks[order[idx:end]] = avg_rank
        idx = end
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    return pearson_corr(_rankdata(x[mask]), _rankdata(y[mask]))


def _auroc(score: np.ndarray, label: np.ndarray) -> float:
    mask = np.isfinite(score) & np.isfinite(label)
    score = score[mask]
    label = (label[mask] >= 0.5).astype(np.float64)
    if score.size == 0 or len(np.unique(label)) < 2:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    label = label[order]
    pos = float(label.sum())
    neg = float(label.size - pos)
    if pos <= 0 or neg <= 0:
        return float("nan")
    cum_tp = np.cumsum(label)
    cum_fp = np.cumsum(1.0 - label)
    tpr = np.concatenate([[0.0], cum_tp / pos, [1.0]])
    fpr = np.concatenate([[0.0], cum_fp / neg, [1.0]])
    return float(safe_trapezoid(tpr, fpr))


def _matrix(columns: Mapping[str, np.ndarray]) -> Dict[str, Dict[str, Optional[float]]]:
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for left, left_values in columns.items():
        out[left] = {}
        for right, right_values in columns.items():
            out[left][right] = _json_float(pearson_corr(left_values, right_values))
    return out


def _cross_matrix(
    rows: Mapping[str, np.ndarray],
    columns: Mapping[str, np.ndarray],
) -> Dict[str, Dict[str, Optional[float]]]:
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for row_name, row_values in rows.items():
        out[row_name] = {}
        for col_name, col_values in columns.items():
            out[row_name][col_name] = _json_float(pearson_corr(row_values, col_values))
    return out


def _residualize(values: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    baseline = baseline.astype(np.float64)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(values) & np.isfinite(baseline)
    if int(mask.sum()) < 3:
        return out
    x = baseline[mask]
    y = values[mask]
    x_centered = x - float(x.mean())
    denom = float(np.dot(x_centered, x_centered))
    if denom <= 1e-12:
        out[mask] = y - float(y.mean())
        return out
    slope = float(np.dot(x_centered, y - float(y.mean())) / denom)
    intercept = float(y.mean()) - slope * float(x.mean())
    out[mask] = y - (slope * x + intercept)
    return out


def _metric_row(
    *,
    dimension: str,
    target_quality: str,
    prediction: np.ndarray,
    target: np.ndarray,
    filter_values: Sequence[str] | None = None,
    filter_name: str | None = None,
    filter_value: str | None = None,
) -> Dict[str, Any]:
    mask = np.isfinite(prediction) & np.isfinite(target)
    if filter_values is not None and filter_value is not None:
        source_mask = np.asarray([str(value) == str(filter_value) for value in filter_values], dtype=bool)
        mask &= source_mask
    pred = prediction[mask]
    tgt = target[mask]
    mse = float(np.mean((pred - tgt) ** 2)) if pred.size else float("nan")
    row: Dict[str, Any] = {
        "dimension": dimension,
        "target_quality": target_quality,
        "n": int(mask.sum()),
        "pearson": _json_float(pearson_corr(pred, tgt)),
        "spearman": _json_float(spearman_corr(pred, tgt)),
        "mse": _json_float(mse),
        "brier": _json_float(mse),
        "AUROC": _json_float(_auroc(pred, tgt)),
    }
    if filter_name:
        row[filter_name] = filter_value
    return row


def _counts(values: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        if not value:
            continue
        out[str(value)] = out.get(str(value), 0) + 1
    return dict(sorted(out.items()))


def _reliability_summary(values: np.ndarray) -> Dict[str, Optional[float] | int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": _json_float(float(finite.mean())),
        "min": _json_float(float(finite.min())),
        "max": _json_float(float(finite.max())),
    }


def matrix_to_rows(
    matrix: Mapping[str, Mapping[str, Any]],
    *,
    matrix_name: str,
) -> List[Dict[str, Any]]:
    return [
        {"matrix": matrix_name, "row": str(row), "column": str(column), "value": value}
        for row, values in matrix.items()
        for column, value in values.items()
    ]


def build_dimension_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    *,
    dimension_names: Sequence[str] = DEFAULT_DIAGNOSTIC_DIMENSIONS,
    prediction_corr_warning: float = 0.95,
    diagonal_margin_warning: float = 0.02,
    residual_corr_warning: float = 0.95,
) -> Dict[str, Any]:
    """Build correlation diagnostics for diagnostic-head validity.

    The diagnostics intentionally warn rather than fail. Related uncertainty
    dimensions can be naturally correlated; this report quantifies when model
    predictions are more homogeneous than the targets justify.
    """
    dims = [name for name in dimension_names if rows and f"dim_{name}" in rows[0]]
    pred_cols = {f"dim_{name}": _column(rows, f"dim_{name}") for name in dims}
    if rows and "mduq_uncertainty" in rows[0]:
        pred_cols["mduq_uncertainty"] = _column(rows, "mduq_uncertainty")
    target_cols = {f"{name}_target": _column(rows, f"{name}_target") for name in dims if rows and f"{name}_target" in rows[0]}
    if rows and "overall_target" in rows[0]:
        target_cols["overall_target"] = _column(rows, "overall_target")

    pred_pred = _matrix(pred_cols)
    target_target = _matrix(target_cols)
    pred_target = _cross_matrix(pred_cols, target_cols)

    baseline = pred_cols.get("mduq_uncertainty")
    residual_cols: Dict[str, np.ndarray] = {}
    if baseline is not None:
        for name in dims:
            pred_name = f"dim_{name}"
            if pred_name in pred_cols:
                residual_cols[pred_name] = _residualize(pred_cols[pred_name], baseline)
    residual_corr = _matrix(residual_cols)

    diagonal: List[Dict[str, Any]] = []
    residual_utility: List[Dict[str, Any]] = []
    for name in dims:
        pred_name = f"dim_{name}"
        own_target = f"{name}_target"
        diag_score = pred_target.get(pred_name, {}).get(own_target)
        offdiag_values = []
        for other in dims:
            if other == name:
                continue
            value = pred_target.get(pred_name, {}).get(f"{other}_target")
            if value is not None:
                offdiag_values.append(float(value))
        best_offdiag = max(offdiag_values) if offdiag_values else float("nan")
        margin = float(diag_score) - best_offdiag if diag_score is not None and math.isfinite(best_offdiag) else float("nan")
        overall_corr = pred_target.get(pred_name, {}).get("overall_target")
        diagonal.append(
            {
                "dimension": name,
                "prediction": pred_name,
                "target": own_target,
                "diagonal_score": diag_score,
                "best_offdiag": _json_float(best_offdiag),
                "diagonal_margin": _json_float(margin),
                "overall_target_corr": overall_corr,
            }
        )
        if own_target in target_cols:
            shared = _column(rows, "shared_uncertainty_score") if rows and "shared_uncertainty_score" in rows[0] else None
            residual = _column(rows, f"residual_{name}_score") if rows and f"residual_{name}_score" in rows[0] else None
            final_corr = pred_target.get(pred_name, {}).get(own_target)
            shared_corr = pearson_corr(shared, target_cols[own_target]) if shared is not None else float("nan")
            residual_corr_value = pearson_corr(residual, target_cols[own_target]) if residual is not None else float("nan")
            gain = (
                abs(float(final_corr)) - abs(float(shared_corr))
                if final_corr is not None and math.isfinite(shared_corr)
                else float("nan")
            )
            residual_utility.append(
                {
                    "dimension": name,
                    "final_pearson": final_corr,
                    "shared_pearson": _json_float(shared_corr),
                    "residual_pearson": _json_float(residual_corr_value),
                    "final_abs_minus_shared_abs": _json_float(gain),
                }
            )

    warnings: List[str] = []
    dim_pred_names = [f"dim_{name}" for name in dims]
    for left_idx, left in enumerate(dim_pred_names):
        for right in dim_pred_names[left_idx + 1:]:
            value = pred_pred.get(left, {}).get(right)
            if value is not None and abs(float(value)) > prediction_corr_warning:
                warnings.append(f"prediction-prediction correlation {left} vs {right} is {float(value):.4f}")
            residual_value = residual_corr.get(left, {}).get(right)
            if residual_value is not None and abs(float(residual_value)) > residual_corr_warning:
                warnings.append(f"residualized prediction correlation {left} vs {right} is {float(residual_value):.4f}")
    for item in diagonal:
        margin = item.get("diagonal_margin")
        if margin is not None and float(margin) < diagonal_margin_warning:
            warnings.append(
                f"diagonal_margin for {item['dimension']} is {float(margin):.4f} < {diagonal_margin_warning:.4f}"
            )
    if diagonal and all(
        item.get("overall_target_corr") is not None
        and item.get("diagonal_score") is not None
        and abs(float(item["overall_target_corr"])) > abs(float(item["diagonal_score"]))
        for item in diagonal
    ):
        warnings.append("all diagnostic heads correlate more strongly with overall_target than with their own target")

    metric_rows: List[Dict[str, Any]] = []
    metric_rows_by_status: List[Dict[str, Any]] = []
    metric_rows_by_source: List[Dict[str, Any]] = []
    metric_rows_by_metric_group: List[Dict[str, Any]] = []
    target_source_summary: Dict[str, Any] = {}
    dataset_name = str(rows[0].get("dataset") or "") if rows else ""
    for name in dims:
        pred = pred_cols[f"dim_{name}"]
        target = target_cols.get(f"{name}_target")
        if target is None:
            continue
        status_col = f"dim_{name}_target_status"
        statuses = [str(row.get(status_col) or "") for row in rows]
        sources = [str(row.get(f"dim_{name}_target_source") or "") for row in rows]
        metric_groups = [str(row.get(f"dim_{name}_metric_group") or "") for row in rows]
        reliabilities = _column(rows, f"dim_{name}_target_reliability")
        metric_rows.append(
            _metric_row(
                dimension=name,
                target_quality="all",
                prediction=pred,
                target=target,
            )
        )
        for target_status in sorted(set(statuses)):
            if not target_status:
                continue
            row = _metric_row(
                dimension=name,
                target_quality=target_status,
                prediction=pred,
                target=target,
                filter_values=statuses,
                filter_name="target_status",
                filter_value=target_status,
            )
            metric_rows.append(row)
            metric_rows_by_status.append(row)
        for target_source in sorted(set(sources)):
            if not target_source:
                continue
            row = _metric_row(
                dimension=name,
                target_quality="source",
                prediction=pred,
                target=target,
                filter_values=sources,
                filter_name="target_source",
                filter_value=target_source,
            )
            metric_rows_by_source.append(row)
        for metric_group in sorted(set(metric_groups)):
            if not metric_group:
                continue
            row = _metric_row(
                dimension=name,
                target_quality="metric_group",
                prediction=pred,
                target=target,
                filter_values=metric_groups,
                filter_name="metric_group",
                filter_value=metric_group,
            )
            metric_rows_by_metric_group.append(row)
        available_statuses = {status for status in statuses if status in AVAILABLE_TARGET_STATUSES}
        target_source_summary[name] = {
            "status_counts": _counts(statuses),
            "source_counts": _counts(sources),
            "metric_group_counts": _counts(metric_groups),
            "reliability": _reliability_summary(reliabilities),
            "available_statuses": sorted(available_statuses),
        }
        if available_statuses == {"proxy"}:
            warnings.append(f"{name} has proxy-only target labels for this evaluation")
        if "main" in set(metric_groups) and not (available_statuses & {"gold", "dataset_grounded"}):
            warnings.append(f"{name} is assigned to metric_group=main but has no gold/dataset_grounded labels")
        if name == "ambiguity" and "proxy" in available_statuses and dataset_name and not dataset_name.lower().startswith("ambigqa"):
            warnings.append("ambiguity target is a proxy on a non-AmbigQA dataset")

    diagnostic_rows: List[Dict[str, Any]] = []
    diagnostic_rows.extend(matrix_to_rows(pred_pred, matrix_name="prediction_prediction"))
    diagnostic_rows.extend(matrix_to_rows(target_target, matrix_name="target_target"))
    diagnostic_rows.extend(matrix_to_rows(pred_target, matrix_name="prediction_target"))
    diagnostic_rows.extend(matrix_to_rows(residual_corr, matrix_name="residual_prediction"))
    for item in diagonal:
        diagnostic_rows.append({"matrix": "diagonal_dominance", **item})
    for item in residual_utility:
        diagnostic_rows.append({"matrix": "residual_utility", **item})

    max_pred_corr = None
    pred_pair_values = [
        abs(float(pred_pred[left][right]))
        for idx, left in enumerate(dim_pred_names)
        for right in dim_pred_names[idx + 1:]
        if pred_pred.get(left, {}).get(right) is not None
    ]
    if pred_pair_values:
        max_pred_corr = max(pred_pair_values)
    max_residual_corr = None
    residual_values = [
        abs(float(residual_corr[left][right]))
        for idx, left in enumerate(dim_pred_names)
        for right in dim_pred_names[idx + 1:]
        if residual_corr.get(left, {}).get(right) is not None
    ]
    if residual_values:
        max_residual_corr = max(residual_values)

    return {
        "status": "success",
        "num_rows": len(rows),
        "dimension_names": list(dims),
        "prediction_prediction_correlation": pred_pred,
        "target_target_correlation": target_target,
        "prediction_target_correlation": pred_target,
        "residual_prediction_correlation": residual_corr,
        "diagonal_dominance": diagonal,
        "residual_utility": residual_utility,
        "summary": {
            "max_prediction_prediction_abs_corr": _json_float(max_pred_corr if max_pred_corr is not None else float("nan")),
            "max_residual_prediction_abs_corr": _json_float(max_residual_corr if max_residual_corr is not None else float("nan")),
            "mean_diagonal_margin": _json_float(float(np.nanmean([_float_or_nan(item.get("diagonal_margin")) for item in diagonal])) if diagonal else float("nan")),
        },
        "warnings": warnings,
        "diagnostic_rows": diagnostic_rows,
        "metric_rows": metric_rows,
        "metric_rows_by_status": metric_rows_by_status,
        "metric_rows_by_source": metric_rows_by_source,
        "metric_rows_by_metric_group": metric_rows_by_metric_group,
        "target_source_summary": target_source_summary,
        "correlation_rows": [
            *matrix_to_rows(pred_pred, matrix_name="prediction_prediction"),
            *matrix_to_rows(target_target, matrix_name="target_target"),
            *matrix_to_rows(pred_target, matrix_name="prediction_target"),
            *matrix_to_rows(residual_corr, matrix_name="residual_prediction"),
        ],
    }
