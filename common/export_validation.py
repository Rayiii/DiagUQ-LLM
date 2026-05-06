"""Validation helpers for prediction and analysis exports."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from common.diagnostic_target_specs import TARGET_STATUS_VALUES

TARGET_STATUS_ALLOWED_VALUES = frozenset(
    TARGET_STATUS_VALUES
)
TARGET_STATUS_SUFFIX = "_target_status"


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def column_coverage(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    n = len(rows)
    for column in columns:
        finite_count = sum(1 for row in rows if _is_finite(row.get(column)))
        missing_count = n - finite_count
        out[column] = {
            "finite_count": finite_count,
            "missing_count": missing_count,
            "coverage": float(finite_count / max(n, 1)),
        }
    return out


def is_target_status_column(column: str) -> bool:
    return column.startswith("dim_") and column.endswith(TARGET_STATUS_SUFFIX)


def categorical_column_coverage(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    *,
    allowed_values: Sequence[str] = tuple(TARGET_STATUS_ALLOWED_VALUES),
) -> dict[str, Any]:
    allowed = set(allowed_values)
    out: dict[str, Any] = {}
    n = len(rows)
    for column in columns:
        present = 0
        null_count = 0
        invalid_values: dict[str, int] = {}
        for row in rows:
            value = row.get(column)
            if value is None or str(value) == "":
                null_count += 1
                continue
            present += 1
            text = str(value)
            if text not in allowed:
                invalid_values[text] = invalid_values.get(text, 0) + 1
        out[column] = {
            "present_count": present,
            "null_count": null_count,
            "invalid_values": invalid_values,
            "coverage": float(present / max(n, 1)),
        }
    return out


def view_weight_row_sum_report(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    *,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    failures: list[str] = []
    bad_rows: list[dict[str, Any]] = []
    max_abs_error = 0.0
    for row_idx, row in enumerate(rows):
        values: list[float] = []
        missing: list[str] = []
        for column in columns:
            value = row.get(column)
            if not _is_finite(value):
                missing.append(column)
                continue
            values.append(float(value))
        if missing:
            failures.append(f"row {row_idx} has missing/non-finite view weights: {missing}")
            bad_rows.append({"row": row_idx, "missing": missing})
            continue
        row_sum = sum(values)
        error = abs(row_sum - 1.0)
        max_abs_error = max(max_abs_error, error)
        if error > tolerance:
            failures.append(
                f"row {row_idx} view-weight sum {row_sum:.8f} differs from 1 by {error:.8f}"
            )
            bad_rows.append({"row": row_idx, "sum": row_sum, "abs_error": error})
    return {
        "columns": list(columns),
        "tolerance": tolerance,
        "max_abs_error": max_abs_error,
        "bad_rows": bad_rows,
        "failures": failures,
    }


def build_export_sanity(
    rows: Sequence[Mapping[str, Any]],
    *,
    required_columns: Sequence[str],
    layer_weight_columns: Sequence[str] = (),
    categorical_columns: Sequence[str] = (),
    view_weight_columns: Sequence[str] = (),
    view_weight_groups: Mapping[str, Sequence[str]] | None = None,
    view_weight_tolerance: float = 1e-4,
) -> dict[str, Any]:
    inferred_categorical = [column for column in required_columns if is_target_status_column(column)]
    categorical = list(dict.fromkeys([*categorical_columns, *inferred_categorical]))
    numeric_required = [column for column in required_columns if column not in categorical]
    grouped_view_columns: dict[str, Sequence[str]] = {}
    if view_weight_columns:
        grouped_view_columns["legacy"] = list(view_weight_columns)
    if view_weight_groups:
        grouped_view_columns.update({name: list(columns) for name, columns in view_weight_groups.items()})
    all_view_columns: list[str] = []
    for columns in grouped_view_columns.values():
        all_view_columns.extend(columns)
    coverage = column_coverage(
        rows,
        list(numeric_required) + list(layer_weight_columns) + list(dict.fromkeys(all_view_columns)),
    )
    categorical_coverage = categorical_column_coverage(rows, categorical)
    failures: list[str] = []
    for column in numeric_required:
        if coverage[column]["finite_count"] == 0:
            failures.append(f"required column is 100% missing or non-finite: {column}")
    for column in categorical:
        stats = categorical_coverage[column]
        if stats["present_count"] == 0:
            failures.append(f"categorical column is missing or empty: {column}")
        if stats["null_count"]:
            failures.append(f"categorical column has null values: {column}")
        if stats["invalid_values"]:
            failures.append(
                f"categorical column has invalid values: {column}: {sorted(stats['invalid_values'].keys())}"
            )
    if layer_weight_columns and not any(coverage[col]["finite_count"] for col in layer_weight_columns):
        failures.append("all layer-weight columns are missing or non-finite")
    view_weight_report = None
    view_weight_reports: dict[str, Any] = {}
    for group_name, columns in grouped_view_columns.items():
        report = view_weight_row_sum_report(rows, columns, tolerance=view_weight_tolerance)
        view_weight_reports[group_name] = report
        if group_name == "legacy":
            view_weight_report = report
        failures.extend([f"{group_name}: {failure}" for failure in report["failures"]])
    return {
        "status": "success" if not failures else "failed",
        "num_rows": len(rows),
        "coverage": coverage,
        "categorical_coverage": categorical_coverage,
        "view_weight_row_sums": view_weight_report,
        "view_weight_row_sum_groups": view_weight_reports,
        "failures": failures,
    }


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as fr:
        return list(csv.DictReader(fr))


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")