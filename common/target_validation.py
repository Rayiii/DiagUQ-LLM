"""Sanity checks for DiagUQ diagnostic target tables."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from common.diagnostic_target_specs import (
    AVAILABLE_TARGET_STATUSES,
    METRIC_GROUP_VALUES,
    TARGET_STATUS_VALUES,
)


TARGET_COLUMNS = (
    "ambiguity_target",
    "knowledge_gap_target",
    "predictive_variability_target",
    "overall_target",
)
DIAGNOSTIC_TARGET_COLUMNS = TARGET_COLUMNS[:3]
TARGET_STATUS_ALLOWED_VALUES = frozenset(TARGET_STATUS_VALUES)
TARGET_METRIC_GROUP_ALLOWED_VALUES = frozenset(METRIC_GROUP_VALUES)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _summary(values: Sequence[Any]) -> dict[str, Any]:
    finite_values = [float(v) for v in values if _finite(v)]
    n = len(values)
    out: dict[str, Any] = {
        "count": n,
        "finite_count": len(finite_values),
        "missing_count": n - len(finite_values),
        "coverage": float(len(finite_values) / max(n, 1)),
    }
    if finite_values:
        out.update(
            {
                "min": min(finite_values),
                "max": max(finite_values),
                "mean": sum(finite_values) / len(finite_values),
                "out_of_unit_interval_count": sum(
                    1 for v in finite_values if v < -1e-6 or v > 1.0 + 1e-6
                ),
            }
        )
    else:
        out["out_of_unit_interval_count"] = 0
    return out


def _compact_example(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = [
        "sample_id",
        "index",
        "question_str",
        "raw_model_answer",
        "extracted_answer",
        "gold_answer",
        "gold_aliases",
        "correct",
        "ask4conf_confidence",
        "exact_match",
        "token_f1",
        "rouge_score",
        "bleu_score",
        "qa_parse_status",
        "qa_parse_error_reason",
        "knowledge_gap_parse_status",
        "knowledge_gap_parse_error_reason",
        "knowledge_gap_error_type",
        "dim_ambiguity_target_status",
        "dim_ambiguity_target_source",
        "dim_ambiguity_target_reliability",
        "dim_ambiguity_metric_group",
    ]
    return {key: row.get(key) for key in keys if key in row}


def _top_examples(
    rows: Sequence[Mapping[str, Any]],
    *,
    score_key: str,
    predicate,
    limit: int = 10,
) -> list[dict[str, Any]]:
    candidates = [row for row in rows if predicate(row) and _finite(row.get(score_key))]
    candidates.sort(key=lambda row: float(row.get(score_key)), reverse=True)
    return [_compact_example(row) for row in candidates[:limit]]


def _bottom_examples(
    rows: Sequence[Mapping[str, Any]],
    *,
    score_key: str,
    predicate,
    limit: int = 10,
) -> list[dict[str, Any]]:
    candidates = [row for row in rows if predicate(row) and _finite(row.get(score_key))]
    candidates.sort(key=lambda row: float(row.get(score_key)))
    return [_compact_example(row) for row in candidates[:limit]]


def _lexical_overlap(row: Mapping[str, Any]) -> float:
    values = [row.get("token_f1"), row.get("rouge_score"), row.get("bleu_score")]
    finite = [float(value) for value in values if _finite(value)]
    return max(finite) if finite else float("nan")


def build_target_sanity(
    rows: Sequence[Mapping[str, Any]],
    *,
    label_threshold: float = 0.5,
    fail_on_degenerate_correct: bool = False,
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    n = len(rows)
    columns = {
        name: _summary([row.get(name) for row in rows])
        for name in TARGET_COLUMNS
    }

    for required in ("knowledge_gap_target", "overall_target"):
        if columns[required]["finite_count"] == 0:
            failures.append(f"{required} has zero finite labels")
    for optional in ("ambiguity_target", "predictive_variability_target"):
        if columns[optional]["finite_count"] == 0:
            warnings.append(f"{optional} has zero finite labels; its loss will be skipped")
    for name, summary in columns.items():
        if summary.get("out_of_unit_interval_count", 0):
            failures.append(
                f"{name} has {summary['out_of_unit_interval_count']} values outside [0,1]"
            )

    correct_values = [row.get("correct") for row in rows if _finite(row.get("correct"))]
    correct_hist = {"0": 0, "1": 0}
    for value in correct_values:
        correct_hist[str(int(float(value) >= label_threshold))] += 1
    correct_total = correct_hist["0"] + correct_hist["1"]
    correct_positive_rate = float(correct_hist["1"] / correct_total) if correct_total else None
    if correct_values and (correct_hist["0"] == 0 or correct_hist["1"] == 0):
        msg = "correct labels contain one class only; ranking metrics will be skipped"
        if fail_on_degenerate_correct:
            failures.append(msg + "; pass --allow-degenerate-labels to override strict mode")
        else:
            warnings.append(msg)
    if not correct_values:
        warnings.append("correct labels are unavailable")

    lexical_rows = []
    for row in rows:
        overlap = _lexical_overlap(row)
        merged = dict(row)
        merged["_lexical_overlap"] = overlap
        lexical_rows.append(merged)

    audit_examples = {
        "high_ask4conf_confidence_but_correct_0": _top_examples(
            rows,
            score_key="ask4conf_confidence",
            predicate=lambda row: _finite(row.get("correct")) and float(row.get("correct")) < label_threshold,
        ),
        "high_lexical_overlap_but_correct_0": _top_examples(
            lexical_rows,
            score_key="_lexical_overlap",
            predicate=lambda row: _finite(row.get("correct")) and float(row.get("correct")) < label_threshold,
        ),
        "parse_failures": [
            _compact_example(row)
            for row in rows
            if row.get("qa_parse_status") not in (None, "ok")
            or row.get("qa_parse_error_reason")
            or row.get("knowledge_gap_parse_status") not in (None, "ok")
            or row.get("knowledge_gap_parse_error_reason")
        ][:10],
        "high_ambiguity_proxy": _top_examples(
            rows,
            score_key="ambiguity_target",
            predicate=lambda row: str(row.get("dim_ambiguity_target_status") or row.get("ambiguity_target_status") or "") == "proxy",
        ),
        "low_ambiguity_proxy": _bottom_examples(
            rows,
            score_key="ambiguity_target",
            predicate=lambda row: str(row.get("dim_ambiguity_target_status") or row.get("ambiguity_target_status") or "") == "proxy",
        ),
        "missing_or_unavailable_ambiguity": [
            _compact_example(row)
            for row in rows
            if str(row.get("dim_ambiguity_target_status") or row.get("ambiguity_target_status") or "") in {"missing", "unavailable", "masked"}
        ][:10],
        "dataset_grounded_ambiguity": [
            _compact_example(row)
            for row in rows
            if str(row.get("dim_ambiguity_target_status") or row.get("ambiguity_target_status") or "") in {"gold", "dataset_grounded"}
        ][:10],
    }
    metric_sources = sorted(
        {
            str(row.get("knowledge_gap_metric_source"))
            for row in rows
            if row.get("knowledge_gap_metric_source")
        }
    )
    target_status_coverage: dict[str, Any] = {}
    target_source_coverage: dict[str, Any] = {}
    target_reliability_summary: dict[str, Any] = {}
    target_metric_group_coverage: dict[str, Any] = {}
    for target in DIAGNOSTIC_TARGET_COLUMNS:
        dimension = target[: -len("_target")]
        status_col = f"dim_{dimension}_target_status"
        source_col = f"dim_{dimension}_target_source"
        reliability_col = f"dim_{dimension}_target_reliability"
        metric_group_col = f"dim_{dimension}_metric_group"
        legacy_status_col = f"{target}_status"
        legacy_source_col = f"{target}_source"
        statuses = [str(row.get(status_col) or "") for row in rows]
        if not any(statuses):
            statuses = [str(row.get(legacy_status_col) or "") for row in rows]
        sources = [str(row.get(source_col) or "") for row in rows]
        if not any(sources):
            sources = [str(row.get(legacy_source_col) or "") for row in rows]
        reliabilities = [row.get(reliability_col, row.get(f"{target}_reliability")) for row in rows]
        metric_groups = [str(row.get(metric_group_col) or row.get(f"{dimension}_metric_group") or "") for row in rows]
        invalid = sorted({value for value in statuses if value and value not in TARGET_STATUS_ALLOWED_VALUES})
        invalid_metric_groups = sorted({value for value in metric_groups if value and value not in TARGET_METRIC_GROUP_ALLOWED_VALUES})
        missing_status = sum(1 for value in statuses if not value)
        missing_source = sum(
            1
            for status, source in zip(statuses, sources)
            if status in AVAILABLE_TARGET_STATUSES and not source
        )
        missing_metric_group = sum(1 for status, group in zip(statuses, metric_groups) if status and not group)
        invalid_reliability = sum(
            1
            for value in reliabilities
            if _finite(value) and (float(value) < -1e-9 or float(value) > 1.0 + 1e-9)
        )
        missing_reliability = sum(
            1
            for status, value in zip(statuses, reliabilities)
            if status in AVAILABLE_TARGET_STATUSES and not _finite(value)
        )
        status_counts = {value: statuses.count(value) for value in sorted(set(statuses)) if value}
        target_status_coverage[target] = {
            "column": status_col,
            "counts": status_counts,
            "missing_count": missing_status,
            "invalid_values": invalid,
        }
        target_source_coverage[target] = {
            "column": source_col,
            "missing_for_available_count": missing_source,
            "unique_sources": sorted({value for value in sources if value}),
        }
        target_reliability_summary[target] = {
            "column": reliability_col,
            **_summary(reliabilities),
            "missing_for_available_count": missing_reliability,
            "invalid_count": invalid_reliability,
        }
        target_metric_group_coverage[target] = {
            "column": metric_group_col,
            "counts": {value: metric_groups.count(value) for value in sorted(set(metric_groups)) if value},
            "missing_count": missing_metric_group,
            "invalid_values": invalid_metric_groups,
        }
        if missing_status:
            failures.append(f"{status_col} has {missing_status} missing values")
        if invalid:
            failures.append(f"{status_col} has invalid values: {invalid}")
        if missing_source:
            failures.append(f"{source_col} is missing for {missing_source} available target rows")
        if missing_reliability:
            failures.append(f"{reliability_col} is missing for {missing_reliability} available target rows")
        if invalid_reliability:
            failures.append(f"{reliability_col} has {invalid_reliability} values outside [0,1]")
        if missing_metric_group:
            failures.append(f"{metric_group_col} has {missing_metric_group} missing values")
        if invalid_metric_groups:
            failures.append(f"{metric_group_col} has invalid values: {invalid_metric_groups}")

    return {
        "status": "success" if not failures else "failed",
        "num_rows": n,
        "columns": columns,
        "correct_label_source": ", ".join(metric_sources) or "correct column unavailable",
        "label_threshold": label_threshold,
        "correct_histogram": correct_hist,
        "correct_positive_rate": correct_positive_rate,
        "answer_metric_distributions": {
            "exact_match": _summary([row.get("exact_match") for row in rows]),
            "token_f1": _summary([row.get("token_f1") for row in rows]),
            "rouge_score": _summary([row.get("rouge_score") for row in rows]),
            "bleu_score": _summary([row.get("bleu_score") for row in rows]),
        },
        "target_status_coverage": target_status_coverage,
        "target_source_coverage": target_source_coverage,
        "target_reliability_summary": target_reliability_summary,
        "target_metric_group_coverage": target_metric_group_coverage,
        "overall_target_policy": rows[0].get("overall_target_policy") if rows else None,
        "overall_target_dimension_weights": rows[0].get("overall_target_dimension_weights") if rows else None,
        "answer_audit_examples": audit_examples,
        "failures": failures,
        "warnings": warnings,
    }


def write_target_reports(
    rows: Sequence[Mapping[str, Any]],
    sanity: Mapping[str, Any],
    *,
    target_dir: str | Path,
) -> dict[str, str]:
    out_dir = Path(target_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "target_sanity.json"
    csv_path = out_dir / "target_sanity.csv"
    json_path.write_text(json.dumps(sanity, indent=2, ensure_ascii=False), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as fw:
        fieldnames = [
            "target",
            "count",
            "finite_count",
            "missing_count",
            "coverage",
            "min",
            "max",
            "mean",
            "out_of_unit_interval_count",
        ]
        writer = csv.DictWriter(fw, fieldnames=fieldnames)
        writer.writeheader()
        for name, summary in (sanity.get("columns") or {}).items():
            writer.writerow({"target": name, **summary})

    targets_csv = out_dir / "targets.csv"
    if rows:
        field_order: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    field_order.append(key)
        with targets_csv.open("w", newline="", encoding="utf-8") as fw:
            writer = csv.DictWriter(fw, fieldnames=field_order)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    targets_json = out_dir / "targets.json"
    targets_json.write_text(json.dumps(list(rows), indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "target_sanity_json": str(json_path),
        "target_sanity_csv": str(csv_path),
        "targets_csv": str(targets_csv),
        "targets_json": str(targets_json),
    }