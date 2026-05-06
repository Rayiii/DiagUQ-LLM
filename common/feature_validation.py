"""Finite-value and shape validation for DiagUQ feature tensors."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch


def tensor_sanity(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    t = tensor.detach().to("cpu")
    finite = torch.isfinite(t)
    finite_count = int(finite.sum().item())
    nan_count = int(torch.isnan(t).sum().item())
    inf_count = int(torch.isinf(t).sum().item())
    out: dict[str, Any] = {
        "name": name,
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "numel": int(t.numel()),
        "finite_count": finite_count,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "finite_fraction": float(finite_count / max(int(t.numel()), 1)),
    }
    if finite_count:
        finite_values = t[finite].float()
        out.update(
            {
                "min": float(finite_values.min().item()),
                "max": float(finite_values.max().item()),
                "mean": float(finite_values.mean().item()),
            }
        )
    return out


def require_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    sanity = tensor_sanity(name, tensor)
    if sanity["nan_count"] or sanity["inf_count"]:
        raise ValueError(
            f"non-finite tensor {name}: nan={sanity['nan_count']} inf={sanity['inf_count']} "
            f"shape={sanity['shape']}"
        )


def _column_sanity(
    name: str,
    tensor: Optional[torch.Tensor],
    *,
    source_path: Optional[str] = None,
    column_names: Sequence[str] = (),
    required: bool = False,
) -> dict[str, Any]:
    if tensor is None:
        return {
            "name": name,
            "source_path": source_path,
            "required": required,
            "exists": False,
            "shape": None,
            "columns": [],
        }
    values = tensor.detach().to("cpu")
    if values.dim() == 1:
        values = values.unsqueeze(-1)
    columns: list[dict[str, Any]] = []
    for idx in range(values.shape[-1]):
        col = values[..., idx].reshape(-1)
        columns.append(
            {
                "column": column_names[idx] if idx < len(column_names) else str(idx),
                "nan_count": int(torch.isnan(col).sum().item()),
                "inf_count": int(torch.isinf(col).sum().item()),
                "finite_count": int(torch.isfinite(col).sum().item()),
                "required": required,
            }
        )
    finite_rows = torch.isfinite(values.float()).reshape(values.shape[0], -1).all(dim=1)
    return {
        "name": name,
        "source_path": source_path,
        "required": required,
        "exists": True,
        "shape": list(values.shape),
        "row_finite_count": int(finite_rows.sum().item()),
        "row_non_finite_count": int((~finite_rows).sum().item()),
        "columns": columns,
    }


def _empty_reason_rows(n: int) -> list[dict[str, Any]]:
    return [
        {
            "index": idx,
            "query_entropy_missing_reason": "missing_entropy_source",
            "query_prob_missing_reason": "missing_entropy_source",
        }
        for idx in range(n)
    ]


def _load_reason_rows(bank_dir: Path, n: int) -> list[dict[str, Any]]:
    path = bank_dir / "entropy_missing_reasons.json"
    rows = _empty_reason_rows(n)
    if not path.is_file():
        return rows
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return rows
    if not isinstance(loaded, list):
        return rows
    for idx, row in enumerate(loaded[:n]):
        if isinstance(row, Mapping):
            rows[idx].update(row)
            rows[idx]["index"] = idx
    return rows


def _bool_mask(value: Any, n: int) -> torch.Tensor:
    out = torch.zeros(n, dtype=torch.bool)
    if isinstance(value, torch.Tensor):
        mask = value.detach().to("cpu").bool().reshape(-1)
        limit = min(n, int(mask.shape[0]))
        out[:limit] = mask[:limit]
    return out


def _finite_row_mask(tensor: Optional[torch.Tensor], n: int) -> torch.Tensor:
    if tensor is None:
        return torch.zeros(n, dtype=torch.bool)
    values = tensor.detach().to("cpu").float()
    if values.dim() == 1:
        values = values.unsqueeze(-1)
    finite = torch.isfinite(values).reshape(values.shape[0], -1).all(dim=1)
    out = torch.zeros(n, dtype=torch.bool)
    limit = min(n, int(finite.shape[0]))
    out[:limit] = finite[:limit]
    return out


def _sanitize_or_create(
    tensor: Optional[torch.Tensor],
    *,
    n: int,
    width: int,
    explicit_mask: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tensor is None:
        return torch.zeros((n, width), dtype=torch.float32), torch.zeros(n, dtype=torch.bool)
    values = tensor.detach().to("cpu").float()
    if values.dim() == 1:
        values = values.unsqueeze(-1)
    if values.shape[0] != n:
        padded = torch.zeros((n, values.shape[-1]), dtype=torch.float32)
        limit = min(n, int(values.shape[0]))
        padded[:limit] = torch.where(
            torch.isfinite(values[:limit]), values[:limit], torch.zeros_like(values[:limit])
        )
        values = padded
    finite = torch.isfinite(values).reshape(values.shape[0], -1).all(dim=1)
    mask = _bool_mask(explicit_mask, n) if explicit_mask is not None else torch.ones(n, dtype=torch.bool)
    available = mask & finite[:n]
    sanitized = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
    sanitized = torch.where(available.view(-1, 1), sanitized, torch.zeros_like(sanitized))
    return sanitized.float(), available


def _reason_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        for key, value in row.items():
            if key.endswith("_missing_reason") and value:
                counter[str(value)] += 1
    return dict(counter)


def _sanitize_entropy_artifacts(
    bank_dir: Path,
    extras: dict[str, Any],
    *,
    n: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from features.hidden_state_ops import ENTROPY_STAT_NAMES, PROB_STAT_NAMES

    query_entropies = extras.get("query_entropies")
    query_probs = extras.get("query_probs")
    entropy_summary = _column_sanity(
        "query_entropies",
        query_entropies if isinstance(query_entropies, torch.Tensor) else None,
        source_path=str(bank_dir / "query_entropies.pt"),
        column_names=ENTROPY_STAT_NAMES,
    )
    prob_summary = _column_sanity(
        "query_probs",
        query_probs if isinstance(query_probs, torch.Tensor) else None,
        source_path=str(bank_dir / "query_probs.pt"),
        column_names=PROB_STAT_NAMES,
    )

    entropy_values, entropy_available = _sanitize_or_create(
        query_entropies if isinstance(query_entropies, torch.Tensor) else None,
        n=n,
        width=len(ENTROPY_STAT_NAMES),
        explicit_mask=extras.get("query_entropy_available"),
    )
    prob_values, prob_available = _sanitize_or_create(
        query_probs if isinstance(query_probs, torch.Tensor) else None,
        n=n,
        width=len(PROB_STAT_NAMES),
        explicit_mask=extras.get("query_prob_available"),
    )

    reason_rows = _load_reason_rows(bank_dir, n)
    raw_entropy_finite = _finite_row_mask(
        query_entropies if isinstance(query_entropies, torch.Tensor) else None, n
    )
    raw_prob_finite = _finite_row_mask(
        query_probs if isinstance(query_probs, torch.Tensor) else None, n
    )
    entropy_source_exists = isinstance(query_entropies, torch.Tensor)
    prob_source_exists = isinstance(query_probs, torch.Tensor)
    for idx, row in enumerate(reason_rows):
        if bool(entropy_available[idx]):
            row["query_entropy_missing_reason"] = None
        elif not entropy_source_exists:
            row["query_entropy_missing_reason"] = "missing_entropy_source"
        elif not bool(raw_entropy_finite[idx]):
            row["query_entropy_missing_reason"] = "non_finite_entropy_source"
        elif row.get("query_entropy_missing_reason") in {None, "not_processed"}:
            row["query_entropy_missing_reason"] = "entropy_unavailable"

        if bool(prob_available[idx]):
            row["query_prob_missing_reason"] = None
        elif not prob_source_exists:
            row["query_prob_missing_reason"] = "missing_probability_source"
        elif not bool(raw_prob_finite[idx]):
            row["query_prob_missing_reason"] = "non_finite_probability_source"
        elif row.get("query_prob_missing_reason") in {None, "not_processed"}:
            row["query_prob_missing_reason"] = "probability_unavailable"

    torch.save(entropy_values, bank_dir / "query_entropies.pt")
    torch.save(prob_values, bank_dir / "query_probs.pt")
    torch.save(entropy_available, bank_dir / "query_entropy_available.pt")
    torch.save(prob_available, bank_dir / "query_prob_available.pt")
    (bank_dir / "entropy_missing_reasons.json").write_text(
        json.dumps(reason_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    extras["query_entropies"] = entropy_values
    extras["query_probs"] = prob_values
    extras["query_entropy_available"] = entropy_available
    extras["query_prob_available"] = prob_available
    extras["entropy_missing_reasons"] = reason_rows
    report = {
        "required": False,
        "source_summaries_before_sanitization": [entropy_summary, prob_summary],
        "availability_rate": float(entropy_available.float().mean().item()) if n else 0.0,
        "probability_availability_rate": float(prob_available.float().mean().item()) if n else 0.0,
        "entropy_available_count": int(entropy_available.sum().item()),
        "probability_available_count": int(prob_available.sum().item()),
        "empty_answer_span_count": sum(
            1 for row in reason_rows if "empty_answer_span" in set(str(v) for v in row.values())
        ),
        "missing_entropy_source_count": sum(
            1 for row in reason_rows if row.get("query_entropy_missing_reason") == "missing_entropy_source"
        ),
        "reason_counts": _reason_counts(reason_rows),
        "sanitized_artifacts": {
            "query_entropies": str(bank_dir / "query_entropies.pt"),
            "query_probs": str(bank_dir / "query_probs.pt"),
            "query_entropy_available": str(bank_dir / "query_entropy_available.pt"),
            "query_prob_available": str(bank_dir / "query_prob_available.pt"),
            "entropy_missing_reasons": str(bank_dir / "entropy_missing_reasons.json"),
        },
    }
    return extras, report


def validate_view_bundle(
    views: Mapping[str, torch.Tensor],
    *,
    required_views: Sequence[str] = ("query", "answer", "relation"),
    entropy: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    failures: list[str] = []
    n_ref: Optional[int] = None

    for view_name in required_views:
        tensor = views.get(view_name)
        if tensor is None:
            failures.append(f"missing required view: {view_name}")
            continue
        summary = tensor_sanity(f"view::{view_name}", tensor)
        summaries.append(summary)
        if summary["nan_count"] or summary["inf_count"]:
            failures.append(
                f"view {view_name} has non-finite values: "
                f"nan={summary['nan_count']} inf={summary['inf_count']}"
            )
        if tensor.dim() != 3:
            failures.append(f"view {view_name} must be 3D (N,L,D), got {tuple(tensor.shape)}")
        if n_ref is None:
            n_ref = int(tensor.shape[0])
        elif int(tensor.shape[0]) != n_ref:
            failures.append(f"view {view_name} N mismatch: {tensor.shape[0]} != {n_ref}")

    if entropy is not None:
        summary = tensor_sanity("view::entropy", entropy)
        summaries.append(summary)
        if summary["nan_count"] or summary["inf_count"]:
            failures.append(
                f"entropy view has non-finite values: nan={summary['nan_count']} "
                f"inf={summary['inf_count']}"
            )
        if n_ref is not None and int(entropy.shape[0]) != n_ref:
            failures.append(f"entropy N mismatch: {entropy.shape[0]} != {n_ref}")

    return {
        "status": "success" if not failures else "failed",
        "num_samples": n_ref or 0,
        "tensors": summaries,
        "failures": failures,
    }


def write_sanity_reports(report: Mapping[str, Any], json_path: str | Path, csv_path: str | Path) -> None:
    json_target = Path(json_path)
    csv_target = Path(csv_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = list(report.get("tensors") or [])
    entropy_report = report.get("entropy", {})
    if isinstance(entropy_report, Mapping):
        for source in entropy_report.get("source_summaries_before_sanitization", []) or []:
            if not isinstance(source, Mapping):
                continue
            for column in source.get("columns", []) or []:
                if isinstance(column, Mapping):
                    rows.append(
                        {
                            "name": f"entropy::{source.get('name')}::{column.get('column')}",
                            "shape": source.get("shape"),
                            "source_path": source.get("source_path"),
                            "required": source.get("required"),
                            "nan_count": column.get("nan_count"),
                            "inf_count": column.get("inf_count"),
                            "finite_count": column.get("finite_count"),
                        }
                    )
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with csv_target.open("w", newline="", encoding="utf-8") as fw:
        writer = csv.DictWriter(fw, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def validate_hidden_bank_dir(
    bank_dir: str | Path,
    *,
    layer_list: Sequence[int],
    model_name: str,
    include_optional_views: bool = True,
    require_entropy: bool = False,
) -> dict[str, Any]:
    """Validate hidden-bank stage outputs without requiring training targets."""
    from features.build_multiview_features import (
        RELATION_OPS_DEFAULT,
        build_answer_view,
        build_entropy_view,
        build_query_view,
        build_relation_view,
    )
    from features.hidden_state_ops import ENTROPY_STAT_NAMES, PROB_STAT_NAMES
    from features.load_feature_tensors import load_multilayer_feature_bank

    bank_path = Path(bank_dir)
    bank = load_multilayer_feature_bank(
        "__unused__",
        model_name,
        layer_list=layer_list,
        bank_dir=str(bank_path),
    )
    query = build_query_view(bank)
    answer = build_answer_view(bank)
    relation = build_relation_view(query, answer, ops=RELATION_OPS_DEFAULT)
    views: dict[str, torch.Tensor] = {
        "query": query,
        "answer": answer,
        "relation": relation,
    }
    n_ref = int(query.shape[0])
    extras = bank.get("extras", {})
    if not isinstance(extras, dict):
        extras = dict(extras or {})
    extra_paths = bank.get("extra_paths", {})

    entropy_report: dict[str, Any]
    if require_entropy:
        entropy_sources = [
            _column_sanity(
                "query_entropies",
                extras.get("query_entropies") if isinstance(extras.get("query_entropies"), torch.Tensor) else None,
                source_path=str(bank_path / "query_entropies.pt"),
                column_names=ENTROPY_STAT_NAMES,
                required=True,
            ),
            _column_sanity(
                "query_probs",
                extras.get("query_probs") if isinstance(extras.get("query_probs"), torch.Tensor) else None,
                source_path=str(bank_path / "query_probs.pt"),
                column_names=PROB_STAT_NAMES,
                required=True,
            ),
        ]
        entropy_failures = []
        for source in entropy_sources:
            if not source.get("exists"):
                entropy_failures.append(f"required entropy source missing: {source['name']}")
            for column in source.get("columns", []):
                if column.get("nan_count") or column.get("inf_count"):
                    entropy_failures.append(
                        f"required {source['name']} column {column.get('column')} has "
                        f"nan={column.get('nan_count')} inf={column.get('inf_count')}"
                    )
        entropy_report = {
            "required": True,
            "source_summaries_before_sanitization": entropy_sources,
            "availability_rate": 0.0 if entropy_failures else 1.0,
            "probability_availability_rate": 0.0 if entropy_failures else 1.0,
            "entropy_available_count": 0 if entropy_failures else n_ref,
            "probability_available_count": 0 if entropy_failures else n_ref,
            "empty_answer_span_count": 0,
            "missing_entropy_source_count": n_ref if any(not s.get("exists") for s in entropy_sources) else 0,
            "reason_counts": {},
            "failures": entropy_failures,
        }
        if entropy_failures:
            report = {
                "status": "failed",
                "num_samples": n_ref,
                "tensors": [
                    tensor_sanity("view::query", query),
                    tensor_sanity("view::answer", answer),
                    tensor_sanity("view::relation", relation),
                ],
                "failures": entropy_failures,
                "entropy": entropy_report,
                "bank_dir": str(bank_path),
                "layer_list": list(layer_list),
                "include_optional_views": include_optional_views,
                "require_entropy": require_entropy,
            }
            write_sanity_reports(
                report,
                bank_path / "hidden_bank_sanity.json",
                bank_path / "hidden_bank_sanity.csv",
            )
            raise ValueError(f"hidden bank entropy validation failed: {entropy_failures}")
        if include_optional_views:
            views["entropy"] = build_entropy_view(
                extras,
                require_entropy=True,
                extra_paths=extra_paths,
            )
    else:
        extras, entropy_report = _sanitize_entropy_artifacts(bank_path, extras, n=n_ref)
        entropy_report["required"] = False
        if include_optional_views:
            views["entropy"] = build_entropy_view(
                extras,
                require_entropy=False,
                extra_paths={
                    **dict(extra_paths or {}),
                    "query_entropies": str(bank_path / "query_entropies.pt"),
                    "query_probs": str(bank_path / "query_probs.pt"),
                },
            )

    report = validate_view_bundle(views, entropy=views.get("entropy"))
    report.update(
        {
            "bank_dir": str(bank_path),
            "layer_list": list(layer_list),
            "include_optional_views": include_optional_views,
            "require_entropy": require_entropy,
            "entropy": entropy_report,
        }
    )
    write_sanity_reports(
        report,
        bank_path / "hidden_bank_sanity.json",
        bank_path / "hidden_bank_sanity.csv",
    )
    if report["status"] != "success":
        raise ValueError(f"hidden bank validation failed: {report['failures']}")
    return report