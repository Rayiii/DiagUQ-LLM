"""Export per-sample DiagUQ outputs for analysis & case studies.

For one ``(dataset, model)`` pair this script writes::

    ./test_output/<dataset__split>/<model>/diaguq/analysis/
        per_sample.csv       -- one row per example (overall + dim + baselines)
        per_sample.json      -- richer JSON copy with question text + raw arrays
        layer_weights.pt     -- {view: tensor (N, L)} from the fusion module
        meta.json            -- run metadata (checkpoint path, layer list, ...)

The DiagUQ outputs come from ``best.pt`` when present, otherwise ``last.pt``.
Baselines are read straight from the dimension-targets payload +
hidden-bank extras under the resolved artifact root.

The internal symbols still use the historical ``MDUQ*`` names
(``export_mduq_outputs``, ``MDUQTrainConfig``, ``MDUQModel``); the
``DiagUQ``-suffixed aliases re-export them at the bottom of the module.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from pipeline.evaluate_diaguq import (
    LABEL_THRESHOLD,
    _build_dataset_from_artifact_root,
    _entropy_features_for_indices,
    _to_np,
)
from pipeline.diaguq_model import MDUQModel
from pipeline.train_diaguq import (
    MDUQTrainConfig,
    _collate_batch,
    _split_views,
)
from common.artifact_locator import locate_response_cache_artifacts
from common.artifact_paths import normalize_split_tag, split_dataset_and_raw
from common.diaguq_existing_artifacts import (
    ExistingDiagUQArtifactRoot,
    require_existing_diaguq_artifact_root,
    resolve_existing_diaguq_artifact_root,
)
from common.artifact_manifest import write_stage_manifest
from common.export_validation import build_export_sanity, read_csv_rows, write_json
from common.pair_context import (
    assert_diaguq_output_path,
    assert_no_duplicate_output_dirs,
    resolve_pair_context,
    INVALID_CHECKPOINT_VARIANT_MESSAGE,
    resolve_checkpoint_context_for_eval,
)
from common.sample_alignment import require_matching_sample_ids


TARGET_STATUS_COLUMNS = (
    "dim_ambiguity_target_status",
    "dim_knowledge_gap_target_status",
    "dim_predictive_variability_target_status",
)
TARGET_SOURCE_COLUMNS = (
    "dim_ambiguity_target_source",
    "dim_knowledge_gap_target_source",
    "dim_predictive_variability_target_source",
)
VIEW_WEIGHT_COLUMNS = (
    "view_weight_query",
    "view_weight_answer",
    "view_weight_relation",
)
VIEW_NAMES = ("query", "answer", "relation")
SAMPLE_ID_FALLBACK_WARNING = "sample_id missing from upstream artifacts; falling back to row index"


def _is_missing_sample_id(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == "" or value.strip().lower() in {"none", "nan"}
    return False


def _sample_id_list(values: Any, n: int) -> Optional[List[Any]]:
    if not isinstance(values, (list, tuple)):
        return None
    out = list(values[:n])
    if len(out) != n or any(_is_missing_sample_id(value) for value in out):
        return None
    return out


def _source_report(source: str, values: Any, n: int) -> Dict[str, Any]:
    if not isinstance(values, (list, tuple)):
        return {"source": source, "available": False, "count": 0, "missing_count": None, "examples": []}
    materialized = list(values)
    return {
        "source": source,
        "available": len(materialized) >= n and not any(_is_missing_sample_id(value) for value in materialized[:n]),
        "count": len(materialized),
        "missing_count": sum(1 for value in materialized[:n] if _is_missing_sample_id(value)),
        "examples": [_safe(value) for value in materialized[:5]],
    }


def _target_sample_ids_from_payload(payload: Mapping[str, Any], n: int) -> Optional[List[Any]]:
    direct = _sample_id_list(payload.get("sample_ids"), n)
    if direct is not None:
        return direct
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    if isinstance(metadata, Mapping):
        return _sample_id_list(metadata.get("sample_ids"), n)
    return None


def _target_sample_id_values_for_report(payload: Mapping[str, Any]) -> Any:
    if isinstance(payload.get("sample_ids"), (list, tuple)):
        return payload.get("sample_ids")
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    if isinstance(metadata, Mapping):
        return metadata.get("sample_ids")
    return None


def _hidden_bank_manifest_sample_ids(hidden_bank_dir: Path, n: int) -> Optional[List[Any]]:
    for name in ("manifest.json", "hidden_bank_manifest.json"):
        path = Path(hidden_bank_dir) / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        candidates = [payload.get("sample_ids")]
        sanity = payload.get("sanity") if isinstance(payload.get("sanity"), Mapping) else {}
        if isinstance(sanity, Mapping):
            candidates.append(sanity.get("sample_ids"))
        for values in candidates:
            sample_ids = _sample_id_list(values, n)
            if sample_ids is not None:
                return sample_ids
    return None


def _hidden_bank_manifest_values_for_report(hidden_bank_dir: Path) -> Any:
    for name in ("manifest.json", "hidden_bank_manifest.json"):
        path = Path(hidden_bank_dir) / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload.get("sample_ids"), (list, tuple)):
            return payload.get("sample_ids")
        sanity = payload.get("sanity") if isinstance(payload.get("sanity"), Mapping) else {}
        if isinstance(sanity, Mapping) and isinstance(sanity.get("sample_ids"), (list, tuple)):
            return sanity.get("sample_ids")
    return None


def _prediction_row_sample_ids(rows: Sequence[Mapping[str, Any]], n: int) -> Optional[List[Any]]:
    if not rows or not any("sample_id" in row for row in rows):
        return None
    return _sample_id_list([row.get("sample_id") for row in rows], n)


def _resolve_export_sample_ids(
    *,
    rows: Sequence[Mapping[str, Any]],
    target_payload: Mapping[str, Any],
    hidden_bank_dir: Path,
) -> tuple[List[Any], Dict[str, Any]]:
    n = len(rows)
    target_ids = _target_sample_ids_from_payload(target_payload, n)
    hidden_ids = _hidden_bank_manifest_sample_ids(hidden_bank_dir, n)
    prediction_ids = _prediction_row_sample_ids(rows, n)
    target_values = target_ids if target_ids is not None else _target_sample_id_values_for_report(target_payload)
    hidden_values = hidden_ids if hidden_ids is not None else _hidden_bank_manifest_values_for_report(hidden_bank_dir)
    prediction_values = prediction_ids if prediction_ids is not None else [row.get("sample_id") for row in rows] if any("sample_id" in row for row in rows) else None
    sources = [
        ("dimension_targets.sample_ids", target_ids, target_values),
        ("hidden_bank.manifest.sample_ids", hidden_ids, hidden_values),
        ("eval.predictions.sample_id", prediction_ids, prediction_values),
    ]
    chosen_source = None
    chosen_ids: Optional[List[Any]] = None
    source_reports = [_source_report(label, raw_values, n) for label, _, raw_values in sources]
    warnings: List[str] = []
    for label, sample_ids, _ in sources:
        if sample_ids is not None:
            chosen_source = label
            chosen_ids = sample_ids
            break
    used_fallback = chosen_ids is None
    if chosen_ids is None:
        chosen_source = "row_index_fallback"
        chosen_ids = list(range(n))
        warnings.append(SAMPLE_ID_FALLBACK_WARNING)
    for label, sample_ids, _ in sources:
        if sample_ids is None or label == chosen_source:
            continue
        require_matching_sample_ids(
            chosen_ids,
            sample_ids,
            expected_label=str(chosen_source),
            actual_label=label,
        )
    sanity = {
        "n_rows": n,
        "chosen_source": chosen_source,
        "used_fallback_index_as_sample_id": used_fallback,
        "sources": source_reports,
        "sample_id_examples": [_safe(value) for value in chosen_ids[:5]],
        "warnings": warnings,
    }
    return chosen_ids, sanity


def _view_weight_groups_from_rows(
    rows: Sequence[Mapping[str, Any]],
    dimension_names: Sequence[str],
) -> Dict[str, Sequence[str]]:
    if not rows:
        return {}
    columns = set(rows[0].keys())
    groups: Dict[str, Sequence[str]] = {}
    for target in ["overall", *dimension_names]:
        group = [f"view_weight_{target}_{view_name}" for view_name in VIEW_NAMES]
        if all(column in columns for column in group):
            groups[target] = group
    return groups


def _float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _categorical_counts(rows: Sequence[Mapping[str, Any]], column: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = str(row.get(column) or "")
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _numeric_summary(rows: Sequence[Mapping[str, Any]], column: str) -> Dict[str, Any]:
    values = np.asarray([_float_or_nan(row.get(column)) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _target_semantic_summary(rows: Sequence[Mapping[str, Any]], dimension_names: Sequence[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for name in dimension_names:
        summary[name] = {
            "target_status_distribution": _categorical_counts(rows, f"dim_{name}_target_status"),
            "target_source_distribution": _categorical_counts(rows, f"dim_{name}_target_source"),
            "metric_group_distribution": _categorical_counts(rows, f"dim_{name}_metric_group"),
            "target_reliability": _numeric_summary(rows, f"dim_{name}_target_reliability"),
        }
    return summary


def analysis_dir(
    dataset_name: str,
    model_name: str,
    output_root: Optional[str] = None,
    *,
    artifact_root: Optional[Path] = None,
    artifact_root_name: Optional[str] = None,
) -> Path:
    """DiagUQ per-sample analysis directory for one (dataset, model) pair."""
    if artifact_root is not None:
        return Path(artifact_root) / "analysis"
    return resolve_pair_context(dataset_name, model_name, runtime_root=output_root).analysis_dir


# ---------------------------------------------------------------------------
# Inference over the full bank
# ---------------------------------------------------------------------------


def _coerce_artifact_root(
    cfg: MDUQTrainConfig,
    artifact_root: Optional[Path],
) -> Path:
    if artifact_root is not None:
        return Path(artifact_root)
    resolved = require_existing_diaguq_artifact_root(
        cfg.dataset_name, cfg.model_name, cfg.output_root
    )
    return Path(resolved.artifact_root)  # type: ignore[arg-type]


def _checkpoint_path_from_root(artifact_root: Path) -> Path:
    ckpt_dir = Path(artifact_root) / "checkpoints"
    best = ckpt_dir / "best.pt"
    last = ckpt_dir / "last.pt"
    target = best if best.is_file() else last
    if not target.is_file():
        raise FileNotFoundError(
            "no DiagUQ checkpoint found; checked paths: "
            f"{best}; {last}"
        )
    _log_export_info("[export-analysis] loading checkpoint file: {}", target)
    return target


def _dimension_targets_pt_from_root(artifact_root: Path) -> Path:
    target_pt = Path(artifact_root) / "dimension_targets" / "dimension_targets.pt"
    _log_export_info("[export-analysis] loading dimension target file: {}", target_pt)
    if not target_pt.is_file():
        raise FileNotFoundError(f"dimension_targets.pt missing: {target_pt}")
    return target_pt


def _load_checkpoint(
    cfg: MDUQTrainConfig,
    artifact_root: Optional[Path] = None,
) -> Dict[str, Any]:
    resolved_root = _coerce_artifact_root(cfg, artifact_root)
    target = _checkpoint_path_from_root(resolved_root)
    return {
        "path": str(target),
        "payload": torch.load(target, map_location="cpu", weights_only=False),
    }


def _forward_full_dataset(
    cfg: MDUQTrainConfig,
    dataset,
    view_dims: Mapping[str, int],
    *,
    state_dict: Mapping[str, torch.Tensor],
    entropy_dim: int,
    dimension_names: Sequence[str],
) -> Dict[str, np.ndarray]:
    device = torch.device(cfg.device)
    model = MDUQModel(
        view_dims=view_dims,
        dimension_names=dimension_names,
        fusion_dim=cfg.fusion_dim,
        fusion_hidden_dim=cfg.fusion_hidden_dim,
        head_hidden_dim=cfg.head_hidden_dim,
        overall_hidden_dim=cfg.overall_hidden_dim,
        dropout=cfg.dropout,
        entropy_dim=entropy_dim,
        layer_softmax_temperature=cfg.layer_softmax_temperature,
        layer_dropout=cfg.layer_dropout,
        gate_logit_clip=cfg.gate_logit_clip,
        view_gate_hidden_dim=cfg.view_gate_hidden_dim,
        view_temperature=cfg.view_temperature,
        view_temperature_min=cfg.view_temperature_min,
        view_temperature_max=cfg.view_temperature_max,
        residual_uniform_alpha=cfg.residual_uniform_alpha,
        view_norm_clip=cfg.view_norm_clip,
        view_dropout_prob=0.0,
        view_gate_scope=cfg.view_gate_scope,
        view_fusion_mode=cfg.view_fusion_mode,
        diagnostic_factorization_mode=cfg.diagnostic_factorization_mode,
        overall_aggregation_mode=cfg.overall_aggregation_mode,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=_collate_batch,
    )

    confidences: List[torch.Tensor] = []
    uncertainties: List[torch.Tensor] = []
    dim_scores: List[torch.Tensor] = []
    layer_weights_per_view: Dict[str, List[torch.Tensor]] = {}
    view_weights_seen: List[torch.Tensor] = []
    gate_logits_seen: Dict[str, List[torch.Tensor]] = {}

    with torch.no_grad():
        for batch in loader:
            views, rest = _split_views(batch)
            views = {k: v.to(device) for k, v in views.items()}
            entropy = rest.get("entropy")
            if entropy is not None:
                entropy = entropy.to(device)
            out = model(views, entropy=entropy)
            confidences.append(out.confidence.detach().cpu())
            uncertainties.append(out.uncertainty.detach().cpu())
            dim_scores.append(out.dimension_scores.detach().cpu())
            for k, w in out.layer_weights.items():
                if k == "_view_weights":
                    view_weights_seen.append(w.detach().cpu())
                    continue
                layer_weights_per_view.setdefault(k, []).append(w.detach().cpu())
            for k, logits in out.gate_logits.items():
                gate_logits_seen.setdefault(k, []).append(logits.detach().cpu())

    payload: Dict[str, np.ndarray] = {
        "confidence": _to_np(torch.cat(confidences, dim=0)),
        "uncertainty": _to_np(torch.cat(uncertainties, dim=0)),
        "dimension_scores": _to_np(torch.cat(dim_scores, dim=0)),
    }
    payload["layer_weights"] = {  # type: ignore[assignment]
        k: torch.cat(v, dim=0) for k, v in layer_weights_per_view.items()
    }
    if view_weights_seen:
        payload["view_weights"] = _to_np(torch.cat(view_weights_seen, dim=0))  # type: ignore[assignment]
    if gate_logits_seen:
        payload["gate_logits"] = {k: torch.cat(v, dim=0) for k, v in gate_logits_seen.items()}  # type: ignore[assignment]
    return payload


# ---------------------------------------------------------------------------
# Baselines + question-text loading
# ---------------------------------------------------------------------------


def _load_dim_targets(
    cfg: MDUQTrainConfig,
    artifact_root: Optional[Path] = None,
) -> Dict[str, torch.Tensor]:
    resolved_root = _coerce_artifact_root(cfg, artifact_root)
    target_pt = _dimension_targets_pt_from_root(resolved_root)
    return torch.load(target_pt, map_location="cpu")


def _load_questions(cfg: MDUQTrainConfig, n: int) -> List[Optional[str]]:
    artifacts = locate_response_cache_artifacts(
        cfg.dataset_name, cfg.model_name, cfg.output_root
    )
    try:
        extend_path = artifacts.require("mextend")
    except FileNotFoundError:
        return [None] * n
    try:
        with open(extend_path, "r", encoding="utf-8") as fr:
            data = json.load(fr)
    except Exception:
        return [None] * n
    out: List[Optional[str]] = []
    for i in range(n):
        if i < len(data) and isinstance(data[i], dict):
            out.append(data[i].get("question_str"))
        else:
            out.append(None)
    return out


def _safe(x: Any) -> Any:
    """JSON / CSV-safe scalar coercion (NaN -> None)."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, np.floating):
        v = float(x)
        return None if not math.isfinite(v) else v
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.ndarray):
        return [_safe(v) for v in x.tolist()]
    return x


def _numeric_matrix_from_rows(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> Optional[np.ndarray]:
    if not rows or not all(column in rows[0] for column in columns):
        return None
    matrix = np.asarray(
        [[_float_or_nan(row.get(column)) for column in columns] for row in rows],
        dtype=np.float64,
    )
    return matrix if np.isfinite(matrix).all() else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_mduq_outputs(
    cfg: MDUQTrainConfig,
    *,
    artifact_root: Optional[Path] = None,
    checkpoint_artifact_root: Optional[Path] = None,
    evaluation_context: Optional[Mapping[str, Any]] = None,
    calibrate_confidence: bool = False,
    run_inference: bool = False,
) -> Dict[str, Any]:
    """Export per-sample files from validated ``eval/predictions.csv``."""
    resolved_root = _coerce_artifact_root(cfg, artifact_root)
    ctx = resolve_pair_context(cfg.dataset_name, cfg.model_name, runtime_root=cfg.output_root)
    assert_diaguq_output_path(ctx, resolved_root, stage_token=None)
    out_dir = analysis_dir(
        cfg.dataset_name,
        cfg.model_name,
        cfg.output_root,
        artifact_root=resolved_root,
    )
    assert_diaguq_output_path(ctx, out_dir, stage_token="analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_root = resolved_root / "eval"
    predictions_csv = eval_root / "predictions.csv"
    predictions_json = eval_root / "predictions.json"
    eval_layer_weights = eval_root / "layer_weights.pt"
    if not predictions_csv.is_file() and run_inference:
        from pipeline.evaluate_diaguq import run_eval

        _log_export_info(
            "[export-analysis] predictions missing; running evaluation inference first: {}",
            predictions_csv,
        )
        run_eval(
            "main",
            cfg,
            artifact_root=resolved_root,
            checkpoint_artifact_root=checkpoint_artifact_root,
            evaluation_context=evaluation_context,
            calibrate_confidence=calibrate_confidence,
        )
    if not predictions_csv.is_file():
        raise FileNotFoundError(
            f"required evaluation predictions missing: {predictions_csv}. "
            "Run `python run.py evaluate-diaguq` first, or pass --run-inference."
        )
    if not eval_layer_weights.is_file():
        raise FileNotFoundError(
            f"required evaluation layer weights missing: {eval_layer_weights}. "
            "Run `python run.py evaluate-diaguq` first."
        )

    rows_csv = read_csv_rows(predictions_csv)
    n = len(rows_csv)

    target_pt_path = resolved_root / "dimension_targets" / "dimension_targets.pt"
    target_payload_metadata: Dict[str, Any] = {}
    target_sample_ids: Optional[Sequence[Any]] = None
    target_payload_for_sample_ids: Mapping[str, Any] = {}
    if target_pt_path.is_file():
        try:
            target_payload = torch.load(target_pt_path, map_location="cpu", weights_only=False)
            if isinstance(target_payload, Mapping):
                target_payload_for_sample_ids = target_payload
                metadata = target_payload.get("metadata")
                if isinstance(metadata, Mapping):
                    target_payload_metadata = dict(metadata)
                    target_sample_ids = metadata.get("sample_ids") if isinstance(metadata.get("sample_ids"), list) else None
                if target_sample_ids is None and isinstance(target_payload.get("sample_ids"), list):
                    target_sample_ids = target_payload.get("sample_ids")
                for key in ("dataset_diagnostic_spec", "overall_target_policy", "overall_target_dimension_weights", "task_type"):
                    if key in target_payload and key not in target_payload_metadata:
                        target_payload_metadata[key] = target_payload[key]
        except Exception:
            target_sample_ids = None
            export_target_metadata_warning = f"failed to read target metadata: {target_pt_path}"
        else:
            export_target_metadata_warning = None
    else:
        export_target_metadata_warning = None
    sample_ids, sample_id_alignment_sanity = _resolve_export_sample_ids(
        rows=rows_csv,
        target_payload=target_payload_for_sample_ids,
        hidden_bank_dir=resolved_root / "hidden_bank",
    )
    sample_id_alignment_path = out_dir / "sample_id_alignment_sanity.json"
    write_json(sample_id_alignment_path, sample_id_alignment_sanity)
    if sample_id_alignment_sanity.get("used_fallback_index_as_sample_id"):
        _log_export_warning("[export-analysis] {}", SAMPLE_ID_FALLBACK_WARNING)
    split_meta = ctx.split_metadata
    for idx, row in enumerate(rows_csv):
        row["sample_id"] = sample_ids[idx]
        row["source_sample_id"] = row.get("source_sample_id") or row.get("sample_id")
        row["dataset"] = row.get("dataset") or ctx.resolved_variant
        row["resolved_variant"] = row.get("resolved_variant") or ctx.resolved_variant
        row["split"] = row.get("split") or target_payload_metadata.get("split") or ctx.split
        row["model"] = row.get("model") or ctx.model
        if split_meta.get("virtual_split") is not None:
            row["virtual_split"] = row.get("virtual_split") or split_meta.get("virtual_split")
    rows_json: List[Dict[str, Any]] = [dict(row) for row in rows_csv]
    if target_sample_ids is not None:
        require_matching_sample_ids(
            target_sample_ids,
            [row.get("sample_id") for row in rows_csv],
            expected_label="dimension_targets.sample_ids",
            actual_label="eval.predictions.sample_ids",
        )
        require_matching_sample_ids(
            [row.get("sample_id") for row in rows_csv],
            [row.get("sample_id") for row in rows_json],
            expected_label="eval.predictions.sample_ids",
            actual_label="analysis.per_sample.sample_ids",
        )

    dim_score_names = {f"dim_{name}" for name in cfg.dimension_names}
    dim_columns = [
        key for key in (rows_csv[0].keys() if rows_csv else [])
        if key in dim_score_names
    ]
    if not dim_columns:
        dim_columns = [f"dim_{name}" for name in cfg.dimension_names]
    layer_columns = [key for key in (rows_csv[0].keys() if rows_csv else []) if key.startswith("layer_weight_")]
    target_status_columns = list(TARGET_STATUS_COLUMNS)
    view_weight_columns = list(VIEW_WEIGHT_COLUMNS)
    view_weight_groups = _view_weight_groups_from_rows(rows_csv, cfg.dimension_names)
    required_columns = ["mduq_uncertainty", "mduq_confidence", *dim_columns]
    export_sanity = build_export_sanity(
        rows_csv,
        required_columns=required_columns,
        layer_weight_columns=layer_columns,
        categorical_columns=target_status_columns,
        view_weight_columns=view_weight_columns,
        view_weight_groups=view_weight_groups,
    )
    export_warnings = export_sanity.setdefault("warnings", [])
    export_sanity["sample_id_alignment"] = sample_id_alignment_sanity
    export_sanity["sample_id_alignment_sanity_path"] = str(sample_id_alignment_path)
    export_warnings.extend(str(warning) for warning in sample_id_alignment_sanity.get("warnings", []))
    target_semantic_summary = _target_semantic_summary(rows_csv, cfg.dimension_names)
    export_sanity["target_semantic_summary"] = target_semantic_summary
    export_sanity["target_status_distribution"] = {
        name: info.get("target_status_distribution", {})
        for name, info in target_semantic_summary.items()
    }
    export_sanity["target_source_distribution"] = {
        name: info.get("target_source_distribution", {})
        for name, info in target_semantic_summary.items()
    }
    export_sanity["target_reliability_summary"] = {
        name: info.get("target_reliability", {})
        for name, info in target_semantic_summary.items()
    }
    for name, info in target_semantic_summary.items():
        status_counts = info.get("target_status_distribution", {}) if isinstance(info, Mapping) else {}
        available_statuses = {status for status in status_counts if status in {"gold", "dataset_grounded", "proxy"}}
        if available_statuses == {"proxy"}:
            export_warnings.append(f"dim_{name} has proxy-only target labels in this export")
    for column in ["mduq_uncertainty", "mduq_confidence", *dim_columns]:
        values = np.asarray([_float_or_nan(row.get(column)) for row in rows_csv], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size and float(finite.max() - finite.min()) < 0.05:
            export_warnings.append(
                f"{column} prediction range is narrow: {float(finite.min()):.4f}..{float(finite.max()):.4f}"
            )
    for dim_col in dim_columns:
        dim_name = dim_col[len("dim_"):]
        status_col = f"{dim_col}_target_status"
        available_col = f"{dim_name}_available"
        if rows_csv and all(
            str(row.get(status_col) or "") in {"unavailable", "missing", "masked"}
            or str(row.get(available_col)).lower() == "false"
            for row in rows_csv
        ):
            export_warnings.append(f"{dim_col} exists but target is unavailable for this dataset")
    eval_context_path = eval_root / "eval_context.json"
    if eval_context_path.is_file():
        try:
            eval_context = json.loads(eval_context_path.read_text(encoding="utf-8"))
            export_sanity["evaluation_context"] = eval_context
            if not bool(eval_context.get("held_out_evaluation", True)):
                export_warnings.append("This is an in-split run, not a held-out evaluation.")
        except Exception:
            pass

    layer_payload = torch.load(eval_layer_weights, map_location="cpu", weights_only=False)
    layer_failures: List[str] = []
    layer_warnings: List[str] = []
    fusion_diagnostics: Dict[str, Any] = {"layer_weights": {}, "view_weights": {}}
    prediction_view_matrix = _numeric_matrix_from_rows(rows_csv, VIEW_WEIGHT_COLUMNS)
    if prediction_view_matrix is not None:
        row_sums = prediction_view_matrix.sum(axis=-1)
        max_row_sum_error = float(np.max(np.abs(row_sums - 1.0))) if row_sums.size else 0.0
        fusion_diagnostics["view_weights"]["prediction_csv"] = {
            "shape": list(prediction_view_matrix.shape),
            "max_row_sum_error": max_row_sum_error,
            "std_by_view": [float(v) for v in prediction_view_matrix.std(axis=0).tolist()],
        }
        existing_view_weights = layer_payload.get("view_weights")
        if isinstance(existing_view_weights, torch.Tensor):
            existing = existing_view_weights.detach().cpu().float()
            if existing.dim() == 2 and list(existing.shape) == list(prediction_view_matrix.shape):
                existing_np = existing.numpy()
                if not np.allclose(existing_np, prediction_view_matrix, atol=1e-6, rtol=1e-6):
                    layer_warnings.append(
                        "eval layer_weights.pt view_weights differ from predictions.csv; "
                        "analysis layer_weights.pt uses predictions.csv as source of truth"
                    )
            elif existing.dim() == 1:
                layer_warnings.append(
                    "eval layer_weights.pt view_weights is a static [num_views] vector; "
                    "analysis layer_weights.pt uses predictions.csv per-sample weights"
                )
            else:
                layer_warnings.append(
                    "eval layer_weights.pt view_weights has unexpected shape; "
                    "analysis layer_weights.pt uses predictions.csv per-sample weights"
                )
        layer_payload["view_weights"] = torch.as_tensor(prediction_view_matrix, dtype=torch.float32)
        layer_payload["view_names"] = [column[len("view_weight_"):] for column in VIEW_WEIGHT_COLUMNS]
    target_view_payload = dict(layer_payload.get("view_weights_by_target") or {})
    for target, columns in view_weight_groups.items():
        matrix = _numeric_matrix_from_rows(rows_csv, columns)
        if matrix is None:
            continue
        row_sums = matrix.sum(axis=-1)
        max_row_sum_error = float(np.max(np.abs(row_sums - 1.0))) if row_sums.size else 0.0
        weights_tensor = torch.as_tensor(matrix, dtype=torch.float32)
        fusion_diagnostics["view_weights"][target] = {
            "shape": list(matrix.shape),
            "max_row_sum_error": max_row_sum_error,
            "mean_by_view": [float(v) for v in matrix.mean(axis=0).tolist()],
            "std_by_view": [float(v) for v in matrix.std(axis=0).tolist()],
            "collapse_rate_gt_095": float((matrix.max(axis=-1) > 0.95).mean()) if matrix.size else 0.0,
        }
        existing = target_view_payload.get(target)
        if isinstance(existing, torch.Tensor):
            existing = existing.detach().cpu().float()
            if tuple(existing.shape) != tuple(weights_tensor.shape) or not torch.allclose(existing, weights_tensor, atol=1e-6):
                layer_warnings.append(
                    f"eval layer_weights.pt view_weights_by_target[{target}] differs from predictions.csv; "
                    "analysis layer_weights.pt uses predictions.csv as source of truth"
                )
        target_view_payload[target] = weights_tensor
    if target_view_payload:
        layer_payload["view_weights_by_target"] = target_view_payload
    for view_name, tensor in (layer_payload.get("layer_weights") or {}).items():
        if not torch.isfinite(tensor).all():
            layer_failures.append(f"layer_weights[{view_name}] contains NaN or Inf")
        elif tensor.dim() == 2 and not torch.allclose(
            tensor.sum(dim=-1), torch.ones(tensor.shape[0]), atol=1e-4
        ):
            layer_failures.append(f"layer_weights[{view_name}] rows do not sum to 1")
        if torch.isfinite(tensor).all() and tensor.dim() == 2 and tensor.numel() > 0:
            max_per_sample = tensor.max(dim=-1).values
            mean_max = float(max_per_sample.mean().item())
            mean_by_layer = tensor.mean(dim=0)
            max_by_layer = tensor.max(dim=0).values
            effectively_unused = [
                int(idx)
                for idx, value in enumerate(max_by_layer.tolist())
                if float(value) <= 1e-6 or float(mean_by_layer[idx].item()) <= 1e-4
            ]
            fusion_diagnostics["layer_weights"][view_name] = {
                "average_max_layer_weight": mean_max,
                "effectively_unused_layers": effectively_unused,
                "mean_by_layer": [float(v) for v in mean_by_layer.tolist()],
            }
            if mean_max > 0.95:
                layer_warnings.append(
                    f"layer_weights[{view_name}] average max layer weight {mean_max:.4f} > 0.95"
                )
            if effectively_unused:
                layer_warnings.append(
                    f"layer_weights[{view_name}] effectively unused layers: {effectively_unused}"
                )
    view_weights = layer_payload.get("view_weights")
    if isinstance(view_weights, torch.Tensor):
        if not torch.isfinite(view_weights).all():
            layer_failures.append("view_weights contains NaN or Inf")
        elif view_weights.dim() == 2:
            row_sums = view_weights.sum(dim=-1)
            if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4):
                layer_failures.append("view_weights rows do not sum to 1")
            if prediction_view_matrix is not None:
                expected = torch.as_tensor(prediction_view_matrix, dtype=view_weights.dtype)
                if tuple(view_weights.shape) != tuple(expected.shape) or not torch.allclose(view_weights, expected, atol=1e-6):
                    layer_failures.append("view_weights do not match predictions.csv per-sample view weights")
        elif view_weights.dim() == 1:
            layer_failures.append("view_weights is a static [num_views] vector; expected [num_samples, num_views]")
        else:
            layer_failures.append(f"view_weights has invalid shape: {tuple(view_weights.shape)}")
        if torch.isfinite(view_weights).all():
            if view_weights.dim() == 2:
                fusion_diagnostics["view_weights"]["tensor_shape"] = list(view_weights.shape)
                fusion_diagnostics["view_weights"]["tensor_mean_by_view"] = [float(v) for v in view_weights.mean(dim=0).tolist()]
            else:
                fusion_diagnostics["view_weights"]["values"] = [float(v) for v in view_weights.reshape(-1).tolist()]
    near_constant_view_columns: List[str] = []
    all_view_weight_columns = list(dict.fromkeys([*view_weight_columns, *[col for group in view_weight_groups.values() for col in group]]))
    for column in all_view_weight_columns:
        values = np.asarray([_float_or_nan(row.get(column)) for row in rows_csv], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size:
            std = float(values.std())
            fusion_diagnostics["view_weights"][column] = {"std": std, "mean": float(values.mean())}
            if std <= 1e-6:
                near_constant_view_columns.append(column)
    for target, columns in {"legacy": view_weight_columns, **view_weight_groups}.items():
        matrix = _numeric_matrix_from_rows(rows_csv, columns)
        if matrix is None or not matrix.size:
            continue
        collapse_rate = float((matrix.max(axis=-1) > 0.95).mean())
        if collapse_rate > 0.5:
            layer_warnings.append(
                f"view-weight group {target} has collapse_rate_gt_095={collapse_rate:.4f}; "
                "inspect fusion diagnostics and ablations before interpreting multi-view fusion"
            )
    target_means: Dict[str, np.ndarray] = {}
    for target, columns in view_weight_groups.items():
        matrix = _numeric_matrix_from_rows(rows_csv, columns)
        if matrix is not None and matrix.size:
            target_means[target] = matrix.mean(axis=0)
    if len(target_means) > 1:
        pairwise_l1: Dict[str, float] = {}
        target_names = sorted(target_means)
        for left_idx, left in enumerate(target_names):
            for right in target_names[left_idx + 1:]:
                pairwise_l1[f"{left}__{right}"] = float(np.mean(np.abs(target_means[left] - target_means[right])))
        fusion_diagnostics["view_weight_pairwise_l1"] = pairwise_l1
        if pairwise_l1 and max(pairwise_l1.values()) <= 1e-4:
            layer_warnings.append(
                "dimension-specific view weights are numerically identical across targets; "
                "treat dimension-specific fusion as unproven for this run"
            )
    if near_constant_view_columns:
        layer_warnings.append(
            "view weights are global/static or have near-zero standard deviation across samples: "
            + ", ".join(near_constant_view_columns)
        )
    if layer_failures:
        export_sanity["status"] = "failed"
        export_sanity.setdefault("failures", []).extend(layer_failures)
    if layer_warnings:
        export_sanity.setdefault("warnings", []).extend(layer_warnings)
    export_sanity["fusion_diagnostics"] = fusion_diagnostics
    eval_dimension_diagnostics_path = eval_root / "dimension_diagnostics.json"
    dimension_diagnostics_path: Optional[Path] = None
    if eval_dimension_diagnostics_path.is_file():
        try:
            dimension_diagnostics = json.loads(eval_dimension_diagnostics_path.read_text(encoding="utf-8"))
            export_sanity["dimension_diagnostics"] = dimension_diagnostics
            dimension_diagnostics_path = out_dir / "dimension_diagnostics.json"
            write_json(dimension_diagnostics_path, dimension_diagnostics)
            for warning in dimension_diagnostics.get("warnings", []) if isinstance(dimension_diagnostics, dict) else []:
                export_warnings.append(str(warning))
        except Exception:
            export_warnings.append(f"failed to read evaluation dimension diagnostics: {eval_dimension_diagnostics_path}")

    export_sanity_path = out_dir / "export_sanity.json"
    write_json(export_sanity_path, export_sanity)

    if export_sanity["status"] != "success":
        meta = {
            "dataset": cfg.dataset_name,
            "model": cfg.model_name,
            "n_samples": n,
            "status": "failed",
            "failure_reason": export_sanity.get("failures", []),
            "predictions_csv": str(predictions_csv),
            "export_sanity_path": str(export_sanity_path),
        }
        with open(out_dir / "meta.json", "w", encoding="utf-8") as fw:
            json.dump(meta, fw, ensure_ascii=False, indent=2)
        write_stage_manifest(
            out_dir,
            stage="export_analysis",
            status="failed",
            dataset=cfg.dataset_name,
            model=cfg.model_name,
            artifacts={"predictions_csv": str(predictions_csv)},
            sanity=export_sanity,
            error="export sanity failed",
            pair_context=ctx,
        )
        raise ValueError(f"export sanity failed: {export_sanity['failures']}")

    csv_path = out_dir / "per_sample.csv"
    if rows_csv:
        fields: List[str] = []
        seen = set()
        for r in rows_csv:
            for k in r.keys():
                if k not in seen:
                    seen.add(k); fields.append(k)
        with open(csv_path, "w", newline="", encoding="utf-8") as fw:
            writer = csv.DictWriter(fw, fieldnames=fields)
            writer.writeheader()
            for r in rows_csv:
                writer.writerow(r)

    json_path = out_dir / "per_sample.json"
    with open(json_path, "w", encoding="utf-8") as fw:
        json.dump(rows_json, fw, ensure_ascii=False, indent=2)

    # ---- raw layer-weight tensors ----
    lw_path = out_dir / "layer_weights.pt"
    torch.save(layer_payload, lw_path)

    view_ablation_summary_path: Optional[Path] = None
    view_ablation_csv = eval_root / "view_ablation.csv"
    if view_ablation_csv.is_file():
        view_rows = read_csv_rows(view_ablation_csv)
        ranked = sorted(
            view_rows,
            key=lambda row: _float_or_nan(row.get("AUROC")),
            reverse=True,
        )
        view_ablation_summary = {
            "source_csv": str(view_ablation_csv),
            "rows": view_rows,
            "best_by_auroc": ranked[0] if ranked and math.isfinite(_float_or_nan(ranked[0].get("AUROC"))) else None,
        }
        view_ablation_summary_path = out_dir / "view_ablation_summary.json"
        write_json(view_ablation_summary_path, view_ablation_summary)

    diagnostic_ablation_summary_path: Optional[Path] = None
    diagnostic_ablation_csv = eval_root / "diagnostic_ablation.csv"
    if diagnostic_ablation_csv.is_file():
        diagnostic_rows = read_csv_rows(diagnostic_ablation_csv)
        ranked = sorted(
            diagnostic_rows,
            key=lambda row: _float_or_nan(row.get("AUROC")),
            reverse=True,
        )
        diagnostic_ablation_summary = {
            "source_csv": str(diagnostic_ablation_csv),
            "rows": diagnostic_rows,
            "best_by_auroc": ranked[0] if ranked and math.isfinite(_float_or_nan(ranked[0].get("AUROC"))) else None,
        }
        diagnostic_ablation_summary_path = out_dir / "diagnostic_ablation_summary.json"
        write_json(diagnostic_ablation_summary_path, diagnostic_ablation_summary)

    # ---- meta ----
    dimension_names = list(layer_payload.get("dimension_names") or [c[len("dim_"):] for c in dim_columns])
    if export_target_metadata_warning:
        export_warnings.append(export_target_metadata_warning)
    meta = {
        "dataset": cfg.dataset_name,
        "model": cfg.model_name,
        "n_samples": n,
        "status": "complete",
        "dimension_names": dimension_names,
        "task_type": rows_csv[0].get("task_type") if rows_csv else target_payload_metadata.get("task_type"),
        "dataset_diagnostic_spec": target_payload_metadata.get("dataset_diagnostic_spec"),
        "target_semantic_summary": target_semantic_summary,
        "target_status_summary": export_sanity.get("target_status_distribution"),
        "overall_target_policy": rows_csv[0].get("overall_target_policy") if rows_csv else target_payload_metadata.get("overall_target_policy"),
        "overall_target_dimension_weights": rows_csv[0].get("overall_target_dimension_weights") if rows_csv else target_payload_metadata.get("overall_target_dimension_weights"),
        "view_names": list(layer_payload.get("view_names") or []),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "layer_weights_path": str(lw_path),
        "predictions_csv": str(predictions_csv),
        "predictions_json": str(predictions_json) if predictions_json.is_file() else None,
        "export_sanity_path": str(export_sanity_path),
        "sample_id_alignment_sanity_path": str(sample_id_alignment_path),
        "label_threshold": LABEL_THRESHOLD,
        "view_weight_groups": {key: list(value) for key, value in view_weight_groups.items()},
        "view_fusion": export_sanity.get("evaluation_context", {}).get("view_fusion", {}),
        "view_ablation_summary_path": str(view_ablation_summary_path) if view_ablation_summary_path else None,
        "dimension_diagnostics_path": str(dimension_diagnostics_path) if dimension_diagnostics_path else None,
        "diagnostic_ablation_summary_path": str(diagnostic_ablation_summary_path) if diagnostic_ablation_summary_path else None,
    }
    manifest_artifacts = {
        "per_sample_csv": str(csv_path),
        "per_sample_json": str(json_path),
        "layer_weights_pt": str(lw_path),
        "export_sanity_json": str(export_sanity_path),
        "sample_id_alignment_sanity_json": str(sample_id_alignment_path),
    }
    if view_ablation_summary_path is not None:
        manifest_artifacts["view_ablation_summary_json"] = str(view_ablation_summary_path)
    if dimension_diagnostics_path is not None:
        manifest_artifacts["dimension_diagnostics_json"] = str(dimension_diagnostics_path)
    if diagnostic_ablation_summary_path is not None:
        manifest_artifacts["diagnostic_ablation_summary_json"] = str(diagnostic_ablation_summary_path)
    manifest_path = write_stage_manifest(
        out_dir,
        stage="export_analysis",
        status="success",
        dataset=cfg.dataset_name,
        model=cfg.model_name,
        artifacts=manifest_artifacts,
        sanity=export_sanity,
        pair_context=ctx,
    )
    meta["manifest_path"] = str(manifest_path)
    with open(out_dir / "meta.json", "w", encoding="utf-8") as fw:
        json.dump(meta, fw, ensure_ascii=False, indent=2)

    if cfg.dataset_name.startswith("truthfulqa__"):
        try:
            from common.truthfulqa_row_count_trace import write_truthfulqa_row_count_trace

            write_truthfulqa_row_count_trace(ctx)
        except Exception:
            pass

    return meta


def _resolved_cfg(
    cfg_template: MDUQTrainConfig,
    resolved: ExistingDiagUQArtifactRoot,
) -> MDUQTrainConfig:
    return MDUQTrainConfig(
        **{
            **cfg_template.__dict__,
            "dataset_name": resolved.dataset_root_name,
            "model_name": resolved.model_root_name,
        }
    )


def _format_missing_subdirs(
    resolved: ExistingDiagUQArtifactRoot,
    required: Sequence[str],
) -> str:
    missing = resolved.missing_subdirs(required)
    return "; ".join(f"{name} missing: {path}" for name, path in missing)


def _log_export_info(message: str, *args) -> None:
    try:
        from loguru import logger

        logger.info(message, *args)
    except Exception:
        print(message.format(*args))


def _log_export_warning(message: str, *args) -> None:
    try:
        from loguru import logger

        logger.warning(message, *args)
    except Exception:
        print(message.format(*args))


def export_mduq_outputs_for_pairs(
    pairs: Sequence[tuple],
    cfg_template: MDUQTrainConfig,
    *,
    skip_on_error: bool = True,
    artifact_root_name: Optional[str] = None,
    checkpoint_artifact_root_name: Optional[str] = None,
    checkpoint_split: Optional[str] = None,
    train_pairs: Optional[Sequence[tuple[str, str]]] = None,
    allow_train_eval_same_split: bool = False,
    allow_train_fallback: bool = False,
    calibrate_confidence: bool = False,
    run_inference: bool = False,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    requested_train_pairs = list(train_pairs or [])
    if checkpoint_split and "," in checkpoint_split:
        raise ValueError(INVALID_CHECKPOINT_VARIANT_MESSAGE)
    required_subdirs = ("hidden_bank", "dimension_targets") if run_inference else ("hidden_bank", "dimension_targets", "eval")
    contexts = [
        resolve_pair_context(
            artifact_root_name or dataset_name,
            model_name,
            runtime_root=cfg_template.output_root,
        )
        for dataset_name, model_name in pairs
    ]
    assert_no_duplicate_output_dirs(contexts, "export_analysis")
    for (dataset_name, model_name), ctx in zip(pairs, contexts):
        resolved_root = ctx.diaguq_root
        if not resolved_root.is_dir() and allow_train_fallback:
            fallback = resolve_existing_diaguq_artifact_root(
                dataset_name,
                model_name,
                cfg_template.output_root,
                artifact_root_name=artifact_root_name,
                allow_train_fallback=True,
            )
            if fallback.found and fallback.artifact_root is not None:
                ctx = resolve_pair_context(
                    str(fallback.dataset_root_name),
                    str(fallback.model_root_name or model_name),
                    runtime_root=cfg_template.output_root,
                )
                resolved_root = Path(fallback.artifact_root)
        if not resolved_root.is_dir():
            reason = (
                "missing exact DiagUQ artifact root for pair context: "
                f"requested_dataset={dataset_name!r} resolved_variant={ctx.resolved_variant!r} "
                f"model={model_name!r} path={resolved_root}. "
                "No train fallback is used unless --allow-train-fallback is passed."
            )
            _log_export_warning(
                "[export-analysis] skipped requested_dataset={} model={} reason={}",
                dataset_name, model_name, reason,
            )
            results.append({
                "dataset": dataset_name,
                "model": model_name,
                "skipped": True,
                "reason": reason,
            })
            continue

        legacy_checkpoint_dataset = None
        if checkpoint_artifact_root_name is None and not requested_train_pairs and checkpoint_split:
            base_dataset, _ = split_dataset_and_raw(dataset_name)
            legacy_checkpoint_dataset = normalize_split_tag(base_dataset, checkpoint_split)
        checkpoint_ctx = resolve_checkpoint_context_for_eval(
            ctx,
            requested_train_pairs,
            checkpoint_dataset=checkpoint_artifact_root_name or legacy_checkpoint_dataset,
        )
        checkpoint_root = checkpoint_ctx.diaguq_root
        checkpoint_missing_reason = ""
        if not checkpoint_root.is_dir():
            checkpoint_missing_reason = f"checkpoint root missing: {checkpoint_root}"
        elif not checkpoint_ctx.checkpoint_dir.is_dir():
            checkpoint_missing_reason = f"checkpoints missing: {checkpoint_ctx.checkpoint_dir}"
        if run_inference and checkpoint_missing_reason:
            _log_export_warning(
                "[export-analysis] skipped requested_dataset={} model={} checkpoint_reason={}",
                dataset_name, model_name, checkpoint_missing_reason,
            )
            results.append({
                "dataset": dataset_name,
                "model": model_name,
                "skipped": True,
                "reason": checkpoint_missing_reason,
                "artifact_root": str(resolved_root),
            })
            continue

        same_split = checkpoint_ctx.resolved_variant == ctx.resolved_variant
        if same_split and not allow_train_eval_same_split:
            _log_export_warning(
                "[export-analysis] This is an in-split run, not a held-out evaluation."
            )

        missing = [
            (name, resolved_root / name)
            for name in required_subdirs
            if not (resolved_root / name).is_dir()
        ]
        missing_reason = "; ".join(f"{name} missing: {path}" for name, path in missing)
        if missing_reason:
            _log_export_warning(
                "[export-analysis] skipped requested_dataset={} model={} "
                "resolved_artifact_root={} reason={}",
                dataset_name, model_name, resolved_root, missing_reason,
            )
            results.append({
                "dataset": dataset_name,
                "model": model_name,
                "skipped": True,
                "reason": missing_reason,
                "artifact_root": str(resolved_root),
            })
            continue

        cfg = MDUQTrainConfig(
            **{
                **cfg_template.__dict__,
                "dataset_name": ctx.resolved_variant,
                "model_name": ctx.model,
            }
        )
        _log_export_info(
            "[export-analysis] eval_dataset={} eval_resolved_variant={} "
            "checkpoint_resolved_variant={} checkpoint_path={} eval_write_root={}",
            dataset_name,
            ctx.resolved_variant,
            checkpoint_ctx.resolved_variant,
            checkpoint_root,
            ctx.analysis_dir,
        )
        try:
            checkpoint_base, _ = split_dataset_and_raw(checkpoint_ctx.resolved_variant)
            eval_base, _ = split_dataset_and_raw(ctx.resolved_variant)
            eval_context = {
                "train_dataset_root": checkpoint_ctx.resolved_variant if checkpoint_root.is_dir() else None,
                "eval_dataset_root": ctx.resolved_variant,
                "train_split": checkpoint_ctx.split if checkpoint_root.is_dir() else None,
                "eval_split": ctx.split,
                "held_out_evaluation": not same_split,
                "allow_train_eval_same_split": allow_train_eval_same_split,
                "allow_train_fallback": allow_train_fallback,
                "checkpoint_dataset": checkpoint_ctx.resolved_variant if checkpoint_root.is_dir() else None,
                "eval_dataset": ctx.resolved_variant,
                "cross_dataset_evaluation": checkpoint_base != eval_base,
                "trained_on_truthfulqa": checkpoint_ctx.resolved_variant.startswith("truthfulqa"),
                "same_split_evaluation": same_split,
            }
            eval_context.update(ctx.split_metadata)
            eval_context["held_out_evaluation"] = not same_split
            eval_context["same_split_evaluation"] = same_split
            if same_split:
                eval_context["split_policy"] = "same_split_debug"
            meta = export_mduq_outputs(
                cfg,
                artifact_root=resolved_root,
                checkpoint_artifact_root=checkpoint_root if checkpoint_root.is_dir() else None,
                evaluation_context=eval_context,
                calibrate_confidence=calibrate_confidence and not same_split,
                run_inference=run_inference,
            )
            meta.update({
                "requested_dataset": dataset_name,
                "resolved_dataset_root": ctx.resolved_variant,
                "resolved_split": ctx.split,
                "artifact_root": str(resolved_root),
            })
            results.append(meta)
        except Exception as exc:  # noqa: BLE001
            res = {
                "dataset": dataset_name,
                "model": model_name,
                "error": repr(exc),
            }
            _log_export_warning(
                "[export-analysis] failed requested_dataset={} model={} "
                "resolved_artifact_root={} error={}",
                dataset_name, model_name, ctx.resolved_variant, repr(exc),
            )
            results.append(res)
            if not skip_on_error:
                raise
    return results


# ---------------------------------------------------------------------------
# DiagUQ-style aliases.
# ---------------------------------------------------------------------------

export_diaguq_outputs = export_mduq_outputs
export_diaguq_outputs_for_pairs = export_mduq_outputs_for_pairs
