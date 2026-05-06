"""Fixed-layer hidden-state baselines for DiagUQ.

This module trains lightweight supervised estimators on existing
``hidden_bank`` and ``dimension_targets`` artifacts. It never calls an LLM or
rebuilds hidden states.
"""

from __future__ import annotations

import csv
import json
import math
import pickle
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.metrics import build_dimension_diagnostics, pearson_corr
from common.pair_context import (
    DiagUQPairContext,
    resolve_checkpoint_context_for_eval,
    resolve_pair_context,
)
from pipeline.evaluate_diaguq import LABEL_THRESHOLD, _metric_row_from_predictions
from features.load_feature_tensors import load_mduq_dataset
from registry.model_registry import get_candidate_layers
from pipeline.train_diaguq import MDUQTrainConfig, _resolve_overall_target, _select_dimension_targets
from analysis.layer_baseline_analysis import build_layer_vs_diaguq_summary_row, write_cross_dataset_layer_summary


DIMENSION_NAMES = ("ambiguity", "knowledge_gap", "predictive_variability")
FEATURE_MODES = (
    "query_answer_relation_concat",
    "answer_only",
    "query_only",
    "relation_only",
)
MLP_METHODS = (
    "last_layer_mlp",
    "fixed_middle_layer_mlp",
    "fixed_layer_scan_mlp",
    "best_fixed_layer_mlp",
    "oracle_best_fixed_layer",
    "uniform_multilayer_mean_mlp",
    "fixed_multilayer_concat_mlp",
)
MODEL_SPECIFIC_MIDDLE_LAYER = {
    "Llama-3.1-8B-Instruct": 16,
    "Qwen2.5-7B-Instruct": 14,
    "gemma-4-E4B-it": 16,
}


@dataclass
class LayerBaselineBundle:
    ctx: DiagUQPairContext
    feature_mode: str
    layers: list[int]
    views: dict[str, torch.Tensor]
    overall_target: torch.Tensor
    dimension_targets: torch.Tensor
    sample_ids: list[str]
    correctness_target_source: list[str]
    target_payload: Mapping[str, Any]


class LayerBaselineMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 4, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        width = int(hidden_dim or min(max(input_dim // 2, 32), 256))
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, width),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(width, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x.float()))


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
            writer.writerow(dict(row))


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def discover_candidate_layers(bank_dir: str | Path, model_name: Optional[str] = None) -> list[int]:
    bank = Path(bank_dir)
    for manifest_name in ("manifest.json", "hidden_bank_sanity.json"):
        path = bank / manifest_name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        candidates = [
            payload.get("layer_list"),
            (payload.get("sanity") or {}).get("layer_list") if isinstance(payload.get("sanity"), Mapping) else None,
        ]
        for candidate in candidates:
            if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
                layers = [int(value) for value in candidate]
                if layers:
                    return layers
    pattern = re.compile(r"^(?:query|answer|relation)_average_layer_(\d+)\.pt$")
    layers = sorted({int(match.group(1)) for path in bank.glob("*_average_layer_*.pt") if (match := pattern.match(path.name))})
    if layers:
        return layers
    if model_name:
        return list(get_candidate_layers(model_name))
    raise FileNotFoundError(f"could not discover candidate layers from hidden_bank: {bank}")


def select_last_layer(layers: Sequence[int]) -> int:
    if not layers:
        raise ValueError("candidate layer list is empty")
    return int(list(layers)[-1])


def select_middle_layer(model_name: str, layers: Sequence[int]) -> int:
    if not layers:
        raise ValueError("candidate layer list is empty")
    values = [int(layer) for layer in layers]
    preferred = MODEL_SPECIFIC_MIDDLE_LAYER.get(model_name)
    if preferred is not None:
        return min(values, key=lambda layer: (abs(layer - int(preferred)), values.index(layer)))
    return values[(len(values) - 1) // 2]


def select_best_layer(scores: Mapping[int, float]) -> int:
    if not scores:
        raise ValueError("no layer scores available for selection")
    return max(scores, key=lambda layer: float(scores[layer]) if math.isfinite(float(scores[layer])) else -float("inf"))


def _text_array(payload: Mapping[str, Any], key: str, n: int, *, default_prefix: str = "row") -> list[str]:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    values = payload.get(key)
    if not isinstance(values, (list, tuple)) and isinstance(metadata, Mapping):
        values = metadata.get(key)
    if isinstance(values, (list, tuple)):
        out = [str(value) if value not in (None, "") else f"{default_prefix}:{idx}" for idx, value in enumerate(values[:n])]
        if len(out) < n:
            out.extend([f"{default_prefix}:{idx}" for idx in range(len(out), n)])
        return out
    return [f"{default_prefix}:{idx}" for idx in range(n)]


def _load_targets(ctx: DiagUQPairContext) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, Any], list[str], list[str]]:
    path = ctx.dimension_targets_dir / "dimension_targets.pt"
    if not path.is_file():
        raise FileNotFoundError(f"missing dimension_targets: {path}")
    payload = torch.load(path, map_location="cpu")
    dims = _select_dimension_targets(payload, DIMENSION_NAMES).float()
    overall = _resolve_overall_target(payload).float()
    n = min(int(overall.shape[0]), int(dims.shape[0]))
    sample_ids = _text_array(payload, "sample_ids", n, default_prefix=ctx.resolved_variant)
    correctness_source = _text_array(payload, "task_error_target_source", n, default_prefix="missing")
    return overall[:n], dims[:n], payload, sample_ids, correctness_source


def load_layer_baseline_bundle(ctx: DiagUQPairContext, *, feature_mode: str = "query_answer_relation_concat") -> LayerBaselineBundle:
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"unknown feature_mode={feature_mode!r}; expected one of {FEATURE_MODES}")
    if not ctx.hidden_bank_dir.is_dir():
        raise FileNotFoundError(f"missing hidden_bank: {ctx.hidden_bank_dir}")
    if not ctx.dimension_targets_dir.is_dir():
        raise FileNotFoundError(f"missing dimension_targets: {ctx.dimension_targets_dir}")
    layers = discover_candidate_layers(ctx.hidden_bank_dir, ctx.model)
    bundle = load_mduq_dataset(
        ctx.resolved_variant,
        ctx.model,
        layer_list=layers,
        output_root=str(ctx.test_output_root),
        bank_dir=str(ctx.hidden_bank_dir),
    )
    views = {name: bundle["views"][name].float() for name in ("query", "answer", "relation")}
    overall, dims, payload, sample_ids, correctness_source = _load_targets(ctx)
    n = min(int(overall.shape[0]), int(next(iter(views.values())).shape[0]))
    views = {name: value[:n] for name, value in views.items()}
    return LayerBaselineBundle(
        ctx=ctx,
        feature_mode=feature_mode,
        layers=layers,
        views=views,
        overall_target=overall[:n],
        dimension_targets=dims[:n],
        sample_ids=sample_ids[:n],
        correctness_target_source=correctness_source[:n],
        target_payload=payload,
    )


def _concat_parts(parts: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([part.reshape(part.shape[0], -1).float() for part in parts], dim=-1)


def build_feature_matrix(bundle: LayerBaselineBundle, *, layer: Optional[int] = None, reduction: str = "single") -> torch.Tensor:
    indices = {layer_id: idx for idx, layer_id in enumerate(bundle.layers)}
    selected_views: list[torch.Tensor] = []
    view_names = {
        "answer_only": ("answer",),
        "query_only": ("query",),
        "relation_only": ("relation",),
        "query_answer_relation_concat": ("query", "answer", "relation"),
    }[bundle.feature_mode]
    for view_name in view_names:
        values = bundle.views[view_name]
        if reduction == "single":
            if layer is None:
                raise ValueError("single-layer feature construction requires layer=<id>")
            selected_views.append(values[:, indices[int(layer)], :])
        elif reduction == "mean":
            selected_views.append(values.mean(dim=1))
        elif reduction == "concat":
            selected_views.append(values.reshape(values.shape[0], -1))
        else:
            raise ValueError(f"unknown reduction={reduction!r}")
    return _concat_parts(selected_views)


def _target_matrix(bundle: LayerBaselineBundle) -> torch.Tensor:
    return torch.cat([bundle.overall_target.view(-1, 1), bundle.dimension_targets], dim=1).float()


def _train_val_indices(n: int, *, val_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if n < 2:
        raise RuntimeError(f"need at least two training rows for layer baseline, got {n}")
    n_val = max(1, min(n - 1, int(round(n * float(val_fraction)))))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(int(seed)))
    return perm[n_val:], perm[:n_val]


def _standardize(train_x: torch.Tensor, *others: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return tuple((x - mean) / std for x in (train_x, *others)), mean.squeeze(0), std.squeeze(0)


def _masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(target)
    if not bool(mask.any()):
        return pred.new_zeros(())
    return F.mse_loss(pred[mask], target[mask])


def _fit_mlp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    device: str,
) -> LayerBaselineMLP:
    torch.manual_seed(int(seed))
    model = LayerBaselineMLP(int(train_x.shape[1]), output_dim=int(train_y.shape[1])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    x = train_x.to(device)
    y = train_y.to(device)
    for _ in range(max(1, int(epochs))):
        perm = torch.randperm(x.shape[0], device=x.device)
        for start in range(0, int(x.shape[0]), max(1, int(batch_size))):
            idx = perm[start:start + max(1, int(batch_size))]
            pred = model(x[idx])
            loss = _masked_mse(pred[:, :1], y[idx, :1]) + 0.5 * _masked_mse(pred[:, 1:], y[idx, 1:])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.cpu()


def _fit_rf(train_x: torch.Tensor, train_y: torch.Tensor):
    try:
        from sklearn.ensemble import RandomForestRegressor
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("fixed_layer_rf requires scikit-learn to be installed") from exc
    reg = RandomForestRegressor(n_estimators=100, random_state=0, min_samples_leaf=2, n_jobs=-1)
    y = torch.where(torch.isfinite(train_y), train_y, torch.zeros_like(train_y)).numpy()
    reg.fit(train_x.numpy(), y)
    return reg


def _predict(model: Any, x: torch.Tensor, *, baseline_model: str) -> torch.Tensor:
    if baseline_model == "rf":
        return torch.tensor(np.asarray(model.predict(x.numpy())), dtype=torch.float32).clamp(0.0, 1.0)
    with torch.no_grad():
        return model(x).detach().cpu().float().clamp(0.0, 1.0)


def _diagnostic_summary(dataset: str, model: str, pred: torch.Tensor, targets: torch.Tensor) -> tuple[dict[str, Any], dict[str, float]]:
    rows = []
    for idx in range(int(pred.shape[0])):
        row = {"dataset": dataset, "model": model, "overall_target": float(targets[idx, 0])}
        for dim_idx, name in enumerate(DIMENSION_NAMES):
            row[f"dim_{name}"] = float(pred[idx, dim_idx + 1])
            row[f"{name}_target"] = float(targets[idx, dim_idx + 1])
        rows.append(row)
    diagnostics = build_dimension_diagnostics(rows, dimension_names=DIMENSION_NAMES)
    summary = diagnostics.get("summary") or {}
    dim_metrics = {
        f"{name}_metric": pearson_corr(pred[:, dim_idx + 1].numpy(), targets[:, dim_idx + 1].numpy())
        for dim_idx, name in enumerate(DIMENSION_NAMES)
    }
    dim_metrics["mean_diagonal_margin"] = float(summary.get("mean_diagonal_margin") or float("nan"))
    dim_metrics["max_prediction_prediction_corr"] = float(summary.get("max_prediction_prediction_abs_corr") or float("nan"))
    return diagnostics, dim_metrics


def _evaluate_predictions(
    *,
    eval_ctx: DiagUQPairContext,
    method: str,
    feature_mode: str,
    layer_id: Optional[int],
    selected_layer: Optional[int],
    is_oracle: bool,
    selection_metric: Optional[str],
    selection_split: Optional[str],
    selected_layer_score: Optional[float],
    n_train: int,
    pred: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, Any]:
    knowledge_gap = targets[:, 2].numpy()
    correct = np.where(np.isfinite(knowledge_gap), (knowledge_gap < LABEL_THRESHOLD).astype(np.float64), np.nan)
    confidence = (1.0 - pred[:, 0].numpy()).astype(np.float64)
    row = _metric_row_from_predictions(eval_ctx.resolved_variant, eval_ctx.model, method, confidence, correct)
    _, dim_metrics = _diagnostic_summary(eval_ctx.resolved_variant, eval_ctx.model, pred, targets)
    row.update(
        {
            "feature_mode": feature_mode,
            "layer_id": layer_id,
            "layer_name": f"layer_{layer_id}" if layer_id is not None else None,
            "selected_layer": selected_layer,
            "is_oracle": bool(is_oracle),
            "selection_metric": selection_metric,
            "selection_split": selection_split,
            "selected_layer_score": selected_layer_score,
            "n_train": int(n_train),
            "ambiguity_metric": dim_metrics["ambiguity_metric"],
            "knowledge_gap_metric": dim_metrics["knowledge_gap_metric"],
            "predictive_variability_metric": dim_metrics["predictive_variability_metric"],
            "mean_diagonal_margin": dim_metrics["mean_diagonal_margin"],
            "max_prediction_prediction_corr": dim_metrics["max_prediction_prediction_corr"],
        }
    )
    if is_oracle:
        row["oracle_note"] = "not valid for fair model selection"
    return row


def _prediction_rows(
    *,
    eval_bundle: LayerBaselineBundle,
    method: str,
    feature_mode: str,
    layer_id: Optional[int],
    selected_layer: Optional[int],
    is_oracle: bool,
    pred: torch.Tensor,
    targets: torch.Tensor,
) -> list[dict[str, Any]]:
    knowledge_gap = targets[:, 2].numpy()
    correct = np.where(
        np.isfinite(knowledge_gap),
        (knowledge_gap < LABEL_THRESHOLD).astype(np.float64),
        np.nan,
    )
    rows: list[dict[str, Any]] = []
    for idx in range(int(pred.shape[0])):
        uncertainty = float(pred[idx, 0])
        confidence = 1.0 - uncertainty if math.isfinite(uncertainty) else float("nan")
        y_true_correct = float(correct[idx]) if math.isfinite(float(correct[idx])) else float("nan")
        rows.append(
            {
                "sample_id": eval_bundle.sample_ids[idx] if idx < len(eval_bundle.sample_ids) else f"{eval_bundle.ctx.resolved_variant}:{idx}",
                "dataset": eval_bundle.ctx.resolved_variant,
                "model": eval_bundle.ctx.model,
                "method": method,
                "feature_mode": feature_mode,
                "layer_id": layer_id,
                "layer_name": f"layer_{layer_id}" if layer_id is not None else None,
                "is_oracle": bool(is_oracle),
                "selected_layer": selected_layer,
                "y_true_correct": y_true_correct,
                "y_true_error": 1.0 - y_true_correct if math.isfinite(y_true_correct) else float("nan"),
                "predicted_uncertainty": uncertainty,
                "predicted_confidence": confidence,
                "overall_target": float(targets[idx, 0]),
                "correctness_target_source": (
                    eval_bundle.correctness_target_source[idx]
                    if idx < len(eval_bundle.correctness_target_source)
                    else "missing"
                ),
                "pred_ambiguity": float(pred[idx, 1]) if pred.shape[1] > 1 else float("nan"),
                "pred_knowledge_gap": float(pred[idx, 2]) if pred.shape[1] > 2 else float("nan"),
                "pred_predictive_variability": float(pred[idx, 3]) if pred.shape[1] > 3 else float("nan"),
            }
        )
    return rows


def _train_and_eval_spec(
    *,
    train_bundle: LayerBaselineBundle,
    eval_bundle: LayerBaselineBundle,
    method: str,
    baseline_model: str,
    layer: Optional[int],
    reduction: str,
    cfg: MDUQTrainConfig,
    checkpoint_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor]:
    train_x_full = build_feature_matrix(train_bundle, layer=layer, reduction=reduction)
    eval_x = build_feature_matrix(eval_bundle, layer=layer, reduction=reduction)
    train_y_full = _target_matrix(train_bundle)
    eval_y = _target_matrix(eval_bundle)
    train_idx, val_idx = _train_val_indices(train_x_full.shape[0], val_fraction=cfg.val_fraction, seed=cfg.seed)
    (train_x, val_x, eval_x), feature_mean, feature_std = _standardize(train_x_full[train_idx], train_x_full[val_idx], eval_x)
    train_y = train_y_full[train_idx]
    val_y = train_y_full[val_idx]
    if baseline_model == "rf":
        estimator = _fit_rf(train_x, train_y)
    else:
        estimator = _fit_mlp(train_x, train_y, epochs=cfg.num_epochs, lr=cfg.learning_rate, batch_size=cfg.batch_size, seed=cfg.seed, device=cfg.device)
    val_pred = _predict(estimator, val_x, baseline_model=baseline_model)
    eval_pred = _predict(estimator, eval_x, baseline_model=baseline_model)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if baseline_model == "rf":
        with (checkpoint_dir / "model.pkl").open("wb") as fw:
            pickle.dump({"model": estimator, "feature_mean": feature_mean, "feature_std": feature_std, "method": method, "layer": layer, "feature_mode": train_bundle.feature_mode}, fw)
    else:
        torch.save({"model_state": estimator.state_dict(), "input_dim": int(train_x.shape[1]), "feature_mean": feature_mean, "feature_std": feature_std, "method": method, "layer": layer, "feature_mode": train_bundle.feature_mode}, checkpoint_dir / "model.pt")
    val_row = _evaluate_predictions(
        eval_ctx=train_bundle.ctx,
        method=method,
        feature_mode=train_bundle.feature_mode,
        layer_id=layer,
        selected_layer=None,
        is_oracle=False,
        selection_metric=None,
        selection_split="train_validation",
        selected_layer_score=None,
        n_train=int(train_idx.numel()),
        pred=val_pred,
        targets=val_y,
    )
    eval_row = _evaluate_predictions(
        eval_ctx=eval_bundle.ctx,
        method=method,
        feature_mode=train_bundle.feature_mode,
        layer_id=layer,
        selected_layer=None,
        is_oracle=False,
        selection_metric=None,
        selection_split=None,
        selected_layer_score=None,
        n_train=int(train_idx.numel()),
        pred=eval_pred,
        targets=eval_y,
    )
    return val_row, eval_row, eval_pred


def _method_suffix(baseline_model: str) -> str:
    return "rf" if baseline_model == "rf" else "mlp"


def run_layer_baselines_for_pair(
    train_ctx: DiagUQPairContext,
    eval_ctx: DiagUQPairContext,
    cfg: MDUQTrainConfig,
    *,
    feature_mode: str,
    baseline_model: str = "mlp",
    force: bool = False,
) -> dict[str, Any]:
    if baseline_model not in {"mlp", "rf"}:
        raise ValueError("baseline_model must be 'mlp' or 'rf'")
    train_bundle = load_layer_baseline_bundle(train_ctx, feature_mode=feature_mode)
    eval_bundle = load_layer_baseline_bundle(eval_ctx, feature_mode=feature_mode)
    baseline_root = train_ctx.diaguq_root / "baselines" / "layer_baselines"
    if force and baseline_root.exists():
        shutil.rmtree(baseline_root)
    suffix = _method_suffix(baseline_model)
    rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    val_scores: dict[int, float] = {}
    eval_scores: dict[int, float] = {}
    scan_rows_by_layer: dict[int, dict[str, Any]] = {}
    scan_preds_by_layer: dict[int, torch.Tensor] = {}

    last_layer = select_last_layer(train_bundle.layers)
    middle_layer = select_middle_layer(train_ctx.model, train_bundle.layers)
    fixed_specs = [
        (f"last_layer_{suffix}", last_layer, "single"),
        (f"fixed_middle_layer_{suffix}", middle_layer, "single"),
    ]
    if baseline_model == "mlp":
        fixed_specs.extend([
            ("uniform_multilayer_mean_mlp", None, "mean"),
            ("fixed_multilayer_concat_mlp", None, "concat"),
        ])
    for method, layer, reduction in fixed_specs:
        _, eval_row, eval_pred = _train_and_eval_spec(train_bundle=train_bundle, eval_bundle=eval_bundle, method=method, baseline_model=baseline_model, layer=layer, reduction=reduction, cfg=cfg, checkpoint_dir=baseline_root / method / feature_mode)
        rows.append(eval_row)
        prediction_rows.extend(_prediction_rows(eval_bundle=eval_bundle, method=method, feature_mode=feature_mode, layer_id=layer, selected_layer=layer, is_oracle=False, pred=eval_pred, targets=_target_matrix(eval_bundle)))

    for layer in train_bundle.layers:
        val_row, eval_row, eval_pred = _train_and_eval_spec(train_bundle=train_bundle, eval_bundle=eval_bundle, method=f"fixed_layer_scan_{suffix}", baseline_model=baseline_model, layer=layer, reduction="single", cfg=cfg, checkpoint_dir=baseline_root / f"fixed_layer_scan_{suffix}" / feature_mode / f"layer_{layer}")
        val_scores[int(layer)] = float(val_row.get("AUROC") or float("nan"))
        eval_scores[int(layer)] = float(eval_row.get("AUROC") or float("nan"))
        scan_rows_by_layer[int(layer)] = eval_row
        scan_preds_by_layer[int(layer)] = eval_pred
        rows.append(eval_row)
        prediction_rows.extend(_prediction_rows(eval_bundle=eval_bundle, method=f"fixed_layer_scan_{suffix}", feature_mode=feature_mode, layer_id=layer, selected_layer=layer, is_oracle=False, pred=eval_pred, targets=_target_matrix(eval_bundle)))

    best_layer = select_best_layer(val_scores)
    best_row = dict(scan_rows_by_layer[best_layer])
    best_row.update({"method": f"best_fixed_layer_{suffix}", "selected_layer": best_layer, "selection_metric": "AUROC", "selection_split": "train_validation", "selected_layer_score": val_scores[best_layer]})
    rows.append(best_row)
    prediction_rows.extend(_prediction_rows(eval_bundle=eval_bundle, method=f"best_fixed_layer_{suffix}", feature_mode=feature_mode, layer_id=best_layer, selected_layer=best_layer, is_oracle=False, pred=scan_preds_by_layer[best_layer], targets=_target_matrix(eval_bundle)))
    if baseline_model == "mlp":
        oracle_layer = select_best_layer(eval_scores)
        oracle_row = dict(scan_rows_by_layer[oracle_layer])
        oracle_row.update({"method": "oracle_best_fixed_layer", "selected_layer": oracle_layer, "is_oracle": True, "selection_metric": "AUROC", "selection_split": "evaluation", "selected_layer_score": eval_scores[oracle_layer], "oracle_note": "not valid for fair model selection"})
        rows.append(oracle_row)
        prediction_rows.extend(_prediction_rows(eval_bundle=eval_bundle, method="oracle_best_fixed_layer", feature_mode=feature_mode, layer_id=oracle_layer, selected_layer=oracle_layer, is_oracle=True, pred=scan_preds_by_layer[oracle_layer], targets=_target_matrix(eval_bundle)))

    return {"rows": rows, "prediction_rows": prediction_rows, "best_layer": best_layer, "oracle_layer": select_best_layer(eval_scores), "baseline_root": str(baseline_root)}


def _load_diaguq_auroc(eval_ctx: DiagUQPairContext) -> float:
    metrics_path = eval_ctx.eval_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"missing DiagUQ metrics for comparison: {metrics_path}")
    rows = json.loads(metrics_path.read_text(encoding="utf-8"))
    for row in rows:
        if isinstance(row, Mapping) and row.get("method") == "mduq":
            value = row.get("AUROC")
            return float(value) if value is not None else float("nan")
    raise KeyError(f"metrics.json has no method='mduq' row: {metrics_path}")


def _row_auroc(rows: Sequence[Mapping[str, Any]], method: str) -> float:
    for row in rows:
        if row.get("method") == method:
            value = row.get("AUROC")
            return float(value) if value is not None else float("nan")
    return float("nan")


def _row_selected_layer(rows: Sequence[Mapping[str, Any]], method: str) -> Optional[int]:
    for row in rows:
        if row.get("method") == method:
            value = row.get("selected_layer") or row.get("layer_id")
            return int(value) if value is not None else None
    return None


def _comparison_row(eval_ctx: DiagUQPairContext, rows: Sequence[Mapping[str, Any]], feature_mode: str) -> dict[str, Any]:
    return build_layer_vs_diaguq_summary_row(eval_ctx, rows, feature_mode=feature_mode)


def run_layer_baseline_pairs(
    eval_pairs: Iterable[tuple[str, str]],
    train_pairs: Iterable[tuple[str, str]],
    cfg_template: MDUQTrainConfig,
    *,
    feature_modes: Sequence[str] = ("query_answer_relation_concat",),
    baseline_model: str = "mlp",
    checkpoint_dataset: Optional[str] = None,
    force: bool = False,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    train_pairs = list(train_pairs)
    selected_layers_by_dataset: dict[str, int] = {}
    comparison_rows_by_model: dict[str, list[dict[str, Any]]] = {}
    for eval_dataset, model in eval_pairs:
        eval_ctx = resolve_pair_context(eval_dataset, model, runtime_root=cfg_template.output_root)
        train_ctx = resolve_checkpoint_context_for_eval(eval_ctx, train_pairs, checkpoint_dataset=checkpoint_dataset)
        pair_baseline_root = train_ctx.diaguq_root / "baselines" / "layer_baselines"
        if force and pair_baseline_root.exists():
            shutil.rmtree(pair_baseline_root)
        all_rows: list[dict[str, Any]] = []
        all_prediction_rows: list[dict[str, Any]] = []
        comparison_rows: list[dict[str, Any]] = []
        for feature_mode in feature_modes:
            cfg = MDUQTrainConfig(**{**cfg_template.__dict__, "dataset_name": train_ctx.resolved_variant, "model_name": train_ctx.model})
            result = run_layer_baselines_for_pair(train_ctx, eval_ctx, cfg, feature_mode=feature_mode, baseline_model=baseline_model, force=False)
            rows = result["rows"]
            all_rows.extend(rows)
            all_prediction_rows.extend(result.get("prediction_rows") or [])
            if baseline_model == "mlp":
                comparison = _comparison_row(eval_ctx, rows, feature_mode)
                comparison_rows.append(comparison)
                comparison_rows_by_model.setdefault(eval_ctx.model, []).append(comparison)
                if comparison.get("selected_best_fixed_layer") is not None:
                    selected_layers_by_dataset[eval_ctx.resolved_variant] = int(comparison["selected_best_fixed_layer"])
        if len(set(selected_layers_by_dataset.values())) > 1:
            for row in comparison_rows:
                warning = str(row.get("warnings") or "")
                extra = "selected best layer differs greatly across datasets"
                row["warnings"] = f"{warning}; {extra}" if warning else extra
        eval_ctx.eval_dir.mkdir(parents=True, exist_ok=True)
        eval_metrics_csv = eval_ctx.eval_dir / "layer_baseline_metrics.csv"
        eval_metrics_json = eval_ctx.eval_dir / "layer_baseline_metrics.json"
        _write_csv(eval_metrics_csv, all_rows)
        _write_json(eval_metrics_json, all_rows)
        eval_predictions_csv = eval_ctx.eval_dir / "layer_baseline_predictions.csv"
        eval_predictions_json = eval_ctx.eval_dir / "layer_baseline_predictions.json"
        _write_csv(eval_predictions_csv, all_prediction_rows)
        _write_json(eval_predictions_json, all_prediction_rows)
        eval_ctx.analysis_dir.mkdir(parents=True, exist_ok=True)
        summary_csv = eval_ctx.analysis_dir / "layer_baseline_summary.csv"
        summary_json = eval_ctx.analysis_dir / "layer_baseline_summary.json"
        _write_csv(summary_csv, all_rows)
        _write_json(summary_json, all_rows)
        comparison_csv = eval_ctx.analysis_dir / "layer_vs_diaguq_summary.csv"
        comparison_json = eval_ctx.analysis_dir / "layer_vs_diaguq_summary.json"
        _write_csv(comparison_csv, comparison_rows)
        _write_json(comparison_json, comparison_rows)
        results.append({"dataset": eval_ctx.resolved_variant, "model": eval_ctx.model, "rows": len(all_rows), "metrics_csv": str(eval_metrics_csv), "comparison_csv": str(comparison_csv)})
    for model, rows in comparison_rows_by_model.items():
        if rows:
            write_cross_dataset_layer_summary(rows, output_root=cfg_template.output_root, model=model)
    return results
