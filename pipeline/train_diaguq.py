"""Training entry-point for the DiagUQ network.

This script is intentionally independent from the baseline
(random-forest) estimator under ``baseline/``. It glues together:

* :func:`features.load_feature_tensors.load_diaguq_dataset` -- multi-view features
* :func:`pipeline.build_diagnostic_targets.dimension_targets_dir` -- diagnostic targets
* :class:`pipeline.diaguq_model.DiagUQModel`                -- the DiagUQ network

and writes checkpoints + per-epoch metrics under

    ./test_output/<dataset>/<model>/diaguq/checkpoints/

so multiple (dataset, model) pairs train in isolation.

The public ``DiagUQTrainConfig`` / ``train_diaguq`` names are the
canonical entry points; the historical ``MDUQTrainConfig`` /
``train_mduq`` names are kept as internal aliases for backward
compatibility.
"""

from __future__ import annotations

import json
import math
import os
import csv
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from common.runtime_paths import get_test_output_dir
from common.artifact_manifest import write_stage_manifest
from common.feature_validation import validate_view_bundle
from common.pair_context import (
    DiagUQPairContext,
    assert_diaguq_output_path,
    resolve_pair_context,
)
from pipeline.diaguq_model import DEFAULT_DIMENSIONS, MDUQModel
from features.load_feature_tensors import load_mduq_dataset
from pipeline.build_diagnostic_targets import (
    DEFAULT_TARGET_NAMES,
    dimension_targets_dir,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MDUQTrainConfig:
    dataset_name: str
    model_name: str
    output_root: str = field(default_factory=lambda: str(get_test_output_dir()))

    # data
    layer_list: Optional[Sequence[int]] = None
    query_kind: str = "average"
    answer_kind: str = "average"
    val_fraction: float = 0.1
    seed: int = 42

    # model
    fusion_dim: int = 256
    fusion_hidden_dim: int = 512
    head_hidden_dim: int = 128
    overall_hidden_dim: int = 128
    dropout: float = 0.1
    layer_softmax_temperature: float = 1.5
    layer_dropout: float = 0.05
    layer_entropy_weight: float = 0.0
    layer_temperature: Optional[float] = None
    layer_residual_uniform_alpha: float = 0.0
    gate_logit_clip: float = 10.0
    view_gate_hidden_dim: Optional[int] = None
    view_temperature: float = 2.0
    view_temperature_min: float = 0.5
    view_temperature_max: float = 10.0
    residual_uniform_alpha: float = 0.05
    view_norm_clip: Optional[float] = 10.0
    view_entropy_weight: float = 0.01
    view_entropy_warmup_epochs: int = 1
    view_entropy_anneal_to: float = 0.002
    view_dropout_prob: float = 0.1
    view_gate_scope: str = "shared"
    view_fusion_mode: str = "dimension_specific"
    diagnostic_factorization_mode: str = "shared_plus_residual"
    dimension_corr_regularization_weight: float = 0.01
    dimension_corr_margin: float = 0.05
    residual_diversity_weight: float = 0.005
    residual_diversity_margin: float = 0.1
    overall_aggregation_mode: str = "hybrid"
    dimension_names: Sequence[str] = field(default_factory=lambda: DEFAULT_DIMENSIONS)

    # training
    batch_size: int = 64
    num_epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    overall_loss_weight: float = 1.0
    dimension_loss_weight: float = 1.0
    ambiguity_loss_weight: float = 1.0
    knowledge_gap_loss_weight: float = 1.0
    predictive_variability_loss_weight: float = 1.0
    gold_target_loss_multiplier: float = 1.0
    dataset_grounded_target_loss_multiplier: float = 1.0
    proxy_target_loss_multiplier: float = 0.7
    unavailable_target_loss_multiplier: float = 0.0
    proxy_target_loss_weight_multiplier: float = 0.7
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------


class _MDUQTensorDataset(Dataset):
    def __init__(
        self,
        views: Mapping[str, torch.Tensor],
        dimension_targets: torch.Tensor,
        overall_target: torch.Tensor,
        entropy: Optional[torch.Tensor] = None,
        dimension_target_status: Optional[torch.Tensor] = None,
        dimension_target_reliability: Optional[torch.Tensor] = None,
        dimension_target_source_code: Optional[torch.Tensor] = None,
        dimension_target_source_names: Optional[Sequence[Sequence[str]]] = None,
    ):
        self.view_names: List[str] = list(views.keys())
        self.views = {name: v.float() for name, v in views.items()}
        self.dimension_targets = dimension_targets.float()
        self.overall_target = overall_target.float()
        self.entropy = entropy.float() if entropy is not None else None
        self.dimension_target_status = dimension_target_status.long() if dimension_target_status is not None else None
        self.dimension_target_reliability = dimension_target_reliability.float() if dimension_target_reliability is not None else None
        self.dimension_target_source_code = dimension_target_source_code.long() if dimension_target_source_code is not None else None
        self.dimension_target_source_names = [list(values) for values in (dimension_target_source_names or [])]

        n = self.dimension_targets.shape[0]
        for name, v in self.views.items():
            if v.shape[0] != n:
                raise ValueError(
                    f"view {name!r} has N={v.shape[0]}, expected {n}"
                )
        if self.overall_target.shape[0] != n:
            raise ValueError("overall_target N mismatch")
        if self.entropy is not None and self.entropy.shape[0] != n:
            raise ValueError("entropy N mismatch")
        if self.dimension_target_status is not None and self.dimension_target_status.shape != self.dimension_targets.shape:
            raise ValueError("dimension_target_status shape mismatch")
        if self.dimension_target_reliability is not None and self.dimension_target_reliability.shape != self.dimension_targets.shape:
            raise ValueError("dimension_target_reliability shape mismatch")
        if self.dimension_target_source_code is not None and self.dimension_target_source_code.shape != self.dimension_targets.shape:
            raise ValueError("dimension_target_source_code shape mismatch")

    def __len__(self) -> int:
        return self.dimension_targets.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item: Dict[str, torch.Tensor] = {
            f"view::{name}": v[idx] for name, v in self.views.items()
        }
        item["dimension_targets"] = self.dimension_targets[idx]
        item["overall_target"] = self.overall_target[idx]
        if self.dimension_target_status is not None:
            item["dimension_target_status"] = self.dimension_target_status[idx]
        if self.dimension_target_reliability is not None:
            item["dimension_target_reliability"] = self.dimension_target_reliability[idx]
        if self.dimension_target_source_code is not None:
            item["dimension_target_source_code"] = self.dimension_target_source_code[idx]
        if self.entropy is not None:
            item["entropy"] = self.entropy[idx]
        return item


def _collate_batch(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    keys = batch[0].keys()
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def _split_views(
    batch: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    views: Dict[str, torch.Tensor] = {}
    rest: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if k.startswith("view::"):
            views[k[len("view::") :]] = v
        else:
            rest[k] = v
    return views, rest


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _masked_mse(
    pred: torch.Tensor, target: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (loss, count) ignoring NaN target entries."""
    mask = torch.isfinite(target)
    if not mask.any():
        return pred.new_zeros(()), pred.new_zeros((), dtype=torch.long)
    diff = (pred[mask] - target[mask]) ** 2
    return diff.mean(), torch.tensor(mask.sum().item(), device=pred.device)


def _masked_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = torch.isfinite(target)
    if sample_weight is not None:
        mask = mask & torch.isfinite(sample_weight) & (sample_weight > 0)
    if not mask.any():
        return pred.new_zeros(()), pred.new_zeros((), dtype=torch.long)
    diff = (pred[mask] - target[mask]) ** 2
    if sample_weight is None:
        return diff.mean(), torch.tensor(mask.sum().item(), device=pred.device)
    weights = sample_weight[mask].to(pred.dtype)
    loss = (diff * weights).sum() / torch.tensor(mask.sum().item(), device=pred.device, dtype=pred.dtype).clamp_min(1.0)
    return loss, torch.tensor(mask.sum().item(), device=pred.device)


def _corrcoef_1d(left: torch.Tensor, right: torch.Tensor) -> Optional[torch.Tensor]:
    left = left.float()
    right = right.float()
    if left.numel() < 4:
        return None
    left = left - left.mean()
    right = right - right.mean()
    denom = torch.sqrt((left * left).sum() * (right * right).sum()).clamp_min(1e-12)
    if float(denom.detach().cpu().item()) <= 1e-10:
        return None
    return (left * right).sum() / denom


def _dimension_excess_corr_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    num_dims = int(predictions.shape[1])
    for left_idx in range(num_dims):
        for right_idx in range(left_idx + 1, num_dims):
            mask = (
                torch.isfinite(targets[:, left_idx])
                & torch.isfinite(targets[:, right_idx])
                & torch.isfinite(predictions[:, left_idx])
                & torch.isfinite(predictions[:, right_idx])
            )
            if int(mask.sum().item()) < 4:
                continue
            pred_corr = _corrcoef_1d(predictions[mask, left_idx], predictions[mask, right_idx])
            target_corr = _corrcoef_1d(targets[mask, left_idx], targets[mask, right_idx])
            if pred_corr is None or target_corr is None:
                continue
            excess = F.relu(torch.abs(pred_corr) - torch.abs(target_corr.detach()) - float(margin))
            losses.append(excess * excess)
    if not losses:
        return predictions.new_zeros(())
    return torch.stack(losses).mean()


def _residual_diversity_loss(
    residual_hiddens: Mapping[str, torch.Tensor],
    dimension_names: Sequence[str],
    *,
    margin: float,
) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    names = [name for name in dimension_names if name in residual_hiddens]
    for left_idx, left_name in enumerate(names):
        for right_name in names[left_idx + 1:]:
            left = residual_hiddens[left_name]
            right = residual_hiddens[right_name]
            if left.shape != right.shape or left.numel() == 0:
                continue
            cosine = F.cosine_similarity(left.float(), right.float(), dim=-1).abs().mean()
            losses.append(F.relu(cosine - float(margin)) ** 2)
    if not losses:
        if residual_hiddens:
            return next(iter(residual_hiddens.values())).new_zeros(())
        return torch.zeros(())
    return torch.stack(losses).mean()


def _binary_aggregate(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> Dict[str, float]:
    mask = torch.isfinite(target)
    if not mask.any():
        return {"acc": float("nan"), "support": 0}
    p = (pred[mask] >= threshold).float()
    t = (target[mask] >= threshold).float()
    return {
        "acc": float((p == t).float().mean().item()),
        "support": int(mask.sum().item()),
    }


def _masked_mean_std(values: torch.Tensor, mask: torch.Tensor) -> Tuple[float, float]:
    if not mask.any():
        return float("nan"), float("nan")
    selected = values[mask].detach().float().cpu()
    std = selected.std(unbiased=False).item() if selected.numel() > 1 else 0.0
    return float(selected.mean().item()), float(std)


def _masked_corr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mask = torch.isfinite(pred) & torch.isfinite(target)
    if int(mask.sum().item()) < 2:
        return float("nan")
    x = pred[mask].detach().float().cpu()
    y = target[mask].detach().float().cpu()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x * x).sum() * (y * y).sum())
    if denom.item() <= 1e-12:
        return float("nan")
    return float(((x * y).sum() / denom).item())


def _dimension_loss_weight(cfg: MDUQTrainConfig, name: str) -> float:
    if name == "ambiguity":
        return float(cfg.ambiguity_loss_weight)
    if name == "knowledge_gap":
        return float(cfg.knowledge_gap_loss_weight)
    if name == "predictive_variability":
        return float(cfg.predictive_variability_loss_weight)
    return 1.0


def _layer_weight_entropy(layer_weights: Mapping[str, torch.Tensor]) -> torch.Tensor:
    entropies: List[torch.Tensor] = []
    for name, weights in layer_weights.items():
        if name.startswith("_") or weights.dim() != 2:
            continue
        entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=-1).mean()
        entropies.append(entropy)
    if not entropies:
        return torch.zeros((), device=next(iter(layer_weights.values())).device)
    return torch.stack(entropies).mean()


def _view_weight_items(layer_weights: Mapping[str, torch.Tensor]) -> List[Tuple[str, torch.Tensor]]:
    keys = [key for key in layer_weights if key.startswith("_view_weights") and "raw" not in key]
    if any(key.startswith("_view_weights_overall") for key in keys):
        keys = [key for key in keys if key != "_view_weights"]
    out: List[Tuple[str, torch.Tensor]] = []
    for key in keys:
        tensor = layer_weights[key]
        if isinstance(tensor, torch.Tensor) and tensor.dim() == 2 and tensor.shape[-1] > 1:
            label = key.replace("_view_weights_", "") if key != "_view_weights" else "overall"
            out.append((label, tensor))
    return out


def _normalized_view_entropy(weights: torch.Tensor) -> torch.Tensor:
    entropy = -(weights.clamp_min(1e-8) * torch.log(weights.clamp_min(1e-8))).sum(dim=-1)
    return entropy / math.log(float(weights.shape[-1]))


def _view_entropy_stats(layer_weights: Mapping[str, torch.Tensor]) -> Dict[str, object]:
    items = _view_weight_items(layer_weights)
    if not items:
        return {
            "loss": torch.zeros((), device=next(iter(layer_weights.values())).device),
            "mean_entropy": 0.0,
            "mean_max_weight": 1.0,
            "collapse_rate_gt_095": 1.0,
            "per_target": {},
        }
    losses: List[torch.Tensor] = []
    ent_values: List[torch.Tensor] = []
    max_values: List[torch.Tensor] = []
    per_target: Dict[str, Dict[str, float]] = {}
    for label, weights in items:
        ent = _normalized_view_entropy(weights)
        max_w = weights.max(dim=-1).values
        losses.append(-ent.mean())
        ent_values.append(ent.detach())
        max_values.append(max_w.detach())
        per_target[label] = {
            "mean_entropy": float(ent.detach().mean().cpu().item()),
            "mean_max_weight": float(max_w.detach().mean().cpu().item()),
            "collapse_rate_gt_095": float((max_w.detach() > 0.95).float().mean().cpu().item()),
        }
    all_entropy = torch.cat([value.reshape(-1) for value in ent_values])
    all_max = torch.cat([value.reshape(-1) for value in max_values])
    return {
        "loss": torch.stack(losses).mean(),
        "mean_entropy": float(all_entropy.mean().cpu().item()),
        "mean_max_weight": float(all_max.mean().cpu().item()),
        "collapse_rate_gt_095": float((all_max > 0.95).float().mean().cpu().item()),
        "per_target": per_target,
    }


def _view_entropy_weight_for_epoch(cfg: MDUQTrainConfig, epoch: int) -> float:
    initial = float(cfg.view_entropy_weight)
    target = float(cfg.view_entropy_anneal_to)
    if initial <= 0:
        return 0.0
    warmup = max(0, int(cfg.view_entropy_warmup_epochs))
    if epoch <= max(warmup, 1):
        return initial
    total = max(int(cfg.num_epochs) - max(warmup, 1), 1)
    progress = min(max((epoch - max(warmup, 1)) / total, 0.0), 1.0)
    return float(initial * (1.0 - progress) + target * progress)


_TARGET_STATUS_CODE_TO_LABEL = {
    3: "gold",
    2: "dataset_grounded",
    1: "proxy",
    0: "unavailable",
    -1: "missing",
    -2: "masked",
}


def _status_multiplier_tensor(status_codes: torch.Tensor, cfg: MDUQTrainConfig, dtype: torch.dtype) -> torch.Tensor:
    weights = torch.zeros_like(status_codes, dtype=dtype)
    weights = torch.where(status_codes == 3, torch.full_like(weights, float(cfg.gold_target_loss_multiplier)), weights)
    weights = torch.where(status_codes == 2, torch.full_like(weights, float(cfg.dataset_grounded_target_loss_multiplier)), weights)
    proxy_multiplier = float(getattr(cfg, "proxy_target_loss_multiplier", cfg.proxy_target_loss_weight_multiplier))
    weights = torch.where(status_codes == 1, torch.full_like(weights, proxy_multiplier), weights)
    unavailable_multiplier = float(cfg.unavailable_target_loss_multiplier)
    weights = torch.where(status_codes <= 0, torch.full_like(weights, unavailable_multiplier), weights)
    return weights


def _source_vocab_from_loader(loader: DataLoader) -> List[List[str]]:
    dataset = loader.dataset
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    vocab = getattr(base, "dimension_target_source_names", None)
    return [list(values) for values in vocab] if vocab else []


# ---------------------------------------------------------------------------
# Build dataset bundle
# ---------------------------------------------------------------------------


def _resolve_overall_target(
    payload: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    overall = payload.get("overall_target")
    if overall is None:
        raise KeyError("dimension_targets.pt missing 'overall_target'")
    fallback = payload.get("overall_ask4conf_target")
    if fallback is not None:
        nan = torch.isnan(overall)
        if nan.any():
            overall = overall.clone()
            overall[nan] = fallback[nan]
    return overall


def _select_dimension_targets(
    payload: Mapping[str, torch.Tensor],
    dimension_names: Sequence[str],
) -> torch.Tensor:
    cols: List[torch.Tensor] = []
    for name in dimension_names:
        # accept either bare name (e.g. "ambiguity") or the JSON suffix
        # ("ambiguity_target") used by generate_dimension_targets.
        key_candidates = [f"{name}_target", name]
        for k in key_candidates:
            if k in payload:
                cols.append(payload[k].float())
                break
        else:
            raise KeyError(
                f"dimension_targets.pt missing target for {name!r}; "
                f"tried {key_candidates}, available keys: "
                f"{[k for k in payload if isinstance(payload[k], torch.Tensor)]}"
            )
    return torch.stack(cols, dim=-1)


def _payload_get_first(payload: Mapping[str, object], *keys: str) -> object:
    for key in keys:
        if key in payload:
            return payload.get(key)
    return None


def _select_dimension_target_status(
    payload: Mapping[str, object],
    dimension_names: Sequence[str],
    n: int,
) -> torch.Tensor:
    status_to_code = {"gold": 3, "dataset_grounded": 2, "proxy": 1, "available": 1, "unavailable": 0, "missing": -1, "masked": -2}
    cols: List[torch.Tensor] = []
    for name in dimension_names:
        values = _payload_get_first(payload, f"{name}_target_status", f"dim_{name}_target_status")
        if isinstance(values, (list, tuple)):
            codes = [status_to_code.get(str(value), 0) for value in values[:n]]
            if len(codes) < n:
                codes.extend([0] * (n - len(codes)))
            cols.append(torch.tensor(codes, dtype=torch.long))
            continue
        available = payload.get(f"{name}_target_available")
        if isinstance(available, torch.Tensor):
            arr = available.reshape(-1)[:n].bool()
            if arr.numel() < n:
                arr = torch.cat([arr, torch.zeros(n - arr.numel(), dtype=torch.bool)])
            cols.append(arr.long())
        else:
            target = payload.get(f"{name}_target")
            if isinstance(target, torch.Tensor):
                arr = torch.isfinite(target.reshape(-1)[:n])
                if arr.numel() < n:
                    arr = torch.cat([arr, torch.zeros(n - arr.numel(), dtype=torch.bool)])
                cols.append(arr.long())
            else:
                cols.append(torch.zeros(n, dtype=torch.long))
    return torch.stack(cols, dim=-1)


def _select_dimension_target_reliability(
    payload: Mapping[str, object],
    dimension_names: Sequence[str],
    n: int,
) -> torch.Tensor:
    cols: List[torch.Tensor] = []
    for name in dimension_names:
        values = _payload_get_first(payload, f"{name}_target_reliability", f"dim_{name}_target_reliability")
        if isinstance(values, torch.Tensor):
            arr = values.reshape(-1)[:n].float()
            if arr.numel() < n:
                arr = torch.cat([arr, torch.zeros(n - arr.numel(), dtype=torch.float32)])
            cols.append(arr)
            continue
        if isinstance(values, (list, tuple)):
            arr = torch.tensor([float(value) if _is_finite_number(value) else 0.0 for value in values[:n]], dtype=torch.float32)
            if arr.numel() < n:
                arr = torch.cat([arr, torch.zeros(n - arr.numel(), dtype=torch.float32)])
            cols.append(arr)
            continue
        status_values = _payload_get_first(payload, f"{name}_target_status", f"dim_{name}_target_status")
        if isinstance(status_values, (list, tuple)):
            arr = torch.tensor([
                1.0 if str(value) in {"gold", "dataset_grounded", "proxy", "available"} else 0.0
                for value in status_values[:n]
            ], dtype=torch.float32)
            if arr.numel() < n:
                arr = torch.cat([arr, torch.zeros(n - arr.numel(), dtype=torch.float32)])
            cols.append(arr)
        else:
            cols.append(torch.ones(n, dtype=torch.float32))
    return torch.stack(cols, dim=-1)


def _is_finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _select_dimension_target_source_codes(
    payload: Mapping[str, object],
    dimension_names: Sequence[str],
    n: int,
) -> Tuple[torch.Tensor, List[List[str]]]:
    cols: List[torch.Tensor] = []
    vocab_by_dim: List[List[str]] = []
    for name in dimension_names:
        values = _payload_get_first(payload, f"{name}_target_source", f"dim_{name}_target_source")
        strings = ["missing"] * n
        if isinstance(values, (list, tuple)):
            strings = [str(value) if value else "missing" for value in values[:n]]
            if len(strings) < n:
                strings.extend(["missing"] * (n - len(strings)))
        vocab = sorted(set(strings)) or ["missing"]
        code_by_value = {value: idx for idx, value in enumerate(vocab)}
        cols.append(torch.tensor([code_by_value[value] for value in strings], dtype=torch.long))
        vocab_by_dim.append(vocab)
    return torch.stack(cols, dim=-1), vocab_by_dim


def _build_dataset(
    cfg: MDUQTrainConfig,
    pair_context: Optional[DiagUQPairContext] = None,
) -> Tuple[_MDUQTensorDataset, Dict[str, int]]:
    ctx = pair_context or resolve_pair_context(cfg.dataset_name, cfg.model_name, runtime_root=cfg.output_root)
    bundle = load_mduq_dataset(
        ctx.resolved_variant,
        ctx.model,
        layer_list=cfg.layer_list,
        output_root=cfg.output_root,
        query_kind=cfg.query_kind,
        answer_kind=cfg.answer_kind,
        bank_dir=str(ctx.hidden_bank_dir),
    )
    views = bundle["views"]
    if "query" not in views or "answer" not in views or "relation" not in views:
        raise RuntimeError(
            "MDUQ feature bundle missing required views; got "
            f"{list(views.keys())}"
        )
    entropy = views.get("entropy")
    core_views = {k: views[k] for k in ("query", "answer", "relation")}

    target_dir = ctx.dimension_targets_dir
    assert_diaguq_output_path(ctx, target_dir, stage_token="dimension_targets")
    target_pt = Path(target_dir) / "dimension_targets.pt"
    if not target_pt.is_file():
        raise FileNotFoundError(
            f"dimension targets not found at {target_pt}; "
            "run `python run.py generate-dim-targets ...` first."
        )
    payload = torch.load(target_pt, map_location="cpu")

    dimension_targets = _select_dimension_targets(payload, cfg.dimension_names)
    overall_target = _resolve_overall_target(payload)
    dimension_target_status = _select_dimension_target_status(payload, cfg.dimension_names, int(dimension_targets.shape[0]))
    dimension_target_reliability = _select_dimension_target_reliability(payload, cfg.dimension_names, int(dimension_targets.shape[0]))
    dimension_target_source_code, dimension_target_source_names = _select_dimension_target_source_codes(payload, cfg.dimension_names, int(dimension_targets.shape[0]))

    n_views = next(iter(core_views.values())).shape[0]
    n_targets = dimension_targets.shape[0]
    if n_views != n_targets:
        n = min(n_views, n_targets)
        for k in list(core_views.keys()):
            core_views[k] = core_views[k][:n]
        dimension_targets = dimension_targets[:n]
        dimension_target_status = dimension_target_status[:n]
        dimension_target_reliability = dimension_target_reliability[:n]
        dimension_target_source_code = dimension_target_source_code[:n]
        overall_target = overall_target[:n]
        if entropy is not None:
            entropy = entropy[:n]

    view_dims = {k: int(v.shape[-1]) for k, v in core_views.items()}
    feature_sanity = validate_view_bundle(core_views, entropy=entropy)
    if feature_sanity["status"] != "success":
        raise ValueError(f"feature validation failed: {feature_sanity['failures']}")
    _validate_training_targets(
        dimension_targets,
        overall_target,
        dimension_names=cfg.dimension_names,
    )
    dataset = _MDUQTensorDataset(
        core_views,
        dimension_targets=dimension_targets,
        overall_target=overall_target,
        entropy=entropy,
        dimension_target_status=dimension_target_status,
        dimension_target_reliability=dimension_target_reliability,
        dimension_target_source_code=dimension_target_source_code,
        dimension_target_source_names=dimension_target_source_names,
    )
    return dataset, view_dims


def _validate_training_targets(
    dimension_targets: torch.Tensor,
    overall_target: torch.Tensor,
    *,
    dimension_names: Sequence[str],
) -> None:
    failures: List[str] = []
    if not torch.isfinite(overall_target).any():
        failures.append("overall_target has zero finite labels")
    finite_overall = overall_target[torch.isfinite(overall_target)]
    if finite_overall.numel() and (
        finite_overall.min().item() < -1e-6 or finite_overall.max().item() > 1.0 + 1e-6
    ):
        failures.append("overall_target has values outside [0,1]")

    for idx, name in enumerate(dimension_names):
        col = dimension_targets[:, idx]
        finite = col[torch.isfinite(col)]
        if finite.numel() == 0:
            continue
        if finite.min().item() < -1e-6 or finite.max().item() > 1.0 + 1e-6:
            failures.append(f"{name}_target has values outside [0,1]")
    if torch.isinf(dimension_targets).any() or torch.isinf(overall_target).any():
        failures.append("targets contain Inf values")
    if failures:
        raise ValueError("; ".join(failures))


def _train_val_split(
    dataset: _MDUQTensorDataset, val_fraction: float, seed: int
) -> Tuple[Subset, Subset]:
    n = len(dataset)
    if n < 2:
        raise RuntimeError(f"need at least 2 examples to split, got {n}")
    n_val = max(1, int(round(n * float(val_fraction))))
    n_val = min(n_val, n - 1)
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


# ---------------------------------------------------------------------------
# Train / evaluate loops
# ---------------------------------------------------------------------------


def _run_epoch(
    model: MDUQModel,
    loader: DataLoader,
    *,
    optimizer: Optional[torch.optim.Optimizer],
    cfg: MDUQTrainConfig,
    device: torch.device,
    epoch: int = 1,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_dim_loss = 0.0
    total_overall_loss = 0.0
    total_layer_entropy = 0.0
    total_view_entropy_loss = 0.0
    total_view_entropy = 0.0
    total_view_max_weight = 0.0
    total_view_collapse_rate = 0.0
    total_corr_regularization = 0.0
    total_residual_diversity = 0.0
    n_batches = 0
    overall_acc_num = 0.0
    overall_acc_den = 0
    per_dim_acc_num = [0.0] * len(cfg.dimension_names)
    per_dim_acc_den = [0] * len(cfg.dimension_names)
    per_dim_loss_sum = [0.0] * len(cfg.dimension_names)
    per_dim_loss_batches = [0] * len(cfg.dimension_names)
    pred_chunks: List[List[torch.Tensor]] = [[] for _ in cfg.dimension_names]
    target_chunks: List[List[torch.Tensor]] = [[] for _ in cfg.dimension_names]
    overall_pred_chunks: List[torch.Tensor] = []
    overall_target_chunks: List[torch.Tensor] = []
    view_dropout_counts: Dict[str, float] = {}
    source_vocab = _source_vocab_from_loader(loader)
    status_loss_sum: Dict[str, float] = {}
    status_loss_count: Dict[str, int] = {}
    status_sample_count: Dict[str, int] = {}
    source_loss_sum: Dict[str, float] = {}
    source_loss_count: Dict[str, int] = {}
    source_sample_count: Dict[str, int] = {}

    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
    with grad_ctx:
        for batch in loader:
            views, rest = _split_views(batch)
            views = {k: v.to(device) for k, v in views.items()}
            entropy = rest.get("entropy")
            if entropy is not None:
                entropy = entropy.to(device)
            dim_target = rest["dimension_targets"].to(device)
            overall_target = rest["overall_target"].to(device)
            dim_target_status = rest.get("dimension_target_status")
            if dim_target_status is not None:
                dim_target_status = dim_target_status.to(device)
            dim_target_reliability = rest.get("dimension_target_reliability")
            if dim_target_reliability is not None:
                dim_target_reliability = dim_target_reliability.to(device)
            dim_target_source_code = rest.get("dimension_target_source_code")
            if dim_target_source_code is not None:
                dim_target_source_code = dim_target_source_code.to(device)

            for view_name, tensor in views.items():
                _require_finite(f"input view {view_name}", tensor)
            if entropy is not None:
                _require_finite("entropy input", entropy)
            if torch.isinf(dim_target).any() or torch.isinf(overall_target).any():
                raise ValueError("target tensor contains Inf")

            out = model(views, entropy=entropy)
            _require_finite("model uncertainty", out.uncertainty)
            _require_finite("model confidence", out.confidence)
            _require_finite("model dimension_scores", out.dimension_scores)
            for weight_name, weight in out.layer_weights.items():
                _require_finite(f"layer weight {weight_name}", weight)

            head_losses: List[torch.Tensor] = []
            head_weights: List[float] = []
            for i, name in enumerate(cfg.dimension_names):
                sample_weight = None
                if dim_target_status is not None:
                    sample_weight = _status_multiplier_tensor(dim_target_status[:, i], cfg, dim_target[:, i].dtype)
                    if dim_target_reliability is not None:
                        sample_weight = sample_weight * dim_target_reliability[:, i].to(dim_target[:, i].dtype).clamp(0.0, 1.0)
                loss_i, count_i = _masked_weighted_mse(out.dimension_scores[:, i], dim_target[:, i], sample_weight)
                finite_mask = torch.isfinite(dim_target[:, i])
                diff_i = (out.dimension_scores[:, i] - dim_target[:, i]) ** 2
                if dim_target_status is not None:
                    status_values = dim_target_status[:, i]
                    for code, label in _TARGET_STATUS_CODE_TO_LABEL.items():
                        status_mask = finite_mask & (status_values == code)
                        count = int(status_mask.sum().item())
                        if count <= 0:
                            continue
                        status_sample_count[label] = status_sample_count.get(label, 0) + count
                        if sample_weight is not None:
                            active_mask = status_mask & torch.isfinite(sample_weight) & (sample_weight > 0)
                        else:
                            active_mask = status_mask
                        active_count = int(active_mask.sum().item())
                        if active_count > 0:
                            status_loss_sum[label] = status_loss_sum.get(label, 0.0) + float(diff_i[active_mask].detach().mean().cpu().item())
                            status_loss_count[label] = status_loss_count.get(label, 0) + 1
                if dim_target_source_code is not None:
                    source_values = dim_target_source_code[:, i]
                    vocab = source_vocab[i] if i < len(source_vocab) else []
                    for code in torch.unique(source_values.detach()).cpu().tolist():
                        source_mask = finite_mask & (source_values == int(code))
                        count = int(source_mask.sum().item())
                        if count <= 0:
                            continue
                        source_label = vocab[int(code)] if int(code) < len(vocab) else f"source_{int(code)}"
                        source_key = f"{name}::{source_label}"
                        source_sample_count[source_key] = source_sample_count.get(source_key, 0) + count
                        if sample_weight is not None:
                            active_mask = source_mask & torch.isfinite(sample_weight) & (sample_weight > 0)
                        else:
                            active_mask = source_mask
                        active_count = int(active_mask.sum().item())
                        if active_count > 0:
                            source_loss_sum[source_key] = source_loss_sum.get(source_key, 0.0) + float(diff_i[active_mask].detach().mean().cpu().item())
                            source_loss_count[source_key] = source_loss_count.get(source_key, 0) + 1
                if int(count_i.item()) > 0:
                    weight_i = _dimension_loss_weight(cfg, name)
                    if weight_i > 0:
                        head_losses.append(loss_i * weight_i)
                        head_weights.append(weight_i)
                    per_dim_loss_sum[i] += float(loss_i.item())
                    per_dim_loss_batches[i] += 1
                pred_chunks[i].append(out.dimension_scores[:, i].detach().cpu())
                target_chunks[i].append(dim_target[:, i].detach().cpu())
            if head_losses:
                dim_loss = torch.stack(head_losses).sum() / max(sum(head_weights), 1e-12)
            else:
                dim_loss = out.dimension_scores.new_zeros(())
            overall_loss, _ = _masked_mse(out.uncertainty, overall_target)
            corr_regularization = _dimension_excess_corr_loss(
                out.dimension_scores,
                dim_target,
                margin=float(cfg.dimension_corr_margin),
            )
            residual_hiddens = {
                name: out.diagnostic_components[f"residual_{name}_hidden"]
                for name in cfg.dimension_names
                if f"residual_{name}_hidden" in out.diagnostic_components
            }
            residual_diversity = _residual_diversity_loss(
                residual_hiddens,
                cfg.dimension_names,
                margin=float(cfg.residual_diversity_margin),
            ).to(out.dimension_scores.device)
            layer_entropy = _layer_weight_entropy(out.layer_weights)
            view_entropy_report = _view_entropy_stats(out.layer_weights)
            view_entropy_loss = view_entropy_report["loss"]
            view_entropy_weight = _view_entropy_weight_for_epoch(cfg, epoch) if is_train else 0.0
            loss = (
                cfg.dimension_loss_weight * dim_loss
                + cfg.overall_loss_weight * overall_loss
                - float(cfg.layer_entropy_weight) * layer_entropy
                + float(view_entropy_weight) * view_entropy_loss
                + (float(cfg.dimension_corr_regularization_weight) if is_train else 0.0) * corr_regularization
                + (float(cfg.residual_diversity_weight) if is_train else 0.0) * residual_diversity
            )
            _require_finite("training loss", loss)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                _require_finite_gradients(model)
                if cfg.grad_clip and cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.grad_clip
                    )
                optimizer.step()

            total_loss += float(loss.item())
            total_dim_loss += float(dim_loss.item())
            total_overall_loss += float(overall_loss.item())
            total_layer_entropy += float(layer_entropy.item())
            total_view_entropy_loss += float(view_entropy_loss.detach().cpu().item())
            total_view_entropy += float(view_entropy_report["mean_entropy"])
            total_view_max_weight += float(view_entropy_report["mean_max_weight"])
            total_view_collapse_rate += float(view_entropy_report["collapse_rate_gt_095"])
            total_corr_regularization += float(corr_regularization.detach().cpu().item())
            total_residual_diversity += float(residual_diversity.detach().cpu().item())
            n_batches += 1
            for key, keep_mask in out.gate_logits.items():
                if not key.startswith("_view_dropout_keep") or not isinstance(keep_mask, torch.Tensor):
                    continue
                dropped = (keep_mask.detach().cpu().float() < 0.5).sum(dim=0)
                for idx, count in enumerate(dropped.tolist()):
                    view_name = model.view_names[idx] if idx < len(model.view_names) else f"view_{idx}"
                    metric_key = f"view_dropout_dropped_{view_name}"
                    view_dropout_counts[metric_key] = view_dropout_counts.get(metric_key, 0.0) + float(count)
            overall_pred_chunks.append(out.uncertainty.detach().cpu())
            overall_target_chunks.append(overall_target.detach().cpu())

            stats = _binary_aggregate(out.uncertainty, overall_target)
            if stats["support"]:
                overall_acc_num += stats["acc"] * stats["support"]
                overall_acc_den += stats["support"]
            for i in range(len(cfg.dimension_names)):
                stats_i = _binary_aggregate(
                    out.dimension_scores[:, i], dim_target[:, i]
                )
                if stats_i["support"]:
                    per_dim_acc_num[i] += stats_i["acc"] * stats_i["support"]
                    per_dim_acc_den[i] += stats_i["support"]

    n_batches = max(n_batches, 1)
    metrics: Dict[str, float] = {
        "loss": total_loss / n_batches,
        "loss_dimension": total_dim_loss / n_batches,
        "loss_overall": total_overall_loss / n_batches,
        "layer_weight_entropy": total_layer_entropy / n_batches,
        "view_entropy_loss": total_view_entropy_loss / n_batches,
        "view_entropy_weight": _view_entropy_weight_for_epoch(cfg, epoch) if is_train else 0.0,
        "view_entropy": total_view_entropy / n_batches,
        "view_max_weight": total_view_max_weight / n_batches,
        "view_collapse_rate_gt_095": total_view_collapse_rate / n_batches,
        "loss_corr_regularization": total_corr_regularization / n_batches,
        "loss_residual_diversity": total_residual_diversity / n_batches,
        "acc_overall": (
            overall_acc_num / overall_acc_den
            if overall_acc_den
            else float("nan")
        ),
    }
    metrics.update(view_dropout_counts)
    for label, count in sorted(status_sample_count.items()):
        metrics[f"target_status_count_{label}"] = count
    for label, total in sorted(status_loss_sum.items()):
        denom = max(status_loss_count.get(label, 0), 1)
        metrics[f"loss_by_status_{label}"] = total / denom
    for label, count in sorted(source_sample_count.items()):
        safe_label = label.replace("::", "__").replace("/", "_").replace(" ", "_")
        metrics[f"target_source_count_{safe_label}"] = count
    for label, total in sorted(source_loss_sum.items()):
        safe_label = label.replace("::", "__").replace("/", "_").replace(" ", "_")
        denom = max(source_loss_count.get(label, 0), 1)
        metrics[f"loss_by_source_{safe_label}"] = total / denom
    if overall_pred_chunks:
        overall_pred = torch.cat(overall_pred_chunks)
        overall_target_all = torch.cat(overall_target_chunks)
        overall_mask = torch.isfinite(overall_target_all)
        metrics["overall_pred_mean"], metrics["overall_pred_std"] = _masked_mean_std(overall_pred, overall_mask)
        metrics["overall_target_mean"], metrics["overall_target_std"] = _masked_mean_std(overall_target_all, overall_mask)
        metrics["corr_overall"] = _masked_corr(overall_pred, overall_target_all)
    for i, name in enumerate(cfg.dimension_names):
        metrics[f"acc_{name}"] = (
            per_dim_acc_num[i] / per_dim_acc_den[i]
            if per_dim_acc_den[i]
            else float("nan")
        )
        metrics[f"loss_{name}"] = (
            per_dim_loss_sum[i] / per_dim_loss_batches[i]
            if per_dim_loss_batches[i]
            else float("nan")
        )
        if pred_chunks[i]:
            pred_all = torch.cat(pred_chunks[i])
            target_all = torch.cat(target_chunks[i])
            mask = torch.isfinite(target_all)
            metrics[f"pred_mean_{name}"], metrics[f"pred_std_{name}"] = _masked_mean_std(pred_all, mask)
            metrics[f"target_mean_{name}"], metrics[f"target_std_{name}"] = _masked_mean_std(target_all, mask)
            metrics[f"corr_{name}"] = _masked_corr(pred_all, target_all)
            metrics[f"support_{name}"] = int(mask.sum().item())
    return metrics


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(
            f"{name} contains non-finite values: "
            f"nan={int(torch.isnan(tensor).sum().item())} "
            f"inf={int(torch.isinf(tensor).sum().item())} "
            f"shape={tuple(tensor.shape)}"
        )


def _require_finite_gradients(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            raise ValueError(f"gradient for {name} contains NaN or Inf")


def _validate_model_on_loader(
    model: MDUQModel,
    loader: DataLoader,
    *,
    device: torch.device,
) -> None:
    model.eval()
    with torch.no_grad():
        for batch in loader:
            views, rest = _split_views(batch)
            views = {k: v.to(device) for k, v in views.items()}
            entropy = rest.get("entropy")
            if entropy is not None:
                entropy = entropy.to(device)
            out = model(views, entropy=entropy)
            _require_finite("validation uncertainty", out.uncertainty)
            _require_finite("validation confidence", out.confidence)
            _require_finite("validation dimension_scores", out.dimension_scores)
            for weight_name, weight in out.layer_weights.items():
                _require_finite(f"validation layer weight {weight_name}", weight)
                if weight_name.startswith("_view_weights") and "raw" not in weight_name:
                    total = weight.sum(dim=-1) if weight.dim() > 1 else weight.sum().reshape(1)
                    if not torch.allclose(total, torch.ones_like(total), atol=1e-4):
                        raise ValueError(f"view weights for {weight_name} do not sum to 1")
                elif weight.dim() == 2:
                    totals = weight.sum(dim=-1)
                    if not torch.allclose(totals, torch.ones_like(totals), atol=1e-4):
                        raise ValueError(f"layer weights for {weight_name} do not sum to 1")
            return
    raise RuntimeError("validation loader produced no batches")


def _write_train_log_csv(path: Path, history: Sequence[Mapping[str, object]]) -> None:
    rows: List[Dict[str, object]] = []
    for record in history:
        row: Dict[str, object] = {"epoch": record["epoch"]}
        for prefix in ("train", "val"):
            metrics = record.get(prefix, {})
            if isinstance(metrics, Mapping):
                for key, value in metrics.items():
                    row[f"{prefix}_{key}"] = value
        rows.append(row)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
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
            writer.writerow(row)


def _json_float(value: float) -> Optional[float]:
    return float(value) if math.isfinite(float(value)) else None


def _stats_from_values(values: Sequence[float]) -> Dict[str, Optional[float]]:
    finite = torch.tensor([v for v in values if math.isfinite(float(v))], dtype=torch.float32)
    if finite.numel() == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": _json_float(float(finite.mean().item())),
        "std": _json_float(float(finite.std(unbiased=False).item() if finite.numel() > 1 else 0.0)),
        "min": _json_float(float(finite.min().item())),
        "max": _json_float(float(finite.max().item())),
    }


def _collect_fusion_diagnostics(
    model: MDUQModel,
    loader: DataLoader,
    *,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    feature_norms: Dict[str, List[float]] = {}
    per_layer_norms: Dict[str, List[List[float]]] = {}
    layer_gate_logits: Dict[str, List[torch.Tensor]] = {}
    view_gate_logits: Dict[str, List[torch.Tensor]] = {}
    layer_entropies: Dict[str, List[float]] = {}
    layer_max_weights: Dict[str, List[float]] = {}
    layer_argmax_counts: Dict[str, Dict[int, int]] = {}
    view_weights_by_target: Dict[str, List[torch.Tensor]] = {}
    dropout_counts: Dict[str, List[float]] = {}
    with torch.no_grad():
        for batch in loader:
            views, rest = _split_views(batch)
            views = {k: v.to(device) for k, v in views.items()}
            for view_name, tensor in views.items():
                norms = tensor.norm(dim=-1).detach().cpu()
                feature_norms.setdefault(view_name, []).extend(norms.reshape(-1).tolist())
                means = norms.mean(dim=0).tolist() if norms.dim() == 2 else []
                per_layer_norms.setdefault(view_name, []).append([float(v) for v in means])
            entropy = rest.get("entropy")
            if entropy is not None:
                entropy = entropy.to(device)
            out = model(views, entropy=entropy)
            for name, logits in out.gate_logits.items():
                if name.startswith("_view_dropout_keep"):
                    keep = logits.detach().cpu().float()
                    dropped = (keep < 0.5).sum(dim=0)
                    target = name.replace("_view_dropout_keep_", "")
                    dropout_counts.setdefault(target, [0.0] * dropped.numel())
                    for idx, count in enumerate(dropped.tolist()):
                        dropout_counts[target][idx] += float(count)
                elif name.startswith("_view_logits"):
                    target = name.replace("_view_logits_", "") if name != "_view_logits" else "overall"
                    view_gate_logits.setdefault(target, []).append(logits.detach().cpu().float())
                else:
                    layer_gate_logits.setdefault(name, []).append(logits.detach().cpu().float())
            for name, weights in out.layer_weights.items():
                if name.startswith("_view_weights"):
                    if "raw" in name or (name == "_view_weights" and "_view_weights_overall" in out.layer_weights):
                        continue
                    target = name.replace("_view_weights_", "") if name != "_view_weights" else "overall"
                    view_weights_by_target.setdefault(target, []).append(weights.detach().cpu().float())
                    continue
                if weights.dim() == 2:
                    ent = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=-1)
                    layer_entropies.setdefault(name, []).extend(ent.detach().cpu().tolist())
                    max_w, argmax = weights.detach().cpu().float().max(dim=-1)
                    layer_max_weights.setdefault(name, []).extend(max_w.tolist())
                    counts = layer_argmax_counts.setdefault(name, {})
                    for idx in argmax.tolist():
                        counts[int(idx)] = counts.get(int(idx), 0) + 1
    per_layer_out: Dict[str, List[Dict[str, Optional[float]]]] = {}
    for view_name, chunks in per_layer_norms.items():
        if not chunks:
            continue
        width = max(len(chunk) for chunk in chunks)
        per_layer_out[view_name] = [
            _stats_from_values(chunk[idx] for chunk in chunks if idx < len(chunk))
            for idx in range(width)
        ]
    view_out: Dict[str, object] = {}
    mean_by_target: Dict[str, torch.Tensor] = {}
    for target, chunks in view_weights_by_target.items():
        weights = torch.cat(chunks, dim=0)
        ent = _normalized_view_entropy(weights)
        max_w = weights.max(dim=-1).values
        mean = weights.mean(dim=0)
        mean_by_target[target] = mean
        view_out[target] = {
            "shape": list(weights.shape),
            "mean_by_view": [float(v) for v in mean.tolist()],
            "std_by_view": [float(v) for v in weights.std(dim=0, unbiased=False).tolist()],
            "entropy": _stats_from_values(ent.tolist()),
            "max_weight": _stats_from_values(max_w.tolist()),
            "collapse_rate_gt_095": float((max_w > 0.95).float().mean().item()),
        }
    pairwise: Dict[str, float] = {}
    targets = sorted(mean_by_target)
    for left_idx, left in enumerate(targets):
        for right in targets[left_idx + 1:]:
            pairwise[f"{left}__{right}"] = float(torch.mean(torch.abs(mean_by_target[left] - mean_by_target[right])).item())
    gate_out: Dict[str, object] = {}
    for target, chunks in view_gate_logits.items():
        logits = torch.cat(chunks, dim=0)
        gate_out[target] = {
            view_name: _stats_from_values(logits[:, idx].tolist())
            for idx, view_name in enumerate(model.view_names)
            if idx < logits.shape[1]
        }
    layer_gate_out: Dict[str, object] = {}
    for name, chunks in layer_gate_logits.items():
        logits = torch.cat(chunks, dim=0)
        layer_gate_out[name] = [
            _stats_from_values(logits[:, idx].tolist())
            for idx in range(logits.shape[1])
        ]
    layer_usage = {
        name: {
            "entropy": _stats_from_values(layer_entropies.get(name, [])),
            "max_weight": _stats_from_values(layer_max_weights.get(name, [])),
            "argmax_counts": {str(k): v for k, v in sorted(layer_argmax_counts.get(name, {}).items())},
        }
        for name in sorted(set(layer_entropies) | set(layer_max_weights) | set(layer_argmax_counts))
    }
    return {
        "feature_norms": {name: _stats_from_values(values) for name, values in feature_norms.items()},
        "per_layer_feature_norms": per_layer_out,
        "gate_logits": gate_out,
        "layer_gate_logits": layer_gate_out,
        "layer_weight_entropy": {name: _stats_from_values(values) for name, values in layer_entropies.items()},
        "layer_usage": layer_usage,
        "view_weights": view_out,
        "view_weight_pairwise_l1": pairwise,
        "view_dropout_counts": {
            target: {
                model.view_names[idx] if idx < len(model.view_names) else f"view_{idx}": count
                for idx, count in enumerate(counts)
            }
            for target, counts in dropout_counts.items()
        },
    }


# ---------------------------------------------------------------------------
# Public training entry point
# ---------------------------------------------------------------------------


def checkpoint_dir(
    dataset_name: str, model_name: str, output_root: Optional[str] = None
) -> Path:
    """DiagUQ checkpoint directory for one (dataset, model) pair.

    Writes use the canonical ``diaguq/`` subtree derived from PairContext.
    """
    return resolve_pair_context(dataset_name, model_name, runtime_root=output_root).checkpoint_dir


def train_mduq(cfg: MDUQTrainConfig) -> Dict[str, object]:
    """Train the MDUQ main model for one (dataset, model) pair."""
    device = torch.device(cfg.device)
    ctx = resolve_pair_context(cfg.dataset_name, cfg.model_name, runtime_root=cfg.output_root)
    ckpt_dir = checkpoint_dir(cfg.dataset_name, cfg.model_name, cfg.output_root)
    assert_diaguq_output_path(ctx, ckpt_dir, stage_token="checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    try:
        dataset, view_dims = _build_dataset(cfg, pair_context=ctx)
        train_set, val_set = _train_val_split(dataset, cfg.val_fraction, cfg.seed)

        train_loader = DataLoader(
            train_set,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=_collate_batch,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=_collate_batch,
        )

        entropy_dim = (
            int(dataset.entropy.shape[-1]) if dataset.entropy is not None else 0
        )
        model = MDUQModel(
            view_dims=view_dims,
            dimension_names=cfg.dimension_names,
            fusion_dim=cfg.fusion_dim,
            fusion_hidden_dim=cfg.fusion_hidden_dim,
            head_hidden_dim=cfg.head_hidden_dim,
            overall_hidden_dim=cfg.overall_hidden_dim,
            dropout=cfg.dropout,
            entropy_dim=entropy_dim,
            layer_softmax_temperature=cfg.layer_temperature or cfg.layer_softmax_temperature,
            layer_dropout=cfg.layer_dropout,
            layer_residual_uniform_alpha=cfg.layer_residual_uniform_alpha,
            gate_logit_clip=cfg.gate_logit_clip,
            view_gate_hidden_dim=cfg.view_gate_hidden_dim,
            view_temperature=cfg.view_temperature,
            view_temperature_min=cfg.view_temperature_min,
            view_temperature_max=cfg.view_temperature_max,
            residual_uniform_alpha=cfg.residual_uniform_alpha,
            view_norm_clip=cfg.view_norm_clip,
            view_dropout_prob=cfg.view_dropout_prob,
            view_gate_scope=cfg.view_gate_scope,
            view_fusion_mode=cfg.view_fusion_mode,
            diagnostic_factorization_mode=cfg.diagnostic_factorization_mode,
            overall_aggregation_mode=cfg.overall_aggregation_mode,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        history: List[Dict[str, object]] = []
        best_val = math.inf
        best_path = ckpt_dir / "best.pt"
        last_path = ckpt_dir / "last.pt"

        for epoch in range(1, cfg.num_epochs + 1):
            train_metrics = _run_epoch(
                model, train_loader, optimizer=optimizer, cfg=cfg, device=device, epoch=epoch
            )
            val_metrics = _run_epoch(
                model, val_loader, optimizer=None, cfg=cfg, device=device, epoch=epoch
            )
            _validate_model_on_loader(model, val_loader, device=device)
            record = {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(record)

            val_loss = val_metrics["loss"]
            if not math.isfinite(float(val_loss)):
                raise ValueError(f"validation loss is non-finite: {val_loss}")
            ckpt_payload = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "view_dims": view_dims,
                "entropy_dim": entropy_dim,
                "config": asdict(cfg),
                "val_metrics": val_metrics,
                "train_metrics": train_metrics,
                "dimension_names": list(cfg.dimension_names),
            }
            torch.save(ckpt_payload, last_path)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(ckpt_payload, best_path)

        metrics_path = ckpt_dir / "metrics.json"
        train_log_path = ckpt_dir / "train_log.csv"
        train_summary_path = ckpt_dir / "train_summary.json"
        fusion_diag_path = ckpt_dir / "fusion_diagnostics.json"
        fusion_train_diag_path = ckpt_dir / "fusion_train_diagnostics.json"
        fusion_diagnostics = _collect_fusion_diagnostics(model, val_loader, device=device)
        fusion_train_diagnostics = _collect_fusion_diagnostics(model, train_loader, device=device)
        summary_payload = {
            "config": asdict(cfg),
            "history": history,
            "best_val_loss": best_val,
            "view_dims": view_dims,
            "entropy_dim": entropy_dim,
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "fusion_diagnostics_path": str(fusion_diag_path),
            "fusion_train_diagnostics_path": str(fusion_train_diag_path),
            "fusion_diagnostics": fusion_diagnostics,
            "fusion_train_diagnostics": fusion_train_diagnostics,
            "num_train": len(train_set),
            "num_val": len(val_set),
            "status": "success",
        }
        with open(metrics_path, "w", encoding="utf-8") as fw:
            json.dump(summary_payload, fw, indent=2, default=str)
        with open(train_summary_path, "w", encoding="utf-8") as fw:
            json.dump(summary_payload, fw, indent=2, default=str)
        with open(fusion_diag_path, "w", encoding="utf-8") as fw:
            json.dump(fusion_diagnostics, fw, indent=2, default=str)
        with open(fusion_train_diag_path, "w", encoding="utf-8") as fw:
            json.dump(fusion_train_diagnostics, fw, indent=2, default=str)
        _write_train_log_csv(train_log_path, history)
        manifest_path = write_stage_manifest(
            ckpt_dir,
            stage="training",
            status="success",
            dataset=cfg.dataset_name,
            model=cfg.model_name,
            artifacts={
                "best_checkpoint": str(best_path),
                "last_checkpoint": str(last_path),
                "metrics_json": str(metrics_path),
                "train_log_csv": str(train_log_path),
                "train_summary_json": str(train_summary_path),
                "fusion_diagnostics_json": str(fusion_diag_path),
                "fusion_train_diagnostics_json": str(fusion_train_diag_path),
            },
            sanity={"best_val_loss": best_val, "num_train": len(train_set), "num_val": len(val_set)},
            pair_context=ctx,
        )

        return {
            "best_val_loss": best_val,
            "best_checkpoint": str(best_path),
            "last_checkpoint": str(last_path),
            "metrics_path": str(metrics_path),
            "train_log_path": str(train_log_path),
            "train_summary_path": str(train_summary_path),
            "manifest_path": str(manifest_path),
            "checkpoint_dir": str(ckpt_dir),
            "num_train": len(train_set),
            "num_val": len(val_set),
            "history": history,
        }
    except Exception as exc:  # noqa: BLE001
        failure_path = ckpt_dir / "train_failure_report.json"
        failure_payload = {
            "status": "failed",
            "dataset": cfg.dataset_name,
            "model": cfg.model_name,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "config": asdict(cfg),
        }
        with open(failure_path, "w", encoding="utf-8") as fw:
            json.dump(failure_payload, fw, indent=2, default=str)
        write_stage_manifest(
            ckpt_dir,
            stage="training",
            status="failed",
            dataset=cfg.dataset_name,
            model=cfg.model_name,
            artifacts={"train_failure_report": str(failure_path)},
            error=repr(exc),
            pair_context=ctx,
        )
        raise


# ---------------------------------------------------------------------------
# DiagUQ-style aliases (the historical `MDUQ` names are kept above for
# backward compatibility).
# ---------------------------------------------------------------------------

DiagUQTrainConfig = MDUQTrainConfig


def train_diaguq(cfg):
    """Alias for :func:	rain_mduq."""
    return train_mduq(cfg)
