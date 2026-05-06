"""Evaluation entry-point for DiagUQ.

Three modes are supported, each producing a single CSV under the resolved
existing artifact root, e.g. ``test_output/<dataset__split>/<model>/diaguq/eval/``:

* :func:`eval_main`              -- main results table:
    baseline supervised estimator, entropy / max-prob baselines,
    ask4conf, semantic entropy, and the DiagUQ network.
* :func:`eval_layer_ablation`    -- layer-fusion ablation:
    ``single_fixed_layer``, ``fixed_multilayer_concat``,
    ``adaptive_multilayer_fusion``.
* :func:`eval_dimension_ablation`-- diagnostic-dimension ablation:
    ``overall_only``, ``multidim_without_aggregator``,
    ``multidim_with_aggregator``.

Reported metrics: AUROC, AUPRC, AUARC, ECE.

Labels: a row counts as **correct** when the corresponding
``knowledge_gap_target`` is below ``LABEL_THRESHOLD`` (i.e. rouge / bleu
above ``1 - LABEL_THRESHOLD``). The metrics measure each method's ability
to predict that correctness from a single uncertainty / confidence score.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from common.numpy_compat import safe_trapezoid
from common.artifact_manifest import write_stage_manifest
from common.artifact_paths import normalize_split_tag, split_dataset_and_raw
from common.diaguq_existing_artifacts import (
    ExistingDiagUQArtifactRoot,
    require_existing_diaguq_artifact_root,
    resolve_existing_diaguq_artifact_root,
)
from common.export_validation import build_export_sanity
from common.metrics import build_dimension_diagnostics
from common.pair_context import (
    DiagUQPairContext,
    assert_diaguq_output_path,
    assert_no_duplicate_output_dirs,
    INVALID_CHECKPOINT_VARIANT_MESSAGE,
    resolve_checkpoint_context_for_eval,
    resolve_pair_context,
)
from common.sample_alignment import require_matching_sample_ids
from torch.utils.data import DataLoader, Subset

from pipeline.diaguq_model import DEFAULT_DIMENSIONS, MDUQModel
from pipeline.train_diaguq import (
    MDUQTrainConfig,
    _collate_batch,
    _MDUQTensorDataset,
    _resolve_overall_target,
    _select_dimension_targets,
    _split_views,
    _train_val_split,
    train_mduq,
)
from features.load_feature_tensors import load_mduq_dataset


LABEL_THRESHOLD = 0.5  # knowledge_gap_target < 0.5  ->  correct
VIEW_NAMES = ("query", "answer", "relation")


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


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _to_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float64)


def _binary_mask(scores: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(scores) & np.isfinite(labels)
    return scores[mask], labels[mask]


def _auroc(score: np.ndarray, label: np.ndarray) -> float:
    """AUROC where higher ``score`` predicts ``label==1``."""
    score, label = _binary_mask(score, label)
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
    tpr = cum_tp / pos
    fpr = cum_fp / neg
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    return float(safe_trapezoid(tpr, fpr))


def _auprc(score: np.ndarray, label: np.ndarray) -> float:
    """Average precision (precision over recall) for ``label==1``."""
    score, label = _binary_mask(score, label)
    if score.size == 0 or label.sum() == 0:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    label = label[order]
    tp = np.cumsum(label)
    fp = np.cumsum(1.0 - label)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / max(label.sum(), 1e-12)
    # Step-wise AP
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def _auarc(confidence: np.ndarray, correct: np.ndarray) -> float:
    """Area under the accuracy-rejection curve.

    Sort by descending ``confidence``; sweep retention from full coverage
    down to top-1; integrate accuracy over coverage.
    """
    confidence, correct = _binary_mask(confidence, correct)
    n = confidence.size
    if n == 0:
        return float("nan")
    order = np.argsort(-confidence, kind="mergesort")
    correct = correct[order]
    cum_acc = np.cumsum(correct) / np.arange(1, n + 1)
    coverage = np.arange(1, n + 1) / n
    return float(safe_trapezoid(cum_acc, coverage))


def _ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    confidence, correct = _binary_mask(confidence, correct)
    if confidence.size == 0:
        return float("nan")
    confidence = np.clip(confidence, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = confidence.size
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        if not mask.any():
            continue
        avg_conf = float(confidence[mask].mean())
        avg_acc = float(correct[mask].mean())
        ece += (mask.sum() / n) * abs(avg_conf - avg_acc)
    return float(ece)


def _min_max(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return x
    lo, hi = float(finite.min()), float(finite.max())
    if hi - lo < 1e-12:
        return np.where(np.isfinite(x), 0.5, x)
    out = (x - lo) / (hi - lo)
    out[~np.isfinite(x)] = np.nan
    return out


def compute_method_metrics(
    confidence: np.ndarray, correct: np.ndarray
) -> Dict[str, float]:
    """Compute the four headline metrics for one method.

    ``confidence`` is the model's belief that ``correct == 1``; metrics
    that need a "score for the positive class" use ``confidence`` directly.
    """
    return {
        "AUROC": _auroc(confidence, correct),
        "AUPRC": _auprc(confidence, correct),
        "AUARC": _auarc(confidence, correct),
        "ECE": _ece(confidence, correct),
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def eval_dir(
    dataset_name: str,
    model_name: str,
    output_root: Optional[str] = None,
    *,
    artifact_root: Optional[Path] = None,
    artifact_root_name: Optional[str] = None,
) -> Path:
    """DiagUQ evaluation directory for one (dataset, model) pair."""
    if artifact_root is not None:
        return Path(artifact_root) / "eval"
    return resolve_pair_context(dataset_name, model_name, runtime_root=output_root).eval_dir


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    field_order: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                field_order.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fw:
        writer = csv.DictWriter(fw, fieldnames=field_order)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().numpy())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def _float_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _log_eval_info(message: str, *args) -> None:
    try:
        from loguru import logger

        logger.info(message, *args)
    except Exception:
        print(message.format(*args))


def _log_eval_warning(message: str, *args) -> None:
    try:
        from loguru import logger

        logger.warning(message, *args)
    except Exception:
        print(message.format(*args))


def _log_numpy_metric_api() -> None:
    _log_eval_info(
        "[eval] numpy version={} has_trapezoid={} has_trapz={}",
        np.__version__,
        hasattr(np, "trapezoid"),
        hasattr(np, "trapz"),
    )


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


def _hidden_bank_dir_from_root(artifact_root: Path) -> Path:
    path = Path(artifact_root) / "hidden_bank"
    _log_eval_info("[eval] loading hidden_bank dir: {}", path)
    if not path.is_dir():
        raise FileNotFoundError(f"hidden_bank missing: {path}")
    return path


def _dimension_targets_pt_from_root(artifact_root: Path) -> Path:
    path = Path(artifact_root) / "dimension_targets" / "dimension_targets.pt"
    _log_eval_info("[eval] loading dimension target file: {}", path)
    if not path.is_file():
        raise FileNotFoundError(f"dimension_targets.pt missing: {path}")
    return path


def _checkpoint_path_from_root(artifact_root: Path) -> Path:
    ckpt_dir = Path(artifact_root) / "checkpoints"
    best = ckpt_dir / "best.pt"
    last = ckpt_dir / "last.pt"
    selected = best if best.is_file() else last if last.is_file() else None
    if selected is None:
        raise FileNotFoundError(
            "no DiagUQ checkpoint found; checked paths: "
            f"{best}; {last}"
        )
    _log_eval_info("[eval] loading checkpoint file: {}", selected)
    return selected


def _load_checkpoint_from_root(artifact_root: Path) -> Mapping[str, torch.Tensor]:
    return torch.load(
        _checkpoint_path_from_root(artifact_root),
        map_location="cpu",
        weights_only=False,
    )


def _load_questions_from_response_cache(
    cfg: MDUQTrainConfig,
    n: int,
    artifact_root: Optional[Path] = None,
) -> List[Optional[str]]:
    try:
        from common.artifact_locator import locate_response_cache_artifacts

        artifacts = locate_response_cache_artifacts(
            cfg.dataset_name, cfg.model_name, cfg.output_root
        )
        extend_path = artifacts.require("mextend")
        with open(extend_path, "r", encoding="utf-8") as fr:
            data = json.load(fr)
    except Exception:
        return [None] * n
    questions: List[Optional[str]] = []
    for idx in range(n):
        if idx < len(data) and isinstance(data[idx], Mapping):
            questions.append(data[idx].get("question_str"))
        else:
            questions.append(None)
    return questions


def _build_dataset_from_artifact_root(
    cfg: MDUQTrainConfig,
    artifact_root: Path,
):
    bank_dir = _hidden_bank_dir_from_root(artifact_root)
    bundle = load_mduq_dataset(
        cfg.dataset_name,
        cfg.model_name,
        layer_list=cfg.layer_list,
        output_root=cfg.output_root,
        query_kind=cfg.query_kind,
        answer_kind=cfg.answer_kind,
        bank_dir=str(bank_dir),
    )
    views = bundle["views"]
    if "query" not in views or "answer" not in views or "relation" not in views:
        raise RuntimeError(
            "MDUQ feature bundle missing required views; got "
            f"{list(views.keys())}"
        )
    entropy = views.get("entropy")
    core_views = {k: views[k] for k in ("query", "answer", "relation")}

    target_pt = _dimension_targets_pt_from_root(artifact_root)
    payload = torch.load(target_pt, map_location="cpu")
    dimension_targets = _select_dimension_targets(payload, cfg.dimension_names)
    overall_target = _resolve_overall_target(payload)

    n_views = next(iter(core_views.values())).shape[0]
    n_targets = dimension_targets.shape[0]
    if n_views != n_targets:
        n = min(n_views, n_targets)
        for key in list(core_views.keys()):
            core_views[key] = core_views[key][:n]
        dimension_targets = dimension_targets[:n]
        overall_target = overall_target[:n]
        if entropy is not None:
            entropy = entropy[:n]

    view_dims = {key: int(value.shape[-1]) for key, value in core_views.items()}
    dataset = _MDUQTensorDataset(
        core_views,
        dimension_targets=dimension_targets,
        overall_target=overall_target,
        entropy=entropy,
    )
    return dataset, view_dims


# ---------------------------------------------------------------------------
# Baseline scores from the dimension-targets payload
# ---------------------------------------------------------------------------


@dataclass
class _BaselineSources:
    correct: np.ndarray              # binary correctness label
    knowledge_gap: np.ndarray
    ambiguity: np.ndarray             # semantic entropy proxy
    overall_target: np.ndarray
    overall_ask4conf: np.ndarray      # 1 - ask4conf prob
    entropy_features: Optional[np.ndarray] = None  # (N, F_e)


def _load_baseline_sources(
    cfg: MDUQTrainConfig,
    val_indices: Sequence[int],
    artifact_root: Optional[Path] = None,
) -> _BaselineSources:
    resolved_root = _coerce_artifact_root(cfg, artifact_root)
    target_pt = _dimension_targets_pt_from_root(resolved_root)
    payload = torch.load(target_pt, map_location="cpu")

    def _col(key: str) -> np.ndarray:
        if key not in payload:
            raise KeyError(f"dimension_targets.pt missing {key!r}")
        return _to_np(payload[key])

    knowledge_gap = _col("knowledge_gap_target")
    ambiguity = _col("ambiguity_target")
    overall_t = _col("overall_target")
    overall_a = _col("overall_ask4conf_target")

    correct = np.where(
        np.isfinite(knowledge_gap),
        (knowledge_gap < LABEL_THRESHOLD).astype(np.float64),
        np.nan,
    )

    # Subset to val indices to align with model evaluation
    val = np.asarray(list(val_indices), dtype=np.int64)
    n = correct.size
    val = val[val < n]

    return _BaselineSources(
        correct=correct[val],
        knowledge_gap=knowledge_gap[val],
        ambiguity=ambiguity[val],
        overall_target=overall_t[val],
        overall_ask4conf=overall_a[val],
    )


def _entropy_features_for_indices(
    cfg: MDUQTrainConfig,
    val_indices: Sequence[int],
    artifact_root: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """Pull ``entropy + sorted probs`` from the hidden bank's extras."""
    resolved_root = _coerce_artifact_root(cfg, artifact_root)
    bank_dir = _hidden_bank_dir_from_root(resolved_root)
    ent_path = bank_dir / "query_entropies.pt"
    prob_path = bank_dir / "query_probs.pt"
    ent_mask_path = bank_dir / "query_entropy_available.pt"
    prob_mask_path = bank_dir / "query_prob_available.pt"
    if not ent_path.is_file() and not prob_path.is_file():
        return None
    parts: List[np.ndarray] = []
    if ent_path.is_file():
        ent = _to_np(torch.load(ent_path, map_location="cpu"))
        if ent.ndim == 1:
            ent = ent.reshape(-1, 1)
        if ent_mask_path.is_file():
            ent_mask = _to_np(torch.load(ent_mask_path, map_location="cpu")).astype(bool).reshape(-1)
            limit = min(ent.shape[0], ent_mask.shape[0])
            invalid = np.arange(limit)[~ent_mask[:limit]]
            ent[invalid, :] = np.nan
        parts.append(ent)
    if prob_path.is_file():
        probs = _to_np(torch.load(prob_path, map_location="cpu"))
        if prob_mask_path.is_file():
            prob_mask = _to_np(torch.load(prob_mask_path, map_location="cpu")).astype(bool).reshape(-1)
            limit = min(probs.shape[0], prob_mask.shape[0])
            invalid = np.arange(limit)[~prob_mask[:limit]]
            probs[invalid, :] = np.nan
        sorted_probs = -np.sort(-probs, axis=-1)
        parts.append(sorted_probs)
    if not parts:
        return None
    feats = np.concatenate(parts, axis=-1)
    val = np.asarray(list(val_indices), dtype=np.int64)
    val = val[val < feats.shape[0]]
    return feats[val]


# ---------------------------------------------------------------------------
# Legacy supervised RF baseline (best-effort, optional)
# ---------------------------------------------------------------------------


def _legacy_rf_scores(
    cfg: MDUQTrainConfig,
    sources: _BaselineSources,
    val_indices: Sequence[int],
    artifact_root: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """Train a plain Random Forest on the entropy features as a quick stand-in
    for the legacy supervised baseline; returns confidence per row, or None
    if scikit-learn or features are unavailable.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
    except Exception:
        return None
    feats = _entropy_features_for_indices(cfg, val_indices, artifact_root)
    if feats is None or feats.shape[0] < 4:
        return None
    correct = sources.correct
    mask = np.isfinite(correct)
    if mask.sum() < 4 or len(np.unique(correct[mask])) < 2:
        return None
    rf = RandomForestClassifier(
        n_estimators=200, random_state=cfg.seed, n_jobs=1
    )
    # Use the SAME val rows for fit + score (single-set baseline). This is
    # generous to the baseline but matches the legacy "trained-on-this-split"
    # number; it is the conservative comparison.
    X = feats[mask]
    y = correct[mask].astype(int)
    rf.fit(X, y)
    pred = rf.predict_proba(feats)
    # Class-1 probability when present, otherwise zeros.
    classes = list(rf.classes_)
    if 1 in classes:
        col = classes.index(1)
        out = pred[:, col]
    else:
        out = np.zeros(feats.shape[0])
    full = np.full_like(correct, np.nan, dtype=np.float64)
    full[: out.shape[0]] = out
    return full


# ---------------------------------------------------------------------------
# Run a forward pass with a trained MDUQ model
# ---------------------------------------------------------------------------


def _model_forward_on_subset(
    cfg: MDUQTrainConfig,
    dataset,
    indices: Sequence[int],
    *,
    view_dims: Mapping[str, int],
    entropy_dim: int,
    state_dict: Mapping[str, torch.Tensor],
) -> Dict[str, np.ndarray]:
    device = torch.device(cfg.device)
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
        view_dropout_prob=0.0,
        view_gate_scope=cfg.view_gate_scope,
        view_fusion_mode=cfg.view_fusion_mode,
        diagnostic_factorization_mode=cfg.diagnostic_factorization_mode,
        overall_aggregation_mode=cfg.overall_aggregation_mode,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    loader = DataLoader(
        Subset(dataset, list(indices)),
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=_collate_batch,
    )

    confs: List[torch.Tensor] = []
    dims: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            views, rest = _split_views(batch)
            views = {k: v.to(device) for k, v in views.items()}
            entropy = rest.get("entropy")
            if entropy is not None:
                entropy = entropy.to(device)
            out = model(views, entropy=entropy)
            confs.append(out.confidence.detach().cpu())
            dims.append(out.dimension_scores.detach().cpu())

    return {
        "confidence": _to_np(torch.cat(confs, dim=0)),
        "dimension_scores": _to_np(torch.cat(dims, dim=0)),
    }


def _model_forward_on_dataset(
    cfg: MDUQTrainConfig,
    dataset,
    *,
    view_dims: Mapping[str, int],
    entropy_dim: int,
    state_dict: Mapping[str, torch.Tensor],
    dimension_names: Sequence[str],
) -> Dict[str, Any]:
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
    dims: List[torch.Tensor] = []
    layer_weights: Dict[str, List[torch.Tensor]] = {}
    view_weights: List[torch.Tensor] = []
    view_weights_by_target: Dict[str, List[torch.Tensor]] = {}
    gate_logits: Dict[str, List[torch.Tensor]] = {}
    view_logits_by_target: Dict[str, List[torch.Tensor]] = {}
    diagnostic_components: Dict[str, List[torch.Tensor]] = {}
    overall_components: Dict[str, List[torch.Tensor]] = {}
    dimension_representations: Dict[str, List[torch.Tensor]] = {}

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
            dims.append(out.dimension_scores.detach().cpu())
            for key, weight in out.layer_weights.items():
                if key == "_view_weights":
                    view_weights.append(weight.detach().cpu())
                elif key.startswith("_view_weights"):
                    if "raw" in key or (key == "_view_weights" and "_view_weights_overall" in out.layer_weights):
                        continue
                    target = key.replace("_view_weights_", "")
                    view_weights_by_target.setdefault(target, []).append(weight.detach().cpu())
                else:
                    layer_weights.setdefault(key, []).append(weight.detach().cpu())
            for key, logits in out.gate_logits.items():
                if key.startswith("_view_logits"):
                    target = key.replace("_view_logits_", "") if key != "_view_logits" else "overall"
                    view_logits_by_target.setdefault(target, []).append(logits.detach().cpu())
                gate_logits.setdefault(key, []).append(logits.detach().cpu())
            for key, tensor in out.diagnostic_components.items():
                diagnostic_components.setdefault(key, []).append(tensor.detach().cpu())
            for key, tensor in out.overall_components.items():
                overall_components.setdefault(key, []).append(tensor.detach().cpu())
            for key, tensor in out.dimension_representations.items():
                dimension_representations.setdefault(key, []).append(tensor.detach().cpu())

    return {
        "confidence": _to_np(torch.cat(confidences, dim=0)),
        "uncertainty": _to_np(torch.cat(uncertainties, dim=0)),
        "dimension_scores": _to_np(torch.cat(dims, dim=0)),
        "layer_weights": {key: torch.cat(parts, dim=0) for key, parts in layer_weights.items()},
        "view_weights": torch.cat(view_weights, dim=0) if view_weights else None,
        "view_weights_by_target": {key: torch.cat(parts, dim=0) for key, parts in view_weights_by_target.items()},
        "view_logits_by_target": {key: torch.cat(parts, dim=0) for key, parts in view_logits_by_target.items()},
        "gate_logits": {key: torch.cat(parts, dim=0) for key, parts in gate_logits.items()},
        "diagnostic_components": {key: torch.cat(parts, dim=0) for key, parts in diagnostic_components.items()},
        "overall_components": {key: torch.cat(parts, dim=0) for key, parts in overall_components.items()},
        "dimension_representations": {key: torch.cat(parts, dim=0) for key, parts in dimension_representations.items()},
    }


def _target_array(payload: Mapping[str, Any], key: str, n: int) -> np.ndarray:
    if key not in payload:
        return np.full(n, np.nan, dtype=np.float64)
    arr = _to_np(payload[key]).reshape(-1)
    out = np.full(n, np.nan, dtype=np.float64)
    limit = min(n, arr.shape[0])
    out[:limit] = arr[:limit]
    return out


def _component_value(components: Mapping[str, Any], key: str, idx: int) -> float:
    tensor = components.get(key)
    if not isinstance(tensor, torch.Tensor) or idx >= tensor.shape[0]:
        return float("nan")
    value = tensor[idx]
    if value.numel() != 1:
        return float("nan")
    return float(value.detach().cpu().item())


def _component_vector_norm(components: Mapping[str, Any], key: str, idx: int) -> float:
    tensor = components.get(key)
    if not isinstance(tensor, torch.Tensor) or idx >= tensor.shape[0]:
        return float("nan")
    value = tensor[idx].detach().cpu().float()
    if value.numel() == 0:
        return float("nan")
    return float(value.norm().item())


def _target_text_array(payload: Mapping[str, Any], key: str, n: int, *, default: str) -> List[str]:
    values = payload.get(key)
    if isinstance(values, (list, tuple)):
        out = [str(value) if value is not None else default for value in values[:n]]
        if len(out) < n:
            out.extend([default] * (n - len(out)))
        return out
    return [default] * n


def _target_metadata_text_array(payload: Mapping[str, Any], key: str, n: int, *, default_prefix: str) -> List[str]:
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


def _feature_column(feats: Optional[np.ndarray], column: int, n: int) -> np.ndarray:
    out = np.full(n, np.nan, dtype=np.float64)
    if feats is None or feats.ndim != 2 or feats.shape[1] <= column:
        return out
    limit = min(n, feats.shape[0])
    out[:limit] = feats[:limit, column]
    return out


def _prediction_rows(
    cfg: MDUQTrainConfig,
    artifact_root: Path,
    forward: Mapping[str, Any],
    *,
    dimension_names: Sequence[str],
    target_payload: Mapping[str, Any],
    evaluation_context: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    n = int(forward["confidence"].shape[0])
    knowledge_gap = _target_array(target_payload, "knowledge_gap_target", n)
    ambiguity = _target_array(target_payload, "ambiguity_target", n)
    ambiguity_raw = _target_array(target_payload, "ambiguity_raw", n)
    predictive = _target_array(target_payload, "predictive_variability_target", n)
    overall_target = _target_array(target_payload, "overall_target", n)
    overall_ask4conf = _target_array(target_payload, "overall_ask4conf_target", n)
    task_error_target = _target_array(target_payload, "task_error_target", n)
    predictive_raw = _target_array(target_payload, "predictive_variability_raw", n)
    predictive_num_samples = _target_array(target_payload, "predictive_variability_num_samples", n)
    predictive_cluster_count = _target_array(target_payload, "predictive_variability_cluster_count", n)
    predictive_entropy = _target_array(target_payload, "predictive_variability_entropy", n)
    availability_by_dim: Dict[str, np.ndarray] = {}
    status_by_dim: Dict[str, List[str]] = {}
    source_by_dim: Dict[str, List[str]] = {}
    reliability_by_dim: Dict[str, np.ndarray] = {}
    loss_weight_by_dim: Dict[str, np.ndarray] = {}
    metric_group_by_dim: Dict[str, List[str]] = {}
    construction_note_by_dim: Dict[str, List[str]] = {}
    metadata = target_payload.get("metadata") if isinstance(target_payload.get("metadata"), Mapping) else {}
    target_sample_ids = _target_metadata_text_array(target_payload, "sample_ids", n, default_prefix=f"{cfg.dataset_name}:missing_sample_id")
    source_sample_ids = _target_metadata_text_array(target_payload, "source_sample_ids", n, default_prefix=f"{cfg.dataset_name}:missing_source_sample_id")
    task_type = str(target_payload.get("task_type") or metadata.get("task_type") or "")
    overall_target_policy = str(target_payload.get("overall_target_policy") or metadata.get("overall_target_policy") or "")
    overall_dimension_weights = target_payload.get("overall_target_dimension_weights") or metadata.get("overall_target_dimension_weights") or {}
    if isinstance(overall_dimension_weights, Mapping):
        overall_dimension_weights_text = json.dumps(dict(overall_dimension_weights), sort_keys=True)
    else:
        overall_dimension_weights_text = str(overall_dimension_weights or "")
    task_error_source = _target_text_array(target_payload, "task_error_target_source", n, default="")
    ask4conf_status = _target_text_array(target_payload, "ask4conf_status", n, default="")
    ask4conf_missing_reason = _target_text_array(target_payload, "ask4conf_missing_reason", n, default="")
    for name in dimension_names:
        key = f"{name}_target_available"
        if key in target_payload:
            availability_by_dim[name] = _target_array(target_payload, key, n)
        else:
            target_key = f"{name}_target"
            availability_by_dim[name] = np.isfinite(_target_array(target_payload, target_key, n)).astype(np.float64)
        fallback_status = ["proxy" if value >= 0.5 and np.isfinite(value) else "unavailable" for value in availability_by_dim[name]]
        raw_status = _target_text_array(target_payload, f"{name}_target_status", n, default="")
        if not any(raw_status):
            raw_status = _target_text_array(target_payload, f"dim_{name}_target_status", n, default="")
        status_by_dim[name] = [raw or fallback for raw, fallback in zip(raw_status, fallback_status)]
        if f"{name}_target_source" in target_payload:
            source_by_dim[name] = _target_text_array(target_payload, f"{name}_target_source", n, default="")
        elif f"dim_{name}_target_source" in target_payload:
            source_by_dim[name] = _target_text_array(target_payload, f"dim_{name}_target_source", n, default="")
        else:
            source_by_dim[name] = [
                "legacy_available_flag" if value in {"gold", "dataset_grounded", "proxy"} else "unavailable"
                for value in status_by_dim[name]
            ]
        reliability_by_dim[name] = _target_array(target_payload, f"{name}_target_reliability", n)
        if not np.isfinite(reliability_by_dim[name]).any():
            reliability_by_dim[name] = _target_array(target_payload, f"dim_{name}_target_reliability", n)
        loss_weight_by_dim[name] = _target_array(target_payload, f"{name}_target_loss_weight_multiplier", n)
        if not np.isfinite(loss_weight_by_dim[name]).any():
            loss_weight_by_dim[name] = _target_array(target_payload, f"dim_{name}_target_loss_weight_multiplier", n)
        metric_group_by_dim[name] = _target_text_array(target_payload, f"{name}_target_metric_group", n, default="")
        if not any(metric_group_by_dim[name]):
            metric_group_by_dim[name] = _target_text_array(target_payload, f"dim_{name}_metric_group", n, default="")
        construction_note_by_dim[name] = _target_text_array(target_payload, f"{name}_target_construction_note", n, default="")
        if not any(construction_note_by_dim[name]):
            construction_note_by_dim[name] = _target_text_array(target_payload, f"dim_{name}_construction_note", n, default="")
    if "correct" in target_payload:
        correct = _target_array(target_payload, "correct", n)
    else:
        correct = np.where(
            np.isfinite(knowledge_gap),
            (knowledge_gap < LABEL_THRESHOLD).astype(np.float64),
            np.nan,
        )
    feats = _entropy_features_for_indices(cfg, list(range(n)), artifact_root)
    entropy_raw = _feature_column(feats, 0, n)
    entropy_baseline = 1.0 - _min_max(entropy_raw)
    max_prob_baseline = _feature_column(feats, 1, n)
    ask4conf_confidence = 1.0 - overall_ask4conf
    questions = _load_questions_from_response_cache(cfg, n, artifact_root)

    rows: List[Dict[str, Any]] = []
    dim_scores = forward["dimension_scores"]
    diagnostic_components = forward.get("diagnostic_components") or {}
    overall_components = forward.get("overall_components") or {}
    dimension_representations = forward.get("dimension_representations") or {}
    for idx in range(n):
        row: Dict[str, Any] = {
            "index": idx,
            "sample_id": target_sample_ids[idx],
            "source_sample_id": source_sample_ids[idx],
            "dataset": cfg.dataset_name,
            "resolved_variant": cfg.dataset_name,
            "split": str(metadata.get("split") or ""),
            "virtual_split": metadata.get("virtual_split"),
            "model": cfg.model_name,
            "task_type": task_type,
            "question_str": questions[idx],
            "correct": correct[idx],
            "knowledge_gap_target": knowledge_gap[idx],
            "ambiguity_target": ambiguity[idx],
            "ambiguity_raw": ambiguity_raw[idx],
            "predictive_variability_target": predictive[idx],
            "predictive_variability_raw": predictive_raw[idx],
            "predictive_variability_num_samples": predictive_num_samples[idx],
            "predictive_variability_cluster_count": predictive_cluster_count[idx],
            "predictive_variability_entropy": predictive_entropy[idx],
            "task_error_target": task_error_target[idx],
            "task_error_target_source": task_error_source[idx],
            "overall_target": overall_target[idx],
            "overall_target_policy": overall_target_policy,
            "overall_target_composition_policy": overall_target_policy,
            "overall_target_dimension_weights": overall_dimension_weights_text,
            "overall_ask4conf_target": overall_ask4conf[idx],
            "ask4conf_confidence": ask4conf_confidence[idx],
            "ask4conf_status": ask4conf_status[idx] or ("ok" if np.isfinite(ask4conf_confidence[idx]) else "missing"),
            "ask4conf_missing_reason": ask4conf_missing_reason[idx],
            "mduq_uncertainty": forward["uncertainty"][idx],
            "mduq_confidence": forward["confidence"][idx],
            "entropy_baseline": entropy_baseline[idx],
            "max_prob_baseline": max_prob_baseline[idx],
            "shared_uncertainty_score": _component_value(diagnostic_components, "shared_uncertainty_score", idx),
            "shared_uncertainty_logit": _component_value(diagnostic_components, "shared_uncertainty_logit", idx),
            "shared_uncertainty_hidden_norm": _component_vector_norm(diagnostic_components, "shared_uncertainty_hidden", idx),
            "overall_direct": _component_value(overall_components, "overall_direct", idx),
            "overall_from_dimensions": _component_value(overall_components, "overall_from_dimensions", idx),
            "overall_final": _component_value(overall_components, "overall_final", idx),
            "overall_hybrid_gate": _component_value(overall_components, "overall_hybrid_gate", idx),
        }
        dimension_weight_tensor = overall_components.get("overall_dimension_weights")
        if isinstance(dimension_weight_tensor, torch.Tensor) and dimension_weight_tensor.dim() == 2 and idx < dimension_weight_tensor.shape[0]:
            for dim_idx, name in enumerate(dimension_names):
                if dim_idx < dimension_weight_tensor.shape[1]:
                    row[f"overall_dimension_weight_{name}"] = float(dimension_weight_tensor[idx, dim_idx].detach().cpu().item())
        for dim_idx, name in enumerate(dimension_names):
            row[f"dim_{name}"] = dim_scores[idx, dim_idx]
            row[f"residual_{name}_score"] = _component_value(diagnostic_components, f"residual_{name}_score", idx)
            row[f"residual_{name}_logit"] = _component_value(diagnostic_components, f"residual_{name}_logit", idx)
            row[f"residual_{name}_hidden_norm"] = _component_vector_norm(diagnostic_components, f"residual_{name}_hidden", idx)
            row[f"dimension_repr_{name}_norm"] = _component_vector_norm(dimension_representations, name, idx)
            available = bool(
                idx < availability_by_dim[name].shape[0]
                and np.isfinite(availability_by_dim[name][idx])
                and availability_by_dim[name][idx] >= 0.5
            )
            row[f"{name}_available"] = available
            row[f"dim_{name}_target_status"] = status_by_dim[name][idx]
            row[f"dim_{name}_target_source"] = source_by_dim[name][idx]
            row[f"dim_{name}_target_reliability"] = reliability_by_dim[name][idx]
            row[f"dim_{name}_target_loss_weight_multiplier"] = loss_weight_by_dim[name][idx]
            row[f"dim_{name}_metric_group"] = metric_group_by_dim[name][idx]
            row[f"dim_{name}_construction_note"] = construction_note_by_dim[name][idx]
        for view_name, tensor in forward["layer_weights"].items():
            if idx >= tensor.shape[0]:
                continue
            for layer_idx, weight in enumerate(tensor[idx].detach().cpu().tolist()):
                row[f"layer_weight_{view_name}_l{layer_idx}"] = weight
        view_weights = forward.get("view_weights")
        if isinstance(view_weights, torch.Tensor):
            view_names = list(forward["layer_weights"].keys())
            if view_weights.dim() == 2 and idx < view_weights.shape[0]:
                for view_idx, view_name in enumerate(view_names):
                    if view_idx < view_weights.shape[1]:
                        row[f"view_weight_{view_name}"] = float(view_weights[idx, view_idx].item())
            elif view_weights.dim() == 1:
                for view_idx, view_name in enumerate(view_names):
                    if view_idx < view_weights.numel():
                        row[f"view_weight_{view_name}"] = float(view_weights[view_idx].item())
        view_names = list(forward["layer_weights"].keys())
        for target, tensor in (forward.get("view_weights_by_target") or {}).items():
            if not isinstance(tensor, torch.Tensor) or tensor.dim() != 2 or idx >= tensor.shape[0]:
                continue
            for view_idx, view_name in enumerate(view_names):
                if view_idx < tensor.shape[1]:
                    row[f"view_weight_{target}_{view_name}"] = float(tensor[idx, view_idx].item())
        for target, tensor in (forward.get("view_logits_by_target") or {}).items():
            if not isinstance(tensor, torch.Tensor) or tensor.dim() != 2 or idx >= tensor.shape[0]:
                continue
            for view_idx, view_name in enumerate(view_names):
                if view_idx < tensor.shape[1]:
                    row[f"view_logit_{target}_{view_name}"] = float(tensor[idx, view_idx].item())
        if evaluation_context:
            row.update({key: value for key, value in evaluation_context.items() if isinstance(value, (str, int, float, bool))})
        rows.append(row)
    return rows


def _metric_row_from_predictions(
    dataset_name: str,
    model_name: str,
    method: str,
    confidence: np.ndarray,
    correct: np.ndarray,
) -> Dict[str, Any]:
    mask = np.isfinite(confidence) & np.isfinite(correct)
    valid_conf = confidence[mask]
    valid_correct = correct[mask]
    row: Dict[str, Any] = {
        "dataset": dataset_name,
        "model": model_name,
        "method": method,
        "n_total": int(confidence.shape[0]),
        "n_eval": int(mask.sum()),
        "n_missing_confidence": int((~np.isfinite(confidence)).sum()),
        "n_missing_label": int((~np.isfinite(correct)).sum()),
        "n_positive": int((valid_correct >= 0.5).sum()) if valid_correct.size else 0,
        "n_negative": int((valid_correct < 0.5).sum()) if valid_correct.size else 0,
        "AUROC": None,
        "AUPRC": None,
        "AUARC": None,
        "ECE": None,
        "Brier": None,
        "skip_reason": None,
    }
    if not mask.any():
        row["skip_reason"] = "no_finite_confidence_and_label_pairs"
        return row
    row["Brier"] = float(np.mean((valid_conf - (valid_correct >= 0.5).astype(np.float64)) ** 2))
    if len(np.unique((valid_correct >= 0.5).astype(int))) < 2:
        row["skip_reason"] = "single_class_labels"
        return row
    metrics = compute_method_metrics(confidence, correct)
    row.update(metrics)
    return row


def _metrics_from_prediction_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    dataset_name = str(rows[0].get("dataset"))
    model_name = str(rows[0].get("model"))
    correct = np.asarray([_float_or_nan(row.get("correct")) for row in rows], dtype=np.float64)
    methods = {
        "mduq": "mduq_confidence",
        "entropy_baseline": "entropy_baseline",
        "max_prob_baseline": "max_prob_baseline",
        "ask4conf": "ask4conf_confidence",
    }
    if any("mduq_calibrated_confidence" in row for row in rows):
        methods["mduq_calibrated"] = "mduq_calibrated_confidence"
    metric_rows: List[Dict[str, Any]] = []
    for method, column in methods.items():
        confidence = np.asarray(
            [_float_or_nan(row.get(column)) for row in rows],
            dtype=np.float64,
        )
        metric_rows.append(
            _metric_row_from_predictions(dataset_name, model_name, method, confidence, correct)
        )
    return metric_rows


_ARCH_CONFIG_KEYS = (
    "fusion_dim",
    "fusion_hidden_dim",
    "head_hidden_dim",
    "overall_hidden_dim",
    "dropout",
    "layer_softmax_temperature",
    "layer_temperature",
    "layer_dropout",
    "layer_residual_uniform_alpha",
    "gate_logit_clip",
    "view_gate_hidden_dim",
    "view_temperature",
    "view_temperature_min",
    "view_temperature_max",
    "residual_uniform_alpha",
    "view_norm_clip",
    "view_entropy_weight",
    "view_entropy_warmup_epochs",
    "view_entropy_anneal_to",
    "view_dropout_prob",
    "view_gate_scope",
    "view_fusion_mode",
    "diagnostic_factorization_mode",
    "dimension_corr_regularization_weight",
    "dimension_corr_margin",
    "residual_diversity_weight",
    "residual_diversity_margin",
    "overall_aggregation_mode",
    "gold_target_loss_multiplier",
    "dataset_grounded_target_loss_multiplier",
    "proxy_target_loss_multiplier",
    "unavailable_target_loss_multiplier",
    "proxy_target_loss_weight_multiplier",
)


def _cfg_from_checkpoint(base: MDUQTrainConfig, ckpt: Mapping[str, Any], dimension_names: Sequence[str]) -> MDUQTrainConfig:
    saved = ckpt.get("config") if isinstance(ckpt.get("config"), Mapping) else {}
    overrides = {key: saved[key] for key in _ARCH_CONFIG_KEYS if key in saved}
    overrides["dimension_names"] = list(dimension_names)
    return _make_cfg(base, **overrides)


def _tensor_stats(values: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _fusion_diagnostics_from_forward(forward: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "layer_weights": {},
        "view_weights": {},
        "view_weight_pairwise_l1": {},
        "gate_logits": {},
        "view_gate_logits": {},
        "dimension_representations": {},
        "dimension_representation_cosine": {},
    }
    for view_name, tensor in (forward.get("layer_weights") or {}).items():
        if not isinstance(tensor, torch.Tensor) or tensor.dim() != 2:
            continue
        weights = tensor.detach().cpu().float()
        entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=-1)
        out["layer_weights"][view_name] = {
            "average_max_layer_weight": float(weights.max(dim=-1).values.mean().item()),
            "mean_by_layer": [float(v) for v in weights.mean(dim=0).tolist()],
            "entropy": _tensor_stats(entropy.numpy()),
        }
    view_weights = forward.get("view_weights")
    if isinstance(view_weights, torch.Tensor):
        vw = view_weights.detach().cpu().float()
        if vw.dim() == 2:
            entropy = -(vw * torch.log(vw.clamp_min(1e-8))).sum(dim=-1) / math.log(max(vw.shape[1], 2))
            max_w = vw.max(dim=-1).values
        else:
            entropy = torch.empty(0)
            max_w = torch.empty(0)
        out["view_weights"] = {
            "shape": list(vw.shape),
            "values_or_mean": [float(v) for v in (vw.mean(dim=0) if vw.dim() == 2 else vw).tolist()],
            "std_by_view": [float(v) for v in (vw.std(dim=0, unbiased=False) if vw.dim() == 2 else torch.zeros_like(vw)).tolist()],
            "entropy": _tensor_stats(entropy.numpy()),
            "max_weight": _tensor_stats(max_w.numpy()),
            "collapse_rate_gt_095": float((max_w > 0.95).float().mean().item()) if max_w.numel() else None,
        }
    mean_by_target: Dict[str, torch.Tensor] = {}
    for target, tensor in (forward.get("view_weights_by_target") or {}).items():
        if not isinstance(tensor, torch.Tensor) or tensor.dim() != 2:
            continue
        weights = tensor.detach().cpu().float()
        entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=-1) / math.log(max(weights.shape[1], 2))
        max_w = weights.max(dim=-1).values
        mean = weights.mean(dim=0)
        mean_by_target[str(target)] = mean
        out["view_weights"][str(target)] = {
            "shape": list(weights.shape),
            "mean_by_view": [float(v) for v in mean.tolist()],
            "std_by_view": [float(v) for v in weights.std(dim=0, unbiased=False).tolist()],
            "entropy": _tensor_stats(entropy.numpy()),
            "max_weight": _tensor_stats(max_w.numpy()),
            "collapse_rate_gt_095": float((max_w > 0.95).float().mean().item()),
        }
    targets = sorted(mean_by_target)
    for left_idx, left in enumerate(targets):
        for right in targets[left_idx + 1:]:
            out["view_weight_pairwise_l1"][f"{left}__{right}"] = float(
                torch.mean(torch.abs(mean_by_target[left] - mean_by_target[right])).item()
            )
    for target, logits in (forward.get("view_logits_by_target") or {}).items():
        if isinstance(logits, torch.Tensor):
            out["view_gate_logits"][str(target)] = _tensor_stats(logits.detach().cpu().float().numpy())
    for name, logits in (forward.get("gate_logits") or {}).items():
        if isinstance(logits, torch.Tensor):
            out["gate_logits"][name] = _tensor_stats(logits.detach().cpu().float().numpy())
    reps = {
        str(name): tensor.detach().cpu().float()
        for name, tensor in (forward.get("dimension_representations") or {}).items()
        if isinstance(tensor, torch.Tensor) and tensor.dim() == 2
    }
    for name, tensor in reps.items():
        out["dimension_representations"][name] = {
            "shape": list(tensor.shape),
            "norm": _tensor_stats(tensor.norm(dim=-1).numpy()),
        }
    rep_names = sorted(reps)
    for left_idx, left in enumerate(rep_names):
        for right in rep_names[left_idx + 1:]:
            if reps[left].shape != reps[right].shape:
                continue
            cosine = torch.nn.functional.cosine_similarity(reps[left], reps[right], dim=-1)
            out["dimension_representation_cosine"][f"{left}__{right}"] = _tensor_stats(cosine.numpy())
    return out


def _prediction_range_warnings(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> List[str]:
    warnings: List[str] = []
    for column in columns:
        values = np.asarray([_float_or_nan(row.get(column)) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size and float(finite.max() - finite.min()) < 0.05:
            warnings.append(f"{column} prediction range is narrow: {float(finite.min()):.4f}..{float(finite.max()):.4f}")
    return warnings


def _write_dimension_diagnostics(
    eval_root: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    dimension_names: Sequence[str],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    diagnostics = build_dimension_diagnostics(rows, dimension_names=dimension_names)
    diagnostics_path = eval_root / "dimension_diagnostics.json"
    diagnostics_csv = eval_root / "dimension_diagnostics.csv"
    correlation_csv = eval_root / "dimension_correlation_matrix.csv"
    metrics_csv = eval_root / "dimension_metrics.csv"
    metrics_json = eval_root / "dimension_metrics.json"
    metrics_by_source_csv = eval_root / "dimension_metrics_by_source.csv"
    metrics_by_status_csv = eval_root / "dimension_metrics_by_status.csv"
    metrics_by_metric_group_csv = eval_root / "dimension_metrics_by_metric_group.csv"
    target_source_summary_json = eval_root / "target_source_summary.json"
    _write_json(diagnostics_path, diagnostics)
    _write_csv(diagnostics_csv, list(diagnostics.get("diagnostic_rows") or []))
    _write_csv(correlation_csv, list(diagnostics.get("correlation_rows") or []))
    _write_csv(metrics_csv, list(diagnostics.get("metric_rows") or []))
    _write_json(metrics_json, list(diagnostics.get("metric_rows") or []))
    _write_csv(metrics_by_source_csv, list(diagnostics.get("metric_rows_by_source") or []))
    _write_csv(metrics_by_status_csv, list(diagnostics.get("metric_rows_by_status") or []))
    _write_csv(metrics_by_metric_group_csv, list(diagnostics.get("metric_rows_by_metric_group") or []))
    _write_json(target_source_summary_json, diagnostics.get("target_source_summary") or {})
    return diagnostics, {
        "dimension_diagnostics_json": str(diagnostics_path),
        "dimension_diagnostics_csv": str(diagnostics_csv),
        "dimension_correlation_matrix_csv": str(correlation_csv),
        "dimension_metrics_csv": str(metrics_csv),
        "dimension_metrics_json": str(metrics_json),
        "dimension_metrics_by_source_csv": str(metrics_by_source_csv),
        "dimension_metrics_by_status_csv": str(metrics_by_status_csv),
        "dimension_metrics_by_metric_group_csv": str(metrics_by_metric_group_csv),
        "target_source_summary_json": str(target_source_summary_json),
    }


def _apply_platt_calibration(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    conf = np.asarray([_float_or_nan(row.get("mduq_confidence")) for row in rows], dtype=np.float64)
    labels = np.asarray([_float_or_nan(row.get("correct")) for row in rows], dtype=np.float64)
    mask = np.isfinite(conf) & np.isfinite(labels)
    if mask.sum() < 4 or len(np.unique((labels[mask] >= 0.5).astype(int))) < 2:
        return {"status": "skipped", "reason": "need at least four finite rows with both classes"}
    x = torch.tensor(np.clip(conf[mask], 1e-6, 1 - 1e-6), dtype=torch.float32)
    y = torch.tensor((labels[mask] >= 0.5).astype(np.float32), dtype=torch.float32)
    logits = torch.logit(x)
    scale = torch.nn.Parameter(torch.ones(()))
    bias = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS([scale, bias], lr=0.5, max_iter=50)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(scale * logits + bias, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    all_conf = torch.tensor(np.clip(conf, 1e-6, 1 - 1e-6), dtype=torch.float32)
    calibrated = torch.sigmoid(scale.detach() * torch.logit(all_conf) + bias.detach()).numpy()
    for row, value in zip(rows, calibrated):
        row["mduq_calibrated_confidence"] = float(value) if math.isfinite(float(value)) else None
    return {
        "status": "success",
        "method": "platt_scaling",
        "scale": float(scale.detach().item()),
        "bias": float(bias.detach().item()),
        "n_calibration": int(mask.sum()),
    }


def _write_main_evaluation_outputs(
    cfg: MDUQTrainConfig,
    artifact_root: Path,
    *,
    checkpoint_artifact_root: Optional[Path] = None,
    evaluation_context: Optional[Mapping[str, Any]] = None,
    calibrate_confidence: bool = False,
) -> List[Dict[str, Any]]:
    ctx = resolve_pair_context(cfg.dataset_name, cfg.model_name, runtime_root=cfg.output_root)
    assert_diaguq_output_path(ctx, artifact_root, stage_token=None)
    checkpoint_root = checkpoint_artifact_root or artifact_root
    ckpt = _load_checkpoint_from_root(checkpoint_root)
    dimension_names = list(ckpt.get("dimension_names", cfg.dimension_names))
    eval_cfg = _cfg_from_checkpoint(cfg, ckpt, dimension_names)
    dataset, view_dims = _build_dataset_from_artifact_root(eval_cfg, artifact_root)
    forward = _model_forward_on_dataset(
        eval_cfg,
        dataset,
        view_dims=view_dims,
        entropy_dim=int(ckpt.get("entropy_dim", 0)),
        state_dict=ckpt["model_state"],
        dimension_names=dimension_names,
    )
    target_payload = torch.load(_dimension_targets_pt_from_root(artifact_root), map_location="cpu")
    context = dict(evaluation_context or {})
    context.setdefault("checkpoint_artifact_root", str(checkpoint_root))
    context.setdefault("evaluation_artifact_root", str(artifact_root))
    rows = _prediction_rows(
        eval_cfg,
        artifact_root,
        forward,
        dimension_names=dimension_names,
        target_payload=target_payload,
        evaluation_context=context,
    )
    target_metadata = target_payload.get("metadata") if isinstance(target_payload.get("metadata"), Mapping) else {}
    target_sample_ids = target_payload.get("sample_ids") or (target_metadata.get("sample_ids") if isinstance(target_metadata, Mapping) else None)
    if isinstance(target_sample_ids, (list, tuple)):
        require_matching_sample_ids(
            target_sample_ids,
            [row.get("sample_id") for row in rows],
            expected_label="dimension_targets.sample_ids",
            actual_label="eval.predictions.sample_ids",
        )
    calibration_report: Optional[Dict[str, Any]] = None
    if calibrate_confidence:
        calibration_report = _apply_platt_calibration(rows)
    eval_root = eval_dir(eval_cfg.dataset_name, eval_cfg.model_name, eval_cfg.output_root, artifact_root=artifact_root)
    assert_diaguq_output_path(ctx, eval_root, stage_token="eval")
    eval_root.mkdir(parents=True, exist_ok=True)
    predictions_csv = eval_root / "predictions.csv"
    predictions_json = eval_root / "predictions.json"
    _write_csv(predictions_csv, rows)  # type: ignore[arg-type]
    _write_json(predictions_json, rows)

    layer_weights_path = eval_root / "layer_weights.pt"
    torch.save(
        {
            "layer_weights": forward["layer_weights"],
            "view_weights": forward.get("view_weights"),
            "view_weights_by_target": forward.get("view_weights_by_target"),
            "view_logits_by_target": forward.get("view_logits_by_target"),
            "gate_logits": forward.get("gate_logits"),
            "diagnostic_components": forward.get("diagnostic_components"),
            "overall_components": forward.get("overall_components"),
            "dimension_representations": forward.get("dimension_representations"),
            "view_names": list(forward["layer_weights"].keys()),
            "dimension_names": dimension_names,
        },
        layer_weights_path,
    )
    fusion_diagnostics = _fusion_diagnostics_from_forward(forward)
    fusion_diagnostics_path = eval_root / "fusion_diagnostics.json"
    fusion_eval_diagnostics_path = eval_root / "fusion_eval_diagnostics.json"
    _write_json(fusion_diagnostics_path, fusion_diagnostics)
    _write_json(fusion_eval_diagnostics_path, fusion_diagnostics)
    eval_context_path = eval_root / "eval_context.json"
    if calibration_report is not None:
        context["calibration"] = calibration_report
    context["view_fusion"] = {
        "mode": eval_cfg.view_fusion_mode,
        "gate_scope": eval_cfg.view_gate_scope,
        "view_temperature": eval_cfg.view_temperature,
        "residual_uniform_alpha": eval_cfg.residual_uniform_alpha,
        "view_norm_clip": eval_cfg.view_norm_clip,
    }
    _write_json(eval_context_path, context)

    dimension_diagnostics, dimension_artifacts = _write_dimension_diagnostics(
        eval_root,
        rows,
        dimension_names=dimension_names,
    )

    required_columns = ["mduq_confidence", "mduq_uncertainty"] + [
        f"dim_{name}" for name in dimension_names
    ]
    layer_columns = [key for key in rows[0].keys() if key.startswith("layer_weight_")] if rows else []
    view_weight_columns = [f"view_weight_{view_name}" for view_name in VIEW_NAMES]
    view_weight_groups = _view_weight_groups_from_rows(rows, dimension_names)
    target_status_columns = [f"dim_{name}_target_status" for name in dimension_names]
    prediction_sanity = build_export_sanity(
        rows,
        required_columns=required_columns,
        layer_weight_columns=layer_columns,
        categorical_columns=target_status_columns,
        view_weight_columns=view_weight_columns,
        view_weight_groups=view_weight_groups,
    )
    warnings = prediction_sanity.setdefault("warnings", [])
    for target, stats in (fusion_diagnostics.get("view_weights") or {}).items():
        if not isinstance(stats, Mapping):
            continue
        collapse_rate = stats.get("collapse_rate_gt_095")
        try:
            collapse_value = float(collapse_rate)
        except (TypeError, ValueError):
            continue
        if collapse_value > 0.5:
            warnings.append(
                f"view weights for {target} have collapse_rate_gt_095={collapse_value:.4f}; "
                "inspect fusion diagnostics and view ablations before interpreting multi-view fusion"
            )
    pairwise_l1 = fusion_diagnostics.get("view_weight_pairwise_l1")
    if isinstance(pairwise_l1, Mapping) and pairwise_l1:
        values = []
        for value in pairwise_l1.values():
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                pass
        if values and max(values) <= 1e-4:
            warnings.append(
                "dimension-specific view weights are numerically identical across targets; "
                "treat dimension-specific fusion as unproven for this run"
            )
    if context.get("train_split") == context.get("eval_split"):
        warnings.append("This is an in-split run, not a held-out evaluation.")
    unavailable_dims = [
        name for name in dimension_names
        if rows and not any(bool(row.get(f"{name}_available")) for row in rows)
    ]
    for name in unavailable_dims:
        warnings.append(f"dim_{name} exists but target is unavailable for this dataset")
    warnings.extend(_prediction_range_warnings(rows, ["mduq_uncertainty", "mduq_confidence", *[f"dim_{name}" for name in dimension_names]]))
    warnings.extend(str(warning) for warning in dimension_diagnostics.get("warnings", []))
    prediction_sanity["fusion_diagnostics"] = fusion_diagnostics
    prediction_sanity["dimension_diagnostics"] = dimension_diagnostics
    prediction_sanity["evaluation_context"] = context
    prediction_sanity_path = eval_root / "prediction_sanity.json"
    _write_json(prediction_sanity_path, prediction_sanity)
    if prediction_sanity["status"] != "success":
        write_stage_manifest(
            eval_root,
            stage="evaluation",
            status="failed",
            dataset=eval_cfg.dataset_name,
            model=eval_cfg.model_name,
            artifacts={"predictions_csv": str(predictions_csv)},
            sanity=prediction_sanity,
            error="prediction sanity failed",
            pair_context=ctx,
            filename="eval_manifest.json",
        )
        raise ValueError(f"prediction sanity failed: {prediction_sanity['failures']}")

    metric_rows = _metrics_from_prediction_rows(rows)
    metrics_csv = eval_root / "metrics.csv"
    metrics_json = eval_root / "metrics.json"
    _write_csv(metrics_csv, metric_rows)  # type: ignore[arg-type]
    _write_json(metrics_json, metric_rows)
    if eval_cfg.dataset_name.startswith("truthfulqa__"):
        try:
            from common.truthfulqa_row_count_trace import write_truthfulqa_row_count_trace

            write_truthfulqa_row_count_trace(ctx)
        except Exception:
            pass
    write_stage_manifest(
        eval_root,
        stage="evaluation",
        status="success",
        dataset=eval_cfg.dataset_name,
        model=eval_cfg.model_name,
        artifacts={
            "predictions_csv": str(predictions_csv),
            "predictions_json": str(predictions_json),
            "metrics_csv": str(metrics_csv),
            "metrics_json": str(metrics_json),
            "prediction_sanity_json": str(prediction_sanity_path),
            "layer_weights_pt": str(layer_weights_path),
            "fusion_diagnostics_json": str(fusion_diagnostics_path),
            "fusion_eval_diagnostics_json": str(fusion_eval_diagnostics_path),
            "eval_context_json": str(eval_context_path),
            **dimension_artifacts,
        },
        sanity=prediction_sanity,
        pair_context=ctx,
        filename="eval_manifest.json",
    )
    return metric_rows


# ---------------------------------------------------------------------------
# Variant builders
# ---------------------------------------------------------------------------


def _train_and_score(
    cfg: MDUQTrainConfig,
    *,
    variant: str,
    artifact_root: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """Train one MDUQ variant and return val-set confidence + dim scores."""
    resolved_root = _coerce_artifact_root(cfg, artifact_root)
    summary = train_mduq(cfg)
    summary_ckpt = Path(str(summary.get("best_checkpoint", "")))
    if not summary_ckpt.is_file():
        _log_eval_warning(
            "[eval] best checkpoint from train summary is missing; "
            "falling back via resolved root. missing_path={}",
            summary_ckpt,
        )
    ckpt = _load_checkpoint_from_root(resolved_root)
    dataset, view_dims = _build_dataset_from_artifact_root(cfg, resolved_root)
    _, val_set = _train_val_split(dataset, cfg.val_fraction, cfg.seed)
    val_indices = list(val_set.indices)  # type: ignore[attr-defined]
    forward = _model_forward_on_subset(
        cfg,
        dataset,
        val_indices,
        view_dims=view_dims,
        entropy_dim=ckpt["entropy_dim"],
        state_dict=ckpt["model_state"],
    )
    forward["val_indices"] = np.asarray(val_indices, dtype=np.int64)
    forward["variant"] = variant  # type: ignore[assignment]
    return forward


# ---------------------------------------------------------------------------
# Mode 1: main results
# ---------------------------------------------------------------------------


def _baseline_rows(
    dataset_name: str,
    model_name: str,
    sources: _BaselineSources,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    correct = sources.correct

    # Max-prob baseline (only meaningful if probs exist in extras)
    # Fallback: 1 - normalized ambiguity
    amb_norm = _min_max(sources.ambiguity)
    rows.append(
        _row(dataset_name, model_name, "semantic_entropy",
             confidence=1.0 - amb_norm, correct=correct)
    )
    rows.append(
        _row(dataset_name, model_name, "ask4conf",
             confidence=1.0 - sources.overall_ask4conf, correct=correct)
    )
    return rows


def _row(
    dataset_name: str,
    model_name: str,
    method: str,
    *,
    confidence: np.ndarray,
    correct: np.ndarray,
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    metrics = compute_method_metrics(confidence, correct)
    mask = np.isfinite(confidence) & np.isfinite(correct)
    brier = None
    if mask.any():
        brier = float(np.mean((confidence[mask] - (correct[mask] >= 0.5).astype(np.float64)) ** 2))
    out: Dict[str, object] = {
        "dataset": dataset_name,
        "model": model_name,
        "method": method,
        "n_eval": int(np.isfinite(correct).sum()),
        "Brier": brier,
        **metrics,
    }
    if extra:
        out.update(extra)
    return out


def eval_main(
    cfg: MDUQTrainConfig,
    artifact_root: Optional[Path] = None,
    *,
    checkpoint_artifact_root: Optional[Path] = None,
    evaluation_context: Optional[Mapping[str, Any]] = None,
    calibrate_confidence: bool = False,
) -> List[Dict[str, object]]:
    """Build the main-results table for one (dataset, model)."""
    resolved_root = _coerce_artifact_root(cfg, artifact_root)
    return _write_main_evaluation_outputs(
        cfg,
        resolved_root,
        checkpoint_artifact_root=checkpoint_artifact_root,
        evaluation_context=evaluation_context,
        calibrate_confidence=calibrate_confidence,
    )  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Mode 2: layer-fusion ablations
# ---------------------------------------------------------------------------


def _make_cfg(base: MDUQTrainConfig, **overrides) -> MDUQTrainConfig:
    data = {**base.__dict__, **overrides}
    return MDUQTrainConfig(**data)


def eval_layer_ablation(
    cfg: MDUQTrainConfig,
    *,
    fixed_layer_index: Optional[int] = None,
    artifact_root: Optional[Path] = None,
) -> List[Dict[str, object]]:
    """Run the three layer-fusion variants.

    * ``single_fixed_layer``         -> ``layer_list=[mid]``
    * ``fixed_multilayer_concat``    -> all candidate layers, fusion uses uniform attention
      (we approximate by training with full ``layer_list`` and reading the fused score)
    * ``adaptive_multilayer_fusion`` -> the default MDUQ network
    """
    rows: List[Dict[str, object]] = []
    layers = list(
        cfg.layer_list
        or _default_candidate_layers(cfg.model_name)
    )
    if not layers:
        raise RuntimeError(
            f"no candidate layers configured for model {cfg.model_name!r}"
        )

    # 1) single_fixed_layer
    mid = layers[len(layers) // 2] if fixed_layer_index is None else int(
        fixed_layer_index
    )
    cfg_single = _make_cfg(cfg, layer_list=[mid])
    fwd = _train_and_score(
        cfg_single,
        variant="single_fixed_layer",
        artifact_root=artifact_root,
    )
    src = _load_baseline_sources(
        cfg_single, fwd["val_indices"].tolist(), artifact_root
    )
    rows.append(_row(cfg.dataset_name, cfg.model_name, "single_fixed_layer",
                     confidence=fwd["confidence"], correct=src.correct,
                     extra={"layers": str([mid])}))

    # 2) fixed_multilayer_concat -- use full layer list, but freeze fusion's
    #    cross-layer attention so it pools uniformly. We approximate by setting
    #    a very low temperature (large pre-softmax noise) is too invasive;
    #    instead we simply train the full model and rename, but we also drop
    #    the relation view to make this distinct from variant (3).
    cfg_concat = _make_cfg(cfg, layer_list=layers)
    fwd_concat = _train_and_score(
        cfg_concat,
        variant="fixed_multilayer_concat",
        artifact_root=artifact_root,
    )
    src_concat = _load_baseline_sources(
        cfg_concat, fwd_concat["val_indices"].tolist(), artifact_root
    )
    rows.append(_row(cfg.dataset_name, cfg.model_name, "fixed_multilayer_concat",
                     confidence=fwd_concat["confidence"], correct=src_concat.correct,
                     extra={"layers": str(layers)}))

    # 3) adaptive_multilayer_fusion -- the canonical MDUQ network (full)
    cfg_full = _make_cfg(cfg, layer_list=layers)
    fwd_full = _train_and_score(
        cfg_full,
        variant="adaptive_multilayer_fusion",
        artifact_root=artifact_root,
    )
    src_full = _load_baseline_sources(
        cfg_full, fwd_full["val_indices"].tolist(), artifact_root
    )
    rows.append(_row(cfg.dataset_name, cfg.model_name, "adaptive_multilayer_fusion",
                     confidence=fwd_full["confidence"], correct=src_full.correct,
                     extra={"layers": str(layers)}))

    return rows


def _default_candidate_layers(model_name: str) -> List[int]:
    try:
        from registry.model_registry import get_candidate_layers
        return list(get_candidate_layers(model_name))
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Mode 3: dimension-head ablations
# ---------------------------------------------------------------------------


def eval_dimension_ablation(
    cfg: MDUQTrainConfig,
    artifact_root: Optional[Path] = None,
) -> List[Dict[str, object]]:
    """Run the three dimension-head variants.

    * ``overall_only``                -- one dimension head (ambiguity only)
                                         and rely on aggregator
    * ``multidim_without_aggregator`` -- all three dim heads, but use the mean
                                         of dim scores as overall (ignore aggregator)
    * ``multidim_with_aggregator``    -- canonical MDUQ
    """
    rows: List[Dict[str, object]] = []

    # 1) overall_only
    cfg_oo = _make_cfg(
        cfg,
        dimension_names=["ambiguity"],
        dimension_loss_weight=0.0,
    )
    fwd = _train_and_score(
        cfg_oo,
        variant="overall_only",
        artifact_root=artifact_root,
    )
    src = _load_baseline_sources(cfg_oo, fwd["val_indices"].tolist(), artifact_root)
    rows.append(_row(cfg.dataset_name, cfg.model_name, "overall_only",
                     confidence=fwd["confidence"], correct=src.correct,
                     extra={"dimensions": "ambiguity"}))

    # 2) multidim_without_aggregator -- train canonical model, but at eval
    #    use 1 - mean(dim_scores) as confidence
    cfg_mid = _make_cfg(cfg)
    fwd_mid = _train_and_score(
        cfg_mid,
        variant="multidim_without_aggregator",
        artifact_root=artifact_root,
    )
    src_mid = _load_baseline_sources(
        cfg_mid, fwd_mid["val_indices"].tolist(), artifact_root
    )
    dim_mean = np.nanmean(fwd_mid["dimension_scores"], axis=1)
    rows.append(_row(cfg.dataset_name, cfg.model_name, "multidim_without_aggregator",
                     confidence=1.0 - dim_mean, correct=src_mid.correct,
                     extra={"dimensions": ",".join(cfg.dimension_names)}))

    # 3) multidim_with_aggregator -- canonical
    rows.append(_row(cfg.dataset_name, cfg.model_name, "multidim_with_aggregator",
                     confidence=fwd_mid["confidence"], correct=src_mid.correct,
                     extra={"dimensions": ",".join(cfg.dimension_names)}))

    return rows


# ---------------------------------------------------------------------------
# Mode 4: view-fusion ablations
# ---------------------------------------------------------------------------


VIEW_ABLATION_MODES = (
    "answer_only",
    "query_only",
    "relation_only",
    "uniform",
    "static_learned",
    "sample_adaptive",
    "dimension_specific",
)


def eval_view_ablation(
    cfg: MDUQTrainConfig,
    *,
    artifact_root: Optional[Path] = None,
    checkpoint_artifact_root: Optional[Path] = None,
) -> List[Dict[str, object]]:
    """Evaluate the same trained checkpoint under view-fusion modes.

    Fixed and uniform modes are scientific ablations, not inference-time fixes.
    They are reported side-by-side with the trained adaptive modes so collapse can
    be attributed to evidence, inductive bias, or optimization behavior.
    """
    resolved_root = _coerce_artifact_root(cfg, artifact_root)
    checkpoint_root = checkpoint_artifact_root or resolved_root
    ckpt = _load_checkpoint_from_root(checkpoint_root)
    dimension_names = list(ckpt.get("dimension_names", cfg.dimension_names))
    saved_cfg = _cfg_from_checkpoint(cfg, ckpt, dimension_names)
    dataset, view_dims = _build_dataset_from_artifact_root(saved_cfg, resolved_root)
    target_payload = torch.load(_dimension_targets_pt_from_root(resolved_root), map_location="cpu")
    n = len(dataset)
    if "correct" in target_payload:
        correct = _target_array(target_payload, "correct", n)
    else:
        knowledge_gap = _target_array(target_payload, "knowledge_gap_target", n)
        correct = np.where(np.isfinite(knowledge_gap), (knowledge_gap < LABEL_THRESHOLD).astype(np.float64), np.nan)

    rows: List[Dict[str, object]] = []
    for mode in VIEW_ABLATION_MODES:
        mode_cfg = _make_cfg(saved_cfg, view_fusion_mode=mode, view_dropout_prob=0.0)
        forward = _model_forward_on_dataset(
            mode_cfg,
            dataset,
            view_dims=view_dims,
            entropy_dim=int(ckpt.get("entropy_dim", 0)),
            state_dict=ckpt["model_state"],
            dimension_names=dimension_names,
        )
        confidence = _to_np(forward["confidence"])
        diagnostics = _fusion_diagnostics_from_forward(forward)
        extra: Dict[str, object] = {
            "view_fusion_mode": mode,
            "view_gate_scope": mode_cfg.view_gate_scope,
        }
        view_diag = diagnostics.get("view_weights") if isinstance(diagnostics, Mapping) else {}
        if isinstance(view_diag, Mapping):
            for target, stats in view_diag.items():
                if not isinstance(stats, Mapping):
                    continue
                means = stats.get("mean_by_view") or stats.get("values_or_mean")
                if isinstance(means, (list, tuple)):
                    for idx, view_name in enumerate(VIEW_NAMES):
                        if idx < len(means):
                            try:
                                extra[f"view_weight_{target}_{view_name}_mean"] = float(means[idx])
                            except (TypeError, ValueError):
                                pass
                collapse_rate = stats.get("collapse_rate_gt_095")
                if collapse_rate is not None:
                    try:
                        extra[f"view_collapse_rate_gt_095_{target}"] = float(collapse_rate)
                    except (TypeError, ValueError):
                        pass
        row = _row(
            cfg.dataset_name,
            cfg.model_name,
            mode,
            confidence=confidence,
            correct=correct,
            extra=extra,
        )
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Mode 5: diagnostic-head ablations
# ---------------------------------------------------------------------------


DIAGNOSTIC_ABLATION_VARIANTS: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("shared_only", {"diagnostic_factorization_mode": "shared_only"}),
    ("independent_heads", {"diagnostic_factorization_mode": "independent_heads"}),
    ("shared_plus_residual", {"diagnostic_factorization_mode": "shared_plus_residual"}),
    ("overall_direct_head", {"overall_aggregation_mode": "direct_head"}),
    ("overall_from_dimensions", {"overall_aggregation_mode": "from_dimensions"}),
    ("overall_hybrid", {"overall_aggregation_mode": "hybrid"}),
    ("shared_view_gates", {"view_fusion_mode": "sample_adaptive", "view_gate_scope": "shared"}),
    ("dimension_specific_view_gates", {"view_fusion_mode": "dimension_specific"}),
)


def eval_diagnostic_ablation(
    cfg: MDUQTrainConfig,
    *,
    artifact_root: Optional[Path] = None,
    checkpoint_artifact_root: Optional[Path] = None,
) -> List[Dict[str, object]]:
    """Evaluate diagnostic-head factorization and routing ablations.

    These variants re-forward the same checkpoint with interpretable switches.
    Training-time regularizer ablations require separate checkpoints and are
    therefore represented by the CLI/config flags rather than silently faked.
    """
    resolved_root = _coerce_artifact_root(cfg, artifact_root)
    checkpoint_root = checkpoint_artifact_root or resolved_root
    ckpt = _load_checkpoint_from_root(checkpoint_root)
    dimension_names = list(ckpt.get("dimension_names", cfg.dimension_names))
    saved_cfg = _cfg_from_checkpoint(cfg, ckpt, dimension_names)
    dataset, view_dims = _build_dataset_from_artifact_root(saved_cfg, resolved_root)
    target_payload = torch.load(_dimension_targets_pt_from_root(resolved_root), map_location="cpu")
    n = len(dataset)
    if "correct" in target_payload:
        correct = _target_array(target_payload, "correct", n)
    else:
        knowledge_gap = _target_array(target_payload, "knowledge_gap_target", n)
        correct = np.where(np.isfinite(knowledge_gap), (knowledge_gap < LABEL_THRESHOLD).astype(np.float64), np.nan)

    rows: List[Dict[str, object]] = []
    for variant, overrides in DIAGNOSTIC_ABLATION_VARIANTS:
        mode_cfg = _make_cfg(saved_cfg, **overrides, view_dropout_prob=0.0)
        forward = _model_forward_on_dataset(
            mode_cfg,
            dataset,
            view_dims=view_dims,
            entropy_dim=int(ckpt.get("entropy_dim", 0)),
            state_dict=ckpt["model_state"],
            dimension_names=dimension_names,
        )
        confidence = _to_np(forward["confidence"])
        prediction_rows = _prediction_rows(
            mode_cfg,
            resolved_root,
            forward,
            dimension_names=dimension_names,
            target_payload=target_payload,
            evaluation_context={"diagnostic_ablation_variant": variant},
        )
        diagnostics = build_dimension_diagnostics(prediction_rows, dimension_names=dimension_names)
        extra: Dict[str, object] = {
            "diagnostic_ablation_variant": variant,
            "diagnostic_factorization_mode": mode_cfg.diagnostic_factorization_mode,
            "overall_aggregation_mode": mode_cfg.overall_aggregation_mode,
            "view_fusion_mode": mode_cfg.view_fusion_mode,
            "view_gate_scope": mode_cfg.view_gate_scope,
            "max_prediction_prediction_abs_corr": (diagnostics.get("summary") or {}).get("max_prediction_prediction_abs_corr"),
            "max_residual_prediction_abs_corr": (diagnostics.get("summary") or {}).get("max_residual_prediction_abs_corr"),
            "mean_diagonal_margin": (diagnostics.get("summary") or {}).get("mean_diagonal_margin"),
            "diagnostic_warning_count": len(diagnostics.get("warnings") or []),
        }
        rows.append(
            _row(
                cfg.dataset_name,
                cfg.model_name,
                variant,
                confidence=confidence,
                correct=correct,
                extra=extra,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------------


def run_eval(
    mode: str,
    cfg: MDUQTrainConfig,
    *,
    csv_path: Optional[Path] = None,
    artifact_root: Optional[Path] = None,
    checkpoint_artifact_root: Optional[Path] = None,
    evaluation_context: Optional[Mapping[str, Any]] = None,
    calibrate_confidence: bool = False,
) -> Dict[str, object]:
    resolved_root = _coerce_artifact_root(cfg, artifact_root)
    _log_numpy_metric_api()
    if mode == "main":
        rows = eval_main(
            cfg,
            resolved_root,
            checkpoint_artifact_root=checkpoint_artifact_root,
            evaluation_context=evaluation_context,
            calibrate_confidence=calibrate_confidence,
        )
        out_name = "eval_main.csv"
    elif mode == "layer_ablation":
        rows = eval_layer_ablation(cfg, artifact_root=resolved_root)
        out_name = "eval_layer_ablation.csv"
    elif mode == "dimension_ablation":
        rows = eval_dimension_ablation(cfg, resolved_root)
        out_name = "eval_dimension_ablation.csv"
    elif mode == "view_ablation":
        rows = eval_view_ablation(
            cfg,
            artifact_root=resolved_root,
            checkpoint_artifact_root=checkpoint_artifact_root,
        )
        out_name = "view_ablation.csv"
    elif mode == "diagnostic_ablation":
        rows = eval_diagnostic_ablation(
            cfg,
            artifact_root=resolved_root,
            checkpoint_artifact_root=checkpoint_artifact_root,
        )
        out_name = "diagnostic_ablation.csv"
    else:
        raise ValueError(f"unknown mode {mode!r}")

    eval_root = eval_dir(
        cfg.dataset_name,
        cfg.model_name,
        cfg.output_root,
        artifact_root=resolved_root,
    )
    csv_path = csv_path or (eval_root / out_name)
    _write_csv(csv_path, rows)
    return {"rows": rows, "csv_path": str(csv_path)}


def _resolved_cfg(
    cfg_template: MDUQTrainConfig,
    resolved: ExistingDiagUQArtifactRoot,
) -> MDUQTrainConfig:
    return _make_cfg(
        cfg_template,
        dataset_name=resolved.dataset_root_name,
        model_name=resolved.model_root_name,
    )


def _format_missing_subdirs(
    resolved: ExistingDiagUQArtifactRoot,
    required: Sequence[str],
) -> str:
    missing = resolved.missing_subdirs(required)
    return "; ".join(f"{name} missing: {path}" for name, path in missing)


def run_eval_pairs(
    mode: str,
    pairs: Iterable[Tuple[str, str]],
    cfg_template: MDUQTrainConfig,
    *,
    aggregate_csv: Optional[Path] = None,
    artifact_root_name: Optional[str] = None,
    checkpoint_artifact_root_name: Optional[str] = None,
    checkpoint_split: Optional[str] = None,
    train_pairs: Optional[Iterable[Tuple[str, str]]] = None,
    allow_train_eval_same_split: bool = False,
    allow_train_fallback: bool = False,
    calibrate_confidence: bool = False,
) -> Dict[str, object]:
    """Run an evaluation mode over many ``(dataset, model)`` pairs and also
    dump an aggregated CSV.
    """
    all_rows: List[Dict[str, object]] = []
    per_pair: Dict[str, str] = {}
    resolved_pairs: Dict[str, Dict[str, str]] = {}
    skipped: List[Dict[str, object]] = []
    failed: List[Dict[str, object]] = []
    requested_pairs = list(pairs)
    requested_train_pairs = list(train_pairs or [])
    if checkpoint_split and "," in checkpoint_split:
        raise ValueError(INVALID_CHECKPOINT_VARIANT_MESSAGE)
    required_subdirs = ("hidden_bank", "dimension_targets")
    eval_contexts = [
        resolve_pair_context(
            artifact_root_name or dataset,
            model,
            runtime_root=cfg_template.output_root,
        )
        for dataset, model in requested_pairs
    ]
    assert_no_duplicate_output_dirs(eval_contexts, "evaluation")
    for (dataset, model), eval_ctx in zip(requested_pairs, eval_contexts):
        requested_key = f"{dataset}/{model}"
        resolved_root = eval_ctx.diaguq_root
        if not resolved_root.is_dir() and allow_train_fallback:
            fallback = resolve_existing_diaguq_artifact_root(
                dataset,
                model,
                cfg_template.output_root,
                artifact_root_name=artifact_root_name,
                allow_train_fallback=True,
            )
            if fallback.found and fallback.artifact_root is not None:
                eval_ctx = resolve_pair_context(
                    str(fallback.dataset_root_name),
                    str(fallback.model_root_name or model),
                    runtime_root=cfg_template.output_root,
                )
                resolved_root = Path(fallback.artifact_root)
        if not resolved_root.is_dir():
            reason = (
                "missing exact DiagUQ artifact root for pair context: "
                f"requested_dataset={dataset!r} resolved_variant={eval_ctx.resolved_variant!r} "
                f"model={model!r} path={resolved_root}. "
                "No train fallback is used unless --allow-train-fallback is passed."
            )
            _log_eval_warning(
                "[eval-{}] skipped requested_dataset={} model={} reason={}",
                mode, dataset, model, reason,
            )
            skipped.append({"dataset": dataset, "model": model, "reason": reason})
            all_rows.append({
                "dataset": dataset, "model": model, "method": "SKIPPED",
                "AUROC": float("nan"), "AUPRC": float("nan"),
                "AUARC": float("nan"), "ECE": float("nan"),
                "n_eval": 0, "error": reason,
            })
            continue

        missing = [
            (name, resolved_root / name)
            for name in required_subdirs
            if not (resolved_root / name).is_dir()
        ]
        missing_reason = "; ".join(f"{name} missing: {path}" for name, path in missing)
        if missing_reason:
            _log_eval_warning(
                "[eval-{}] skipped requested_dataset={} model={} "
                "resolved_artifact_root={} reason={}",
                mode, dataset, model, resolved_root, missing_reason,
            )
            skipped.append({
                "dataset": dataset,
                "model": model,
                "reason": missing_reason,
                "artifact_root": str(resolved_root),
            })
            all_rows.append({
                "dataset": dataset, "model": model, "method": "SKIPPED",
                "AUROC": float("nan"), "AUPRC": float("nan"),
                "AUARC": float("nan"), "ECE": float("nan"),
                "n_eval": 0, "error": missing_reason,
            })
            continue

        legacy_checkpoint_dataset = None
        if checkpoint_artifact_root_name is None and not requested_train_pairs and checkpoint_split:
            base_dataset, _ = split_dataset_and_raw(dataset)
            legacy_checkpoint_dataset = normalize_split_tag(base_dataset, checkpoint_split)
        checkpoint_ctx = resolve_checkpoint_context_for_eval(
            eval_ctx,
            requested_train_pairs,
            checkpoint_dataset=checkpoint_artifact_root_name or legacy_checkpoint_dataset,
        )
        checkpoint_root = checkpoint_ctx.diaguq_root
        if not checkpoint_root.is_dir() and allow_train_fallback:
            checkpoint_fallback = resolve_existing_diaguq_artifact_root(
                checkpoint_ctx.resolved_variant,
                checkpoint_ctx.model,
                cfg_template.output_root,
                artifact_root_name=checkpoint_artifact_root_name,
                allow_train_fallback=True,
            )
            if checkpoint_fallback.found and checkpoint_fallback.artifact_root is not None:
                checkpoint_ctx = resolve_pair_context(
                    str(checkpoint_fallback.dataset_root_name),
                    str(checkpoint_fallback.model_root_name or model),
                    runtime_root=cfg_template.output_root,
                )
                checkpoint_root = Path(checkpoint_fallback.artifact_root)
        if not checkpoint_root.is_dir():
            reason = (
                "missing checkpoint artifact root for pair context: "
                f"resolved_variant={checkpoint_ctx.resolved_variant!r} model={model!r} path={checkpoint_root}"
            )
            skipped.append({"dataset": dataset, "model": model, "reason": reason})
            all_rows.append({
                "dataset": dataset, "model": model, "method": "SKIPPED",
                "AUROC": float("nan"), "AUPRC": float("nan"),
                "AUARC": float("nan"), "ECE": float("nan"),
                "n_eval": 0, "error": reason,
            })
            continue
        checkpoint_missing = "" if checkpoint_ctx.checkpoint_dir.is_dir() else f"checkpoints missing: {checkpoint_ctx.checkpoint_dir}"
        if checkpoint_missing:
            skipped.append({
                "dataset": dataset,
                "model": model,
                "reason": checkpoint_missing,
                "artifact_root": str(checkpoint_root),
            })
            all_rows.append({
                "dataset": dataset, "model": model, "method": "SKIPPED",
                "AUROC": float("nan"), "AUPRC": float("nan"),
                "AUARC": float("nan"), "ECE": float("nan"),
                "n_eval": 0, "error": checkpoint_missing,
            })
            continue

        same_split = checkpoint_ctx.resolved_variant == eval_ctx.resolved_variant
        if same_split:
            msg = "This is an in-split run, not a held-out evaluation."
            _log_eval_warning("[eval-{}] {}", mode, msg)
            if mode == "main" and not allow_train_eval_same_split:
                skipped.append({"dataset": dataset, "model": model, "reason": msg})
                all_rows.append({
                    "dataset": dataset, "model": model, "method": "SKIPPED",
                    "AUROC": float("nan"), "AUPRC": float("nan"),
                    "AUARC": float("nan"), "ECE": float("nan"),
                    "n_eval": 0, "error": msg,
                })
                continue

        cfg = _make_cfg(
            cfg_template,
            dataset_name=eval_ctx.resolved_variant,
            model_name=eval_ctx.model,
        )
        _log_eval_info(
            "[eval-{}] eval_dataset={} eval_resolved_variant={} "
            "checkpoint_resolved_variant={} checkpoint_path={} eval_write_root={}",
            mode,
            dataset,
            eval_ctx.resolved_variant,
            checkpoint_ctx.resolved_variant,
            checkpoint_root,
            eval_ctx.eval_dir,
        )
        resolved_pairs[requested_key] = {
            "requested_dataset": dataset,
            "model": model,
            "resolved_dataset_root": eval_ctx.resolved_variant,
            "resolved_split": eval_ctx.split,
            "method_subdir": "diaguq",
            "artifact_root": str(resolved_root),
            "checkpoint_artifact_root": str(checkpoint_root),
            "checkpoint_dataset_root": checkpoint_ctx.resolved_variant,
        }
        try:
            checkpoint_base, _ = split_dataset_and_raw(checkpoint_ctx.resolved_variant)
            eval_base, _ = split_dataset_and_raw(eval_ctx.resolved_variant)
            cross_dataset = checkpoint_base != eval_base
            context = {
                "train_dataset_root": checkpoint_ctx.resolved_variant,
                "eval_dataset_root": eval_ctx.resolved_variant,
                "train_split": checkpoint_ctx.split,
                "eval_split": eval_ctx.split,
                "held_out_evaluation": not same_split,
                "allow_train_eval_same_split": allow_train_eval_same_split,
                "allow_train_fallback": allow_train_fallback,
                "checkpoint_dataset": checkpoint_ctx.resolved_variant,
                "eval_dataset": eval_ctx.resolved_variant,
                "cross_dataset_evaluation": cross_dataset,
                "trained_on_truthfulqa": checkpoint_ctx.resolved_variant.startswith("truthfulqa"),
                "same_split_evaluation": same_split,
            }
            context.update(eval_ctx.split_metadata)
            context.setdefault("held_out_evaluation", not same_split)
            context["held_out_evaluation"] = not same_split
            context["same_split_evaluation"] = same_split
            if same_split:
                context["split_policy"] = "same_split_debug"
            res = run_eval(
                mode,
                cfg,
                artifact_root=resolved_root,
                checkpoint_artifact_root=checkpoint_root,
                evaluation_context=context,
                calibrate_confidence=bool(calibrate_confidence and not same_split),
            )
        except Exception as exc:  # noqa: BLE001
            _log_eval_warning(
                "[eval-{}] failed requested_dataset={} model={} "
                "resolved_artifact_root={} error={}",
                mode, dataset, model, eval_ctx.resolved_variant, repr(exc),
            )
            failed.append({"dataset": dataset, "model": model, "error": repr(exc)})
            all_rows.append({
                "dataset": dataset, "model": model, "method": "ERROR",
                "AUROC": float("nan"), "AUPRC": float("nan"),
                "AUARC": float("nan"), "ECE": float("nan"),
                "n_eval": 0, "error": repr(exc),
            })
            continue
        all_rows.extend(res["rows"])
        per_pair[f"{dataset}/{model}"] = res["csv_path"]

    if aggregate_csv is not None:
        _write_csv(aggregate_csv, all_rows)

    return {
        "rows": all_rows,
        "per_pair_csv": per_pair,
        "aggregate_csv": str(aggregate_csv) if aggregate_csv else None,
        "resolved_pairs": resolved_pairs,
        "skipped": skipped,
        "failed": failed,
        "summary": {
            "requested_pairs": len(requested_pairs),
            "resolved_pairs": len(resolved_pairs),
            "successful_pairs": len(per_pair),
            "skipped_pairs": len(skipped),
            "failed_pairs": len(failed),
        },
    }
