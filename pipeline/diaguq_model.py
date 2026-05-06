"""MDUQ main model.

Three small ``nn.Module``\\s are combined into the end-to-end MDUQ network:

* :class:`LayerFusionModule` -- attention-pool the per-layer
  query / answer / relation tensors produced by
    ``features.build_multiview_features`` into a single fused vector and expose
  the (per-view) layer weights for inspection.
* :class:`DimensionHeads` -- one MLP head per uncertainty dimension
  (defaults: ``ambiguity``, ``knowledge_gap``, ``predictive_variability``);
  produces a scalar score per dimension.
* :class:`OverallAggregator` -- mixes the dimension scores with the fused
  internal representation to predict the final overall uncertainty.

The composite :class:`MDUQModel` returns both the predictions and the
fusion diagnostics (layer weights, dimension scores) so trainers can log
or persist them. This module is **independent** of the legacy
``RandomForest`` calibration code in ``supervised_calibration.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


DEFAULT_VIEWS: Tuple[str, ...] = ("query", "answer", "relation")
DEFAULT_DIMENSIONS: Tuple[str, ...] = (
    "ambiguity",
    "knowledge_gap",
    "predictive_variability",
)
VIEW_FUSION_MODES: Tuple[str, ...] = (
    "answer_only",
    "query_only",
    "relation_only",
    "uniform",
    "static_learned",
    "sample_adaptive",
    "sample_adaptive_regularized",
    "dimension_specific",
)
VIEW_GATE_SCOPES: Tuple[str, ...] = ("shared", "dimension_specific")
DIAGNOSTIC_FACTORIZATION_MODES: Tuple[str, ...] = (
    "shared_only",
    "independent_heads",
    "shared_plus_residual",
)
OVERALL_AGGREGATION_MODES: Tuple[str, ...] = (
    "direct_head",
    "from_dimensions",
    "hybrid",
)


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(
            f"{name} contains non-finite values: "
            f"nan={int(torch.isnan(tensor).sum().item())} "
            f"inf={int(torch.isinf(tensor).sum().item())} "
            f"shape={tuple(tensor.shape)}"
        )


def _row_sum_assert(name: str, weights: torch.Tensor, *, atol: float = 1e-4) -> None:
    row_sums = weights.sum(dim=-1)
    if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=atol):
        max_error = float((row_sums - 1.0).abs().max().detach().cpu().item())
        raise ValueError(f"{name} rows do not sum to 1 within {atol}: max_error={max_error:.8f}")


# ---------------------------------------------------------------------------
# Layer fusion
# ---------------------------------------------------------------------------


class _PerViewLayerAttention(nn.Module):
    """Soft-attention pooling over the layer axis for a single view."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float,
        *,
        softmax_temperature: float = 1.5,
        layer_dropout: float = 0.05,
        gate_logit_clip: float = 10.0,
        residual_uniform_alpha: float = 0.0,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.attn = nn.Linear(in_dim, 1)
        self.softmax_temperature = max(float(softmax_temperature), 1e-6)
        self.layer_dropout = float(layer_dropout)
        self.gate_logit_clip = float(gate_logit_clip)
        self.residual_uniform_alpha = float(max(0.0, min(1.0, residual_uniform_alpha)))

    def _apply_layer_dropout(self, scores: torch.Tensor) -> torch.Tensor:
        if not self.training or self.layer_dropout <= 0 or scores.shape[-1] <= 1:
            return scores
        keep = torch.rand_like(scores) >= self.layer_dropout
        empty = ~keep.any(dim=-1, keepdim=True)
        if empty.any():
            fallback = torch.zeros_like(keep)
            fallback.scatter_(-1, scores.argmax(dim=-1, keepdim=True), True)
            keep = torch.where(empty, fallback, keep)
        return scores.masked_fill(~keep, -1e4)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (N, L, D_in)
        if x.dim() != 3:
            raise ValueError(
                f"per-view tensor must be (N, L, D); got {tuple(x.shape)}"
            )
        _require_finite("per-view fusion input", x)
        x_norm = self.input_norm(x)
        scores = self.attn(x_norm).squeeze(-1) / self.softmax_temperature
        if self.gate_logit_clip > 0:
            scores = scores.clamp(-self.gate_logit_clip, self.gate_logit_clip)
        scores = self._apply_layer_dropout(scores)
        _require_finite("layer attention logits", scores)
        weights = torch.softmax(scores, dim=-1)         # (N, L)
        if self.residual_uniform_alpha > 0 and weights.shape[-1] > 1:
            uniform = torch.full_like(weights, 1.0 / float(weights.shape[-1]))
            weights = (1.0 - self.residual_uniform_alpha) * weights + self.residual_uniform_alpha * uniform
        _require_finite("layer attention weights", weights)
        _row_sum_assert("layer attention weights", weights)
        projected = self.proj(x_norm)                   # (N, L, D_out)
        _require_finite("per-view projected features", projected)
        pooled = (weights.unsqueeze(-1) * projected).sum(dim=1)  # (N, D_out)
        _require_finite("per-view pooled features", pooled)
        return pooled, weights, scores


class LayerFusionModule(nn.Module):
    """Fuse multi-layer per-view tensors into one shared representation.

    Parameters
    ----------
    view_dims:
        Mapping from view name (``"query"``, ``"answer"``, ``"relation"``) to
        its per-layer hidden size ``D_v``. Only views present in this map
        are accepted at ``forward`` time.
    fusion_dim:
        Output dim of each per-view projection and of the fused vector.
    hidden_dim:
        Inner width of the per-view projection MLP.
    dropout:
        Dropout applied inside the projection MLP.
    """

    def __init__(
        self,
        view_dims: Mapping[str, int],
        dimension_names: Sequence[str] = DEFAULT_DIMENSIONS,
        fusion_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        layer_softmax_temperature: float = 1.5,
        layer_dropout: float = 0.05,
        layer_residual_uniform_alpha: float = 0.0,
        gate_logit_clip: float = 10.0,
        view_gate_hidden_dim: Optional[int] = None,
        view_temperature: float = 2.0,
        view_temperature_min: float = 0.5,
        view_temperature_max: float = 10.0,
        residual_uniform_alpha: float = 0.05,
        view_norm_clip: Optional[float] = None,
        view_dropout_prob: float = 0.1,
        view_gate_scope: str = "shared",
        view_fusion_mode: str = "dimension_specific",
    ):
        super().__init__()
        if not view_dims:
            raise ValueError("view_dims must be non-empty")
        if view_fusion_mode not in VIEW_FUSION_MODES:
            raise ValueError(f"unsupported view_fusion_mode: {view_fusion_mode!r}")
        if view_gate_scope not in VIEW_GATE_SCOPES:
            raise ValueError(f"unsupported view_gate_scope: {view_gate_scope!r}")
        self.view_names: List[str] = list(view_dims.keys())
        self.dimension_names: List[str] = list(dimension_names)
        self.fusion_dim = int(fusion_dim)
        self.view_fusion_mode = view_fusion_mode
        self.view_gate_scope = "dimension_specific" if view_fusion_mode == "dimension_specific" else view_gate_scope
        self.view_temperature = float(min(max(view_temperature, view_temperature_min), view_temperature_max))
        self.view_temperature_min = float(view_temperature_min)
        self.view_temperature_max = float(view_temperature_max)
        self.residual_uniform_alpha = float(max(0.0, min(1.0, residual_uniform_alpha)))
        self.view_norm_clip = None if view_norm_clip is None else float(view_norm_clip)
        self.view_dropout_prob = float(max(0.0, min(1.0, view_dropout_prob)))
        self.per_view = nn.ModuleDict(
            {
                name: _PerViewLayerAttention(
                    in_dim=int(dim),
                    hidden_dim=int(hidden_dim),
                    out_dim=self.fusion_dim,
                    dropout=float(dropout),
                    softmax_temperature=layer_softmax_temperature,
                    layer_dropout=layer_dropout,
                    gate_logit_clip=gate_logit_clip,
                    residual_uniform_alpha=layer_residual_uniform_alpha,
                )
                for name, dim in view_dims.items()
            }
        )
        for name, module in self.per_view.items():
            setattr(self, f"{name}_view_proj", module.proj)
        view_gate_hidden_dim = int(view_gate_hidden_dim or max(32, self.fusion_dim))
        self.view_logit_prior = nn.Parameter(torch.zeros(len(self.view_names)))
        self.static_view_logits = nn.Parameter(torch.zeros(len(self.view_names)))
        self.view_gate = nn.Sequential(
            nn.LayerNorm(self.fusion_dim * len(self.view_names)),
            nn.Linear(self.fusion_dim * len(self.view_names), view_gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(view_gate_hidden_dim, len(self.view_names)),
        )
        gate_targets = ["overall", *self.dimension_names]
        self.dimension_view_gates = nn.ModuleDict(
            {
                target: nn.Sequential(
                    nn.LayerNorm(self.fusion_dim * len(self.view_names)),
                    nn.Linear(self.fusion_dim * len(self.view_names), view_gate_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(view_gate_hidden_dim, len(self.view_names)),
                )
                for target in gate_targets
            }
        )
        self.gate_logit_clip = float(gate_logit_clip)
        self.norm = nn.LayerNorm(self.fusion_dim)

    @property
    def output_dim(self) -> int:
        return self.fusion_dim

    @property
    def uses_dimension_specific_gates(self) -> bool:
        return self.view_fusion_mode == "dimension_specific" or self.view_gate_scope == "dimension_specific"

    def _clip_view_norms(self, stacked: torch.Tensor) -> torch.Tensor:
        if self.view_norm_clip is None or self.view_norm_clip <= 0:
            return stacked
        norms = stacked.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        scale = torch.clamp(float(self.view_norm_clip) / norms, max=1.0)
        return stacked * scale

    def _fixed_weights(self, batch_size: int, *, device: torch.device, dtype: torch.dtype, mode: str) -> torch.Tensor:
        weights = torch.zeros(batch_size, len(self.view_names), device=device, dtype=dtype)
        if mode == "uniform":
            weights.fill_(1.0 / float(len(self.view_names)))
            return weights
        view_name = mode.removesuffix("_only")
        if view_name not in self.view_names:
            raise ValueError(f"view_fusion_mode={mode!r} requires view {view_name!r}; available={self.view_names}")
        weights[:, self.view_names.index(view_name)] = 1.0
        return weights

    def _apply_view_dropout(self, logits: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not self.training or self.view_dropout_prob <= 0 or logits.shape[-1] <= 1:
            return logits, None
        n, num_views = logits.shape
        drop_sample = torch.rand(n, device=logits.device) < self.view_dropout_prob
        if not bool(drop_sample.any()):
            return logits, None
        drop_idx = torch.randint(num_views, (n,), device=logits.device)
        keep = torch.ones_like(logits, dtype=torch.bool)
        keep[torch.arange(n, device=logits.device), drop_idx] = ~drop_sample
        return logits.masked_fill(~keep, -1e4), keep.float()

    def _weights_from_logits(
        self,
        logits: torch.Tensor,
        *,
        allow_dropout: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.gate_logit_clip > 0:
            logits = logits.clamp(-self.gate_logit_clip, self.gate_logit_clip)
        dropped_mask = None
        if allow_dropout:
            logits, dropped_mask = self._apply_view_dropout(logits)
        _require_finite("view logits", logits)
        raw = torch.softmax(logits / max(self.view_temperature, 1e-6), dim=-1)
        if self.residual_uniform_alpha > 0 and raw.shape[-1] > 1:
            uniform = torch.full_like(raw, 1.0 / float(raw.shape[-1]))
            weights = (1.0 - self.residual_uniform_alpha) * raw + self.residual_uniform_alpha * uniform
        else:
            weights = raw
        _require_finite("view weights", weights)
        _row_sum_assert("view weights", weights)
        return weights, raw, dropped_mask

    def _fuse(self, stacked: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)
        fused = self.norm(fused)
        _require_finite("fused features", fused)
        return fused

    def forward(
        self, views: Mapping[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        missing = [n for n in self.view_names if n not in views]
        if missing:
            raise KeyError(f"missing views in input: {missing}")

        per_view_pool: List[torch.Tensor] = []
        layer_weights: Dict[str, torch.Tensor] = {}
        gate_logits: Dict[str, torch.Tensor] = {}
        for name in self.view_names:
            pooled, w, logits = self.per_view[name](views[name])
            per_view_pool.append(pooled)
            layer_weights[name] = w
            gate_logits[name] = logits

        batch_size = per_view_pool[0].shape[0]
        for name, pooled in zip(self.view_names, per_view_pool):
            if pooled.shape != (batch_size, self.fusion_dim):
                raise ValueError(
                    f"view {name!r} representation shape mismatch: {tuple(pooled.shape)}; "
                    f"expected {(batch_size, self.fusion_dim)}"
                )
            _require_finite(f"projected view representation {name}", pooled)

        stacked = torch.stack(per_view_pool, dim=1)       # (N, V, F)
        stacked = self._clip_view_norms(stacked)
        gate_input = torch.cat([stacked[:, idx, :] for idx in range(stacked.shape[1])], dim=-1)
        fused_by_target: Dict[str, torch.Tensor] = {}
        mode = self.view_fusion_mode
        fixed_modes = {"answer_only", "query_only", "relation_only", "uniform"}

        if mode in fixed_modes:
            view_weights = self._fixed_weights(batch_size, device=stacked.device, dtype=stacked.dtype, mode=mode)
            view_logits = torch.log(view_weights.clamp_min(1e-8))
            fused = self._fuse(stacked, view_weights)
            fused_by_target = {"overall": fused, **{name: fused for name in self.dimension_names}}
            layer_weights["_view_weights"] = view_weights
            layer_weights["_view_weights_overall"] = view_weights
            gate_logits["_view_logits"] = view_logits.detach()
            gate_logits["_view_logits_overall"] = view_logits.detach()
            return fused, layer_weights, gate_logits, fused_by_target

        if mode == "static_learned":
            view_logits = self.static_view_logits.view(1, -1).expand(batch_size, -1)
            view_weights, raw_weights, _ = self._weights_from_logits(view_logits, allow_dropout=False)
            fused = self._fuse(stacked, view_weights)
            fused_by_target = {"overall": fused, **{name: fused for name in self.dimension_names}}
            layer_weights["_view_weights"] = view_weights
            layer_weights["_view_weights_raw"] = raw_weights
            layer_weights["_view_weights_overall"] = view_weights
            gate_logits["_view_logits"] = view_logits.detach()
            gate_logits["_view_logits_overall"] = view_logits.detach()
            return fused, layer_weights, gate_logits, fused_by_target

        if self.uses_dimension_specific_gates:
            targets = ["overall", *self.dimension_names]
            for target in targets:
                view_logits = self.dimension_view_gates[target](gate_input) + self.view_logit_prior.view(1, -1)
                view_weights, raw_weights, keep_mask = self._weights_from_logits(view_logits, allow_dropout=True)
                fused_by_target[target] = self._fuse(stacked, view_weights)
                suffix = target
                layer_weights[f"_view_weights_{suffix}"] = view_weights
                layer_weights[f"_view_weights_raw_{suffix}"] = raw_weights
                gate_logits[f"_view_logits_{suffix}"] = view_logits.detach()
                if keep_mask is not None:
                    gate_logits[f"_view_dropout_keep_{suffix}"] = keep_mask.detach()
            layer_weights["_view_weights"] = layer_weights["_view_weights_overall"]
            gate_logits["_view_logits"] = gate_logits["_view_logits_overall"]
            return fused_by_target["overall"], layer_weights, gate_logits, fused_by_target

        view_logits = self.view_gate(gate_input) + self.view_logit_prior.view(1, -1)
        allow_dropout = mode == "sample_adaptive_regularized"
        view_weights, raw_weights, keep_mask = self._weights_from_logits(view_logits, allow_dropout=allow_dropout)
        fused = self._fuse(stacked, view_weights)
        fused_by_target = {"overall": fused, **{name: fused for name in self.dimension_names}}
        layer_weights["_view_weights"] = view_weights
        layer_weights["_view_weights_raw"] = raw_weights
        layer_weights["_view_weights_overall"] = view_weights
        gate_logits["_view_logits"] = view_logits.detach()
        gate_logits["_view_logits_overall"] = view_logits.detach()
        if keep_mask is not None:
            gate_logits["_view_dropout_keep_overall"] = keep_mask.detach()
        return fused, layer_weights, gate_logits, fused_by_target


# ---------------------------------------------------------------------------
# Dimension heads
# ---------------------------------------------------------------------------


class DimensionHeads(nn.Module):
    """Diagnostic heads with shared uncertainty plus dimension residuals.

    ``shared_plus_residual`` keeps the natural shared uncertainty factor while
    giving each diagnostic dimension its own residual representation and score.
    ``independent_heads`` keeps the legacy behavior conceptually, and
    ``shared_only`` is an ablation that intentionally removes residual evidence.
    """

    def __init__(
        self,
        in_dim: int,
        dimension_names: Sequence[str] = DEFAULT_DIMENSIONS,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        extra_in_dim: int = 0,
        output_activation: Optional[str] = "sigmoid",
        diagnostic_factorization_mode: str = "shared_plus_residual",
    ):
        super().__init__()
        if not dimension_names:
            raise ValueError("dimension_names must be non-empty")
        if diagnostic_factorization_mode not in DIAGNOSTIC_FACTORIZATION_MODES:
            raise ValueError(f"unsupported diagnostic_factorization_mode: {diagnostic_factorization_mode!r}")
        self.dimension_names: List[str] = list(dimension_names)
        self.in_dim = int(in_dim) + int(extra_in_dim)
        self.hidden_dim = int(hidden_dim)
        self.diagnostic_factorization_mode = diagnostic_factorization_mode
        self.shared_block = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.shared_head = nn.Linear(hidden_dim, 1)
        self.dimension_blocks = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(self.in_dim),
                    nn.Linear(self.in_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(hidden_dim),
                )
                for name in self.dimension_names
            }
        )
        self.heads = nn.ModuleDict(
            {
                name: nn.Linear(hidden_dim, 1)
                for name in self.dimension_names
            }
        )
        self.shared_alpha = nn.Parameter(torch.ones(len(self.dimension_names)))
        if output_activation not in (None, "sigmoid"):
            raise ValueError(
                f"unsupported output_activation: {output_activation!r}"
            )
        self.output_activation = output_activation

    @property
    def num_dimensions(self) -> int:
        return len(self.dimension_names)

    def forward(
        self,
        fused: torch.Tensor,
        extra: Optional[torch.Tensor] = None,
        fused_by_dimension: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        def _with_extra(base: torch.Tensor) -> torch.Tensor:
            if extra is not None:
                _require_finite("dimension-head extra features", extra)
                return torch.cat([base, extra], dim=-1)
            return base

        shared_input = _with_extra(fused)
        _require_finite("shared diagnostic input", shared_input)
        shared_hidden = self.shared_block(shared_input)
        shared_logit = self.shared_head(shared_hidden).squeeze(-1)
        shared_score = torch.sigmoid(shared_logit) if self.output_activation == "sigmoid" else shared_logit
        components: Dict[str, torch.Tensor] = {
            "shared_uncertainty_hidden": shared_hidden,
            "shared_uncertainty_logit": shared_logit,
            "shared_uncertainty_score": shared_score,
        }
        representations: Dict[str, torch.Tensor] = {}
        per_dim_logits: List[torch.Tensor] = []
        per_dim_named: Dict[str, torch.Tensor] = {}
        fused_by_dimension = fused_by_dimension or {}
        for idx, name in enumerate(self.dimension_names):
            dim_fused = fused_by_dimension.get(name, fused)
            dim_input = _with_extra(dim_fused)
            _require_finite(f"diagnostic residual input {name}", dim_input)
            residual_hidden = self.dimension_blocks[name](dim_input)
            residual_logit = self.heads[name](residual_hidden).squeeze(-1)
            residual_score = torch.sigmoid(residual_logit) if self.output_activation == "sigmoid" else residual_logit
            if self.diagnostic_factorization_mode == "shared_only":
                logit = shared_logit
            elif self.diagnostic_factorization_mode == "independent_heads":
                logit = residual_logit
            else:
                logit = self.shared_alpha[idx] * shared_logit + residual_logit
            per_dim_logits.append(logit)
            per_dim_named[name] = logit
            representations[name] = residual_hidden
            components[f"residual_{name}_hidden"] = residual_hidden
            components[f"residual_{name}_logit"] = residual_logit
            components[f"residual_{name}_score"] = residual_score
            components[f"diagnostic_alpha_{name}"] = self.shared_alpha[idx].expand_as(logit)
        logits = torch.stack(per_dim_logits, dim=-1)  # (N, K)
        if self.output_activation == "sigmoid":
            scores = torch.sigmoid(logits)
        else:
            scores = logits
        _require_finite("dimension scores", scores)
        return scores, per_dim_named, components, representations


# ---------------------------------------------------------------------------
# Overall aggregator
# ---------------------------------------------------------------------------


class OverallAggregator(nn.Module):
    """Combine dimension scores with the fused representation.

    The overall head is intentionally light-weight: it concatenates the
    dimension score vector with a small projection of the fused
    representation so the trainer can supervise it against
    ``overall_target`` (or its ``1 - ask4conf`` fallback).
    """

    def __init__(
        self,
        fused_dim: int,
        num_dimensions: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        output_activation: Optional[str] = "sigmoid",
        overall_aggregation_mode: str = "hybrid",
    ):
        super().__init__()
        if overall_aggregation_mode not in OVERALL_AGGREGATION_MODES:
            raise ValueError(f"unsupported overall_aggregation_mode: {overall_aggregation_mode!r}")
        self.overall_aggregation_mode = overall_aggregation_mode
        self.num_dimensions = int(num_dimensions)
        self.proj = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim + int(num_dimensions), 1)
        self.dimension_weight_logits = nn.Parameter(torch.zeros(int(num_dimensions)))
        self.hybrid_gate = nn.Sequential(
            nn.Linear(hidden_dim + int(num_dimensions), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        if output_activation not in (None, "sigmoid"):
            raise ValueError(
                f"unsupported output_activation: {output_activation!r}"
            )
        self.output_activation = output_activation

    def forward(
        self, fused: torch.Tensor, dimension_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        h = self.proj(fused)
        x = torch.cat([h, dimension_scores], dim=-1)
        direct_logit = self.head(x).squeeze(-1)
        weights = torch.softmax(self.dimension_weight_logits, dim=0)
        from_dimensions = (dimension_scores * weights.view(1, -1)).sum(dim=-1)
        if self.output_activation == "sigmoid":
            direct = torch.sigmoid(direct_logit)
            if self.overall_aggregation_mode == "direct_head":
                uncertainty = direct
            elif self.overall_aggregation_mode == "from_dimensions":
                uncertainty = from_dimensions
            else:
                gate = torch.sigmoid(self.hybrid_gate(x).squeeze(-1))
                uncertainty = gate * direct + (1.0 - gate) * from_dimensions
        else:
            direct = direct_logit
            gate = torch.sigmoid(self.hybrid_gate(x).squeeze(-1))
            if self.overall_aggregation_mode == "direct_head":
                uncertainty = direct
            elif self.overall_aggregation_mode == "from_dimensions":
                uncertainty = from_dimensions
            else:
                uncertainty = gate * direct + (1.0 - gate) * from_dimensions
        if self.output_activation == "sigmoid" and self.overall_aggregation_mode != "hybrid":
            gate = torch.ones_like(uncertainty) if self.overall_aggregation_mode == "direct_head" else torch.zeros_like(uncertainty)
        confidence = 1.0 - uncertainty
        _require_finite("overall uncertainty", uncertainty)
        _require_finite("overall confidence", confidence)
        components = {
            "overall_direct": direct,
            "overall_from_dimensions": from_dimensions,
            "overall_final": uncertainty,
            "overall_hybrid_gate": gate,
            "overall_dimension_weights": weights.view(1, -1).expand(dimension_scores.shape[0], -1),
        }
        return uncertainty, confidence, components


# ---------------------------------------------------------------------------
# Composite model
# ---------------------------------------------------------------------------


@dataclass
class MDUQOutput:
    """Container for one forward pass."""

    uncertainty: torch.Tensor                       # (N,)
    confidence: torch.Tensor                        # (N,)
    dimension_scores: torch.Tensor                  # (N, K)
    dimension_named: Dict[str, torch.Tensor] = field(default_factory=dict)
    layer_weights: Dict[str, torch.Tensor] = field(default_factory=dict)
    gate_logits: Dict[str, torch.Tensor] = field(default_factory=dict)
    fused: Optional[torch.Tensor] = None            # (N, F)
    fused_by_target: Dict[str, torch.Tensor] = field(default_factory=dict)
    diagnostic_components: Dict[str, torch.Tensor] = field(default_factory=dict)
    overall_components: Dict[str, torch.Tensor] = field(default_factory=dict)
    dimension_representations: Dict[str, torch.Tensor] = field(default_factory=dict)

    def to_serializable(self) -> Dict[str, torch.Tensor]:
        """Flatten to a ``{str: cpu tensor}`` dict for ``torch.save``."""
        out: Dict[str, torch.Tensor] = {
            "uncertainty": self.uncertainty.detach().cpu(),
            "confidence": self.confidence.detach().cpu(),
            "dimension_scores": self.dimension_scores.detach().cpu(),
        }
        for k, v in self.dimension_named.items():
            out[f"dim/{k}"] = v.detach().cpu()
        for k, v in self.layer_weights.items():
            out[f"layer_weights/{k}"] = v.detach().cpu()
        for k, v in self.gate_logits.items():
            out[f"gate_logits/{k}"] = v.detach().cpu()
        for k, v in self.diagnostic_components.items():
            out[f"diagnostic/{k}"] = v.detach().cpu()
        for k, v in self.overall_components.items():
            out[f"overall/{k}"] = v.detach().cpu()
        for k, v in self.dimension_representations.items():
            out[f"dimension_repr/{k}"] = v.detach().cpu()
        if self.fused is not None:
            out["fused"] = self.fused.detach().cpu()
        for k, v in self.fused_by_target.items():
            out[f"fused/{k}"] = v.detach().cpu()
        return out


class MDUQModel(nn.Module):
    """End-to-end MDUQ network.

    Example
    -------
    >>> model = MDUQModel(
    ...     view_dims={"query": 4096, "answer": 4096, "relation": 8192 + 17},
    ...     dimension_names=("ambiguity", "knowledge_gap", "predictive_variability"),
    ...     entropy_dim=6,
    ... )
    >>> out = model(views, entropy=entropy_features)
    >>> out.uncertainty.shape, out.dimension_scores.shape
    (torch.Size([N]), torch.Size([N, 3]))
    """

    def __init__(
        self,
        view_dims: Mapping[str, int],
        *,
        dimension_names: Sequence[str] = DEFAULT_DIMENSIONS,
        fusion_dim: int = 256,
        fusion_hidden_dim: int = 512,
        head_hidden_dim: int = 128,
        overall_hidden_dim: int = 128,
        dropout: float = 0.1,
        entropy_dim: int = 0,
        layer_softmax_temperature: float = 1.5,
        layer_dropout: float = 0.05,
        layer_residual_uniform_alpha: float = 0.0,
        gate_logit_clip: float = 10.0,
        view_gate_hidden_dim: Optional[int] = None,
        view_temperature: float = 2.0,
        view_temperature_min: float = 0.5,
        view_temperature_max: float = 10.0,
        residual_uniform_alpha: float = 0.05,
        view_norm_clip: Optional[float] = None,
        view_dropout_prob: float = 0.1,
        view_gate_scope: str = "shared",
        view_fusion_mode: str = "dimension_specific",
        diagnostic_factorization_mode: str = "shared_plus_residual",
        overall_aggregation_mode: str = "hybrid",
        head_output_activation: Optional[str] = "sigmoid",
        overall_output_activation: Optional[str] = "sigmoid",
    ):
        super().__init__()
        self.fusion = LayerFusionModule(
            view_dims=view_dims,
            dimension_names=dimension_names,
            fusion_dim=fusion_dim,
            hidden_dim=fusion_hidden_dim,
            dropout=dropout,
            layer_softmax_temperature=layer_softmax_temperature,
            layer_dropout=layer_dropout,
            layer_residual_uniform_alpha=layer_residual_uniform_alpha,
            gate_logit_clip=gate_logit_clip,
            view_gate_hidden_dim=view_gate_hidden_dim,
            view_temperature=view_temperature,
            view_temperature_min=view_temperature_min,
            view_temperature_max=view_temperature_max,
            residual_uniform_alpha=residual_uniform_alpha,
            view_norm_clip=view_norm_clip,
            view_dropout_prob=view_dropout_prob,
            view_gate_scope=view_gate_scope,
            view_fusion_mode=view_fusion_mode,
        )
        self.heads = DimensionHeads(
            in_dim=self.fusion.output_dim,
            dimension_names=dimension_names,
            hidden_dim=head_hidden_dim,
            dropout=dropout,
            extra_in_dim=int(entropy_dim),
            output_activation=head_output_activation,
            diagnostic_factorization_mode=diagnostic_factorization_mode,
        )
        self.aggregator = OverallAggregator(
            fused_dim=self.fusion.output_dim,
            num_dimensions=self.heads.num_dimensions,
            hidden_dim=overall_hidden_dim,
            dropout=dropout,
            output_activation=overall_output_activation,
            overall_aggregation_mode=overall_aggregation_mode,
        )
        self.entropy_dim = int(entropy_dim)
        self.view_fusion_mode = view_fusion_mode
        self.view_gate_scope = view_gate_scope
        self.diagnostic_factorization_mode = diagnostic_factorization_mode
        self.overall_aggregation_mode = overall_aggregation_mode

    @property
    def dimension_names(self) -> List[str]:
        return self.heads.dimension_names

    @property
    def view_names(self) -> List[str]:
        return self.fusion.view_names

    def forward(
        self,
        views: Mapping[str, torch.Tensor],
        entropy: Optional[torch.Tensor] = None,
    ) -> MDUQOutput:
        if self.entropy_dim > 0 and entropy is None:
            raise ValueError(
                "model was configured with entropy_dim>0 but no entropy "
                "tensor was provided"
            )
        if entropy is not None and self.entropy_dim == 0:
            # Quietly ignore: the heads weren't sized for it.
            entropy = None
        for view_name, tensor in views.items():
            _require_finite(f"model input view {view_name}", tensor)
        if entropy is not None:
            _require_finite("model entropy input", entropy)

        fused, layer_weights, gate_logits, fused_by_target = self.fusion(views)
        fused_by_dimension = {name: fused_by_target.get(name, fused) for name in self.dimension_names}
        dim_scores, dim_named, diagnostic_components, dimension_representations = self.heads(
            fused,
            extra=entropy,
            fused_by_dimension=fused_by_dimension,
        )
        uncertainty, confidence, overall_components = self.aggregator(fused, dim_scores)
        return MDUQOutput(
            uncertainty=uncertainty,
            confidence=confidence,
            dimension_scores=dim_scores,
            dimension_named=dim_named,
            layer_weights=layer_weights,
            gate_logits=gate_logits,
            fused=fused,
            fused_by_target=fused_by_target,
            diagnostic_components=diagnostic_components,
            overall_components=overall_components,
            dimension_representations=dimension_representations,
        )


# ---------------------------------------------------------------------------
# DiagUQ-style aliases. The class name `MDUQModel` is preserved for
# backward compatibility; new code should use `DiagUQModel`.
# ---------------------------------------------------------------------------

DiagUQModel = MDUQModel
DiagUQOutput = MDUQOutput
