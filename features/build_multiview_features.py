"""Multi-view feature builders for the DiagUQ pipeline.

Given a multi-layer hidden bank (see
:func:`features.load_feature_tensors.load_multilayer_feature_bank`), the
four ``build_*_view`` helpers in this module construct the canonical
feature views consumed by the DiagUQ trainer:

* :func:`build_query_view`     -- ``(N, L, D)`` per-layer question-side hidden
* :func:`build_answer_view`    -- ``(N, L, D)`` per-layer answer-side hidden
* :func:`build_relation_view`  -- ``(N, L, F)`` query<->answer interaction
  feature; default ops cover ``concat``, ``abs_diff``, ``prod``, ``cosine``
  and ``norm_stats``
* :func:`build_entropy_view`   -- ``(N, F_e)`` entropy / sorted-probability
  side-channel features

All inputs/outputs are :class:`torch.Tensor` (CPU). Functions are pure and
do not touch disk -- I/O is the loader's job.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------

# Op names accepted by build_relation_view
RELATION_OPS_DEFAULT: Tuple[str, ...] = (
    "concat",
    "abs_diff",
    "prod",
    "cosine",
    "norm_stats",
)


def _select_kind(view_dict: Mapping[str, torch.Tensor], kind: str) -> torch.Tensor:
    if kind not in view_dict:
        available = sorted(view_dict.keys())
        raise KeyError(
            f"kind {kind!r} not present in view dict; available: {available}"
        )
    return view_dict[kind]


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(
            f"{name} contains non-finite values: "
            f"nan={int(torch.isnan(tensor).sum().item())} "
            f"inf={int(torch.isinf(tensor).sum().item())} "
            f"shape={tuple(tensor.shape)}"
        )


def _source_path(extra_paths: Optional[Mapping[str, Any]], key: str) -> str:
    if not extra_paths:
        return "<unknown>"
    value = extra_paths.get(key)
    return str(value) if value is not None else "<unknown>"


def _per_column_counts(tensor: torch.Tensor) -> list[dict[str, int]]:
    values = tensor.detach().to("cpu")
    if values.dim() == 1:
        values = values.unsqueeze(-1)
    rows = []
    for idx in range(values.shape[-1]):
        col = values[..., idx].reshape(-1)
        rows.append(
            {
                "column": idx,
                "nan_count": int(torch.isnan(col).sum().item()),
                "inf_count": int(torch.isinf(col).sum().item()),
                "finite_count": int(torch.isfinite(col).sum().item()),
            }
        )
    return rows


def _availability_mask(
    tensor: Optional[torch.Tensor],
    explicit_mask: Optional[Any],
    n_ref: int,
) -> torch.Tensor:
    if tensor is None:
        return torch.zeros(n_ref, dtype=torch.bool)
    finite_mask = torch.isfinite(tensor.float())
    if finite_mask.dim() > 1:
        finite_mask = finite_mask.reshape(finite_mask.shape[0], -1).all(dim=1)
    else:
        finite_mask = finite_mask.reshape(-1)
    if explicit_mask is None:
        return finite_mask[:n_ref]
    mask = explicit_mask.detach().to("cpu").bool().reshape(-1)
    out = torch.zeros(n_ref, dtype=torch.bool)
    limit = min(n_ref, int(mask.shape[0]), int(finite_mask.shape[0]))
    out[:limit] = mask[:limit] & finite_mask[:limit]
    return out


def _sanitize_optional_tensor(
    name: str,
    tensor: torch.Tensor,
    available: torch.Tensor,
    *,
    required: bool,
    source_path: str,
) -> torch.Tensor:
    values = tensor.float().detach().to("cpu")
    if values.dim() == 1:
        values = values.unsqueeze(-1)
    if required:
        try:
            _require_finite(name, values)
        except ValueError as exc:
            raise ValueError(
                f"{exc}; source_path={source_path}; per_column={_per_column_counts(values)}"
            ) from exc
        return values
    finite_values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))
    if available.shape[0] != finite_values.shape[0]:
        aligned = torch.zeros(finite_values.shape[0], dtype=torch.bool)
        limit = min(int(available.shape[0]), int(finite_values.shape[0]))
        aligned[:limit] = available[:limit]
        available = aligned
    return torch.where(available.view(-1, 1), finite_values, torch.zeros_like(finite_values))


def _ensure_3d(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Coerce ``(N, D)`` -> ``(N, 1, D)`` so callers can mix average and
    last-token tensors uniformly. ``(N, L, D)`` is returned as-is. Anything
    else raises.
    """
    if tensor.dim() == 2:
        out = tensor.unsqueeze(1)
        _require_finite(name, out)
        return out
    if tensor.dim() == 3:
        _require_finite(name, tensor)
        return tensor
    raise ValueError(
        f"{name} must have 2 or 3 dims (N, D) or (N, L, D); "
        f"got {tuple(tensor.shape)}"
    )


def _collapse_options(tensor: torch.Tensor, option_idx: Optional[torch.Tensor]) -> torch.Tensor:
    """Reduce an MMLU answer tensor of shape ``(N, L, num_options, D)`` to
    ``(N, L, D)`` by gathering along the option axis.

    If ``option_idx`` is ``None`` we average over options as a safe default;
    callers that want the model's argmax option should pass it in.
    """
    if tensor.dim() != 4:
        _require_finite("answer option tensor", tensor)
        return tensor
    if option_idx is None:
        out = tensor.mean(dim=2)
        _require_finite("collapsed answer options", out)
        return out
    if option_idx.dim() != 1 or option_idx.shape[0] != tensor.shape[0]:
        raise ValueError(
            "option_idx must be shape (N,); got "
            f"{tuple(option_idx.shape)} for tensor {tuple(tensor.shape)}"
        )
    n, l, _, d = tensor.shape
    gather_idx = (
        option_idx.to(tensor.device).long().view(n, 1, 1, 1).expand(n, l, 1, d)
    )
    out = tensor.gather(dim=2, index=gather_idx).squeeze(2)
    _require_finite("collapsed answer options", out)
    return out


# ---------------------------------------------------------------------------
# Public view builders
# ---------------------------------------------------------------------------


def build_query_view(
    bank: Mapping[str, object],
    kind: str = "average",
) -> torch.Tensor:
    """Return the multi-layer query view as ``(N, L, D)``.

    ``bank`` is the mapping returned by ``load_multilayer_feature_bank``; the
    ``"query"`` entry must contain ``kind`` (typically ``"average"`` or
    ``"last_1_token"``).
    """
    query_dict = bank["query"]  # type: ignore[index]
    if not isinstance(query_dict, Mapping):
        raise TypeError("bank['query'] must be a mapping of kind -> tensor")
    tensor = _select_kind(query_dict, kind)
    return _ensure_3d(tensor, f"query[{kind}]")


def build_answer_view(
    bank: Mapping[str, object],
    kind: str = "average",
    option_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return the multi-layer answer view as ``(N, L, D)``.

    For MMLU the bank stores per-option tensors of shape
    ``(N, L, num_options, D)``; pass ``option_idx`` (e.g. the model's argmax
    over option logits) to gather the chosen option, otherwise the options
    are averaged.
    """
    answer_dict = bank["answer"]  # type: ignore[index]
    if not isinstance(answer_dict, Mapping):
        raise TypeError("bank['answer'] must be a mapping of kind -> tensor")
    tensor = _select_kind(answer_dict, kind)
    tensor = _collapse_options(tensor, option_idx)
    return _ensure_3d(tensor, f"answer[{kind}]")


def build_relation_view(
    query: torch.Tensor,
    answer: torch.Tensor,
    ops: Sequence[str] = RELATION_OPS_DEFAULT,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Build the query<->answer relation view.

    ``query`` and ``answer`` are both ``(N, L, D)``. The output concatenates
    one block per requested op along the last axis, yielding ``(N, L, F)``.

    Supported ops:

    * ``"concat"``      -> ``[q ; a]``                        (2D)
    * ``"abs_diff"``    -> ``|q - a|``                        (D)
    * ``"prod"``        -> ``q * a``                          (D)
    * ``"cosine"``      -> per-layer cosine similarity        (1)
    * ``"norm_stats"``  -> ``[||q||, ||a||, ||q-a||, q.a]``   (4)
    """
    if query.shape != answer.shape:
        raise ValueError(
            "query and answer must have identical shape; got "
            f"{tuple(query.shape)} vs {tuple(answer.shape)}"
        )
    if query.dim() != 3:
        raise ValueError(
            f"query/answer must be 3D (N, L, D); got {tuple(query.shape)}"
        )
    _require_finite("relation query input", query)
    _require_finite("relation answer input", answer)

    pieces = []
    diff = query - answer
    prod = query * answer
    q_norm = query.norm(dim=-1, keepdim=True)
    a_norm = answer.norm(dim=-1, keepdim=True)
    diff_norm = diff.norm(dim=-1, keepdim=True)
    dot = (query * answer).sum(dim=-1, keepdim=True)

    for op in ops:
        if op == "concat":
            pieces.append(torch.cat([query, answer], dim=-1))
        elif op == "abs_diff":
            pieces.append(diff.abs())
        elif op == "prod":
            pieces.append(prod)
        elif op == "cosine":
            cos = dot / (q_norm * a_norm).clamp_min(eps)
            pieces.append(cos)
        elif op == "norm_stats":
            pieces.append(torch.cat([q_norm, a_norm, diff_norm, dot], dim=-1))
        else:
            raise ValueError(
                f"unknown relation op {op!r}; "
                f"supported: {RELATION_OPS_DEFAULT}"
            )

    relation = torch.cat(pieces, dim=-1)
    _require_finite("relation view", relation)
    return relation


def build_entropy_view(
    extras: Mapping[str, torch.Tensor],
    entropy_key: str = "query_entropies",
    probs_key: str = "query_probs",
    top_k: Optional[int] = None,
    *,
    require_entropy: bool = False,
    include_availability: bool = True,
    extra_paths: Optional[Mapping[str, Any]] = None,
) -> torch.Tensor:
    """Build the entropy / probability side-channel view as ``(N, F_e)``.

    The output concatenates the per-example entropy with the sorted (desc)
    output probabilities, matching the legacy ``entropy_features`` block.
    In optional mode, non-finite or unavailable rows are represented by finite
    placeholders plus appended availability indicator columns. Required mode
    raises with source-path and per-column context instead.
    """
    entropy = extras.get(entropy_key)
    probs = extras.get(probs_key)
    entropy_mask = extras.get(f"{entropy_key[:-1] if entropy_key.endswith('s') else entropy_key}_available")
    if entropy_mask is None:
        entropy_mask = extras.get("query_entropy_available")
    probs_mask = extras.get(f"{probs_key[:-1] if probs_key.endswith('s') else probs_key}_available")
    if probs_mask is None:
        probs_mask = extras.get("query_prob_available")

    if require_entropy and entropy is None:
        raise KeyError(
            f"required entropy source missing: {entropy_key!r}; "
            f"available: {sorted(extras.keys())}"
        )
    if require_entropy and probs is None:
        raise KeyError(
            f"required probability source missing: {probs_key!r}; "
            f"available: {sorted(extras.keys())}"
        )
    if entropy is None and probs is None:
        raise KeyError(
            f"extras has neither {entropy_key!r} nor {probs_key!r}; "
            f"available: {sorted(extras.keys())}"
        )

    parts = []
    availability_parts = []
    n_ref: Optional[int] = None

    if entropy is not None:
        ent = entropy
        if ent.dim() == 1:
            ent = ent.unsqueeze(-1)
        n_ref = ent.shape[0]
        available = _availability_mask(ent, entropy_mask, int(n_ref))
        if require_entropy and not bool(available.all().item()):
            raise ValueError(
                "required entropy features are unavailable for "
                f"{int((~available).sum().item())} row(s); "
                f"source_path={_source_path(extra_paths, entropy_key)}"
            )
        ent = _sanitize_optional_tensor(
            "entropy features",
            ent,
            available,
            required=require_entropy,
            source_path=_source_path(extra_paths, entropy_key),
        )
        parts.append(ent.float())
        availability_parts.append(available.float().view(-1, 1))

    if probs is not None:
        prob_values = probs
        if prob_values.dim() == 1:
            prob_values = prob_values.unsqueeze(-1)
        if n_ref is None:
            n_ref = prob_values.shape[0]
        available = _availability_mask(prob_values, probs_mask, int(n_ref))
        if require_entropy and not bool(available.all().item()):
            raise ValueError(
                "required probability features are unavailable for "
                f"{int((~available).sum().item())} row(s); "
                f"source_path={_source_path(extra_paths, probs_key)}"
            )
        prob_values = _sanitize_optional_tensor(
            "probability features",
            prob_values,
            available,
            required=require_entropy,
            source_path=_source_path(extra_paths, probs_key),
        )
        sorted_probs = torch.sort(prob_values.float(), dim=-1, descending=True).values
        sorted_probs = torch.where(
            torch.isfinite(sorted_probs), sorted_probs, torch.zeros_like(sorted_probs)
        )
        if top_k is not None:
            sorted_probs = sorted_probs[..., :top_k]
        if n_ref is None:
            n_ref = sorted_probs.shape[0]
        elif sorted_probs.shape[0] != n_ref:
            raise ValueError(
                "entropy and probs disagree on N: "
                f"{n_ref} vs {sorted_probs.shape[0]}"
            )
        if available.shape[0] != sorted_probs.shape[0]:
            aligned = torch.zeros(sorted_probs.shape[0], dtype=torch.bool)
            limit = min(int(available.shape[0]), int(sorted_probs.shape[0]))
            aligned[:limit] = available[:limit]
            available = aligned
        sorted_probs = torch.where(
            available.view(-1, 1), sorted_probs.float(), torch.zeros_like(sorted_probs.float())
        )
        _require_finite("sorted probability features", sorted_probs)
        parts.append(sorted_probs.float())
        availability_parts.append(available.float().view(-1, 1))

    if include_availability and availability_parts:
        parts.extend(availability_parts)

    out = torch.cat(parts, dim=-1)
    _require_finite("entropy view", out)
    return out


# ---------------------------------------------------------------------------
# Convenience: build all four views in one call
# ---------------------------------------------------------------------------


def build_all_views(
    bank: Mapping[str, object],
    *,
    query_kind: str = "average",
    answer_kind: str = "average",
    relation_ops: Sequence[str] = RELATION_OPS_DEFAULT,
    option_idx: Optional[torch.Tensor] = None,
    entropy_key: str = "query_entropies",
    probs_key: str = "query_probs",
    require_entropy: bool = False,
    include_entropy_availability: bool = True,
    extra_paths: Optional[Mapping[str, Any]] = None,
) -> Dict[str, torch.Tensor]:
    """Construct ``{"query","answer","relation","entropy"}`` in one shot.

    The ``entropy`` view is omitted from the result dict if the bank's
    ``extras`` carry neither the entropy tensor nor the probabilities (so
    downstream code can simply do ``views.get("entropy")``).
    """
    query = build_query_view(bank, kind=query_kind)
    answer = build_answer_view(bank, kind=answer_kind, option_idx=option_idx)
    relation = build_relation_view(query, answer, ops=relation_ops)

    out: Dict[str, torch.Tensor] = {
        "query": query,
        "answer": answer,
        "relation": relation,
    }

    extras = bank.get("extras")  # type: ignore[union-attr]
    if isinstance(extras, Mapping) and (
        entropy_key in extras or probs_key in extras
    ):
        out["entropy"] = build_entropy_view(
            extras,
            entropy_key=entropy_key,
            probs_key=probs_key,
            require_entropy=require_entropy,
            include_availability=include_entropy_availability,
            extra_paths=extra_paths,
        )
    return out
