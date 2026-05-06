"""Generate the four DiagUQ diagnostic-uncertainty targets.

For each ``(dataset, model)`` pair we read whatever intermediate
response-cache artefacts already exist on disk -- ``*_mextend.json``,
``*_mextend_rouge.json`` / ``*_mextend_bleu.json``,
``*_semantic_entropy.json``, the ask4conf JSON, and the sampled outputs
inside ``*_mextend.json["generated_answers"]`` -- and emit the four
canonical DiagUQ targets to

    ./test_output/<dataset>/<model>/diaguq/dimension_targets/

Targets:
    * ``ambiguity_target``               -- semantic entropy of sampled answers
    * ``knowledge_gap_target``           -- 1 - QA token-F1/EM score or BLEU
    * ``predictive_variability_target``  -- normalized diversity of sampled answers
    * ``overall_target``                 -- average of the (min-max scaled) three;
                                            falls back to (1 - ask4conf) when
                                            individual components are missing

Missing components are kept as ``NaN`` in the JSON dump and as ``nan`` in
the tensor dump so downstream trainers can mask them.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from common.artifact_manifest import write_stage_manifest
from common.runtime_paths import get_test_output_dir
from common.artifact_locator import (
    locate_response_cache_artifacts,
)
from common.artifact_paths import (
    ask4conf_aggregate_json_path,
    ask4conf_dir,
)
from common.target_validation import build_target_sanity, write_target_reports
from common.diagnostic_target_specs import (
    CORE_DIAGNOSTIC_DIMENSIONS,
    TARGET_STATUS_VALUES,
    AVAILABLE_TARGET_STATUSES,
    dataset_spec_to_dict,
    get_dataset_diagnostic_spec,
    target_status_for_value,
)
from common.pair_context import (
    DiagUQPairContext,
    assert_diaguq_output_path,
    assert_no_duplicate_output_dirs,
    assert_pair_output_path,
    resolve_pair_context,
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


DEFAULT_TARGET_NAMES: Tuple[str, ...] = (
    "ambiguity_target",
    "knowledge_gap_target",
    "predictive_variability_target",
    "overall_target",
)


def _safe_load_json(path: Optional[Path]) -> Optional[Any]:
    if path is None:
        return None
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as fr:
        return json.load(fr)


def _artifact_entry(artifacts: Any, key: str) -> Dict[str, Any]:
    entry = getattr(artifacts, "manifest_artifacts", {}).get(key, {})
    return dict(entry) if isinstance(entry, dict) else {}


def _artifact_status(artifacts: Any, key: str) -> str:
    entry = _artifact_entry(artifacts, key)
    path = artifacts.paths.get(key)
    if entry.get("status"):
        return str(entry["status"])
    if path is not None and path.is_file():
        return "generated"
    return "missing"


def _artifact_reason(artifacts: Any, key: str, default: str) -> str:
    entry = _artifact_entry(artifacts, key)
    return str(entry.get("reason") or default)


def _missing_artifact_error(
    *,
    artifact_name: str,
    dataset_name: str,
    resolved_dataset: str,
    model_name: str,
    artifacts: Any,
    rerun_hint: str,
    flag_hint: Optional[str] = None,
) -> FileNotFoundError:
    path = artifacts.paths.get(artifact_name)
    manifest_entry = _artifact_entry(artifacts, artifact_name)
    flag_msg = f" Required flag: {flag_hint}." if flag_hint else ""
    return FileNotFoundError(
        "strict mode: missing required response-cache artifact "
        f"{artifact_name!r} for dataset={dataset_name!r} "
        f"resolved_variant={resolved_dataset!r} model={model_name!r}. "
        "This artifact should be produced by stage `build-response-cache`. "
        f"Rerun: {rerun_hint}.{flag_msg} "
        f"Resolved path: {path}. Manifest contained entry: {bool(manifest_entry)} "
        f"entry={manifest_entry}. Checked paths: {[str(p) for p in artifacts.checked_paths]}"
    )


def dimension_targets_dir(
    dataset_name: str, model_name: str, output_root: Optional[str] = None
) -> Path:
    """Canonical destination for one (dataset, model)'s diagnostic targets.

    Writes always go to the new ``diaguq/`` subtree derived from the
    pair-scoped context. Existing sibling split directories are never used
    to resolve a different pair's output path.
    """
    return resolve_pair_context(dataset_name, model_name, runtime_root=output_root).dimension_targets_dir


def _assert_response_cache_owned(ctx: DiagUQPairContext, artifacts: Any) -> None:
    if artifacts.dataset_variant != ctx.resolved_variant:
        raise AssertionError(
            "response-cache resolved_variant mismatch: "
            f"ctx={ctx.resolved_variant} artifacts={artifacts.dataset_variant}"
        )
    if Path(artifacts.response_cache_dir) != ctx.response_cache_root:
        raise AssertionError(
            "response-cache root mismatch: "
            f"ctx={ctx.response_cache_root} artifacts={artifacts.response_cache_dir}"
        )
    for key in ("mextend", "mextend_rouge", "mextend_bleu", "extend", "semantic_entropy"):
        path = artifacts.paths.get(key)
        if path is not None:
            assert_pair_output_path(ctx, path)


def _resolve_metric_key(dataset_name: str) -> str:
    return "bleu" if dataset_name.startswith("wmt") else "qa_score"


def _load_ask4conf(
    model_name: str, dataset_name: str, output_root: str
) -> Optional[Dict[int, float]]:
    """Best-effort load of ask4conf probabilities keyed by example index."""
    base_ds = dataset_name.split("__", 1)[0]
    ask_dir = ask4conf_dir(model_name, output_root)
    candidates = [
        ask4conf_aggregate_json_path(base_ds, model_name, output_root),
        ask4conf_aggregate_json_path(dataset_name, model_name, output_root),
    ]
    for cand in candidates:
        if not cand.is_file():
            continue
        try:
            with open(cand, "r", encoding="utf-8") as fr:
                payload = json.load(fr)
        except Exception:
            continue
        probs = payload.get("ask4conf_prob") if isinstance(payload, dict) else None
        if isinstance(probs, dict):
            try:
                return {int(k): float(v) for k, v in probs.items()}
            except Exception:
                continue
    # Fallback: scan jsonl shards in ask_dir
    if ask_dir.is_dir():
        out: Dict[int, float] = {}
        for shard in sorted(ask_dir.glob(f"{dataset_name}*.jsonl")):
            if shard.name.endswith("_source_errors.jsonl"):
                continue
            try:
                with open(shard, "r", encoding="utf-8") as fr:
                    for line in fr:
                        row = json.loads(line)
                        if "prob" in row:
                            idx = row.get("sample_idx")
                            if idx is None:
                                idx = len(out)
                            out[int(idx)] = float(row["prob"])
            except Exception:
                continue
        if out:
            return out
    return None


# ---------------------------------------------------------------------------
# Per-row target computation
# ---------------------------------------------------------------------------


_ANSWERS_KEY = "generated_answers"
_SEMANTIC_KEY = "semantic_entropy"


def _flatten_answers(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        out: List[str] = []
        for inner in raw:
            out.extend(str(a) for a in inner)
        return out
    if isinstance(raw, list):
        return [str(a) for a in raw]
    return []


def _predictive_variability(answers: Sequence[str]) -> float:
    """Diversity of sampled answers, in ``[0, 1]``.

    Uses ``1 - max_cluster_share`` over normalized strings: 0 when every
    sample is identical, approaches 1 as samples become uniformly diverse.
    Returns ``NaN`` when fewer than two samples are available.
    """
    cleaned = [a.strip().lower() for a in answers if a is not None]
    cleaned = [a for a in cleaned if a]
    if len(cleaned) < 2:
        return float("nan")
    counts = Counter(cleaned)
    max_share = max(counts.values()) / len(cleaned)
    return float(1.0 - max_share)


def _predictive_variability_stats(answers: Sequence[str]) -> Dict[str, Any]:
    cleaned = [a.strip().lower() for a in answers if a is not None]
    cleaned = [a for a in cleaned if a]
    if len(cleaned) < 2:
        return {
            "value": float("nan"),
            "num_samples": len(cleaned),
            "cluster_count": len(set(cleaned)),
            "entropy": float("nan"),
        }
    counts = Counter(cleaned)
    total = float(len(cleaned))
    entropy = -sum((count / total) * math.log(max(count / total, 1e-12)) for count in counts.values())
    max_share = max(counts.values()) / total
    return {
        "value": float(1.0 - max_share),
        "num_samples": len(cleaned),
        "cluster_count": len(counts),
        "entropy": float(entropy),
    }


def _coerce_float(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _coerce_optional_bool_float(value: Any) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return 1.0
        if lowered in {"false", "0", "no"}:
            return 0.0
    numeric = _coerce_float(value)
    if math.isfinite(numeric):
        return float(numeric >= 0.5)
    return float("nan")


def _clip_unit(value: float) -> float:
    if not math.isfinite(value):
        return float("nan")
    return float(min(1.0, max(0.0, value)))


def _min_max_normalize(values: Sequence[float]) -> List[float]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return [float("nan")] * len(values)
    lo, hi = min(finite), max(finite)
    if hi - lo < 1e-12:
        return [0.0 if math.isfinite(v) else float("nan") for v in values]
    return [
        (v - lo) / (hi - lo) if math.isfinite(v) else float("nan")
        for v in values
    ]

def _weighted_average(values: Mapping[str, float], weights: Mapping[str, float]) -> float:
    numerator = 0.0
    denominator = 0.0
    for name, value in values.items():
        weight = float(weights.get(name, 0.0))
        if weight <= 0 or not math.isfinite(_coerce_float(value)):
            continue
        numerator += float(value) * weight
        denominator += weight
    if denominator <= 0:
        return float("nan")
    return _clip_unit(numerator / denominator)


def _target_semantic_fields(
    *,
    dimension: str,
    value: float,
    spec: Any,
) -> Dict[str, Any]:
    status = target_status_for_value(spec, value)
    available = status in AVAILABLE_TARGET_STATUSES and math.isfinite(_coerce_float(value))
    source = spec.source if status in AVAILABLE_TARGET_STATUSES else status
    reliability = float(spec.reliability) if available else 0.0
    loss_multiplier = float(spec.effective_loss_weight_multiplier) if available else 0.0
    return {
        f"{dimension}_target_available": bool(available),
        f"{dimension}_target_status": status,
        f"{dimension}_target_source": source,
        f"{dimension}_target_reliability": reliability,
        f"{dimension}_target_loss_weight_multiplier": loss_multiplier,
        f"{dimension}_metric_group": spec.metric_group,
        f"{dimension}_construction_note": spec.construction_method,
        f"dim_{dimension}_target_status": status,
        f"dim_{dimension}_target_source": source,
        f"dim_{dimension}_target_reliability": reliability,
        f"dim_{dimension}_target_loss_weight_multiplier": loss_multiplier,
        f"dim_{dimension}_metric_group": spec.metric_group,
        f"dim_{dimension}_construction_note": spec.construction_method,
    }


def _metric_payload(row_metric: Optional[Dict[str, Any]], metric_key: str) -> Dict[str, Any]:
    if row_metric is None:
        return {
            "metric_score": float("nan"),
            "correct": float("nan"),
            "metric_source": "missing_metric_row",
        }
    if metric_key == "bleu":
        metric_score = _coerce_float(row_metric.get("bleu"))
        correct = float(metric_score >= 0.5) if math.isfinite(metric_score) else float("nan")
        return {
            "metric_score": metric_score,
            "correct": correct,
            "metric_source": "bleu_threshold_0.5",
            "bleu_score": metric_score,
        }

    if "qa_score" in row_metric:
        metric_score = _coerce_float(row_metric.get("qa_score"))
        raw_correct = row_metric.get("qa_correct")
        correct = _coerce_optional_bool_float(raw_correct) if raw_correct is not None else (
            float(metric_score >= _coerce_float(row_metric.get("qa_f1_threshold", 0.5)))
            if math.isfinite(metric_score)
            else float("nan")
        )
        return {
            "metric_score": metric_score,
            "correct": correct,
            "metric_source": "normalized_em_token_f1_alias",
            "exact_match": bool(row_metric.get("exact_match", False)),
            "token_f1": _coerce_float(row_metric.get("token_f1")),
            "qa_f1_threshold": _coerce_float(row_metric.get("qa_f1_threshold", 0.5)),
            "qa_parse_status": row_metric.get("parse_status"),
            "qa_parse_error_reason": row_metric.get("parse_error_reason"),
            "rouge_score": _coerce_float(row_metric.get("rouge1_most")),
            "bleu_score": _coerce_float(row_metric.get("bleu")),
            "raw_model_answer": row_metric.get("raw_model_answer") or row_metric.get("most_likely_answer"),
            "extracted_answer": row_metric.get("extracted_answer"),
            "gold_answer": row_metric.get("gold_answer") or row_metric.get("answer_str"),
            "gold_aliases": row_metric.get("gold_aliases"),
            "normalized_prediction": row_metric.get("normalized_prediction"),
            "normalized_gold_answers": row_metric.get("normalized_gold_answers"),
        }

    legacy_score = _coerce_float(row_metric.get("rouge1_most"))
    correct = float(legacy_score >= 0.5) if math.isfinite(legacy_score) else float("nan")
    return {
        "metric_score": legacy_score,
        "correct": correct,
        "metric_source": "legacy_rouge1_threshold_0.5",
        "rouge_score": legacy_score,
    }


def _row_targets(
    row_extend: Dict[str, Any],
    row_metric: Optional[Dict[str, Any]],
    row_semantic: Optional[Dict[str, Any]],
    row_samples: Optional[Dict[str, Any]],
    metric_key: str,
    ask4conf_value: Optional[float],
    semantic_missing_reason: Optional[str],
    samples_missing_reason: Optional[str],
) -> Dict[str, Any]:
    # ambiguity_target = semantic entropy
    if row_semantic is not None and _SEMANTIC_KEY in row_semantic:
        ambiguity = _coerce_float(row_semantic[_SEMANTIC_KEY])
        semantic_available = math.isfinite(ambiguity)
        row_semantic_missing_reason = None if semantic_available else "semantic_entropy_non_finite"
    else:
        ambiguity = float("nan")
        semantic_available = False
        row_semantic_missing_reason = semantic_missing_reason or "semantic_entropy_missing_for_row"

    # knowledge_gap_target = 1 - greedy correctness metric
    metric_info = _metric_payload(row_metric, metric_key)
    metric_score = _coerce_float(metric_info.get("metric_score"))
    knowledge_gap = (
        _clip_unit(1.0 - metric_score)
        if math.isfinite(metric_score)
        else float("nan")
    )

    # predictive_variability_target from sampled answers
    if row_semantic is not None and _ANSWERS_KEY in row_semantic:
        answers_src = row_semantic
    elif row_samples is not None and _ANSWERS_KEY in row_samples:
        answers_src = row_samples
    else:
        answers_src = row_extend
    answers = _flatten_answers(answers_src.get(_ANSWERS_KEY))
    variability_stats = _predictive_variability_stats(answers)
    variability = _coerce_float(variability_stats.get("value"))
    predictive_available = math.isfinite(variability)
    if predictive_available:
        predictive_missing_reason = None
    elif not answers:
        predictive_missing_reason = samples_missing_reason or "missing_sampled_answers"
    else:
        predictive_missing_reason = "too_few_sampled_answers"

    # ask4conf-derived overall fallback
    ask4conf_confidence = _coerce_float(ask4conf_value)
    ask4conf_available = math.isfinite(ask4conf_confidence)
    overall_ask4conf = (
        _clip_unit(1.0 - ask4conf_confidence)
        if ask4conf_available
        else float("nan")
    )

    return {
        "ambiguity_raw": ambiguity,
        "ambiguity_target": float("nan"),
        "semantic_available": semantic_available,
        "semantic_missing_reason": row_semantic_missing_reason,
        "knowledge_gap_target": knowledge_gap,
        "correct": metric_info.get("correct", float("nan")),
        "knowledge_gap_metric_score": metric_score,
        "knowledge_gap_metric_source": metric_info.get("metric_source"),
        "knowledge_gap_parse_status": metric_info.get("qa_parse_status"),
        "knowledge_gap_parse_error_reason": metric_info.get("qa_parse_error_reason"),
        "knowledge_gap_error_type": (
            "parse_failure"
            if metric_info.get("qa_parse_status") not in (None, "ok") or metric_info.get("qa_parse_error_reason")
            else ("incorrect" if math.isfinite(_coerce_float(metric_info.get("correct"))) and _coerce_float(metric_info.get("correct")) < 0.5 else "none")
        ),
        "exact_match": metric_info.get("exact_match"),
        "token_f1": metric_info.get("token_f1"),
        "qa_f1_threshold": metric_info.get("qa_f1_threshold"),
        "qa_parse_status": metric_info.get("qa_parse_status"),
        "qa_parse_error_reason": metric_info.get("qa_parse_error_reason"),
        "rouge_score": metric_info.get("rouge_score"),
        "bleu_score": metric_info.get("bleu_score"),
        "raw_model_answer": metric_info.get("raw_model_answer"),
        "extracted_answer": metric_info.get("extracted_answer"),
        "gold_answer": metric_info.get("gold_answer"),
        "gold_aliases": metric_info.get("gold_aliases"),
        "normalized_prediction": metric_info.get("normalized_prediction"),
        "normalized_gold_answers": metric_info.get("normalized_gold_answers"),
        "ask4conf_confidence": ask4conf_confidence,
        "ask4conf_status": "ok" if ask4conf_available else "missing",
        "ask4conf_missing_reason": None if ask4conf_available else "ask4conf_missing_or_source_error",
        "predictive_variability_target": _clip_unit(variability),
        "predictive_variability_raw": variability,
        "predictive_variability_num_samples": int(variability_stats.get("num_samples") or 0),
        "predictive_variability_cluster_count": int(variability_stats.get("cluster_count") or 0),
        "predictive_variability_entropy": _coerce_float(variability_stats.get("entropy")),
        "predictive_variability_available": predictive_available,
        "predictive_variability_missing_reason": predictive_missing_reason,
        # placeholder; filled in after a global normalization pass
        "overall_target": float("nan"),
        "_overall_ask4conf": overall_ask4conf,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_dimension_targets(
    dataset_name: str,
    model_name: str,
    output_root: Optional[str] = None,
    *,
    strict: bool = False,
    require_semantic_entropy: bool = False,
    allow_degenerate_labels: bool = False,
    pair_context: Optional[DiagUQPairContext] = None,
) -> Dict[str, Any]:
    """Compute and persist the four DiagUQ diagnostic-uncertainty targets.

    Returns a dict with ``"path"``, ``"num_rows"``, ``"target_names"`` and
    ``"missing"`` (per-target count of NaN entries) for the caller's logs.
    """
    if output_root is None:
        output_root = str(get_test_output_dir())
    ctx = pair_context or resolve_pair_context(dataset_name, model_name, runtime_root=output_root)
    diagnostic_spec = get_dataset_diagnostic_spec(ctx.resolved_variant)
    diagnostic_spec_payload = dataset_spec_to_dict(diagnostic_spec)
    artifacts = locate_response_cache_artifacts(
        ctx.resolved_variant, ctx.model, ctx.test_output_root, split=ctx.split
    )
    _assert_response_cache_owned(ctx, artifacts)
    extend_path = artifacts.require("mextend")
    resolved_dataset = ctx.resolved_variant
    logger_msg = (
        "[diag-targets] artifact_resolution requested_dataset={} "
        "resolved_variant={} split={} model={} read_root={} write_root={} "
        "manifest={} mextend={} semantic_entropy={} strategy={}"
    )
    try:
        from loguru import logger

        logger.info(
            logger_msg,
            ctx.requested_dataset,
            resolved_dataset,
            ctx.split,
            ctx.model,
            ctx.response_cache_root,
            ctx.dimension_targets_dir,
            artifacts.manifest_path,
            extend_path,
            artifacts.paths.get("semantic_entropy"),
            artifacts.resolution_strategy,
        )
    except Exception:
        pass

    extend = _safe_load_json(extend_path) or []
    sampled_path = artifacts.paths.get("extend")
    sampled_data = _safe_load_json(sampled_path)
    metric_key = _resolve_metric_key(resolved_dataset)
    metric_path = (
        artifacts.paths.get("mextend_bleu")
        if metric_key == "bleu"
        else artifacts.paths.get("mextend_rouge")
    )
    if metric_path is not None and not metric_path.is_file():
        metric_path = None
    metric_data = _safe_load_json(metric_path) if metric_path else None
    semantic_path = artifacts.paths.get("semantic_entropy")
    semantic_data = _safe_load_json(semantic_path) if semantic_path else None
    ask4conf_map = _load_ask4conf(ctx.model, resolved_dataset, output_root)

    semantic_status = _artifact_status(artifacts, "semantic_entropy")
    semantic_missing_reason = None
    if semantic_data is None:
        semantic_missing_reason = _artifact_reason(
            artifacts,
            "semantic_entropy",
            "semantic entropy artifact missing; rerun build-response-cache without --skip-semantic-entropy",
        )
    samples_status = _artifact_status(artifacts, "extend")
    samples_missing_reason = None
    if sampled_data is None:
        samples_missing_reason = _artifact_reason(
            artifacts,
            "extend",
            "sampled-answer artifact missing; rerun build-response-cache",
        )

    if strict:
        if metric_path is None or not metric_path.is_file():
            metric_artifact = "mextend_bleu" if metric_key == "bleu" else "mextend_rouge"
            raise _missing_artifact_error(
                artifact_name=metric_artifact,
                dataset_name=ctx.requested_dataset,
                resolved_dataset=resolved_dataset,
                model_name=ctx.model,
                artifacts=artifacts,
                rerun_hint="python run.py build-response-cache --scope custom --force",
            )
        if require_semantic_entropy and (semantic_path is None or not semantic_path.is_file()):
            raise _missing_artifact_error(
                artifact_name="semantic_entropy",
                dataset_name=ctx.requested_dataset,
                resolved_dataset=resolved_dataset,
                model_name=ctx.model,
                artifacts=artifacts,
                rerun_hint="python run.py build-response-cache --scope custom --force",
                flag_hint="do not pass --skip-semantic-entropy",
            )

    rows_out: List[Dict[str, Any]] = []
    n = len(extend)
    for idx in range(n):
        row_extend = extend[idx] if idx < len(extend) else {}
        row_metric = metric_data[idx] if (
            metric_data is not None and idx < len(metric_data)
        ) else None
        row_sem = semantic_data[idx] if (
            semantic_data is not None and idx < len(semantic_data)
        ) else None
        row_samples = sampled_data[idx] if (
            sampled_data is not None and idx < len(sampled_data)
        ) else None
        ask_val = ask4conf_map.get(idx) if ask4conf_map else None
        targets = _row_targets(
            row_extend,
            row_metric,
            row_sem,
            row_samples,
            metric_key,
            ask_val,
            semantic_missing_reason,
            samples_missing_reason,
        )
        targets["index"] = idx
        targets["question_str"] = row_extend.get("question_str")
        targets.update(ctx.row_provenance(row_extend.get("sample_id", idx)))
        rows_out.append(targets)

    # Build min-max-normalized columns and the composite overall_target.
    norm_amb = _min_max_normalize([r["ambiguity_raw"] for r in rows_out])
    norm_var = _min_max_normalize(
        [r["predictive_variability_target"] for r in rows_out]
    )
    # knowledge_gap is already in [0,1] under rouge/bleu so no rescale needed.
    for r, na, nv in zip(rows_out, norm_amb, norm_var):
        r["task_type"] = diagnostic_spec.task_type
        r["target_semantics_version"] = "dataset_aware_v1"
        r["ambiguity_target"] = _clip_unit(na)
        r["predictive_variability_target"] = _clip_unit(nv)
        r["task_error_target"] = _clip_unit(r["knowledge_gap_target"])
        r["task_error_target_source"] = r.get("knowledge_gap_metric_source") or "missing"
        components = {
            "ambiguity": r["ambiguity_target"],
            "knowledge_gap": r["knowledge_gap_target"],
            "predictive_variability": r["predictive_variability_target"],
        }
        weighted_dimensions = _weighted_average(components, diagnostic_spec.overall_dimension_weights)
        if diagnostic_spec.overall_target_policy == "task_error_only":
            overall = r["task_error_target"]
        elif diagnostic_spec.overall_target_policy == "hybrid_task_error_and_dimensions":
            hybrid_parts = [r["task_error_target"], weighted_dimensions]
            finite_hybrid = [value for value in hybrid_parts if math.isfinite(_coerce_float(value))]
            overall = _clip_unit(sum(finite_hybrid) / len(finite_hybrid)) if finite_hybrid else float("nan")
        else:
            overall = weighted_dimensions
        r["overall_target"] = overall if math.isfinite(overall) else r["_overall_ask4conf"]
        r["overall_ask4conf_target"] = r["_overall_ask4conf"]
        r["overall_target_composition_policy"] = diagnostic_spec.overall_target_policy
        r["overall_target_dimension_weights"] = json.dumps(
            dict(diagnostic_spec.overall_dimension_weights),
            sort_keys=True,
        )
        for dimension in CORE_DIAGNOSTIC_DIMENSIONS:
            r.update(
                _target_semantic_fields(
                    dimension=dimension,
                    value=_coerce_float(r[f"{dimension}_target"]),
                    spec=diagnostic_spec.dimension_spec(dimension),
                )
            )
        overall_spec = diagnostic_spec.dimension_spec("overall")
        r.update(
            _target_semantic_fields(
                dimension="overall",
                value=_coerce_float(r["overall_target"]),
                spec=overall_spec,
            )
        )
        r["predictive_variability_available"] = bool(
            r["predictive_variability_target_available"]
        )
        if r["predictive_variability_available"]:
            r["predictive_variability_missing_reason"] = None
        r["overall_ask4conf_target_available"] = bool(
            math.isfinite(r["overall_ask4conf_target"])
        )
        r["overall_target_policy"] = diagnostic_spec.overall_target_policy
        r["overall_target_source"] = r["overall_target_source"] if math.isfinite(_coerce_float(r["overall_target"])) else "missing"
        r["correct"] = _coerce_float(r.get("correct"))
        r.update(ctx.row_provenance(r.get("sample_id", r.get("index"))))

    target_dir = ctx.dimension_targets_dir
    assert_diaguq_output_path(ctx, target_dir, stage_token="dimension_targets")
    assert resolved_dataset in str(target_dir)
    assert ctx.model in str(target_dir)
    assert "dimension_targets" in str(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / "dimension_targets.json"
    with open(json_path, "w", encoding="utf-8") as fw:
        json.dump(rows_out, fw, ensure_ascii=False, indent=2)

    # Tensor dump for the trainer
    tensor_payload: Dict[str, torch.Tensor] = {}
    columns: List[torch.Tensor] = []
    for name in DEFAULT_TARGET_NAMES:
        dimension = name[: -len("_target")]
        col = torch.tensor(
            [_coerce_float(r[name]) for r in rows_out], dtype=torch.float32
        )
        tensor_payload[name] = col
        columns.append(col)
        tensor_payload[f"{name}_available"] = torch.tensor(
            [bool(r[f"{name}_available"]) for r in rows_out], dtype=torch.bool
        )
        tensor_payload[f"{name}_status"] = [str(r[f"{name}_status"]) for r in rows_out]  # type: ignore[assignment]
        tensor_payload[f"{name}_source"] = [str(r[f"{name}_source"]) for r in rows_out]  # type: ignore[assignment]
        tensor_payload[f"{name}_reliability"] = torch.tensor(
            [_coerce_float(r.get(f"{name}_reliability")) for r in rows_out], dtype=torch.float32
        )
        tensor_payload[f"{name}_loss_weight_multiplier"] = torch.tensor(
            [_coerce_float(r.get(f"{name}_loss_weight_multiplier")) for r in rows_out], dtype=torch.float32
        )
        tensor_payload[f"{name}_metric_group"] = [str(r.get(f"{dimension}_metric_group") or "") for r in rows_out]  # type: ignore[assignment]
        tensor_payload[f"{name}_construction_note"] = [str(r.get(f"{dimension}_construction_note") or "") for r in rows_out]  # type: ignore[assignment]
        tensor_payload[f"dim_{dimension}_target_status"] = [str(r.get(f"dim_{dimension}_target_status") or "") for r in rows_out]  # type: ignore[assignment]
        tensor_payload[f"dim_{dimension}_target_source"] = [str(r.get(f"dim_{dimension}_target_source") or "") for r in rows_out]  # type: ignore[assignment]
        tensor_payload[f"dim_{dimension}_target_reliability"] = torch.tensor(
            [_coerce_float(r.get(f"dim_{dimension}_target_reliability")) for r in rows_out], dtype=torch.float32
        )
        tensor_payload[f"dim_{dimension}_target_loss_weight_multiplier"] = torch.tensor(
            [_coerce_float(r.get(f"dim_{dimension}_target_loss_weight_multiplier")) for r in rows_out], dtype=torch.float32
        )
        tensor_payload[f"dim_{dimension}_metric_group"] = [str(r.get(f"dim_{dimension}_metric_group") or "") for r in rows_out]  # type: ignore[assignment]
        tensor_payload[f"dim_{dimension}_construction_note"] = [str(r.get(f"dim_{dimension}_construction_note") or "") for r in rows_out]  # type: ignore[assignment]
    tensor_payload["task_error_target"] = torch.tensor(
        [_coerce_float(r.get("task_error_target")) for r in rows_out], dtype=torch.float32
    )
    tensor_payload["task_error_target_source"] = [str(r.get("task_error_target_source") or "") for r in rows_out]  # type: ignore[assignment]
    tensor_payload["overall_ask4conf_target"] = torch.tensor(
        [_coerce_float(r["_overall_ask4conf"]) for r in rows_out],
        dtype=torch.float32,
    )
    tensor_payload["overall_ask4conf_target_available"] = torch.tensor(
        [bool(r["overall_ask4conf_target_available"]) for r in rows_out],
        dtype=torch.bool,
    )
    tensor_payload["ask4conf_status"] = [str(r.get("ask4conf_status") or "") for r in rows_out]  # type: ignore[assignment]
    tensor_payload["ask4conf_missing_reason"] = [str(r.get("ask4conf_missing_reason") or "") for r in rows_out]  # type: ignore[assignment]
    tensor_payload["ambiguity_raw"] = torch.tensor(
        [_coerce_float(r["ambiguity_raw"]) for r in rows_out],
        dtype=torch.float32,
    )
    tensor_payload["predictive_variability_raw"] = torch.tensor(
        [_coerce_float(r.get("predictive_variability_raw")) for r in rows_out], dtype=torch.float32
    )
    tensor_payload["predictive_variability_num_samples"] = torch.tensor(
        [int(r.get("predictive_variability_num_samples") or 0) for r in rows_out], dtype=torch.long
    )
    tensor_payload["predictive_variability_cluster_count"] = torch.tensor(
        [int(r.get("predictive_variability_cluster_count") or 0) for r in rows_out], dtype=torch.long
    )
    tensor_payload["predictive_variability_entropy"] = torch.tensor(
        [_coerce_float(r.get("predictive_variability_entropy")) for r in rows_out], dtype=torch.float32
    )
    tensor_payload["correct"] = torch.tensor(
        [_coerce_float(r["correct"]) for r in rows_out], dtype=torch.float32
    )
    tensor_payload["semantic_available"] = torch.tensor(
        [bool(r.get("semantic_available")) for r in rows_out], dtype=torch.bool
    )
    tensor_payload["predictive_variability_available"] = torch.tensor(
        [bool(r.get("predictive_variability_available")) for r in rows_out],
        dtype=torch.bool,
    )
    tensor_payload["targets_matrix"] = torch.stack(columns, dim=1)
    tensor_payload["target_names"] = DEFAULT_TARGET_NAMES  # type: ignore[assignment]
    tensor_payload["task_type"] = diagnostic_spec.task_type  # type: ignore[assignment]
    tensor_payload["overall_target_policy"] = diagnostic_spec.overall_target_policy  # type: ignore[assignment]
    tensor_payload["overall_target_dimension_weights"] = dict(diagnostic_spec.overall_dimension_weights)  # type: ignore[assignment]
    tensor_payload["dataset_diagnostic_spec"] = diagnostic_spec_payload  # type: ignore[assignment]
    tensor_payload["sample_ids"] = [r.get("sample_id") for r in rows_out]  # type: ignore[assignment]
    tensor_payload["source_sample_ids"] = [r.get("source_sample_id", r.get("sample_id")) for r in rows_out]  # type: ignore[assignment]
    tensor_payload["metadata"] = {
        "requested_dataset": ctx.requested_dataset,
        "dataset": ctx.resolved_variant,
        "resolved_variant": ctx.resolved_variant,
        "split": ctx.split,
        "virtual_split": ctx.split_metadata.get("virtual_split"),
        "source_dataset": ctx.split_metadata.get("source_dataset"),
        "model": ctx.model,
        "task_type": diagnostic_spec.task_type,
        "sample_ids": [r.get("sample_id") for r in rows_out],
        "source_sample_ids": [r.get("source_sample_id", r.get("sample_id")) for r in rows_out],
        "target_status_values": list(TARGET_STATUS_VALUES),
        "target_semantics_version": "dataset_aware_v1",
        "dataset_diagnostic_spec": diagnostic_spec_payload,
        "overall_target_policy": diagnostic_spec.overall_target_policy,
        "overall_target_dimension_weights": dict(diagnostic_spec.overall_dimension_weights),
    }  # type: ignore[assignment]
    pt_path = target_dir / "dimension_targets.pt"
    torch.save(tensor_payload, pt_path)

    target_sanity = build_target_sanity(
        rows_out,
        fail_on_degenerate_correct=bool(strict and not allow_degenerate_labels),
    )
    report_paths = write_target_reports(rows_out, target_sanity, target_dir=target_dir)

    missing = {
        name: int(torch.isnan(tensor_payload[name]).sum().item())
        for name in DEFAULT_TARGET_NAMES
    }
    meta = {
        "dataset": ctx.resolved_variant,
        "requested_dataset": ctx.requested_dataset,
        "dataset_variant": resolved_dataset,
        "resolved_variant": ctx.resolved_variant,
        "split": ctx.split,
        "model": ctx.model,
        "pair_root": str(ctx.pair_root),
        "diaguq_root": str(ctx.diaguq_root),
        "stage_output_dir": str(ctx.dimension_targets_dir),
        "num_rows": n,
        "target_names": list(DEFAULT_TARGET_NAMES),
        "target_semantics_version": "dataset_aware_v1",
        "task_type": diagnostic_spec.task_type,
        "dataset_diagnostic_spec": diagnostic_spec_payload,
        "overall_target_policy": diagnostic_spec.overall_target_policy,
        "overall_target_dimension_weights": dict(diagnostic_spec.overall_dimension_weights),
        "missing": missing,
        "metric_key": metric_key,
        "response_cache_dir": str(artifacts.response_cache_dir),
        "response_cache_manifest": (
            str(artifacts.manifest_path) if artifacts.manifest_path else None
        ),
        "artifact_resolution_strategy": artifacts.resolution_strategy,
        "had_semantic_entropy": semantic_data is not None,
        "semantic_status": semantic_status,
        "semantic_missing_reason": semantic_missing_reason,
        "semantic_path": str(semantic_path) if semantic_path else None,
        "had_sampled_answers": sampled_data is not None,
        "sampled_answers_status": samples_status,
        "sampled_answers_missing_reason": samples_missing_reason,
        "sampled_answers_path": str(sampled_path) if sampled_path else None,
        "had_metric_file": metric_data is not None,
        "allow_degenerate_labels": allow_degenerate_labels,
        "had_ask4conf": ask4conf_map is not None,
        "json_path": str(json_path),
        "pt_path": str(pt_path),
        "target_sanity": target_sanity,
        "target_reports": report_paths,
    }
    manifest_path = write_stage_manifest(
        target_dir,
        stage="diagnostic_targets",
        status="success" if target_sanity["status"] == "success" else "failed",
        dataset=resolved_dataset,
        model=ctx.model,
        artifacts={
            "dimension_targets_json": str(json_path),
            "dimension_targets_pt": str(pt_path),
            **report_paths,
        },
        sanity=target_sanity,
        pair_context=ctx,
    )
    meta["manifest_path"] = str(manifest_path)

    with open(target_dir / "meta.json", "w", encoding="utf-8") as fw:
        json.dump(meta, fw, ensure_ascii=False, indent=2)

    if target_sanity["status"] != "success":
        raise ValueError(
            "diagnostic target sanity check failed: "
            f"{target_sanity.get('failures', [])}; report={report_paths['target_sanity_json']}"
        )

    return meta


def generate_dimension_targets_for_pairs(
    pairs: Iterable[Tuple[str, str]],
    output_root: Optional[str] = None,
    *,
    strict: bool = False,
    require_semantic_entropy: bool = False,
    allow_degenerate_labels: bool = False,
    skip_on_error: bool = True,
) -> List[Dict[str, Any]]:
    """Run ``generate_dimension_targets`` over many ``(dataset, model)`` pairs."""
    if output_root is None:
        output_root = str(get_test_output_dir())
    results: List[Dict[str, Any]] = []
    contexts = [resolve_pair_context(dataset_name, model_name, runtime_root=output_root) for dataset_name, model_name in pairs]
    assert_no_duplicate_output_dirs(contexts, "dimension_targets")
    for ctx in contexts:
        try:
            meta = generate_dimension_targets(
                ctx.requested_dataset,
                ctx.model,
                output_root,
                strict=strict,
                require_semantic_entropy=require_semantic_entropy,
                allow_degenerate_labels=allow_degenerate_labels,
                pair_context=ctx,
            )
            results.append(meta)
        except Exception as exc:  # noqa: BLE001
            err = {
                "dataset": ctx.resolved_variant,
                "requested_dataset": ctx.requested_dataset,
                "model": ctx.model,
                "error": repr(exc),
            }
            results.append(err)
            if not skip_on_error:
                raise
        finally:
            if ctx.resolved_variant.startswith("truthfulqa__"):
                try:
                    from common.truthfulqa_row_count_trace import write_truthfulqa_row_count_trace

                    write_truthfulqa_row_count_trace(ctx)
                except Exception:
                    pass
    return results


# ---------------------------------------------------------------------------
# DiagUQ-style alias
# ---------------------------------------------------------------------------

def build_diagnostic_targets_for_pairs(*args, **kwargs):
    """Alias for :func:generate_dimension_targets_for_pairs."""
    return generate_dimension_targets_for_pairs(*args, **kwargs)
