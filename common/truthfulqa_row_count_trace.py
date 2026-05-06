"""TruthfulQA internal-split row-count trace utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Optional

from common.artifact_paths import (
    ask4conf_metadata_path,
    extend_path,
    mextend_path,
    response_answer_audit_json_path,
)
from common.pair_context import DiagUQPairContext, resolve_pair_context
from common.sample_alignment import require_no_sample_id_overlap
from common.single_split_policy import (
    DEFAULT_INTERNAL_SPLIT_SEED,
    DEFAULT_INTERNAL_TRAIN_RATIO,
    internal_split_variant,
    parse_internal_split_variant,
    row_belongs_to_internal_split,
)


TRACE_FILENAME = "truthfulqa_row_count_trace.json"


class _TraceTokenizer:
    bos_token_id = -101
    eos_token_id = -102

    def encode(self, text: Any, *args: Any, **kwargs: Any) -> list[int]:
        del args, kwargs
        return [ord(ch) % 32000 for ch in str(text)]


def _json_len(path: Path) -> Optional[int]:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, Mapping):
        for key in ("rows", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return len(value)
    return None


def _csv_len(path: Path) -> Optional[int]:
    if not path.is_file():
        return None
    try:
        with path.open(newline="", encoding="utf-8") as fr:
            return sum(1 for _ in csv.DictReader(fr))
    except Exception:  # noqa: BLE001
        return None


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, Mapping)]
    return []


def _ask4conf_meta(split_tag: str, model: str, root: Path) -> dict[str, Any]:
    path = ask4conf_metadata_path(split_tag, model, root)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _meta_int(meta: Mapping[str, Any], *keys: str) -> Optional[int]:
    for key in keys:
        value = meta.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except Exception:  # noqa: BLE001
            continue
    return None


def _raw_local_generation_rows() -> Optional[int]:
    try:
        import datasets
        from data.formatters import TRUTHFULQA_LOCAL

        raw = datasets.load_from_disk(TRUTHFULQA_LOCAL)
        if isinstance(raw, datasets.Dataset):
            return len(raw)
        if "validation" in raw:
            return len(raw["validation"])
    except Exception:  # noqa: BLE001
        return None
    return None


def _raw_hf_multiple_choice_rows() -> Optional[int]:
    try:
        import datasets

        download_config = datasets.DownloadConfig(local_files_only=True)
        ds = datasets.load_dataset(
            "truthful_qa",
            "multiple_choice",
            split="validation",
            download_config=download_config,
        )
        return len(ds)
    except Exception:  # noqa: BLE001
        return None


def _formatted_truthfulqa_rows() -> list[dict[str, Any]]:
    try:
        from data.formatters import truthfulqa_formatter

        formatted = truthfulqa_formatter(_TraceTokenizer(), cache=False)
        key = "truthfulqa__validation"
        if key not in formatted:
            return []
        return [dict(row) for row in formatted[key]]
    except Exception:  # noqa: BLE001
        return []


def _hidden_bank_rows(ctx: DiagUQPairContext) -> Optional[int]:
    sanity = ctx.hidden_bank_dir / "hidden_bank_sanity.json"
    if sanity.is_file():
        try:
            payload = json.loads(sanity.read_text(encoding="utf-8"))
            value = payload.get("num_samples")
            if value is not None:
                return int(value)
        except Exception:  # noqa: BLE001
            pass
    return None


def _target_rows(ctx: DiagUQPairContext) -> Optional[int]:
    sanity = ctx.dimension_targets_dir / "target_sanity.json"
    if sanity.is_file():
        try:
            payload = json.loads(sanity.read_text(encoding="utf-8"))
            for key in ("num_rows", "n_rows"):
                if payload.get(key) is not None:
                    return int(payload[key])
        except Exception:  # noqa: BLE001
            pass
    return _json_len(ctx.dimension_targets_dir / "dimension_targets.json")


def _split_rows(rows: list[dict[str, Any]], *, virtual_split: str, seed: int, train_ratio: float) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row_belongs_to_internal_split(
            row,
            virtual_split,
            seed=seed,
            train_ratio=train_ratio,
        )
    ]


def _drop_delta(before: Optional[int], after: Optional[int]) -> Optional[int]:
    if before is None or after is None:
        return None
    return int(before) - int(after)


def write_truthfulqa_row_count_trace(
    ctx: DiagUQPairContext,
    *,
    internal_train_ratio: float = DEFAULT_INTERNAL_TRAIN_RATIO,
    internal_split_seed: int = DEFAULT_INTERNAL_SPLIT_SEED,
) -> Optional[Path]:
    info = parse_internal_split_variant(ctx.resolved_variant)
    base = ctx.resolved_variant.split("__", 1)[0]
    if base != "truthfulqa" and info is None:
        return None
    source_split = info.source_split if info is not None else (ctx.split or "validation")
    train_variant = internal_split_variant("truthfulqa", source_split, "train")
    eval_variant = internal_split_variant("truthfulqa", source_split, "eval")
    train_ctx = resolve_pair_context(train_variant, ctx.model, runtime_root=ctx.test_output_root)
    eval_ctx = resolve_pair_context(eval_variant, ctx.model, runtime_root=ctx.test_output_root)

    formatted_rows = _formatted_truthfulqa_rows()
    formatter_rows = len(formatted_rows) if formatted_rows else None
    train_rows = _split_rows(
        formatted_rows,
        virtual_split="train",
        seed=internal_split_seed,
        train_ratio=internal_train_ratio,
    ) if formatted_rows else []
    eval_rows = _split_rows(
        formatted_rows,
        virtual_split="eval",
        seed=internal_split_seed,
        train_ratio=internal_train_ratio,
    ) if formatted_rows else []
    if train_rows or eval_rows:
        require_no_sample_id_overlap(
            [row.get("sample_id") for row in train_rows],
            [row.get("sample_id") for row in eval_rows],
            left_label=train_variant,
            right_label=eval_variant,
        )

    response_cache_train_rows = _json_len(mextend_path(train_variant, ctx.model, ctx.test_output_root))
    response_cache_eval_rows = _json_len(mextend_path(eval_variant, ctx.model, ctx.test_output_root))
    ask4conf_train_meta = _ask4conf_meta(train_variant, ctx.model, ctx.test_output_root)
    ask4conf_eval_meta = _ask4conf_meta(eval_variant, ctx.model, ctx.test_output_root)
    response_train_json = _load_json_rows(mextend_path(train_variant, ctx.model, ctx.test_output_root))
    response_eval_json = _load_json_rows(mextend_path(eval_variant, ctx.model, ctx.test_output_root))
    if response_train_json and response_eval_json:
        require_no_sample_id_overlap(
            [row.get("sample_id") for row in response_train_json],
            [row.get("sample_id") for row in response_eval_json],
            left_label=f"response_cache:{train_variant}",
            right_label=f"response_cache:{eval_variant}",
        )
    extend_eval_rows = _json_len(extend_path(eval_variant, ctx.model, ctx.test_output_root))
    audit_eval_rows = _json_len(response_answer_audit_json_path(eval_variant, ctx.model, ctx.test_output_root))
    hidden_bank_train_rows = _hidden_bank_rows(train_ctx)
    hidden_bank_eval_rows = _hidden_bank_rows(eval_ctx)
    target_train_rows = _target_rows(train_ctx)
    target_eval_rows = _target_rows(eval_ctx)
    target_train_json = _load_json_rows(train_ctx.dimension_targets_dir / "dimension_targets.json")
    target_eval_json = _load_json_rows(eval_ctx.dimension_targets_dir / "dimension_targets.json")
    if target_train_json and target_eval_json:
        require_no_sample_id_overlap(
            [row.get("sample_id") for row in target_train_json],
            [row.get("sample_id") for row in target_eval_json],
            left_label=f"dimension_targets:{train_variant}",
            right_label=f"dimension_targets:{eval_variant}",
        )
    eval_prediction_rows = _csv_len(eval_ctx.eval_dir / "predictions.csv")
    export_rows = _json_len(eval_ctx.analysis_dir / "per_sample.json")
    raw_generation_rows = _raw_local_generation_rows()
    raw_multiple_choice_rows = _raw_hf_multiple_choice_rows()
    expected_eval = None
    guardrail_warning = None
    if raw_generation_rows is not None:
        expected_eval = raw_generation_rows * (1.0 - float(internal_train_ratio))
        if response_cache_eval_rows is not None and response_cache_eval_rows < 0.5 * expected_eval:
            guardrail_warning = (
                "TruthfulQA internal eval split has fewer than 50% of expected rows: "
                f"actual={response_cache_eval_rows} expected≈{expected_eval:.1f} "
                f"raw_rows={raw_generation_rows} train_ratio={internal_train_ratio}"
            )

    dropped_rows_by_stage = {
        "raw_to_formatter": _drop_delta(raw_generation_rows, formatter_rows),
        "formatter_to_internal_total": _drop_delta(formatter_rows, (len(train_rows) + len(eval_rows)) if formatted_rows else None),
        "internal_train_to_response_cache_train": _drop_delta(len(train_rows) if formatted_rows else None, response_cache_train_rows),
        "internal_eval_to_response_cache_eval": _drop_delta(len(eval_rows) if formatted_rows else None, response_cache_eval_rows),
        "response_cache_eval_to_extend_eval": _drop_delta(response_cache_eval_rows, extend_eval_rows),
        "response_cache_eval_to_hidden_bank_eval": _drop_delta(response_cache_eval_rows, hidden_bank_eval_rows),
        "response_cache_train_to_hidden_bank_train": _drop_delta(response_cache_train_rows, hidden_bank_train_rows),
        "response_cache_eval_to_target_eval": _drop_delta(response_cache_eval_rows, target_eval_rows),
        "response_cache_train_to_target_train": _drop_delta(response_cache_train_rows, target_train_rows),
        "target_eval_to_predictions": _drop_delta(target_eval_rows, eval_prediction_rows),
        "predictions_to_export": _drop_delta(eval_prediction_rows, export_rows),
    }
    drop_reasons: dict[str, str] = {}
    for stage, dropped in dropped_rows_by_stage.items():
        if dropped is None:
            drop_reasons[stage] = "not_available_yet"
        elif dropped == 0:
            drop_reasons[stage] = "no_drop"
        elif stage == "raw_to_formatter":
            drop_reasons[stage] = "formatter_filtered_or_grouped_rows"
        elif "internal" in stage:
            drop_reasons[stage] = "response_cache_missing_or_stale_for_internal_split"
        else:
            drop_reasons[stage] = "downstream_artifact_row_count_mismatch"
    if guardrail_warning:
        drop_reasons["truthfulqa_internal_eval_guardrail"] = guardrail_warning

    trace = {
        "dataset": "truthfulqa",
        "source_split": source_split,
        "model": ctx.model,
        "internal_train_ratio": float(internal_train_ratio),
        "internal_split_seed": int(internal_split_seed),
        "raw_hf_rows": {
            "generation_validation": raw_generation_rows,
            "multiple_choice_validation": raw_multiple_choice_rows,
        },
        "formatter_rows": formatter_rows,
        "project_loader_rows": {
            "truthfulqa__validation": formatter_rows,
            "truthfulqa__validation_train": len(train_rows) if formatted_rows else None,
            "truthfulqa__validation_eval": len(eval_rows) if formatted_rows else None,
        },
        "usable_rows_before_internal_split": formatter_rows,
        "internal_train_rows": len(train_rows) if formatted_rows else None,
        "internal_eval_rows": len(eval_rows) if formatted_rows else None,
        "response_cache_train_rows": response_cache_train_rows,
        "response_cache_eval_rows": response_cache_eval_rows,
        "greedy_rows": {
            "train": response_cache_train_rows,
            "eval": response_cache_eval_rows,
        },
        "ask4conf_valid_rows": {
            "train": _meta_int(ask4conf_train_meta, "written_count", "valid_answer_rows"),
            "eval": _meta_int(ask4conf_eval_meta, "written_count", "valid_answer_rows"),
        },
        "ask4conf_skipped_rows": {
            "train": _meta_int(ask4conf_train_meta, "skipped_ask4conf_rows", "source_failed_count"),
            "eval": _meta_int(ask4conf_eval_meta, "skipped_ask4conf_rows", "source_failed_count"),
        },
        "final_response_cache_rows": {
            "train": response_cache_train_rows,
            "eval": response_cache_eval_rows,
        },
        "hidden_bank_train_rows": hidden_bank_train_rows,
        "hidden_bank_eval_rows": hidden_bank_eval_rows,
        "target_train_rows": target_train_rows,
        "target_eval_rows": target_eval_rows,
        "eval_prediction_rows": eval_prediction_rows,
        "export_rows": export_rows,
        "response_cache_extend_eval_rows": extend_eval_rows,
        "response_answer_audit_eval_rows": audit_eval_rows,
        "expected_eval_rows": expected_eval,
        "dropped_rows_by_stage": dropped_rows_by_stage,
        "drop_reasons": drop_reasons,
        "warnings": [guardrail_warning] if guardrail_warning else [],
    }
    written: list[Path] = []
    for target_ctx in (train_ctx, eval_ctx):
        target_ctx.diaguq_root.mkdir(parents=True, exist_ok=True)
        path = target_ctx.diaguq_root / TRACE_FILENAME
        path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(path)
    return eval_ctx.diaguq_root / TRACE_FILENAME if written else None
