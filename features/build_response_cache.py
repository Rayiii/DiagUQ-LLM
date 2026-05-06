"""Build the response cache used by both DiagUQ and the baseline.

The "response cache" comprises the artefacts produced by greedy decoding
and post-hoc scoring. All path construction goes through
:mod:`common.artifact_paths` -- see that module for the canonical layout.

* ``<split_tag>_mextend.json``         -- tokenised prompts + greedy answers
* ``<split_tag>_mextend_samples.json`` -- early-progress snapshot
* ``<split_tag>_mextend_rouge.json``   -- per-sample correctness for QA
* ``<split_tag>_mextend_bleu.json``    -- per-sample correctness for WMT
* ``<split_tag>_extend.json``          -- sampled answers (semantic entropy input)
* ``<split_tag>_semantic_entropy.json``
* ``ask4conf/<model>/<split_tag>.jsonl``
"""

from typing import List, Optional, Sequence, Tuple

from loguru import logger

from common.artifact_paths import normalize_split_tag, split_dataset_and_raw
from common.artifact_locator import write_response_cache_manifest
from common.pair_context import DiagUQPairContext, assert_no_duplicate_output_dirs, resolve_pair_context
from registry.dataset_registry import get_split_names
from features.response_pipeline import (
    generate_answer_most,
    generate_answers,
    generate_ask4conf,
    generate_uncertainty_score,
    generate_y_most_MMLU,
    generate_y_most_QA,
    generate_y_most_WMT,
)


# ``raw_split`` values to drive each dataset's pipeline. ``raw_split`` is
# control-flow only -- file naming uses ``split_tag`` derived via
# :func:`common.artifact_paths.normalize_split_tag`.
def _raw_splits_for_dataset(dataset: str) -> Tuple[str, ...]:
    base_dataset, raw_split = split_dataset_and_raw(dataset)
    if raw_split:
        return (raw_split,)
    try:
        return tuple(get_split_names(base_dataset, prefer="mduq"))
    except Exception:
        return ("train",)


def _eval_raw_split(dataset: str) -> str:
    """The raw split used for sampled answers + semantic entropy."""
    base_dataset, raw_split = split_dataset_and_raw(dataset)
    if raw_split:
        return raw_split
    return "test" if base_dataset == "wmt" else "train"


def _build_one(
    dataset: str,
    model: str,
    *,
    output_root: Optional[str] = None,
    pair_context: Optional[DiagUQPairContext] = None,
    skip_semantic_entropy: bool = False,
    ask4conf_debug_limit: Optional[int] = None,
    limit: Optional[int] = None,
    qa_f1_threshold: float = 0.5,
    allow_full_formatting: bool = False,
    internal_train_ratio: float = 0.7,
    internal_split_seed: int = 42,
    source_error_policy: Optional[str] = None,
    source_error_threshold: float = 0.005,
) -> None:
    ctx = pair_context or resolve_pair_context(dataset, model, runtime_root=output_root)
    base_dataset, _ = split_dataset_and_raw(ctx.resolved_variant)
    raw_splits = (ctx.split,) if ctx.split and ctx.split != "unknown" else _raw_splits_for_dataset(dataset)
    built_variants = []
    semantic_failed_tag = None
    semantic_failure = None
    # 1. greedy answers, per raw split
    for raw_split in raw_splits:
        split_tag = normalize_split_tag(base_dataset, raw_split)
        built_variants.append(split_tag)
        logger.info(
            "[response-cache] stage=greedy dataset={} raw_split={} "
            "split_tag={} model={} read_root={} write_root={}",
            base_dataset, raw_split, split_tag, model, ctx.response_cache_root, ctx.response_cache_root,
        )
        generate_answer_most(
            model,
            split_tag,
            limit=limit,
            allow_full_formatting=allow_full_formatting,
            internal_train_ratio=internal_train_ratio,
            internal_split_seed=internal_split_seed,
        )

    # 2. correctness label, per raw split (must use split_tag, not bare
    # dataset, otherwise we'd read the wrong _mextend.json).
    for raw_split in raw_splits:
        split_tag = normalize_split_tag(base_dataset, raw_split)
        logger.info(
            "[response-cache] stage=metric dataset={} raw_split={} "
            "split_tag={} model={} read_root={} write_root={}",
            base_dataset, raw_split, split_tag, model, ctx.response_cache_root, ctx.response_cache_root,
        )
        if base_dataset == "wmt":
            generate_y_most_WMT(model, split_tag)
        elif base_dataset == "mmlu":
            generate_y_most_MMLU(model, split_tag)
        else:
            generate_y_most_QA(model, split_tag, f1_threshold=qa_f1_threshold)

    # 3. ask-4-confidence (iterates internally over the dataset's splits)
    logger.info(
        "[response-cache] stage=ask4conf requested_dataset={} resolved_variant={} "
        "split={} model={} read_root={} write_root={} debug_limit={}",
        ctx.requested_dataset, ctx.resolved_variant, ctx.split, model,
        ctx.response_cache_root, ctx.response_cache_root, ask4conf_debug_limit,
    )
    generate_ask4conf(
        model,
        dataset,
        debug_limit=ask4conf_debug_limit,
        source_error_policy=source_error_policy,
        source_error_threshold=source_error_threshold,
    )

    # 4. sampled answers + semantic entropy on the eval split
    if base_dataset != "mmlu":
        eval_raw = ctx.split if ctx.split and ctx.split != "unknown" else _eval_raw_split(dataset)
        eval_tag = normalize_split_tag(base_dataset, eval_raw)
        if eval_tag not in built_variants:
            built_variants.append(eval_tag)
        logger.info(
            "[response-cache] stage=samples dataset={} raw_split={} "
            "split_tag={} model={} read_root={} write_root={}",
            base_dataset, eval_raw, eval_tag, model, ctx.response_cache_root, ctx.response_cache_root,
        )
        generate_answers(
            model,
            eval_tag,
            limit=limit,
            allow_full_formatting=allow_full_formatting,
            internal_train_ratio=internal_train_ratio,
            internal_split_seed=internal_split_seed,
        )
        if skip_semantic_entropy:
            logger.info(
                "[response-cache] stage=semantic_entropy SKIPPED dataset={} "
                "split_tag={} model={} (use --include-reference-models on "
                "setup-models to enable)",
                base_dataset, eval_tag, model,
            )
        else:
            logger.info(
                "[response-cache] stage=semantic_entropy dataset={} raw_split={} "
                "split_tag={} model={} read_root={} write_root={}",
                base_dataset, eval_raw, eval_tag, model, ctx.response_cache_root, ctx.response_cache_root,
            )
            try:
                generate_uncertainty_score(model, eval_tag)
            except Exception as exc:  # noqa: BLE001
                semantic_failed_tag = eval_tag
                semantic_failure = exc
                logger.exception(
                    "[response-cache] stage=semantic_entropy FAILED dataset={} "
                    "split_tag={} model={}",
                    base_dataset, eval_tag, model,
                )

    for split_tag in built_variants:
        artifact_status = {}
        artifact_reasons = {}
        if base_dataset == "wmt":
            artifact_status["mextend_rouge"] = "skipped"
            artifact_reasons["mextend_rouge"] = "WMT uses BLEU metric"
            artifact_status["response_answer_audit_csv"] = "skipped"
            artifact_status["response_answer_audit_json"] = "skipped"
            artifact_reasons["response_answer_audit_csv"] = "answer audit is generated for QA datasets"
            artifact_reasons["response_answer_audit_json"] = "answer audit is generated for QA datasets"
        elif base_dataset == "mmlu":
            artifact_status["mextend_bleu"] = "skipped"
            artifact_reasons["mextend_bleu"] = "MMLU uses multiple-choice correctness in mextend_rouge"
            artifact_status["extend"] = "skipped"
            artifact_status["extend_samples"] = "skipped"
            artifact_status["semantic_entropy"] = "skipped"
            artifact_status["response_answer_audit_csv"] = "skipped"
            artifact_status["response_answer_audit_json"] = "skipped"
            artifact_reasons["extend"] = "MMLU response-cache uses option-level scoring, not sampled open generation"
            artifact_reasons["extend_samples"] = artifact_reasons["extend"]
            artifact_reasons["semantic_entropy"] = "MMLU semantic entropy is skipped for multiple-choice response-cache"
            artifact_reasons["response_answer_audit_csv"] = "MMLU uses multiple-choice scoring, not QA answer audit"
            artifact_reasons["response_answer_audit_json"] = artifact_reasons["response_answer_audit_csv"]
        else:
            artifact_status["mextend_bleu"] = "skipped"
            artifact_reasons["mextend_bleu"] = "QA datasets use ROUGE metric"
        if skip_semantic_entropy:
            artifact_status["semantic_entropy"] = "skipped"
            artifact_reasons["semantic_entropy"] = "semantic entropy stage skipped by --skip-semantic-entropy"
        elif split_tag != normalize_split_tag(base_dataset, _eval_raw_split(dataset)):
            artifact_status["semantic_entropy"] = "skipped"
            artifact_reasons["semantic_entropy"] = "semantic entropy is generated only for the eval split"
        elif semantic_failed_tag == split_tag and semantic_failure is not None:
            artifact_status["semantic_entropy"] = "failed"
            artifact_reasons["semantic_entropy"] = repr(semantic_failure)
        manifest = write_response_cache_manifest(
            dataset,
            split_tag,
            model,
            runtime_root=output_root,
            artifact_status=artifact_status,
            artifact_reasons=artifact_reasons,
            pair_context=ctx if split_tag == ctx.resolved_variant else None,
        )
        logger.info(
            "[response-cache] stage=manifest dataset={} split_tag={} "
            "model={} manifest={}",
            dataset, split_tag, model, manifest,
        )
    if semantic_failure is not None:
        raise semantic_failure


def build_response_cache_for_pairs(
    pairs: Sequence[Tuple[str, str]],
    *,
    output_root: Optional[str] = None,
    skip_on_error: bool = True,
    skip_semantic_entropy: bool = False,
    ask4conf_debug_limit: Optional[int] = None,
    limit: Optional[int] = None,
    qa_f1_threshold: float = 0.5,
    allow_full_formatting: bool = False,
    internal_train_ratio: float = 0.7,
    internal_split_seed: int = 42,
    source_error_policy: Optional[str] = None,
    source_error_threshold: float = 0.005,
) -> List[dict]:
    """Build the response cache for each ``(dataset, model)`` pair."""
    results: List[dict] = []
    contexts = [resolve_pair_context(dataset, model, runtime_root=output_root) for dataset, model in pairs]
    assert_no_duplicate_output_dirs(contexts, "response_cache")
    for ctx in contexts:
        try:
            _build_one(
                ctx.requested_dataset, ctx.model,
                output_root=output_root,
                pair_context=ctx,
                skip_semantic_entropy=skip_semantic_entropy,
                ask4conf_debug_limit=ask4conf_debug_limit,
                limit=limit,
                qa_f1_threshold=qa_f1_threshold,
                allow_full_formatting=allow_full_formatting,
                internal_train_ratio=internal_train_ratio,
                internal_split_seed=internal_split_seed,
                source_error_policy=source_error_policy,
                source_error_threshold=source_error_threshold,
            )
            results.append({"dataset": ctx.resolved_variant, "requested_dataset": ctx.requested_dataset, "model": ctx.model, "ok": True})
        except Exception as exc:  # noqa: BLE001
            results.append(
                {"dataset": ctx.resolved_variant, "requested_dataset": ctx.requested_dataset, "model": ctx.model, "error": repr(exc)}
            )
            if not skip_on_error:
                raise
        finally:
            if ctx.resolved_variant.startswith("truthfulqa__"):
                try:
                    from common.truthfulqa_row_count_trace import write_truthfulqa_row_count_trace

                    write_truthfulqa_row_count_trace(
                        ctx,
                        internal_train_ratio=internal_train_ratio,
                        internal_split_seed=internal_split_seed,
                    )
                except Exception as trace_exc:  # noqa: BLE001
                    logger.warning(
                        "[response-cache] truthfulqa row-count trace failed dataset={} model={} error={!r}",
                        ctx.resolved_variant,
                        ctx.model,
                        trace_exc,
                    )
    return results
