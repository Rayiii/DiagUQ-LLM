"""Build the multi-layer hidden-state bank used by DiagUQ.

Outputs land under ``./test_output/<dataset>/<model>/diaguq/hidden_bank/``
with files ``{view}_{kind}_layer_{idx}.pt`` plus side files
(``query_entropies.pt``, ``query_probs.pt``, ``labels.pt`` ...).

Wraps :func:`features.response_pipeline.generate_X` (and the MMLU
variants) with the multi-layer extraction flag enabled (the underlying
keyword is still called ``mduq_mode=True`` -- a historical name kept for
backward compatibility with the response-pipeline internals).
"""

import json
from typing import List, Sequence, Tuple

from common.artifact_locator import ArtifactResolutionError, locate_response_cache_artifacts
from common.artifact_manifest import write_stage_manifest
from common.pair_context import (
    DiagUQPairContext,
    assert_diaguq_output_path,
    assert_no_duplicate_output_dirs,
    resolve_pair_context,
)
from features.dataset_variant_loader import resolve_supported_dataset_variant
from registry.model_registry import get_candidate_layers


def _hidden_bank_sample_ids(ctx: DiagUQPairContext) -> list:
    artifacts = locate_response_cache_artifacts(
        ctx.resolved_variant,
        ctx.model,
        runtime_root=ctx.test_output_root,
    )
    with open(artifacts.require("mextend"), "r", encoding="utf-8") as fr:
        rows = json.load(fr)
    if not isinstance(rows, list):
        return []
    num_queries = sum(1 for row in rows if isinstance(row, dict) and "most_likely_answer" in row)
    return [row.get("sample_id") for row in rows[:num_queries] if isinstance(row, dict)]


def preflight_hidden_bank_for_pairs(
    pairs: Sequence[Tuple[str, str]],
    *,
    output_root: str | None = None,
    require_response_cache: bool = True,
) -> List[DiagUQPairContext]:
    """Validate hidden-bank pair support before loading an LLM."""
    contexts = [resolve_pair_context(dataset, model, runtime_root=output_root) for dataset, model in pairs]
    assert_no_duplicate_output_dirs(contexts, "hidden_bank")
    for ctx in contexts:
        resolve_supported_dataset_variant(ctx.resolved_variant)
        assert_diaguq_output_path(ctx, ctx.hidden_bank_dir, stage_token="hidden_bank")
        if require_response_cache:
            artifacts = locate_response_cache_artifacts(
                ctx.resolved_variant,
                ctx.model,
                runtime_root=ctx.test_output_root,
            )
            try:
                artifacts.require("mextend")
            except ArtifactResolutionError as exc:
                raise ArtifactResolutionError(
                    "build-hidden-bank preflight failed: response-cache mextend "
                    f"artifact is required before hidden-bank extraction for "
                    f"resolved_variant={ctx.resolved_variant!r} model={ctx.model!r}. "
                    "Run build-response-cache for the same pair first.\n"
                    + str(exc)
                ) from exc
    return contexts


def _build_one(
    dataset: str,
    model_name: str,
    *,
    mmlu_phases: Sequence[str],
    output_root: str | None = None,
    pair_context: DiagUQPairContext | None = None,
    internal_train_ratio: float = 0.7,
    internal_split_seed: int = 42,
) -> List[str]:
    from features.response_pipeline import generate_X

    ctx = pair_context or resolve_pair_context(dataset, model_name, runtime_root=output_root)
    built_variants: List[str] = []
    del mmlu_phases
    generate_X(
        ctx.model,
        ctx.resolved_variant,
        ctx.model,
        mduq_mode=True,
        pair_context=ctx,
        internal_train_ratio=internal_train_ratio,
        internal_split_seed=internal_split_seed,
    )
    built_variants.append(ctx.resolved_variant)
    return built_variants


def build_hidden_bank_for_pairs(
    pairs: Sequence[Tuple[str, str]],
    *,
    output_root: str | None = None,
    mmlu_phases: Sequence[str] = ("validation", "test"),
    skip_on_error: bool = True,
    internal_train_ratio: float = 0.7,
    internal_split_seed: int = 42,
) -> List[dict]:
    """Build the DiagUQ multi-layer hidden bank for each ``(dataset, model)``."""
    results: List[dict] = []
    contexts = preflight_hidden_bank_for_pairs(
        pairs,
        output_root=output_root,
        require_response_cache=True,
    )
    for ctx in contexts:
        try:
            variants = _build_one(
                ctx.requested_dataset,
                ctx.model,
                mmlu_phases=mmlu_phases,
                output_root=output_root,
                pair_context=ctx,
                internal_train_ratio=internal_train_ratio,
                internal_split_seed=internal_split_seed,
            )
            reports = []
            from common.feature_validation import validate_hidden_bank_dir

            layer_list = get_candidate_layers(ctx.model)
            for variant in variants:
                if variant.startswith("MMLU/"):
                    continue
                bank_dir = ctx.hidden_bank_dir
                assert_diaguq_output_path(ctx, bank_dir, stage_token="hidden_bank")
                report = validate_hidden_bank_dir(
                    bank_dir,
                    layer_list=layer_list,
                    model_name=ctx.model,
                    include_optional_views=True,
                    require_entropy=False,
                )
                manifest = write_stage_manifest(
                    bank_dir,
                    stage="hidden_bank",
                    status="success",
                    dataset=ctx.resolved_variant,
                    model=ctx.model,
                    artifacts={
                        "hidden_bank_dir": str(bank_dir),
                        "hidden_bank_sanity_json": str(bank_dir / "hidden_bank_sanity.json"),
                        "hidden_bank_sanity_csv": str(bank_dir / "hidden_bank_sanity.csv"),
                    },
                    sanity=report,
                    extra={"sample_ids": _hidden_bank_sample_ids(ctx)},
                    pair_context=ctx,
                )
                reports.append({"variant": variant, "manifest": str(manifest), "sanity": report})
            results.append({"dataset": ctx.resolved_variant, "requested_dataset": ctx.requested_dataset, "model": ctx.model, "ok": True, "reports": reports})
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
                except Exception:
                    pass
    return results
