"""Two-layer feature extraction for the baseline random-forest estimator.

Wraps :func:`features.response_pipeline.generate_X` (and the MMLU
variants) in single-layer mode (i.e. the multi-layer extraction flag is
off). These are the two-fixed-layer features consumed by the baseline
random-forest uncertainty estimator under :mod:`baseline.models`.
"""

from features.response_pipeline import (
    generate_X,
    generate_answer_X_mmlu,
    generate_query_X_mmlu,
)


def build_baseline_features(model_type: str, dataset_split: str) -> None:
    """Extract the two-layer baseline feature tensors for one split.

    ``dataset_split`` should look like ``"triviaqa__train"`` or
    ``"wmt__test"``; the ``model_type`` is used both as the target and tool
    LLM (the legacy "self" branch).
    """
    return generate_X(model_type, dataset_split, model_type)


def build_baseline_features_mmlu(model_type: str, phase: str) -> None:
    """Extract the two-layer baseline MMLU feature tensors for one phase."""
    generate_query_X_mmlu(model_type, phase)
    generate_answer_X_mmlu(model_type, phase)
