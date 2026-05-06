"""Dataset-aware semantics for DiagUQ diagnostic targets.

This module is the single source of truth for whether a diagnostic target is
gold, dataset-grounded, a behavior-derived proxy, or unavailable for a dataset.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping, Sequence


CORE_DIAGNOSTIC_DIMENSIONS = (
    "ambiguity",
    "knowledge_gap",
    "predictive_variability",
)
ALL_DIAGNOSTIC_DIMENSIONS = (*CORE_DIAGNOSTIC_DIMENSIONS, "overall")

TARGET_STATUS_VALUES = (
    "gold",
    "dataset_grounded",
    "proxy",
    "unavailable",
    "missing",
    "masked",
)
AVAILABLE_TARGET_STATUSES = ("gold", "dataset_grounded", "proxy")
MASKED_TARGET_STATUSES = ("unavailable", "missing", "masked")
METRIC_GROUP_VALUES = ("main", "proxy", "auxiliary", "unavailable")

DEFAULT_STATUS_LOSS_MULTIPLIERS = {
    "gold": 1.0,
    "dataset_grounded": 1.0,
    "proxy": 0.7,
    "unavailable": 0.0,
    "missing": 0.0,
    "masked": 0.0,
}


@dataclass(frozen=True)
class DimensionTargetSpec:
    dimension: str
    status: str
    source: str
    construction_method: str
    metric_group: str
    reliability: float
    loss_weight_multiplier: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.dimension not in ALL_DIAGNOSTIC_DIMENSIONS:
            raise ValueError(f"unknown diagnostic dimension: {self.dimension!r}")
        if self.status not in TARGET_STATUS_VALUES:
            raise ValueError(f"invalid target status: {self.status!r}")
        if self.metric_group not in METRIC_GROUP_VALUES:
            raise ValueError(f"invalid metric group: {self.metric_group!r}")
        reliability = float(self.reliability)
        if reliability < 0.0 or reliability > 1.0:
            raise ValueError(f"reliability must be in [0,1], got {self.reliability!r}")

    @property
    def effective_loss_weight_multiplier(self) -> float:
        if self.loss_weight_multiplier is not None:
            return float(self.loss_weight_multiplier)
        return float(DEFAULT_STATUS_LOSS_MULTIPLIERS[self.status])

    @property
    def contributes_to_loss(self) -> bool:
        return self.status in AVAILABLE_TARGET_STATUSES and self.effective_loss_weight_multiplier > 0.0


@dataclass(frozen=True)
class DatasetDiagnosticSpec:
    dataset_name: str
    task_type: str
    dimensions: Mapping[str, DimensionTargetSpec]
    overall_target_policy: str
    overall_dimension_weights: Mapping[str, float] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        missing = [name for name in ALL_DIAGNOSTIC_DIMENSIONS if name not in self.dimensions]
        if missing:
            raise ValueError(f"dataset diagnostic spec missing dimensions: {missing}")
        invalid = [name for name in self.overall_dimension_weights if name not in CORE_DIAGNOSTIC_DIMENSIONS]
        if invalid:
            raise ValueError(f"overall weights use unknown dimensions: {invalid}")

    def dimension_spec(self, dimension: str) -> DimensionTargetSpec:
        return self.dimensions[dimension]


def base_dataset_name(dataset_name: str) -> str:
    return str(dataset_name).split("__", 1)[0].strip().lower()


def _dim(
    dimension: str,
    *,
    status: str,
    source: str,
    construction_method: str,
    metric_group: str,
    reliability: float,
    notes: str = "",
    loss_weight_multiplier: float | None = None,
) -> DimensionTargetSpec:
    return DimensionTargetSpec(
        dimension=dimension,
        status=status,
        source=source,
        construction_method=construction_method,
        metric_group=metric_group,
        reliability=reliability,
        loss_weight_multiplier=loss_weight_multiplier,
        notes=notes,
    )


def _overall(
    *,
    status: str = "proxy",
    source: str,
    construction_method: str,
    metric_group: str = "main",
    reliability: float = 0.7,
    notes: str = "",
) -> DimensionTargetSpec:
    return _dim(
        "overall",
        status=status,
        source=source,
        construction_method=construction_method,
        metric_group=metric_group,
        reliability=reliability,
        notes=notes,
    )


DEFAULT_DATASET_DIAGNOSTIC_SPECS: Mapping[str, DatasetDiagnosticSpec] = {
    "triviaqa": DatasetDiagnosticSpec(
        dataset_name="triviaqa",
        task_type="open_domain_qa",
        overall_target_policy="hybrid_task_error_and_dimensions",
        overall_dimension_weights={
            "ambiguity": 0.2,
            "knowledge_gap": 0.5,
            "predictive_variability": 0.3,
        },
        dimensions={
            "ambiguity": _dim(
                "ambiguity",
                status="proxy",
                source="semantic_dispersion_proxy",
                construction_method="min-max normalized semantic entropy of sampled answers",
                metric_group="proxy",
                reliability=0.5,
                notes="TriviaQA has answer aliases but no gold ambiguity annotation.",
            ),
            "knowledge_gap": _dim(
                "knowledge_gap",
                status="proxy",
                source="qa_alias_f1_correctness_proxy",
                construction_method="1 - QA alias-aware exact/F1 correctness score",
                metric_group="main",
                reliability=0.8,
                notes="Open-domain QA correctness is dataset-grounded by aliases but still model-output parsed.",
            ),
            "predictive_variability": _dim(
                "predictive_variability",
                status="proxy",
                source="sampling_disagreement_proxy",
                construction_method="sampled-answer disagreement from normalized response clusters",
                metric_group="proxy",
                reliability=0.7,
            ),
            "overall": _overall(
                source="hybrid_task_error_and_dimension_proxy",
                construction_method="hybrid of task error and weighted diagnostic proxies",
                reliability=0.75,
            ),
        },
    ),
    "ambigqa": DatasetDiagnosticSpec(
        dataset_name="ambigqa",
        task_type="ambiguous_qa",
        overall_target_policy="hybrid_task_error_and_dimensions",
        overall_dimension_weights={
            "ambiguity": 0.5,
            "knowledge_gap": 0.25,
            "predictive_variability": 0.25,
        },
        dimensions={
            "ambiguity": _dim(
                "ambiguity",
                status="dataset_grounded",
                source="answer_cluster_or_ambiguity_annotation",
                construction_method="dataset-provided plausible answer set or ambiguity annotations when available",
                metric_group="main",
                reliability=1.0,
            ),
            "knowledge_gap": _dim(
                "knowledge_gap",
                status="proxy",
                source="qa_alias_f1_correctness_proxy",
                construction_method="1 - QA alias-aware correctness score against plausible answers",
                metric_group="proxy",
                reliability=0.8,
            ),
            "predictive_variability": _dim(
                "predictive_variability",
                status="proxy",
                source="sampling_disagreement_proxy",
                construction_method="sampled-answer disagreement from normalized response clusters",
                metric_group="proxy",
                reliability=0.7,
            ),
            "overall": _overall(
                source="hybrid_ambiguity_task_error_dimension_proxy",
                construction_method="ambiguity-weighted hybrid of dataset-grounded ambiguity and proxy dimensions",
                reliability=0.85,
            ),
        },
    ),
    "truthfulqa": DatasetDiagnosticSpec(
        dataset_name="truthfulqa",
        task_type="truthfulness_qa",
        overall_target_policy="hybrid_task_error_and_dimensions",
        overall_dimension_weights={
            "ambiguity": 0.1,
            "knowledge_gap": 0.6,
            "predictive_variability": 0.3,
        },
        dimensions={
            "ambiguity": _dim(
                "ambiguity",
                status="proxy",
                source="semantic_dispersion_proxy",
                construction_method="min-max normalized semantic entropy of sampled answers",
                metric_group="auxiliary",
                reliability=0.5,
                notes="TruthfulQA is not a gold ambiguity dataset.",
            ),
            "knowledge_gap": _dim(
                "knowledge_gap",
                status="dataset_grounded",
                source="truthfulness_correctness_label",
                construction_method="truthfulness correctness against correct/incorrect answer sets",
                metric_group="main",
                reliability=1.0,
            ),
            "predictive_variability": _dim(
                "predictive_variability",
                status="proxy",
                source="sampling_disagreement_proxy",
                construction_method="sampled-answer disagreement from normalized response clusters",
                metric_group="proxy",
                reliability=0.7,
            ),
            "overall": _overall(
                source="hybrid_truthfulness_error_dimension_proxy",
                construction_method="hybrid of truthfulness task error and diagnostic proxies",
                reliability=0.85,
            ),
        },
    ),
    "mmlu": DatasetDiagnosticSpec(
        dataset_name="mmlu",
        task_type="multiple_choice",
        overall_target_policy="hybrid_task_error_and_dimensions",
        overall_dimension_weights={
            "ambiguity": 0.0,
            "knowledge_gap": 0.6,
            "predictive_variability": 0.4,
        },
        dimensions={
            "ambiguity": _dim(
                "ambiguity",
                status="unavailable",
                source="unavailable_no_option_entropy_or_response_dispersion",
                construction_method="not used unless an option-entropy or response-dispersion proxy is explicitly implemented",
                metric_group="unavailable",
                reliability=0.0,
                notes="MMLU has no gold ambiguity labels.",
            ),
            "knowledge_gap": _dim(
                "knowledge_gap",
                status="proxy",
                source="multiple_choice_correctness_proxy",
                construction_method="1 - multiple-choice correctness proxy",
                metric_group="main",
                reliability=0.8,
            ),
            "predictive_variability": _dim(
                "predictive_variability",
                status="proxy",
                source="sampled_option_distribution_entropy",
                construction_method="sampled option distribution entropy or response-cluster disagreement",
                metric_group="proxy",
                reliability=0.7,
            ),
            "overall": _overall(
                source="hybrid_task_error_and_dimension_proxy",
                construction_method="hybrid of multiple-choice task error and available diagnostic proxies",
                reliability=0.75,
            ),
        },
    ),
    "wmt": DatasetDiagnosticSpec(
        dataset_name="wmt",
        task_type="machine_translation",
        overall_target_policy="weighted_dimension_proxy",
        overall_dimension_weights={
            "ambiguity": 0.2,
            "knowledge_gap": 0.3,
            "predictive_variability": 0.5,
        },
        dimensions={
            "ambiguity": _dim(
                "ambiguity",
                status="proxy",
                source="translation_diversity_proxy",
                construction_method="semantic or lexical diversity among sampled translations",
                metric_group="auxiliary",
                reliability=0.5,
            ),
            "knowledge_gap": _dim(
                "knowledge_gap",
                status="proxy",
                source="translation_quality_proxy",
                construction_method="1 - translation quality score such as BLEU",
                metric_group="proxy",
                reliability=0.6,
            ),
            "predictive_variability": _dim(
                "predictive_variability",
                status="proxy",
                source="translation_sampling_diversity_proxy",
                construction_method="sampled translation disagreement from normalized response clusters",
                metric_group="main",
                reliability=0.7,
            ),
            "overall": _overall(
                source="weighted_translation_quality_variability_proxy",
                construction_method="weighted quality and variability proxy for translation uncertainty",
                reliability=0.65,
            ),
        },
    ),
}


def _fallback_spec(dataset_name: str) -> DatasetDiagnosticSpec:
    base = base_dataset_name(dataset_name)
    trivia = DEFAULT_DATASET_DIAGNOSTIC_SPECS["triviaqa"]
    return replace(
        trivia,
        dataset_name=base,
        task_type="unknown",
        notes="Fallback diagnostic semantics; add a dataset-specific spec before paper claims.",
    )


def get_dataset_diagnostic_spec(
    dataset_name: str,
    overrides: Mapping[str, Any] | None = None,
) -> DatasetDiagnosticSpec:
    """Return the dataset semantic spec, with optional shallow overrides.

    ``dataset_name`` may include split suffixes such as ``triviaqa__validation``.
    ``overrides`` is intentionally simple so experiments can replace selected
    fields without duplicating this module.
    """
    base = base_dataset_name(dataset_name)
    spec = DEFAULT_DATASET_DIAGNOSTIC_SPECS.get(base, _fallback_spec(dataset_name))
    if not overrides:
        return spec

    dimensions = dict(spec.dimensions)
    for dimension, dim_override in dict(overrides.get("dimensions") or {}).items():
        if dimension in dimensions and isinstance(dim_override, Mapping):
            dimensions[dimension] = replace(dimensions[dimension], **dict(dim_override))
    top_level = {
        key: value
        for key, value in overrides.items()
        if key in {"task_type", "overall_target_policy", "overall_dimension_weights", "notes"}
    }
    return replace(spec, dimensions=dimensions, **top_level)


def get_dimension_target_spec(dataset_name: str, dimension: str) -> DimensionTargetSpec:
    return get_dataset_diagnostic_spec(dataset_name).dimension_spec(dimension)


def status_loss_multiplier(status: str, overrides: Mapping[str, float] | None = None) -> float:
    values = dict(DEFAULT_STATUS_LOSS_MULTIPLIERS)
    if overrides:
        values.update({str(key): float(value) for key, value in overrides.items()})
    return float(values.get(str(status), 0.0))


def dimension_spec_to_dict(spec: DimensionTargetSpec) -> dict[str, Any]:
    payload = asdict(spec)
    payload["loss_weight_multiplier"] = spec.effective_loss_weight_multiplier
    payload["contributes_to_loss"] = spec.contributes_to_loss
    return payload


def dataset_spec_to_dict(spec: DatasetDiagnosticSpec) -> dict[str, Any]:
    return {
        "dataset_name": spec.dataset_name,
        "task_type": spec.task_type,
        "overall_target_policy": spec.overall_target_policy,
        "overall_dimension_weights": dict(spec.overall_dimension_weights),
        "notes": spec.notes,
        "dimensions": {
            name: dimension_spec_to_dict(dim_spec)
            for name, dim_spec in spec.dimensions.items()
        },
        "target_status_values": list(TARGET_STATUS_VALUES),
        "metric_group_values": list(METRIC_GROUP_VALUES),
        "status_loss_multipliers": dict(DEFAULT_STATUS_LOSS_MULTIPLIERS),
    }


def target_status_for_value(spec: DimensionTargetSpec, value: float) -> str:
    if spec.status == "unavailable":
        return "unavailable"
    try:
        finite = math.isfinite(float(value))
    except (TypeError, ValueError):
        finite = False
    return spec.status if finite else "missing"
