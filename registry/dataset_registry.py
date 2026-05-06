"""Unified dataset registry for the baseline and DiagUQ pipelines.

The registry is split in two:

* ``LEGACY_DATASET_REGISTRY`` -- datasets used by the baseline
  supervised-uncertainty estimator (kept under the historical name for
  backward compatibility; also exported as ``BASELINE_DATASET_REGISTRY``
  via the ``DIAGUQ_*`` alias block at the bottom of this module).
* ``MDUQ_DATASET_REGISTRY`` -- datasets used by the DiagUQ pipeline
  (also exported as ``DIAGUQ_DATASET_REGISTRY``).

Each entry exposes the following metadata keys:

``task_type``                : Coarse task tag, one of ``TASK_TYPES`` below.
``split_names``              : Tuple of split names available locally
                               (e.g. ``("train", "validation", "test")``).
``supports_generation``      : ``True`` when the dataset is suitable for
                               open-ended sequence generation evaluation.
``supports_multiple_choice`` : ``True`` when answers are constrained to a
                               fixed letter set (A/B/C/D...).
``supports_dimension_proxy`` : ``True`` when the dataset can produce labels
                               for at least one of the MDUQ uncertainty
                               dimensions (ambiguity, knowledge_gap,
                               predictive_variability, ...).
``hf_repo_id`` (optional)    : Source HF dataset, used by download helpers.
``local_dir_name`` (optional): On-disk folder name. Resolved through
                               :func:`common.runtime_paths.get_dataset_dir`
                               (uses ``<runtime_root>/data`` in autodl
                               mode and ``./artifacts/data`` in local
                               fallback mode). The source-code package
                               ``./data`` is NEVER used as a dataset
                               cache root.
"""

from __future__ import annotations

from typing import Dict, List

# ---------------------------------------------------------------------------
# Task-type tags shared by dataset entries.
# ---------------------------------------------------------------------------

TASK_TYPES = (
    "multiple_choice_qa",
    "open_domain_qa",
    "ambiguity_qa",
    "truthfulness_qa",
    "translation",
)


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

LEGACY_DATASET_REGISTRY: Dict[str, Dict] = {
    "coqa": {
        "task_type": "open_domain_qa",
        "split_names": ("train", "validation"),
        "supports_generation": True,
        "supports_multiple_choice": False,
        "supports_dimension_proxy": False,
        "hf_repo_id": "stanfordnlp/coqa",
        "local_dir_name": "coqa",
    },
    "triviaqa": {
        "task_type": "open_domain_qa",
        "split_names": ("train", "validation"),
        "supports_generation": True,
        "supports_multiple_choice": False,
        "supports_dimension_proxy": False,
        "hf_repo_id": "trivia_qa",
        "hf_config": "rc.nocontext",
        "local_dir_name": "trivia_qa",
    },
    "mmlu": {
        "task_type": "multiple_choice_qa",
        "split_names": ("validation", "test"),
        "supports_generation": False,
        "supports_multiple_choice": True,
        "supports_dimension_proxy": False,
        "hf_repo_id": "cais/mmlu",
        "local_dir_name": "mmlu",
    },
    "wmt": {
        "task_type": "translation",
        "split_names": ("train", "test"),
        "supports_generation": True,
        "supports_multiple_choice": False,
        "supports_dimension_proxy": False,
        "hf_repo_id": "wmt14",
        "hf_config": "fr-en",
        "local_dir_name": "wmt14",
    },
}


MDUQ_DATASET_REGISTRY: Dict[str, Dict] = {
    "mmlu": {
        "task_type": "multiple_choice_qa",
        "split_names": ("train", "validation", "dev", "test"),
        "supports_generation": False,
        "supports_multiple_choice": True,
        "supports_dimension_proxy": True,
        "hf_repo_id": "cais/mmlu",
        "local_dir_name": "mmlu",
    },
    "triviaqa": {
        "task_type": "open_domain_qa",
        "split_names": ("train", "validation"),
        "supports_generation": True,
        "supports_multiple_choice": False,
        "supports_dimension_proxy": True,
        "hf_repo_id": "trivia_qa",
        "hf_config": "rc.nocontext",
        "local_dir_name": "trivia_qa",
    },
    "ambigqa": {
        "task_type": "ambiguity_qa",
        "split_names": ("train", "validation"),
        "supports_generation": True,
        "supports_multiple_choice": False,
        "supports_dimension_proxy": True,
        "hf_repo_id": "sewon/ambig_qa",
        "hf_config": "light",
        "local_dir_name": "ambig_qa",
    },
    "truthfulqa": {
        "task_type": "truthfulness_qa",
        "split_names": ("validation",),
        "available_splits": ("validation",),
        "default_single_split_policy": "internal_split",
        "allow_silent_train_fallback": False,
        "supports_generation": True,
        "supports_multiple_choice": True,
        "supports_dimension_proxy": True,
        "hf_repo_id": "truthful_qa",
        "hf_config": "generation",
        "local_dir_name": "truthful_qa",
    },
    "wmt": {
        "task_type": "translation",
        "split_names": ("train", "test"),
        "supports_generation": True,
        "supports_multiple_choice": False,
        "supports_dimension_proxy": True,
        "hf_repo_id": "wmt14",
        "hf_config": "fr-en",
        "local_dir_name": "wmt14",
    },
}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def list_legacy_datasets() -> List[str]:
    """Return the legacy-pipeline dataset names."""
    return list(LEGACY_DATASET_REGISTRY.keys())


def list_mduq_datasets() -> List[str]:
    """Return the MDUQ-pipeline dataset names."""
    return list(MDUQ_DATASET_REGISTRY.keys())


def list_all_datasets() -> List[str]:
    """Return the union of legacy and MDUQ dataset names (deduplicated)."""
    seen = []
    for name in list_legacy_datasets() + list_mduq_datasets():
        if name not in seen:
            seen.append(name)
    return seen


def get_dataset_spec(dataset_name: str, *, prefer: str = "auto") -> Dict:
    """Return the registry entry for ``dataset_name``.

    ``prefer`` controls which registry is consulted first when a name lives
    in both registries (currently ``mmlu``, ``triviaqa`` and ``wmt``):

    * ``"auto"`` (default) -- legacy first, then MDUQ.
    * ``"legacy"`` -- legacy only.
    * ``"mduq"``   -- MDUQ only.
    """
    if prefer not in ("auto", "legacy", "mduq"):
        raise ValueError(
            f"`prefer` must be 'auto', 'legacy' or 'mduq', got {prefer!r}"
        )
    if prefer in ("auto", "legacy") and dataset_name in LEGACY_DATASET_REGISTRY:
        return LEGACY_DATASET_REGISTRY[dataset_name]
    if prefer in ("auto", "mduq") and dataset_name in MDUQ_DATASET_REGISTRY:
        return MDUQ_DATASET_REGISTRY[dataset_name]
    raise KeyError(
        f"Unknown dataset '{dataset_name}'. Known datasets: "
        f"{list_all_datasets()}"
    )


def is_mduq_dataset(dataset_name: str) -> bool:
    return dataset_name in MDUQ_DATASET_REGISTRY


def is_legacy_dataset(dataset_name: str) -> bool:
    return dataset_name in LEGACY_DATASET_REGISTRY


def get_task_type(dataset_name: str, *, prefer: str = "auto") -> str:
    return get_dataset_spec(dataset_name, prefer=prefer)["task_type"]


def get_local_dir_name(dataset_name: str, *, prefer: str = "auto") -> str:
    return get_dataset_spec(dataset_name, prefer=prefer)["local_dir_name"]


def get_split_names(dataset_name: str, *, prefer: str = "auto"):
    return tuple(get_dataset_spec(dataset_name, prefer=prefer)["split_names"])


# ---------------------------------------------------------------------------
# DiagUQ-style aliases (the historical `MDUQ_*` / `list_mduq_*` names
# above are kept for backward compatibility).
# ---------------------------------------------------------------------------

DIAGUQ_DATASET_REGISTRY = MDUQ_DATASET_REGISTRY


def list_diaguq_datasets():
    """Alias for :func:list_mduq_datasets."""
    return list_mduq_datasets()
