from pathlib import Path

import datasets

from registry.dataset_registry import (
    LEGACY_DATASET_REGISTRY,
    MDUQ_DATASET_REGISTRY,
    get_dataset_spec,
    list_legacy_datasets,
    list_mduq_datasets,
)

MMLU_TASKS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


def _save_hf_to_disk(repo_id, hf_config, save_path):
    if hf_config is None:
        ds = datasets.load_dataset(repo_id)
    else:
        ds = datasets.load_dataset(repo_id, hf_config)
    ds.save_to_disk(str(save_path))


def prepare_dataset_by_name(dataset_name, save_dir, prefer: str = "auto"):
    """Download a registered dataset by name.

    MMLU is special-cased because it is shipped per-task on HuggingFace.
    """
    spec = get_dataset_spec(dataset_name, prefer=prefer)
    local_dir = Path(save_dir) / spec["local_dir_name"]
    if dataset_name == "mmlu":
        for task in MMLU_TASKS:
            ds = datasets.load_dataset(spec["hf_repo_id"], task)
            ds.save_to_disk(str(local_dir / task))
        return
    _save_hf_to_disk(spec["hf_repo_id"], spec.get("hf_config"), local_dir)


def prepare_legacy_datasets(save_dir):
    for name in list_legacy_datasets():
        prepare_dataset_by_name(name, save_dir, prefer="legacy")


def prepare_mduq_datasets(save_dir):
    for name in list_mduq_datasets():
        prepare_dataset_by_name(name, save_dir, prefer="mduq")


# ---------------------------------------------------------------------------
# Legacy thin wrappers (kept so the original CLI keeps working).
# ---------------------------------------------------------------------------


def prepare_triviaqa(save_dir):
    prepare_dataset_by_name("triviaqa", save_dir, prefer="legacy")


def prepare_coqa(save_dir):
    prepare_dataset_by_name("coqa", save_dir, prefer="legacy")


def prepare_mmlu(save_dir):
    prepare_dataset_by_name("mmlu", save_dir, prefer="legacy")


def prepare_wmt(save_dir):
    prepare_dataset_by_name("wmt", save_dir, prefer="legacy")


# ---------------------------------------------------------------------------
# New MDUQ-only helpers
# ---------------------------------------------------------------------------


def prepare_ambigqa(save_dir):
    prepare_dataset_by_name("ambigqa", save_dir, prefer="mduq")


def prepare_truthfulqa(save_dir):
    prepare_dataset_by_name("truthfulqa", save_dir, prefer="mduq")
