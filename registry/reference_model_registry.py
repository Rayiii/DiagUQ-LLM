"""Auxiliary / reference model registry.

DiagUQ distinguishes two kinds of model dependencies:

* **Target models** -- the LLMs whose uncertainty we are estimating. These
  are managed by :mod:`registry.model_registry`.
* **Reference / auxiliary models** -- helper models used to compute
  comparison signals (e.g. an NLI model used by the semantic-entropy
  baseline). These are managed here.

Reference models live under the same runtime ``models/`` directory as the
target models, but are deliberately tracked in a separate registry so the
default ``setup-models`` flow does not pull them in unless the user opts
in via ``--include-reference-models`` (or runs ``setup-reference-models``
directly). The two namespaces are kept apart so that switching target
models never silently breaks comparison signals -- and vice versa.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from common.runtime_paths import get_models_dir


__all__ = [
    "ReferenceModelSpec",
    "REFERENCE_MODEL_REGISTRY",
    "list_reference_models",
    "get_reference_model_spec",
    "reference_model_local_dir",
    "reference_model_is_available_locally",
    "format_reference_model_setup_hint",
]


@dataclass(frozen=True)
class ReferenceModelSpec:
    """One auxiliary/reference model used for comparison signals."""

    canonical_name: str
    hf_repo_id: str
    local_dir_name: str
    usage: str            # e.g. "semantic_entropy"
    model_type: str       # e.g. "sequence_classification"
    requires_hf_token: bool = False


REFERENCE_MODEL_REGISTRY: Dict[str, ReferenceModelSpec] = {
    "deberta-large-mnli": ReferenceModelSpec(
        canonical_name="deberta-large-mnli",
        hf_repo_id="microsoft/deberta-large-mnli",
        local_dir_name="deberta-large-mnli",
        usage="semantic_entropy",
        model_type="sequence_classification",
        requires_hf_token=False,
    ),
}


def list_reference_models() -> List[str]:
    """Return all canonical reference-model names."""
    return list(REFERENCE_MODEL_REGISTRY.keys())


def get_reference_model_spec(name: str) -> ReferenceModelSpec:
    """Return the :class:`ReferenceModelSpec` for ``name``.

    Raises ``KeyError`` for unknown names so callers fail loudly instead
    of silently passing an unregistered string to Hugging Face APIs.
    """
    if name in REFERENCE_MODEL_REGISTRY:
        return REFERENCE_MODEL_REGISTRY[name]
    raise KeyError(
        f"Unknown reference model '{name}'. Known: "
        f"{sorted(REFERENCE_MODEL_REGISTRY.keys())}"
    )


def reference_model_local_dir(
    name: str, models_dir: Optional[Path] = None
) -> Path:
    """Resolve the on-disk directory for the given reference model."""
    spec = get_reference_model_spec(name)
    base = Path(models_dir) if models_dir is not None else Path(get_models_dir())
    return base / spec.local_dir_name


def reference_model_is_available_locally(
    name: str, models_dir: Optional[Path] = None
) -> bool:
    """Return True if a usable local snapshot of the model exists.

    A directory is considered usable when it contains a ``config.json`` --
    this matches what ``transformers.AutoModel*.from_pretrained`` actually
    needs to load offline, and avoids false positives from empty stub
    folders left over by a failed download.
    """
    local_dir = reference_model_local_dir(name, models_dir=models_dir)
    return local_dir.is_dir() and (local_dir / "config.json").is_file()


def format_reference_model_setup_hint(name: str) -> str:
    """Return a one-line, copy-pasteable command to install the model."""
    spec = get_reference_model_spec(name)
    local = reference_model_local_dir(name)
    return (
        f"reference model '{spec.canonical_name}' "
        f"(hf_repo_id='{spec.hf_repo_id}') is not available locally at "
        f"{local}. Fix with one of: "
        f"`python run.py setup-reference-models` (recommended -- "
        f"downloads only the auxiliary model), "
        f"`python run.py setup-models --only-reference-models`, "
        f"or `python run.py setup-models --include-reference-models`."
    )
