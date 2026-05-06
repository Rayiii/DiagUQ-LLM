"""Unified model registry for the baseline and DiagUQ pipelines.

All model-specific metadata (HF repo id, local directory name, hidden size,
candidate hidden-state layers, and HF-token requirement) is centralized here
so that switching or adding a model only requires editing this file.

The registry is split in two:

* ``LEGACY_MODEL_REGISTRY`` -- models used by the baseline
  supervised-uncertainty estimator (also exported as
  ``BASELINE_MODEL_REGISTRY`` via the ``DIAGUQ_*`` alias block).
* ``MDUQ_MODEL_REGISTRY`` -- models used by the DiagUQ pipeline
  (also exported as ``DIAGUQ_MODEL_REGISTRY``).

Every entry exposes the following keys:

``hf_repo_id``        : Hugging Face repo identifier, e.g. ``"meta-llama/Llama-2-7b-hf"``.
``local_dir_name``    : Folder name used as the local cache. Resolved
                        through :func:`common.runtime_paths.get_models_dir`
                        (which uses ``<runtime_root>/models`` in autodl
                        mode and ``./artifacts/models`` in local fallback
                        mode). The source-code package ``./models`` is
                        NEVER used as a model cache root.
``hidden_size``       : Hidden state dimension of the transformer.
``candidate_layers``  : Layer indices used as features. For legacy models this
                        is the pair ``(mid, last)`` historically used in the
                        codebase. For MDUQ models this is a richer multi-layer
                        bank that downstream fusion modules can sample from.
``requires_hf_token`` : ``True`` when downloading or loading the weights needs
                        an authenticated Hugging Face token (read from the
                        ``HF_TOKEN`` environment variable).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

LEGACY_MODEL_REGISTRY: Dict[str, Dict] = {
    "llama_2_7b": {
        "hf_repo_id": "meta-llama/Llama-2-7b-hf",
        "local_dir_name": "Llama-2-7b-hf-local",
        "hidden_size": 4096,
        "candidate_layers": [16, 32],
        "requires_hf_token": True,
    },
    "llama_2_13b": {
        "hf_repo_id": "meta-llama/Llama-2-13b-hf",
        "local_dir_name": "Llama-2-13b-hf-local",
        "hidden_size": 5120,
        "candidate_layers": [20, 40],
        "requires_hf_token": True,
    },
    "llama_3_8b": {
        "hf_repo_id": "meta-llama/Meta-Llama-3-8B",
        "local_dir_name": "Llama-3-8b-hf-local",
        "hidden_size": 4096,
        "candidate_layers": [16, 32],
        "requires_hf_token": True,
    },
    "gemma_7b": {
        "hf_repo_id": "google/gemma-7b",
        "local_dir_name": "gemma-7b",
        "hidden_size": 3072,
        "candidate_layers": [14, 28],
        "requires_hf_token": True,
    },
    "gemma_2b": {
        "hf_repo_id": "google/gemma-2b",
        "local_dir_name": "gemma-2b",
        "hidden_size": 2048,
        "candidate_layers": [9, 18],
        "requires_hf_token": True,
    },
}


MDUQ_MODEL_REGISTRY: Dict[str, Dict] = {
    "Qwen2.5-7B-Instruct": {
        "canonical_name": "Qwen2.5-7B-Instruct",
        "hf_repo_id": "Qwen/Qwen2.5-7B-Instruct",
        "local_dir_name": "Qwen2.5-7B-Instruct",
        "aliases": ["qwen_2_5_7b_instruct", "qwen_2_5_7b"],
        "hidden_size": 3584,
        # 28 transformer blocks -> hidden_states has 29 entries (0..28)
        "candidate_layers": [7, 14, 21, 28],
        "requires_hf_token": False,
    },
    "Llama-3.1-8B-Instruct": {
        "canonical_name": "Llama-3.1-8B-Instruct",
        "hf_repo_id": "meta-llama/Llama-3.1-8B-Instruct",
        "local_dir_name": "Llama-3.1-8B-Instruct",
        "aliases": ["llama_3_1_8b_instruct"],
        "hidden_size": 4096,
        # 32 transformer blocks -> hidden_states has 33 entries (0..32)
        "candidate_layers": [8, 16, 24, 32],
        "requires_hf_token": True,
    },
    "gemma-4-E4B-it": {
        "canonical_name": "gemma-4-E4B-it",
        "hf_repo_id": "google/gemma-4-E4B-it",
        "local_dir_name": "gemma-4-E4B-it",
        "aliases": ["gemma_4_e4b_it", "gemma_4_4b"],
        "hidden_size": 3072,
        # Conservative placeholder layer bank; adjust once the model is on hand.
        "candidate_layers": [8, 16, 24, 32],
        "requires_hf_token": True,
    },
}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def list_legacy_models() -> List[str]:
    """Return the list of legacy model names registered in the project."""
    return list(LEGACY_MODEL_REGISTRY.keys())


def list_mduq_models() -> List[str]:
    """Return the list of MDUQ model names registered in the project."""
    return list(MDUQ_MODEL_REGISTRY.keys())


def list_all_models() -> List[str]:
    """Return all known model names (legacy first, then MDUQ)."""
    return list_legacy_models() + list_mduq_models()


def normalize_model_name(model_name: str) -> str:
    """Map any registry key or declared alias to its canonical registry key.

    Accepts both registry keys (e.g. ``"Llama-3.1-8B-Instruct"``) and the
    user-facing aliases declared on each entry (e.g.
    ``"llama_3_1_8b_instruct"``). Raises ``KeyError`` for unknown names so
    callers fail loudly instead of silently passing an alias to Hugging
    Face.
    """
    mapping = iter_model_aliases()
    if model_name in mapping:
        return mapping[model_name]
    raise KeyError(
        f"Unknown model '{model_name}'. Known models / aliases: "
        f"{sorted(mapping.keys())}"
    )


def get_model_spec(model_name: str) -> Dict:
    """Return the registry entry for ``model_name`` (alias-aware).

    Looks up ``model_name`` first in the legacy registry, then in the MDUQ
    registry. If the name is neither a registry key nor a declared alias,
    raises ``KeyError``.
    """
    if model_name in LEGACY_MODEL_REGISTRY:
        return LEGACY_MODEL_REGISTRY[model_name]
    if model_name in MDUQ_MODEL_REGISTRY:
        return MDUQ_MODEL_REGISTRY[model_name]
    canonical = normalize_model_name(model_name)
    if canonical in LEGACY_MODEL_REGISTRY:
        return LEGACY_MODEL_REGISTRY[canonical]
    return MDUQ_MODEL_REGISTRY[canonical]


def is_mduq_model(model_name: str) -> bool:
    try:
        return normalize_model_name(model_name) in MDUQ_MODEL_REGISTRY
    except KeyError:
        return False


def is_legacy_model(model_name: str) -> bool:
    try:
        return normalize_model_name(model_name) in LEGACY_MODEL_REGISTRY
    except KeyError:
        return False


# ---------------------------------------------------------------------------
# Path / metadata helpers used by the rest of the codebase
# ---------------------------------------------------------------------------

def get_local_dir_name(model_name: str) -> str:
    """Return the folder name (no parent prefix) used to cache the model."""
    return get_model_spec(model_name)["local_dir_name"]


def get_model_paths(
    model_name: str, prefix: Optional[str] = None
) -> Tuple[str, str]:
    """Return ``(model_path, tokenizer_path)`` strings for ``model_name``.

    ``prefix`` (optional) is prepended to the local directory name. When left
    as ``None`` the bare local directory name is returned, matching the
    historical behavior in ``supervised_generation.py``. Pass the resolved
    runtime models root (e.g. from
    :func:`common.runtime_paths.get_models_dir`) to obtain a
    fully-qualified local path. Note: the source-code package
    ``./models`` is not a valid models root.
    """
    local_dir = get_local_dir_name(model_name)
    if prefix is None:
        return local_dir, local_dir
    full = str(Path(prefix) / local_dir)
    return full, full


def get_hidden_size(model_name: str) -> int:
    return int(get_model_spec(model_name)["hidden_size"])


def get_candidate_layers(model_name: str) -> List[int]:
    return list(get_model_spec(model_name)["candidate_layers"])


def get_layer_list_and_dim(model_name: str) -> Tuple[List[int], int]:
    """Convenience helper that returns ``(candidate_layers, hidden_size)``."""
    spec = get_model_spec(model_name)
    return list(spec["candidate_layers"]), int(spec["hidden_size"])


def requires_hf_token(model_name: str) -> bool:
    return bool(get_model_spec(model_name)["requires_hf_token"])


def get_hf_token(model_name: Optional[str] = None) -> Optional[str]:
    """Return the HF access token from the ``HF_TOKEN`` env var.

    If ``model_name`` is provided and that model does not require a token,
    ``None`` is returned even when the env var is set. The token is never
    persisted to disk by this function and must never be hard-coded.
    """
    if model_name is not None and not requires_hf_token(model_name):
        return None
    return os.environ.get("HF_TOKEN")


# ---------------------------------------------------------------------------
# Legacy peer-model mapping (used by transfer experiments on MMLU)
# ---------------------------------------------------------------------------

# For each "primary" legacy model the transfer evaluation uses two peer
# models from a different architecture family. Centralizing this here so
# Feature loaders consume the registry mapping directly.
LEGACY_PEER_MODELS: Dict[str, Tuple[str, str]] = {
    "gemma_7b": ("llama_2_7b", "llama_2_13b"),
    "llama_2_7b": ("gemma_7b", "gemma_2b"),
    "llama_3_8b": ("gemma_7b", "gemma_2b"),
}


def get_peer_models(model_name: str) -> Tuple[str, str]:
    """Return the two peer model names used for transferability evaluation."""
    if model_name not in LEGACY_PEER_MODELS:
        raise KeyError(
            f"No peer-model mapping for '{model_name}'. Known: "
            f"{list(LEGACY_PEER_MODELS.keys())}"
        )
    return LEGACY_PEER_MODELS[model_name]


# ---------------------------------------------------------------------------
# DiagUQ-style aliases.
# ---------------------------------------------------------------------------

DIAGUQ_MODEL_REGISTRY = MDUQ_MODEL_REGISTRY


def list_diaguq_models():
    """Alias for :func:list_mduq_models."""
    return list_mduq_models()


# ---------------------------------------------------------------------------
# HF repo / local-path resolution helpers
# ---------------------------------------------------------------------------

def get_hf_repo_id(model_name: str) -> str:
    """Return the canonical Hugging Face repo id for ``model_name``."""
    return get_model_spec(model_name)["hf_repo_id"]


def get_canonical_name(model_name: str) -> str:
    """Return the canonical registry key for ``model_name``.

    Accepts aliases. Falls back to the normalized key for legacy entries
    that do not explicitly carry a ``canonical_name`` field.
    """
    canonical = normalize_model_name(model_name)
    spec = get_model_spec(canonical)
    return str(spec.get("canonical_name", canonical))


def iter_model_aliases() -> Dict[str, str]:
    """Return a ``{alias: canonical_name}`` mapping across all registries.

    Each canonical key is also mapped to itself so callers can use the
    returned dict as a single normalization table.
    """
    out: Dict[str, str] = {}
    for registry in (LEGACY_MODEL_REGISTRY, MDUQ_MODEL_REGISTRY):
        for canonical, spec in registry.items():
            out[canonical] = canonical
            for alias in spec.get("aliases", []) or []:
                out[alias] = canonical
    return out


def resolve_model_id(
    model_name: str,
    models_root: Optional[str] = None,
) -> Tuple[str, bool]:
    """Resolve the HF-loadable identifier for ``model_name``.

    Returns ``(model_id, is_local)`` where:

    * ``model_id`` is the local cache path when it exists on disk,
      otherwise the canonical ``hf_repo_id``.
    * ``is_local`` is ``True`` iff a local directory with model files was
      found.

    ``models_root`` defaults to :func:`common.runtime_paths.get_models_dir`
    so the same resolution rule applies in both AutoDL and local fallback
    modes. Never returns the ``canonical_name`` (which is not a valid HF
    repo id for namespaced models). Aliases are accepted and normalized
    automatically.
    """
    canonical_key = normalize_model_name(model_name)
    spec = get_model_spec(canonical_key)
    if models_root is None:
        try:
            from common.runtime_paths import get_models_dir
            models_root = str(get_models_dir())
        except Exception:
            # Fallback uses the local artifact root, not the source package.
            models_root = "artifacts/models"
    local_dir = Path(models_root) / spec["local_dir_name"]
    # Use the strict validity check from common.load_models if available, so
    # we never claim a half-downloaded directory is usable. Falls back to a
    # cheap "non-empty dir" probe when transformers is not importable.
    is_local = False
    try:
        from common.load_models import is_valid_local_model_dir
        is_local, _ = is_valid_local_model_dir(local_dir)
    except Exception:
        is_local = local_dir.is_dir() and any(local_dir.iterdir())
    if is_local:
        return str(local_dir), True
    return str(spec["hf_repo_id"]), False
