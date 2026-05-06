"""Model download utilities driven by ``registry.model_registry``.

All weights are downloaded under the resolved model artifact root. Tokens
for gated models are read only from the ``HF_TOKEN`` environment variable.

Public API:

* :func:`download_registered_model(model_name, save_dir)` -- download one
  registered model.
* :func:`download_registered_models(model_names, save_dir)` -- download a
  list of registered models.
* :func:`download_model_by_name(model_name, save_dir)` -- download one
    registered model by name.
* :func:`download_legacy_models(save_dir)` /
  :func:`download_mduq_models(save_dir)` -- batch helpers.

Model-specific wrappers call :func:`download_registered_model`.
"""

from pathlib import Path
from typing import Iterable, List, Optional

from huggingface_hub import snapshot_download
from loguru import logger

from registry.model_registry import (
    get_canonical_name,
    get_hf_token,
    get_local_dir_name,
    get_model_spec,
    list_legacy_models,
    list_mduq_models,
)


def _resolve_local_dir(save_dir, local_dir_name: str) -> Path:
    """Return ``<save_dir>/<local_dir_name>`` as an absolute-free Path."""
    return Path(save_dir) / local_dir_name


def _snapshot_download_repo(
    repo_id: str,
    local_dir: Path,
    access_token: Optional[str] = None,
) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "repo_id": repo_id,
        "local_dir": str(local_dir),
        "local_dir_use_symlinks": False,
    }
    if access_token is not None:
        kwargs["token"] = access_token
    snapshot_download(**kwargs)


def _target_model_is_complete(local_dir: Path) -> bool:
    """Heuristic completeness check for a downloaded target HF model.

    A directory is considered complete when it contains:
      * ``config.json``
      * at least one tokenizer artefact (``tokenizer.json`` or
        ``tokenizer.model`` or ``tokenizer_config.json``)
      * at least one weight file: a sharded index
        (``*.safetensors.index.json`` / ``pytorch_model.bin.index.json``)
        or a single weight blob (``model.safetensors`` / ``pytorch_model.bin``).
    """
    if not local_dir.is_dir():
        return False
    if not (local_dir / "config.json").exists():
        return False
    tokenizer_candidates = (
        "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
    )
    if not any((local_dir / f).exists() for f in tokenizer_candidates):
        return False
    single_weights = ("model.safetensors", "pytorch_model.bin")
    if any((local_dir / f).exists() for f in single_weights):
        return True
    # Sharded weights -> presence of an index json + at least one shard.
    index_files = list(local_dir.glob("*.index.json"))
    if not index_files:
        return False
    shards = (
        list(local_dir.glob("*.safetensors"))
        + list(local_dir.glob("pytorch_model-*.bin"))
    )
    return len(shards) > 0


def download_registered_model(
    model_name: str, save_dir, force: bool = False,
) -> Path:
    """Download the weights and tokenizer for one registered target model.

    Skips the network call if ``local_dir`` already looks complete (see
    :func:`_target_model_is_complete`) unless ``force=True``.

    The HF token (if required) is read from the ``HF_TOKEN`` environment
    variable; it is never written to disk or logged.
    """
    spec = get_model_spec(model_name)
    local_dir = _resolve_local_dir(save_dir, get_local_dir_name(model_name))
    token = get_hf_token(model_name)
    already = _target_model_is_complete(local_dir)
    logger.info(
        "[setup-models] scope=target model='{can}' local_path='{loc}' "
        "status={st} action={act} hf_repo_id='{repo}' uses_token={tok}",
        can=get_canonical_name(model_name),
        loc=str(local_dir),
        st="already_present" if already else "missing",
        act=("skip" if already and not force else "download"),
        repo=spec["hf_repo_id"],
        tok=token is not None,
    )
    if already and not force:
        return local_dir
    _snapshot_download_repo(spec["hf_repo_id"], local_dir, access_token=token)
    return local_dir


def download_registered_models(
    model_names: Iterable[str], save_dir, force: bool = False,
) -> List[dict]:
    """Download every model name in ``model_names``.

    Returns a per-model summary list with keys ``name``, ``local_dir``,
    ``status`` (``already_present``/``downloaded``/``error``), and
    optionally ``error``.
    """
    summary: List[dict] = []
    for name in model_names:
        spec = get_model_spec(name)
        local_dir = _resolve_local_dir(save_dir, get_local_dir_name(name))
        was_present = _target_model_is_complete(local_dir)
        try:
            download_registered_model(name, save_dir, force=force)
            summary.append({
                "name": get_canonical_name(name),
                "hf_repo_id": spec["hf_repo_id"],
                "local_dir": str(local_dir),
                "status": (
                    "already_present" if was_present and not force
                    else "downloaded"
                ),
            })
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[setup-models] scope=target model='{n}' FAILED error={e!r}",
                n=get_canonical_name(name), e=exc,
            )
            summary.append({
                "name": get_canonical_name(name),
                "hf_repo_id": spec["hf_repo_id"],
                "local_dir": str(local_dir),
                "status": "error",
                "error": repr(exc),
            })
    return summary


# Backward-compatible alias used by earlier refactors / external scripts.
def download_model_by_name(model_name: str, save_dir) -> Path:
    return download_registered_model(model_name, save_dir)


# ---------------------------------------------------------------------------
# Legacy thin wrappers (kept so the original CLI keeps working).
# ---------------------------------------------------------------------------


def download_llama2(save_dir):
    download_registered_model("llama_2_7b", save_dir)


def download_llama3(save_dir):
    download_registered_model("llama_3_8b", save_dir)


def download_gemma(save_dir):
    download_registered_model("gemma_7b", save_dir)


def download_deberta(save_dir):
    """Backwards-compatible shim. Prefer :func:`download_reference_model`."""
    download_reference_model("deberta-large-mnli", save_dir)


def download_reference_model(name: str, save_dir) -> Path:
    """Download one auxiliary/reference model (e.g. the NLI scorer used by
    semantic entropy). Reference models live in the same on-disk
    ``models/`` tree as target models but are tracked separately in
    :mod:`registry.reference_model_registry`.

    Public models do not require ``HF_TOKEN``; gated reference models
    (``spec.requires_hf_token=True``) read it from the environment.
    """
    from registry.reference_model_registry import (
        get_reference_model_spec,
        reference_model_is_available_locally,
    )

    spec = get_reference_model_spec(name)
    local_dir = _resolve_local_dir(save_dir, spec.local_dir_name)
    already = reference_model_is_available_locally(name, models_dir=save_dir)
    token = None
    if spec.requires_hf_token:
        import os
        token = os.environ.get("HF_TOKEN")
    logger.info(
        "[setup-models] scope=reference model='{n}' local_path='{l}' "
        "status={st} action={act} hf_repo_id='{r}' usage='{u}' "
        "uses_token={t}",
        n=spec.canonical_name, l=str(local_dir),
        st="already_present" if already else "missing",
        act="skip" if already else "download",
        r=spec.hf_repo_id, u=spec.usage, t=token is not None,
    )
    if already:
        return local_dir
    _snapshot_download_repo(spec.hf_repo_id, local_dir, access_token=token)
    return local_dir


# Spec-friendly alias requested in TASK 2.
def download_one_reference_model(name: str, save_dir) -> Path:
    """Alias for :func:`download_reference_model`."""
    return download_reference_model(name, save_dir)


def download_reference_models(
    save_dir,
    model_names: Optional[Iterable[str]] = None,
) -> List[dict]:
    """Download every reference/auxiliary model (or a supplied subset).

    Returns a list of per-model summary dicts with keys
    ``name``, ``hf_repo_id``, ``local_dir``, ``status`` (one of
    ``"downloaded"`` / ``"already_present"`` / ``"error"``), and on
    failure ``error``.
    """
    from registry.reference_model_registry import (
        list_reference_models,
        get_reference_model_spec,
        reference_model_is_available_locally,
    )

    targets = list(model_names) if model_names is not None else list_reference_models()
    logger.info(
        "[download-reference] starting batch save_dir='{d}' targets={t}",
        d=str(save_dir), t=targets,
    )
    summary: List[dict] = []
    for n in targets:
        spec = get_reference_model_spec(n)
        local_dir = _resolve_local_dir(save_dir, spec.local_dir_name)
        was_present = reference_model_is_available_locally(n, models_dir=save_dir)
        try:
            download_reference_model(n, save_dir)
            summary.append({
                "name": spec.canonical_name,
                "hf_repo_id": spec.hf_repo_id,
                "local_dir": str(local_dir),
                "status": "already_present" if was_present else "downloaded",
            })
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[download-reference] failed name='{n}' error={e!r}",
                n=spec.canonical_name, e=exc,
            )
            summary.append({
                "name": spec.canonical_name,
                "hf_repo_id": spec.hf_repo_id,
                "local_dir": str(local_dir),
                "status": "error",
                "error": repr(exc),
            })
    logger.info(
        "[download-reference] batch complete summary={s}", s=summary,
    )
    return summary


def download_legacy_models(save_dir) -> List[dict]:
    """Download every legacy model registered in the project."""
    return download_registered_models(list_legacy_models(), save_dir)


def download_mduq_models(save_dir) -> List[dict]:
    """Download every MDUQ model registered in the project."""
    return download_registered_models(list_mduq_models(), save_dir)
