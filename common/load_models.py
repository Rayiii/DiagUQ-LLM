import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from registry.model_registry import (
    get_canonical_name,
    get_hf_repo_id,
    get_hf_token,
    get_local_dir_name,
    get_model_spec,
    normalize_model_name,
    resolve_model_id,
)


# ---------------------------------------------------------------------------
# Generation-config compatibility shims
# ---------------------------------------------------------------------------
#
# Some Llama-3.x release artifacts ship a ``generation_config.json`` whose
# ``pad_token_id`` field is stored as a list of ints (e.g.
# ``[128001, 128008, 128009]``). Recent ``transformers`` releases validate
# that field through a strict dataclass that only accepts ``int`` or
# ``None``, so ``AutoModelForCausalLM.from_pretrained`` aborts before any
# of our code runs.
#
# Per project policy we MUST NOT modify the downloaded HF files on disk
# (no edits to ``config.json`` / ``generation_config.json``). Instead we
# install a one-shot monkey-patch on ``GenerationConfig.from_dict`` that
# scalarizes ``pad_token_id`` in memory just before the strict validator
# runs. ``eos_token_id`` is intentionally left as-is because Llama-3.x
# legitimately uses a list of multiple EOS ids and the rest of the stack
# (including ``model.generate``) handles that correctly.


# Files that mark a directory as a self-contained HF model snapshot.
_REQUIRED_CONFIG_FILES = ("config.json",)
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")
_WEIGHT_INDEX_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


def is_valid_local_model_dir(path) -> Tuple[bool, Dict[str, object]]:
    """Return ``(is_valid, details)`` for a candidate local HF model dir.

    A directory is considered a usable local model snapshot when:

    * ``config.json`` is present,
    * at least one tokenizer file is present
      (``tokenizer.json`` / ``tokenizer_config.json`` / ``tokenizer.model``),
    * at least one weight file or weight index is present
      (``model.safetensors`` / ``model.safetensors.index.json`` /
      ``pytorch_model.bin`` / ``pytorch_model.bin.index.json``).

    ``details`` is a dict suitable for log lines.
    """
    p = Path(path)
    details: Dict[str, object] = {
        "expected_local_path": str(p),
        "config_exists": False,
        "tokenizer_exists": False,
        "weights_index_exists": False,
        "safetensors_count": 0,
        "local_valid": False,
    }
    if not p.is_dir():
        return False, details
    details["config_exists"] = all((p / f).is_file() for f in _REQUIRED_CONFIG_FILES)
    details["tokenizer_exists"] = any((p / f).is_file() for f in _TOKENIZER_FILES)
    details["weights_index_exists"] = any(
        (p / f).is_file() for f in _WEIGHT_INDEX_FILES
    )
    details["safetensors_count"] = sum(
        1 for _ in p.glob("*.safetensors")
    )
    details["local_valid"] = bool(
        details["config_exists"]
        and details["tokenizer_exists"]
        and details["weights_index_exists"]
    )
    return bool(details["local_valid"]), details


# ---------------------------------------------------------------------------
# Token-id normalization helpers (in-memory only)
# ---------------------------------------------------------------------------


def _first_scalar_token_id(value: Any) -> Optional[int]:
    """Return the first int element of ``value`` if it is a list/tuple,
    return ``value`` itself if it is already an int, else ``None``.
    """
    if value is None:
        return None
    if isinstance(value, bool):  # bool is a subclass of int -- ignore
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, (list, tuple)) and len(value) > 0:
        first = value[0]
        if isinstance(first, int) and not isinstance(first, bool):
            return int(first)
    return None


def _candidate_pad_sources(model, tokenizer) -> List[Tuple[str, Any]]:
    """Sources for the pad-token fallback chain, in priority order.

    The chain only reads ``pad_token_id`` from the tokenizer and
    ``eos_token_id`` from {tokenizer, generation_config, config}; it
    NEVER overwrites any ``eos_token_id`` field.
    """
    gen_cfg = getattr(model, "generation_config", None)
    cfg = getattr(model, "config", None)
    return [
        ("tokenizer.pad_token_id", getattr(tokenizer, "pad_token_id", None)),
        ("tokenizer.eos_token_id", getattr(tokenizer, "eos_token_id", None)),
        ("model.generation_config.eos_token_id",
         getattr(gen_cfg, "eos_token_id", None) if gen_cfg is not None else None),
        ("model.config.eos_token_id",
         getattr(cfg, "eos_token_id", None) if cfg is not None else None),
    ]


def normalize_special_token_ids(model, tokenizer) -> Optional[int]:
    """Pick a scalar ``pad_token_id`` and propagate it consistently.

    Selection order (first scalar int wins):

    1. ``tokenizer.pad_token_id``                          (already int)
    2. ``tokenizer.eos_token_id``                          (int)
    3. first int of ``tokenizer.eos_token_id``             (list)
    4. ``model.generation_config.eos_token_id``            (int)
    5. first int of ``model.generation_config.eos_token_id`` (list)
    6. ``model.config.eos_token_id``                       (int)
    7. first int of ``model.config.eos_token_id``          (list)

    The chosen scalar is then mirrored onto:

    * ``tokenizer.pad_token_id``
    * ``model.config.pad_token_id``
    * ``model.generation_config.pad_token_id``

    ``eos_token_id`` is never modified -- multi-EOS semantics for
    Llama-3.x are preserved. Returns the chosen scalar (or ``None`` if
    nothing usable was found, in which case the caller may decide to
    leave generation un-padded).
    """
    pad_id: Optional[int] = None
    chosen_source = "<none>"
    for source_name, source_value in _candidate_pad_sources(model, tokenizer):
        cand = _first_scalar_token_id(source_value)
        if cand is not None:
            pad_id = cand
            chosen_source = source_name
            break

    gen_cfg = getattr(model, "generation_config", None)
    cfg = getattr(model, "config", None)

    if pad_id is not None:
        # Tokenizer
        try:
            tokenizer.pad_token_id = pad_id
            if getattr(tokenizer, "pad_token", None) is None:
                # Best-effort: re-decode the chosen id into a string token.
                try:
                    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(pad_id)
                except Exception:
                    pass
        except Exception:
            pass
        # Model config
        if cfg is not None:
            try:
                cfg.pad_token_id = pad_id
            except Exception:
                pass
        # Generation config
        if gen_cfg is not None:
            try:
                gen_cfg.pad_token_id = pad_id
            except Exception:
                pass

    # Defensive: if generation_config still holds a list pad_token_id
    # (e.g. our patch didn't install in time), scalarize it now without
    # touching eos_token_id.
    if gen_cfg is not None:
        v = getattr(gen_cfg, "pad_token_id", None)
        if isinstance(v, (list, tuple)):
            scalar = _first_scalar_token_id(v)
            if scalar is not None:
                gen_cfg.pad_token_id = scalar

    eos_summary = _summarize_eos_field(model, tokenizer)
    logger.info(
        "[token-ids] chosen pad_token_id={pid} (source={src}); "
        "tokenizer.pad_token_id={tpi}, model.config.pad_token_id={mcp}, "
        "model.generation_config.pad_token_id={mgp}; eos_token_id={eos}",
        pid=pad_id,
        src=chosen_source,
        tpi=getattr(tokenizer, "pad_token_id", None),
        mcp=getattr(cfg, "pad_token_id", None) if cfg is not None else None,
        mgp=getattr(gen_cfg, "pad_token_id", None) if gen_cfg is not None else None,
        eos=eos_summary,
    )
    return pad_id


def _summarize_eos_field(model, tokenizer) -> str:
    """Compact, log-friendly summary of all observed ``eos_token_id`` values."""
    gen_cfg = getattr(model, "generation_config", None)
    cfg = getattr(model, "config", None)
    parts = []
    for name, value in (
        ("tok", getattr(tokenizer, "eos_token_id", None)),
        ("gen_cfg", getattr(gen_cfg, "eos_token_id", None) if gen_cfg is not None else None),
        ("cfg", getattr(cfg, "eos_token_id", None) if cfg is not None else None),
    ):
        parts.append(f"{name}={type(value).__name__}:{value}")
    return ", ".join(parts)


# Backward-compatible alias used by older imports.
def normalize_generation_config(model, tokenizer) -> None:
    normalize_special_token_ids(model, tokenizer)


def assert_pad_token_id_scalar(model) -> None:
    """Lightweight regression guard for generation call sites.

    Raises ``ValueError`` if ``model.generation_config.pad_token_id`` is
    a list/tuple at the moment of generation. If it is ``None`` we leave
    it alone (some generation paths intentionally generate un-padded).
    """
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is None:
        return
    v = getattr(gen_cfg, "pad_token_id", None)
    if isinstance(v, (list, tuple)):
        scalar = _first_scalar_token_id(v)
        logger.warning(
            "[token-ids] generation called with list pad_token_id={old}; "
            "scalarizing to {new}",
            old=v,
            new=scalar,
        )
        if scalar is not None:
            gen_cfg.pad_token_id = scalar
        else:
            raise ValueError(
                "model.generation_config.pad_token_id is a list with no "
                "usable int element."
            )


# ---------------------------------------------------------------------------
# Monkey-patch GenerationConfig.from_dict so list-typed pad_token_id is
# scalarized BEFORE the strict dataclass validator runs. Installed once.
# ---------------------------------------------------------------------------


def _install_generation_config_loader_patch() -> None:
    try:
        from transformers import GenerationConfig
    except Exception:
        return
    if getattr(GenerationConfig, "_diaguq_pad_token_patched", False):
        return
    original_from_dict = GenerationConfig.from_dict

    def _patched_from_dict(cls, config_dict, **kwargs):
        try:
            cd = dict(config_dict) if config_dict else config_dict
        except Exception:
            cd = config_dict
        if isinstance(cd, dict):
            v = cd.get("pad_token_id")
            if isinstance(v, (list, tuple)) and len(v) > 0:
                first = next(
                    (x for x in v if isinstance(x, int) and not isinstance(x, bool)),
                    None,
                )
                if first is not None:
                    cd["pad_token_id"] = int(first)
                    logger.info(
                        "[gen-config] in-memory loader patch: pad_token_id "
                        "{old} -> {new}",
                        old=v,
                        new=cd["pad_token_id"],
                    )
        return original_from_dict.__func__(cls, cd, **kwargs)

    GenerationConfig.from_dict = classmethod(_patched_from_dict)
    GenerationConfig._diaguq_pad_token_patched = True
    logger.debug(
        "[gen-config] installed in-memory pad_token_id normalization patch "
        "on transformers.GenerationConfig.from_dict"
    )


_install_generation_config_loader_patch()


# ---------------------------------------------------------------------------
# No-op placeholder for older callers. DiagUQ never modifies HF model files
# on disk; generation-config normalization happens in memory.
# ---------------------------------------------------------------------------


def patch_generation_config_file_if_needed(model_dir) -> bool:
    """Return ``False`` and never modify disk.

    Pad-token normalization now happens entirely in memory through
    :func:`_install_generation_config_loader_patch` and
    :func:`normalize_special_token_ids`.
    """
    return False


def _maybe_normalize_local_generation_config(model_id) -> None:
    """No-op; local generation configs are normalized in memory."""
    return None


# ---------------------------------------------------------------------------
# from_pretrained kwarg shim: prefer ``dtype`` (transformers >= 4.45) and
# fall back to the deprecated ``torch_dtype`` keyword on older versions.
# ---------------------------------------------------------------------------


def _dtype_kwarg(value):
    """Return the right kwarg dict for torch dtype across transformers versions."""
    try:
        sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
        params = sig.parameters
    except (TypeError, ValueError):
        return {"torch_dtype": value}
    if "dtype" in params:
        return {"dtype": value}
    return {"torch_dtype": value}


def load_gemma(model_path, tokenizer_path):
    """
    Load the gemma model and tokenizer.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path, **_dtype_kwarg(torch.float16)
    ).cuda()

    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    normalize_special_token_ids(model, tokenizer)
    return model, tokenizer


def load_llama2(model_path, tokenizer_path, access_token=None):
    """
    Load the llama2 model and tokenizer.

    The pad/eos token-id normalization happens in
    :func:`normalize_special_token_ids` after load. The previous
    assignment ``model.config.pad_token_id = model.config.eos_token_id``
    was the original source of the ``StrictDataclassFieldValidationError``
    on Llama-3.1, since Llama-3 ships ``eos_token_id`` as a list.
    """
    common_kwargs: Dict[str, Any] = {
        "device_map": "auto",
        **_dtype_kwarg(torch.float16),
    }
    if access_token is not None:
        common_kwargs["token"] = access_token

    model = AutoModelForCausalLM.from_pretrained(model_path, **common_kwargs)

    tokenizer_kwargs: Dict[str, Any] = {}
    if access_token is not None:
        tokenizer_kwargs["token"] = access_token
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)

    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Centralized token-id normalization. Replaces the previous unsafe
    # ``tokenizer.pad_token_id = tokenizer.eos_token_id`` /
    # ``model.config.pad_token_id = model.config.eos_token_id`` chain.
    normalize_special_token_ids(model, tokenizer)

    return model, tokenizer


def load_model_by_name(model_name, models_root=None):
    """Load a registered model + tokenizer by its registry name.

    Resolution order for the HF identifier passed to
    ``AutoModelForCausalLM.from_pretrained``:

    1. ``<models_root>/<local_dir_name>`` if it is a *valid* local model
       directory (see :func:`is_valid_local_model_dir`). When
       ``models_root`` is omitted it defaults to
       :func:`common.runtime_paths.get_models_dir`, which on AutoDL
       resolves to ``/root/autodl-tmp/DiagUQ_runtime/models``.
    2. otherwise the registry ``hf_repo_id`` (e.g.
       ``"Qwen/Qwen2.5-7B-Instruct"``).

    The display ``canonical_name`` is **never** used as a load identifier
    because it lacks the HF namespace prefix. For gated models the
    ``HF_TOKEN`` env variable is forwarded to ``from_pretrained``.
    """
    spec = get_model_spec(model_name)
    canonical = get_canonical_name(model_name)
    canonical_key = normalize_model_name(model_name)
    hf_repo_id = get_hf_repo_id(model_name)

    if models_root is None:
        from common.runtime_paths import get_models_dir
        models_root = str(get_models_dir())

    expected_local = Path(models_root) / spec["local_dir_name"]
    local_valid, details = is_valid_local_model_dir(expected_local)

    if local_valid:
        model_id = str(expected_local)
        is_local = True
    else:
        model_id = hf_repo_id
        is_local = False

    token = get_hf_token(model_name)

    logger.info(
        "[load_model_by_name] requested='{req}' normalized_key='{key}' "
        "canonical='{can}' hf_repo_id='{repo}' model_id='{mid}' "
        "expected_local_path='{elp}' config_exists={cfg} "
        "tokenizer_exists={tok_files} weights_index_exists={wix} "
        "safetensors_count={sct} local_valid={lv} uses_token={tok}",
        req=model_name,
        key=canonical_key,
        can=canonical,
        repo=hf_repo_id,
        mid=model_id,
        elp=details["expected_local_path"],
        cfg=details["config_exists"],
        tok_files=details["tokenizer_exists"],
        wix=details["weights_index_exists"],
        sct=details["safetensors_count"],
        lv=local_valid,
        tok=token is not None,
    )

    if is_local:
        # NOTE: invalid pad_token_id (list of ints, e.g. Llama-3.x) is
        # repaired purely in memory by the
        # ``GenerationConfig.from_dict`` monkey-patch installed at
        # module import. We do NOT modify any file under ``model_id``.
        pass

    return load_llama2(model_id, model_id, access_token=token)


def load_registered_causal_lm(model_name, models_root=None):
    """Load a registered causal LM by name using **relative** paths.

    The resolved load path is ``<models_root>/<local_dir_name>``.
    When ``models_root`` is omitted
    it is taken from :func:`common.runtime_paths.get_models_dir`, which
    on AutoDL points at ``/root/autodl-tmp/DiagUQ_runtime/models``.

    The HF access token, when required, is read from the ``HF_TOKEN``
    environment variable via :func:`registry.model_registry.get_hf_token`.
    Tokens are never read from constants or written to disk.

    Returns
    -------
    Tuple[AutoModelForCausalLM, AutoTokenizer]
    """
    if models_root is None:
        from common.runtime_paths import get_models_dir
        models_root = str(get_models_dir())
    return load_model_by_name(model_name, models_root=models_root)
