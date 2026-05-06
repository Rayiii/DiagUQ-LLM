"""Centralized runtime path resolution for DiagUQ.

DiagUQ separates *runtime artifacts* (large; should live on a data disk)
from *analysis outputs* (small; safe to keep inside the repository on
the system disk). This module is the single source of truth for those
locations across the whole codebase.

Namespace policy
----------------
The repository's top-level ``data/`` directory is a **Python source-code
package** (e.g. ``data.formatters``) and MUST NOT be used as a filesystem
root for downloaded datasets. Model definitions live under
``pipeline/``; downloaded model weights live under the resolved artifact
model root. The old source/artifact namespace collision produced bugs such as
``build-response-cache`` trying to read ``./data/trivia_qa`` (a
non-existent path inside the source-code package directory).

Artifact roots
--------------
Two modes are supported:

* **autodl / runtime mode** -- triggered by the env var
  ``DIAGUQ_RUNTIME_ROOT`` or by the presence of ``/root/autodl-tmp``.
  Layout:

  - ``<runtime_root>/data``
  - ``<runtime_root>/models``
  - ``<runtime_root>/test_output``
  - ``<runtime_root>/cache``

* **local fallback mode** -- everything else. Layout (under repo root):

    - ``./artifacts/data``
    - ``./artifacts/models``
    - ``./artifacts/pipeline``
    - ``./artifacts/cache``

Environment variables
---------------------

``DIAGUQ_RUNTIME_ROOT``
    Override the runtime root. Default in autodl mode:
    ``/root/autodl-tmp/DiagUQ_runtime``.

``DIAGUQ_ANALYSIS_ROOT``
    Override the analysis output root (csv summaries, plots, markdown
    tables). Default: ``<repo_root>/artifacts/results``.

``HF_TOKEN``
    Required by gated-model downloads (Llama / Gemma).
"""

from __future__ import annotations

import os
import sys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from loguru import logger
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal envs
    logger = logging.getLogger(__name__)

__all__ = [
    "get_repo_root",
    "get_runtime_root",
    "get_analysis_root",
    "get_models_dir",
    "get_data_dir",
    "get_test_output_dir",
    "get_cache_dir",
    "get_analysis_output_dir",
    "get_dataset_dir",
    "get_model_dir",
    "ensure_runtime_dirs",
    "describe_runtime_layout",
    "validate_environment",
    "is_autodl_mode",
    "RuntimeLayout",
    "MissingDatasetError",
    "MissingModelError",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_AUTODL_TMP = Path("/root/autodl-tmp")
_AUTODL_RUNTIME_DEFAULT = _AUTODL_TMP / "DiagUQ_runtime"
_LOCAL_ARTIFACT_ROOT = "artifacts"
_ANALYSIS_DIR_NAME = f"{_LOCAL_ARTIFACT_ROOT}/results"

# Runtime-mode subdirectory names (under DIAGUQ_RUNTIME_ROOT).
_RUNTIME_SUBDIR_DATA = "data"
_RUNTIME_SUBDIR_MODELS = "models"
_RUNTIME_SUBDIR_TEST_OUTPUT = "test_output"
_RUNTIME_SUBDIR_CACHE = "cache"

# Local-fallback directory names (under repo root). Keeping them under a
# single artifact root keeps generated files out of the source tree.
_LOCAL_DIR_DATA = f"{_LOCAL_ARTIFACT_ROOT}/data"
_LOCAL_DIR_MODELS = f"{_LOCAL_ARTIFACT_ROOT}/models"
_LOCAL_DIR_TEST_OUTPUT = f"{_LOCAL_ARTIFACT_ROOT}/pipeline"
_LOCAL_DIR_CACHE = f"{_LOCAL_ARTIFACT_ROOT}/cache"

# Local-fallback names that ``setup-autodl`` may symlink to runtime
# subdirectories.
_LOCAL_LINK_NAMES: Tuple[Tuple[str, str], ...] = (
    (_LOCAL_DIR_DATA, _RUNTIME_SUBDIR_DATA),
    (_LOCAL_DIR_MODELS, _RUNTIME_SUBDIR_MODELS),
    (_LOCAL_DIR_TEST_OUTPUT, _RUNTIME_SUBDIR_TEST_OUTPUT),
    (_LOCAL_DIR_CACHE, _RUNTIME_SUBDIR_CACHE),
)

# Names that must NEVER be touched by setup-autodl because they are
# source-code packages.
_RESERVED_PACKAGE_NAMES: Tuple[str, ...] = ("data", "pipeline")


# ---------------------------------------------------------------------------
# Repo-root resolution
# ---------------------------------------------------------------------------


def get_repo_root() -> Path:
    """Return the repository root.

    Resolved as the parent directory of this file's ``common/`` folder.
    """
    return Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------


def is_autodl_mode() -> bool:
    """Return True when DiagUQ should use the runtime data-disk layout."""
    if os.environ.get("DIAGUQ_RUNTIME_ROOT"):
        return True
    return _AUTODL_TMP.exists()


# ---------------------------------------------------------------------------
# Path accessors
# ---------------------------------------------------------------------------


def get_runtime_root() -> Optional[Path]:
    """Resolve the runtime root, or return ``None`` in local-fallback mode.

    * If ``DIAGUQ_RUNTIME_ROOT`` is set: return that path.
    * Else if ``/root/autodl-tmp`` exists: return the AutoDL default
      ``/root/autodl-tmp/DiagUQ_runtime``.
    * Else: return ``None`` -- callers should use the ``runtime_*``
      local-fallback directories under the repo root.
    """
    env = os.environ.get("DIAGUQ_RUNTIME_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    if _AUTODL_TMP.exists():
        return _AUTODL_RUNTIME_DEFAULT
    return None


def get_analysis_root() -> Path:
    """Resolve the root directory for small analysis outputs."""
    env = os.environ.get("DIAGUQ_ANALYSIS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return get_repo_root() / _ANALYSIS_DIR_NAME


def _resolve_artifact_root(runtime_subdir: str, local_dir: str) -> Path:
    """Pick ``<runtime_root>/<runtime_subdir>`` if a runtime root is set,
    otherwise ``<repo_root>/<local_dir>``.
    """
    rt = get_runtime_root()
    if rt is not None:
        return rt / runtime_subdir
    return get_repo_root() / local_dir


def get_models_dir() -> Path:
    """Directory holding downloaded HF model weights."""
    return _resolve_artifact_root(_RUNTIME_SUBDIR_MODELS, _LOCAL_DIR_MODELS)


def get_data_dir() -> Path:
    """Directory holding HF dataset caches."""
    return _resolve_artifact_root(_RUNTIME_SUBDIR_DATA, _LOCAL_DIR_DATA)


def get_test_output_dir() -> Path:
    """Root of the per-pair pipeline tree
    (``<dataset>/<model>/diaguq/...``)."""
    return _resolve_artifact_root(
        _RUNTIME_SUBDIR_TEST_OUTPUT, _LOCAL_DIR_TEST_OUTPUT
    )


def get_cache_dir() -> Path:
    """Generic working-cache directory."""
    return _resolve_artifact_root(_RUNTIME_SUBDIR_CACHE, _LOCAL_DIR_CACHE)


def get_analysis_output_dir() -> Path:
    """Default place for csv summaries, plots, markdown tables."""
    return get_analysis_root()


# ---------------------------------------------------------------------------
# Dataset / model directory helpers (single source of truth for the
# logical-name -> on-disk-folder mapping).
# ---------------------------------------------------------------------------


def get_dataset_dir(dataset_name: str, *, prefer: str = "auto") -> Path:
    """Resolve the on-disk directory for a registered dataset.

    Folder-name normalization (e.g. logical ``triviaqa`` -> on-disk
    ``trivia_qa``) is centralized in
    :func:`registry.dataset_registry.get_local_dir_name`.
    """
    # Local import to avoid a circular dependency (registry has no need
    # to import this module, but some downstream modules import both).
    from registry.dataset_registry import get_local_dir_name
    local_name = get_local_dir_name(dataset_name, prefer=prefer)
    return get_data_dir() / local_name


def get_model_dir(model_name: str) -> Path:
    """Resolve the on-disk directory for a registered model."""
    from registry.model_registry import get_local_dir_name as _get_model_dir_name
    return get_models_dir() / _get_model_dir_name(model_name)


# ---------------------------------------------------------------------------
# Missing-asset errors
# ---------------------------------------------------------------------------


class MissingDatasetError(FileNotFoundError):
    """Raised when a registered dataset is not present at its canonical
    on-disk location. The message lists both the runtime and local
    fallback paths and explicitly notes the namespace policy."""


class MissingModelError(FileNotFoundError):
    """Raised when a registered model snapshot is not present at its
    canonical on-disk location. The message lists both the runtime and
    local fallback paths and explicitly notes the namespace policy."""


def _format_missing_asset(
    kind: str,
    logical_name: str,
    on_disk_name: str,
    package_name: str,
    runtime_subdir: str,
    local_dir: str,
) -> str:
    rt = get_runtime_root()
    runtime_path = (
        f"'{rt / runtime_subdir / on_disk_name}' (runtime mode)"
        if rt is not None
        else (
            f"'<DIAGUQ_RUNTIME_ROOT>/{runtime_subdir}/{on_disk_name}' "
            "(runtime mode; not active because DIAGUQ_RUNTIME_ROOT is unset "
            "and /root/autodl-tmp is absent)"
        )
    )
    local_path = f"'./{local_dir}/{on_disk_name}' (local fallback)"
    return (
        f"{kind} '{logical_name}' expected at {runtime_path} or "
        f"{local_path}, but was not found. "
        f"Note: './{package_name}' is a source-code package, not a "
        f"{kind.lower()} root."
    )


def require_dataset_dir(dataset_name: str, *, prefer: str = "auto") -> Path:
    """Return the on-disk directory for ``dataset_name`` if it exists,
    otherwise raise :class:`MissingDatasetError` with a message that
    explains the namespace policy.
    """
    from registry.dataset_registry import get_local_dir_name
    local_name = get_local_dir_name(dataset_name, prefer=prefer)
    path = get_data_dir() / local_name
    if not path.exists():
        raise MissingDatasetError(
            _format_missing_asset(
                kind="Dataset",
                logical_name=dataset_name,
                on_disk_name=local_name,
                package_name="data",
                runtime_subdir=_RUNTIME_SUBDIR_DATA,
                local_dir=_LOCAL_DIR_DATA,
            )
        )
    return path


def require_model_dir(model_name: str) -> Path:
    """Return the on-disk directory for ``model_name`` if it exists,
    otherwise raise :class:`MissingModelError`.
    """
    from registry.model_registry import get_local_dir_name as _get_model_dir_name
    local_name = _get_model_dir_name(model_name)
    path = get_models_dir() / local_name
    if not path.exists():
        raise MissingModelError(
            _format_missing_asset(
                kind="Model",
                logical_name=model_name,
                on_disk_name=local_name,
                package_name="models",
                runtime_subdir=_RUNTIME_SUBDIR_MODELS,
                local_dir=_LOCAL_DIR_MODELS,
            )
        )
    return path


# ---------------------------------------------------------------------------
# Layout description
# ---------------------------------------------------------------------------


@dataclass
class RuntimeLayout:
    repo_root: Path
    runtime_root: Optional[Path]
    analysis_root: Path
    models_dir: Path
    data_dir: Path
    test_output_dir: Path
    cache_dir: Path
    autodl_mode: bool

    def as_lines(self) -> List[str]:
        return [
            f"  repo_root      : {self.repo_root}",
            f"  runtime_root   : {self.runtime_root if self.runtime_root else '<unset; using ./artifacts/* local fallback>'}",
            f"  analysis_root  : {self.analysis_root}",
            f"  models_dir     : {self.models_dir}",
            f"  data_dir       : {self.data_dir}",
            f"  test_output_dir: {self.test_output_dir}",
            f"  cache_dir      : {self.cache_dir}",
            f"  autodl_mode    : {self.autodl_mode}",
        ]


def describe_runtime_layout() -> RuntimeLayout:
    """Return a snapshot of all resolved DiagUQ paths."""
    return RuntimeLayout(
        repo_root=get_repo_root(),
        runtime_root=get_runtime_root(),
        analysis_root=get_analysis_root(),
        models_dir=get_models_dir(),
        data_dir=get_data_dir(),
        test_output_dir=get_test_output_dir(),
        cache_dir=get_cache_dir(),
        autodl_mode=is_autodl_mode(),
    )


# ---------------------------------------------------------------------------
# Directory creation + (safe) symlink convenience
# ---------------------------------------------------------------------------


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def ensure_runtime_dirs(create_local_links: bool = True) -> RuntimeLayout:
    """Create runtime + analysis directories (idempotent).

    If a runtime root is configured and ``create_local_links`` is True,
    also create symlinks under ``./artifacts`` so local paths point at the
    runtime root subdirectories.

    Source directories such as ``data/`` and ``pipeline/`` are NEVER
    touched by this function; runtime data and model weights live under
    the resolved artifact roots.
    """
    layout = describe_runtime_layout()

    layout.analysis_root.mkdir(parents=True, exist_ok=True)
    for d in (
        layout.models_dir,
        layout.data_dir,
        layout.test_output_dir,
        layout.cache_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)
    if layout.runtime_root is not None:
        layout.runtime_root.mkdir(parents=True, exist_ok=True)

    if (
        create_local_links
        and layout.runtime_root is not None
        and layout.runtime_root.resolve() != layout.repo_root.resolve()
    ):
        for local_name, runtime_subdir in _LOCAL_LINK_NAMES:
            link_path = layout.repo_root / local_name
            target = layout.runtime_root / runtime_subdir
            target.mkdir(parents=True, exist_ok=True)
            _maybe_create_symlink(link_path, target)

    return layout


def _maybe_create_symlink(link_path: Path, target: Path) -> None:
    """Create ``link_path`` -> ``target`` if it is safe to do so.

    Refuses to touch top-level source package directories.
    """
    repo_root = get_repo_root().resolve()
    if link_path.parent.resolve() == repo_root and link_path.name in _RESERVED_PACKAGE_NAMES:
        return
    try:
        link_path.parent.mkdir(parents=True, exist_ok=True)
        if link_path.is_symlink():
            try:
                if link_path.resolve() == target.resolve():
                    return
            except OSError:
                pass
            link_path.unlink()
        elif link_path.exists():
            try:
                if link_path.is_dir() and not any(link_path.iterdir()):
                    link_path.rmdir()
                else:
                    return
            except OSError:
                return
        link_path.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pass


# ---------------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------------


def _is_writable(p: Path) -> bool:
    try:
        p.mkdir(parents=True, exist_ok=True)
        probe = p / ".diaguq_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def validate_environment(
    require_hf_token: bool = False,
) -> Dict[str, object]:
    """Validate runtime + analysis storage and (optionally) HF_TOKEN."""
    layout = describe_runtime_layout()
    problems: List[str] = []

    for d in (layout.models_dir, layout.data_dir,
              layout.test_output_dir, layout.cache_dir):
        if not _is_writable(d):
            problems.append(f"artifact dir not writable: {d}")
    if not _is_writable(layout.analysis_root):
        problems.append(f"analysis_root not writable: {layout.analysis_root}")
    if require_hf_token and not os.environ.get("HF_TOKEN"):
        problems.append(
            "HF_TOKEN is not set; gated model downloads (Llama, Gemma) will fail."
        )

    return {"ok": not problems, "problems": problems, "layout": layout}


# ---------------------------------------------------------------------------
# Convenience: log the resolved layout once per process
# ---------------------------------------------------------------------------

_LAYOUT_LOGGED = False


def log_layout_once(printer=None) -> None:
    """Print the resolved runtime layout the first time it's called.

    Quiet on subsequent calls. ``printer`` defaults to ``print``.
    """
    global _LAYOUT_LOGGED
    if _LAYOUT_LOGGED:
        return
    _LAYOUT_LOGGED = True
    layout = describe_runtime_layout()
    out = printer if printer is not None else print
    out("[diaguq] runtime layout:")
    for line in layout.as_lines():
        out(line)
    # Mirror the same info into loguru for log files.
    logger.debug("[runtime-paths] resolved layout:")
    for line in layout.as_lines():
        logger.debug(line)


if __name__ == "__main__":  # pragma: no cover
    layout = describe_runtime_layout()
    print("DiagUQ runtime layout:")
    for line in layout.as_lines():
        print(line)
    sys.exit(0)
