"""DiagUQ command-line entry point.

Run ``python run.py --help`` for the grouped command list. Primary commands
are those in the DiagUQ section; comparison commands are kept under the
``baseline-*`` namespace.
"""

import csv
import json
import shutil
from pathlib import Path
from typing import *

import click
from loguru import logger

# ---------------------------------------------------------------------------
# Imports of the underlying functionality (canonical implementations live
# in their respective modules; registry/, features/, pipeline/, analysis/,
# and baseline/ provide the runtime entry points.)
# ---------------------------------------------------------------------------

# NOTE: heavier imports (transformers/torch via `data.download_models`,
# `data.download_datasets`, `baseline.*`, `pipeline.*`, `analysis.*`) are deferred
# into the command bodies so `python run.py --help` and the lightweight
# `setup-autodl` command stay fast and work even when optional deps
# (transformers, datasets, sklearn ...) are missing.

from registry.dataset_registry import (
    get_dataset_spec,
    get_split_names,
    list_diaguq_datasets,
    list_legacy_datasets,
)
from registry.model_registry import (
    list_diaguq_models,
    list_legacy_models,
    iter_model_aliases,
)
from common.runtime_paths import (
    describe_runtime_layout,
    ensure_runtime_dirs,
    get_analysis_output_dir,
    get_data_dir,
    get_models_dir,
    get_test_output_dir,
    log_layout_once,
    validate_environment,
)
from common.artifact_paths import normalize_split_tag, split_dataset_and_raw
from common.response_cache_limits import resolve_response_cache_limit
from common.single_split_policy import (
    DEFAULT_INTERNAL_SPLIT_SEED,
    DEFAULT_INTERNAL_TRAIN_RATIO,
    TRUTHFULQA_SINGLE_SPLIT_MESSAGE,
    internal_split_name,
    split_metadata_for_variant,
)


# ---------------------------------------------------------------------------
# Choice tables
# ---------------------------------------------------------------------------

AVAILABLE_BASELINE_DATASETS = tuple(list_legacy_datasets())
AVAILABLE_DIAGUQ_DATASETS = tuple(list_diaguq_datasets())
AVAILABLE_ALL_DATASETS = tuple(
    dict.fromkeys(AVAILABLE_BASELINE_DATASETS + AVAILABLE_DIAGUQ_DATASETS)
)
AVAILABLE_BASELINE_MODELS = tuple(list_legacy_models())
AVAILABLE_DIAGUQ_MODELS = tuple(list_diaguq_models())
AVAILABLE_ALL_MODELS = AVAILABLE_BASELINE_MODELS + AVAILABLE_DIAGUQ_MODELS


# ---------------------------------------------------------------------------
# Flexible model-name aliases
#
# Users frequently type model names in a "snake_case" form that does not
# match the canonical registry keys (which retain the upstream punctuation,
# e.g. ``Qwen2.5-7B-Instruct``). We accept a small set of explicit aliases
# and additionally fall back to a fuzzy match that ignores case and the
# difference between ``-``, ``_`` and ``.``.  Registry keys themselves are
# left untouched.
# ---------------------------------------------------------------------------

MODEL_ALIASES: Dict[str, str] = {
    "llama_3_1_8b_instruct": "Llama-3.1-8B-Instruct",
    "qwen_2_5_7b": "Qwen2.5-7B-Instruct",
    "gemma_4_4b": "gemma-4-E4B-it",
}
# Merge in any aliases declared in the registry itself so the registry
# stays the single source of truth. Existing entries above act as a
# stable baseline and are preserved for backward compatibility.
for _alias, _canonical in iter_model_aliases().items():
    if _alias != _canonical:
        MODEL_ALIASES.setdefault(_alias, _canonical)


def _canonicalize(name: str) -> str:
    """Normalize a model name for fuzzy comparison."""
    return name.lower().replace("-", "").replace("_", "").replace(".", "")


_FUZZY_MODEL_INDEX: Dict[str, str] = {
    _canonicalize(k): k for k in AVAILABLE_ALL_MODELS
}
_FUZZY_MODEL_INDEX.update(
    {_canonicalize(alias): canonical for alias, canonical in MODEL_ALIASES.items()}
)


def resolve_model_name(name: str) -> str:
    """Resolve a user-provided model name to a canonical registry key.

    Resolution order:
      1. Exact registry key -> return as-is.
      2. Explicit alias in ``MODEL_ALIASES`` -> convert and warn.
      3. Fuzzy match (case-insensitive, ignoring ``-``/``_``/``.``) ->
         convert and warn.
      4. Otherwise raise ``click.BadParameter`` listing available models.
    """
    if name in AVAILABLE_ALL_MODELS:
        return name

    if name in MODEL_ALIASES:
        canonical = MODEL_ALIASES[name]
        click.echo(
            f"[warn] Using alias '{name}' \u2192 '{canonical}'", err=True
        )
        return canonical

    fuzzy_key = _canonicalize(name)
    if fuzzy_key in _FUZZY_MODEL_INDEX:
        canonical = _FUZZY_MODEL_INDEX[fuzzy_key]
        if canonical != name:
            click.echo(
                f"[warn] Using alias '{name}' \u2192 '{canonical}'", err=True
            )
        return canonical

    available = "\n  ".join(AVAILABLE_ALL_MODELS)
    aliases = "\n  ".join(f"{a} -> {c}" for a, c in MODEL_ALIASES.items())
    raise click.BadParameter(
        f"Unknown model '{name}'.\n"
        f"Available registry keys:\n  {available}\n"
        f"Recognized aliases:\n  {aliases}"
    )


class ModelNameType(click.ParamType):
    """Click parameter type that accepts model aliases and fuzzy names."""

    name = "model"

    def __init__(self, allowed: Optional[Tuple[str, ...]] = None):
        self.allowed = allowed  # restrict to a subset (e.g. baseline-only)

    def convert(self, value, param, ctx):
        if value is None:
            return value
        try:
            canonical = resolve_model_name(value)
        except click.BadParameter as exc:
            self.fail(str(exc), param, ctx)
        if self.allowed is not None and canonical not in self.allowed:
            allowed = ", ".join(self.allowed)
            self.fail(
                f"Model '{canonical}' is not allowed for this command "
                f"(allowed: {allowed}).",
                param,
                ctx,
            )
        return canonical

    def get_metavar(self, param, ctx=None):
        return "MODEL"


MODEL_CHOICE_ALL = ModelNameType()
MODEL_CHOICE_BASELINE = ModelNameType(allowed=AVAILABLE_BASELINE_MODELS)
VIEW_FUSION_MODE_CHOICES = (
    "answer_only",
    "query_only",
    "relation_only",
    "uniform",
    "static_learned",
    "sample_adaptive",
    "sample_adaptive_regularized",
    "dimension_specific",
)
VIEW_GATE_SCOPE_CHOICES = ("shared", "dimension_specific")
DIAGNOSTIC_FACTORIZATION_MODE_CHOICES = (
    "shared_only",
    "independent_heads",
    "shared_plus_residual",
)
OVERALL_AGGREGATION_MODE_CHOICES = (
    "direct_head",
    "from_dimensions",
    "hybrid",
)


# ---------------------------------------------------------------------------
# Click group with sectioned --help.
# ---------------------------------------------------------------------------

DIAGUQ_COMMAND_NAMES: Tuple[str, ...] = (
    "setup-autodl",
    "setup-data",
    "setup-models",
    "setup-reference-models",
    "build-response-cache",
    "build-hidden-bank",
    "build-diagnostic-targets",
    "train-diaguq",
    "evaluate-diaguq",
    "ablate-layers",
    "ablate-dimensions",
    "ablate-views",
    "ablate-diagnostic-heads",
    "export-analysis",
    "download-one-dataset",
    "download-one-model",
)
BASELINE_COMMAND_NAMES: Tuple[str, ...] = (
    "baseline-setup-data",
    "baseline-setup-models",
    "baseline-build-features",
    "baseline-train",
    "baseline-evaluate",
    "baseline-transfer",
)

class _SectionedGroup(click.Group):
    """Click group that prints commands in labelled sections."""

    def format_commands(self, ctx, formatter):  # noqa: D401
        all_cmds = {name: self.get_command(ctx, name) for name in self.list_commands(ctx)}
        all_cmds = {n: c for n, c in all_cmds.items() if c is not None}

        def _section(title: str, names: Sequence[str]) -> None:
            rows = []
            for n in names:
                cmd = all_cmds.get(n)
                if cmd is None:
                    continue
                short = (cmd.get_short_help_str(limit=64) or "").strip()
                rows.append((n, short))
            if rows:
                with formatter.section(title):
                    formatter.write_dl(rows)

        _section("DiagUQ commands", DIAGUQ_COMMAND_NAMES)
        _section("Baseline commands", BASELINE_COMMAND_NAMES)


@click.group(cls=_SectionedGroup)
@click.option(
    "--show-paths/--no-show-paths", default=False,
    help="Print the resolved DiagUQ runtime layout (repo / runtime / analysis roots) before running.",
)
@click.pass_context
def run(ctx: click.Context, show_paths: bool):
    """DiagUQ -- Diagnostic Uncertainty Quantification from Multi-Layer
    Internal Activations.

    The primary commands live under the DiagUQ namespace; baseline routines
    are retained for controlled comparison.

    Storage roots are resolved by ``common.runtime_paths``. On AutoDL
    (when ``/root/autodl-tmp`` is present, or ``DIAGUQ_RUNTIME_ROOT`` is
    set) large artifacts go to the data disk; small analysis outputs
    stay under ``artifacts/results`` by default. Run
    ``python run.py setup-autodl`` once after cloning on a new VM.
    """
    ctx.ensure_object(dict)
    # Resolve canonical roots via the centralized runtime-path module so
    # large artifacts go to the data disk on AutoDL, while still working
    # transparently in local mode.
    data_dir = get_data_dir()
    model_dir = get_models_dir()
    ctx.obj["data_dir"] = data_dir
    ctx.obj["model_dir"] = model_dir
    ctx.obj["output_root"] = str(get_test_output_dir())
    ctx.obj["analysis_root"] = str(get_analysis_output_dir())
    if show_paths:
        log_layout_once()


def _resolve_diaguq_run_plan(
    *,
    command: str,
    scope: str,
    ds: Tuple[str, ...],
    models: Tuple[str, ...],
    dataset_alias: Tuple[str, ...] = (),
    model_alias: Tuple[str, ...] = (),
    split: Optional[str] = None,
    limit: Optional[int] = None,
    train_split: Optional[str] = None,
    eval_split: Optional[str] = None,
    train_limit: Optional[int] = None,
    eval_limit: Optional[int] = None,
    single_split_policy: Optional[str] = None,
    internal_train_ratio: float = DEFAULT_INTERNAL_TRAIN_RATIO,
    internal_split_seed: int = DEFAULT_INTERNAL_SPLIT_SEED,
    allow_single_split_same_eval: bool = False,
) -> Dict[str, Any]:
    selected_ds = tuple(dataset_alias or ds)
    selected_models = tuple(model_alias or models)
    if scope == "all":
        selected_ds = AVAILABLE_DIAGUQ_DATASETS
        selected_models = AVAILABLE_DIAGUQ_MODELS
    elif scope != "custom":
        raise click.UsageError(f"unknown scope: {scope}")
    base_pairs = [(dataset, model) for model in selected_models for dataset in selected_ds]

    def _available_splits(dataset: str) -> Tuple[str, ...]:
        base_dataset, raw_split = split_dataset_and_raw(dataset)
        if raw_split:
            return (raw_split,)
        try:
            return tuple(get_split_names(base_dataset, prefer="mduq"))
        except Exception:
            return ("train",)

    def _default_train_split(dataset: str) -> str:
        splits = _available_splits(dataset)
        if "train" in splits:
            return "train"
        return splits[0] if splits else "train"

    def _default_eval_split(dataset: str) -> str:
        splits = _available_splits(dataset)
        if "train" not in splits and "validation" in splits and "test" in splits:
            return "test"
        for candidate in ("validation", "dev", "test"):
            if candidate in splits:
                return candidate
        if len(splits) > 1:
            return splits[1]
        return splits[0] if splits else "train"

    def _single_split_spec(dataset: str) -> Dict[str, Any]:
        base_dataset, _ = split_dataset_and_raw(dataset)
        try:
            spec = dict(get_dataset_spec(base_dataset, prefer="mduq"))
        except Exception:
            return {}
        available = tuple(spec.get("available_splits") or spec.get("split_names") or ())
        if len(available) == 1 and spec.get("default_single_split_policy"):
            return {**spec, "source_split": available[0]}
        return {}

    def _qualify(dataset: str, requested_split: Optional[str]) -> str:
        base_dataset, raw_split = split_dataset_and_raw(dataset)
        split_name = requested_split or raw_split
        if not split_name:
            return dataset
        return normalize_split_tag(base_dataset, split_name)

    train_pairs: List[Tuple[str, str]] = []
    eval_pairs: List[Tuple[str, str]] = []
    pair_limits: Dict[Tuple[str, str], Optional[int]] = {}
    pair_context: Dict[Tuple[str, str], Dict[str, Any]] = {}
    checkpoint_pairs: List[Tuple[Tuple[str, str], Tuple[str, str]]] = []
    split_fallbacks: List[Dict[str, Any]] = []
    train_split_labels: List[str] = []
    eval_split_labels: List[str] = []
    for dataset, model in base_pairs:
        base_dataset, explicit_raw = split_dataset_and_raw(dataset)
        available_splits = _available_splits(dataset)
        split_policy = None
        single_spec = _single_split_spec(dataset)
        source_split = single_spec.get("source_split")
        if single_spec:
            if allow_single_split_same_eval:
                resolved_train_split = str(train_split or split or explicit_raw or source_split)
                resolved_eval_split = str(eval_split or split or explicit_raw or source_split)
                if resolved_train_split != source_split or resolved_eval_split != source_split:
                    raise click.UsageError(TRUTHFULQA_SINGLE_SPLIT_MESSAGE)
                split_policy = "same_split_debug"
            else:
                requested_policy = single_split_policy or str(single_spec.get("default_single_split_policy") or "")
                if requested_policy != "internal_split":
                    raise click.UsageError(TRUTHFULQA_SINGLE_SPLIT_MESSAGE)
                if train_split and train_split != internal_split_name(source_split, "train"):
                    raise click.UsageError(TRUTHFULQA_SINGLE_SPLIT_MESSAGE)
                if eval_split and eval_split != internal_split_name(source_split, "eval"):
                    raise click.UsageError(TRUTHFULQA_SINGLE_SPLIT_MESSAGE)
                if split and split != source_split:
                    raise click.UsageError(TRUTHFULQA_SINGLE_SPLIT_MESSAGE)
                if explicit_raw and explicit_raw != source_split:
                    raise click.UsageError(TRUTHFULQA_SINGLE_SPLIT_MESSAGE)
                resolved_train_split = internal_split_name(source_split, "train")
                resolved_eval_split = internal_split_name(source_split, "eval")
                split_policy = "internal_split"
        else:
            resolved_train_split = train_split or split or explicit_raw or _default_train_split(dataset)
            resolved_eval_split = eval_split or split or explicit_raw or _default_eval_split(dataset)
            if not (train_split or split or explicit_raw) and "train" not in available_splits and resolved_train_split:
                split_fallbacks.append({
                    "dataset": base_dataset,
                    "role": "train",
                    "requested_split": "train",
                    "resolved_split": resolved_train_split,
                    "available_splits": available_splits,
                    "reason": "registry has no train split",
                })
        train_dataset = _qualify(dataset, resolved_train_split)
        eval_dataset = _qualify(dataset, resolved_eval_split)
        train_pair = (train_dataset, model)
        eval_pair = (eval_dataset, model)
        train_pairs.append(train_pair)
        eval_pairs.append(eval_pair)
        checkpoint_pairs.append((eval_pair, train_pair))
        pair_limits[train_pair] = train_limit if train_limit is not None else limit
        pair_limits[eval_pair] = eval_limit if eval_limit is not None else limit
        context = {
            "base_dataset": base_dataset,
            "train_split": resolved_train_split,
            "eval_split": resolved_eval_split,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "held_out_evaluation": train_dataset != eval_dataset,
            "split_policy": split_policy,
            "internal_train_ratio": internal_train_ratio,
            "internal_split_seed": internal_split_seed,
            "same_split_evaluation": bool(train_dataset == eval_dataset),
        }
        context.update(split_metadata_for_variant(
            eval_dataset,
            policy=split_policy,
            seed=internal_split_seed,
            train_ratio=internal_train_ratio,
            same_split_debug=(split_policy == "same_split_debug"),
        ))
        pair_context[eval_pair] = context
        pair_context[train_pair] = context
        train_split_labels.append(resolved_train_split)
        eval_split_labels.append(resolved_eval_split)

    artifact_pairs = list(dict.fromkeys([*train_pairs, *eval_pairs]))
    if command in {"train-diaguq"}:
        pairs = list(dict.fromkeys(train_pairs))
    elif command.startswith("eval-") or command in {"export-analysis", "ablate-layers"}:
        pairs = list(dict.fromkeys(eval_pairs))
    else:
        pairs = artifact_pairs
    return {
        "command": command,
        "scope": scope,
        "split": split,
        "limit": limit,
        "train_split": ",".join(dict.fromkeys(train_split_labels)),
        "eval_split": ",".join(dict.fromkeys(eval_split_labels)),
        "train_limit": train_limit if train_limit is not None else limit,
        "eval_limit": eval_limit if eval_limit is not None else limit,
        "single_split_policy": single_split_policy,
        "internal_train_ratio": internal_train_ratio,
        "internal_split_seed": internal_split_seed,
        "allow_single_split_same_eval": allow_single_split_same_eval,
        "base_pairs": base_pairs,
        "train_pairs": list(dict.fromkeys(train_pairs)),
        "eval_pairs": list(dict.fromkeys(eval_pairs)),
        "checkpoint_pairs": list(dict.fromkeys(checkpoint_pairs)),
        "artifact_pairs": artifact_pairs,
        "pair_limits": pair_limits,
        "pair_context": pair_context,
        "split_fallbacks": split_fallbacks,
        "pairs": pairs,
        "num_pairs": len(pairs),
    }


def _print_run_plan(plan: Mapping[str, Any]) -> None:
    click.echo(
        f"[{plan['command']}] run_plan scope={plan['scope']} "
        f"pairs={plan['num_pairs']} train_split={plan.get('train_split') or 'default'} "
        f"eval_split={plan.get('eval_split') or 'default'} "
        f"train_limit={plan.get('train_limit') or 'default'} "
        f"eval_limit={plan.get('eval_limit') or 'default'}"
    )
    if plan.get("train_pairs") or plan.get("eval_pairs"):
        click.echo(f"  train_pairs={plan.get('train_pairs')}")
        click.echo(f"  eval_pairs={plan.get('eval_pairs')}")
    if plan.get("checkpoint_pairs") and (
        str(plan.get("command", "")).startswith("eval-")
        or plan.get("command") in {"export-analysis", "ablate-layers"}
    ):
        for eval_pair, train_pair in plan["checkpoint_pairs"]:
            click.echo(
                f"  checkpoint_pair eval={eval_pair[0]} "
                f"checkpoint={train_pair[0]} model={eval_pair[1]}"
            )
    policies = sorted({
        ctx.get("split_policy")
        for ctx in plan.get("pair_context", {}).values()
        if ctx.get("split_policy")
    })
    if policies:
        click.echo(
            "  split_policy={} internal_train_ratio={} internal_split_seed={}".format(
                ",".join(policies),
                plan.get("internal_train_ratio"),
                plan.get("internal_split_seed"),
            )
        )
    for fallback in plan.get("split_fallbacks", []):
        click.echo(
            "  split_fallback "
            f"dataset={fallback['dataset']} role={fallback['role']} "
            f"requested={fallback['requested_split']} resolved={fallback['resolved_split']} "
            f"available={fallback['available_splits']} reason={fallback['reason']}"
        )
    for idx, (dataset, model) in enumerate(plan["pairs"], start=1):
        click.echo(f"  {idx}. dataset={dataset} model={model}")


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _force_clear_stage_outputs(stage: str, pairs: Sequence[Tuple[str, str]], output_root: str) -> None:
    """Clear only stage-owned runtime outputs. Models and raw data are untouched."""
    from common.artifact_locator import locate_response_cache_artifacts
    from common.pair_context import resolve_pair_context

    for dataset, model in pairs:
        try:
            pair_ctx = resolve_pair_context(dataset, model, runtime_root=output_root)
            if stage == "response-cache":
                artifacts = locate_response_cache_artifacts(dataset, model, output_root)
                for key, path in artifacts.paths.items():
                    if key == "response_cache_dir":
                        continue
                    if path.exists():
                        _remove_path(path)
            elif stage == "hidden-bank":
                path = pair_ctx.hidden_bank_dir
                if path.exists():
                    _remove_path(path)
            elif stage == "diagnostic-targets":
                path = pair_ctx.dimension_targets_dir
                if path.exists():
                    _remove_path(path)
            elif stage == "training":
                path = pair_ctx.checkpoint_dir
                if path.exists():
                    _remove_path(path)
            elif stage == "evaluation":
                path = pair_ctx.eval_dir
                if path.exists():
                    _remove_path(path)
            elif stage == "export-analysis":
                path = pair_ctx.analysis_dir
                if path.exists():
                    _remove_path(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[{}] --force cleanup skipped dataset={} model={} error={!r}",
                stage, dataset, model, exc,
            )


def _assert_unique_stage_outputs(stage: str, pairs: Sequence[Tuple[str, str]], output_root: str) -> None:
    from common.pair_context import assert_no_duplicate_output_dirs, contexts_for_pairs

    contexts = contexts_for_pairs(pairs, runtime_root=output_root)
    assert_no_duplicate_output_dirs(contexts, stage)


def _preflight_layer_baseline_artifacts(
    plan: Mapping[str, Any],
    output_root: str,
    *,
    checkpoint_dataset: Optional[str] = None,
    strict: bool = False,
) -> None:
    from common.pair_context import resolve_checkpoint_context_for_eval, resolve_pair_context

    missing: List[str] = []
    train_pairs = list(plan.get("train_pairs") or [])
    for eval_pair, _train_pair in plan.get("checkpoint_pairs") or []:
        eval_ctx = resolve_pair_context(eval_pair[0], eval_pair[1], runtime_root=output_root)
        train_ctx = resolve_checkpoint_context_for_eval(
            eval_ctx,
            train_pairs,
            checkpoint_dataset=checkpoint_dataset,
        )
        required = [
            ("train_hidden_bank", train_ctx.hidden_bank_dir),
            ("train_dimension_targets", train_ctx.dimension_targets_dir / "dimension_targets.pt"),
            ("eval_hidden_bank", eval_ctx.hidden_bank_dir),
            ("eval_dimension_targets", eval_ctx.dimension_targets_dir / "dimension_targets.pt"),
            ("diaguq_metrics_for_comparison", eval_ctx.eval_dir / "metrics.json"),
        ]
        click.echo(
            "  layer_baseline_artifacts "
            f"eval={eval_ctx.resolved_variant} train={train_ctx.resolved_variant} "
            f"model={eval_ctx.model}"
        )
        for label, path in required:
            exists = path.is_dir() if label.endswith("hidden_bank") else path.is_file()
            click.echo(f"    {label}={path} exists={exists}")
            if not exists:
                missing.append(f"{label}: {path}")
    if strict and missing:
        detail = "\n".join(f"  - {item}" for item in missing)
        raise click.ClickException("missing required layer-baseline artifacts:\n" + detail)


# =======================  DiagUQ: setup commands  ==========================


@run.command(name="setup-autodl")
@click.option(
    "--require-hf-token/--no-require-hf-token",
    default=True, show_default=True,
    help="Fail when HF_TOKEN is not exported (needed for gated Llama / Gemma weights).",
)
@click.option(
    "--no-symlinks", is_flag=True, default=False,
    help="Skip creation of local ./artifacts links to the runtime root.",
)
def setup_autodl_cmd(require_hf_token: bool, no_symlinks: bool):
    """Initialize the AutoDL runtime artifact layout.

    Creates the runtime root (large artifacts on the data disk) and the
    analysis root (small outputs on the system disk), then validates
    that all artifact subdirectories are writable and that ``HF_TOKEN``
    is exported.

    On Linux, also creates local links under ``./artifacts`` that point
    at their counterparts under the runtime root. The repo's source-code
    package directories (``./data`` and ``./models``) are never touched.

    Safe to re-run; never overwrites existing user data.
    """
    layout = ensure_runtime_dirs(create_local_links=not no_symlinks)
    print("[setup-autodl] resolved runtime layout:")
    for line in layout.as_lines():
        print(line)
    report = validate_environment(require_hf_token=require_hf_token)
    if report["ok"]:
        print("[setup-autodl] OK -- runtime + analysis roots are writable.")
        if not require_hf_token:
            print("[setup-autodl] note: HF_TOKEN was not validated.")
    else:
        print("[setup-autodl] PROBLEMS:")
        for prob in report["problems"]:
            print(f"  - {prob}")
        raise click.ClickException(
            "setup-autodl failed; export the missing env vars and retry."
        )


@run.command(name="setup-data")
@click.pass_context
def setup_data(ctx: click.Context):
    """Download all datasets used by DiagUQ (mmlu, triviaqa, ambigqa,
    truthfulqa, wmt)."""
    from data.download_datasets import prepare_mduq_datasets
    log_layout_once()
    ctx.obj["data_dir"].mkdir(parents=True, exist_ok=True)
    prepare_mduq_datasets(ctx.obj["data_dir"])


@run.command(name="setup-models")
@click.pass_context
@click.option(
    "--include-reference-models/--no-include-reference-models",
    default=False, show_default=True,
    help=(
        "Also download auxiliary reference models (e.g. "
        "microsoft/deberta-large-mnli for the semantic-entropy baseline). "
        "Required for full-comparison runs."
    ),
)
@click.option(
    "--only-reference-models", is_flag=True, default=False,
    help=(
        "Download ONLY reference / auxiliary models (skip the target "
        "LLMs entirely). Useful when the target models are already cached "
        "and you just need the NLI scorer."
    ),
)
@click.option(
    "--only-target-models", is_flag=True, default=False,
    help="Download ONLY target LLMs (skip reference models).",
)
@click.option(
    "--force/--no-force", default=False, show_default=True,
    help=(
        "Re-download even if a target model directory already looks "
        "complete locally."
    ),
)
def setup_models(
    ctx: click.Context,
    include_reference_models: bool,
    only_reference_models: bool,
    only_target_models: bool,
    force: bool,
):
    """Download DiagUQ models. By default downloads only **target LLMs**
    (the models whose uncertainty we estimate) and skips any directories
    that already look complete locally.

    Reference / auxiliary models (helpers used for comparison signals
    such as semantic entropy) are tracked in
    :mod:`registry.reference_model_registry`. Use one of:

    \b
      --include-reference-models   target + reference (skip cached targets)
      --only-reference-models      reference only (skip targets entirely)
      --only-target-models         target only (default behavior)

    Or run the dedicated ``setup-reference-models`` command.
    """
    from data.download_models import (
        download_mduq_models, download_reference_models,
    )

    if only_reference_models and only_target_models:
        raise click.UsageError(
            "--only-reference-models and --only-target-models are mutually "
            "exclusive."
        )
    if only_reference_models and include_reference_models:
        # --only-* implies the include flag; just inform the user.
        logger.info(
            "[setup-models] --only-reference-models supersedes "
            "--include-reference-models"
        )

    do_targets = not only_reference_models
    do_refs = include_reference_models or only_reference_models
    if only_target_models:
        do_refs = False

    log_layout_once()
    if do_targets and do_refs:
        scope = "target+reference"
    elif do_targets:
        scope = "target"
    else:
        scope = "reference"
    logger.info(
        "[setup-models] starting scope={} model_dir={} force={}",
        scope, ctx.obj["model_dir"], force,
    )
    ctx.obj["model_dir"].mkdir(parents=True, exist_ok=True)

    target_summary: list = []
    reference_summary: list = []
    if do_targets:
        logger.info("[setup-models] downloading TARGET models")
        target_summary = download_mduq_models(ctx.obj["model_dir"]) or []
    else:
        logger.info("[setup-models] target models SKIPPED (per CLI flags)")

    if do_refs:
        logger.info("[setup-models] downloading REFERENCE / auxiliary models")
        reference_summary = download_reference_models(ctx.obj["model_dir"]) or []
    else:
        logger.info(
            "[setup-models] reference models SKIPPED. Pass "
            "--include-reference-models, --only-reference-models, or run "
            "setup-reference-models to fetch the semantic-entropy NLI scorer."
        )

    def _count(summary, status):
        return sum(1 for s in summary if s.get("status") == status)

    logger.info(
        "[setup-models] DONE scope={} target_total={} target_downloaded={} "
        "target_skipped={} target_failed={} reference_total={} "
        "reference_downloaded={} reference_skipped={} reference_failed={}",
        scope,
        len(target_summary),
        _count(target_summary, "downloaded"),
        _count(target_summary, "already_present"),
        _count(target_summary, "error"),
        len(reference_summary),
        _count(reference_summary, "downloaded"),
        _count(reference_summary, "already_present"),
        _count(reference_summary, "error"),
    )


@run.command(name="setup-reference-models")
@click.pass_context
def setup_reference_models(ctx: click.Context):
    """Download auxiliary/reference models used for comparison signals
    (e.g. ``microsoft/deberta-large-mnli`` for the semantic-entropy
    baseline). Idempotent and safe to re-run."""
    from data.download_models import download_reference_models
    log_layout_once()
    logger.info(
        "[setup-reference-models] downloading REFERENCE / auxiliary "
        "models into model_dir={}", ctx.obj["model_dir"],
    )
    ctx.obj["model_dir"].mkdir(parents=True, exist_ok=True)
    download_reference_models(ctx.obj["model_dir"])


@run.command(name="download-one-dataset")
@click.pass_context
@click.option(
    "--name", "-n",
    type=click.Choice(AVAILABLE_ALL_DATASETS),
    required=True,
    help="Registry name of the dataset to download.",
)
@click.option(
    "--prefer",
    type=click.Choice(["auto", "baseline", "diaguq", "legacy", "mduq"]),
    default="auto", show_default=True,
    help=(
        "Which registry to consult when the name exists in both. "
        "`baseline` and `diaguq` are the public choices; `legacy` and "
        "`mduq` are accepted for existing scripts."
    ),
)
def download_one_dataset(ctx: click.Context, name: str, prefer: str):
    """Download a single dataset by its registry name."""
    from data.download_datasets import prepare_dataset_by_name
    # Normalize user-facing names to the historical internal tokens that
    # `prepare_dataset_by_name` understands.
    prefer = {"baseline": "legacy", "diaguq": "mduq"}.get(prefer, prefer)
    ctx.obj["data_dir"].mkdir(parents=True, exist_ok=True)
    prepare_dataset_by_name(name, ctx.obj["data_dir"], prefer=prefer)


@run.command(name="inspect-dataset")
@click.option("--dataset", "dataset_name", type=click.Choice(AVAILABLE_ALL_DATASETS), required=True)
@click.option("--split", default=None, help="Requested DiagUQ split, e.g. train/validation/test.")
@click.option("--limit", type=int, default=3, show_default=True)
def inspect_dataset(dataset_name: str, split: Optional[str], limit: int):
    """Inspect local dataset formatting without loading an LLM."""
    base_dataset, embedded_split = split_dataset_and_raw(dataset_name)
    requested_split = split or embedded_split or "train"
    if base_dataset != "mmlu":
        raise click.ClickException("inspect-dataset currently supports --dataset mmlu.")

    from registry.dataset_registry import get_local_dir_name
    from data.mmlu_loader import (
        discover_mmlu_subjects,
        load_mmlu_subject_split,
        mmlu_actual_split,
        normalize_mmlu_row,
    )

    root = get_data_dir() / get_local_dir_name("mmlu", prefer="mduq")
    actual_split = mmlu_actual_split(requested_split)
    click.echo(f"dataset=mmlu local_path={root}")
    click.echo(f"requested_split={requested_split} actual_mmlu_split={actual_split}")
    subjects = discover_mmlu_subjects(root)
    if not subjects:
        raise click.ClickException(f"No MMLU subject directories found under {root}")
    shown = 0
    for subject_dir in subjects:
        if shown >= limit:
            break
        try:
            ds = load_mmlu_subject_split(subject_dir, actual_split)
        except Exception as exc:  # noqa: BLE001
            click.echo(f"subject={subject_dir.name} skipped error={exc!r}")
            continue
        if len(ds) == 0:
            continue
        row = dict(ds[0])
        formatted = normalize_mmlu_row(
            row,
            subject_dir.name,
            actual_split,
            requested_split=requested_split,
            row_index=0,
        )
        prompt_preview = formatted["prompt"].replace("\n", " ")[:240]
        click.echo(f"subject={subject_dir.name}")
        click.echo(f"  first_row_keys={sorted(row.keys())}")
        click.echo(f"  gold_option={formatted['gold_option']} gold_answer_text={formatted['gold_answer_text']}")
        click.echo(f"  prompt_preview={prompt_preview}")
        shown += 1
    if shown == 0:
        raise click.ClickException(
            f"No rows found for requested_split={requested_split} actual_mmlu_split={actual_split} under {root}"
        )


@run.command(name="download-one-model")
@click.pass_context
@click.option(
    "--name", "-n",
    type=MODEL_CHOICE_ALL,
    required=True,
    help="Registry name (or alias) of the model to download.",
)
def download_one_model(ctx: click.Context, name: str):
    """Download a single model by its registry name."""
    from data.download_models import download_model_by_name
    ctx.obj["model_dir"].mkdir(parents=True, exist_ok=True)
    download_model_by_name(name, ctx.obj["model_dir"])


# =======================  DiagUQ: feature builders  ========================


def _diaguq_pair_options(func):
    func = click.option(
        "--allow-single-split-same-eval",
        is_flag=True,
        default=False,
        help="Use one physical split for both train and eval as an explicit non-held-out run.",
    )(func)
    func = click.option(
        "--internal-split-seed",
        type=int,
        default=DEFAULT_INTERNAL_SPLIT_SEED,
        show_default=True,
        help="Seed for deterministic sample_id hashing in single-split internal splits.",
    )(func)
    func = click.option(
        "--internal-train-ratio",
        type=float,
        default=DEFAULT_INTERNAL_TRAIN_RATIO,
        show_default=True,
        help="Train fraction for deterministic single-split internal splits.",
    )(func)
    func = click.option(
        "--single-split-policy",
        type=click.Choice(["internal_split"]),
        default=None,
        help="Protocol for datasets that only expose one physical split, e.g. TruthfulQA.",
    )(func)
    func = click.option(
        "--allow-train-eval-same-split",
        is_flag=True,
        default=False,
        help="Allow evaluation/reporting when train and eval splits resolve to the same artifact root.",
    )(func)
    func = click.option(
        "--eval-limit",
        type=int,
        default=None,
        help="Optional row cap for the evaluation split.",
    )(func)
    func = click.option(
        "--train-limit",
        type=int,
        default=None,
        help="Optional row cap for the training split.",
    )(func)
    func = click.option(
        "--eval-split",
        type=str,
        default=None,
        help="Held-out split used for evaluation/export artifacts. Defaults to validation/test when available.",
    )(func)
    func = click.option(
        "--train-split",
        type=str,
        default=None,
        help="Split used for training artifacts. Defaults to train when available.",
    )(func)
    func = click.option(
        "--continue-on-error/--fail-fast",
        default=False,
        show_default=True,
        help="Continue across dataset/model pairs after a failure. Default is fail-fast.",
    )(func)
    func = click.option(
        "--force/--no-force",
        default=False,
        show_default=True,
        help="Clear this stage's existing outputs for the planned pairs before running.",
    )(func)
    func = click.option(
        "--dry-run",
        is_flag=True,
        default=False,
        help="Print the resolved run plan and exit without running the heavy stage.",
    )(func)
    func = click.option(
        "--limit",
        type=int,
        default=None,
        help="Optional row cap where the stage supports it.",
    )(func)
    func = click.option(
        "--split",
        type=str,
        default=None,
        help="Optional split label for plan reporting and split-aware future stages.",
    )(func)
    func = click.option(
        "--scope",
        type=click.Choice(["custom", "all"]),
        default="custom",
        show_default=True,
        help="Run the explicit -d/-m selection or the full registered grid.",
    )(func)
    func = click.option(
        "--dataset",
        "dataset_alias",
        multiple=True,
        type=click.Choice(AVAILABLE_ALL_DATASETS),
        default=(),
        help="Alias for -d/--ds; may be repeated.",
    )(func)
    func = click.option(
        "--model",
        "model_alias",
        multiple=True,
        type=MODEL_CHOICE_ALL,
        default=(),
        help="Alias for -m/--models; may be repeated.",
    )(func)
    func = click.option(
        "--ds", "-d",
        multiple=True,
        type=click.Choice(AVAILABLE_ALL_DATASETS),
        default=("triviaqa",),
        show_default=True,
    )(func)
    func = click.option(
        "--models", "-m",
        multiple=True,
        type=MODEL_CHOICE_ALL,
        default=("llama_3_1_8b_instruct",),
        show_default=True,
    )(func)
    return func


@run.command(name="build-response-cache")
@click.pass_context
@_diaguq_pair_options
@click.option(
    "--skip-semantic-entropy/--no-skip-semantic-entropy",
    default=False, show_default=True,
    help=(
        "Skip the semantic-entropy stage. When False (default), the "
        "auxiliary NLI model 'microsoft/deberta-large-mnli' must be "
        "available locally; this is checked up front before any expensive "
        "decoding starts."
    ),
)
@click.option(
    "--ask4conf-limit",
    "ask4conf_debug_limit",
    type=int,
    default=None,
    help=(
        "Optional cap for ask4conf rows per split. By default "
        "ask4conf processes the full canonical response-cache artifact."
    ),
)
@click.option(
    "--source-error-policy",
    type=click.Choice(["fail", "skip", "retry_then_skip"]),
    default=None,
    help=(
        "Policy for ask4conf placeholder/missing source answers. Defaults to "
        "retry_then_skip for full runs and fail when --ask4conf-limit is used."
    ),
)
@click.option(
    "--source-error-threshold",
    type=float,
    default=0.005,
    show_default=True,
    help="Maximum ask4conf source-error rate before fail policy aborts.",
)
@click.option(
    "--qa-f1-threshold",
    type=float,
    default=0.5,
    show_default=True,
    help="Token-F1 threshold used with normalized exact match for open-domain QA correctness.",
)
@click.option(
    "--allow-full-formatting",
    is_flag=True,
    default=False,
    help="Allow uncapped formatter calls for very large dataset splits. Use only for intentional full runs.",
)
def build_response_cache(
    ctx,
    models: Tuple[str],
    ds: Tuple[str],
    dataset_alias: Tuple[str],
    model_alias: Tuple[str],
    scope: str,
    split: Optional[str],
    train_split: Optional[str],
    eval_split: Optional[str],
    limit: Optional[int],
    train_limit: Optional[int],
    eval_limit: Optional[int],
    single_split_policy: Optional[str],
    internal_train_ratio: float,
    internal_split_seed: int,
    allow_single_split_same_eval: bool,
    allow_train_eval_same_split: bool,
    dry_run: bool,
    force: bool,
    continue_on_error: bool,
    skip_semantic_entropy: bool,
    ask4conf_debug_limit: Optional[int],
    source_error_policy: Optional[str],
    source_error_threshold: float,
    qa_f1_threshold: float,
    allow_full_formatting: bool,
):
    """Build the dataset-level response cache: greedy answers,
    correctness labels, ask-4-confidence and sampled answers.

    These artefacts feed both DiagUQ and the baseline pipeline.
    """
    plan = _resolve_diaguq_run_plan(
        command="build-response-cache",
        scope=scope,
        ds=ds,
        models=models,
        dataset_alias=dataset_alias,
        model_alias=model_alias,
        split=split,
        limit=limit,
        train_split=train_split,
        eval_split=eval_split,
        train_limit=train_limit,
        eval_limit=eval_limit,
        single_split_policy=single_split_policy,
        internal_train_ratio=internal_train_ratio,
        internal_split_seed=internal_split_seed,
        allow_single_split_same_eval=allow_single_split_same_eval,
    )
    _print_run_plan(plan)
    pairs = plan["pairs"]
    _assert_unique_stage_outputs("response_cache", pairs, ctx.obj["output_root"])
    if dry_run:
        return

    resolved_source_error_policy = source_error_policy or (
        "fail" if ask4conf_debug_limit is not None else "retry_then_skip"
    )

    try:
        from data.formatters import validate_response_cache_dataset_request
    except Exception as exc:  # noqa: BLE001
        requested = ", ".join(dataset for dataset, _ in pairs)
        raise click.ClickException(
            "build-response-cache preflight failed before loading the LLM: "
            f"could not import dataset formatters for requested datasets [{requested}]. "
            f"Error: {exc!r}. Suggested fix: install/activate the environment "
            "from environment.yaml, then rerun the same build-response-cache command."
        ) from exc

    pair_limits = plan.get("pair_limits", {})
    for dataset, model in pairs:
        pair_limit = pair_limits.get((dataset, model), limit)
        effective_limit = resolve_response_cache_limit(
            dataset,
            pair_limit,
            allow_full_formatting=allow_full_formatting,
        )
        try:
            summary = validate_response_cache_dataset_request(
                dataset,
                sample_limit=min(int(effective_limit or pair_limit or 2), 2),
                requested_limit=effective_limit,
                allow_full_formatting=allow_full_formatting,
                internal_train_ratio=internal_train_ratio,
                internal_split_seed=internal_split_seed,
            )
        except Exception as exc:  # noqa: BLE001
            raise click.ClickException(str(exc)) from exc
        logger.info(
            "[response-cache] preflight dataset={} base_dataset={} split={} "
            "actual_mmlu_split={} formatter={} available_splits={} raw_sample_count={} "
            "formatted_sample_count={} internal_train_count={} internal_eval_count={} "
            "requested_limit={} effective_sample_count={} limit_applied_before_formatting={} "
            "sample_count={} source_error_policy={} source_error_threshold={:.4f} suggestion={}",
            dataset,
            summary.get("base_dataset"),
            summary.get("split"),
            summary.get("actual_mmlu_split"),
            summary.get("formatter"),
            summary.get("available_splits"),
            summary.get("raw_sample_count"),
            summary.get("formatted_sample_count"),
            summary.get("internal_train_count"),
            summary.get("internal_eval_count"),
            summary.get("requested_limit"),
            summary.get("effective_sample_count"),
            summary.get("limit_applied_before_formatting"),
            summary.get("sample_count"),
            resolved_source_error_policy,
            source_error_threshold,
            summary.get("suggestion"),
        )

    if force:
        _force_clear_stage_outputs("response-cache", pairs, ctx.obj["output_root"])

    # --- preflight: fail fast if a required reference model is missing.
    needs_semantic_entropy = (not skip_semantic_entropy) and any(
        split_dataset_and_raw(d)[0] != "mmlu" for d, _ in pairs
    )
    if needs_semantic_entropy:
        from registry.reference_model_registry import (
            format_reference_model_setup_hint,
            get_reference_model_spec,
            reference_model_is_available_locally,
            reference_model_local_dir,
        )
        ref_name = "deberta-large-mnli"
        spec = get_reference_model_spec(ref_name)
        local_dir = reference_model_local_dir(ref_name)
        local_exists = reference_model_is_available_locally(ref_name)
        logger.info(
            "[response-cache] preflight auxiliary_model={} hf_repo_id={} "
            "resolved_local_path={} local_exists={} usage={}",
            spec.canonical_name, spec.hf_repo_id, local_dir,
            local_exists, spec.usage,
        )
        if not local_exists:
            raise click.ClickException(
                "build-response-cache preflight failed: "
                + format_reference_model_setup_hint(ref_name)
                + " (Or rerun with --skip-semantic-entropy to disable that"
                  " stage.)"
            )

    from features.build_response_cache import build_response_cache_for_pairs

    results = []
    for pair in pairs:
        pair_limit = pair_limits.get(pair, limit)
        results.extend(build_response_cache_for_pairs(
            [pair],
            output_root=ctx.obj["output_root"],
            skip_on_error=continue_on_error,
            skip_semantic_entropy=skip_semantic_entropy,
            ask4conf_debug_limit=ask4conf_debug_limit if ask4conf_debug_limit is not None else pair_limit,
            limit=pair_limit,
            qa_f1_threshold=qa_f1_threshold,
            allow_full_formatting=allow_full_formatting,
            internal_train_ratio=internal_train_ratio,
            internal_split_seed=internal_split_seed,
            source_error_policy=resolved_source_error_policy,
            source_error_threshold=source_error_threshold,
        ))
    failed = []
    for r in results:
        if "error" in r:
            failed.append(r)
            logger.warning(
                f"[response-cache] {r['dataset']} / {r['model']} failed: {r['error']}"
            )
        else:
            logger.info(f"[response-cache] {r['dataset']} / {r['model']} ok")
    if failed and not continue_on_error:
        raise click.ClickException(f"build-response-cache failed for {len(failed)} pair(s).")


@run.command(name="build-hidden-bank")
@click.pass_context
@_diaguq_pair_options
@click.option(
    "--mmlu-phase",
    type=click.Choice(["validation", "test"]),
    multiple=True,
    default=("validation", "test"), show_default=True,
    help="Which MMLU phase(s) to extract when 'mmlu' is in --ds.",
)
def build_hidden_bank(
    ctx,
    models: Tuple[str],
    ds: Tuple[str],
    dataset_alias: Tuple[str],
    model_alias: Tuple[str],
    scope: str,
    split: Optional[str],
    train_split: Optional[str],
    eval_split: Optional[str],
    limit: Optional[int],
    train_limit: Optional[int],
    eval_limit: Optional[int],
    single_split_policy: Optional[str],
    internal_train_ratio: float,
    internal_split_seed: int,
    allow_single_split_same_eval: bool,
    allow_train_eval_same_split: bool,
    dry_run: bool,
    force: bool,
    continue_on_error: bool,
    mmlu_phase: Tuple[str],
):
    """Build the multi-layer hidden-state bank consumed by DiagUQ.

    Outputs land under the resolved runtime test_output variant, e.g.
    ``<dataset__split>/<model>/diaguq/hidden_bank/``. Requires
    ``build-response-cache`` to have run for each pair first.
    """
    plan = _resolve_diaguq_run_plan(
        command="build-hidden-bank",
        scope=scope,
        ds=ds,
        models=models,
        dataset_alias=dataset_alias,
        model_alias=model_alias,
        split=split,
        limit=limit,
        train_split=train_split,
        eval_split=eval_split,
        train_limit=train_limit,
        eval_limit=eval_limit,
        single_split_policy=single_split_policy,
        internal_train_ratio=internal_train_ratio,
        internal_split_seed=internal_split_seed,
        allow_single_split_same_eval=allow_single_split_same_eval,
    )
    _print_run_plan(plan)
    pairs = plan["pairs"]
    _assert_unique_stage_outputs("hidden_bank", pairs, ctx.obj["output_root"])
    from features.build_hidden_bank import (
        build_hidden_bank_for_pairs,
        preflight_hidden_bank_for_pairs,
    )

    try:
        preflight_hidden_bank_for_pairs(
            pairs,
            output_root=ctx.obj["output_root"],
            require_response_cache=not dry_run,
        )
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc
    if dry_run:
        return
    if force:
        _force_clear_stage_outputs("hidden-bank", pairs, ctx.obj["output_root"])

    results = build_hidden_bank_for_pairs(
        pairs,
        output_root=ctx.obj["output_root"],
        mmlu_phases=mmlu_phase,
        skip_on_error=continue_on_error,
        internal_train_ratio=internal_train_ratio,
        internal_split_seed=internal_split_seed,
    )
    failed = []
    for r in results:
        if "error" in r:
            failed.append(r)
            logger.warning(
                f"[hidden-bank] {r['dataset']} / {r['model']} failed: {r['error']}"
            )
        else:
            logger.info(
                f"[hidden-bank] {r['dataset']} / {r['model']} -> "
                "resolved diaguq/hidden_bank under runtime test_output"
            )
    if failed and not continue_on_error:
        raise click.ClickException(f"build-hidden-bank failed for {len(failed)} pair(s).")


@run.command(name="build-diagnostic-targets")
@click.pass_context
@_diaguq_pair_options
@click.option(
    "--output-root",
    type=click.Path(file_okay=False),
    default=None,
    help="Override the per-pair output tree root. "
         "Defaults to the runtime test_output (data disk on AutoDL, repo root locally).",
)
@click.option(
    "--strict/--no-strict",
    default=False, show_default=True,
    help="Fail when required intermediate scoring files are missing.",
)
@click.option(
    "--require-semantic-entropy/--allow-missing-semantic-entropy",
    default=False,
    show_default=True,
    help="Treat semantic entropy as required instead of marking ambiguity unavailable.",
)
@click.option(
    "--allow-degenerate-labels/--fail-degenerate-labels",
    default=False,
    show_default=True,
    help="Allow strict diagnostic-target builds to continue when correct has only one class.",
)
def build_diagnostic_targets(
    ctx,
    models: Tuple[str],
    ds: Tuple[str],
    dataset_alias: Tuple[str],
    model_alias: Tuple[str],
    scope: str,
    split: Optional[str],
    train_split: Optional[str],
    eval_split: Optional[str],
    limit: Optional[int],
    train_limit: Optional[int],
    eval_limit: Optional[int],
    single_split_policy: Optional[str],
    internal_train_ratio: float,
    internal_split_seed: int,
    allow_single_split_same_eval: bool,
    allow_train_eval_same_split: bool,
    dry_run: bool,
    force: bool,
    continue_on_error: bool,
    output_root: Optional[str],
    strict: bool,
    require_semantic_entropy: bool,
    allow_degenerate_labels: bool,
):
    """Build the four DiagUQ diagnostic targets per (dataset, model) pair.

    Reads existing ``*_mextend.json`` / ``*_mextend_rouge.json`` /
    ``*_mextend_bleu.json`` / ``*_semantic_entropy.json`` / ask4conf
    artefacts and writes ``dimension_targets.{json,pt}`` plus
    ``meta.json`` under
    ``./test_output/<dataset>/<model>/diaguq/dimension_targets/``.
    """
    if output_root is None:
        output_root = ctx.obj["output_root"]
    plan = _resolve_diaguq_run_plan(
        command="build-diagnostic-targets",
        scope=scope,
        ds=ds,
        models=models,
        dataset_alias=dataset_alias,
        model_alias=model_alias,
        split=split,
        limit=limit,
        train_split=train_split,
        eval_split=eval_split,
        train_limit=train_limit,
        eval_limit=eval_limit,
        single_split_policy=single_split_policy,
        internal_train_ratio=internal_train_ratio,
        internal_split_seed=internal_split_seed,
        allow_single_split_same_eval=allow_single_split_same_eval,
    )
    _print_run_plan(plan)
    pairs = plan["pairs"]
    _assert_unique_stage_outputs("dimension_targets", pairs, output_root)
    if dry_run:
        return
    if force:
        _force_clear_stage_outputs("diagnostic-targets", pairs, output_root)
    from pipeline.build_diagnostic_targets import (
        build_diagnostic_targets_for_pairs,
    )

    results = build_diagnostic_targets_for_pairs(
        pairs,
        output_root=output_root,
        strict=strict,
        require_semantic_entropy=require_semantic_entropy,
        allow_degenerate_labels=allow_degenerate_labels,
        skip_on_error=continue_on_error,
    )
    failed = []
    for res in results:
        if "error" in res:
            failed.append(res)
            logger.warning(
                f"[diag-targets] {res['dataset']} / {res['model']}: {res['error']}"
            )
        else:
            logger.info(
                f"[diag-targets] {res['dataset']} / {res['model']}: "
                f"{res['num_rows']} rows, missing={res['missing']} -> {res['json_path']}"
            )
    if failed and not continue_on_error:
        raise click.ClickException(f"build-diagnostic-targets failed for {len(failed)} pair(s).")


# =======================  DiagUQ: training  ================================


@run.command(name="train-diaguq")
@click.pass_context
@_diaguq_pair_options
@click.option("--output-root", type=click.Path(file_okay=False),
              default=None,
              help="Override the per-pair output tree root. Defaults to the runtime test_output.")
@click.option("--epochs", type=int, default=20, show_default=True)
@click.option("--batch-size", type=int, default=64, show_default=True)
@click.option("--lr", type=float, default=1e-3, show_default=True)
@click.option("--fusion-dim", type=int, default=256, show_default=True)
@click.option("--layer-softmax-temperature", type=float, default=1.5, show_default=True)
@click.option("--layer-temperature", type=float, default=None,
              help="Alias for layer-softmax-temperature; saved in the training config when set.")
@click.option("--layer-dropout", type=float, default=0.05, show_default=True)
@click.option("--layer-residual-uniform-alpha", type=float, default=0.0, show_default=True,
              help="Mix layer attention with uniform weights to reduce last-layer collapse.")
@click.option("--layer-entropy-weight", type=float, default=0.0, show_default=True)
@click.option("--gate-logit-clip", type=float, default=10.0, show_default=True)
@click.option("--view-gate-hidden-dim", type=int, default=None)
@click.option("--view-fusion-mode", type=click.Choice(VIEW_FUSION_MODE_CHOICES), default="dimension_specific", show_default=True)
@click.option("--view-gate-scope", type=click.Choice(VIEW_GATE_SCOPE_CHOICES), default="shared", show_default=True)
@click.option("--diagnostic-factorization-mode", type=click.Choice(DIAGNOSTIC_FACTORIZATION_MODE_CHOICES), default="shared_plus_residual", show_default=True)
@click.option("--overall-aggregation-mode", type=click.Choice(OVERALL_AGGREGATION_MODE_CHOICES), default="hybrid", show_default=True)
@click.option("--dimension-corr-regularization-weight", type=float, default=0.01, show_default=True)
@click.option("--dimension-corr-margin", type=float, default=0.05, show_default=True)
@click.option("--residual-diversity-weight", type=float, default=0.005, show_default=True)
@click.option("--residual-diversity-margin", type=float, default=0.1, show_default=True)
@click.option("--view-temperature", type=float, default=2.0, show_default=True)
@click.option("--view-temperature-min", type=float, default=0.5, show_default=True)
@click.option("--view-temperature-max", type=float, default=10.0, show_default=True)
@click.option("--residual-uniform-alpha", type=float, default=0.05, show_default=True,
              help="Mix adaptive view weights with uniform weights during training/evaluation.")
@click.option("--view-norm-clip", type=float, default=10.0, show_default=True,
              help="Clamp per-view fused feature norms before the view gate; use negative to disable.")
@click.option("--view-entropy-weight", type=float, default=0.01, show_default=True)
@click.option("--view-entropy-warmup-epochs", type=int, default=1, show_default=True)
@click.option("--view-entropy-anneal-to", type=float, default=0.002, show_default=True)
@click.option("--view-dropout-prob", type=float, default=0.1, show_default=True)
@click.option("--knowledge-gap-loss-weight", type=float, default=1.0, show_default=True)
@click.option("--predictive-variability-loss-weight", type=float, default=1.0, show_default=True)
@click.option("--ambiguity-loss-weight", type=float, default=1.0, show_default=True)
@click.option("--proxy-target-loss-weight-multiplier", type=float, default=0.7, show_default=True)
@click.option("--val-fraction", type=float, default=0.1, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--device", type=str, default=None,
              help="Override torch device. Defaults to cuda when available.")
def train_diaguq_cmd(
    ctx,
    models: Tuple[str],
    ds: Tuple[str],
    dataset_alias: Tuple[str],
    model_alias: Tuple[str],
    scope: str,
    split: Optional[str],
    train_split: Optional[str],
    eval_split: Optional[str],
    limit: Optional[int],
    train_limit: Optional[int],
    eval_limit: Optional[int],
    single_split_policy: Optional[str],
    internal_train_ratio: float,
    internal_split_seed: int,
    allow_single_split_same_eval: bool,
    allow_train_eval_same_split: bool,
    dry_run: bool,
    force: bool,
    continue_on_error: bool,
    output_root: Optional[str],
    epochs: int, batch_size: int, lr: float, fusion_dim: int,
    layer_softmax_temperature: float, layer_temperature: Optional[float],
    layer_dropout: float, layer_residual_uniform_alpha: float,
    layer_entropy_weight: float, gate_logit_clip: float,
    view_gate_hidden_dim: Optional[int], view_fusion_mode: str,
    view_gate_scope: str, diagnostic_factorization_mode: str,
    overall_aggregation_mode: str, dimension_corr_regularization_weight: float,
    dimension_corr_margin: float, residual_diversity_weight: float,
    residual_diversity_margin: float, view_temperature: float, view_temperature_min: float,
    view_temperature_max: float, residual_uniform_alpha: float,
    view_norm_clip: float, view_entropy_weight: float,
    view_entropy_warmup_epochs: int, view_entropy_anneal_to: float,
    view_dropout_prob: float, knowledge_gap_loss_weight: float,
    predictive_variability_loss_weight: float, ambiguity_loss_weight: float,
    proxy_target_loss_weight_multiplier: float, val_fraction: float,
    seed: int, device: Optional[str],
):
    """Train the DiagUQ network for each (dataset, model) pair.

    Writes checkpoints + train log under
    ``<runtime_root>/test_output/<dataset>/<model>/diaguq/checkpoints/``.
    """
    if output_root is None:
        output_root = ctx.obj["output_root"]
    log_layout_once()
    plan = _resolve_diaguq_run_plan(
        command="train-diaguq",
        scope=scope,
        ds=ds,
        models=models,
        dataset_alias=dataset_alias,
        model_alias=model_alias,
        split=split,
        limit=limit,
        train_split=train_split,
        eval_split=eval_split,
        train_limit=train_limit,
        eval_limit=eval_limit,
        single_split_policy=single_split_policy,
        internal_train_ratio=internal_train_ratio,
        internal_split_seed=internal_split_seed,
        allow_single_split_same_eval=allow_single_split_same_eval,
    )
    _print_run_plan(plan)
    pairs = plan["pairs"]
    _assert_unique_stage_outputs("training", pairs, output_root)
    if dry_run:
        return
    import torch as _torch
    from pipeline.train_diaguq import DiagUQTrainConfig, train_diaguq

    resolved = device or ("cuda" if _torch.cuda.is_available() else "cpu")
    if force:
        _force_clear_stage_outputs("training", pairs, output_root)
    failed = []
    for dataset, model in pairs:
        cfg = DiagUQTrainConfig(
            dataset_name=dataset,
            model_name=model,
            output_root=output_root,
            num_epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            fusion_dim=fusion_dim,
            layer_softmax_temperature=layer_softmax_temperature,
            layer_temperature=layer_temperature,
            layer_dropout=layer_dropout,
            layer_residual_uniform_alpha=layer_residual_uniform_alpha,
            layer_entropy_weight=layer_entropy_weight,
            gate_logit_clip=gate_logit_clip,
            view_gate_hidden_dim=view_gate_hidden_dim,
            view_fusion_mode=view_fusion_mode,
            view_gate_scope=view_gate_scope,
            diagnostic_factorization_mode=diagnostic_factorization_mode,
            overall_aggregation_mode=overall_aggregation_mode,
            dimension_corr_regularization_weight=dimension_corr_regularization_weight,
            dimension_corr_margin=dimension_corr_margin,
            residual_diversity_weight=residual_diversity_weight,
            residual_diversity_margin=residual_diversity_margin,
            view_temperature=view_temperature,
            view_temperature_min=view_temperature_min,
            view_temperature_max=view_temperature_max,
            residual_uniform_alpha=residual_uniform_alpha,
            view_norm_clip=None if view_norm_clip < 0 else view_norm_clip,
            view_entropy_weight=view_entropy_weight,
            view_entropy_warmup_epochs=view_entropy_warmup_epochs,
            view_entropy_anneal_to=view_entropy_anneal_to,
            view_dropout_prob=view_dropout_prob,
            knowledge_gap_loss_weight=knowledge_gap_loss_weight,
            predictive_variability_loss_weight=predictive_variability_loss_weight,
            ambiguity_loss_weight=ambiguity_loss_weight,
            proxy_target_loss_weight_multiplier=proxy_target_loss_weight_multiplier,
            val_fraction=val_fraction,
            seed=seed,
            device=resolved,
        )
        try:
            summary = train_diaguq(cfg)
            logger.info(
                f"[train] {dataset} / {model} done -> {summary['checkpoint_dir']}"
            )
        except Exception as exc:  # noqa: BLE001
            failed.append({"dataset": dataset, "model": model, "error": repr(exc)})
            logger.warning(
                f"[train] {dataset} / {model} failed: {exc}"
            )
            if not continue_on_error:
                raise
    if failed and not continue_on_error:
        raise click.ClickException(f"train-diaguq failed for {len(failed)} pair(s).")


# =======================  DiagUQ: evaluation  ==============================


_DOWNSTREAM_REQUIRED_SUBDIRS: Tuple[str, ...] = (
    "hidden_bank",
    "dimension_targets",
)


def _preflight_downstream_roots(
    label: str,
    pairs: Sequence[Tuple[str, str]],
    output_root: Optional[str],
    artifact_root_name: Optional[str],
    allow_train_fallback: bool = False,
) -> None:
    from common.diaguq_existing_artifacts import (
        resolve_existing_diaguq_artifact_root,
    )

    usable = 0
    for dataset, model in pairs:
        resolved = resolve_existing_diaguq_artifact_root(
            dataset,
            model,
            output_root,
            artifact_root_name=artifact_root_name,
            allow_train_fallback=allow_train_fallback,
        )
        if not resolved.found:
            logger.warning(
                "[{}] skipped requested_dataset={} model={} reason={}",
                label, dataset, model, resolved.describe(),
            )
            continue
        missing = resolved.missing_subdirs(_DOWNSTREAM_REQUIRED_SUBDIRS)
        if missing:
            reason = "; ".join(
                f"{name} missing: {path}" for name, path in missing
            )
            logger.warning(
                "[{}] skipped requested_dataset={} model={} "
                "resolved_artifact_root={} reason={}",
                label, dataset, model, resolved.artifact_root, reason,
            )
            continue
        usable += 1
        logger.info(
            "[{}] requested_dataset={} model={} resolved_artifact_root={} "
            "resolved_split={} resolved_path={}",
            label,
            dataset,
            model,
            resolved.dataset_root_name,
            resolved.split_label,
            resolved.artifact_root,
        )
    logger.info(
        "[{}] preflight summary requested_pairs={} usable_pairs={} skipped_pairs={}",
        label, len(pairs), usable, len(pairs) - usable,
    )
    if usable == 0:
        raise click.ClickException(
            f"{label} found no usable DiagUQ artifact roots; see logs above."
        )


def _shared_eval_options(func):
    func = click.option(
        "--allow-train-fallback",
        is_flag=True,
        default=False,
        help="Allow evaluation/export to fall back to train artifacts when the requested eval split is missing.",
    )(func)
    func = click.option("--device", type=str, default=None,
                        help="Override torch device.")(func)
    func = click.option("--seed", type=int, default=42, show_default=True)(func)
    func = click.option("--val-fraction", type=float, default=0.1,
                        show_default=True)(func)
    func = click.option("--fusion-dim", type=int, default=256,
                        show_default=True)(func)
    func = click.option("--layer-temperature", type=float, default=None,
                        help="Alias for layer-softmax-temperature when evaluating configs without saved architecture metadata.")(func)
    func = click.option("--layer-residual-uniform-alpha", type=float, default=0.0,
                        show_default=True)(func)
    func = click.option("--view-fusion-mode", type=click.Choice(VIEW_FUSION_MODE_CHOICES),
                        default="dimension_specific", show_default=True)(func)
    func = click.option("--view-gate-scope", type=click.Choice(VIEW_GATE_SCOPE_CHOICES),
                        default="shared", show_default=True)(func)
    func = click.option("--diagnostic-factorization-mode", type=click.Choice(DIAGNOSTIC_FACTORIZATION_MODE_CHOICES),
                        default="shared_plus_residual", show_default=True)(func)
    func = click.option("--overall-aggregation-mode", type=click.Choice(OVERALL_AGGREGATION_MODE_CHOICES),
                        default="hybrid", show_default=True)(func)
    func = click.option("--dimension-corr-regularization-weight", type=float, default=0.01,
                        show_default=True)(func)
    func = click.option("--dimension-corr-margin", type=float, default=0.05,
                        show_default=True)(func)
    func = click.option("--residual-diversity-weight", type=float, default=0.005,
                        show_default=True)(func)
    func = click.option("--residual-diversity-margin", type=float, default=0.1,
                        show_default=True)(func)
    func = click.option("--proxy-target-loss-weight-multiplier", type=float, default=0.7,
                        show_default=True)(func)
    func = click.option("--view-temperature", type=float, default=2.0,
                        show_default=True)(func)
    func = click.option("--residual-uniform-alpha", type=float, default=0.05,
                        show_default=True)(func)
    func = click.option("--view-norm-clip", type=float, default=10.0,
                        show_default=True, help="Use negative to disable.")(func)
    func = click.option("--view-entropy-weight", type=float, default=0.01,
                        show_default=True)(func)
    func = click.option("--view-dropout-prob", type=float, default=0.0,
                        show_default=True)(func)
    func = click.option("--epochs", type=int, default=20, show_default=True)(func)
    func = click.option("--batch-size", type=int, default=64,
                        show_default=True)(func)
    func = click.option("--lr", type=float, default=1e-3, show_default=True)(func)
    func = click.option("--output-root", type=click.Path(file_okay=False),
                        default=None,
                        help="Override the per-pair output tree root. "
                             "Defaults to the runtime test_output.")(func)
    func = click.option(
        "--artifact-root-name",
        type=str,
        default=None,
        help=(
            "Optional existing dataset artifact root to read from, e.g. "
            "triviaqa__train. Defaults to automatic discovery."
        ),
    )(func)
    func = click.option(
        "--checkpoint-artifact-root-name",
        "--checkpoint-dataset",
        "checkpoint_artifact_root_name",
        type=str,
        default=None,
        help=(
            "Optional training/checkpoint artifact root to load from, e.g. "
            "triviaqa__train. Defaults to the resolved train split. "
            "--checkpoint-dataset is an alias for cross-dataset evaluation."
        ),
    )(func)
    func = click.option(
        "--calibrate-confidence/--no-calibrate-confidence",
        default=False,
        show_default=True,
        help="Fit a post-hoc Platt calibrator on the evaluation rows and save calibrated confidence separately.",
    )(func)
    func = _diaguq_pair_options(func)
    func = click.option(
        "--aggregate-csv", type=click.Path(dir_okay=False), default=None,
        help="Optional path to dump a single CSV combining every (dataset, model) row.",
    )(func)
    return func


def _build_eval_cfg(
    output_root: Optional[str], epochs: int, batch_size: int, lr: float,
    fusion_dim: int, layer_temperature: Optional[float],
    layer_residual_uniform_alpha: float, view_fusion_mode: str,
    view_gate_scope: str, diagnostic_factorization_mode: str,
    overall_aggregation_mode: str, dimension_corr_regularization_weight: float,
    dimension_corr_margin: float, residual_diversity_weight: float,
    residual_diversity_margin: float, proxy_target_loss_weight_multiplier: float,
    view_temperature: float, residual_uniform_alpha: float,
    view_norm_clip: float, view_entropy_weight: float, view_dropout_prob: float,
    val_fraction: float, seed: int, device: Optional[str],
):
    import torch as _torch
    from pipeline.train_diaguq import DiagUQTrainConfig
    if output_root is None:
        output_root = str(get_test_output_dir())
    resolved = device or ("cuda" if _torch.cuda.is_available() else "cpu")
    return DiagUQTrainConfig(
        dataset_name="__placeholder__",
        model_name="__placeholder__",
        output_root=output_root,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        fusion_dim=fusion_dim,
        layer_temperature=layer_temperature,
        layer_residual_uniform_alpha=layer_residual_uniform_alpha,
        view_fusion_mode=view_fusion_mode,
        view_gate_scope=view_gate_scope,
        diagnostic_factorization_mode=diagnostic_factorization_mode,
        overall_aggregation_mode=overall_aggregation_mode,
        dimension_corr_regularization_weight=dimension_corr_regularization_weight,
        dimension_corr_margin=dimension_corr_margin,
        residual_diversity_weight=residual_diversity_weight,
        residual_diversity_margin=residual_diversity_margin,
        proxy_target_loss_weight_multiplier=proxy_target_loss_weight_multiplier,
        view_temperature=view_temperature,
        residual_uniform_alpha=residual_uniform_alpha,
        view_norm_clip=None if view_norm_clip < 0 else view_norm_clip,
        view_entropy_weight=view_entropy_weight,
        view_dropout_prob=view_dropout_prob,
        val_fraction=val_fraction,
        seed=seed,
        device=resolved,
    )


def _run_eval(
    mode: str,
    models,
    ds,
    dataset_alias,
    model_alias,
    scope,
    split,
    train_split,
    eval_split,
    limit,
    train_limit,
    eval_limit,
    single_split_policy,
    internal_train_ratio,
    internal_split_seed,
    allow_single_split_same_eval,
    allow_train_eval_same_split,
    allow_train_fallback,
    dry_run,
    force,
    continue_on_error,
    aggregate_csv,
    output_root,
    artifact_root_name,
    checkpoint_artifact_root_name,
    calibrate_confidence,
    epochs,
    batch_size,
    lr,
    fusion_dim,
    layer_temperature,
    layer_residual_uniform_alpha,
    view_fusion_mode,
    view_gate_scope,
    diagnostic_factorization_mode,
    overall_aggregation_mode,
    dimension_corr_regularization_weight,
    dimension_corr_margin,
    residual_diversity_weight,
    residual_diversity_margin,
    proxy_target_loss_weight_multiplier,
    view_temperature,
    residual_uniform_alpha,
    view_norm_clip,
    view_entropy_weight,
    view_dropout_prob,
    val_fraction,
    seed,
    device,
):
    if output_root is None:
        output_root = str(get_test_output_dir())
    plan = _resolve_diaguq_run_plan(
        command=f"eval-{mode}",
        scope=scope,
        ds=ds,
        models=models,
        dataset_alias=dataset_alias,
        model_alias=model_alias,
        split=split,
        limit=limit,
        train_split=train_split,
        eval_split=eval_split,
        train_limit=train_limit,
        eval_limit=eval_limit,
        single_split_policy=single_split_policy,
        internal_train_ratio=internal_train_ratio,
        internal_split_seed=internal_split_seed,
        allow_single_split_same_eval=allow_single_split_same_eval,
    )
    _print_run_plan(plan)
    pairs = plan["pairs"]
    _assert_unique_stage_outputs("evaluation", pairs, output_root)
    if dry_run:
        return
    if force:
        _force_clear_stage_outputs("evaluation", pairs, output_root)
    _preflight_downstream_roots(
        f"eval-{mode}", pairs, output_root, artifact_root_name, allow_train_fallback
    )
    from pipeline.evaluate_diaguq import run_eval_pairs as _run_eval_pairs
    cfg = _build_eval_cfg(
        output_root, epochs, batch_size, lr, fusion_dim,
        layer_temperature, layer_residual_uniform_alpha, view_fusion_mode,
        view_gate_scope, diagnostic_factorization_mode,
        overall_aggregation_mode, dimension_corr_regularization_weight,
        dimension_corr_margin, residual_diversity_weight,
        residual_diversity_margin, proxy_target_loss_weight_multiplier,
        view_temperature, residual_uniform_alpha,
        view_norm_clip, view_entropy_weight, view_dropout_prob,
        val_fraction, seed, device,
    )
    res = _run_eval_pairs(
        mode, pairs, cfg,
        aggregate_csv=Path(aggregate_csv) if aggregate_csv else None,
        artifact_root_name=artifact_root_name,
        checkpoint_artifact_root_name=checkpoint_artifact_root_name,
        train_pairs=plan.get("train_pairs") or [],
        allow_train_eval_same_split=allow_train_eval_same_split or allow_single_split_same_eval,
        allow_train_fallback=allow_train_fallback,
        calibrate_confidence=calibrate_confidence,
    )
    for k, info in res["resolved_pairs"].items():
        logger.info(
            "[eval-{}] resolved {} requested_dataset={} model={} "
            "resolved_artifact_root={} resolved_path={}",
            mode, k, info["requested_dataset"], info["model"],
            info["resolved_dataset_root"], info["artifact_root"],
        )
    for item in res["skipped"]:
        logger.warning(
            "[eval-{}] skipped {}/{}: {}",
            mode, item["dataset"], item["model"], item["reason"],
        )
    for item in res["failed"]:
        logger.warning(
            "[eval-{}] failed {}/{}: {}",
            mode, item["dataset"], item["model"], item["error"],
        )
    for k, v in res["per_pair_csv"].items():
        logger.info(f"[eval-{mode}] {k} -> {v}")
    if res["aggregate_csv"]:
        logger.info(f"[eval-{mode}] aggregate -> {res['aggregate_csv']}")
    summary = res["summary"]
    logger.info(
        "[eval-{}] summary requested_pairs={} resolved_pairs={} "
        "successful_pairs={} skipped_pairs={} failed_pairs={}",
        mode,
        summary["requested_pairs"],
        summary["resolved_pairs"],
        summary["successful_pairs"],
        summary["skipped_pairs"],
        summary["failed_pairs"],
    )
    if summary["successful_pairs"] == 0:
        raise click.ClickException(
            f"eval-{mode} produced no successful pairs; see skipped/failed logs above."
        )
    if res["failed"] and not continue_on_error:
        raise click.ClickException(f"eval-{mode} failed for {len(res['failed'])} pair(s).")


@run.command(name="evaluate-diaguq")
@click.pass_context
@_shared_eval_options
def evaluate_diaguq_cmd(ctx, models, ds, aggregate_csv, output_root, epochs,
                        batch_size, lr, fusion_dim, layer_temperature,
                        layer_residual_uniform_alpha, view_fusion_mode,
                        view_gate_scope, diagnostic_factorization_mode,
                        overall_aggregation_mode, dimension_corr_regularization_weight,
                        dimension_corr_margin, residual_diversity_weight,
                        residual_diversity_margin, proxy_target_loss_weight_multiplier,
                        view_temperature, residual_uniform_alpha,
                        view_norm_clip, view_entropy_weight, view_dropout_prob,
                        val_fraction, seed, device,
                        artifact_root_name, checkpoint_artifact_root_name,
                        calibrate_confidence, allow_train_fallback, dataset_alias, model_alias, scope,
                        split, train_split, eval_split, limit, train_limit,
                        eval_limit, single_split_policy, internal_train_ratio,
                        internal_split_seed, allow_single_split_same_eval,
                        allow_train_eval_same_split, dry_run,
                        force, continue_on_error):
    """Main DiagUQ evaluation: DiagUQ vs. entropy / max-prob / ask4conf /
    semantic-entropy / baseline supervised estimator."""
    _run_eval("main", models, ds, dataset_alias, model_alias, scope, split,
              train_split, eval_split, limit, train_limit, eval_limit,
              single_split_policy, internal_train_ratio, internal_split_seed,
              allow_single_split_same_eval, allow_train_eval_same_split,
              allow_train_fallback, dry_run, force, continue_on_error,
              aggregate_csv, output_root, artifact_root_name,
              checkpoint_artifact_root_name, calibrate_confidence, epochs,
              batch_size, lr, fusion_dim, layer_temperature,
              layer_residual_uniform_alpha, view_fusion_mode, view_gate_scope,
              diagnostic_factorization_mode, overall_aggregation_mode,
              dimension_corr_regularization_weight, dimension_corr_margin,
              residual_diversity_weight, residual_diversity_margin,
              proxy_target_loss_weight_multiplier, view_temperature,
              residual_uniform_alpha, view_norm_clip,
              view_entropy_weight, view_dropout_prob, val_fraction, seed, device)


@run.command(name="ablate-layers")
@click.pass_context
@click.option(
    "--feature-mode",
    "feature_modes",
    type=click.Choice((
        "query_answer_relation_concat",
        "answer_only",
        "query_only",
        "relation_only",
    )),
    multiple=True,
    default=("query_answer_relation_concat",),
    show_default=True,
    help="Hidden-state feature view to train on. Repeat to run multiple modes.",
)
@click.option(
    "--baseline-model",
    type=click.Choice(("mlp", "rf")),
    default="mlp",
    show_default=True,
    help="Supervised estimator family for fixed-layer baselines.",
)
@_shared_eval_options
def ablate_layers_cmd(ctx, models, ds, aggregate_csv, output_root, epochs,
                      batch_size, lr, fusion_dim, layer_temperature,
                      layer_residual_uniform_alpha, view_fusion_mode,
                      view_gate_scope, diagnostic_factorization_mode,
                      overall_aggregation_mode, dimension_corr_regularization_weight,
                      dimension_corr_margin, residual_diversity_weight,
                      residual_diversity_margin, proxy_target_loss_weight_multiplier,
                      view_temperature, residual_uniform_alpha,
                      view_norm_clip, view_entropy_weight, view_dropout_prob,
                      val_fraction, seed, device,
                      artifact_root_name, checkpoint_artifact_root_name,
                      calibrate_confidence, allow_train_fallback, dataset_alias, model_alias, scope,
                      split, train_split, eval_split, limit, train_limit,
                      eval_limit, single_split_policy, internal_train_ratio,
                      internal_split_seed, allow_single_split_same_eval,
                      allow_train_eval_same_split, dry_run,
                      force, continue_on_error, feature_modes, baseline_model):
    """Train fixed-layer hidden-state baselines from existing DiagUQ artifacts."""
    if output_root is None:
        output_root = str(get_test_output_dir())
    canonical_models = tuple(resolve_model_name(m) for m in (models or ())) or AVAILABLE_DIAGUQ_MODELS
    canonical_ds = tuple(ds or ())
    plan = _resolve_diaguq_run_plan(
        command="ablate-layers",
        models=canonical_models,
        ds=canonical_ds,
        dataset_alias=dataset_alias,
        model_alias=model_alias,
        scope=scope,
        split=split,
        train_split=train_split,
        eval_split=eval_split,
        limit=limit,
        train_limit=train_limit,
        eval_limit=eval_limit,
        single_split_policy=single_split_policy,
        internal_train_ratio=internal_train_ratio,
        internal_split_seed=internal_split_seed,
        allow_single_split_same_eval=allow_single_split_same_eval,
    )
    _print_run_plan(plan)
    _preflight_layer_baseline_artifacts(
        plan,
        output_root,
        checkpoint_dataset=checkpoint_artifact_root_name,
        strict=not dry_run,
    )
    if dry_run:
        click.echo(
            "[ablate-layers] dry-run complete: no hidden states, targets, checkpoints, "
            "or DiagUQ metrics were modified."
        )
        return
    from baseline.layer_baselines import run_layer_baseline_pairs

    cfg = _build_eval_cfg(
        output_root, epochs, batch_size, lr, fusion_dim, layer_temperature,
        layer_residual_uniform_alpha, view_fusion_mode, view_gate_scope,
        diagnostic_factorization_mode, overall_aggregation_mode,
        dimension_corr_regularization_weight, dimension_corr_margin,
        residual_diversity_weight, residual_diversity_margin,
        proxy_target_loss_weight_multiplier, view_temperature,
        residual_uniform_alpha, view_norm_clip, view_entropy_weight,
        view_dropout_prob, val_fraction, seed, device,
    )
    results = run_layer_baseline_pairs(
        plan.get("eval_pairs") or plan.get("pairs") or [],
        plan.get("train_pairs") or [],
        cfg,
        feature_modes=feature_modes,
        baseline_model=baseline_model,
        checkpoint_dataset=checkpoint_artifact_root_name,
        force=force,
    )
    if aggregate_csv:
        rows = []
        for result in results:
            rows.append(result)
        with Path(aggregate_csv).open("w", newline="", encoding="utf-8") as fw:
            fieldnames = list(rows[0].keys()) if rows else ["dataset", "model", "rows", "metrics_csv", "comparison_csv"]
            writer = csv.DictWriter(fw, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    for result in results:
        click.echo(
            "[ablate-layers] dataset={} model={} rows={} metrics={} comparison={}".format(
                result["dataset"],
                result["model"],
                result["rows"],
                result["metrics_csv"],
                result["comparison_csv"],
            )
        )


@run.command(name="bootstrap-layer-comparison")
@click.pass_context
@_diaguq_pair_options
@click.option("--n-bootstrap", type=int, default=1000, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option("--ci-level", type=float, default=0.95, show_default=True)
def bootstrap_layer_comparison_cmd(
    ctx,
    models: Tuple[str],
    ds: Tuple[str],
    dataset_alias: Tuple[str],
    model_alias: Tuple[str],
    scope: str,
    split: Optional[str],
    train_split: Optional[str],
    eval_split: Optional[str],
    limit: Optional[int],
    train_limit: Optional[int],
    eval_limit: Optional[int],
    single_split_policy: Optional[str],
    internal_train_ratio: float,
    internal_split_seed: int,
    allow_single_split_same_eval: bool,
    allow_train_eval_same_split: bool,
    dry_run: bool,
    force: bool,
    continue_on_error: bool,
    n_bootstrap: int,
    seed: int,
    ci_level: float,
):
    """Compute paired bootstrap CIs for DiagUQ vs fixed-layer baselines."""
    plan = _resolve_diaguq_run_plan(
        command="bootstrap-layer-comparison",
        models=models,
        ds=ds,
        dataset_alias=dataset_alias,
        model_alias=model_alias,
        scope=scope,
        split=split,
        train_split=train_split,
        eval_split=eval_split,
        limit=limit,
        train_limit=train_limit,
        eval_limit=eval_limit,
        single_split_policy=single_split_policy,
        internal_train_ratio=internal_train_ratio,
        internal_split_seed=internal_split_seed,
        allow_single_split_same_eval=allow_single_split_same_eval,
    )
    _print_run_plan(plan)
    pairs = plan.get("eval_pairs") or plan.get("pairs") or []
    if dry_run:
        click.echo("[bootstrap-layer-comparison] dry-run complete: no files written.")
        return
    from analysis.layer_baseline_analysis import LayerAnalysisError, run_bootstrap_layer_comparison

    try:
        rows = run_bootstrap_layer_comparison(
            pairs,
            output_root=ctx.obj["output_root"],
            n_bootstrap=n_bootstrap,
            seed=seed,
            ci_level=ci_level,
            force=force,
        )
    except LayerAnalysisError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"[bootstrap-layer-comparison] wrote {len(rows)} metric rows")


@run.command(name="plot-layer-heatmaps")
@click.pass_context
@_diaguq_pair_options
@click.option("--metric", type=click.Choice(("AUROC", "AUPRC", "AUARC", "ECE", "Brier")), default="AUROC", show_default=True)
@click.option("--output-format", type=click.Choice(("png", "pdf", "both", "none")), default="both", show_default=True)
@click.option("--include-dimension-heatmaps", is_flag=True, default=False)
def plot_layer_heatmaps_cmd(
    ctx,
    models: Tuple[str],
    ds: Tuple[str],
    dataset_alias: Tuple[str],
    model_alias: Tuple[str],
    scope: str,
    split: Optional[str],
    train_split: Optional[str],
    eval_split: Optional[str],
    limit: Optional[int],
    train_limit: Optional[int],
    eval_limit: Optional[int],
    single_split_policy: Optional[str],
    internal_train_ratio: float,
    internal_split_seed: int,
    allow_single_split_same_eval: bool,
    allow_train_eval_same_split: bool,
    dry_run: bool,
    force: bool,
    continue_on_error: bool,
    metric: str,
    output_format: str,
    include_dimension_heatmaps: bool,
):
    """Create dataset-layer and optional layer-dimension heatmaps."""
    plan = _resolve_diaguq_run_plan(
        command="plot-layer-heatmaps",
        models=models,
        ds=ds,
        dataset_alias=dataset_alias,
        model_alias=model_alias,
        scope=scope,
        split=split,
        train_split=train_split,
        eval_split=eval_split,
        limit=limit,
        train_limit=train_limit,
        eval_limit=eval_limit,
        single_split_policy=single_split_policy,
        internal_train_ratio=internal_train_ratio,
        internal_split_seed=internal_split_seed,
        allow_single_split_same_eval=allow_single_split_same_eval,
    )
    _print_run_plan(plan)
    pairs = plan.get("eval_pairs") or plan.get("pairs") or []
    from analysis.layer_baseline_analysis import LayerAnalysisError, aggregate_existing_layer_summaries, run_layer_heatmap_analysis

    try:
        aggregate_existing_layer_summaries(pairs, output_root=ctx.obj["output_root"])
        result = run_layer_heatmap_analysis(
            pairs,
            output_root=ctx.obj["output_root"],
            metric=metric,
            output_format=output_format,
            include_dimension_heatmaps=include_dimension_heatmaps,
            dry_run=dry_run,
        )
    except LayerAnalysisError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"[plot-layer-heatmaps] {json.dumps(result, ensure_ascii=False)}")


@run.command(name="ablate-dimensions")
@click.pass_context
@_shared_eval_options
def ablate_dimensions_cmd(ctx, models, ds, aggregate_csv, output_root, epochs,
                          batch_size, lr, fusion_dim, layer_temperature,
                          layer_residual_uniform_alpha, view_fusion_mode,
                          view_gate_scope, diagnostic_factorization_mode,
                          overall_aggregation_mode, dimension_corr_regularization_weight,
                          dimension_corr_margin, residual_diversity_weight,
                          residual_diversity_margin, proxy_target_loss_weight_multiplier,
                          view_temperature, residual_uniform_alpha,
                          view_norm_clip, view_entropy_weight, view_dropout_prob,
                          val_fraction, seed, device,
                          artifact_root_name, checkpoint_artifact_root_name,
                          calibrate_confidence, allow_train_fallback, dataset_alias, model_alias,
                          scope, split, train_split, eval_split, limit,
                          train_limit, eval_limit, single_split_policy,
                          internal_train_ratio, internal_split_seed,
                          allow_single_split_same_eval, allow_train_eval_same_split,
                          dry_run, force, continue_on_error):
    """Diagnostic-dimension ablation: overall-only vs.
    multidim-without-aggregator vs. multidim-with-aggregator."""
    _run_eval("dimension_ablation", models, ds, dataset_alias, model_alias,
              scope, split, train_split, eval_split, limit, train_limit,
              eval_limit, single_split_policy, internal_train_ratio,
              internal_split_seed, allow_single_split_same_eval,
              allow_train_eval_same_split, allow_train_fallback, dry_run, force,
              continue_on_error, aggregate_csv, output_root, artifact_root_name,
              checkpoint_artifact_root_name, calibrate_confidence, epochs,
              batch_size, lr, fusion_dim, layer_temperature,
              layer_residual_uniform_alpha, view_fusion_mode, view_gate_scope,
              diagnostic_factorization_mode, overall_aggregation_mode,
              dimension_corr_regularization_weight, dimension_corr_margin,
              residual_diversity_weight, residual_diversity_margin,
              proxy_target_loss_weight_multiplier, view_temperature,
              residual_uniform_alpha, view_norm_clip,
              view_entropy_weight, view_dropout_prob, val_fraction, seed, device)


@run.command(name="ablate-views")
@click.pass_context
@_shared_eval_options
def ablate_views_cmd(ctx, models, ds, aggregate_csv, output_root, epochs,
                     batch_size, lr, fusion_dim, layer_temperature,
                     layer_residual_uniform_alpha, view_fusion_mode,
                     view_gate_scope, diagnostic_factorization_mode,
                     overall_aggregation_mode, dimension_corr_regularization_weight,
                     dimension_corr_margin, residual_diversity_weight,
                     residual_diversity_margin, proxy_target_loss_weight_multiplier,
                     view_temperature, residual_uniform_alpha,
                     view_norm_clip, view_entropy_weight, view_dropout_prob,
                     val_fraction, seed, device,
                     artifact_root_name, checkpoint_artifact_root_name,
                     calibrate_confidence, allow_train_fallback, dataset_alias, model_alias,
                     scope, split, train_split, eval_split, limit,
                     train_limit, eval_limit, single_split_policy,
                     internal_train_ratio, internal_split_seed,
                     allow_single_split_same_eval, allow_train_eval_same_split,
                     dry_run, force, continue_on_error):
    """View-fusion ablation: answer-only, query-only, relation-only,
    uniform, static-learned, sample-adaptive, and dimension-specific gates."""
    _run_eval("view_ablation", models, ds, dataset_alias, model_alias,
              scope, split, train_split, eval_split, limit, train_limit,
              eval_limit, single_split_policy, internal_train_ratio,
              internal_split_seed, allow_single_split_same_eval,
              allow_train_eval_same_split, allow_train_fallback, dry_run, force,
              continue_on_error, aggregate_csv, output_root, artifact_root_name,
              checkpoint_artifact_root_name, calibrate_confidence, epochs,
              batch_size, lr, fusion_dim, layer_temperature,
              layer_residual_uniform_alpha, view_fusion_mode, view_gate_scope,
              diagnostic_factorization_mode, overall_aggregation_mode,
              dimension_corr_regularization_weight, dimension_corr_margin,
              residual_diversity_weight, residual_diversity_margin,
              proxy_target_loss_weight_multiplier, view_temperature,
              residual_uniform_alpha, view_norm_clip,
              view_entropy_weight, view_dropout_prob, val_fraction, seed, device)


@run.command(name="ablate-diagnostic-heads")
@click.pass_context
@_shared_eval_options
def ablate_diagnostic_heads_cmd(ctx, models, ds, aggregate_csv, output_root, epochs,
                                batch_size, lr, fusion_dim, layer_temperature,
                                layer_residual_uniform_alpha, view_fusion_mode,
                                view_gate_scope, diagnostic_factorization_mode,
                                overall_aggregation_mode, dimension_corr_regularization_weight,
                                dimension_corr_margin, residual_diversity_weight,
                                residual_diversity_margin, proxy_target_loss_weight_multiplier,
                                view_temperature, residual_uniform_alpha,
                                view_norm_clip, view_entropy_weight, view_dropout_prob,
                                val_fraction, seed, device,
                                artifact_root_name, checkpoint_artifact_root_name,
                                calibrate_confidence, allow_train_fallback, dataset_alias, model_alias,
                                scope, split, train_split, eval_split, limit,
                                train_limit, eval_limit, single_split_policy,
                                internal_train_ratio, internal_split_seed,
                                allow_single_split_same_eval, allow_train_eval_same_split,
                                dry_run, force, continue_on_error):
    """Diagnostic-head ablation for shared-only, independent, residual, and regularized variants."""
    _run_eval("diagnostic_ablation", models, ds, dataset_alias, model_alias,
              scope, split, train_split, eval_split, limit, train_limit,
              eval_limit, single_split_policy, internal_train_ratio,
              internal_split_seed, allow_single_split_same_eval,
              allow_train_eval_same_split, allow_train_fallback, dry_run, force,
              continue_on_error, aggregate_csv, output_root, artifact_root_name,
              checkpoint_artifact_root_name, calibrate_confidence, epochs,
              batch_size, lr, fusion_dim, layer_temperature,
              layer_residual_uniform_alpha, view_fusion_mode, view_gate_scope,
              diagnostic_factorization_mode, overall_aggregation_mode,
              dimension_corr_regularization_weight, dimension_corr_margin,
              residual_diversity_weight, residual_diversity_margin,
              proxy_target_loss_weight_multiplier, view_temperature,
              residual_uniform_alpha, view_norm_clip,
              view_entropy_weight, view_dropout_prob, val_fraction, seed, device)


@run.command(name="export-analysis")
@click.pass_context
@_shared_eval_options
@click.option(
    "--run-inference/--no-run-inference",
    default=False,
    show_default=True,
    help="Run evaluate-diaguq inference first if eval/predictions.csv is missing.",
)
def export_analysis_cmd(ctx, models, ds, aggregate_csv, output_root, epochs,
                        batch_size, lr, fusion_dim, layer_temperature,
                        layer_residual_uniform_alpha, view_fusion_mode,
                        view_gate_scope, diagnostic_factorization_mode,
                        overall_aggregation_mode, dimension_corr_regularization_weight,
                        dimension_corr_margin, residual_diversity_weight,
                        residual_diversity_margin, proxy_target_loss_weight_multiplier,
                        view_temperature, residual_uniform_alpha,
                        view_norm_clip, view_entropy_weight, view_dropout_prob,
                        val_fraction, seed, device,
                        artifact_root_name, checkpoint_artifact_root_name,
                        calibrate_confidence, allow_train_fallback, dataset_alias, model_alias, scope,
                        split, train_split, eval_split, limit, train_limit,
                        eval_limit, single_split_policy, internal_train_ratio,
                        internal_split_seed, allow_single_split_same_eval,
                        allow_train_eval_same_split, dry_run, force, continue_on_error,
                        run_inference):
    """Per-sample DiagUQ export (overall / per-dim / per-layer weights /
    baselines) for paper figures and case studies. Outputs land under
    the resolved existing artifact root's ``diaguq/analysis/`` subtree."""
    if output_root is None:
        output_root = str(get_test_output_dir())
    plan = _resolve_diaguq_run_plan(
        command="export-analysis",
        scope=scope,
        ds=ds,
        models=models,
        dataset_alias=dataset_alias,
        model_alias=model_alias,
        split=split,
        limit=limit,
        train_split=train_split,
        eval_split=eval_split,
        train_limit=train_limit,
        eval_limit=eval_limit,
        single_split_policy=single_split_policy,
        internal_train_ratio=internal_train_ratio,
        internal_split_seed=internal_split_seed,
        allow_single_split_same_eval=allow_single_split_same_eval,
    )
    _print_run_plan(plan)
    pairs = plan["pairs"]
    _assert_unique_stage_outputs("export_analysis", pairs, output_root)
    if dry_run:
        return
    if force:
        _force_clear_stage_outputs("export-analysis", pairs, output_root)
    _preflight_downstream_roots(
        "export-analysis", pairs, output_root, artifact_root_name, allow_train_fallback
    )
    from analysis.export_diaguq_outputs import (
        export_diaguq_outputs_for_pairs as _export_diaguq_outputs_for_pairs,
    )
    cfg = _build_eval_cfg(
        output_root, epochs, batch_size, lr, fusion_dim,
        layer_temperature, layer_residual_uniform_alpha, view_fusion_mode,
        view_gate_scope, diagnostic_factorization_mode,
        overall_aggregation_mode, dimension_corr_regularization_weight,
        dimension_corr_margin, residual_diversity_weight,
        residual_diversity_margin, proxy_target_loss_weight_multiplier,
        view_temperature, residual_uniform_alpha,
        view_norm_clip, view_entropy_weight, view_dropout_prob,
        val_fraction, seed, device,
    )
    results = _export_diaguq_outputs_for_pairs(
        pairs,
        cfg,
        artifact_root_name=artifact_root_name,
        checkpoint_artifact_root_name=checkpoint_artifact_root_name,
        train_pairs=plan.get("train_pairs") or [],
        allow_train_eval_same_split=allow_train_eval_same_split or allow_single_split_same_eval,
        allow_train_fallback=allow_train_fallback,
        calibrate_confidence=calibrate_confidence,
        skip_on_error=continue_on_error,
        run_inference=run_inference,
    )
    requested_pairs = len(pairs)
    successful_pairs = 0
    skipped_pairs = 0
    failed_pairs = 0
    for r in results:
        if r.get("skipped"):
            skipped_pairs += 1
            logger.warning(
                f"[export-analysis] {r['dataset']} / {r['model']} skipped: {r['reason']}"
            )
        elif "error" in r:
            failed_pairs += 1
            logger.warning(
                f"[export-analysis] {r['dataset']} / {r['model']} failed: {r['error']}"
            )
        else:
            successful_pairs += 1
            logger.info(
                "[export-analysis] requested_dataset={} model={} "
                "resolved_artifact_root={} resolved_path={}",
                r.get("requested_dataset", r.get("dataset")),
                r["model"],
                r.get("resolved_dataset_root"),
                r.get("artifact_root"),
            )
            logger.info(
                f"[export-analysis] {r['dataset']} / {r['model']} -> {r['csv_path']}"
            )
    logger.info(
        "[export-analysis] summary requested_pairs={} successful_pairs={} "
        "skipped_pairs={} failed_pairs={}",
        requested_pairs, successful_pairs, skipped_pairs, failed_pairs,
    )
    if successful_pairs == 0:
        raise click.ClickException(
            "export-analysis produced no successful pairs; see skipped/failed logs above."
        )
    if failed_pairs and not continue_on_error:
        raise click.ClickException(f"export-analysis failed for {failed_pairs} pair(s).")


# =======================  Baseline commands  ===============================


def _baseline_pair_options(func):
    func = click.option(
        "--ds", "-d", multiple=True,
        type=click.Choice(AVAILABLE_BASELINE_DATASETS),
        default=("coqa", "triviaqa"), show_default=True,
    )(func)
    func = click.option(
        "--models", "-m", multiple=True,
        type=MODEL_CHOICE_BASELINE,
        default=("gemma_7b", "llama_2_7b"), show_default=True,
    )(func)
    return func


@run.command(name="baseline-setup-data")
@click.pass_context
def baseline_setup_data(ctx: click.Context):
    """Download baseline datasets (coqa, triviaqa, mmlu, wmt)."""
    from data.download_datasets import prepare_legacy_datasets
    ctx.obj["data_dir"].mkdir(parents=True, exist_ok=True)
    prepare_legacy_datasets(ctx.obj["data_dir"])


@run.command(name="baseline-setup-models")
@click.pass_context
def baseline_setup_models(ctx: click.Context):
    """Download baseline models (gemma_*, llama_2_*, llama_3_*) + DeBERTa."""
    from data.download_models import download_legacy_models, download_deberta
    ctx.obj["model_dir"].mkdir(parents=True, exist_ok=True)
    download_legacy_models(ctx.obj["model_dir"])
    download_deberta(ctx.obj["model_dir"])


@run.command(name="baseline-build-features")
@click.pass_context
@_baseline_pair_options
def baseline_build_features(ctx, models: Tuple[str], ds: Tuple[str]):
    """Extract two-layer features + greedy answers + RoUGE/BLEU + ask4conf
    consumed by the baseline random-forest estimator."""
    from features.response_pipeline import (
        generate_X, generate_answer_X_mmlu, generate_query_X_mmlu,
    )
    for model in models:
        for dataset in ds:
            logger.info(f"[baseline-features] {dataset} / {model}")
            try:
                if dataset != "mmlu":
                    from features.response_pipeline import generate_answer_most
                    generate_answer_most(model, dataset + "__train")
                    if dataset == "wmt":
                        generate_answer_most(model, dataset + "__test")
                if dataset == "mmlu":
                    generate_query_X_mmlu(model, "validation")
                    generate_query_X_mmlu(model, "test")
                    generate_answer_X_mmlu(model, "validation")
                    generate_answer_X_mmlu(model, "test")
                else:
                    generate_X(model, dataset + "__train", model)
                    if dataset == "wmt":
                        generate_X(model, dataset + "__test", model)
                if dataset == "wmt":
                    from features.response_pipeline import generate_y_most_WMT
                    generate_y_most_WMT(model, dataset + "__train")
                    generate_y_most_WMT(model, dataset + "__test")
                elif dataset != "mmlu":
                    from features.response_pipeline import generate_y_most_QA
                    generate_y_most_QA(model, dataset + "__train")
                from features.response_pipeline import (
                    generate_ask4conf, generate_answers,
                    generate_uncertainty_score,
                )
                generate_ask4conf(model, dataset)
                if dataset != "mmlu":
                    test_split = "__test" if dataset == "wmt" else "__train"
                    generate_answers(model, dataset + test_split)
                    generate_uncertainty_score(model, dataset + test_split)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"[baseline-features] failed for {model} / {dataset}: {exc}"
                )


@run.command(name="baseline-train")
@click.pass_context
@_baseline_pair_options
def baseline_train(ctx, models: Tuple[str], ds: Tuple[str]):
    """Train the baseline random-forest uncertainty estimator."""
    from baseline.baseline_rf_estimator import (
        train_baseline_estimator, train_baseline_estimator_mmlu,
    )
    for model in models:
        for dataset in ds:
            logger.info(f"[baseline-train] {dataset} / {model}")
            try:
                if dataset == "mmlu":
                    train_baseline_estimator_mmlu(model)
                else:
                    train_baseline_estimator(model, dataset)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"[baseline-train] failed for {model} / {dataset}: {exc}"
                )


@run.command(name="baseline-evaluate")
@click.pass_context
@_baseline_pair_options
def baseline_evaluate(ctx, models: Tuple[str], ds: Tuple[str]):
    """Evaluate the baseline random-forest uncertainty estimator."""
    from baseline.baseline_rf_estimator import (
        evaluate_baseline_estimator, evaluate_baseline_estimator_mmlu,
    )
    for model in models:
        for dataset in ds:
            logger.info(f"[baseline-evaluate] {dataset} / {model}")
            try:
                if dataset == "mmlu":
                    evaluate_baseline_estimator_mmlu(model)
                else:
                    evaluate_baseline_estimator(model, dataset)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"[baseline-evaluate] failed for {model} / {dataset}: {exc}"
                )


@run.command(name="baseline-transfer")
@click.pass_context
@_baseline_pair_options
def baseline_transfer(ctx, models: Tuple[str], ds: Tuple[str]):
    """Cross-model transfer evaluation for the baseline estimator."""
    from baseline.baseline_rf_estimator import train_baseline_estimator_mmlu
    from baseline.transfer_eval import (
        test_transferability, test_transferability_mmlu,
    )
    for model in models:
        for dataset in ds:
            logger.info(f"[baseline-transfer] {dataset} / {model}")
            try:
                if dataset == "mmlu":
                    train_baseline_estimator_mmlu(model, "mmlu", mmlu_tasks="Group1")
                    train_baseline_estimator_mmlu(model, "mmlu", mmlu_tasks="Group2")
                    test_transferability_mmlu(model, "Group1")
                    test_transferability_mmlu(model, "Group2")
                else:
                    test_transferability(model, dataset)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"[baseline-transfer] failed for {model} / {dataset}: {exc}"
                )

if __name__ == "__main__":
    run()
