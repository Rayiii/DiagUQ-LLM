import csv
import json
import os
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from time import time
from typing import *

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers
from joblib import Parallel, delayed
from loguru import logger
from nltk.translate.bleu_score import sentence_bleu
from torch.nn import functional as F
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

from data.download_datasets import MMLU_TASKS
from data.formatters import (
    coqa_formatter_hf,
    load_formatted_dataset,
    mmlu_formatter,
    triviaqa_formatter,
    validate_response_cache_dataset_request,
    wmt_formatter,
)
from data.mmlu_loader import parse_mmlu_option_answer
from features.dataset_variant_loader import (
    load_dataset_for_variant,
    resolve_supported_dataset_variant,
)
from features.hidden_state_ops import (
    ENTROPY_STAT_NAMES,
    PROB_STAT_NAMES,
    get_average_hidden_states,
    get_entropy_statistics,
    get_last_token_hidden_states,
    get_prob_statistics,
)
from common.load_models import (
    load_llama2,
    load_model_by_name,
)
from features.generators import MMLUGenerator, _encode_single_letter_token
from registry.model_registry import (
    get_layer_list_and_dim,
    get_model_paths,
)
from common.runtime_paths import get_models_dir, get_test_output_dir
from common.artifact_locator import locate_response_cache_artifacts
from common.artifact_paths import (
    ask4conf_jsonl_path,
    ask4conf_metadata_path,
    ask4conf_success_marker,
    extend_path,
    extend_samples_path,
    mextend_bleu_path,
    mextend_path,
    mextend_rouge_path,
    mextend_samples_path,
    response_answer_audit_csv_path,
    response_answer_audit_json_path,
    semantic_entropy_path,
    split_artifact_dir,
    split_dataset_and_raw,
)
from common.response_cache_limits import resolve_response_cache_limit
from common.qa_answer_scoring import (
    DEFAULT_QA_F1_THRESHOLD,
    score_open_domain_qa,
    write_answer_audit,
)


def _load_bool_sidecar(path: str, n: int) -> torch.Tensor:
    if not os.path.exists(path):
        return torch.zeros(n, dtype=torch.bool)
    loaded = torch.load(path, map_location="cpu").bool().reshape(-1)
    out = torch.zeros(n, dtype=torch.bool)
    limit = min(n, int(loaded.shape[0]))
    out[:limit] = loaded[:limit]
    return out


def _empty_entropy_reason_rows(n: int) -> List[Dict[str, Any]]:
    return [
        {
            "index": idx,
            "query_entropy_missing_reason": "not_processed",
            "answer_entropy_missing_reason": "not_processed",
            "query_prob_missing_reason": "not_processed",
            "answer_prob_missing_reason": "not_processed",
        }
        for idx in range(n)
    ]


def _load_entropy_reason_rows(path: str, n: int) -> List[Dict[str, Any]]:
    rows = _empty_entropy_reason_rows(n)
    if not os.path.exists(path):
        return rows
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return rows
    if not isinstance(loaded, list):
        return rows
    for idx, row in enumerate(loaded[:n]):
        if isinstance(row, dict):
            rows[idx].update(row)
            rows[idx]["index"] = idx
    return rows


def _set_missing_reason(
    rows: List[Dict[str, Any]],
    idx: int,
    key: str,
    available: bool,
    reason: Optional[str],
) -> None:
    rows[idx][key] = None if available else (reason or "unavailable")


def _write_entropy_sidecars(
    hidden_state_output_dir: str,
    *,
    query_entropy_available: torch.Tensor,
    answer_entropy_available: torch.Tensor,
    query_prob_available: torch.Tensor,
    answer_prob_available: torch.Tensor,
    reason_rows: List[Dict[str, Any]],
) -> None:
    Path(hidden_state_output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(query_entropy_available, hidden_state_output_dir + "query_entropy_available.pt")
    torch.save(answer_entropy_available, hidden_state_output_dir + "answer_entropy_available.pt")
    torch.save(query_prob_available, hidden_state_output_dir + "query_prob_available.pt")
    torch.save(answer_prob_available, hidden_state_output_dir + "answer_prob_available.pt")
    Path(hidden_state_output_dir, "entropy_missing_reasons.json").write_text(
        json.dumps(reason_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _feature_done(tensor: torch.Tensor, available: torch.Tensor, reason: Any) -> bool:
    return bool(
        torch.isfinite(tensor.float()).all()
        and (bool(available.item()) or reason not in {None, "not_processed"})
    )


def _require_input(stage: str, path, *, dataset: str = "", split_tag: str = "") -> None:
    """Raise FileNotFoundError with stage / split context if ``path`` is missing."""
    from pathlib import Path as _Path
    p = _Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"[response-cache] stage={stage} dataset={dataset} "
            f"split_tag={split_tag} missing input path: {p}"
        )


# Canonical artifact roots. Computed once at import time so that any
# ``./test_output`` / direct model-cache literals removed below cannot drift
# back. ``data/`` is a source package; generated artifacts belong under
# the resolved artifact roots.
_TEST_OUTPUT_ROOT = str(get_test_output_dir())
_MODELS_ROOT = get_models_dir()


class StopWordStoppingCriteria(StoppingCriteria):
    """StopWord stopping criteria."""

    def __init__(self, tokenizer, stop_word):
        self.tokenizer = tokenizer
        self.stop_word = stop_word
        self.length = len(self.stop_word)

    def __call__(self, input_ids, *args, **kwargs) -> bool:
        cur_text = self.tokenizer.decode(input_ids[0])
        return cur_text[-self.length :] == self.stop_word


def generate_stopword_stopping_criteria(
    eos_words: list[str],
    tokenizer: transformers.AutoTokenizer,
) -> StoppingCriteriaList:
    stop_criteria = StoppingCriteriaList()
    for word in eos_words:
        stop_criteria.append(StopWordStoppingCriteria(tokenizer, word))
    return stop_criteria


def _assert_hidden_bank_sample_alignment(
    resolved_variant: str,
    formatter_rows: list[dict[str, Any]],
    response_rows: list[dict[str, Any]],
) -> None:
    formatter_ids = [row.get("sample_id") for row in formatter_rows]
    response_ids = [row.get("sample_id") for row in response_rows]
    if any(sample_id in (None, "") for sample_id in formatter_ids):
        raise ValueError(
            f"hidden-bank formatter rows missing sample_id for resolved_variant={resolved_variant!r}"
        )
    if any(sample_id in (None, "") for sample_id in response_ids):
        raise ValueError(
            f"response-cache rows missing sample_id for resolved_variant={resolved_variant!r}"
        )
    if formatter_ids != response_ids:
        raise ValueError(
            "hidden-bank sample_id alignment mismatch: "
            f"resolved_variant={resolved_variant!r} "
            f"formatter_sample_ids={formatter_ids[:5]} "
            f"response_cache_sample_ids={response_ids[:5]} "
            f"formatter_count={len(formatter_ids)} response_count={len(response_ids)}"
        )


def _response_cache_sample_ids_match(expected_rows: Sequence[Mapping[str, Any]], cached_rows: Sequence[Mapping[str, Any]]) -> bool:
    expected_ids = [row.get("sample_id") for row in expected_rows]
    cached_ids = [row.get("sample_id") for row in cached_rows]
    return expected_ids == cached_ids and all(sample_id not in (None, "") for sample_id in cached_ids)


def _load_or_initialize_response_rows(
    path: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    split_tag: str,
    stage: str,
) -> list[dict[str, Any]]:
    expected = [dict(row) for row in rows]
    if not os.path.exists(path):
        return expected
    with open(path, "r", encoding="utf-8") as fr:
        cached = json.load(fr)
    if not isinstance(cached, list):
        logger.warning(
            "[response-cache] stage={} split_tag={} ignoring stale cache with non-list payload: {}",
            stage, split_tag, path,
        )
        return expected
    cached_rows = [dict(row) for row in cached if isinstance(row, Mapping)]
    if len(cached_rows) == len(expected) and _response_cache_sample_ids_match(expected, cached_rows):
        return cached_rows
    logger.warning(
        "[response-cache] stage={} split_tag={} ignoring stale cache because row/sample_id alignment changed: "
        "cached_count={} expected_count={} cached_ids={} expected_ids={}",
        stage,
        split_tag,
        len(cached_rows),
        len(expected),
        [row.get("sample_id") for row in cached_rows[:5]],
        [row.get("sample_id") for row in expected[:5]],
    )
    return expected


def generate_X(
    data_model: str,
    resolved_variant: str,
    model_type: str,
    mduq_mode: bool = False,
    *,
    pair_context=None,
    runtime_root=None,
    internal_train_ratio: float = 0.7,
    internal_split_seed: int = 42,
):
    """Extract per-layer hidden-state features and entropy/prob statistics.

    When ``mduq_mode=True`` the candidate-layer list comes from
    :mod:`registry.model_registry` (full multi-layer hidden bank) and outputs
    are saved under ``./test_output/<dataset>/<model>/diaguq/hidden_bank/``.
    The cross-model branch is forced off in MDUQ mode -- only the self
    pipeline is supported for the multi-layer bank.
    """
    if pair_context is not None:
        resolved_variant = pair_context.resolved_variant
        runtime_root = pair_context.test_output_root

    resolve_supported_dataset_variant(resolved_variant)
    output_dir = str(runtime_root or _TEST_OUTPUT_ROOT)

    artifacts = locate_response_cache_artifacts(
        resolved_variant,
        data_model,
        runtime_root=runtime_root,
    )
    dataset_variant = artifacts.dataset_variant
    data_extend_path = str(artifacts.require("mextend"))
    _require_input(
        "generate_X",
        data_extend_path,
        dataset=artifacts.dataset_name,
        split_tag=dataset_variant,
    )

    with open(data_extend_path) as f:
        data_extend = json.load(f)

    if mduq_mode:
        if pair_context is not None:
            hidden_state_output_dir = str(pair_context.hidden_bank_dir) + "/"
        else:
            hidden_state_output_dir = (
                output_dir
                + "/"
                + dataset_variant
                + "/"
                + model_type
                + "/diaguq/hidden_bank/"
            )
        # DiagUQ multi-layer bank only runs the self-extraction pipeline.
        data_model = model_type
    else:
        hidden_state_output_dir = (
            output_dir + "/" + dataset_variant + "/" + model_type + "/"
        )

    MOST_ANSWER = "most_likely_answer"
    PROMPT_TOKENS = "tokenized_prompt"
    Q_BEGIN = "question_token_start_idx"
    Q_END = "answer_token_start_idx"
    STEP_SIZE = 500

    num_queries = 0
    for i in range(len(data_extend)):
        if MOST_ANSWER in data_extend[i]:
            num_queries += 1
    data_extend = data_extend[:num_queries]

    model, tokenizer = load_model_by_name(model_type)

    answer_strs = [data_extend[i][MOST_ANSWER] for i in range(len(data_extend))]

    # tokenize answer_strs without special tokens
    tokenized_answers = [
        tokenizer.encode(answer_str, add_special_tokens=False)
        for answer_str in answer_strs
    ]

    num_queries = len(answer_strs)

    data = load_dataset_for_variant(
        dataset_variant,
        limit=num_queries,
        tokenizer=tokenizer,
        cache=True,
        internal_train_ratio=internal_train_ratio,
        internal_split_seed=internal_split_seed,
    )
    if len(data) < num_queries:
        raise ValueError(
            f"hidden-bank formatter returned fewer rows than response-cache: "
            f"resolved_variant={dataset_variant!r} formatter_count={len(data)} "
            f"response_cache_count={num_queries}"
        )
    data = data[:num_queries]
    _assert_hidden_bank_sample_alignment(dataset_variant, data, data_extend)

    output_token_average_hidden_states = True
    len_of_token_hidden_states_output = 1  # if set to zero, then not used
    get_query_entropies = True  # whether to get the entropy of the output token
    get_query_probs = True

    layer_list, num_dim = get_layer_list_and_dim(model_type)

    num_entropy_statistics = 4

    # initialize output_tensor as num_layers x num_queries x num_dim
    if output_token_average_hidden_states:
        query_output_average_tensor = torch.zeros(
            (num_queries, len(layer_list), num_dim), dtype=torch.float16
        )
        answer_output_average_tensor = torch.zeros(
            (num_queries, len(layer_list), num_dim), dtype=torch.float16
        )
    if len_of_token_hidden_states_output > 0:
        query_output_last_token_tensor = torch.zeros(
            (
                num_queries,
                len(layer_list),
                len_of_token_hidden_states_output,
                num_dim,
            ),
            dtype=torch.float16,
        )
        answer_output_last_token_tensor = torch.zeros(
            (
                num_queries,
                len(layer_list),
                len_of_token_hidden_states_output,
                num_dim,
            ),
            dtype=torch.float16,
        )
    if get_query_entropies:
        query_entropy_output_tensor = torch.zeros(
            (num_queries, num_entropy_statistics), dtype=torch.float16
        )
        answer_entropy_output_tensor = torch.zeros(
            (num_queries, num_entropy_statistics), dtype=torch.float16
        )
    if get_query_probs:
        query_prob_output_tensor = torch.zeros(
            (num_queries, 6), dtype=torch.float16
        )
        answer_prob_output_tensor = torch.zeros(
            (num_queries, 6), dtype=torch.float16
        )
    query_entropy_available_tensor = torch.zeros(num_queries, dtype=torch.bool)
    answer_entropy_available_tensor = torch.zeros(num_queries, dtype=torch.bool)
    query_prob_available_tensor = torch.zeros(num_queries, dtype=torch.bool)
    answer_prob_available_tensor = torch.zeros(num_queries, dtype=torch.bool)
    entropy_reason_rows = _empty_entropy_reason_rows(num_queries)

    # load the tensors if they have existed

    for idx, layer_idx in enumerate(layer_list):
        if model_type == data_model:
            if os.path.exists(
                hidden_state_output_dir
                + "answer_last_"
                + str(len_of_token_hidden_states_output)
                + "_token_layer_"
                + str(layer_idx)
                + ".pt"
            ):
                query_output_average_tensor[:, idx, :] = torch.load(
                    hidden_state_output_dir
                    + "query_average_layer_"
                    + str(layer_idx)
                    + ".pt"
                )
                query_output_last_token_tensor[:, idx, :, :] = torch.load(
                    hidden_state_output_dir
                    + "query_last_"
                    + str(len_of_token_hidden_states_output)
                    + "_token_layer_"
                    + str(layer_idx)
                    + ".pt"
                )
                answer_output_average_tensor[:, idx, :] = torch.load(
                    hidden_state_output_dir
                    + "answer_average_layer_"
                    + str(layer_idx)
                    + ".pt"
                )
                answer_output_last_token_tensor[:, idx, :, :] = torch.load(
                    hidden_state_output_dir
                    + "answer_last_"
                    + str(len_of_token_hidden_states_output)
                    + "_token_layer_"
                    + str(layer_idx)
                    + ".pt"
                )
        else:
            if os.path.exists(
                hidden_state_output_dir
                + "cross_answer_last_"
                + str(len_of_token_hidden_states_output)
                + "_token_layer_"
                + str(layer_idx)
                + ".pt"
            ):
                query_output_average_tensor[:, idx, :] = torch.load(
                    hidden_state_output_dir
                    + "cross_query_average_layer_"
                    + str(layer_idx)
                    + ".pt"
                )
                query_output_last_token_tensor[:, idx, :, :] = torch.load(
                    hidden_state_output_dir
                    + "cross_query_last_"
                    + str(len_of_token_hidden_states_output)
                    + "_token_layer_"
                    + str(layer_idx)
                    + ".pt"
                )
                answer_output_average_tensor[:, idx, :] = torch.load(
                    hidden_state_output_dir
                    + "cross_answer_average_layer_"
                    + str(layer_idx)
                    + ".pt"
                )
                answer_output_last_token_tensor[:, idx, :, :] = torch.load(
                    hidden_state_output_dir
                    + "cross_answer_last_"
                    + str(len_of_token_hidden_states_output)
                    + "_token_layer_"
                    + str(layer_idx)
                    + ".pt"
                )

    if model_type == data_model:
        if os.path.exists(hidden_state_output_dir + "query_entropies.pt"):
            query_entropy_output_tensor = torch.load(
                hidden_state_output_dir + "query_entropies.pt"
            )
            query_prob_output_tensor = torch.load(
                hidden_state_output_dir + "query_probs.pt"
            )
            answer_entropy_output_tensor = torch.load(
                hidden_state_output_dir + "answer_entropies.pt"
            )
            answer_prob_output_tensor = torch.load(
                hidden_state_output_dir + "answer_probs.pt"
            )
            query_entropy_available_tensor = _load_bool_sidecar(
                hidden_state_output_dir + "query_entropy_available.pt", num_queries
            )
            answer_entropy_available_tensor = _load_bool_sidecar(
                hidden_state_output_dir + "answer_entropy_available.pt", num_queries
            )
            query_prob_available_tensor = _load_bool_sidecar(
                hidden_state_output_dir + "query_prob_available.pt", num_queries
            )
            answer_prob_available_tensor = _load_bool_sidecar(
                hidden_state_output_dir + "answer_prob_available.pt", num_queries
            )
            entropy_reason_rows = _load_entropy_reason_rows(
                hidden_state_output_dir + "entropy_missing_reasons.json", num_queries
            )
    else:
        if os.path.exists(hidden_state_output_dir + "cross_query_entropies.pt"):
            query_entropy_output_tensor = torch.load(
                hidden_state_output_dir + "cross_query_entropies.pt"
            )
            query_prob_output_tensor = torch.load(
                hidden_state_output_dir + "cross_query_probs.pt"
            )
            answer_entropy_output_tensor = torch.load(
                hidden_state_output_dir + "cross_answer_entropies.pt"
            )
            answer_prob_output_tensor = torch.load(
                hidden_state_output_dir + "cross_answer_probs.pt"
            )
            query_entropy_available_tensor = _load_bool_sidecar(
                hidden_state_output_dir + "query_entropy_available.pt", num_queries
            )
            answer_entropy_available_tensor = _load_bool_sidecar(
                hidden_state_output_dir + "answer_entropy_available.pt", num_queries
            )
            query_prob_available_tensor = _load_bool_sidecar(
                hidden_state_output_dir + "query_prob_available.pt", num_queries
            )
            answer_prob_available_tensor = _load_bool_sidecar(
                hidden_state_output_dir + "answer_prob_available.pt", num_queries
            )
            entropy_reason_rows = _load_entropy_reason_rows(
                hidden_state_output_dir + "entropy_missing_reasons.json", num_queries
            )

    # set the device as the device the model is on
    device = model.device

    # forward and get features of the query
    for data_i, d in tqdm(enumerate(data)):
        if data_i >= num_queries:
            break

        # If all hidden-state tensors and entropy sidecars for this row are already valid,
        # keep the cached row. Legacy all-NaN entropy files intentionally fall through.
        if (
            torch.sum(torch.abs(answer_output_average_tensor[data_i].float())) > 0
            and _feature_done(
                query_entropy_output_tensor[data_i],
                query_entropy_available_tensor[data_i],
                entropy_reason_rows[data_i].get("query_entropy_missing_reason"),
            )
            and _feature_done(
                query_prob_output_tensor[data_i],
                query_prob_available_tensor[data_i],
                entropy_reason_rows[data_i].get("query_prob_missing_reason"),
            )
        ):
            continue

        q_begin = d[Q_BEGIN]
        q_end = d[Q_END]
        a_begin = q_end - 1
        a_end = q_end + len(tokenized_answers[data_i])
        answer_token_count = len(tokenized_answers[data_i])
        query_prompt_token = d[PROMPT_TOKENS][:q_end]
        answer_token = tokenized_answers[data_i]
        # concatenate the prompt token and the answer token
        prompt_token = query_prompt_token + answer_token

        # convert prompt_token to tensor
        prompt_token = torch.tensor(prompt_token).unsqueeze(0)
        prompt_token = prompt_token.to(device)

        outputs = model.forward(prompt_token, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        if not os.path.exists(hidden_state_output_dir):
            os.makedirs(hidden_state_output_dir)

        if output_token_average_hidden_states:
            query_output_average_tensor[data_i] = get_average_hidden_states(
                hidden_states, layer_list, q_begin, q_end, num_dim=num_dim
            )
            answer_output_average_tensor[data_i] = get_average_hidden_states(
                hidden_states, layer_list, a_begin, a_end, num_dim=num_dim
            )
        if len_of_token_hidden_states_output > 0:
            query_output_last_token_tensor[data_i] = (
                get_last_token_hidden_states(
                    hidden_states,
                    layer_list,
                    q_end,
                    len_of_token_hidden_states_output,
                    num_dim=num_dim,
                )
            )
            answer_output_last_token_tensor[data_i] = (
                get_last_token_hidden_states(
                    hidden_states,
                    layer_list,
                    a_end,
                    len_of_token_hidden_states_output,
                    num_dim=num_dim,
                )
            )

        if get_query_entropies:
            q_entropy, q_entropy_ok, q_entropy_reason = get_entropy_statistics(
                outputs.logits,
                q_begin,
                q_end,
                span_label="query",
                return_metadata=True,
            )
            query_entropy_output_tensor[data_i, :] = q_entropy.to(torch.float16)
            query_entropy_available_tensor[data_i] = q_entropy_ok
            _set_missing_reason(
                entropy_reason_rows,
                data_i,
                "query_entropy_missing_reason",
                q_entropy_ok,
                q_entropy_reason,
            )
            if answer_token_count <= 0:
                a_entropy = torch.zeros(len(ENTROPY_STAT_NAMES), dtype=torch.float32)
                a_entropy_ok = False
                a_entropy_reason = "empty_answer_span"
            else:
                a_entropy, a_entropy_ok, a_entropy_reason = get_entropy_statistics(
                    outputs.logits,
                    a_begin,
                    a_end,
                    query=False,
                    span_label="answer",
                    return_metadata=True,
                )
            answer_entropy_output_tensor[data_i, :] = a_entropy.to(torch.float16)
            answer_entropy_available_tensor[data_i] = a_entropy_ok
            _set_missing_reason(
                entropy_reason_rows,
                data_i,
                "answer_entropy_missing_reason",
                a_entropy_ok,
                a_entropy_reason,
            )

        if get_query_probs:
            q_prob, q_prob_ok, q_prob_reason = get_prob_statistics(
                outputs.logits,
                prompt_token,
                q_begin,
                q_end,
                query=False,
                span_label="query",
                return_metadata=True,
            )
            query_prob_output_tensor[data_i, :] = q_prob.to(torch.float16)
            query_prob_available_tensor[data_i] = q_prob_ok
            _set_missing_reason(
                entropy_reason_rows,
                data_i,
                "query_prob_missing_reason",
                q_prob_ok,
                q_prob_reason,
            )
            if answer_token_count <= 0:
                a_prob = torch.zeros(len(PROB_STAT_NAMES), dtype=torch.float32)
                a_prob_ok = False
                a_prob_reason = "empty_answer_span"
            else:
                a_prob, a_prob_ok, a_prob_reason = get_prob_statistics(
                    outputs.logits,
                    prompt_token,
                    a_begin,
                    a_end,
                    query=False,
                    span_label="answer",
                    return_metadata=True,
                )
            answer_prob_output_tensor[data_i, :] = a_prob.to(torch.float16)
            answer_prob_available_tensor[data_i] = a_prob_ok
            _set_missing_reason(
                entropy_reason_rows,
                data_i,
                "answer_prob_missing_reason",
                a_prob_ok,
                a_prob_reason,
            )

        if (data_i + 1) % STEP_SIZE == 0 or (data_i + 1) == num_queries:
            # save the hidden_states output
            for idx, layer_idx in enumerate(layer_list):
                if model_type == data_model:
                    torch.save(
                        query_output_average_tensor[:, idx, :],
                        hidden_state_output_dir
                        + "query_average_layer_"
                        + str(layer_idx)
                        + ".pt",
                    )
                    torch.save(
                        query_output_last_token_tensor[:, idx, :, :],
                        hidden_state_output_dir
                        + "query_last_"
                        + str(len_of_token_hidden_states_output)
                        + "_token_layer_"
                        + str(layer_idx)
                        + ".pt",
                    )
                    torch.save(
                        answer_output_average_tensor[:, idx, :],
                        hidden_state_output_dir
                        + "answer_average_layer_"
                        + str(layer_idx)
                        + ".pt",
                    )
                    torch.save(
                        answer_output_last_token_tensor[:, idx, :, :],
                        hidden_state_output_dir
                        + "answer_last_"
                        + str(len_of_token_hidden_states_output)
                        + "_token_layer_"
                        + str(layer_idx)
                        + ".pt",
                    )
                else:
                    torch.save(
                        query_output_average_tensor[:, idx, :],
                        hidden_state_output_dir
                        + "cross_query_average_layer_"
                        + str(layer_idx)
                        + ".pt",
                    )
                    torch.save(
                        query_output_last_token_tensor[:, idx, :, :],
                        hidden_state_output_dir
                        + "cross_query_last_"
                        + str(len_of_token_hidden_states_output)
                        + "_token_layer_"
                        + str(layer_idx)
                        + ".pt",
                    )
                    torch.save(
                        answer_output_average_tensor[:, idx, :],
                        hidden_state_output_dir
                        + "cross_answer_average_layer_"
                        + str(layer_idx)
                        + ".pt",
                    )
                    torch.save(
                        answer_output_last_token_tensor[:, idx, :, :],
                        hidden_state_output_dir
                        + "cross_answer_last_"
                        + str(len_of_token_hidden_states_output)
                        + "_token_layer_"
                        + str(layer_idx)
                        + ".pt",
                    )

            if model_type == data_model:
                torch.save(
                    query_entropy_output_tensor,
                    hidden_state_output_dir + "query_entropies.pt",
                )
                torch.save(
                    query_prob_output_tensor,
                    hidden_state_output_dir + "query_probs.pt",
                )
                torch.save(
                    answer_entropy_output_tensor,
                    hidden_state_output_dir + "answer_entropies.pt",
                )
                torch.save(
                    answer_prob_output_tensor,
                    hidden_state_output_dir + "answer_probs.pt",
                )
                _write_entropy_sidecars(
                    hidden_state_output_dir,
                    query_entropy_available=query_entropy_available_tensor,
                    answer_entropy_available=answer_entropy_available_tensor,
                    query_prob_available=query_prob_available_tensor,
                    answer_prob_available=answer_prob_available_tensor,
                    reason_rows=entropy_reason_rows,
                )
            else:
                torch.save(
                    query_entropy_output_tensor,
                    hidden_state_output_dir + "cross_query_entropies.pt",
                )
                torch.save(
                    query_prob_output_tensor,
                    hidden_state_output_dir + "cross_query_probs.pt",
                )
                torch.save(
                    answer_entropy_output_tensor,
                    hidden_state_output_dir + "cross_answer_entropies.pt",
                )
                torch.save(
                    answer_prob_output_tensor,
                    hidden_state_output_dir + "cross_answer_probs.pt",
                )
                _write_entropy_sidecars(
                    hidden_state_output_dir,
                    query_entropy_available=query_entropy_available_tensor,
                    answer_entropy_available=answer_entropy_available_tensor,
                    query_prob_available=query_prob_available_tensor,
                    answer_prob_available=answer_prob_available_tensor,
                    reason_rows=entropy_reason_rows,
                )


def generate_answer_most(
    model_type: str,
    dataset_name: str,
    limit: Optional[int] = None,
    dry_run: bool = False,
    allow_full_formatting: bool = False,
    internal_train_ratio: float = 0.7,
    internal_split_seed: int = 42,
):
    # ``dataset_name`` here is the canonical split_tag (e.g. ``triviaqa__train``).
    split_tag = dataset_name
    base_dataset, raw_split = split_dataset_and_raw(split_tag)
    row_limit = resolve_response_cache_limit(
        split_tag,
        limit,
        allow_full_formatting=allow_full_formatting,
    )

    if dry_run:
        return validate_response_cache_dataset_request(
            split_tag,
            sample_limit=min(int(row_limit or limit or 2), 2),
            requested_limit=row_limit,
            allow_full_formatting=allow_full_formatting,
            internal_train_ratio=internal_train_ratio,
            internal_split_seed=internal_split_seed,
        )

    if base_dataset not in {"triviaqa", "coqa", "ambigqa", "truthfulqa", "mmlu", "wmt"}:
        raise NotImplementedError(
            f"response-cache generation does not support dataset={base_dataset}, split={raw_split}"
        )

    validate_response_cache_dataset_request(
        split_tag,
        sample_limit=min(int(row_limit or limit or 2), 2),
        requested_limit=row_limit,
        allow_full_formatting=allow_full_formatting,
        internal_train_ratio=internal_train_ratio,
        internal_split_seed=internal_split_seed,
    )

    model, tokenizer = load_model_by_name(model_type)

    data = load_formatted_dataset(
        base_dataset,
        raw_split,
        tokenizer,
        limit=row_limit,
        dataset_variant=split_tag,
        allow_full_formatting=allow_full_formatting,
        internal_train_ratio=internal_train_ratio,
        internal_split_seed=internal_split_seed,
    )

    pair_dir = split_artifact_dir(split_tag, model_type)
    hidden_state_output_dir = str(pair_dir) + "/"

    PROMPT_TOKENS = "tokenized_prompt"
    Q_END = "answer_token_start_idx"

    dataset_extend_path = str(mextend_path(split_tag, model_type))
    samples_snapshot_path = str(mextend_samples_path(split_tag, model_type))
    logger.info(
        "[response-cache] stage=greedy dataset={} raw_split={} split_tag={} "
        "model={} output={}",
        base_dataset, raw_split, split_tag, model_type, dataset_extend_path,
    )
    pair_dir.mkdir(parents=True, exist_ok=True)

    # if the path not exists, then create the path
    if not os.path.exists(hidden_state_output_dir):
        os.makedirs(hidden_state_output_dir)

    if os.path.exists(dataset_extend_path):
        data_extend = _load_or_initialize_response_rows(
            dataset_extend_path,
            data,
            split_tag=split_tag,
            stage="greedy",
        )
    else:
        time1 = time()
        data_extend = list(data)
        time2 = time()
        print("Time to list the data:", time2 - time1)

    GREEDY = "most_likely_answer"

    if base_dataset in {"triviaqa", "coqa", "ambigqa", "truthfulqa"}:
        MAX_LENGTH_OF_GENERATED_SEQUENCE = 50
        period_words = [
            "Question:",
            "Question:",
            "\n",
            "\n\n",
            "\n\n\n",
            "\n\n\n\n",
            "\n\n\n\n\n",
            "<eos>",
            "Answer:",
            "Answer:",
            "?",
            "\nQ",
            "\nQ:",
            "\n2.",
        ]
        eos_words = [
            "Question:",
            " Question:",
            "\n",
            "\n\n",
            "\n\n\n",
            "\n\n\n\n",
            "\n\n\n\n\n",
            "<eos>",
            "Answer:",
            " Answer:",
            "Q:",
            "?",
            "<u>",
            "<h3>",
            "\nQ",
            "\nQ:",
            "\n2.",
            "Q",
            "Q ",
            "\n Q",
        ]
        STEP_SIZE = 50
    elif base_dataset == "mmlu":
        _generate_mmlu_answer_most(
            model,
            tokenizer,
            data_extend,
            dataset_extend_path,
            samples_snapshot_path,
            greedy_key=GREEDY,
        )
        return None
    elif dataset_name.startswith("cnndaily"):
        MAX_LENGTH_OF_GENERATED_SEQUENCE = 200
        eos_words = [
            "<end_of_turn>",
            "end_of_turn",
            "<start_of_turn>",
            "start_of_turn",
        ]
        STEP_SIZE = 20
    elif base_dataset == "wmt":
        MAX_LENGTH_OF_GENERATED_SEQUENCE = 100
        period_words = [
            "Question:",
            "Question:",
            "\n",
            "\n\n",
            "\n\n\n",
            "\n\n\n\n",
            "\n\n\n\n\n",
            "<eos>",
            "Answer:",
            "Answer:",
            "Q:",
            "?",
            "\nQ",
            "\nQ:",
            "\n2.",
        ]
        eos_words = [
            "Q:",
            "\n",
            "\n\n",
            "\n\n\n",
            "\n\n\n\n",
            "\n\n\n\n\n",
            "<eos>",
            "A:",
            "</s><s>",
            "\nQ",
            "\nQ:",
            "Q :",
            "Q. ",
            " Q. ",
            "What is the English",
        ]
        STEP_SIZE = 50
    else:
        raise NotImplementedError(
            f"response-cache generation does not support dataset={base_dataset}, split={raw_split}"
        )

    # question_framing_ids = [tokenizer.encode(word,add_special_tokens=False) for word in eos_words]
    period_token_id = [
        tokenizer.encode(word, add_special_tokens=False)[0]
        for word in period_words
    ]
    if model_type == "llama_2_7b":
        period_token_id.append(13)

    # unique the period_token_id
    period_token_id = list(set(period_token_id))

    TEMPERATURE = 1.0

    with torch.no_grad():

        for data_i in tqdm(range(len(data_extend))):
            d = data_extend[data_i]

            # check if this data has been processed before
            if GREEDY in d:
                continue

            input_length = d[Q_END]

            prompt_tokens = d[PROMPT_TOKENS][: d[Q_END]]

            prompt_tokens = (
                torch.tensor(prompt_tokens).to(model.device).unsqueeze(0)
            )

            try:
                answer_token = model.generate(
                    prompt_tokens,
                    max_length=input_length + MAX_LENGTH_OF_GENERATED_SEQUENCE,
                    num_return_sequences=1,
                    do_sample=False,
                    eos_token_id=period_token_id,
                )
            except:
                answer_token = model.generate(
                    prompt_tokens,
                    max_length=input_length + MAX_LENGTH_OF_GENERATED_SEQUENCE,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    eos_token_id=period_token_id,
                )

            sequence = tokenizer.decode(
                answer_token[0][input_length:], skip_special_tokens=True
            )

            # truncate the sequence with the first eos word
            for word in eos_words:
                if word in sequence:
                    sequence = sequence[: sequence.index(word)]
                    break
            data_extend[data_i][GREEDY] = sequence

            if (data_i + 1) % STEP_SIZE == 0 or data_i == len(data_extend) - 1:

                # save the extended data
                with open(dataset_extend_path, "w") as f:
                    json.dump(data_extend, f)

            if data_i == 100:
                with open(samples_snapshot_path, "w") as f:
                    json.dump(data_extend[: data_i + 1], f)


def _generate_mmlu_answer_most(
    model,
    tokenizer,
    data_extend: List[dict],
    dataset_extend_path: str,
    samples_snapshot_path: str,
    *,
    greedy_key: str,
) -> None:
    option_letters = ["A", "B", "C", "D"]
    option_token_ids = [_encode_single_letter_token(tokenizer, letter) for letter in option_letters]
    prompt_key = "tokenized_prompt"
    answer_start_key = "answer_token_start_idx"
    step_size = 50
    device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except Exception:  # noqa: BLE001
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for data_i in tqdm(range(len(data_extend))):
            row = data_extend[data_i]
            if greedy_key in row:
                continue
            input_length = int(row[answer_start_key])
            prompt_tokens = row[prompt_key][:input_length]
            prompt_tensor = torch.tensor(prompt_tokens).to(device).unsqueeze(0)
            outputs = model.forward(prompt_tensor)
            logits = outputs.logits[0, -1]
            scores = [float(logits[token_id].detach().float().cpu()) for token_id in option_token_ids]
            best_idx = int(max(range(len(scores)), key=lambda idx: scores[idx]))
            row[greedy_key] = option_letters[best_idx]
            row["mmlu_choice_scores"] = dict(zip(option_letters, scores))
            row["mmlu_prediction_method"] = "single_token_option_logit"

            if (data_i + 1) % step_size == 0 or data_i == len(data_extend) - 1:
                with open(dataset_extend_path, "w") as f:
                    json.dump(data_extend, f)
            if data_i == 100:
                with open(samples_snapshot_path, "w") as f:
                    json.dump(data_extend[: data_i + 1], f)


def generate_y_most_QA(
    model_type,
    dataset_name,
    *,
    f1_threshold: float = DEFAULT_QA_F1_THRESHOLD,
):
    split_tag = dataset_name
    base_dataset, raw_split = split_dataset_and_raw(split_tag)
    data_json_path = mextend_path(split_tag, model_type)
    data_extend_path = mextend_rouge_path(split_tag, model_type)
    audit_csv_path = response_answer_audit_csv_path(split_tag, model_type)
    audit_json_path = response_answer_audit_json_path(split_tag, model_type)
    logger.info(
        "[response-cache] stage=rouge dataset={} raw_split={} split_tag={} "
        "model={} input={} output={}",
        base_dataset, raw_split, split_tag, model_type,
        data_json_path, data_extend_path,
    )
    data_extend_path.parent.mkdir(parents=True, exist_ok=True)

    _require_input("qa_scoring", data_json_path,
                   dataset=base_dataset, split_tag=split_tag)
    with open(data_json_path, encoding="utf-8") as fr:
        data = json.load(fr)
    existing_metric_rows = []
    if data_extend_path.exists():
        with open(data_extend_path, encoding="utf-8") as fr:
            existing_metric_rows = json.load(fr)

    rouge_type_list = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    rouge_most = ["rouge1_most", "rouge2_most", "rougeL_most", "rougeLsum_most"]
    # Local, deterministic ROUGE backend. We deliberately do NOT use
    # ``evaluate.load("rouge")`` / ``datasets.load_metric("rouge")`` because
    # those resolve metric scripts via the HF Hub (or local module paths
    # like ``./rouge/rouge.py``), which fails offline with errors of the
    # form: ``Couldn't find a module script at .../rouge/rouge.py. Module
    # 'rouge' doesn't exist on the Hugging Face Hub either.``
    from rouge_score import rouge_scorer  # local, no network

    scorer = rouge_scorer.RougeScorer(rouge_type_list, use_stemmer=True)
    logger.info(
        "[response-cache] stage=rouge scorer_backend=local_rouge_score "
        "dataset={} split_tag={} model={} output={} n_rows={}",
        base_dataset, split_tag, model_type, data_extend_path,
        len(data),
    )
    _stage_t0 = time()

    def calculate_qa_metrics(d, sample_id: int, scorer):
        audit = score_open_domain_qa(
            d,
            sample_id=sample_id,
            f1_threshold=f1_threshold,
        )
        gold_answers = audit.get("gold_aliases") or [audit.get("gold_answer")]
        generated_answer = audit["extracted_answer"]
        best_scores = {rouge_type: 0.0 for rouge_type in rouge_type_list}
        if generated_answer:
            for reference in gold_answers:
                if not reference:
                    continue
                score = scorer.score(str(reference), generated_answer)
                for rouge_type in rouge_type_list:
                    best_scores[rouge_type] = max(
                        best_scores[rouge_type],
                        float(score[rouge_type].fmeasure),
                    )
        for rouge_idx, rouge_type in enumerate(rouge_type_list):
            d[rouge_most[rouge_idx]] = best_scores[rouge_type]
        d["raw_model_answer"] = audit["raw_model_answer"]
        d["extracted_answer"] = audit["extracted_answer"]
        d["gold_answer"] = audit["gold_answer"]
        d["gold_aliases"] = audit["gold_aliases"]
        d["normalized_prediction"] = audit["normalized_prediction"]
        d["normalized_gold_answers"] = audit["normalized_gold_answers"]
        d["exact_match"] = bool(audit["exact_match"])
        d["token_f1"] = float(audit["token_f1"])
        d["qa_score"] = float(max(float(audit["token_f1"]), 1.0 if audit["exact_match"] else 0.0))
        d["qa_correct"] = bool(audit["qa_correct"])
        d["qa_f1_threshold"] = float(f1_threshold)
        d["parse_status"] = audit["parse_status"]
        d["parse_error_reason"] = audit["parse_error_reason"]
        audit["rouge_score"] = d["rouge1_most"]
        audit["bleu_score"] = None
        return audit

    _PROGRESS_EVERY = 500
    data_extend_rouge = []
    audit_rows = []
    for from_idx in tqdm(range(len(data))):
        row = deepcopy(data[from_idx])
        if from_idx < len(existing_metric_rows) and isinstance(existing_metric_rows[from_idx], dict):
            row.update(existing_metric_rows[from_idx])
        try:
            previous_threshold = float(row.get("qa_f1_threshold", -1.0))
        except (TypeError, ValueError):
            previous_threshold = -1.0
        needs_recompute = (
            "qa_correct" not in row
            or "token_f1" not in row
            or previous_threshold != float(f1_threshold)
        )
        if needs_recompute:
            audit = calculate_qa_metrics(row, from_idx, scorer)
        else:
            audit = score_open_domain_qa(
                row,
                sample_id=from_idx,
                f1_threshold=f1_threshold,
                rouge_score=float(row.get("rouge1_most", 0.0)),
            )
        data_extend_rouge.append(row)
        audit_rows.append(audit)

        if (from_idx + 1) % _PROGRESS_EVERY == 0:
            logger.info(
                "[response-cache] stage=rouge split_tag={} progress={}/{} "
                "elapsed={:.1f}s",
                split_tag, from_idx + 1, len(data_extend_rouge),
                time() - _stage_t0,
            )
            with open(data_extend_path, "w") as fw:
                json.dump(data_extend_rouge, fw)
            write_answer_audit(
                audit_rows,
                csv_path=audit_csv_path,
                json_path=audit_json_path,
            )
        if from_idx > 18000:
            with open(data_extend_path, "w") as fw:
                json.dump(data_extend_rouge, fw)
            write_answer_audit(
                audit_rows,
                csv_path=audit_csv_path,
                json_path=audit_json_path,
            )
            break
    # Final flush -- previous loop only writes every 500 rows or after
    # 18000; small datasets would otherwise lose their last partial batch.
    with open(data_extend_path, "w") as fw:
        json.dump(data_extend_rouge, fw)
    write_answer_audit(
        audit_rows,
        csv_path=audit_csv_path,
        json_path=audit_json_path,
    )
    logger.info(
        "[response-cache] stage=rouge split_tag={} done elapsed={:.1f}s "
        "output={} audit_csv={} audit_json={}",
        split_tag, time() - _stage_t0, data_extend_path,
        audit_csv_path, audit_json_path,
    )


def generate_y_most_MMLU(model_type, dataset_name):
    split_tag = dataset_name
    base_dataset, raw_split = split_dataset_and_raw(split_tag)
    data_json_path = mextend_path(split_tag, model_type)
    data_extend_path = mextend_rouge_path(split_tag, model_type)
    logger.info(
        "[response-cache] stage=mmlu_metric dataset={} raw_split={} split_tag={} "
        "model={} input={} output={}",
        base_dataset, raw_split, split_tag, model_type,
        data_json_path, data_extend_path,
    )
    data_extend_path.parent.mkdir(parents=True, exist_ok=True)
    _require_input("mmlu_scoring", data_json_path, dataset=base_dataset, split_tag=split_tag)
    with open(data_json_path, encoding="utf-8") as fr:
        data = json.load(fr)

    rows = []
    for sample_id, row in enumerate(data):
        row = deepcopy(row)
        predicted, parse_status, parse_error_reason = parse_mmlu_option_answer(
            row.get("most_likely_answer"),
            choices=row.get("choices"),
        )
        gold = str(row.get("gold_option") or row.get("target") or row.get("gold_answer") or row.get("answer_str") or "").strip().upper()[:1]
        correct = bool(predicted and gold and predicted == gold)
        row["mmlu_predicted_choice"] = predicted
        row["mmlu_gold_choice"] = gold
        row["qa_score"] = 1.0 if correct else 0.0
        row["qa_correct"] = correct
        row["exact_match"] = correct
        row["token_f1"] = row["qa_score"]
        row["qa_f1_threshold"] = 1.0
        row["parse_status"] = parse_status
        row["parse_error_reason"] = parse_error_reason
        row["metric_source"] = "multiple_choice_option_logit"
        row["raw_model_answer"] = row.get("most_likely_answer")
        row["extracted_answer"] = predicted
        row["gold_answer"] = gold
        row["gold_aliases"] = [gold] if gold else []
        rows.append(row)

    with open(data_extend_path, "w", encoding="utf-8") as fw:
        json.dump(rows, fw)
    logger.info(
        "[response-cache] stage=mmlu_metric split_tag={} done output={} n_rows={}",
        split_tag, data_extend_path, len(rows),
    )


def generate_y_most_WMT(model_type, dataset_name):
    split_tag = dataset_name
    base_dataset, raw_split = split_dataset_and_raw(split_tag)
    data_json_path = mextend_path(split_tag, model_type)
    data_extend_path = mextend_bleu_path(split_tag, model_type)
    logger.info(
        "[response-cache] stage=bleu dataset={} raw_split={} split_tag={} "
        "model={} input={} output={}",
        base_dataset, raw_split, split_tag, model_type,
        data_json_path, data_extend_path,
    )
    data_extend_path.parent.mkdir(parents=True, exist_ok=True)

    MOST_ANSWER = "most_likely_answer"
    ANSWER_REF = "answer_str"

    if not data_extend_path.exists():
        _require_input("bleu", data_json_path,
                       dataset=base_dataset, split_tag=split_tag)
        with open(data_json_path) as fr:
            data = json.load(fr)
        data_extend_rouge = deepcopy(data)
    else:
        with open(data_extend_path) as fr:
            data_extend_rouge = json.load(fr)

    # Local sentence-level BLEU using NLTK (already a hard dependency).
    # Avoids ``evaluate.load("bleu")`` which resolves a metric script via
    # the HF Hub and breaks offline. We use the standard BLEU-4 with
    # method1 smoothing, then rescale into [0,1] -- the same range HF's
    # ``bleu`` metric returns -- so downstream consumers don't change.
    from nltk.translate.bleu_score import (
        SmoothingFunction,
        sentence_bleu,
    )

    metric = "bleu"
    _bleu_smoothing = SmoothingFunction().method1
    logger.info(
        "[response-cache] stage=bleu scorer_backend=local_nltk_sentence_bleu "
        "dataset={} split_tag={} model={} output={} n_rows={}",
        base_dataset, split_tag, model_type, data_extend_path,
        len(data_extend_rouge),
    )
    _stage_t0 = time()

    def calculate_bleu(d):
        generated_raw = d.get(MOST_ANSWER, "")
        if isinstance(generated_raw, list):
            generated_raw = generated_raw[0] if generated_raw else ""
        generated_answer = str(generated_raw).lstrip()
        reference = d[ANSWER_REF].lstrip()

        if generated_answer == "":
            d[metric] = 0.0
            return 0.0

        score = sentence_bleu(
            [reference.split()],
            generated_answer.split(),
            smoothing_function=_bleu_smoothing,
        )
        d[metric] = float(score)
        return score

    _PROGRESS_EVERY = 500
    for from_idx in tqdm(
        range(len(data_extend_rouge))
    ):  # len(data_extend_rouge)
        calculate_bleu(data_extend_rouge[from_idx])

        if (from_idx + 1) % _PROGRESS_EVERY == 0:
            logger.info(
                "[response-cache] stage=bleu split_tag={} progress={}/{} "
                "elapsed={:.1f}s",
                split_tag, from_idx + 1, len(data_extend_rouge),
                time() - _stage_t0,
            )
            with open(data_extend_path, "w") as fw:
                json.dump(data_extend_rouge, fw)
        if from_idx > 18000:
            with open(data_extend_path, "w") as fw:
                json.dump(data_extend_rouge, fw)
            break
    # Final flush so short datasets persist their last partial batch.
    with open(data_extend_path, "w") as fw:
        json.dump(data_extend_rouge, fw)
    logger.info(
        "[response-cache] stage=bleu split_tag={} done elapsed={:.1f}s "
        "output={}",
        split_tag, time() - _stage_t0, data_extend_path,
    )


def generate_answers(
    model_type,
    dataset_name,
    limit: Optional[int] = None,
    allow_full_formatting: bool = False,
    internal_train_ratio: float = 0.7,
    internal_split_seed: int = 42,
):
    split_tag = dataset_name
    base_dataset, raw_split = split_dataset_and_raw(split_tag)

    if base_dataset == "mmlu":
        raise NotImplementedError(
            f"response-cache generation does not support sampled answers for dataset={base_dataset}, split={raw_split}"
        )
    if base_dataset not in {"triviaqa", "coqa", "ambigqa", "truthfulqa", "wmt"}:
        raise NotImplementedError(
            f"response-cache generation does not support dataset={base_dataset}, split={raw_split}"
        )

    row_limit = resolve_response_cache_limit(
        split_tag,
        limit,
        allow_full_formatting=allow_full_formatting,
    )
    validate_response_cache_dataset_request(
        split_tag,
        sample_limit=min(int(row_limit or limit or 2), 2),
        requested_limit=row_limit,
        allow_full_formatting=allow_full_formatting,
        internal_train_ratio=internal_train_ratio,
        internal_split_seed=internal_split_seed,
    )

    model, tokenizer = load_model_by_name(model_type)

    data = load_formatted_dataset(
        base_dataset,
        raw_split,
        tokenizer,
        limit=row_limit,
        dataset_variant=split_tag,
        allow_full_formatting=allow_full_formatting,
        internal_train_ratio=internal_train_ratio,
        internal_split_seed=internal_split_seed,
    )

    pair_dir = split_artifact_dir(split_tag, model_type)
    hidden_state_output_dir = str(pair_dir) + "/"

    PROMPT_TOKENS = "tokenized_prompt"

    Q_END = "answer_token_start_idx"

    # generate multiple answers and get the features (statistics of entropy of output logits) of answers
    dataset_extend_path = str(extend_path(split_tag, model_type))
    samples_snapshot_path = str(extend_samples_path(split_tag, model_type))
    logger.info(
        "[response-cache] stage=samples dataset={} raw_split={} split_tag={} "
        "model={} output={}",
        base_dataset, raw_split, split_tag, model_type, dataset_extend_path,
    )
    pair_dir.mkdir(parents=True, exist_ok=True)

    # if the path not exists, then create the path
    if not os.path.exists(hidden_state_output_dir):
        os.makedirs(hidden_state_output_dir)

    if os.path.exists(dataset_extend_path):
        data_extend = _load_or_initialize_response_rows(
            dataset_extend_path,
            data,
            split_tag=split_tag,
            stage="samples",
        )
    else:
        time1 = time()
        data_extend = list(data)
        time2 = time()
        print("Time to list the data:", time2 - time1)

    ANSWERS = "generated_answers"

    if base_dataset in {"triviaqa", "coqa", "ambigqa", "truthfulqa"}:
        MAX_LENGTH_OF_GENERATED_SEQUENCE = 50
        period_words = [
            "Question:",
            "\n",
            "\n\n",
            "\n\n\n",
            "\n\n\n\n",
            "\n\n\n\n\n",
            "<eos>",
            "?",
            "<u>",
            "<h3>",
            "\nQ",
            "\nQ:",
            "\n2.",
        ]
        eos_words = [
            "Question:",
            " Question:",
            "\n",
            "\n\n",
            "\n\n\n",
            "\n\n\n\n",
            "\n\n\n\n\n",
            "<eos>",
            "Answer:",
            " Answer:",
            "Q:",
            "?",
            "<u>",
            "<h3>",
            "\nQ",
            "\nQ:",
            "\n2.",
            "Q",
            "Q ",
            "\n Q",
        ]
        NUM_GENERATION_PER_PROMPT = 10
        STEP_SIZE = 500
    elif dataset_name.startswith("cnndaily"):
        MAX_LENGTH_OF_GENERATED_SEQUENCE = 200
        eos_words = [
            "<end_of_turn>",
            "end_of_turn",
            "<start_of_turn>",
            "start_of_turn",
        ]
        NUM_GENERATION_PER_PROMPT = 10
        STEP_SIZE = 20
    elif base_dataset == "wmt":
        MAX_LENGTH_OF_GENERATED_SEQUENCE = 100
        period_words = [
            "Question:",
            "Question:",
            "\n",
            "\n\n",
            "\n\n\n",
            "\n\n\n\n",
            "\n\n\n\n\n",
            "<eos>",
            "Answer:",
            "Answer:",
            "Q:",
            "?",
            "\nQ",
            "\nQ:",
            "\n2.",
        ]
        eos_words = [
            "Q:",
            "\n",
            "\n\n",
            "\n\n\n",
            "\n\n\n\n",
            "\n\n\n\n\n",
            "<eos>",
            "A:",
            "</s><s>",
            "\nQ",
            "\nQ:",
            "Q :",
            "Q. ",
            " Q. ",
            "What is the English",
        ]
        NUM_GENERATION_PER_PROMPT = 5
        STEP_SIZE = 50
    else:
        raise NotImplementedError(
            f"response-cache generation does not support dataset={base_dataset}, split={raw_split}"
        )

    # question_framing_ids = [tokenizer.encode(word,add_special_tokens=False) for word in eos_words]
    period_token_id = [
        tokenizer.encode(word, add_special_tokens=False)[0]
        for word in period_words
    ]
    if model_type == "llama_2_7b":
        period_token_id.append(13)

    # unique the period_token_id
    period_token_id = list(set(period_token_id))

    TEMPERATURE = 1.0
    TOP_P = 1.0

    with torch.no_grad():

        for data_i in tqdm(range(len(data_extend))):
            d = data_extend[data_i]

            # check if this data has been processed before
            if (ANSWERS in d) and len(d[ANSWERS]) > 0:
                continue

            input_length = d[Q_END]
            data_extend[data_i][ANSWERS] = []
            prompt_tokens = d[PROMPT_TOKENS][: d[Q_END]]
            prompt_tokens = (
                torch.tensor(prompt_tokens).to(model.device).unsqueeze(0)
            )

            for i in range(NUM_GENERATION_PER_PROMPT):
                answer_token = model.generate(
                    prompt_tokens,
                    max_length=input_length + MAX_LENGTH_OF_GENERATED_SEQUENCE,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    eos_token_id=period_token_id,
                    top_p=TOP_P,
                )

                sequence = tokenizer.decode(
                    answer_token[0][input_length:], skip_special_tokens=True
                )

                # truncate the sequence with the first eos word
                for word in eos_words:
                    if word in sequence:
                        sequence = sequence[: sequence.index(word)]
                        break
                data_extend[data_i][ANSWERS].append(sequence)

            if (data_i + 1) % STEP_SIZE == 0 or data_i == len(data_extend) - 1:

                # save the extended data
                with open(dataset_extend_path, "w") as f:
                    json.dump(data_extend, f)

            if data_i == 100:
                with open(samples_snapshot_path, "w") as f:
                    json.dump(data_extend[: data_i + 1], f)


# Strict-output protocol version for ask4conf. Bump when changing the
# prompt template, parser, source policy, or jsonl schema so already-written
# shards are transparently re-generated by build-response-cache.
ASK4CONF_PROTOCOL_VERSION = 5
ASK4CONF_GENERATED_ANSWER_KEYS: Tuple[str, ...] = (
    "most_likely_answer",
    "greedy_answer",
    "generated_answer",
    "model_answer",
)
ASK4CONF_QUESTION_KEYS: Tuple[str, ...] = (
    "question_str",
    "question",
    "prompt_question",
    "input",
)
ASK4CONF_MAX_SOURCE_ERROR_RATE = 0.005
ASK4CONF_MAX_GENERATION_ERROR_RATE = 0.0
ASK4CONF_MAX_PARSE_FAILURE_RATE = 0.05
ASK4CONF_SOURCE_ERROR_POLICIES: Tuple[str, ...] = ("fail", "skip", "retry_then_skip")
ASK4CONF_DEFAULT_FORMAL_SOURCE_ERROR_POLICY = "retry_then_skip"
ASK4CONF_DEFAULT_DEBUG_SOURCE_ERROR_POLICY = "fail"


def _ask4conf_resolve_source_error_policy(policy: Optional[str], *, debug_limit: Optional[int] = None) -> str:
    resolved = policy or (
        ASK4CONF_DEFAULT_DEBUG_SOURCE_ERROR_POLICY
        if debug_limit is not None
        else ASK4CONF_DEFAULT_FORMAL_SOURCE_ERROR_POLICY
    )
    if resolved not in ASK4CONF_SOURCE_ERROR_POLICIES:
        raise ValueError(
            f"source_error_policy must be one of {ASK4CONF_SOURCE_ERROR_POLICIES}, got {resolved!r}"
        )
    return resolved


def _ask4conf_policy_allows_skip(policy: str) -> bool:
    return policy in {"skip", "retry_then_skip"}


def _ask4conf_validate_existing(
    jsonl_path: Path,
    meta_path: Path,
    success_marker: Path,
    source_errors_path: Path,
    *,
    protocol_version: int,
    expected_count: int,
    source_count: int,
    source_failed_count: Optional[int],
    source_error_policy: str,
    source_error_threshold: float,
) -> Tuple[bool, str]:
    """Return (is_valid, reason) for an existing ask4conf shard."""
    if not jsonl_path.exists() or not meta_path.exists():
        return False, "missing_jsonl_or_meta"
    if not success_marker.exists():
        return False, "missing_success_marker"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, f"meta_unreadable: {exc!r}"
    if int(meta.get("protocol_version", -1)) != protocol_version:
        return False, "protocol_version_mismatch"
    if str(meta.get("source_error_policy") or "") != str(source_error_policy):
        return False, "source_error_policy_changed"
    try:
        meta_threshold = float(meta.get("source_validation_threshold_used"))
    except Exception:
        return False, "source_error_threshold_missing"
    if abs(meta_threshold - float(source_error_threshold)) > 1e-12:
        return False, "source_error_threshold_changed"
    meta_expected = int(meta.get("expected_count", -1))
    meta_source = int(meta.get("source_count", -1))
    written = int(meta.get("written_count", -1))
    meta_source_failed = int(
        meta.get("source_failed_count", meta.get("source_error_count", -1))
    )
    if meta_expected != expected_count:
        return False, (
            "expected_count_changed: "
            f"meta={meta_expected} current={expected_count}"
        )
    if meta_source != source_count:
        return False, (
            "source_count_changed: "
            f"meta={meta_source} current={source_count}"
        )
    if source_failed_count is not None and meta_source_failed != source_failed_count:
        return False, (
            "source_failed_count_changed: "
            f"meta={meta_source_failed} current={source_failed_count}"
        )
    if expected_count < 0 or written + meta_source_failed != expected_count:
        return False, "incomplete_run"

    try:
        line_count = 0
        row_parse_failed = 0
        row_generation_failed = 0
        row_source_failed = 0
        with open(jsonl_path, "r", encoding="utf-8") as fr:
            for line in fr:
                line_count += 1
                row = json.loads(line)
                if int(row.get("protocol_version", -1)) != protocol_version:
                    return False, "row_protocol_version_mismatch"
                error_type = row.get("error_type")
                if error_type == "source_error":
                    row_source_failed += 1
                elif error_type == "generation_error":
                    row_generation_failed += 1
                if row.get("parse_status") != "ok":
                    row_parse_failed += 1
    except Exception as exc:  # noqa: BLE001
        return False, f"jsonl_unreadable: {exc!r}"

    if line_count != written:
        return False, f"line_count_mismatch: file={line_count} meta={written}"
    if row_source_failed:
        return False, f"source_error_rows_in_main_jsonl: {row_source_failed}"
    if meta_source_failed > 0:
        if not source_errors_path.exists():
            return False, "missing_source_errors_jsonl"
        try:
            with open(source_errors_path, "r", encoding="utf-8") as fr:
                source_error_lines = sum(1 for _ in fr)
        except Exception as exc:  # noqa: BLE001
            return False, f"source_errors_unreadable: {exc!r}"
        if source_error_lines != meta_source_failed:
            return False, (
                "source_error_line_count_mismatch: "
                f"file={source_error_lines} meta={meta_source_failed}"
            )
    if row_generation_failed:
        return False, f"generation_error_rows_present: {row_generation_failed}"
    parse_rate = row_parse_failed / max(line_count, 1)
    if parse_rate > ASK4CONF_MAX_PARSE_FAILURE_RATE:
        return False, (
            "parse_failure_rate_too_high: "
            f"{parse_rate:.4f}>{ASK4CONF_MAX_PARSE_FAILURE_RATE:.4f}"
        )
    return True, "ok"


def _ask4conf_build_prompt(
    tokenizer,
    *,
    question_text: str,
    answer_text: str,
) -> str:
    """Strict prompt: model must reply with ONE float in [0, 1] only.

    Uses the tokenizer's chat template when available (instruct models
    such as Llama-3.1-Instruct, Qwen2.5-Instruct, Gemma-it). Falls back
    to a plain template otherwise.
    """
    system_msg = (
        "You are a calibration assistant. Given a question and a "
        "candidate answer, output ONLY a single decimal number between "
        "0 and 1 representing the probability that the candidate answer "
        "is correct. Output the number on its own with no words, no "
        "punctuation other than the decimal point, no explanation."
    )
    user_msg = (
        f"Question: {question_text.strip()}\n"
        f"Candidate answer: {answer_text.strip()}\n\n"
        "Reply with only a single number between 0 and 1."
    )
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        try:
            return tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:  # noqa: BLE001
            pass
    # Plain-text fallback for base models without a chat template.
    return (
        f"{system_msg}\n\n{user_msg}\nProbability: "
    )


# Match a float; capture group 1 is the bare number (no leading sign).
_ASK4CONF_FLOAT_RE = re.compile(r"(?<![\w.])((?:0?\.\d+|1(?:\.0+)?|0))(?![\w.])")


def _ask4conf_parse_confidence(text: str) -> Tuple[Optional[float], str, str]:
    """Strict parser: returns (confidence, status, error_reason).

    status is one of ``"ok"`` or ``"parse_error"``. Confidence is None on
    failure. We deliberately accept only numbers in [0, 1].
    """
    if not text:
        return None, "parse_error", "empty_generation"
    s = text.strip()
    # Trim common chat-suffix artefacts.
    for sep in ("\n", "<|", "</s>", "<eos>"):
        idx = s.find(sep)
        if idx != -1:
            s = s[:idx].strip()
    # Accept JSON like {"confidence": 0.7}.
    if s.startswith("{") and "confidence" in s:
        try:
            obj = json.loads(s)
            v = float(obj["confidence"])
            if 0.0 <= v <= 1.0:
                return v, "ok", ""
            return None, "parse_error", f"json_out_of_range: {v}"
        except Exception as exc:  # noqa: BLE001
            return None, "parse_error", f"json_parse_error: {exc!r}"
    m = _ASK4CONF_FLOAT_RE.search(s)
    if not m:
        return None, "parse_error", f"no_float_in: {s[:64]!r}"
    try:
        v = float(m.group(1))
    except ValueError as exc:
        return None, "parse_error", f"float_cast_error: {exc!r}"
    if not (0.0 <= v <= 1.0):
        return None, "parse_error", f"out_of_range: {v}"
    return v, "ok", ""


def _ask4conf_stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        for item in value:
            text = _ask4conf_stringify(item)
            if text:
                return text
        return ""
    if isinstance(value, dict):
        for key in ("text", "answer", "generated_text", "content"):
            if key in value:
                text = _ask4conf_stringify(value[key])
                if text:
                    return text
        return ""
    return str(value).strip()


def _ask4conf_generated_answer(row: Mapping[str, Any]) -> Tuple[str, str]:
    for key in ASK4CONF_GENERATED_ANSWER_KEYS:
        if key in row:
            return _ask4conf_stringify(row.get(key)), key
    return "", "missing_generated_answer_key"


def _ask4conf_answer_error(answer_text: str) -> str:
    text = (answer_text or "").strip()
    if not text:
        return "missing_generated_answer"
    stripped = re.sub(
        r"^(?:candidate\s+answer|answer|a)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    lowered = stripped.lower()
    compact = re.sub(r"\s+", "", lowered)
    if not stripped:
        return "empty_after_answer_prefix"
    placeholders = {
        "<unk>",
        "[unk]",
        "unk",
        "<unknown>",
        "<pad>",
        "[pad]",
        "null",
        "none",
        "nan",
        "n/a",
    }
    if lowered in placeholders or compact in placeholders:
        return f"placeholder_generated_answer: {text[:80]!r}"
    if "<unk>" in lowered or "[unk]" in lowered:
        return f"placeholder_generated_answer: {text[:80]!r}"
    return ""


def _ask4conf_source_errors_path(split_tag: str, model_type: str) -> Path:
    return ask4conf_jsonl_path(split_tag, model_type).with_name(
        f"{split_tag}_source_errors.jsonl"
    )


def _ask4conf_source_error_audit_json_path(split_tag: str, model_type: str) -> Path:
    return ask4conf_jsonl_path(split_tag, model_type).with_name(
        f"{split_tag}_source_error_audit.json"
    )


def _ask4conf_source_error_audit_csv_path(split_tag: str, model_type: str) -> Path:
    return ask4conf_jsonl_path(split_tag, model_type).with_name(
        f"{split_tag}_source_error_audit.csv"
    )


def _ask4conf_mextend_validation_path(split_tag: str, source_path: Path) -> Path:
    return source_path.with_name(f"{split_tag}_mextend_validation.json")


def _ask4conf_bad_patterns(source_errors: Sequence[Mapping[str, Any]]) -> List[dict]:
    counts = Counter(
        (
            str(err.get("reason", "")),
            str(err.get("answer_preview", "")),
        )
        for err in source_errors
    )
    return [
        {"reason": reason, "answer_preview": preview, "count": count}
        for (reason, preview), count in counts.most_common(10)
    ]


def _ask4conf_write_source_errors(path: Path, records: Sequence[dict], *, split_tag: str, model_type: str) -> dict:
    audit_json = _ask4conf_source_error_audit_json_path(split_tag, model_type)
    audit_csv = _ask4conf_source_error_audit_csv_path(split_tag, model_type)
    if not records:
        for stale in (path, audit_json, audit_csv):
            try:
                stale.unlink()
            except FileNotFoundError:
                pass
        return {
            "jsonl": str(path),
            "json": str(audit_json),
            "csv": str(audit_csv),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fw:
        for record in records:
            fw.write(json.dumps(record, ensure_ascii=False))
            fw.write("\n")
    tmp_path.replace(path)

    audit_json_tmp = audit_json.with_suffix(audit_json.suffix + ".tmp")
    audit_json_tmp.write_text(
        json.dumps(list(records), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    audit_json_tmp.replace(audit_json)

    fieldnames = [
        "sample_idx",
        "sample_id",
        "question_id",
        "split_tag",
        "answer_source_key",
        "question_source_key",
        "reason",
        "answer_preview",
        "retry_attempted",
        "retry_success",
    ]
    audit_csv_tmp = audit_csv.with_suffix(audit_csv.suffix + ".tmp")
    with open(audit_csv_tmp, "w", newline="", encoding="utf-8") as fw:
        writer = csv.DictWriter(fw, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    audit_csv_tmp.replace(audit_csv)
    return {
        "jsonl": str(path),
        "json": str(audit_json),
        "csv": str(audit_csv),
    }


def _ask4conf_question_id(row: Mapping[str, Any], sample_idx: int) -> Any:
    for key in ("question_id", "id", "sample_id", "uid"):
        if row.get(key) is not None:
            return row.get(key)
    return sample_idx


def _ask4conf_sample_id(row: Mapping[str, Any], sample_idx: int, split_tag: str) -> str:
    value = row.get("sample_id")
    if value not in (None, ""):
        return str(value)
    return f"{split_tag}:{sample_idx}"


def _ask4conf_source_error_record(
    row: Mapping[str, Any],
    *,
    sample_idx: int,
    split_tag: str,
    answer_source_key: str,
    question_source_key: str = "",
    reason: str,
    answer_preview: str,
    question_str: str = "",
    retry_attempted: bool = False,
    retry_success: bool = False,
) -> dict:
    return {
        "sample_idx": sample_idx,
        "sample_id": _ask4conf_sample_id(row, sample_idx, split_tag),
        "question_id": _ask4conf_question_id(row, sample_idx),
        "split_tag": split_tag,
        "answer_source_key": answer_source_key,
        "question_source_key": question_source_key,
        "reason": reason,
        "answer_preview": str(answer_preview or "")[:120],
        "question_str": str(question_str or "")[:500],
        "retry_attempted": bool(retry_attempted),
        "retry_success": bool(retry_success),
    }


def _ask4conf_question_text(
    row: Mapping[str, Any], tokenizer=None
) -> Tuple[str, str]:
    for key in ASK4CONF_QUESTION_KEYS:
        text = _ask4conf_stringify(row.get(key))
        if text:
            return text, key
    if tokenizer is not None:
        try:
            prompt_tokens = row.get("tokenized_prompt")
            answer_idx = int(row.get("answer_token_start_idx"))
            if prompt_tokens is not None and answer_idx > 0:
                return (
                    tokenizer.decode(
                        prompt_tokens[:answer_idx],
                        skip_special_tokens=True,
                    ).strip(),
                    "decoded_prompt_prefix",
                )
        except Exception:  # noqa: BLE001
            pass
    return "", "missing_question"


def _ask4conf_source_split_tags(dataset_name: str, model_type: str) -> List[str]:
    if "__" in dataset_name:
        return [dataset_name]
    if dataset_name in ("coqa", "triviaqa"):
        return [f"{dataset_name}__train"]
    if dataset_name == "wmt":
        return ["wmt__train", "wmt__test"]

    root = Path(_TEST_OUTPUT_ROOT)
    split_tags: List[str] = []
    for split_dir in sorted(root.glob(f"{dataset_name}__*")):
        if not split_dir.is_dir():
            continue
        split_tag = split_dir.name
        if mextend_path(split_tag, model_type).is_file():
            split_tags.append(split_tag)
    if split_tags:
        return split_tags
    return [dataset_name]


def _ask4conf_load_response_cache(
    split_tag: str, model_type: str
) -> Tuple[List[dict], Path]:
    source_path = mextend_path(split_tag, model_type)
    base_dataset, _ = split_dataset_and_raw(split_tag)
    _require_input(
        "ask4conf_source",
        source_path,
        dataset=base_dataset,
        split_tag=split_tag,
    )
    with open(source_path, "r", encoding="utf-8") as fr:
        payload = json.load(fr)
    if not isinstance(payload, list):
        raise ValueError(
            f"[response-cache] stage=ask4conf split_tag={split_tag} "
            f"model={model_type} expected list in {source_path}, "
            f"got {type(payload).__name__}"
        )
    return payload, source_path


def _ask4conf_source_summary(
    rows: Sequence[Any],
    split_tag: str,
    source_path: Path,
    *,
    model_type: str,
    source_error_policy: str,
    source_error_threshold: float,
    write_validation_report: bool = True,
    write_source_error_audit: bool = False,
) -> dict:
    anomalies: List[dict] = []
    valid_answers = 0
    for sample_idx, row in enumerate(rows):
        if not isinstance(row, Mapping):
            anomalies.append(
                {
                    "sample_idx": sample_idx,
                    "sample_id": f"{split_tag}:{sample_idx}",
                    "split_tag": split_tag,
                    "reason": "source_row_not_object",
                    "answer_preview": repr(row)[:120],
                    "question_id": sample_idx,
                    "answer_source_key": "source_row_not_object",
                }
            )
            continue
        answer_text, answer_key = _ask4conf_generated_answer(row)
        reason = _ask4conf_answer_error(answer_text)
        if reason:
            anomalies.append(
                _ask4conf_source_error_record(
                    row,
                    sample_idx=sample_idx,
                    split_tag=split_tag,
                    answer_source_key=answer_key,
                    reason=reason,
                    answer_preview=answer_text,
                )
            )
        else:
            valid_answers += 1
    loaded = len(rows)
    source_error_count = len(anomalies)
    source_error_rate = source_error_count / max(loaded, 1)
    if loaded == 0:
        source_error_count = 1
        source_error_rate = 1.0
        anomalies.append(
            {
                "sample_idx": None,
                "sample_id": None,
                "split_tag": split_tag,
                "reason": "empty_response_cache_artifact",
                "answer_preview": "",
                "question_id": None,
                "answer_source_key": "missing_generated_answer_key",
            }
        )
    source_errors_path = _ask4conf_source_errors_path(split_tag, model_type)
    audit_paths = {
        "jsonl": str(source_errors_path),
        "json": str(_ask4conf_source_error_audit_json_path(split_tag, model_type)),
        "csv": str(_ask4conf_source_error_audit_csv_path(split_tag, model_type)),
    }
    should_write_audit = write_source_error_audit or (
        source_error_rate > source_error_threshold
        and not _ask4conf_policy_allows_skip(source_error_policy)
    )
    if should_write_audit:
        audit_paths = _ask4conf_write_source_errors(
            source_errors_path,
            anomalies,
            split_tag=split_tag,
            model_type=model_type,
        )
    summary = {
        "split_tag": split_tag,
        "source_path": str(source_path),
        "loaded_count": loaded,
        "valid_answer_count": valid_answers,
        "placeholder_or_missing_count": source_error_count,
        "source_error_count": source_error_count,
        "source_error_rate": source_error_rate,
        "source_error_policy": source_error_policy,
        "source_validation_threshold_used": float(source_error_threshold),
        "source_errors_path": str(source_errors_path),
        "source_error_audit_json": audit_paths["json"],
        "source_error_audit_csv": audit_paths["csv"],
        "top_source_errors": anomalies[:5],
        "top_bad_patterns": _ask4conf_bad_patterns(anomalies),
        "mextend_validation_path": str(
            _ask4conf_mextend_validation_path(split_tag, source_path)
        ),
    }
    if write_validation_report:
        validation_report = {
            "split_tag": split_tag,
            "source_path": str(source_path),
            "total_count": loaded,
            "valid_answer_count": valid_answers,
            "placeholder_or_missing_count": source_error_count,
            "placeholder_rate": source_error_rate,
            "source_error_policy": source_error_policy,
            "source_validation_threshold_used": float(source_error_threshold),
            "source_error_audit_json": audit_paths["json"],
            "source_error_audit_csv": audit_paths["csv"],
            "top_bad_patterns": summary["top_bad_patterns"],
            "bad_example_indices": [
                err.get("sample_idx") for err in anomalies[:20]
            ],
            "top_source_errors": anomalies[:5],
        }
        validation_path = _ask4conf_mextend_validation_path(split_tag, source_path)
        try:
            validation_path.write_text(
                json.dumps(validation_report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[response-cache] stage=ask4conf source_validation "
                "split_tag={} validation_report_write_failed path={} error={!r}",
                split_tag, validation_path, exc,
            )
    logger.info(
        "[response-cache] stage=ask4conf source_validation split_tag={} "
        "source={} loaded={} valid_answers={} placeholder_or_missing={} "
        "source_error_rate={:.4f}",
        split_tag,
        source_path,
        loaded,
        valid_answers,
        source_error_count,
        source_error_rate,
    )
    if anomalies:
        logger.warning(
            "[response-cache] stage=ask4conf source_validation "
            "split_tag={} below_threshold={} top_source_errors={}",
            split_tag,
            source_error_rate <= source_error_threshold,
            anomalies[:5],
        )
    if source_error_rate > source_error_threshold:
        logger.warning(
            "[response-cache] stage=ask4conf source_validation "
            "split_tag={} source_error_rate={:.4f} threshold={:.4f}",
            split_tag, source_error_rate, source_error_threshold,
        )
    if source_error_rate > source_error_threshold and not _ask4conf_policy_allows_skip(source_error_policy):
        raise ValueError(
            "[response-cache] stage=ask4conf source_error: "
            f"split_tag={split_tag} source={source_path} "
            f"loaded={loaded} valid_answers={valid_answers} "
            f"placeholder_or_missing={source_error_count} "
            f"policy={source_error_policy} "
            f"threshold={source_error_threshold:.4f} "
            f"source_error_audit_json={audit_paths['json']} "
            f"source_error_audit_csv={audit_paths['csv']} "
            f"top_source_errors={anomalies[:5]}"
        )
    return summary


def _ask4conf_retry_generated_answer(
    row: Mapping[str, Any],
    *,
    model,
    tokenizer,
    device,
    generation_config: GenerationConfig,
) -> Tuple[str, str]:
    question_text, _ = _ask4conf_question_text(row, tokenizer)
    try:
        prompt_tokens = row.get("tokenized_prompt")
        answer_idx = int(row.get("answer_token_start_idx") or 0)
    except Exception:
        prompt_tokens = None
        answer_idx = 0

    try:
        if prompt_tokens is not None and answer_idx > 0:
            prompt_ids = torch.tensor([list(prompt_tokens[:answer_idx])], device=device)
            attention_mask = torch.ones_like(prompt_ids)
        else:
            retry_prompt = (
                "Answer the question concisely. Do not output None, null, or placeholders.\n\n"
                f"Question: {question_text.strip()}\nAnswer:"
            )
            enc = tokenizer(
                retry_prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(prompt_ids)
            else:
                attention_mask = attention_mask.to(device)
        prompt_len = prompt_ids.shape[1]
        with torch.no_grad():
            retry_out = model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
            )
        raw_text = tokenizer.decode(
            retry_out[0][prompt_len:],
            skip_special_tokens=True,
        ).strip()
    except Exception as exc:  # noqa: BLE001
        return "", f"retry_generation_error: {exc!r}"

    text = raw_text
    for sep in ("\n\n", "\nQ:", "\nQuestion:"):
        if sep in text:
            text = text.split(sep, 1)[0].strip()
    reason = _ask4conf_answer_error(text)
    if reason:
        return "", f"retry_returned_invalid_answer: {reason}"
    return text, ""


def _ask4conf_prepare_rows(
    rows: Sequence[Mapping[str, Any]],
    split_tag: str,
    tokenizer,
    *,
    model=None,
    device=None,
    retry_source_errors: bool = False,
    retry_generation_config: Optional[GenerationConfig] = None,
) -> Tuple[List[dict], List[dict], dict]:
    prepared: List[dict] = []
    source_errors: List[dict] = []
    retry_stats = {"retried_rows": 0, "retry_success_rows": 0}
    for sample_idx, row in enumerate(rows):
        if not isinstance(row, Mapping):
            source_errors.append(
                {
                    "sample_idx": sample_idx,
                    "sample_id": f"{split_tag}:{sample_idx}",
                    "question_id": sample_idx,
                    "split_tag": split_tag,
                    "answer_source_key": "source_row_not_object",
                    "question_source_key": "source_row_not_object",
                    "reason": "source_row_not_object",
                    "answer_preview": repr(row)[:120],
                    "retry_attempted": False,
                    "retry_success": False,
                }
            )
            continue
        answer_text, answer_key = _ask4conf_generated_answer(row)
        answer_reason = _ask4conf_answer_error(answer_text)
        question_text, question_key = _ask4conf_question_text(row, tokenizer)
        retry_attempted = False
        retry_success = False
        retry_error = ""
        if (
            answer_reason
            and retry_source_errors
            and model is not None
            and retry_generation_config is not None
            and question_text
        ):
            retry_attempted = True
            retry_stats["retried_rows"] += 1
            retry_answer, retry_error = _ask4conf_retry_generated_answer(
                row,
                model=model,
                tokenizer=tokenizer,
                device=device,
                generation_config=retry_generation_config,
            )
            if retry_answer:
                retry_success = True
                retry_stats["retry_success_rows"] += 1
                answer_text = retry_answer
                answer_key = "retry_generated_answer"
                answer_reason = ""
                if isinstance(row, dict):
                    row["most_likely_answer"] = retry_answer
                    row["ask4conf_answer_retry_status"] = "success"
                    row["ask4conf_answer_retry_reason"] = "placeholder_source_answer"
            elif isinstance(row, dict):
                row["ask4conf_answer_retry_status"] = "failed"
                row["ask4conf_answer_retry_error"] = retry_error
        if answer_reason or not question_text:
            reason = answer_reason or "missing_question_text"
            source_errors.append(
                _ask4conf_source_error_record(
                    row,
                    sample_idx=sample_idx,
                    split_tag=split_tag,
                    answer_source_key=answer_key,
                    question_source_key=question_key,
                    reason=retry_error or reason,
                    answer_preview=answer_text,
                    question_str=question_text,
                    retry_attempted=retry_attempted,
                    retry_success=retry_success,
                )
            )
            if isinstance(row, dict):
                row["ask4conf_status"] = "source_error"
                row["ask4conf_missing_reason"] = retry_error or reason
                row["ask4conf_confidence"] = float("nan")
            continue
        if isinstance(row, dict):
            row["ask4conf_status"] = "ready" if not retry_success else "retry_success"
            row.pop("ask4conf_missing_reason", None)
        prepared.append(
            {
                "sample_idx": sample_idx,
                "question_id": _ask4conf_question_id(row, sample_idx),
                "sample_id": _ask4conf_sample_id(row, sample_idx, split_tag),
                "question_str": question_text,
                "question_source_key": question_key,
                "answer_str": answer_text,
                "answer_source_key": answer_key,
                "answer_retry_status": "success" if retry_success else "not_needed",
            }
        )
    return prepared, source_errors, retry_stats


def _ask4conf_confidence_histogram(values: Sequence[float]) -> dict:
    bins = Counter()
    for value in values:
        if value < 0.2:
            bins["[0.0,0.2)"] += 1
        elif value < 0.4:
            bins["[0.2,0.4)"] += 1
        elif value < 0.6:
            bins["[0.4,0.6)"] += 1
        elif value < 0.8:
            bins["[0.6,0.8)"] += 1
        elif value < 1.0:
            bins["[0.8,1.0)"] += 1
        else:
            bins["[1.0,1.0]"] += 1
    return dict(bins)


def generate_ask4conf(
    model_type,
    dataset_name,
    debug_limit: Optional[int] = None,
    *,
    source_error_policy: Optional[str] = None,
    source_error_threshold: float = ASK4CONF_MAX_SOURCE_ERROR_RATE,
):
    """Hardened ask4conf stage of build-response-cache.

    * Strict prompt/output protocol (single float in [0, 1]).
    * Always passes ``attention_mask`` and an explicit ``pad_token_id``.
    * Deterministic decoding (``do_sample=False``).
        * Reads candidate answers from the canonical ``*_mextend.json`` response
            cache; never from formatter ``answer_str`` placeholders.
        * Per-row audit fields (``parse_status``, ``parsed_confidence``,
            ``raw_generation_text``, ``error_reason``, ``error_type``,
            ``sample_idx``).
    * Writes a sidecar ``<split_tag>.meta.json`` and only then writes
      the ``SUCCESSFUL__<split_tag>`` marker after re-reading the jsonl
      and verifying ``line_count == written_count``.
    * Auto-resumes safely: if a previous shard is complete and matches
      the current protocol version it is skipped; otherwise the broken
      shard (and its stale marker) is removed and re-generated.
    """
    resolved_source_error_policy = _ask4conf_resolve_source_error_policy(
        source_error_policy,
        debug_limit=debug_limit,
    )
    output_dir = Path(_TEST_OUTPUT_ROOT) / "ask4conf" / model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "[response-cache] stage=ask4conf dataset={} model={} output_dir={} "
        "protocol_version={} debug_limit={} source_error_policy={} source_error_threshold={:.4f}",
        dataset_name, model_type, output_dir, ASK4CONF_PROTOCOL_VERSION,
        debug_limit, resolved_source_error_policy, float(source_error_threshold),
    )

    if debug_limit is not None and debug_limit <= 0:
        raise ValueError("debug_limit must be positive when provided")

    split_tags = _ask4conf_source_split_tags(dataset_name, model_type)
    pending: List[Tuple[str, List[dict], dict, Path, bool]] = []
    for split_tag in split_tags:
        rows, source_path = _ask4conf_load_response_cache(split_tag, model_type)
        source_summary = _ask4conf_source_summary(
            rows,
            split_tag,
            source_path,
            model_type=model_type,
            source_error_policy=resolved_source_error_policy,
            source_error_threshold=float(source_error_threshold),
        )
        process_rows = rows[:debug_limit] if debug_limit is not None else rows
        process_count = len(process_rows)
        if debug_limit is not None and process_count < len(rows):
            logger.warning(
                "[response-cache] stage=ask4conf split_tag={} model={} "
                "explicit_debug_limit={} source_count={} process_count={}",
                split_tag, model_type, debug_limit, len(rows), process_count,
            )
            source_summary = _ask4conf_source_summary(
                process_rows,
                split_tag,
                source_path,
                model_type=model_type,
                source_error_policy=resolved_source_error_policy,
                source_error_threshold=float(source_error_threshold),
                write_validation_report=False,
            )
        expected_count = int(source_summary["loaded_count"])
        source_failed_count_for_validation: Optional[int] = int(source_summary["source_error_count"])
        if resolved_source_error_policy == "retry_then_skip" and source_failed_count_for_validation > 0:
            source_failed_count_for_validation = None

        success_marker = ask4conf_success_marker(split_tag, model_type)
        meta_path = ask4conf_metadata_path(split_tag, model_type)
        jsonl_out = ask4conf_jsonl_path(split_tag, model_type)
        source_errors_path = _ask4conf_source_errors_path(split_tag, model_type)

        ok, reason = _ask4conf_validate_existing(
            jsonl_out,
            meta_path,
            success_marker,
            source_errors_path,
            protocol_version=ASK4CONF_PROTOCOL_VERSION,
            expected_count=expected_count,
            source_count=expected_count,
            source_failed_count=source_failed_count_for_validation,
            source_error_policy=resolved_source_error_policy,
            source_error_threshold=float(source_error_threshold),
        )
        if ok:
            logger.info(
                "[response-cache] stage=ask4conf split_tag={} model={} "
                "skip=already_valid expected_count={} output={}",
                split_tag, model_type, expected_count, jsonl_out,
            )
            continue

        if jsonl_out.exists() or meta_path.exists() or success_marker.exists():
            logger.warning(
                "[response-cache] stage=ask4conf split_tag={} model={} "
                "rerun reason={} (clearing stale shard)",
                split_tag, model_type, reason,
            )
            for stale in (
                jsonl_out,
                jsonl_out.with_suffix(jsonl_out.suffix + ".tmp"),
                meta_path,
                success_marker,
                source_errors_path,
                source_errors_path.with_suffix(source_errors_path.suffix + ".tmp"),
                _ask4conf_source_error_audit_json_path(split_tag, model_type),
                _ask4conf_source_error_audit_csv_path(split_tag, model_type),
            ):
                try:
                    stale.unlink()
                except FileNotFoundError:
                    pass
        pending.append((split_tag, process_rows, source_summary, source_path, debug_limit is None))

    if not pending:
        logger.info(
            "[response-cache] stage=ask4conf dataset={} model={} "
            "all_shards_valid",
            dataset_name, model_type,
        )
        return

    model, tokenizer = load_model_by_name(model_type)

    # Make sure we have a usable pad token. Many causal LMs (Llama,
    # Qwen, Gemma) ship without a pad token, so transformers falls back
    # to ``pad_token_id == eos_token_id`` and emits the well-known
    # "attention mask not set" warning. We set it explicitly here AND
    # always pass attention_mask below.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    device = next(model.parameters()).device

    # Deterministic decoding for the confidence query itself.
    confidence_gen_config = GenerationConfig(
        max_new_tokens=8,
        do_sample=False,
        temperature=1.0,  # ignored when do_sample=False; set to neutral
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )
    source_retry_gen_config = GenerationConfig(
        max_new_tokens=64,
        do_sample=False,
        temperature=1.0,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )

    for split_tag, rows, source_summary, source_path, persist_source_status in pending:
        logger.info(
            "[response-cache] stage=ask4conf split_tag={} model={} output={}",
            split_tag, model_type, ask4conf_jsonl_path(split_tag, model_type),
        )
        prepared_rows, source_errors, retry_stats = _ask4conf_prepare_rows(
            rows,
            split_tag,
            tokenizer,
            model=model,
            device=device,
            retry_source_errors=resolved_source_error_policy == "retry_then_skip",
            retry_generation_config=source_retry_gen_config,
        )
        source_failed_count = len(source_errors)
        source_error_rate = source_failed_count / max(len(rows), 1)
        if source_error_rate > source_error_threshold and not _ask4conf_policy_allows_skip(resolved_source_error_policy):
            raise ValueError(
                "[response-cache] stage=ask4conf source_error: "
                f"split_tag={split_tag} loaded={len(rows)} "
                f"source_failed={source_failed_count} "
                f"policy={resolved_source_error_policy} "
                f"threshold={source_error_threshold:.4f} "
                f"top_source_errors={source_errors[:5]}"
            )
        expected_count = len(rows)
        source_count = int(source_summary["loaded_count"])
        success_marker = ask4conf_success_marker(split_tag, model_type)
        meta_path = ask4conf_metadata_path(split_tag, model_type)
        jsonl_out = ask4conf_jsonl_path(split_tag, model_type)
        source_errors_path = _ask4conf_source_errors_path(split_tag, model_type)
        tmp_out = jsonl_out.with_suffix(jsonl_out.suffix + ".tmp")
        audit_paths = _ask4conf_write_source_errors(
            source_errors_path,
            source_errors,
            split_tag=split_tag,
            model_type=model_type,
        )
        if source_errors:
            logger.warning(
                "[response-cache] stage=ask4conf split_tag={} model={} "
                "source_failed_count={} source_error_rate={:.4f} policy={} "
                "source_error_audit_json={} source_error_audit_csv={} top_source_errors={}",
                split_tag, model_type, source_failed_count, source_error_rate,
                resolved_source_error_policy,
                audit_paths["json"], audit_paths["csv"], source_errors[:5],
            )

        attempted = 0
        written = 0
        generation_failed = 0
        parse_failed = 0
        confidences: List[float] = []

        with open(tmp_out, "w", encoding="utf-8") as fw:
            for item in tqdm(
                prepared_rows,
                total=len(prepared_rows),
                desc=f"ask4conf {split_tag}",
            ):
                attempted += 1
                sample_idx = int(item["sample_idx"])
                question_str = item["question_str"]
                answer_str = item["answer_str"]

                # ---- 1) ask-for-confidence prompt ------------------
                ask_prompt = _ask4conf_build_prompt(
                    tokenizer,
                    question_text=question_str,
                    answer_text=answer_str,
                )
                enc = tokenizer(
                    ask_prompt,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                ask_input_ids = enc["input_ids"].to(device)
                ask_attention_mask = enc.get("attention_mask")
                if ask_attention_mask is None:
                    ask_attention_mask = torch.ones_like(ask_input_ids)
                else:
                    ask_attention_mask = ask_attention_mask.to(device)
                prompt_len = ask_input_ids.shape[1]

                try:
                    with torch.no_grad():
                        prob_out = model.generate(
                            input_ids=ask_input_ids,
                            attention_mask=ask_attention_mask,
                            generation_config=confidence_gen_config,
                        )
                    raw_gen_ids = prob_out[0][prompt_len:]
                    raw_gen_text = tokenizer.decode(
                        raw_gen_ids, skip_special_tokens=True
                    )
                except Exception as exc:  # noqa: BLE001
                    generation_failed += 1
                    logger.warning(
                        "[response-cache] stage=ask4conf split_tag={} "
                        "sample_idx={} confidence_generation_failed: {!r}",
                        split_tag, sample_idx, exc,
                    )
                    raw_gen_text = ""
                    raw_gen_ids = None
                    conf = None
                    parse_status = "not_run"
                    err_reason = f"generation_error: {exc!r}"
                    error_type = "generation_error"
                    legacy_prob = 0.5
                else:
                    # ---- 2) strict parse ---------------------------
                    conf, parse_status, err_reason = _ask4conf_parse_confidence(
                        raw_gen_text
                    )
                    if parse_status != "ok":
                        parse_failed += 1
                        # Legacy ``prob`` field kept at 0.5 only as a derived
                        # numeric fallback so downstream consumers that index
                        # by row position do not break. The truth is in
                        # ``parse_status`` / ``parsed_confidence``.
                        legacy_prob = 0.5
                        error_type = "parse_error"
                    else:
                        legacy_prob = float(conf)
                        confidences.append(float(conf))
                        error_type = None

                row = {
                    "sample_idx": sample_idx,
                    "question_id": item["question_id"],
                    "question_str": question_str,
                    "question_source_key": item["question_source_key"],
                    "answer_str": answer_str,
                    "answer_source": "response_cache_mextend",
                    "answer_source_key": item["answer_source_key"],
                    "greedy_answer_tokens": tokenizer.encode(
                        answer_str, add_special_tokens=False
                    ),
                    "prob_answer_tokens": (
                        raw_gen_ids.tolist()
                        if isinstance(raw_gen_ids, torch.Tensor) else None
                    ),
                    "raw_generation_text": raw_gen_text,
                    "parse_status": parse_status,
                    "parsed_confidence": conf,
                    "error_type": error_type,
                    "error_reason": err_reason,
                    "prob": legacy_prob,
                    "protocol_version": ASK4CONF_PROTOCOL_VERSION,
                }
                if sample_idx < len(rows) and isinstance(rows[sample_idx], dict):
                    rows[sample_idx]["ask4conf_status"] = "ok" if error_type is None else str(error_type)
                    rows[sample_idx]["ask4conf_confidence"] = float(conf) if conf is not None else float("nan")
                    rows[sample_idx]["ask4conf_parse_status"] = parse_status
                    rows[sample_idx]["ask4conf_error_reason"] = err_reason
                fw.write(json.dumps(row, ensure_ascii=False))
                fw.write("\n")
                written += 1

        # ---- post-write integrity validation -----------------------
        with open(tmp_out, "r", encoding="utf-8") as fr:
            line_count = sum(1 for _ in fr)

        parse_failure_rate = parse_failed / max(written, 1)
        generation_error_rate = generation_failed / max(written, 1)
        placeholder_or_missing_count = int(
            source_summary.get("placeholder_or_missing_count", source_failed_count)
        )
        confidence_histogram = _ask4conf_confidence_histogram(confidences)
        unique_confidence_count = len(set(confidences))

        failures: List[str] = []
        if written + source_failed_count != expected_count:
            failures.append(
                "written_plus_source_failed_mismatch: "
                f"written={written} source_failed={source_failed_count} "
                f"expected={expected_count}"
            )
        if line_count != written:
            failures.append(f"line_count_mismatch: {line_count}!={written}")
        if source_error_rate > source_error_threshold and not _ask4conf_policy_allows_skip(resolved_source_error_policy):
            failures.append(
                "source_error_rate_too_high: "
                f"{source_error_rate:.4f}>{source_error_threshold:.4f}"
            )
        if generation_error_rate > ASK4CONF_MAX_GENERATION_ERROR_RATE:
            failures.append(
                "generation_error_rate_too_high: "
                f"{generation_error_rate:.4f}>"
                f"{ASK4CONF_MAX_GENERATION_ERROR_RATE:.4f}"
            )
        if parse_failure_rate > ASK4CONF_MAX_PARSE_FAILURE_RATE:
            failures.append(
                "parse_failure_rate_too_high: "
                f"{parse_failure_rate:.4f}>{ASK4CONF_MAX_PARSE_FAILURE_RATE:.4f}"
            )

        logger.info(
            "[response-cache] stage=ask4conf summary split_tag={} model={} "
            "expected_count={} written_count={} source_count={} "
            "source_failed_count={} placeholder_or_missing_answers={} "
            "policy={} retried_rows={} retry_success_rows={} skipped_ask4conf_rows={} generation_failed={} "
            "parse_failed={} parse_failure_rate={:.4f} "
            "unique_confidence_count={} confidence_histogram={} output={}",
            split_tag,
            model_type,
            expected_count,
            written,
            source_count,
            source_failed_count,
            placeholder_or_missing_count,
            resolved_source_error_policy,
            int(retry_stats.get("retried_rows", 0)),
            int(retry_stats.get("retry_success_rows", 0)),
            source_failed_count,
            generation_failed,
            parse_failed,
            parse_failure_rate,
            unique_confidence_count,
            confidence_histogram,
            jsonl_out,
        )

        tmp_out.replace(jsonl_out)
        if persist_source_status:
            source_path.write_text(
                json.dumps(rows, ensure_ascii=False),
                encoding="utf-8",
            )

        if failures:
            for stale in (meta_path, success_marker):
                try:
                    stale.unlink()
                except FileNotFoundError:
                    pass
            raise RuntimeError(
                "[response-cache] stage=ask4conf integrity_check_failed: "
                f"split_tag={split_tag} model={model_type} failures={failures} "
                f"output={jsonl_out}"
            )

        meta = {
            "protocol_version": ASK4CONF_PROTOCOL_VERSION,
            "dataset": split_dataset_and_raw(split_tag)[0],
            "split_tag": split_tag,
            "model": model_type,
            "source_path": source_summary["source_path"],
            "loaded_rows": expected_count,
            "source_count": source_count,
            "expected_count": expected_count,
            "valid_answer_rows": len(prepared_rows),
            "valid_answer_count": len(prepared_rows),
            "attempted_count": attempted,
            "written_count": written,
            "source_failed_count": source_failed_count,
            "source_error_count": source_failed_count,
            "placeholder_or_missing_count": placeholder_or_missing_count,
            "placeholder_or_missing_rows": placeholder_or_missing_count,
            "source_error_rate": source_error_rate,
            "source_error_policy": resolved_source_error_policy,
            "source_validation_threshold_used": float(source_error_threshold),
            "source_error_threshold": float(source_error_threshold),
            "skipped_ask4conf_rows": source_failed_count,
            "retried_rows": int(retry_stats.get("retried_rows", 0)),
            "retry_success_rows": int(retry_stats.get("retry_success_rows", 0)),
            "source_errors_path": str(source_errors_path),
            "source_error_audit_json": audit_paths["json"],
            "source_error_audit_csv": audit_paths["csv"],
            "mextend_validation_path": source_summary.get("mextend_validation_path"),
            "generation_failed_count": generation_failed,
            "generation_error_rate": generation_error_rate,
            "parse_failed_count": parse_failed,
            "parse_failure_rate": parse_failure_rate,
            "unique_confidence_count": unique_confidence_count,
            "confidence_histogram": confidence_histogram,
            "line_count": line_count,
            "output_path": str(jsonl_out),
            "debug_limit": debug_limit,
        }
        meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        success_marker.write_text("successful", encoding="utf-8")

        logger.info(
            "[response-cache] stage=ask4conf success split_tag={} model={} "
            "expected_count={} written_count={} meta={} success_marker={}",
            split_tag, model_type, expected_count, written, meta_path,
            success_marker,
        )


def generate_uncertainty_score(model_type, dataset_name):
    split_tag = dataset_name
    base_dataset, raw_split = split_dataset_and_raw(split_tag)
    GENERATED_QA_LOCAL = extend_path(split_tag, model_type)
    QUESTION_KEY = "question_str"  # string
    ANSWERS_KEY = "generated_answers"  # list[list[str]]
    SEMANTIC_ENTROPY_KEY = "semantic_entropy"
    save_path = semantic_entropy_path(split_tag, model_type)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # --- resolve the auxiliary NLI model.
    # ``transformers.AutoModel*.from_pretrained`` accepts EITHER an HF repo
    # id OR a local directory path, but it MUST be one or the other --
    # passing a local filesystem path that doesn't exist makes it fall
    # back to a Hub lookup with a confusing error. We pick exactly one
    # and log which mode we used.
    from registry.reference_model_registry import (
        format_reference_model_setup_hint,
        get_reference_model_spec,
        reference_model_is_available_locally,
        reference_model_local_dir,
    )
    _REF_NAME = "deberta-large-mnli"
    _ref_spec = get_reference_model_spec(_REF_NAME)
    _ref_local = reference_model_local_dir(_REF_NAME)
    _ref_local_exists = reference_model_is_available_locally(_REF_NAME)
    _ref_load_target = (
        str(_ref_local) if _ref_local_exists else _ref_spec.hf_repo_id
    )
    _ref_load_mode = "local" if _ref_local_exists else "remote"
    logger.info(
        "[response-cache] stage=semantic_entropy auxiliary_model={} "
        "hf_repo_id={} resolved_local_path={} local_exists={} load_mode={}",
        _ref_spec.canonical_name, _ref_spec.hf_repo_id,
        _ref_local, _ref_local_exists, _ref_load_mode,
    )
    logger.info(
        "[response-cache] stage=semantic_entropy dataset={} raw_split={} "
        "split_tag={} model={} input={} output={}",
        base_dataset, raw_split, split_tag, model_type,
        GENERATED_QA_LOCAL, save_path,
    )
    _require_input(
        "semantic_entropy", GENERATED_QA_LOCAL,
        dataset=base_dataset, split_tag=split_tag,
    )

    try:
        entailment_tokenizer = AutoTokenizer.from_pretrained(
            _ref_load_target,
            local_files_only=_ref_local_exists,
        )
        entailment_model = AutoModelForSequenceClassification.from_pretrained(
            _ref_load_target,
            local_files_only=_ref_local_exists,
        ).cuda()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "[response-cache] stage=semantic_entropy failed to load "
            f"auxiliary model. {format_reference_model_setup_hint(_REF_NAME)} "
            f"underlying error: {exc!r}"
        ) from exc

    with open(GENERATED_QA_LOCAL, "r") as f:
        data_with_answers = json.load(f)

    if save_path.exists():
        with open(save_path, "r") as f:
            data_with_score = json.load(f)
    else:
        data_with_score = data_with_answers

    for ridx in tqdm(range(len(data_with_answers))):
        row = data_with_score[ridx]
        if SEMANTIC_ENTROPY_KEY in row:
            continue
        if ANSWERS_KEY not in row or row[ANSWERS_KEY] == []:
            # check if there is also no answers in data_with_answers
            if (
                ANSWERS_KEY not in data_with_answers[ridx]
                or data_with_answers[ridx][ANSWERS_KEY] == []
            ):
                continue
            else:
                # check if they are the same question
                if row[QUESTION_KEY] != data_with_answers[ridx][QUESTION_KEY]:
                    logger.warning(f"Not the same question in row {ridx}")
                    break
                else:
                    row[ANSWERS_KEY] = data_with_answers[ridx][ANSWERS_KEY]

        question = row[QUESTION_KEY]

        try:
            answers = sum(row[ANSWERS_KEY], [])  # flatten the list
        except TypeError:
            answers = row[ANSWERS_KEY]

        # use only unique answers - follow semantic entropy implementation
        answers_set = list(set(answers))
        num_answers = len(answers_set)

        alist1, alist2, entailment_prompts = [], [], []

        # records answer and its semantic cluster - used for semantic entropy
        ans2smt = {answer: i for i, answer in enumerate(answers_set)}

        if num_answers == 1:
            row[SEMANTIC_ENTROPY_KEY] = 0
        else:
            for i, ref_answer in enumerate(answers_set):
                for j in range(i + 1, len(answers_set)):
                    alist1.append(answers_set[i])
                    alist2.append(answers_set[j])

                    qa_1 = question + " " + answers[i]
                    qa_2 = question + " " + answers[j]

                    # not sure, but this seperator is used in semantic uncertainty
                    entailment_prompt = qa_1 + "[SEP]" + qa_2
                    entailment_prompts.append(entailment_prompt)

                    # here we just follow semantic uncertainty
                    encoded_prompt = entailment_tokenizer.encode(
                        entailment_prompt, padding=True
                    )
                    pred = entailment_model(
                        # torch.tensor(
                        #     torch.tensor([encoded_prompt]),
                        #     device="cuda"
                        # )
                        torch.tensor([encoded_prompt], device="cuda")
                    )["logits"]
                    pred_label = torch.argmax(pred, dim=1)

                    reversed_prompt = qa_2 + "[SEP]" + qa_1
                    encoded_reversed_prompt = entailment_tokenizer.encode(
                        reversed_prompt, padding=True
                    )
                    reversed_pred = entailment_model(
                        # torch.tensor(
                        #     torch.tensor([encoded_reversed_prompt]),
                        #     device="cuda"
                        # )
                        torch.tensor([encoded_reversed_prompt], device="cuda")
                    )["logits"]
                    reversed_pred_label = torch.argmax(reversed_pred, dim=1)

                    if 0 in pred_label or 0 in reversed_pred_label:
                        pass  # semantically different, do nothing
                    else:  # semantically same, merge clusters
                        ans2smt[answers_set[j]] = ans2smt[answers_set[i]]

            semantic_group = list(ans2smt.values())
            group_of_answer = [ans2smt[answer] for answer in answers]
            semantic_group_set = set(semantic_group)

            # calculate the number of samples in each cluster
            num_samples_in_cluster = [
                group_of_answer.count(group_idx)
                for group_idx in semantic_group_set
            ]

            N = num_answers

            semantic_entropy = (
                -1
                / len(semantic_group_set)
                * sum(
                    [
                        np.log(num_sample / N)
                        for num_sample in num_samples_in_cluster
                    ]
                )
            )
            row[SEMANTIC_ENTROPY_KEY] = semantic_entropy

        # save the data
        if (ridx + 1) % 500 == 0:
            with open(save_path, "w") as f:
                json.dump(data_with_score, f)


def generate_query_X_mmlu(model_type, phase, mduq_mode: bool = False):
    output_dir = _TEST_OUTPUT_ROOT

    if model_type.startswith("gemma"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    model_path, tokenizer_path = get_model_paths(model_type)

    model, tokenizer = load_llama2(model_path, tokenizer_path)

    # raise error if phase=="train":
    if phase == "train":
        raise ValueError("The phase cannot be train")

    if mduq_mode:
        hidden_state_output_dir = (
            output_dir
            + "/MMLU/"
            + model_type
            + "/diaguq/hidden_bank/"
            + phase
            + "/"
        )
    else:
        hidden_state_output_dir = (
            output_dir + "/MMLU/" + model_type + "/" + phase + "/"
        )

    PROMPT_TOKENS = "tokenized_prompt"
    Q_BEGIN = "question_token_start_idx"
    Q_END = "answer_token_start_idx"

    data_tasks = [
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

    output_token_average_hidden_states = True
    len_of_token_hidden_states_output = 1  # if set to zero, then not used
    get_query_entropies = True  # whether to get the entropy of the output token

    num_entropy_statistics = 4
    num_letters = 4

    data_total = mmlu_formatter(
        tokenizer=tokenizer,
        num_example=5,
        merge_split=False,
        conv_generation=True,
    )

    # if the path not exists, then create the path
    if not os.path.exists(hidden_state_output_dir):
        os.makedirs(hidden_state_output_dir)

    layer_list, num_dim = get_layer_list_and_dim(model_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for task in tqdm(data_tasks):
            dataset_name = "mmlu__" + task + "__" + phase
            task_output_dir = hidden_state_output_dir + task + "/"
            if not os.path.exists(task_output_dir):
                os.makedirs(task_output_dir)
            if os.path.exists(task_output_dir + "query_logits.pt"):
                continue
            data = data_total[dataset_name]

            num_queries = len(data)

            print("queries to be processed: ", num_queries)

            # initialize output_tensor as num_layers x num_queries x num_dim
            if output_token_average_hidden_states:
                output_average_tensor = torch.zeros(
                    (num_queries, len(layer_list), num_dim), dtype=torch.float16
                )
            if len_of_token_hidden_states_output > 0:
                output_last_token_tensor = torch.zeros(
                    (
                        num_queries,
                        len(layer_list),
                        len_of_token_hidden_states_output,
                        num_dim,
                    ),
                    dtype=torch.float16,
                )
            if get_query_entropies:
                entropy_output_tensor = torch.zeros(
                    (num_queries, num_entropy_statistics), dtype=torch.float16
                )

            logits_output_tensor = torch.zeros(
                (num_queries, num_letters), dtype=torch.float16
            )
            letter_tokens = [
                tokenizer.encode(letter)[1] for letter in ["A", "B", "C", "D"]
            ]

            # forward and get features of the query
            for data_i, d in tqdm(enumerate(data)):

                q_begin = d[Q_BEGIN]
                q_end = d[Q_END]
                prompt_token = d[PROMPT_TOKENS][:q_end]

                # convert prompt_token to tensor
                prompt_token = torch.tensor(prompt_token).unsqueeze(0)
                prompt_token = prompt_token.to(device)

                outputs = model.forward(prompt_token, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                logits = outputs.logits
                logits_output_tensor[data_i, :] = torch.tensor(
                    [logits[0, -1, token_idx] for token_idx in letter_tokens],
                    dtype=torch.float16,
                )

            if output_token_average_hidden_states:
                output_average_tensor[data_i] = get_average_hidden_states(
                    hidden_states, layer_list, q_begin, q_end, num_dim=num_dim
                )
            if len_of_token_hidden_states_output > 0:
                output_last_token_tensor[data_i] = get_last_token_hidden_states(
                    hidden_states,
                    layer_list,
                    q_end,
                    len_of_token_hidden_states_output,
                    num_dim=num_dim,
                )

            if get_query_entropies:
                entropy_output_tensor[data_i, :] = get_entropy_statistics(
                    outputs.logits, q_begin, q_end
                )

            # save the hidden_states output
            for idx, layer_idx in enumerate(layer_list):
                if output_token_average_hidden_states:
                    torch.save(
                        output_average_tensor[:, idx, :],
                        task_output_dir
                        + "query_average_layer_"
                        + str(layer_idx)
                        + ".pt",
                    )
                if len_of_token_hidden_states_output > 0:
                    torch.save(
                        output_last_token_tensor[:, idx, :, :],
                        task_output_dir
                        + "query_last_"
                        + str(len_of_token_hidden_states_output)
                        + "_token_layer_"
                        + str(layer_idx)
                        + ".pt",
                    )

            # release the memory
            if output_token_average_hidden_states:
                del output_average_tensor
            if len_of_token_hidden_states_output > 0:
                del output_last_token_tensor

            # save the entropy output
            if get_query_entropies:
                torch.save(
                    entropy_output_tensor,
                    task_output_dir + "query_entropies.pt",
                )
                # release the memory
                del entropy_output_tensor

            # save the logits output
            torch.save(
                logits_output_tensor, task_output_dir + "query_logits.pt"
            )


def generate_answer_X_mmlu(model_type, phase, mduq_mode: bool = False):
    output_dir = _TEST_OUTPUT_ROOT

    if model_type.startswith("gemma"):
        os.environ["CUDA_VISIBLE_DEVICES"] = "3, 2"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    model_path, tokenizer_path = get_model_paths(model_type)

    model, tokenizer = load_llama2(model_path, tokenizer_path)

    # raise error if phase=="train":
    if phase == "train":
        raise ValueError("The phase cannot be train")

    if mduq_mode:
        hidden_state_output_dir = (
            output_dir
            + "/MMLU/"
            + model_type
            + "/diaguq/hidden_bank/"
            + phase
            + "/"
        )
    else:
        hidden_state_output_dir = (
            output_dir + "/MMLU/" + model_type + "/" + phase + "/"
        )

    PROMPT_TOKENS = "tokenized_prompt"

    Q_END = "answer_token_start_idx"

    data_tasks = [
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

    layer_list, num_dim = get_layer_list_and_dim(model_type)

    data_total = mmlu_formatter(
        tokenizer=tokenizer,
        num_example=5,
        merge_split=False,
        conv_generation=True,
    )

    # if the path not exists, then create the path
    if not os.path.exists(hidden_state_output_dir):
        os.makedirs(hidden_state_output_dir)

    with torch.no_grad():
        generator = MMLUGenerator(model, tokenizer, layer_list, num_dim)

        for task in tqdm(data_tasks):
            dataset_name = "mmlu__" + task + "__" + phase
            task_output_dir = hidden_state_output_dir + task + "/"
            if not os.path.exists(task_output_dir):
                os.makedirs(task_output_dir)
            if os.path.exists(
                task_output_dir + str(layer_list[0]) + "_output_answer_X.pt"
            ):
                continue
            data = data_total[dataset_name]
            print(len(data))
            num_tokens = 4
            output_answer_X = torch.zeros(
                (len(data), num_tokens, len(layer_list), num_dim)
            )

            data = list(data)
            for i in tqdm(range(0, len(data))):
                d = data[i]
                prompt_tokens = d[PROMPT_TOKENS][: d[Q_END]]
                output_answer_X[i] = generator.generate_single(prompt_tokens)

            # save the result

            for idx, layer_idx in enumerate(layer_list):
                torch.save(
                    output_answer_X[:, :, idx, :],
                    task_output_dir + str(layer_idx) + "_output_answer_X.pt",
                )
