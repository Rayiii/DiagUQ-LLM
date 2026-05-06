import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


ENTROPY_STAT_NAMES = (
    "entropy_max",
    "entropy_min",
    "entropy_mean",
    "entropy_std",
)
PROB_STAT_NAMES = (
    "neg_prob_max",
    "neg_prob_min",
    "neg_prob_mean",
    "neg_prob_std",
    "neg_log_prob_mean",
    "neg_log_prob_std",
)


def _valid_token_span(q_begin, q_end, token_len):
    begin = max(0, min(int(q_begin), int(token_len) - 1))
    end = max(begin + 1, min(int(q_end), int(token_len)))
    return begin, end


def _checked_token_span(q_begin, q_end, token_len, span_label):
    try:
        begin = int(q_begin)
        end = int(q_end)
        token_len = int(token_len)
    except (TypeError, ValueError):
        return None, None, f"malformed_{span_label}_span"
    if token_len <= 0:
        return None, None, "empty_logits_sequence"
    if end <= begin:
        return None, None, f"empty_{span_label}_span"
    if end <= 0 or begin >= token_len:
        return None, None, f"out_of_bounds_{span_label}_span"
    begin = max(0, begin)
    end = min(token_len, end)
    if end <= begin:
        return None, None, f"empty_{span_label}_span"
    return begin, end, None


def _stats_result(values, available, reason, return_metadata):
    values = values.detach().to("cpu").float()
    if return_metadata:
        return values, bool(available), reason
    return values


def _missing_stats(length, device, reason, return_metadata):
    values = torch.zeros(length, dtype=torch.float32, device=device)
    return _stats_result(values, False, reason, return_metadata)


def get_average_hidden_states(
    hidden_states, layer_list, q_begin, q_end, num_dim=4096
):
    """
    Get the average hidden states of the query.
    Inputs:
    - hidden_states: the hidden_states of the query, shape: (num_hidden_layers,(batch_size, token_len, layer_dim))
    - layer_list: the list of layers to be used
    - q_begin: the beginning index of the calculated sequence
    - q_end: the ending index of the calculated sequence
    - num_dim: the unique(consistent) dimension of the hidden states
    """
    q_begin, q_end = _valid_token_span(
        q_begin, q_end, hidden_states[0].shape[1]
    )

    result = torch.zeros(
        (hidden_states[0].shape[0], len(layer_list), num_dim),
        dtype=torch.float16,
    )
    for idx, layer_idx in enumerate(layer_list):
        result[:, idx, :] = torch.mean(
            hidden_states[layer_idx][:, q_begin:q_end, :], dim=1
        )
    return result


def get_last_token_hidden_states(
    hidden_states,
    layer_list,
    q_end,
    len_of_token_hidden_states_output,
    num_dim=4096,
):
    """
    Get the hidden states of the last token of the query.
    Inputs:
    - hidden_states: the hidden_states of the query, shape: (num_hidden_layers,(batch_size, token_len, hidden_size))
    - layer_list: the list of layers to be used
    - q_begin: the beginning index of the calculated sequence
    - q_end: the ending index of the calculated sequence
    - len_of_token_hidden_states_output: the number of hidden states of the last token to be output
    - num_dim: the unique(consistent) dimension of the hidden states
    """
    token_len = hidden_states[0].shape[1]
    q_end = max(1, min(int(q_end), int(token_len)))
    start = max(0, q_end - int(len_of_token_hidden_states_output))
    actual_len = q_end - start
    result = torch.zeros(
        (
            hidden_states[0].shape[0],
            len(layer_list),
            len_of_token_hidden_states_output,
            num_dim,
        ),
        dtype=torch.float16,
    )
    for idx, layer_idx in enumerate(layer_list):
        result[:, idx, -actual_len:, :] = hidden_states[layer_idx][
            :, start:q_end, :
        ]
    return result


def get_prob_statistics(
    logits,
    tokens,
    q_begin,
    q_end,
    query=True,
    *,
    span_label="token",
    return_metadata=False,
):
    tokens = tokens.squeeze().reshape(-1)
    begin, end, reason = _checked_token_span(
        q_begin, q_end, logits.shape[1], span_label
    )
    if reason is not None:
        return _missing_stats(len(PROB_STAT_NAMES), logits.device, reason, return_metadata)

    selected_logits = logits[:, begin:end, :].float()
    if selected_logits.numel() == 0:
        return _missing_stats(
            len(PROB_STAT_NAMES), logits.device, f"empty_{span_label}_span", return_metadata
        )
    if not torch.isfinite(selected_logits).all():
        return _missing_stats(
            len(PROB_STAT_NAMES), logits.device, "non_finite_logits", return_metadata
        )
    log_probs = F.log_softmax(selected_logits, dim=-1)
    if not torch.isfinite(log_probs).all():
        return _missing_stats(
            len(PROB_STAT_NAMES), logits.device, "non_finite_log_probs", return_metadata
        )
    probs = log_probs.exp().squeeze(0)
    log_probs = log_probs.squeeze(0)
    if probs.dim() < 2:
        probs = probs.unsqueeze(0)
        log_probs = log_probs.unsqueeze(0)

    next_token = torch.argmax(probs[-1, :])
    max_observed_steps = min(
        int(probs.shape[0]) - 1,
        max(0, int(tokens.numel()) - begin - 1),
    )
    selected_probs = []
    selected_log_probs = []
    if max_observed_steps > 0:
        vocab_size = int(probs.shape[-1])
        for i in range(max_observed_steps):
            token_idx = int(tokens[i + begin + 1].item())
            if token_idx < 0 or token_idx >= vocab_size:
                return _missing_stats(
                    len(PROB_STAT_NAMES),
                    logits.device,
                    "token_index_out_of_bounds",
                    return_metadata,
                )
            selected_probs.append(probs[i, token_idx])
            selected_log_probs.append(log_probs[i, token_idx])
    selected_probs.append(probs[-1, next_token])
    selected_log_probs.append(log_probs[-1, next_token])

    probs = torch.stack(selected_probs).float()
    token_log_probs = torch.stack(selected_log_probs).float()
    probs_max = torch.max(-probs)
    probs_min = torch.min(-probs)
    probs_mean = torch.mean(-probs)
    probs_std = torch.std(-probs, unbiased=False)
    probs_log_mean = torch.mean(-token_log_probs)
    probs_log_std = torch.std(-token_log_probs, unbiased=False)

    result = torch.stack(
        [
            probs_max,
            probs_min,
            probs_mean,
            probs_std,
            probs_log_mean,
            probs_log_std,
        ],
        dim=0,
    )

    if not torch.isfinite(result).all():
        return _missing_stats(
            len(PROB_STAT_NAMES), logits.device, "non_finite_probability_result", return_metadata
        )
    return _stats_result(result, True, None, return_metadata)


def get_entropy_statistics(
    logits,
    q_begin,
    q_end,
    query=True,
    *,
    span_label="token",
    return_metadata=False,
):
    """
    Get the entropy statistics of the output token.
    Inputs:
    - logits: the logits of the output token, shape: (batch_size, token_len, token_len)
    - q_begin: the beginning index of the calculated sequence
    - q_end: the ending index of the calculated sequence
    """
    begin, end, reason = _checked_token_span(
        q_begin, q_end, logits.shape[1], span_label
    )
    if reason is not None:
        return _missing_stats(len(ENTROPY_STAT_NAMES), logits.device, reason, return_metadata)

    selected_logits = logits[:, begin:end, :].float()
    if selected_logits.numel() == 0:
        return _missing_stats(
            len(ENTROPY_STAT_NAMES), logits.device, f"empty_{span_label}_span", return_metadata
        )
    if not torch.isfinite(selected_logits).all():
        return _missing_stats(
            len(ENTROPY_STAT_NAMES), logits.device, "non_finite_logits", return_metadata
        )
    log_probs = F.log_softmax(selected_logits, dim=-1)
    if not torch.isfinite(log_probs).all():
        return _missing_stats(
            len(ENTROPY_STAT_NAMES), logits.device, "non_finite_log_probs", return_metadata
        )
    probs = log_probs.exp()
    entropy = -torch.sum(probs * log_probs, dim=2)
    entropy_max = entropy.max(dim=1).values
    entropy_min = entropy.min(dim=1).values
    entropy_mean = entropy.mean(dim=1)
    entropy_std = entropy.std(dim=1, unbiased=False)
    result = torch.stack(
        [entropy_max, entropy_min, entropy_mean, entropy_std], dim=1
    )
    if result.shape[0] == 1:
        result = result.squeeze(0)
    if not torch.isfinite(result).all():
        return _missing_stats(
            len(ENTROPY_STAT_NAMES), logits.device, "non_finite_entropy_result", return_metadata
        )
    return _stats_result(result, True, None, return_metadata)


# ---------------------------------------------------------------------------
# MDUQ multi-layer hidden-bank IO
# ---------------------------------------------------------------------------

# Default sub-folder under ``./test_output/<dataset>/<model>/`` where the
# multi-layer hidden bank lives. Keeping it as a module-level constant so
# downstream code (training, analysis) can import it instead of hard-coding
# the path.
# Canonical (DiagUQ) subdirectory under ``./test_output/<dataset>/<model>/``.
# The historical name was ``mduq/hidden_bank`` -- old directories are still
# read transparently by the multi-layer feature-bank loader.
MDUQ_HIDDEN_BANK_SUBDIR = "diaguq/hidden_bank"
LEGACY_HIDDEN_BANK_SUBDIR = "mduq/hidden_bank"


def mduq_hidden_bank_dir(
    dataset_name: str,
    model_name: str,
    output_root: Optional[str] = None,
) -> str:
    """Return the directory used to store the multi-layer hidden bank.

    For MMLU pass ``dataset_name="MMLU/<phase>"`` to keep the existing
    ``phase`` subdivision; for QA / WMT / AmbigQA / TruthfulQA the bare
    dataset name is enough.

    When ``output_root`` is ``None`` it resolves via
    :func:`common.runtime_paths.get_test_output_dir`.
    """
    from common.artifact_locator import locate_hidden_bank_dir

    return str(locate_hidden_bank_dir(dataset_name, model_name, output_root))


def hidden_bank_filename(view: str, layer_idx: int, kind: str) -> str:
    """Canonical filename for one entry in the multi-layer hidden bank.

    ``view``  : ``"query"`` or ``"answer"``
    ``kind``  : ``"average"`` or ``"last_1_token"``
    """
    if view not in {"query", "answer"}:
        raise ValueError(f"view must be 'query' or 'answer', got {view!r}")
    if kind not in {"average", "last_1_token"}:
        raise ValueError(
            f"kind must be 'average' or 'last_1_token', got {kind!r}"
        )
    return f"{view}_{kind}_layer_{layer_idx}.pt"


def save_hidden_bank(
    bank_dir: str,
    layer_list,
    *,
    query_average=None,
    query_last_token=None,
    answer_average=None,
    answer_last_token=None,
    extras: dict | None = None,
) -> None:
    """Save the multi-layer hidden bank with the canonical naming.

    Each ``*_average`` tensor is expected with shape ``(N, L, D)`` and each
    ``*_last_token`` tensor with shape ``(N, L, T, D)`` where ``L =
    len(layer_list)``. ``extras`` is a flat ``{name: tensor}`` mapping for
    side-channel stats such as ``query_entropies.pt`` / ``query_probs.pt``
    that should live alongside the bank.
    """
    Path(bank_dir).mkdir(parents=True, exist_ok=True)
    for idx, layer_idx in enumerate(layer_list):
        if query_average is not None:
            torch.save(
                query_average[:, idx, :],
                os.path.join(
                    bank_dir, hidden_bank_filename("query", layer_idx, "average")
                ),
            )
        if query_last_token is not None:
            torch.save(
                query_last_token[:, idx, :, :],
                os.path.join(
                    bank_dir,
                    hidden_bank_filename(
                        "query", layer_idx, "last_1_token"
                    ),
                ),
            )
        if answer_average is not None:
            torch.save(
                answer_average[:, idx, :],
                os.path.join(
                    bank_dir,
                    hidden_bank_filename("answer", layer_idx, "average"),
                ),
            )
        if answer_last_token is not None:
            torch.save(
                answer_last_token[:, idx, :, :],
                os.path.join(
                    bank_dir,
                    hidden_bank_filename(
                        "answer", layer_idx, "last_1_token"
                    ),
                ),
            )
    if extras:
        for name, tensor in extras.items():
            torch.save(tensor, os.path.join(bank_dir, name))
