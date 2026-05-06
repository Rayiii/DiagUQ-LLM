from __future__ import annotations

from itertools import islice
import re
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

import datasets
import pandas as pd
import transformers
from loguru import logger
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# Canonical artifact roots come from common.runtime_paths so that
# ``data/`` (the source-code package containing this file) is never
# treated as a dataset storage root. See common/runtime_paths.py for
# the full namespace policy.
from common.runtime_paths import get_cache_dir, get_data_dir
from common.dataset_variants import DatasetVariant, parse_dataset_variant
from common.response_cache_limits import resolve_response_cache_limit
from common.single_split_policy import (
    DEFAULT_INTERNAL_SPLIT_SEED,
    DEFAULT_INTERNAL_TRAIN_RATIO,
    parse_internal_split_variant,
    row_belongs_to_internal_split,
)
from data.mmlu_loader import (
    MMLU_LOCAL_SPLITS,
    discover_mmlu_subjects,
    load_mmlu_subject_split,
    mmlu_actual_split,
    normalize_mmlu_row,
)
from registry.dataset_registry import get_dataset_spec, get_split_names

_DATA_ROOT = get_data_dir()
_CACHE_ROOT = get_cache_dir()

COQA_LOCAL = str(_DATA_ROOT / "coqa")
TRIVIA_LOCAL = str(_DATA_ROOT / "trivia_qa")
MMLU_LOCAL = str(_DATA_ROOT / "mmlu")
CNN_LOCAL = str(_DATA_ROOT / "cnn_dailymail")
WMT_LOCAL = str(_DATA_ROOT / "wmt14")
WEBGPT_LOCAL = str(_DATA_ROOT / "webgpt_comparisons")
AMBIGQA_LOCAL = str(_DATA_ROOT / "ambig_qa")
TRUTHFULQA_LOCAL = str(_DATA_ROOT / "truthful_qa")

CACHE_LOCAL = str(_CACHE_ROOT / "data_cache")

ENTER_PAT = re.compile(r"\n")
WMT_FORMATTER_VERSION = "limitaware_v2"
TRUTHFULQA_FORMATTER_VERSION = "fullrows_v2"
FULL_FORMATTING_GUARD_RAW_EXAMPLES = 100_000


def normalize_text(text):
    # Remove space before punctuation
    text = re.sub(r"\s+([.,;?!:])", r"\1", text)

    # Fix spacing after punctuation if missing
    text = re.sub(r"([.,;?!:])([^\s])", r"\1 \2", text)

    return text


def coqa_formatter_hf(
    tokenizer: transformers.PreTrainedTokenizer,
    dpath: str = COQA_LOCAL,
    num_example: int = 3,
    cache: bool = True,
) -> datasets.DatasetDict:
    step_size = 1 + num_example
    dd = datasets.load_from_disk(dpath)
    merged_datasets = {}

    caching_path = str(
        Path(CACHE_LOCAL)
        / f"coqa_{tokenizer.__class__.__name__}_exmp{num_example}"
    )

    if cache:
        if Path(caching_path).exists():
            logger.info(f"Loading cached dataset from {caching_path}")
            try:
                merged_datasetdict = datasets.load_from_disk(caching_path)
                return merged_datasetdict
            except:
                logger.warning(
                    f"Failed to load cached dataset from {caching_path}, need regeneration"
                )

    # here we manually add "id" column to the dataset based on the "story"
    dd = dd.map(lambda x: {"id": sha256(x["story"].encode()).hexdigest()})

    for ds_key, ds in dd.items():
        merged_datasets[ds_key] = []

        batch_cache = []
        batch_id = None

        for ditem in tqdm(ds, desc=f"Formatting {ds_key} dataset"):
            if len(ditem["questions"]) < num_example + 1:
                logger.debug(
                    f"Skipping {ditem['id']} with {len(ditem['questions'])} questions, need at least {num_example+1}"
                )
                continue
            for i in range(len(ditem["questions"]) - num_example):
                chunk_low, chunk_high = i, i + num_example
                story_str = (
                    f"Reading the passage and answer given questions accordingly.\n\nPassage:\n{ditem['story']}\n\n"
                    f"Examples:\n"
                    + "\n".join(
                        [
                            f"Q: {question}\nA: {answer}"
                            for question, answer in zip(
                                ditem["questions"][chunk_low:chunk_high],
                                ditem["answers"]["input_text"][
                                    chunk_low:chunk_high
                                ],
                            )
                        ]
                    )
                    + "\n"
                )

                question_str = f"Q: {ditem['questions'][chunk_high]}\n"
                answer_str = f"A: {ditem['answers']['input_text'][chunk_high]}"

                if tokenizer is not None:
                    story = tokenizer.encode(story_str)
                    question = tokenizer.encode(question_str)
                    answer = tokenizer.encode(answer_str)

                    if story[-1] == tokenizer.eos_token_id:
                        story = story[:-1]
                    if question[-1] == tokenizer.eos_token_id:
                        question = question[:-1]

                    if answer[0] == tokenizer.bos_token_id:
                        answer = answer[1:]
                    if question[0] == tokenizer.bos_token_id:
                        question = question[1:]

                    question_start_idx = len(story)
                    answer_start_idx = len(story) + len(question)

                    merged_datasets[ds_key].append(
                        {
                            "tokenized_prompt": story + question + answer,
                            "question_token_start_idx": question_start_idx,
                            "answer_token_start_idx": answer_start_idx,
                            "answer_str": answer_str,
                            "question_str": question_str,
                        }
                    )

                else:
                    logger.warning("no tokenizer offered, printing to stdout")
                    print(story_str + question_str + answer_str)
                batch_cache.append(ditem)
            merged_datasets[ds_key].append(ditem)

        merged_datasetdict = datasets.DatasetDict(
            {
                k: datasets.Dataset.from_pandas(pd.DataFrame(v))
                for k, v in merged_datasets.items()
            }
        )

        if cache:
            merged_datasetdict.save_to_disk(caching_path)

        return merged_datasetdict


def coqa_formatter(
    tokenizer: transformers.PreTrainedTokenizer,
    dpath: str = COQA_LOCAL,
    num_example: int = 3,
    cache: bool = True,
) -> datasets.DatasetDict:
    step_size = 1 + num_example
    dd = datasets.load_from_disk(dpath)
    merged_datasets = {}

    caching_path = str(
        Path(CACHE_LOCAL)
        / f"coqa_{tokenizer.__class__.__name__}_exmp{num_example}"
    )

    if cache:
        if Path(caching_path).exists():
            logger.info(f"Loading cached dataset from {caching_path}")
            try:
                merged_datasetdict = datasets.load_from_disk(caching_path)
                return merged_datasetdict
            except:
                logger.warning(
                    f"Failed to load cached dataset from {caching_path}, need regeneration"
                )

    for ds_key, ds in dd.items():
        merged_datasets[ds_key] = []

        batch_cache = []
        batch_id = None

        for ditem in tqdm(ds, desc=f"Formatting {ds_key} dataset"):
            if batch_id != ditem["id"].split("_")[0]:
                for i in range(len(batch_cache) // step_size):
                    try:
                        chunk = batch_cache[i * step_size : (i + 1) * step_size]
                    except IndexError as e:
                        logger.warning(
                            f"Failed to chunk {batch_cache} with step_size {step_size}, could be too small chunk"
                        )
                        break

                    if not len(chunk) == step_size:
                        break
                    else:
                        story_str = (
                            f"Reading the passage and answer given questions accordingly.\n\nPassage:\n{chunk[0]['story']}\n\n"
                            f"Examples:\n"
                            + "\n".join(
                                [
                                    f"Q: {question}\nA: {answer}"
                                    for question, answer in zip(
                                        [_["question"] for _ in chunk[:-1]],
                                        [
                                            _["answer"]["text"]
                                            for _ in chunk[:-1]
                                        ],
                                    )
                                ]
                            )
                            + "\n"
                        )

                        question_str = f"Q: {chunk[-1]['question']}\n"
                        answer_str = f"A: {chunk[-1]['answer']['text']}"

                        if tokenizer is not None:
                            story = tokenizer.encode(story_str)
                            question = tokenizer.encode(question_str)
                            answer = tokenizer.encode(answer_str)

                            if story[-1] == tokenizer.eos_token_id:
                                story = story[:-1]
                            if question[-1] == tokenizer.eos_token_id:
                                question = question[:-1]

                            if answer[0] == tokenizer.bos_token_id:
                                answer = answer[1:]
                            if question[0] == tokenizer.bos_token_id:
                                question = question[1:]

                            question_start_idx = len(story)
                            answer_start_idx = len(story) + len(question)

                            merged_datasets[ds_key].append(
                                {
                                    "tokenized_prompt": story
                                    + question
                                    + answer,
                                    "question_token_start_idx": question_start_idx,
                                    "answer_token_start_idx": answer_start_idx,
                                    "answer_str": answer_str,
                                    "question_str": question_str,
                                }
                            )

                        else:
                            logger.warning(
                                "no tokenizer offered, printing to stdout"
                            )
                            print(story_str + question_str + answer_str)

                batch_cache = []

                batch_id = ditem["id"].split("_")[0]
                batch_cache.append(ditem)
            else:
                batch_cache.append(ditem)

    merged_datasetdict = {("coqa__" + k): v for k, v in merged_datasets.items()}

    merged_datasetdict = datasets.DatasetDict(
        {
            k: datasets.Dataset.from_pandas(pd.DataFrame(v))
            for k, v in merged_datasets.items()
        }
    )

    if cache:
        merged_datasetdict.save_to_disk(caching_path)

    return merged_datasetdict


def triviaqa_formatter(
    tokenizer: transformers.PreTrainedTokenizer,
    dpath: str = TRIVIA_LOCAL,
    num_example: int = 3,
    cache: bool = True,
) -> datasets.DatasetDict:
    step_size = 1 + num_example
    dd = datasets.load_from_disk(dpath)
    merged_datasets = {}

    caching_path = str(
        Path(CACHE_LOCAL)
        / f"triviaqa_{tokenizer.__class__.__name__}_exmp{num_example}_answerauditv2"
    )

    if cache:
        if Path(caching_path).exists():
            logger.info(f"Loading cached dataset from {caching_path}")
            try:
                merged_datasetdict = datasets.load_from_disk(caching_path)
                return merged_datasetdict
            except:
                logger.warning(
                    f"Failed to load cached dataset from {caching_path}, need regeneration"
                )

    for ds_key, ds in dd.items():
        merged_datasets[ds_key] = []

        chunk_cache = []
        for idx, ditem in tqdm(
            enumerate(ds), desc=f"Formatting {ds_key} dataset"
        ):
            chunk_cache.append(ditem)

            if (idx + 1) % step_size == 0:
                prompt_str = (
                    f"Answer the question like following examples.\n\n"
                    + "\n".join(
                        [
                            f"Q: {_['question']}\nA: {_['answer']['value']}"
                            for _ in chunk_cache[:-1]
                        ]
                    )
                    + "\n"
                )
                question_str = f"Q: {chunk_cache[-1]['question']}\n"
                answer_obj = chunk_cache[-1].get("answer", {})
                gold_answer = answer_obj.get("value", "") if isinstance(answer_obj, dict) else ""
                gold_aliases = []
                if isinstance(answer_obj, dict):
                    for alias_key in ("aliases", "normalized_aliases"):
                        aliases = answer_obj.get(alias_key)
                        if isinstance(aliases, list):
                            gold_aliases.extend(str(alias) for alias in aliases if alias is not None)
                answer_str = f"A: {gold_answer}"
                if tokenizer is not None:
                    prompt = tokenizer.encode(prompt_str)
                    question = tokenizer.encode(question_str)
                    answer = tokenizer.encode(answer_str)

                    if prompt[-1] == tokenizer.eos_token_id:
                        prompt = prompt[:-1]
                    if question[-1] == tokenizer.eos_token_id:
                        question = question[:-1]

                    if answer[0] == tokenizer.bos_token_id:
                        answer = answer[1:]
                    if question[0] == tokenizer.bos_token_id:
                        question = question[1:]

                    question_start_idx = len(prompt)
                    answer_start_idx = len(prompt) + len(question)

                    merged_datasets[ds_key].append(
                        {
                            "sample_id": f"triviaqa:{ds_key}:{idx}",
                            "tokenized_prompt": prompt + question + answer,
                            "question_token_start_idx": question_start_idx,
                            "answer_token_start_idx": answer_start_idx,
                            "answer_str": answer_str,
                            "gold_answer": gold_answer,
                            "gold_aliases": gold_aliases,
                            "question_str": question_str,
                        }
                    )

                else:
                    logger.warning("no tokenizer offered, printing to stdout")
                    print(prompt_str + question_str + answer_str)

                # finish & clean cache
                chunk_cache = []

    merged_datasets = {
        ("triviaqa__" + k): v for k, v in merged_datasets.items()
    }

    merged_datasetdict = datasets.DatasetDict(
        {
            k: datasets.Dataset.from_pandas(pd.DataFrame(v))
            for k, v in merged_datasets.items()
        }
    )

    if cache:
        merged_datasetdict.save_to_disk(caching_path)

    return merged_datasetdict


def mmlu_formatter(
    tokenizer: transformers.PreTrainedTokenizer,
    dpath: str = MMLU_LOCAL,
    num_example: int = 5,
    cache: bool = True,
    merge_split: bool = False,
    conv_generation: bool = True,
    requested_split: Optional[str] = None,
) -> datasets.DatasetDict:
    merged_datasets = {}

    cache_split = requested_split or "all"
    caching_path = str(
        Path(CACHE_LOCAL)
        / f"mmlu_localarrow_v2_{tokenizer.__class__.__name__}_exmp{num_example}_merge{merge_split}_conv{conv_generation}_split{cache_split}"
    )

    if cache:
        if Path(caching_path).exists():
            logger.info(f"Loading cached dataset from {caching_path}")
            try:
                merged_datasetdict = datasets.load_from_disk(caching_path)
                return merged_datasetdict
            except:
                logger.warning(
                    f"Failed to load cached dataset from {caching_path}, need regeneration"
                )

    subject_dirs = discover_mmlu_subjects(dpath)
    use_requested_split_policy = requested_split is not None
    if requested_split is not None:
        requested_splits = (requested_split,)
    else:
        requested_splits = MMLU_LOCAL_SPLITS

    def _encode(text: str) -> list[int]:
        tokens = tokenizer.encode(text)
        if tokens and tokens[-1] == tokenizer.eos_token_id:
            tokens = tokens[:-1]
        if tokens and tokens[0] == tokenizer.bos_token_id:
            tokens = tokens[1:]
        return tokens

    def _example_text(row: Mapping[str, Any]) -> str:
        return f"{row['prompt']}{row['gold_option']}"

    def _format_rows(rows: list[dict[str, Any]], split_key: str) -> list[dict[str, Any]]:
        formatted = []
        for idx, row in enumerate(rows):
            if conv_generation:
                examples = rows[max(0, idx - int(num_example)):idx]
            else:
                window_start = (idx // max(1, int(num_example) + 1)) * (int(num_example) + 1)
                examples = rows[window_start:idx]
            prefix = ""
            if examples:
                prefix = (
                    "You will be given multiple-choice questions. "
                    "Answer each question using only A, B, C, or D.\n\n"
                    + "\n\n".join(_example_text(example) for example in examples)
                    + "\n\nNow answer the question.\n\n"
                )
            question = row["prompt"]
            answer = row["gold_option"]
            output_row = dict(row)
            output_row["resolved_variant"] = f"mmlu__{split_key}"
            output_row["split"] = split_key
            output_row["requested_split"] = split_key
            output_row["sample_id"] = f"mmlu:{row['subject']}:{split_key}:{idx}"
            output_row["prompt"] = prefix + question
            output_row["input"] = prefix + question
            output_row["question_str"] = question
            output_row["answer_str"] = answer
            output_row["gold_answer"] = answer
            output_row["gold_aliases"] = [answer, row["gold_answer_text"]]
            if tokenizer is not None:
                prefix_tokens = _encode(prefix) if prefix else []
                question_tokens = _encode(question)
                answer_tokens = _encode(answer)
                output_row["tokenized_prompt"] = prefix_tokens + question_tokens + answer_tokens
                output_row["question_token_start_idx"] = len(prefix_tokens)
                output_row["answer_token_start_idx"] = len(prefix_tokens) + len(question_tokens)
            formatted.append(output_row)
        return formatted

    for requested in requested_splits:
        actual = mmlu_actual_split(requested) if use_requested_split_policy else requested
        aggregate_rows: list[dict[str, Any]] = []
        for subject_dir in tqdm(subject_dirs, desc=f"Formatting MMLU requested_split={requested}"):
            subject = subject_dir.name
            try:
                ds = load_mmlu_subject_split(subject_dir, actual)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Skipping MMLU subject split subject={} requested_split={} actual_mmlu_split={} error={!r}",
                    subject, requested, actual, exc,
                )
                continue
            raw_rows = [
                normalize_mmlu_row(row, subject, actual, requested_split=requested, row_index=idx)
                for idx, row in enumerate(ds)
            ]
            formatted_rows = _format_rows(raw_rows, requested)
            if not use_requested_split_policy and requested in MMLU_LOCAL_SPLITS and requested == actual:
                merged_datasets[f"{subject}__{actual}"] = formatted_rows
            aggregate_rows.extend(formatted_rows)
        if aggregate_rows and (use_requested_split_policy or merge_split):
            merged_datasets[requested] = aggregate_rows

    merged_datasets = {("mmlu__" + k): v for k, v in merged_datasets.items() if len(v) > 0}

    merged_datasetdict = datasets.DatasetDict(
        {
            k: datasets.Dataset.from_pandas(pd.DataFrame(v))
            for k, v in merged_datasets.items()
        }
    )

    if cache:
        merged_datasetdict.save_to_disk(caching_path)

    return merged_datasetdict


def cnndaily_formatter(
    tokenizer: transformers.PreTrainedTokenizer,
    dpath: str = CNN_LOCAL,
    num_example: int = 3,
    cache: bool = True,
) -> datasets.DatasetDict:
    step_size = 1 + num_example
    dd = datasets.load_from_disk(dpath)
    merged_datasets = {}

    caching_path = str(
        Path(CACHE_LOCAL)
        / f"cnndaily_{tokenizer.__class__.__name__}_exmp{num_example}"
    )

    if cache:
        if Path(caching_path).exists():
            logger.info(f"Loading cached dataset from {caching_path}")
            try:
                merged_datasetdict = datasets.load_from_disk(caching_path)
                return merged_datasetdict
            except:
                logger.warning(
                    f"Failed to load cached dataset from {caching_path}, need regeneration"
                )

    for ds_key, ds in dd.items():
        merged_datasets[ds_key] = []

        chunk_cache = []
        for idx, ditem in tqdm(
            enumerate(ds), desc=f"Formatting {ds_key} dataset"
        ):

            chunk_cache.append(ditem)

            if (idx + 1) % step_size == 0:
                prompt = "".join(
                    [
                        (
                            "<start_of_turn>user\n What are the highlights in this paragraph?: "
                            f"{c['article']}"
                            "<end_of_turn>\n"
                            "<start_of_turn>model\n the highlights of the paragraph: "
                            f"{ENTER_PAT.sub(' ', c['highlights'])}"
                            "<end_of_turn>\n"
                        )
                        for c in chunk_cache[:-1]
                    ]
                )
                question = (
                    "<start_of_turn>user\n What are the highlights in this paragraph?: "
                    f"{chunk_cache[-1]['article']}"
                    "<end_of_turn>\n"
                    "<start_of_turn>model\n the highlights of the paragraph: "
                )

                answer = (
                    f"{ENTER_PAT.sub(' ', chunk_cache[-1]['highlights'])}"
                    "<end_of_turn>\n"
                )

                prompt_tokens = tokenizer.encode(prompt)
                question_tokens = tokenizer.encode(question)
                answer_tokens = tokenizer.encode(answer)

                if prompt_tokens[-1] == tokenizer.eos_token_id:
                    prompt_tokens = prompt_tokens[:-1]
                if question_tokens[-1] == tokenizer.eos_token_id:
                    question_tokens = question_tokens[:-1]

                if question_tokens[0] == tokenizer.bos_token_id:
                    question_tokens = question_tokens[1:]
                if answer_tokens[0] == tokenizer.bos_token_id:
                    answer_tokens = answer_tokens[1:]

                question_start_idx = len(prompt_tokens)
                answer_start_idx = question_start_idx + len(question_tokens)

                merged_datasets[ds_key].append(
                    {
                        "tokenized_prompt": prompt_tokens
                        + question_tokens
                        + answer_tokens,
                        "question_token_start_idx": question_start_idx,
                        "answer_token_start_idx": answer_start_idx,
                    }
                )

                chunk_cache = []

    merged_datasets = {
        ("cnndaily__" + k): v for k, v in merged_datasets.items()
    }

    merged_datasetdict = datasets.DatasetDict(
        {
            k: datasets.Dataset.from_pandas(pd.DataFrame(v))
            for k, v in merged_datasets.items()
        }
    )

    if cache:
        merged_datasetdict.save_to_disk(caching_path)

    return merged_datasetdict


def cnndailymail_formatter(
    path: str,
    tokenizer: transformers.PreTrainedTokenizer,
    n_shot: int = 2,
    cache: bool = False,
):
    dd = datasets.load_from_disk(path)
    step_size = 1 + n_shot
    merged_datasets = {}

    caching_path = str(
        Path(CACHE_LOCAL) / f"coqa_nexmp{n_shot}_{Path(path).name}"
    )

    if cache:
        if Path(caching_path).exists():
            try:
                merged_datasetdict = datasets.load_from_disk(caching_path)
                return merged_datasetdict
            except:
                pass

    for ds_key, ds in dd.items():
        merged_datasets[ds_key] = []

        chunk_cache = []
        for idx, ditem in tqdm(
            enumerate(ds), desc=f"Formatting {ds_key} dataset"
        ):

            chunk_cache.append(ditem)

            if (idx + 1) % step_size == 0:
                prompt = "".join(
                    [
                        (
                            "<start_of_turn>user\n What are the highlights in this paragraph?: "
                            f"{c['article']}"
                            "<end_of_turn>\n"
                            "<start_of_turn>model\n the highlights of the paragraph: "
                            f"{ENTER_PAT.sub(' ', c['highlights'])}"
                            "<end_of_turn>\n"
                        )
                        for c in chunk_cache[:-1]
                    ]
                )
                question = (
                    "<start_of_turn>user\n What are the highlights in this paragraph?: "
                    f"{chunk_cache[-1]['article']}"
                    "<end_of_turn>\n"
                    "<start_of_turn>model\n the highlights of the paragraph: "
                )

                answer = f"{ENTER_PAT.sub(' ', chunk_cache[-1]['highlights'])}"

                prompt_tokens = tokenizer.encode(prompt)
                question_tokens = tokenizer.encode(question)
                answer_tokens = tokenizer.encode(answer)

                question_start_idx = len(prompt_tokens)
                answer_start_idx = question_start_idx + len(question_tokens)

                merged_datasets[ds_key].append(
                    {
                        "prompt_str": prompt,
                        "question_str": question,
                        "answer_str": answer,
                        "tokenized_prompt": prompt_tokens
                        + question_tokens
                        + answer_tokens,
                        "tokenized_prompt_no_answer": prompt_tokens
                        + question_tokens,
                        "question_token_start_idx": question_start_idx,
                        "answer_token_start_idx": answer_start_idx,
                    }
                )

                chunk_cache = []

    merged_datasets = {
        ("cnndaily__" + k): v for k, v in merged_datasets.items()
    }

    merged_datasetdict = datasets.DatasetDict(
        {
            k: datasets.Dataset.from_pandas(pd.DataFrame(v))
            for k, v in merged_datasets.items()
        }
    )

    if cache:
        merged_datasetdict.save_to_disk(caching_path)

    return merged_datasetdict


def wmt_formatter(
    tokenizer: transformers.PreTrainedTokenizer,
    dpath: str = WMT_LOCAL,
    num_example: int = 3,
    cache: bool = True,
    conv_generation: bool = True,
    Q_LANG: str = "fr",
    A_LANG: str = "en",
    requested_split: Optional[str] = None,
    max_samples: Optional[int] = None,
    limit: Optional[int] = None,
    allow_full_formatting: bool = False,
) -> datasets.DatasetDict:
    step_size = 1 + num_example
    if max_samples is not None and limit is not None and int(max_samples) != int(limit):
        raise ValueError(f"Conflicting WMT limits: max_samples={max_samples} limit={limit}")
    requested_limit = int(max_samples if max_samples is not None else limit) if (max_samples is not None or limit is not None) else None
    dd = datasets.load_from_disk(dpath)
    merged_datasets = {}

    split_key = requested_split or "all"
    limit_key = "full" if requested_limit is None else f"limit{requested_limit}"
    tokenizer_name = tokenizer.__class__.__name__ if tokenizer is not None else "no_tokenizer"

    caching_path = str(
        Path(CACHE_LOCAL)
        / f"wmt_{WMT_FORMATTER_VERSION}_{split_key}_{Q_LANG}2{A_LANG}_{tokenizer_name}_exmp{num_example}_conv{conv_generation}_{limit_key}"
    )

    if cache:
        if Path(caching_path).exists():
            logger.info(f"Loading cached dataset from {caching_path}")
            try:
                merged_datasetdict = datasets.load_from_disk(caching_path)
                return merged_datasetdict
            except:
                logger.warning(
                    f"Failed to load cached dataset from {caching_path}, need regeneration"
                )

    if requested_split is not None:
        if requested_split not in dd:
            available = tuple(dd.keys()) if hasattr(dd, "keys") else ()
            raise KeyError(f"WMT split {requested_split!r} not found at {dpath}; available_splits={available}")
        split_items = [(requested_split, dd[requested_split])]
    else:
        split_items = list(dd.items())

    def _dataset_len_optional(ds: Any) -> Optional[int]:
        if hasattr(ds, "num_rows"):
            return int(ds.num_rows)
        try:
            return len(ds)
        except Exception:  # noqa: BLE001
            return None

    def _max_wmt_outputs(raw_count: Optional[int]) -> Optional[int]:
        if raw_count is None:
            return None
        if conv_generation:
            return max(0, int(raw_count) - int(num_example))
        return max(0, int(raw_count) // int(step_size))

    def _effective_wmt_outputs(raw_count: Optional[int]) -> Optional[int]:
        possible = _max_wmt_outputs(raw_count)
        if requested_limit is None:
            return possible
        if possible is None:
            return requested_limit
        return min(requested_limit, possible)

    def _select_raw_prefix(ds: Any, raw_take: Optional[int]):
        if raw_take is None:
            return ds
        if hasattr(ds, "select"):
            return ds.select(range(raw_take))
        return islice(ds, raw_take)

    def _format_chunk(ds_key: str, idx: int, chunk_cache: list[Mapping[str, Any]]) -> dict[str, Any]:
        prompt = "".join(
            [
                (
                    f"Q: What is the English translation of the following sentence? {sen['translation'][Q_LANG]}\n"
                    f"A: {sen['translation'][A_LANG]}\n"
                )
                for sen in chunk_cache[:-1]
            ]
        )
        question = f"Q: What is the English translation of the following sentence? {chunk_cache[-1]['translation'][Q_LANG]}\nA: "
        answer = f"{chunk_cache[-1]['translation'][A_LANG]}"

        if tokenizer is None:
            print(prompt + question + answer)
            return {
                "sample_id": f"wmt:{ds_key}:{idx}",
                "prompt_str": prompt,
                "question_str": question,
                "answer_str": answer,
                "gold_answer": answer,
                "gold_aliases": [answer],
            }

        prompt_tokens = tokenizer.encode(prompt)
        question_tokens = tokenizer.encode(question)
        answer_tokens = tokenizer.encode(answer)

        if prompt_tokens and prompt_tokens[-1] == tokenizer.eos_token_id:
            prompt_tokens = prompt_tokens[:-1]
        if question_tokens and question_tokens[-1] == tokenizer.eos_token_id:
            question_tokens = question_tokens[:-1]
        if question_tokens and question_tokens[0] == tokenizer.bos_token_id:
            question_tokens = question_tokens[1:]
        if answer_tokens and answer_tokens[0] == tokenizer.bos_token_id:
            answer_tokens = answer_tokens[1:]

        question_start_idx = len(prompt_tokens)
        answer_start_idx = question_start_idx + len(question_tokens)
        return {
            "sample_id": f"wmt:{ds_key}:{idx}",
            "tokenized_prompt": prompt_tokens + question_tokens + answer_tokens,
            "question_token_start_idx": question_start_idx,
            "answer_token_start_idx": answer_start_idx,
            "answer_str": answer,
            "gold_answer": answer,
            "gold_aliases": [answer],
            "question_str": question,
        }

    for ds_key, ds in split_items:
        merged_datasets[ds_key] = []
        raw_count = _dataset_len_optional(ds)
        if (
            requested_limit is None
            and not allow_full_formatting
            and raw_count is not None
            and raw_count > FULL_FORMATTING_GUARD_RAW_EXAMPLES
        ):
            raise ValueError(
                f"Refusing to format full WMT split {ds_key!r} with {raw_count} raw examples. "
                "Pass max_samples/limit or set allow_full_formatting=True intentionally."
            )
        effective_samples = _effective_wmt_outputs(raw_count)
        if requested_limit is not None:
            raw_take = requested_limit + num_example if conv_generation else requested_limit * step_size
            if raw_count is not None:
                raw_take = min(raw_take, raw_count)
        else:
            raw_take = None
        logger.info(
            "[formatter:wmt] split={} raw_split_size={} requested_limit={} "
            "effective_num_samples={} cache_path={}",
            ds_key, raw_count, requested_limit, effective_samples, caching_path,
        )
        chunk_cache = []
        progress = tqdm(
            total=effective_samples,
            desc=f"Formatting {ds_key} dataset",
            unit="sample",
        )
        try:
            for idx, ditem in enumerate(_select_raw_prefix(ds, raw_take)):
                chunk_cache.append(ditem)
                if len(chunk_cache) != step_size:
                    continue
                merged_datasets[ds_key].append(_format_chunk(ds_key, idx, chunk_cache))
                progress.update(1)
                if requested_limit is not None and len(merged_datasets[ds_key]) >= requested_limit:
                    break
                if conv_generation:
                    chunk_cache.pop(0)
                else:
                    chunk_cache = []
        finally:
            progress.close()
        logger.info(
            "[formatter:wmt] split={} actual_formatted_sample_count={}",
            ds_key, len(merged_datasets[ds_key]),
        )

    merged_datasets = {("wmt__" + k): v for k, v in merged_datasets.items()}

    merged_datasetdict = datasets.DatasetDict(
        {
            k: datasets.Dataset.from_pandas(pd.DataFrame(v))
            for k, v in merged_datasets.items()
        }
    )

    if cache:
        merged_datasetdict.save_to_disk(caching_path)

    return merged_datasetdict


def webgpt_formatter(
    tokenizer: transformers.PreTrainedTokenizer,
    dpath: str = WEBGPT_LOCAL,
    num_example: int = 3,
    cache: bool = True,
    conv_generation: bool = True,
) -> datasets.DatasetDict:
    step_size = 1 + num_example
    dd = datasets.load_from_disk(dpath)
    merged_datasets: dict = {}

    IDX_PATTERN = re.compile(r"\s*\[\d+.*\]\s*")

    caching_path = str(
        Path(CACHE_LOCAL)
        / f"webgpt_{tokenizer.__class__.__name__}_exmp{num_example}"
    )

    if cache:
        if Path(caching_path).exists():
            try:
                merged_datasetdict = datasets.load_from_disk(caching_path)
                return merged_datasetdict
            except:
                pass

    for ds_key, ds in dd.items():
        merged_datasets[ds_key] = []

        chunk_cache = []
        if not conv_generation:
            for idx, ditem in tqdm(
                enumerate(ds), desc=f"Formatting {ds_key} dataset"
            ):
                if ditem["score_0"] == 0 and ditem["score_1"] == 0:
                    continue
                else:
                    chunk_cache.append(ditem)

                if (idx + 1) % step_size == 0:
                    prompt = "".join(
                        [
                            "You will receive a question with two options. Please select either answer A or B to indicate your preference. Here are some examples:\n\n"
                        ]
                        + [
                            f"Question: {d['question']['full_text']}\nA. {IDX_PATTERN.sub(' ', d['answer_0']).strip()}\nB. {IDX_PATTERN.sub(' ', d['answer_1']).strip()}\nAnswer: {'A' if d['score_0'] > d['score_1'] else 'B'}\n"
                            for d in chunk_cache[:-1]
                        ]
                    )
                    question = f"Question: {chunk_cache[-1]['question']['full_text']}\nA. {IDX_PATTERN.sub(' ', chunk_cache[-1]['answer_0']).strip()}\nB. {IDX_PATTERN.sub(' ', chunk_cache[-1]['answer_1']).strip()}\nAnswer: "
                    answer = f"{'A' if chunk_cache[-1]['score_0'] > chunk_cache[-1]['score_1'] else 'B'}"

                    prompt = normalize_text(prompt)

                    if tokenizer is not None:
                        prompt_tokens = tokenizer.encode(prompt)
                        question_tokens = tokenizer.encode(question)
                        answer_tokens = tokenizer.encode(answer)

                        if prompt_tokens[-1] == tokenizer.eos_token_id:
                            prompt_tokens = prompt_tokens[:-1]
                        if question_tokens[-1] == tokenizer.eos_token_id:
                            question_tokens = question_tokens[:-1]

                        if question_tokens[0] == tokenizer.bos_token_id:
                            question_tokens = question_tokens[1:]
                        if answer_tokens[0] == tokenizer.bos_token_id:
                            answer_tokens = answer_tokens[1:]

                        question_start_idx = len(prompt_tokens)
                        answer_start_idx = question_start_idx + len(
                            question_tokens
                        )

                        merged_datasets[ds_key].append(
                            {
                                "tokenized_prompt": prompt_tokens
                                + question_tokens
                                + answer_tokens,
                                "question_token_start_idx": question_start_idx,
                                "answer_token_start_idx": answer_start_idx,
                                "prompt_str": prompt,
                                "answer_str": answer,
                                "question_str": question,
                            }
                        )
                    else:
                        print(prompt + question + answer)
                        print("=" * 50)

                    chunk_cache = []
        else:
            for idx, ditem in tqdm(
                enumerate(ds), desc=f"Formatting {ds_key} dataset"
            ):
                if ditem["score_0"] == 0 and ditem["score_1"] == 0:
                    continue
                else:
                    chunk_cache.append(ditem)

                if len(chunk_cache) == step_size:
                    prompt = "".join(
                        [
                            "You will receive a question with two options. Please select either answer A or B to indicate your preference. Here are some examples:\n\n"
                        ]
                        + [
                            f"Question: {d['question']['full_text']}\nA. {IDX_PATTERN.sub(' ', d['answer_0']).strip()}\nB. {IDX_PATTERN.sub(' ', d['answer_1']).strip()}\nAnswer: {'A' if d['score_0'] > d['score_1'] else 'B'}\n"
                            for d in chunk_cache[:-1]
                        ]
                    )
                    question = f"Question: {chunk_cache[-1]['question']['full_text']}\nA. {IDX_PATTERN.sub(' ', chunk_cache[-1]['answer_0']).strip()}\nB. {IDX_PATTERN.sub(' ', chunk_cache[-1]['answer_1']).strip()}\nAnswer: "
                    answer = f"{'A' if chunk_cache[-1]['score_0'] > chunk_cache[-1]['score_1'] else 'B'}"

                    prompt = normalize_text(prompt)

                    if tokenizer is not None:
                        prompt_tokens = tokenizer.encode(prompt)
                        question_tokens = tokenizer.encode(question)
                        answer_tokens = tokenizer.encode(answer)

                        if prompt_tokens[-1] == tokenizer.eos_token_id:
                            prompt_tokens = prompt_tokens[:-1]
                        if question_tokens[-1] == tokenizer.eos_token_id:
                            question_tokens = question_tokens[:-1]

                        if question_tokens[0] == tokenizer.bos_token_id:
                            question_tokens = question_tokens[1:]
                        if answer_tokens[0] == tokenizer.bos_token_id:
                            answer_tokens = answer_tokens[1:]

                        question_start_idx = len(prompt_tokens)
                        answer_start_idx = question_start_idx + len(
                            question_tokens
                        )

                        merged_datasets[ds_key].append(
                            {
                                "tokenized_prompt": prompt_tokens
                                + question_tokens
                                + answer_tokens,
                                "question_token_start_idx": question_start_idx,
                                "answer_token_start_idx": answer_start_idx,
                                "prompt_str": prompt,
                                "answer_str": answer,
                                "question_str": question,
                            }
                        )
                    else:
                        print(prompt + question + answer)
                        print("=" * 50)

                    chunk_cache.pop(0)

    merged_datasets = {("webgpt__" + k): v for k, v in merged_datasets.items()}

    merged_datasetdict = datasets.DatasetDict(
        {
            k: datasets.Dataset.from_pandas(pd.DataFrame(v))
            for k, v in merged_datasets.items()
        }
    )

    if cache:
        merged_datasetdict.save_to_disk(caching_path)

    return merged_datasetdict


# ---------------------------------------------------------------------------
# MDUQ formatters: AmbigQA and TruthfulQA
# ---------------------------------------------------------------------------


def _tokenize_qa_triplet(
    tokenizer: transformers.PreTrainedTokenizer,
    prompt_str: str,
    question_str: str,
    answer_str: str,
):
    """Encode prompt/question/answer and return the merged token sequence
    plus the boundary indices, mirroring :func:`triviaqa_formatter`.
    """
    prompt = tokenizer.encode(prompt_str)
    question = tokenizer.encode(question_str)
    answer = tokenizer.encode(answer_str)
    if prompt and prompt[-1] == tokenizer.eos_token_id:
        prompt = prompt[:-1]
    if question and question[-1] == tokenizer.eos_token_id:
        question = question[:-1]
    if answer and answer[0] == tokenizer.bos_token_id:
        answer = answer[1:]
    if question and question[0] == tokenizer.bos_token_id:
        question = question[1:]
    question_start_idx = len(prompt)
    answer_start_idx = len(prompt) + len(question)
    return (
        prompt + question + answer,
        question_start_idx,
        answer_start_idx,
    )


def _ambigqa_extract_answers(item) -> list[str]:
    """Pull the flat list of plausible answers from an AmbigQA row.

    AmbigQA's ``light`` config exposes ``annotations`` with either a
    ``singleAnswer`` (unambiguous) or a ``multipleQAs`` field (ambiguous,
    several disambiguated QA pairs). We collapse them into a flat,
    deduplicated list of answer strings.
    """
    answers: list[str] = []
    annotations = item.get("annotations") or {}
    single = annotations.get("answer") or annotations.get("singleAnswer")
    if single:
        for a in single:
            if isinstance(a, list):
                answers.extend([str(x) for x in a])
            else:
                answers.append(str(a))
    multi = annotations.get("qaPairs") or annotations.get("multipleQAs")
    if multi:
        for qa in multi:
            qa_answers = qa.get("answer") if isinstance(qa, dict) else None
            if not qa_answers:
                continue
            for a in qa_answers:
                if isinstance(a, list):
                    answers.extend([str(x) for x in a])
                else:
                    answers.append(str(a))
    # dedupe while preserving order
    seen, dedup = set(), []
    for a in answers:
        a = a.strip()
        if a and a not in seen:
            seen.add(a)
            dedup.append(a)
    return dedup


def ambigqa_formatter(
    tokenizer: transformers.PreTrainedTokenizer,
    dpath: str = AMBIGQA_LOCAL,
    num_example: int = 3,
    cache: bool = True,
) -> datasets.DatasetDict:
    """Formatter for the ``ambigqa`` dataset (sewon/ambig_qa, ``light`` config).

    Output schema matches the legacy QA formatters
    (``tokenized_prompt``, ``question_token_start_idx``,
    ``answer_token_start_idx``, ``question_str``, ``answer_str``) and adds
    ``all_answers`` (list of plausible answers) and ``num_answers`` so that
    downstream MDUQ code can derive an ambiguity-dimension proxy label.
    """
    step_size = 1 + num_example
    dd = datasets.load_from_disk(dpath)
    merged_datasets = {}

    caching_path = str(
        Path(CACHE_LOCAL)
        / f"ambigqa_{tokenizer.__class__.__name__}_exmp{num_example}"
    )
    if cache and Path(caching_path).exists():
        logger.info(f"Loading cached dataset from {caching_path}")
        try:
            return datasets.load_from_disk(caching_path)
        except Exception:
            logger.warning(
                f"Failed to load cached dataset from {caching_path}, regenerating"
            )

    for ds_key, ds in dd.items():
        merged_datasets[ds_key] = []
        chunk_cache = []
        for idx, ditem in tqdm(
            enumerate(ds), desc=f"Formatting ambigqa/{ds_key}"
        ):
            chunk_cache.append(ditem)
            if (idx + 1) % step_size != 0:
                continue

            example_lines = []
            for ex in chunk_cache[:-1]:
                ex_answers = _ambigqa_extract_answers(ex)
                ex_answer = ex_answers[0] if ex_answers else ""
                example_lines.append(
                    f"Q: {ex['question']}\nA: {ex_answer}"
                )
            prompt_str = (
                "Answer the question like following examples.\n\n"
                + "\n".join(example_lines)
                + "\n"
            )
            target = chunk_cache[-1]
            target_answers = _ambigqa_extract_answers(target)
            target_answer = target_answers[0] if target_answers else ""

            question_str = f"Q: {target['question']}\n"
            answer_str = f"A: {target_answer}"

            if tokenizer is None:
                logger.warning("no tokenizer offered, printing to stdout")
                print(prompt_str + question_str + answer_str)
                chunk_cache = []
                continue

            tokens, q_start, a_start = _tokenize_qa_triplet(
                tokenizer, prompt_str, question_str, answer_str
            )
            merged_datasets[ds_key].append(
                {
                    "sample_id": f"ambigqa:{ds_key}:{idx}",
                    "tokenized_prompt": tokens,
                    "question_token_start_idx": q_start,
                    "answer_token_start_idx": a_start,
                    "answer_str": answer_str,
                    "gold_answer": target_answer,
                    "gold_aliases": target_answers,
                    "question_str": question_str,
                    "all_answers": target_answers,
                    "num_answers": len(target_answers),
                }
            )
            chunk_cache = []

    merged_datasets = {
        ("ambigqa__" + k): v for k, v in merged_datasets.items()
    }
    merged_datasetdict = datasets.DatasetDict(
        {
            k: datasets.Dataset.from_pandas(pd.DataFrame(v))
            for k, v in merged_datasets.items()
        }
    )

    if cache:
        merged_datasetdict.save_to_disk(caching_path)

    return merged_datasetdict


def truthfulqa_formatter(
    tokenizer: transformers.PreTrainedTokenizer,
    dpath: str = TRUTHFULQA_LOCAL,
    num_example: int = 3,
    cache: bool = True,
) -> datasets.DatasetDict:
    """Formatter for the ``truthfulqa`` dataset (truthful_qa, ``generation`` config).

    Each row exposes ``question``, ``best_answer``, ``correct_answers`` and
    ``incorrect_answers``. We emit one formatted row per raw TruthfulQA row;
    few-shot examples are prompt context only and never consume target rows.
    """
    dd = datasets.load_from_disk(dpath)
    merged_datasets = {}

    caching_path = str(
        Path(CACHE_LOCAL)
        / f"truthfulqa_{TRUTHFULQA_FORMATTER_VERSION}_{tokenizer.__class__.__name__}_exmp{num_example}"
    )
    if cache and Path(caching_path).exists():
        logger.info(f"Loading cached dataset from {caching_path}")
        try:
            return datasets.load_from_disk(caching_path)
        except Exception:
            logger.warning(
                f"Failed to load cached dataset from {caching_path}, regenerating"
            )

    # truthful_qa "generation" only has the validation split.
    if isinstance(dd, datasets.Dataset):
        split_items = {"validation": dd}
    else:
        split_items = dict(dd.items())

    for ds_key, ds in split_items.items():
        merged_datasets[ds_key] = []
        raw_rows = [dict(row) for row in ds]
        for idx, target in tqdm(
            enumerate(raw_rows), total=len(raw_rows), desc=f"Formatting truthfulqa/{ds_key}"
        ):
            examples = []
            for ex_idx, ex in enumerate(raw_rows):
                if ex_idx == idx:
                    continue
                examples.append(ex)
                if len(examples) >= int(num_example):
                    break
            example_lines = [
                f"Q: {ex['question']}\nA: {ex.get('best_answer', '')}"
                for ex in examples
            ]
            prompt_str = (
                "Answer the question like following examples.\n\n"
                + "\n".join(example_lines)
                + "\n"
            )
            best_answer = target.get("best_answer", "") or ""
            correct = list(target.get("correct_answers") or [])
            incorrect = list(target.get("incorrect_answers") or [])

            question_str = f"Q: {target['question']}\n"
            answer_str = f"A: {best_answer}"

            if tokenizer is None:
                logger.warning("no tokenizer offered, printing to stdout")
                print(prompt_str + question_str + answer_str)
                chunk_cache = []
                continue

            tokens, q_start, a_start = _tokenize_qa_triplet(
                tokenizer, prompt_str, question_str, answer_str
            )
            merged_datasets[ds_key].append(
                {
                    "sample_id": f"truthfulqa:{ds_key}:{idx}",
                    "source_sample_id": f"truthfulqa:{ds_key}:{idx}",
                    "split": ds_key,
                    "tokenized_prompt": tokens,
                    "question_token_start_idx": q_start,
                    "answer_token_start_idx": a_start,
                    "answer_str": answer_str,
                    "gold_answer": best_answer,
                    "gold_aliases": correct or ([best_answer] if best_answer else []),
                    "question_str": question_str,
                    "correct_answers": correct,
                    "incorrect_answers": incorrect,
                }
            )

    merged_datasets = {
        ("truthfulqa__" + k): v for k, v in merged_datasets.items()
    }
    merged_datasetdict = datasets.DatasetDict(
        {
            k: datasets.Dataset.from_pandas(pd.DataFrame(v))
            for k, v in merged_datasets.items()
        }
    )

    if cache:
        merged_datasetdict.save_to_disk(caching_path)

    return merged_datasetdict


# ---------------------------------------------------------------------------
# Registry-aware formatter dispatch
# ---------------------------------------------------------------------------

# Map a dataset registry name to the formatter callable in this module.
_FORMATTER_REGISTRY = {
    "coqa": coqa_formatter_hf,
    "triviaqa": triviaqa_formatter,
    "mmlu": mmlu_formatter,
    "wmt": wmt_formatter,
    "ambigqa": ambigqa_formatter,
    "truthfulqa": truthfulqa_formatter,
}


def get_formatter(dataset_name: str):
    """Return the formatter callable for ``dataset_name``.

    Looks up the bare dataset name (e.g. ``"triviaqa"``) and tolerates the
    ``"<name>__<split>"`` convention used elsewhere in the codebase by
    splitting on ``"__"``.
    """
    base = dataset_name.split("__", 1)[0]
    if base not in _FORMATTER_REGISTRY:
        raise KeyError(
            f"No formatter registered for dataset '{dataset_name}'. "
            f"Available: {sorted(_FORMATTER_REGISTRY.keys())}"
        )
    return _FORMATTER_REGISTRY[base]


class ResponseCacheDatasetError(RuntimeError):
    """Dataset/formatter validation error with response-cache context."""


_FORMATTER_DEFAULT_KWARGS = {
    "coqa": {"num_example": 3, "cache": True},
    "triviaqa": {"num_example": 3, "cache": True},
    "ambigqa": {"num_example": 3, "cache": True},
    "truthfulqa": {"num_example": 3, "cache": True},
    "wmt": {"num_example": 3, "cache": True, "conv_generation": True},
    "mmlu": {"num_example": 5, "cache": True, "merge_split": False, "conv_generation": True},
}

_LOCAL_DATASET_PATHS = {
    "coqa": COQA_LOCAL,
    "triviaqa": TRIVIA_LOCAL,
    "mmlu": MMLU_LOCAL,
    "wmt": WMT_LOCAL,
    "ambigqa": AMBIGQA_LOCAL,
    "truthfulqa": TRUTHFULQA_LOCAL,
}

_RESPONSE_CACHE_REQUIRED_FIELDS = (
    "tokenized_prompt",
    "question_token_start_idx",
    "answer_token_start_idx",
    "question_str",
    "answer_str",
    "sample_id",
)


def _formatter_name(base_dataset: str) -> str:
    formatted_sample_count = None
    internal_train_count = None
    internal_eval_count = None

    try:
        return getattr(get_formatter(base_dataset), "__name__", repr(get_formatter(base_dataset)))
    except Exception:
        return "<unregistered>"


def _suggest_response_cache_command(variant: DatasetVariant) -> str:
    split_flag = f" --split {variant.split}" if variant.split else ""
    return (
        "python run.py build-response-cache --scope custom "
        f"-d {variant.base_dataset}{split_flag} -m <model> --train-limit 2 --eval-limit 2 --dry-run"
    )


def _dataset_error(
    message: str,
    *,
    variant: DatasetVariant,
    formatter_name: Optional[str] = None,
    available_splits: Optional[tuple[str, ...]] = None,
) -> ResponseCacheDatasetError:
    splits = tuple(available_splits if available_splits is not None else variant.available_splits)
    return ResponseCacheDatasetError(
        f"{message} requested_dataset={variant.requested_dataset!r} "
        f"base_dataset={variant.base_dataset!r} split={variant.split!r} "
        f"formatter={formatter_name or _formatter_name(variant.base_dataset)} "
        f"available_splits={splits} suggestion={_suggest_response_cache_command(variant)}"
    )


def _validate_registry_split(variant: DatasetVariant) -> None:
    try:
        available = tuple(get_split_names(variant.base_dataset, prefer="mduq"))
    except Exception as exc:  # noqa: BLE001
        raise _dataset_error(
            f"response-cache generation does not support dataset={variant.base_dataset!r}.",
            variant=variant,
            available_splits=(),
        ) from exc
    if not variant.split:
        raise _dataset_error(
            "response-cache generation requires an explicit split-qualified dataset.",
            variant=variant,
            available_splits=available,
        )
    internal_info = parse_internal_split_variant(variant.split_tag)
    if internal_info is not None and internal_info.source_split in available:
        return
    if variant.split not in available:
        raise _dataset_error(
            "response-cache generation requested a split not present in the registry.",
            variant=variant,
            available_splits=available,
        )


def _matching_formatted_keys(formatted: Mapping[str, Any], variant: DatasetVariant) -> list[str]:
    keys = list(formatted.keys())
    internal_info = parse_internal_split_variant(variant.split_tag)
    if internal_info is not None and internal_info.source_variant in formatted:
        return [internal_info.source_variant]
    if variant.split_tag in formatted:
        return [variant.split_tag]
    if variant.split in formatted:
        return [variant.split]
    if variant.base_dataset == "mmlu" and variant.split:
        suffix = f"__{variant.split}"
        matches = [key for key in keys if key.startswith("mmlu__") and key.endswith(suffix)]
        if matches:
            return sorted(matches)
    raise _dataset_error(
        "formatter did not return the requested split.",
        variant=variant,
        available_splits=tuple(keys),
    )


def _dataset_len(ds: Any) -> int:
    if hasattr(ds, "num_rows"):
        return int(ds.num_rows)
    return len(ds)


def _select_rows(ds: Any, limit: Optional[int]) -> list[dict[str, Any]]:
    count = _dataset_len(ds)
    row_count = count if limit is None else min(int(limit), count)
    if hasattr(ds, "select"):
        return [dict(row) for row in ds.select(range(row_count))]
    return [dict(row) for row in list(ds)[:row_count]]


def _strip_answer_prefix_for_gold(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return re.sub(r"^\s*A\s*[:：]\s*", "", text, flags=re.IGNORECASE).strip()


def _normalize_response_cache_row(row: dict[str, Any], *, split_tag: str, row_index: int) -> dict[str, Any]:
    row = dict(row)
    row.setdefault("sample_id", f"{split_tag}:{row_index}")
    if not row.get("question_str") and row.get("question"):
        row["question_str"] = str(row.get("question"))
    if not row.get("gold_answer") and row.get("answer_str"):
        row["gold_answer"] = _strip_answer_prefix_for_gold(row.get("answer_str"))
    if not row.get("gold_aliases"):
        aliases = row.get("all_answers") or row.get("correct_answers") or row.get("aliases")
        if aliases:
            row["gold_aliases"] = list(aliases) if isinstance(aliases, list) else [aliases]
        elif row.get("gold_answer"):
            row["gold_aliases"] = [row["gold_answer"]]
    missing = [field for field in _RESPONSE_CACHE_REQUIRED_FIELDS if field not in row]
    if missing:
        raise ResponseCacheDatasetError(
            f"formatter row missing required response-cache fields: {missing}. "
            f"split_tag={split_tag!r} sample_id={row.get('sample_id')!r}"
        )
    return row


def load_formatted_dataset(
    base_dataset: str,
    split: str,
    tokenizer: transformers.PreTrainedTokenizer,
    limit: Optional[int] = None,
    *,
    cache: bool = True,
    dataset_variant: Optional[str] = None,
    allow_full_formatting: bool = False,
    internal_train_ratio: float = DEFAULT_INTERNAL_TRAIN_RATIO,
    internal_split_seed: int = DEFAULT_INTERNAL_SPLIT_SEED,
) -> list[dict[str, Any]]:
    """Load response-cache-ready rows through the registered formatter."""
    requested = dataset_variant or f"{base_dataset}__{split}"
    variant = parse_dataset_variant(requested, split=split, prefer="mduq")
    _validate_registry_split(variant)
    if variant.base_dataset not in _FORMATTER_REGISTRY:
        raise _dataset_error(
            f"response-cache generation does not support dataset={variant.base_dataset!r}, split={variant.split!r}.",
            variant=variant,
        )
    formatter = get_formatter(variant.base_dataset)
    kwargs = dict(_FORMATTER_DEFAULT_KWARGS.get(variant.base_dataset, {}))
    kwargs["cache"] = bool(cache)
    if variant.base_dataset == "mmlu":
        kwargs["requested_split"] = variant.split
    if variant.base_dataset == "wmt":
        kwargs["requested_split"] = variant.split
        kwargs["max_samples"] = limit
        kwargs["allow_full_formatting"] = allow_full_formatting
    formatted = formatter(tokenizer=tokenizer, **kwargs)
    keys = _matching_formatted_keys(formatted, variant)
    internal_info = parse_internal_split_variant(variant.split_tag)

    rows: list[dict[str, Any]] = []
    remaining = None if limit is None else max(0, int(limit))
    for key in keys:
        if remaining == 0:
            break
        key_limit = None if internal_info is not None else remaining
        selected = _select_rows(formatted[key], key_limit)
        source_count_before_internal = len(selected)
        if internal_info is not None:
            selected = [
                row for row in selected
                if row_belongs_to_internal_split(
                    row,
                    internal_info.virtual_split,
                    seed=internal_split_seed,
                    train_ratio=internal_train_ratio,
                )
            ]
            selected = selected if remaining is None else selected[:remaining]
            if (
                variant.base_dataset == "truthfulqa"
                and internal_info.virtual_split == "eval"
                and source_count_before_internal > 0
            ):
                expected_eval = source_count_before_internal * (1.0 - float(internal_train_ratio))
                if len(selected) < 0.5 * expected_eval:
                    logger.warning(
                        "[formatter:truthfulqa] internal eval split has fewer than 50% of expected rows: "
                        "actual={} expected≈{:.1f} source_rows={} train_ratio={} variant={}",
                        len(selected), expected_eval, source_count_before_internal,
                        internal_train_ratio, variant.split_tag,
                    )
        for row in selected:
            if internal_info is not None:
                row = dict(row)
                row.update(
                    {
                        "source_dataset": internal_info.source_variant,
                        "virtual_split": internal_info.virtual_split,
                        "internal_split_seed": int(internal_split_seed),
                        "internal_train_ratio": float(internal_train_ratio),
                        "split_policy": "internal_split",
                        "held_out_evaluation": True,
                        "same_split_evaluation": False,
                    }
                )
            rows.append(_normalize_response_cache_row(row, split_tag=variant.split_tag, row_index=len(rows)))
        if remaining is not None:
            remaining = max(0, remaining - len(selected))
    if not rows:
        raise _dataset_error(
            "formatter returned zero usable rows for response-cache generation.",
            variant=variant,
            available_splits=tuple(keys),
        )
    return rows


def _raw_dataset_split_count(raw: Any, split: str) -> tuple[int, tuple[str, ...]]:
    if isinstance(raw, datasets.Dataset):
        return len(raw), (split,)
    available = tuple(raw.keys())
    if split not in raw:
        return 0, available
    return len(raw[split]), available


def _formatted_wmt_sample_capacity(
    raw_count: Optional[int],
    *,
    num_example: int,
    conv_generation: bool,
) -> Optional[int]:
    if raw_count is None:
        return None
    if conv_generation:
        return max(0, int(raw_count) - int(num_example))
    return max(0, int(raw_count) // int(num_example + 1))


def _effective_wmt_sample_count(
    raw_count: Optional[int],
    requested_limit: Optional[int],
    *,
    num_example: int,
    conv_generation: bool,
) -> Optional[int]:
    capacity = _formatted_wmt_sample_capacity(
        raw_count,
        num_example=num_example,
        conv_generation=conv_generation,
    )
    if requested_limit is None:
        return capacity
    if capacity is None:
        return int(requested_limit)
    return min(int(requested_limit), int(capacity))


def validate_response_cache_dataset_request(
    dataset_name: str,
    *,
    sample_limit: int = 2,
    requested_limit: Optional[int] = None,
    allow_full_formatting: bool = False,
    internal_train_ratio: float = DEFAULT_INTERNAL_TRAIN_RATIO,
    internal_split_seed: int = DEFAULT_INTERNAL_SPLIT_SEED,
) -> dict[str, Any]:
    """Fail fast on formatter/split availability before loading an LLM."""
    variant = parse_dataset_variant(dataset_name, prefer="mduq")
    _validate_registry_split(variant)
    internal_info = parse_internal_split_variant(variant.split_tag)
    formatter = get_formatter(variant.base_dataset)
    formatter_name = getattr(formatter, "__name__", repr(formatter))
    local_path = Path(_LOCAL_DATASET_PATHS.get(variant.base_dataset, ""))
    if not local_path.exists():
        raise _dataset_error(
            f"formatted source dataset is missing at {local_path}.",
            variant=variant,
            formatter_name=formatter_name,
        )

    try:
        if variant.base_dataset == "mmlu":
            actual_split = mmlu_actual_split(variant.split)
            subject_dirs = sorted(path for path in local_path.glob("*") if path.is_dir())
            if not subject_dirs:
                raise FileNotFoundError(f"no MMLU subject directories under {local_path}")
            sample_count = 0
            available_seen: set[str] = set()
            for subject_dir in subject_dirs[:3]:
                subject_available = tuple(
                    split for split in MMLU_LOCAL_SPLITS if (subject_dir / split).exists()
                )
                try:
                    raw = load_mmlu_subject_split(subject_dir, actual_split)
                except Exception:
                    available_seen.update(subject_available)
                    continue
                count = len(raw)
                available_seen.update(subject_available or (actual_split,))
                sample_count += min(int(sample_limit), count)
                if sample_count:
                    break
            available_raw_splits = tuple(sorted(available_seen)) or variant.available_splits
            raw_sample_count = None
            effective_sample_count = sample_count
            limit_applied_before_formatting = False
        elif variant.base_dataset == "wmt":
            raw = datasets.load_from_disk(str(local_path))
            count, available_raw_splits = _raw_dataset_split_count(raw, variant.split)
            formatter_kwargs = dict(_FORMATTER_DEFAULT_KWARGS.get("wmt", {}))
            if (
                requested_limit is None
                and not allow_full_formatting
                and count > FULL_FORMATTING_GUARD_RAW_EXAMPLES
            ):
                requested_limit = resolve_response_cache_limit(
                    variant.split_tag,
                    None,
                    allow_full_formatting=False,
                )
            effective = _effective_wmt_sample_count(
                count,
                requested_limit,
                num_example=int(formatter_kwargs.get("num_example", 3)),
                conv_generation=bool(formatter_kwargs.get("conv_generation", True)),
            )
            sample_count = int(effective or 0)
            raw_sample_count = count
            effective_sample_count = sample_count
            limit_applied_before_formatting = requested_limit is not None
        else:
            raw = datasets.load_from_disk(str(local_path))
            source_split = internal_info.source_split if internal_info is not None else variant.split
            count, available_raw_splits = _raw_dataset_split_count(raw, source_split)
            if variant.base_dataset == "truthfulqa" and internal_info is not None:
                formatted_sample_count = count
                internal_train_count = sum(
                    1
                    for idx in range(count)
                    if row_belongs_to_internal_split(
                        {"sample_id": f"truthfulqa:{source_split}:{idx}"},
                        "train",
                        seed=internal_split_seed,
                        train_ratio=internal_train_ratio,
                    )
                )
                internal_eval_count = count - internal_train_count
                internal_count = internal_train_count if internal_info.virtual_split == "train" else internal_eval_count
                sample_count = min(int(requested_limit), internal_count) if requested_limit is not None else internal_count
            else:
                sample_count = min(int(sample_limit), count)
            raw_sample_count = count
            effective_sample_count = sample_count
            limit_applied_before_formatting = internal_info is not None
    except Exception as exc:  # noqa: BLE001
        raise _dataset_error(
            f"formatter preflight failed while reading local data: {exc!r}.",
            variant=variant,
            formatter_name=formatter_name,
        ) from exc

    if variant.base_dataset == "mmlu":
        expected_local_split = mmlu_actual_split(variant.split)
    elif internal_info is not None:
        expected_local_split = internal_info.source_split
    else:
        expected_local_split = variant.split
    if expected_local_split not in available_raw_splits:
        raise _dataset_error(
            "local dataset does not contain the requested split.",
            variant=variant,
            formatter_name=formatter_name,
            available_splits=available_raw_splits,
        )
    if sample_count <= 0:
        raise _dataset_error(
            "local dataset split exists but contains no rows.",
            variant=variant,
            formatter_name=formatter_name,
            available_splits=available_raw_splits,
        )

    return {
        **asdict(variant),
        "formatter": formatter_name,
        "local_path": str(local_path),
        "available_splits": available_raw_splits,
        "actual_mmlu_split": mmlu_actual_split(variant.split) if variant.base_dataset == "mmlu" else None,
        "source_dataset": internal_info.source_variant if internal_info is not None else None,
        "virtual_split": internal_info.virtual_split if internal_info is not None else None,
        "split_policy": "internal_split" if internal_info is not None else None,
        "raw_sample_count": raw_sample_count,
        "formatted_sample_count": formatted_sample_count,
        "internal_train_count": internal_train_count,
        "internal_eval_count": internal_eval_count,
        "requested_limit": requested_limit,
        "effective_sample_count": effective_sample_count,
        "limit_applied_before_formatting": limit_applied_before_formatting,
        "sample_count": sample_count,
        "suggestion": _suggest_response_cache_command(variant),
    }


if __name__ == "__main__":  # do cache generation
    pass
