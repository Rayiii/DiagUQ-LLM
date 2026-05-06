"""Local MMLU Arrow dataset loading and formatting helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


MMLU_LOCAL_SPLITS = ("dev", "validation", "test")
MMLU_REQUESTED_SPLITS = ("train", "validation", "dev", "test")
MMLU_OPTION_LETTERS = ("A", "B", "C", "D")
MMLU_EXPECTED_SCHEMAS = (
    "Schema A: question, choices, answer",
    "Schema B: input, choices or A/B/C/D, target or answer",
    "Schema C: Question, A, B, C, D, Answer",
)


def mmlu_actual_split(requested_split: str) -> str:
    """Map DiagUQ requested MMLU splits to local Arrow split names."""
    split = str(requested_split or "").strip().lower()
    if split == "train":
        return "validation"
    if split in {"validation", "eval"}:
        return "test"
    if split == "dev":
        return "dev"
    if split == "test":
        return "test"
    raise ValueError(
        f"Unsupported MMLU requested split {requested_split!r}. "
        f"Expected one of {MMLU_REQUESTED_SPLITS}."
    )


def discover_mmlu_subjects(mmlu_root: str | Path) -> list[Path]:
    root = Path(mmlu_root)
    if not root.exists():
        raise FileNotFoundError(f"MMLU root does not exist: {root}")
    subjects = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and (
            (path / "dataset_dict.json").exists()
            or any((path / split).exists() for split in MMLU_LOCAL_SPLITS)
        )
    ]
    return sorted(subjects, key=lambda path: path.name)


def _load_datasets_module():
    import datasets  # type: ignore

    return datasets


def load_mmlu_subject_split(subject_dir: str | Path, split: str):
    """Load one subject split from ``<subject_dir>/<split>`` or DatasetDict."""
    subject_path = Path(subject_dir)
    split_name = str(split)
    split_path = subject_path / split_name
    datasets = _load_datasets_module()
    if split_path.exists():
        return datasets.load_from_disk(str(split_path))

    payload = datasets.load_from_disk(str(subject_path))
    if split_name not in payload:
        available = tuple(payload.keys()) if hasattr(payload, "keys") else ()
        raise KeyError(
            f"MMLU subject split missing: subject={subject_path.name!r} "
            f"split={split_name!r} available_splits={available} path={subject_path}"
        )
    return payload[split_name]


def load_local_mmlu_split(
    mmlu_root: str | Path,
    split: str,
    subjects: Optional[Sequence[str]] = None,
):
    """Load and concatenate local MMLU rows for a requested DiagUQ split."""
    requested_split = str(split)
    actual_split = mmlu_actual_split(requested_split)
    requested_subjects = set(subjects or [])
    datasets = _load_datasets_module()
    pieces = []
    for subject_dir in discover_mmlu_subjects(mmlu_root):
        subject = subject_dir.name
        if requested_subjects and subject not in requested_subjects:
            continue
        ds = load_mmlu_subject_split(subject_dir, actual_split)

        def add_metadata(row, idx, *, subject=subject):
            row = dict(row)
            row["subject"] = subject
            row["requested_split"] = requested_split
            row["actual_mmlu_split"] = actual_split
            row.setdefault("sample_id", f"mmlu:{subject}:{requested_split}:{idx}")
            return row

        pieces.append(ds.map(add_metadata, with_indices=True))
    if not pieces:
        raise FileNotFoundError(
            f"No MMLU subject splits found: root={mmlu_root} "
            f"requested_split={requested_split!r} actual_mmlu_split={actual_split!r} "
            f"subjects={sorted(requested_subjects) if requested_subjects else 'all'}"
        )
    return pieces[0] if len(pieces) == 1 else datasets.concatenate_datasets(pieces)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _short_row_sample(row: Mapping[str, Any], max_chars: int = 500) -> str:
    sample = {}
    for idx, (key, value) in enumerate(row.items()):
        if idx >= 12:
            sample["..."] = "..."
            break
        text = repr(value)
        sample[str(key)] = text[:120] + ("..." if len(text) > 120 else "")
    rendered = repr(sample)
    return rendered[:max_chars] + ("..." if len(rendered) > max_chars else "")


def _unsupported_schema_error(row: Mapping[str, Any], subject: str, split: str) -> ValueError:
    return ValueError(
        "Unsupported MMLU row schema: "
        f"subject={subject!r} split={split!r} row_keys={sorted(map(str, row.keys()))} "
        f"row_sample={_short_row_sample(row)} expected_schemas={MMLU_EXPECTED_SCHEMAS}"
    )


def _normalize_choice_list(raw_choices: Any, row: Mapping[str, Any], subject: str, split: str) -> list[str]:
    choices: list[str] = []
    if raw_choices is not None:
        if isinstance(raw_choices, Mapping):
            if all(letter in raw_choices for letter in MMLU_OPTION_LETTERS):
                choices = [_stringify(raw_choices[letter]) for letter in MMLU_OPTION_LETTERS]
            elif "text" in raw_choices:
                choices = [_stringify(item) for item in raw_choices.get("text") or []]
        elif isinstance(raw_choices, Sequence) and not isinstance(raw_choices, (str, bytes, bytearray)):
            choices = [_stringify(item) for item in raw_choices]
    if not choices and all(letter in row for letter in MMLU_OPTION_LETTERS):
        choices = [_stringify(row[letter]) for letter in MMLU_OPTION_LETTERS]
    choices = [choice for choice in choices if choice]
    if len(choices) < 4:
        raise _unsupported_schema_error(row, subject, split)
    return choices[:4]


def _normalize_text_for_match(text: Any) -> str:
    value = _stringify(text).lower()
    value = re.sub(r"^\s*(?:answer|the answer is|option)\s*[:\-]?\s*", "", value)
    value = re.sub(r"^[\(\[]?([abcd])[\)\].:]\s*", r"\1", value)
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^a-z0-9 ]", "", value)
    return value.strip()


def normalize_mmlu_gold_answer(answer: Any, choices: Sequence[str]) -> tuple[str, str]:
    if isinstance(answer, int) and not isinstance(answer, bool):
        if 0 <= int(answer) < len(MMLU_OPTION_LETTERS):
            option = MMLU_OPTION_LETTERS[int(answer)]
            return option, str(choices[int(answer)])

    text = _stringify(answer)
    upper = text.upper().strip()
    if upper in MMLU_OPTION_LETTERS:
        idx = MMLU_OPTION_LETTERS.index(upper)
        return upper, str(choices[idx])

    normalized = _normalize_text_for_match(text)
    for idx, choice in enumerate(choices):
        if normalized and normalized == _normalize_text_for_match(choice):
            return MMLU_OPTION_LETTERS[idx], str(choice)

    parsed, status, _ = parse_mmlu_option_answer(text, choices=choices)
    if status == "ok" and parsed:
        idx = MMLU_OPTION_LETTERS.index(parsed)
        return parsed, str(choices[idx])

    raise ValueError(f"Unsupported MMLU gold answer: answer={answer!r} choices={list(choices)!r}")


def format_mmlu_prompt(question: str, choices: Sequence[str]) -> str:
    lines = ["Question:", question.strip(), "", "Choices:"]
    lines.extend(f"{letter}. {choice}" for letter, choice in zip(MMLU_OPTION_LETTERS, choices))
    lines.extend(["", "Answer with only the option letter.", "Answer: "])
    return "\n".join(lines)


def normalize_mmlu_row(
    row: Mapping[str, Any],
    subject: str,
    split: str,
    *,
    requested_split: Optional[str] = None,
    row_index: Optional[int] = None,
) -> dict[str, Any]:
    row = dict(row)
    actual_split = str(split)
    requested = requested_split or str(split)

    if "question" in row and "choices" in row and "answer" in row:
        question = _stringify(row["question"])
        choices = _normalize_choice_list(row.get("choices"), row, subject, actual_split)
        answer = row.get("answer")
    elif "input" in row and ("target" in row or "answer" in row):
        question = _stringify(row["input"])
        choices = _normalize_choice_list(row.get("choices"), row, subject, actual_split)
        answer = row.get("target", row.get("answer"))
    elif "Question" in row and all(letter in row for letter in MMLU_OPTION_LETTERS) and "Answer" in row:
        question = _stringify(row["Question"])
        choices = _normalize_choice_list(None, row, subject, actual_split)
        answer = row.get("Answer")
    else:
        raise _unsupported_schema_error(row, subject, actual_split)

    if not question:
        raise _unsupported_schema_error(row, subject, actual_split)
    gold_option, gold_answer_text = normalize_mmlu_gold_answer(answer, choices)
    sample_id = row.get("sample_id") or f"mmlu:{subject}:{requested}:{row_index if row_index is not None else 0}"
    resolved_variant = f"mmlu__{requested}"
    prompt = format_mmlu_prompt(question, choices)
    return {
        "sample_id": sample_id,
        "dataset": "mmlu",
        "resolved_variant": resolved_variant,
        "split": requested,
        "requested_split": requested,
        "actual_mmlu_split": actual_split,
        "subject": subject,
        "question": question,
        "choices": list(choices),
        "prompt": prompt,
        "input": prompt,
        "gold_option": gold_option,
        "gold_answer_text": gold_answer_text,
        "target": gold_option,
        "answer_str": gold_option,
        "gold_answer": gold_option,
        "gold_aliases": [gold_option, gold_answer_text],
        "question_str": prompt,
    }


def parse_mmlu_option_answer(text: Any, choices: Optional[Sequence[str] | Mapping[str, str]] = None) -> tuple[str, str, Optional[str]]:
    raw = _stringify(text)
    if not raw:
        return "", "empty", "answer text is empty"

    match = re.search(
        r"(?:^|\b)(?:the\s+answer\s+is\s+|answer\s*[:\-]?\s*|option\s*)?[\(\[]?([ABCDabcd])[\)\].:]?(?:\b|$)",
        raw,
    )
    if match:
        return match.group(1).upper(), "ok", None

    if choices is not None:
        if isinstance(choices, Mapping):
            choice_items = [(letter, choices.get(letter, "")) for letter in MMLU_OPTION_LETTERS]
        else:
            choice_items = list(zip(MMLU_OPTION_LETTERS, choices))
        normalized = _normalize_text_for_match(raw)
        for letter, choice_text in choice_items:
            if normalized and normalized == _normalize_text_for_match(choice_text):
                return letter, "ok", None

    return "", "unparseable", "could not parse an A/B/C/D option"
