"""Open-domain QA answer normalization, scoring, and audit helpers."""

from __future__ import annotations

import csv
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


DEFAULT_QA_F1_THRESHOLD = 0.5

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_PREFIX_RE = re.compile(
    r"^\s*(?:final\s+answer|answer|ans|a)\s*[:：\-]\s*",
    flags=re.IGNORECASE,
)
_EXPLANATION_MARKERS = (
    " because ",
    " since ",
    " therefore ",
    " thus ",
    " so ",
    " -- ",
    " - ",
    ";",
)
_PLACEHOLDER_ANSWERS = {
    "",
    "none",
    "null",
    "n/a",
    "na",
    "unknown",
    "i don't know",
    "i do not know",
    "not sure",
    "no answer",
}

AUDIT_FIELDNAMES = [
    "sample_id",
    "question",
    "raw_model_answer",
    "extracted_answer",
    "gold_answer",
    "gold_aliases",
    "normalized_prediction",
    "normalized_gold_answers",
    "exact_match",
    "token_f1",
    "qa_correct",
    "qa_f1_threshold",
    "rouge_score",
    "bleu_score",
    "parse_status",
    "parse_error_reason",
]


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            text = _stringify(item)
            if text.strip():
                return text
        return ""
    return str(value)


def strip_answer_prefix(text: Any) -> str:
    out = _stringify(text).strip()
    previous = None
    while out and out != previous:
        previous = out
        out = _PREFIX_RE.sub("", out, count=1).strip()
    return out


def extract_answer(raw_answer: Any) -> tuple[str, str, Optional[str]]:
    raw = _stringify(raw_answer)
    if raw_answer is None:
        return "", "missing", "raw answer is None"
    if not raw.strip():
        return "", "empty", "raw answer is empty"

    first_line = raw.strip().splitlines()[0].strip()
    answer = strip_answer_prefix(first_line).strip().strip('"\'`')
    lowered = answer.lower().strip()
    if lowered in _PLACEHOLDER_ANSWERS:
        return "", "placeholder", f"placeholder answer: {answer}"

    cut_at = None
    lowered_for_markers = f" {answer.lower()} "
    for marker in _EXPLANATION_MARKERS:
        idx = lowered_for_markers.find(marker)
        if idx > 0:
            cut_at = idx - 1
            break
    if cut_at is not None:
        answer = answer[:cut_at].strip()

    if not answer:
        return "", "empty_after_parse", "answer became empty after parsing"
    return answer, "ok", None


def normalize_answer(text: Any) -> str:
    out = strip_answer_prefix(text).lower()
    out = out.translate(str.maketrans({ch: " " for ch in string.punctuation}))
    out = _ARTICLE_RE.sub(" ", out)
    out = _WHITESPACE_RE.sub(" ", out).strip()
    return out


def _flatten_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        out: list[str] = []
        for key in ("value", "text", "normalized_value", "aliases", "normalized_aliases"):
            out.extend(_flatten_values(value.get(key)))
        return out
    if isinstance(value, Iterable):
        out: list[str] = []
        for item in value:
            out.extend(_flatten_values(item))
        return out
    return [str(value)]


def gold_answers_from_row(row: Mapping[str, Any]) -> tuple[str, list[str]]:
    gold_answer = _stringify(
        row.get("gold_answer")
        or row.get("answer_value")
        or row.get("answer_str")
        or row.get("answer")
    )
    candidates: list[str] = []
    for key in (
        "gold_aliases",
        "answer_aliases",
        "aliases",
        "normalized_aliases",
        "answer",
        "answer_str",
        "gold_answer",
    ):
        candidates.extend(_flatten_values(row.get(key)))
    if gold_answer:
        candidates.append(gold_answer)

    seen: set[str] = set()
    aliases: list[str] = []
    for candidate in candidates:
        text = strip_answer_prefix(candidate)
        if not text:
            continue
        key = normalize_answer(text)
        if key and key not in seen:
            seen.add(key)
            aliases.append(text)
    return strip_answer_prefix(gold_answer), aliases


def token_f1(prediction: Any, ground_truth: Any) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(gold_tokens)
    return float(2 * precision * recall / (precision + recall))


def score_open_domain_qa(
    row: Mapping[str, Any],
    *,
    sample_id: int,
    f1_threshold: float = DEFAULT_QA_F1_THRESHOLD,
    rouge_score: Optional[float] = None,
    bleu_score: Optional[float] = None,
) -> dict[str, Any]:
    raw_model_answer = _stringify(row.get("most_likely_answer"))
    extracted, parse_status, parse_error_reason = extract_answer(raw_model_answer)
    gold_answer, gold_aliases = gold_answers_from_row(row)
    normalized_prediction = normalize_answer(extracted)
    normalized_gold_answers = [normalize_answer(ans) for ans in gold_aliases]
    normalized_gold_answers = [ans for ans in normalized_gold_answers if ans]

    exact_match = bool(
        normalized_prediction
        and any(normalized_prediction == gold for gold in normalized_gold_answers)
    )
    f1_scores = [token_f1(extracted, alias) for alias in gold_aliases]
    best_f1 = float(max(f1_scores) if f1_scores else 0.0)
    qa_correct = bool(exact_match or best_f1 >= float(f1_threshold))

    if not normalized_gold_answers:
        parse_status = "missing_gold"
        parse_error_reason = "gold answer and aliases are missing"
        qa_correct = False

    return {
        "sample_id": sample_id,
        "question": _stringify(row.get("question_str") or row.get("question")),
        "raw_model_answer": raw_model_answer,
        "extracted_answer": extracted,
        "gold_answer": gold_answer,
        "gold_aliases": gold_aliases,
        "normalized_prediction": normalized_prediction,
        "normalized_gold_answers": normalized_gold_answers,
        "exact_match": exact_match,
        "token_f1": best_f1,
        "qa_correct": qa_correct,
        "qa_f1_threshold": float(f1_threshold),
        "rouge_score": rouge_score,
        "bleu_score": bleu_score,
        "parse_status": parse_status,
        "parse_error_reason": parse_error_reason,
    }


def write_answer_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    csv_path: str | Path,
    json_path: str | Path,
) -> None:
    csv_file = Path(csv_path)
    json_file = Path(json_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    json_file.parent.mkdir(parents=True, exist_ok=True)

    normalized_rows = []
    for row in rows:
        out = {field: row.get(field) for field in AUDIT_FIELDNAMES}
        normalized_rows.append(out)

    with csv_file.open("w", newline="", encoding="utf-8") as fw:
        writer = csv.DictWriter(fw, fieldnames=AUDIT_FIELDNAMES)
        writer.writeheader()
        for row in normalized_rows:
            csv_row = dict(row)
            for key in ("gold_aliases", "normalized_gold_answers"):
                csv_row[key] = json.dumps(csv_row.get(key) or [], ensure_ascii=False)
            writer.writerow(csv_row)

    json_file.write_text(
        json.dumps(normalized_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
