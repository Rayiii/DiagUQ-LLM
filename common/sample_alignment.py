"""Sample-id alignment checks shared by DiagUQ stages."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def sample_ids_from_rows(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(row.get("sample_id")) for row in rows]


def require_no_sample_id_overlap(
    left_ids: Sequence[Any],
    right_ids: Sequence[Any],
    *,
    left_label: str,
    right_label: str,
) -> None:
    left = {str(value) for value in left_ids if value not in (None, "")}
    right = {str(value) for value in right_ids if value not in (None, "")}
    overlap = sorted(left & right)
    if overlap:
        raise ValueError(
            f"sample_id overlap between {left_label} and {right_label}: "
            f"count={len(overlap)} examples={overlap[:5]}"
        )


def require_matching_sample_ids(
    expected_ids: Sequence[Any],
    actual_ids: Sequence[Any],
    *,
    expected_label: str,
    actual_label: str,
) -> None:
    expected = [str(value) for value in expected_ids]
    actual = [str(value) for value in actual_ids]
    if any(value in {"None", ""} for value in expected):
        raise ValueError(f"{expected_label} contains missing sample_id values")
    if any(value in {"None", ""} for value in actual):
        raise ValueError(f"{actual_label} contains missing sample_id values")
    if expected == actual:
        return
    expected_set = set(expected)
    actual_set = set(actual)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    first_mismatch = None
    for idx, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            first_mismatch = {"index": idx, expected_label: left, actual_label: right}
            break
    raise ValueError(
        f"sample_id alignment mismatch: {expected_label} vs {actual_label}; "
        f"expected_count={len(expected)} actual_count={len(actual)} "
        f"missing_in_actual={missing[:5]} extra_in_actual={extra[:5]} "
        f"first_mismatch={first_mismatch}"
    )
