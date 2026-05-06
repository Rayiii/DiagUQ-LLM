"""Response-cache row-limit policy shared by CLI preflight and generation."""

from __future__ import annotations

from typing import Optional


DEFAULT_RESPONSE_CACHE_LIMIT = 20_000
DEFAULT_RESPONSE_CACHE_SPLIT_LIMITS = {
    "wmt__test": 2_000,
}


def resolve_response_cache_limit(
    split_tag: str,
    requested_limit: Optional[int],
    *,
    allow_full_formatting: bool = False,
) -> Optional[int]:
    """Return the formatter cap for one response-cache split.

    Explicit user limits are honored exactly. Without a user limit, response-cache
    generation uses conservative defaults; passing ``allow_full_formatting`` is
    the only way to intentionally request an uncapped formatter call.
    """
    if requested_limit is not None:
        limit = int(requested_limit)
        if limit < 0:
            raise ValueError(f"response-cache limit must be non-negative, got {requested_limit!r}")
        return limit
    if allow_full_formatting:
        return None
    return int(DEFAULT_RESPONSE_CACHE_SPLIT_LIMITS.get(split_tag, DEFAULT_RESPONSE_CACHE_LIMIT))
