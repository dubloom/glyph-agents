"""Usage normalization helpers for provider backends."""

from collections.abc import Mapping
from typing import Any


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _extract_detail(usage: dict[str, Any], key: str) -> dict[str, Any]:
    details = usage.get(key)
    if isinstance(details, list) and details:
        details = details[0]
    return _as_dict(details)


def normalize_usage(provider: str, usage: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Normalize backend usage into a stable, flat schema."""
    if usage is None:
        return None

    usage_dict = _as_dict(usage)
    if not usage_dict:
        return None

    normalized_provider = provider.strip().lower()
    input_tokens = _as_int(usage_dict.get("input_tokens"), default=0)
    output_tokens = _as_int(usage_dict.get("output_tokens"), default=0)
    total_tokens = _as_int(usage_dict.get("total_tokens"), default=input_tokens + output_tokens)
    requests = _as_int(usage_dict.get("requests"), default=0)
    if requests <= 0:
        if isinstance(usage_dict.get("request_usage_entries"), list):
            requests = len(usage_dict["request_usage_entries"])
        elif normalized_provider in {"claude", "openai"}:
            requests = 1

    cached_input_tokens = _as_int(usage_dict.get("cached_input_tokens"), default=0)
    cache_creation_input_tokens = _as_int(usage_dict.get("cache_creation_input_tokens"), default=0)
    cache_read_input_tokens = _as_int(usage_dict.get("cache_read_input_tokens"), default=0)
    reasoning_tokens = _as_int(usage_dict.get("reasoning_tokens"), default=0)

    if normalized_provider == "openai":
        input_details = _extract_detail(usage_dict, "input_tokens_details")
        output_details = _extract_detail(usage_dict, "output_tokens_details")
        cached_input_tokens = _as_int(input_details.get("cached_tokens"), default=cached_input_tokens)
        reasoning_tokens = _as_int(output_details.get("reasoning_tokens"), default=reasoning_tokens)
    elif normalized_provider == "claude":
        # Claude exposes cache read/create counters directly.
        cache_creation_input_tokens = _as_int(
            usage_dict.get("cache_creation_input_tokens"), default=cache_creation_input_tokens
        )
        cache_read_input_tokens = _as_int(
            usage_dict.get("cache_read_input_tokens"), default=cache_read_input_tokens
        )
        cached_input_tokens = max(cached_input_tokens, cache_read_input_tokens)

    return {
        "requests": max(0, requests),
        "input_tokens": max(0, input_tokens),
        "output_tokens": max(0, output_tokens),
        "total_tokens": max(0, total_tokens),
        "cached_input_tokens": max(0, cached_input_tokens),
        "cache_creation_input_tokens": max(0, cache_creation_input_tokens),
        "cache_read_input_tokens": max(0, cache_read_input_tokens),
        "reasoning_tokens": max(0, reasoning_tokens),
    }
