"""OpenAI usage-cost estimation."""

from dataclasses import dataclass
from typing import Any


_TOKENS_PER_MILLION = 1_000_000
_REGIONAL_UPLIFT_MODELS = ("gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.4-pro")
_REGIONAL_UPLIFT_MULTIPLIER = 1.10


@dataclass(frozen=True)
class OpenAIModelPricing:
    """Per-1M-token pricing for one model family."""

    input_usd_per_million: float
    output_usd_per_million: float
    cached_input_usd_per_million: float | None = None


# Standard OpenAI pricing (USD / 1M tokens).
# Keep keys as model-family prefixes; longest prefix wins.
_OPENAI_PRICING_BY_PREFIX: dict[str, OpenAIModelPricing] = {
    "gpt-5.4-pro": OpenAIModelPricing(30.00, 180.00, None),
    "gpt-5.4-mini": OpenAIModelPricing(0.75, 4.50, 0.075),
    "gpt-5.4-nano": OpenAIModelPricing(0.20, 1.25, 0.02),
    "gpt-5.4": OpenAIModelPricing(2.50, 15.00, 0.25),
    "gpt-5.1-codex-mini": OpenAIModelPricing(0.25, 0.025, 2.00),
    "gpt-5.2-pro": OpenAIModelPricing(21.00, 168.00, None),
    "gpt-5.2": OpenAIModelPricing(1.75, 14.00, 0.175),
    "gpt-5-pro": OpenAIModelPricing(15.00, 120.00, None),
    "gpt-5-mini": OpenAIModelPricing(0.25, 2.00, 0.025),
    "gpt-5-nano": OpenAIModelPricing(0.05, 0.40, 0.005),
    "gpt-5.1": OpenAIModelPricing(1.25, 10.00, 0.125),
    "gpt-5": OpenAIModelPricing(1.25, 10.00, 0.125),
    "gpt-4.1-nano": OpenAIModelPricing(0.10, 0.40, 0.025),
    "gpt-4.1-mini": OpenAIModelPricing(0.40, 1.60, 0.10),
    "gpt-4.1": OpenAIModelPricing(2.00, 8.00, 0.50),
    "gpt-4o-mini": OpenAIModelPricing(0.15, 0.60, 0.075),
    "gpt-4o-2024-05-13": OpenAIModelPricing(5.00, 15.00, None),
    "gpt-4o": OpenAIModelPricing(2.50, 10.00, 1.25),
    "gpt-4-turbo-2024-04-09": OpenAIModelPricing(10.00, 30.00, None),
    "gpt-4-0125-preview": OpenAIModelPricing(10.00, 30.00, None),
    "gpt-4-1106-preview": OpenAIModelPricing(10.00, 30.00, None),
    "gpt-4-1106-vision-preview": OpenAIModelPricing(10.00, 30.00, None),
    "gpt-4-0613": OpenAIModelPricing(30.00, 60.00, None),
    "gpt-4-0314": OpenAIModelPricing(30.00, 60.00, None),
    "gpt-4-32k": OpenAIModelPricing(60.00, 120.00, None),
    "o4-mini": OpenAIModelPricing(1.10, 4.40, 0.275),
    "o3-mini": OpenAIModelPricing(1.10, 4.40, 0.55),
    "o3-pro": OpenAIModelPricing(20.00, 80.00, None),
    "o3": OpenAIModelPricing(2.00, 8.00, 0.50),
    "o1-mini": OpenAIModelPricing(1.10, 4.40, 0.55),
    "o1-pro": OpenAIModelPricing(150.00, 600.00, None),
    "o1": OpenAIModelPricing(15.00, 60.00, 7.50),
    "gpt-3.5-turbo-16k-0613": OpenAIModelPricing(3.00, 4.00, None),
    "gpt-3.5-turbo-instruct": OpenAIModelPricing(1.50, 2.00, None),
    "gpt-3.5-turbo-0125": OpenAIModelPricing(0.50, 1.50, None),
    "gpt-3.5-turbo-1106": OpenAIModelPricing(1.00, 2.00, None),
    "gpt-3.5-turbo-0613": OpenAIModelPricing(1.50, 2.00, None),
    "gpt-3.5-turbo": OpenAIModelPricing(0.50, 1.50, None),
    "gpt-3.5-0301": OpenAIModelPricing(1.50, 2.00, None),
    "davinci-002": OpenAIModelPricing(2.00, 2.00, None),
    "babbage-002": OpenAIModelPricing(0.40, 0.40, None),
}


def get_openai_model_pricing(model: str) -> OpenAIModelPricing | None:
    """Return pricing for a model id, using prefix matching."""
    normalized = model.strip().lower()
    if not normalized:
        return None
    for prefix in sorted(_OPENAI_PRICING_BY_PREFIX, key=len, reverse=True):
        if normalized.startswith(prefix):
            return _OPENAI_PRICING_BY_PREFIX[prefix]
    return None


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _cached_input_tokens(usage: dict[str, Any] | None) -> int:
    if not usage:
        return 0
    normalized_value = usage.get("cached_input_tokens")
    if normalized_value is not None:
        return _as_int(normalized_value, default=0)
    details = usage.get("input_tokens_details")
    if isinstance(details, list) and details:
        details = details[0]
    if not isinstance(details, dict):
        return 0
    return _as_int(details.get("cached_tokens"), default=0)


def estimate_openai_total_cost_usd(
    *,
    model: str,
    usage: dict[str, Any] | None,
    regional_processing: bool = False,
) -> float | None:
    """Estimate total request cost (USD) from OpenAI token usage.

    Returns ``None`` if the model has no configured pricing table.
    """
    pricing = get_openai_model_pricing(model)
    if pricing is None:
        return None

    input_tokens = _as_int((usage or {}).get("input_tokens"), default=0)
    output_tokens = _as_int((usage or {}).get("output_tokens"), default=0)
    cached_tokens = max(0, _cached_input_tokens(usage))
    non_cached_input_tokens = max(0, input_tokens - cached_tokens)

    input_cost_usd = (non_cached_input_tokens / _TOKENS_PER_MILLION) * pricing.input_usd_per_million
    output_cost_usd = (output_tokens / _TOKENS_PER_MILLION) * pricing.output_usd_per_million

    if pricing.cached_input_usd_per_million is not None:
        cached_input_cost_usd = (cached_tokens / _TOKENS_PER_MILLION) * pricing.cached_input_usd_per_million
    else:
        cached_input_cost_usd = (cached_tokens / _TOKENS_PER_MILLION) * pricing.input_usd_per_million

    total = input_cost_usd + cached_input_cost_usd + output_cost_usd
    normalized_model = model.strip().lower()
    if regional_processing and any(normalized_model.startswith(m) for m in _REGIONAL_UPLIFT_MODELS):
        total *= _REGIONAL_UPLIFT_MULTIPLIER
    return total
