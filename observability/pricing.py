"""Cost estimation from configured prices.

**This table is a configured snapshot, not a live feed.** Prices change, and a
stale number presented as fact is worse than no number. So:

* A model not in the table produces ``usd = None`` with a reason. It is never
  estimated from a "similar" model.
* Usage the provider did not report produces ``usd = None``. It is never
  inferred from text length.
* Every cost is labelled an *estimate*, because the authoritative figure is
  the provider's own billing.

Override at runtime with ``LLM_PRICE_INPUT_PER_MTOK`` and
``LLM_PRICE_OUTPUT_PER_MTOK`` when using a model that is not listed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from observability.events import TokenCost
from providers.base import TokenUsage

# Source: Anthropic published list prices, recorded 2026-10-06. USD per
# million tokens. Verify before relying on these for real accounting.
PRICING_SNAPSHOT_DATE = "2026-10-06"


@dataclass(frozen=True)
class ModelPricing:
    input_per_mtok: float
    output_per_mtok: float


PRICING: dict[str, ModelPricing] = {
    "claude-opus-5-5": ModelPricing(4.00, 20.00),
    "claude-opus-5": ModelPricing(5.00, 25.00),
    "claude-sonnet-5-5": ModelPricing(2.00, 10.00),
    "claude-sonnet-5": ModelPricing(2.00, 10.00),
    "claude-haiku-5-5": ModelPricing(0.10, 0.50),
    "claude-fable-5-1": ModelPricing(10.00, 50.00),
}


def _override_from_env() -> ModelPricing | None:
    """Prices supplied by the operator, for an unlisted model."""
    raw_in = os.getenv("LLM_PRICE_INPUT_PER_MTOK")
    raw_out = os.getenv("LLM_PRICE_OUTPUT_PER_MTOK")
    if not raw_in or not raw_out:
        return None
    try:
        return ModelPricing(float(raw_in), float(raw_out))
    except ValueError:
        return None


def get_pricing(model: str) -> ModelPricing | None:
    """Look up a model's configured price, if there is one."""
    override = _override_from_env()
    if override is not None:
        return override
    return PRICING.get(model)


def estimate_cost(model: str, usage: TokenUsage | None, provider: str = "") -> TokenCost:
    """Estimate the USD cost of one call, or explain why it cannot be.

    Returns a ``TokenCost`` whose ``usd`` is ``None`` whenever the inputs do
    not support an honest number, with ``reason`` saying which input was
    missing.
    """
    if provider == "mock":
        return TokenCost(
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
            usd=None,
            priced=False,
            reason="mock provider: token counts are estimates and nothing was billed",
        )

    if usage is None or usage.input_tokens is None or usage.output_tokens is None:
        return TokenCost(
            input_tokens=usage.input_tokens if usage else None,
            output_tokens=usage.output_tokens if usage else None,
            usd=None,
            priced=False,
            reason="provider did not report token usage",
        )

    pricing = get_pricing(model)
    if pricing is None:
        return TokenCost(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            usd=None,
            priced=False,
            reason=(
                f"no configured price for {model!r}; set LLM_PRICE_INPUT_PER_MTOK "
                "and LLM_PRICE_OUTPUT_PER_MTOK to enable cost estimates"
            ),
        )

    usd = (
        usage.input_tokens / 1_000_000 * pricing.input_per_mtok
        + usage.output_tokens / 1_000_000 * pricing.output_per_mtok
    )
    return TokenCost(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        usd=usd,
        priced=True,
        reason=None,
    )
