from __future__ import annotations

from dataclasses import dataclass

from ragelo.types.formats import LLMUsage


@dataclass(frozen=True, slots=True)
class Price:
    """USD per million tokens. Cached input tokens are billed at `cached`, the rest of the input at `input`."""

    input: float
    output: float
    cached: float

    @classmethod
    def parse(cls, spec: str) -> tuple[str, Price]:
        model, _, rates = spec.rpartition("=")
        values = [float(rate) for rate in rates.split(",")]
        if not model or len(values) != 3:
            raise ValueError(f"Expected MODEL=INPUT,OUTPUT,CACHED in USD per million tokens, got {spec}")
        return model, cls(*values)

    def cost(self, usage: LLMUsage) -> float:
        fresh_input = usage.input_tokens - usage.cached_tokens
        return (
            fresh_input * self.input + usage.cached_tokens * self.cached + usage.output_tokens * self.output
        ) / 1_000_000


def usage_cells(usages: list[LLMUsage], n_judged: int, price: Price | None) -> list[str]:
    """Mean input, cached and output tokens, and USD per 1,000 judged. Blank unless every judged item has
    its usage: a mean over the few that do says nothing about the rest."""
    if len(usages) < n_judged or not usages:
        return ["-", "-", "-", "-"]
    n = len(usages)
    cells = [
        f"{sum(usage.input_tokens for usage in usages) / n:.0f}",
        f"{sum(usage.cached_tokens for usage in usages) / n:.0f}",
        f"{sum(usage.output_tokens for usage in usages) / n:.0f}",
    ]
    if price is None:
        return [*cells, "-"]
    return [*cells, f"{1000 * sum(price.cost(usage) for usage in usages) / n:.4f}"]
