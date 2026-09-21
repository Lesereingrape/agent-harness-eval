"""Statistics for paired harness comparisons.

Two harnesses evaluated on the same task set produce *paired* scores; naive
mean-difference reporting hides whether a gain is real. `paired_bootstrap`
gives a CI and win-probability on the mean delta; `mcnemar_exact` is the
standard exact test for binary success outcomes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import comb


@dataclass(frozen=True)
class BootstrapResult:
    mean_delta: float
    ci_low: float
    ci_high: float
    p_win: float  # bootstrap probability that harness B beats harness A

    def summary(self) -> str:
        return (
            f"delta={self.mean_delta:+.3f} "
            f"95% CI=[{self.ci_low:+.3f}, {self.ci_high:+.3f}] "
            f"P(B>A)={self.p_win:.3f}"
        )


def paired_bootstrap(
    a: list[float],
    b: list[float],
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapResult:
    if len(a) != len(b) or not a:
        raise ValueError("paired samples must be equal-length and non-empty")
    rng = random.Random(seed)
    n = len(a)
    deltas = [bi - ai for ai, bi in zip(a, b)]
    means = sorted(
        sum(deltas[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(n_resamples)
    )
    lo = means[int((alpha / 2) * n_resamples)]
    hi = means[int((1 - alpha / 2) * n_resamples)]
    p_win = sum(1 for m in means if m > 0) / n_resamples
    return BootstrapResult(
        mean_delta=sum(deltas) / n, ci_low=lo, ci_high=hi, p_win=p_win
    )


def mcnemar_exact(success_a: list[bool], success_b: list[bool]) -> float:
    """Two-sided exact McNemar p-value on paired binary outcomes."""
    if len(success_a) != len(success_b):
        raise ValueError("samples must be equal-length")
    b01 = sum(1 for x, y in zip(success_a, success_b) if not x and y)  # A wrong, B right
    b10 = sum(1 for x, y in zip(success_a, success_b) if x and not y)
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)
