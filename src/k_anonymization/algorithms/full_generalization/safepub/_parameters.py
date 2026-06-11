"""SafePub parameter calculation.

This module ports the ARX SafePub parameter calculation from
`org.deidentifier.arx.dp.ParameterCalculation` into standalone Python.
The Java implementation uses interval arithmetic for conservative floating
point bounds. This Python version follows the same formulae with regular
double precision floats and preserves the ARX-compatible conservative loop
that returns the post-incremented `k`.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import MutableMapping


class ParameterCalculationError(RuntimeError):
    """Raised when SafePub parameters cannot be calculated."""


@dataclass(frozen=True)
class SafePubParameters:
    """Calculated SafePub parameters."""

    epsilon: float
    delta: float
    beta: float
    gamma: float
    k: int


class ParameterCalculation:
    """Calculate SafePub's `(k, beta)` parameters from `(epsilon, delta)`."""

    def __init__(
        self,
        epsilon: float,
        delta: float,
        *,
        cache_size: int = 1000,
        max_k: int = 100_000,
        max_delta_scan_steps: int = 1_000_000,
    ) -> None:
        if epsilon <= 0 or not math.isfinite(epsilon):
            raise ValueError("epsilon must be a finite number > 0")
        if delta <= 0 or delta >= 1 or not math.isfinite(delta):
            raise ValueError("delta must be a finite number in (0, 1)")
        if cache_size < 1:
            raise ValueError("cache_size must be >= 1")

        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.cache_size = int(cache_size)
        self.max_k = int(max_k)
        self.max_delta_scan_steps = int(max_delta_scan_steps)

        self.beta = self._calculate_beta(self.epsilon)
        self.gamma = self._calculate_gamma(self.epsilon, self.beta)
        self.k = self._calculate_k()

    def get_beta(self) -> float:
        """Return SafePub's sampling probability beta."""

        return self.beta

    def get_k(self) -> int:
        """Return SafePub's k threshold."""

        return self.k

    def as_parameters(self) -> SafePubParameters:
        """Return all calculated values as an immutable dataclass."""

        return SafePubParameters(
            epsilon=self.epsilon,
            delta=self.delta,
            beta=self.beta,
            gamma=self.gamma,
            k=self.k,
        )

    @staticmethod
    def _calculate_beta(epsilon: float) -> float:
        return -math.expm1(-epsilon)

    @staticmethod
    def _calculate_gamma(epsilon: float, beta: float) -> float:
        # Equivalent to `(exp(epsilon) - 1 + beta) / exp(epsilon)`.
        # The expm1 form avoids overflow for large epsilon.
        gamma = -math.expm1(-2.0 * epsilon)
        if not (0.0 < beta < gamma < 1.0):
            raise ParameterCalculationError(
                f"invalid SafePub beta/gamma values: beta={beta}, gamma={gamma}"
            )
        return gamma

    def _calculate_a(self, n: int, beta: float, gamma: float) -> float:
        lower = math.floor(n * gamma) + 1
        return _binomial_tail(n, beta, lower)

    def _calculate_c(self, n: int, beta: float, gamma: float) -> float:
        divergence = gamma * math.log(gamma / beta) - (gamma - beta)
        if divergence <= 0.0:
            raise ParameterCalculationError(
                f"invalid SafePub divergence term: {divergence}"
            )
        exponent = -float(n) * divergence
        if exponent < -745.0:
            return 0.0
        return math.exp(exponent)

    def _calculate_delta(
        self,
        k: int,
        beta: float,
        gamma: float,
        a_cache: MutableMapping[int, float],
        c_cache: MutableMapping[int, float],
    ) -> float:
        n = max(0, math.ceil(float(k) / gamma - 1.0))
        delta_k = -math.inf
        bound = math.inf
        steps = 0

        while delta_k <= bound:
            if n not in a_cache:
                _lru_put(a_cache, n, self._calculate_a(n, beta, gamma), self.cache_size)
                _lru_put(c_cache, n, self._calculate_c(n, beta, gamma), self.cache_size)
            delta_k = max(delta_k, a_cache[n])
            bound = c_cache[n]
            n += 1
            steps += 1
            if steps > self.max_delta_scan_steps:
                raise ParameterCalculationError(
                    f"delta calculation did not converge for k={k}"
                )

        return delta_k

    def _calculate_k(self) -> int:
        a_cache: OrderedDict[int, float] = OrderedDict()
        c_cache: OrderedDict[int, float] = OrderedDict()

        k = 1
        delta_k = math.inf

        while not (delta_k < self.delta):
            if k > self.max_k:
                raise ParameterCalculationError(
                    f"k calculation exceeded max_k={self.max_k}"
                )
            delta_k = self._calculate_delta(k, self.beta, self.gamma, a_cache, c_cache)
            k += 1

        return k


def _lru_put(cache: MutableMapping[int, float], key: int, value: float, limit: int) -> None:
    cache[key] = value
    if isinstance(cache, OrderedDict):
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)


def _binomial_tail(n: int, probability: float, lower: int) -> float:
    """Return P[X >= lower] for X ~ Binomial(n, probability)."""

    if n < 0:
        raise ValueError("n must be >= 0")
    if lower <= 0:
        return 1.0
    if lower > n:
        return 0.0
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 1.0

    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    logs: list[float] = []
    max_log = -math.inf

    for j in range(lower, n + 1):
        log_value = (
            math.lgamma(n + 1)
            - math.lgamma(j + 1)
            - math.lgamma(n - j + 1)
            + j * log_p
            + (n - j) * log_q
        )
        logs.append(log_value)
        if log_value > max_log:
            max_log = log_value

    if max_log == -math.inf:
        return 0.0

    total = sum(math.exp(value - max_log) for value in logs)
    result = math.exp(max_log) * total
    return min(1.0, max(0.0, result))
