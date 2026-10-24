"""Discrete exponential mechanism used by SafePub's data-dependent search."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Generic, Iterable, Sequence, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class WeightedValue(Generic[T]):
    """A value and its normalized sampling probability."""

    value: T
    probability: float


class ExponentialMechanism(Generic[T]):
    """Sample candidates with probability proportional to exp(0.5 * epsilon * score)."""

    def __init__(
        self,
        epsilon: float,
        *,
        deterministic: bool = False,
        seed: int = 0xDEADBEEF,
    ) -> None:
        if epsilon < 0 or not math.isfinite(epsilon):
            raise ValueError("epsilon must be a finite number >= 0")
        self.epsilon = float(epsilon)
        self._random = random.Random(seed) if deterministic else random.SystemRandom()
        self._weighted_values: list[WeightedValue[T]] = []

    def set_distribution(self, values: Sequence[T], scores: Sequence[float]) -> None:
        """Set the candidate distribution."""

        if not values:
            raise ValueError("No values supplied")
        if len(values) != len(scores):
            raise ValueError("Number of scores and values must be identical")
        if any(not math.isfinite(score) for score in scores):
            raise ValueError("scores must be finite numbers")

        exponents = [0.5 * self.epsilon * float(score) for score in scores]
        shift = max(exponents)
        weights = [math.exp(exponent - shift) for exponent in exponents]
        total = sum(weights)
        if total <= 0.0 or not math.isfinite(total):
            raise ValueError("Could not normalize exponential-mechanism weights")

        self._weighted_values = [
            WeightedValue(value=value, probability=weight / total)
            for value, weight in zip(values, weights)
        ]

    def distribution(self) -> tuple[WeightedValue[T], ...]:
        """Return the current normalized distribution."""

        return tuple(self._weighted_values)

    def sample(self) -> T:
        """Sample one value from the current distribution."""

        if not self._weighted_values:
            raise ValueError("Distribution has not been set")

        threshold = self._random.random()
        cumulative = 0.0
        for weighted_value in self._weighted_values:
            cumulative += weighted_value.probability
            if threshold <= cumulative:
                return weighted_value.value
        return self._weighted_values[-1].value

    def choose(self, values_and_scores: Iterable[tuple[T, float]]) -> T:
        """Set a distribution from `(value, score)` pairs and sample from it."""

        values: list[T] = []
        scores: list[float] = []
        for value, score in values_and_scores:
            values.append(value)
            scores.append(score)
        self.set_distribution(values, scores)
        return self.sample()
