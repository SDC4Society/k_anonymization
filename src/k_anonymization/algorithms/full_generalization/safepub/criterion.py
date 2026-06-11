"""Python representation of ARX's `(epsilon, delta)` SafePub criterion."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

from .parameters import ParameterCalculation, SafePubParameters

DEFAULT_SEARCH_BUDGET_RATIO = 0.10


@dataclass(frozen=True)
class SafePubInitialization:
    """Initialization output for the SafePub criterion."""

    parameters: SafePubParameters
    sampled_indices: tuple[int, ...]


class EDDifferentialPrivacy:
    """SafePub-backed `(epsilon, delta)` differential privacy criterion."""

    def __init__(
        self,
        epsilon: float,
        delta: float,
        *,
        data_dependent: bool = True,
        dp_search_budget: float | None = None,
        deterministic: bool = False,
        seed: int = 0xDEADBEEF,
    ) -> None:
        resolved_search_budget = (
            float(epsilon) * DEFAULT_SEARCH_BUDGET_RATIO
            if data_dependent and dp_search_budget is None
            else float(dp_search_budget or 0.0)
        )
        if resolved_search_budget < 0:
            raise ValueError("dp_search_budget must be >= 0")
        if data_dependent and resolved_search_budget >= epsilon:
            raise ValueError("dp_search_budget must be smaller than epsilon")

        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.data_dependent = bool(data_dependent)
        self.dp_search_budget = resolved_search_budget
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self._initialization: SafePubInitialization | None = None

    @property
    def parameters(self) -> SafePubParameters:
        if self._initialization is None:
            raise RuntimeError("criterion has not been initialized")
        return self._initialization.parameters

    @property
    def beta(self) -> float:
        return self.parameters.beta

    @property
    def k(self) -> int:
        return self.parameters.k

    @property
    def sampled_indices(self) -> tuple[int, ...]:
        if self._initialization is None:
            raise RuntimeError("criterion has not been initialized")
        return self._initialization.sampled_indices

    def initialize(self, record_count: int) -> SafePubInitialization:
        """Calculate `(beta, k)` and sample the SafePub subset."""

        if record_count < 0:
            raise ValueError("record_count must be >= 0")
        if self.data_dependent and self.dp_search_budget <= 0.0:
            raise ValueError("dp_search_budget must be > 0 for data-dependent SafePub")

        epsilon_for_anonymization = self.epsilon
        if self.data_dependent:
            epsilon_for_anonymization -= self.dp_search_budget

        parameters = ParameterCalculation(
            epsilon_for_anonymization,
            self.delta,
        ).as_parameters()

        rng = random.Random(self.seed) if self.deterministic else random.SystemRandom()
        sampled_indices = tuple(
            index for index in range(record_count) if rng.random() < parameters.beta
        )

        self._initialization = SafePubInitialization(
            parameters=parameters,
            sampled_indices=sampled_indices,
        )
        return self._initialization

    def is_anonymous(self, equivalence_class_size: int) -> bool:
        """Return whether an equivalence class satisfies SafePub's k threshold."""

        return equivalence_class_size >= self.k

    def are_classes_anonymous(self, class_sizes: Iterable[int]) -> bool:
        """Return whether all provided class sizes satisfy SafePub's k threshold."""

        return all(self.is_anonymous(size) for size in class_sizes)
