"""Generic data-dependent SafePub-style search loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Hashable, TypeVar

from .exponential_mechanism import ExponentialMechanism


TransformationID = TypeVar("TransformationID", bound=Hashable)


@dataclass(frozen=True)
class SearchStep(Generic[TransformationID]):
    """One SafePub data-dependent search step."""

    step: int
    pivot: TransformationID
    score: float
    candidate_count: int


@dataclass(frozen=True)
class SearchResult(Generic[TransformationID]):
    """Result from the data-dependent SafePub-style search."""

    best: TransformationID
    best_score: float
    final_pivot: TransformationID
    steps: tuple[SearchStep[TransformationID], ...]
    stopped_early: bool


class DataDependentEDDPSearch(Generic[TransformationID]):
    """Port of ARX's `DataDependentEDDPAlgorithm` control flow.

    The ARX class is tightly coupled to the ARX transformation lattice and
    quality metrics. This Python class keeps the same pivot/candidate update
    loop but accepts small callables for predecessor lookup and score
    calculation, so it can be used with a custom Python lattice.

    Following ARX, every lattice predecessor is admitted as a candidate; with
    SafePub's record suppression there is no validity filtering, only scores.
    The optimum is tracked like `AbstractAlgorithm#trackOptimum`: a pivot
    becomes the new best when its score is higher, or equal with a lower
    transformation level (given via the optional `level` callable).
    """

    def __init__(
        self,
        *,
        top: TransformationID,
        predecessors: Callable[[TransformationID], list[TransformationID]],
        score: Callable[[TransformationID], float],
        expansion_limit: int,
        epsilon_search: float,
        deterministic: bool = False,
        level: Callable[[TransformationID], float] | None = None,
    ) -> None:
        if expansion_limit < 0:
            raise ValueError("expansion_limit must be >= 0")
        if epsilon_search < 0:
            raise ValueError("epsilon_search must be >= 0")

        self.top = top
        self.predecessors = predecessors
        self.score = score
        self.level = level
        self.expansion_limit = expansion_limit
        epsilon_per_step = (
            epsilon_search / float(expansion_limit) if expansion_limit != 0 else 0.0
        )
        self.exponential_mechanism: ExponentialMechanism[TransformationID] = (
            ExponentialMechanism(epsilon_per_step, deterministic=deterministic)
        )

    def traverse(self) -> SearchResult[TransformationID]:
        """Run the SafePub-style search."""

        score_cache: dict[TransformationID, float] = {}

        def assure_score(identifier: TransformationID) -> float:
            if identifier not in score_cache:
                score_cache[identifier] = float(self.score(identifier))
            return score_cache[identifier]

        pivot = self.top
        pivot_score = assure_score(pivot)
        best = pivot
        best_score = pivot_score
        candidate_scores: dict[TransformationID, float] = {pivot: pivot_score}
        steps: list[SearchStep[TransformationID]] = [
            SearchStep(step=0, pivot=pivot, score=pivot_score, candidate_count=1)
        ]

        for step in range(1, self.expansion_limit + 1):
            for predecessor in self.predecessors(pivot):
                if predecessor not in candidate_scores:
                    candidate_scores[predecessor] = assure_score(predecessor)

            candidate_scores.pop(pivot, None)
            if not candidate_scores:
                return SearchResult(
                    best=best,
                    best_score=best_score,
                    final_pivot=pivot,
                    steps=tuple(steps),
                    stopped_early=True,
                )

            pivot = self.exponential_mechanism.choose(candidate_scores.items())
            pivot_score = assure_score(pivot)

            if pivot_score > best_score or (
                pivot_score == best_score
                and self.level is not None
                and self.level(pivot) < self.level(best)
            ):
                best = pivot
                best_score = pivot_score

            steps.append(
                SearchStep(
                    step=step,
                    pivot=pivot,
                    score=pivot_score,
                    candidate_count=len(candidate_scores),
                )
            )

        return SearchResult(
            best=best,
            best_score=best_score,
            final_pivot=pivot,
            steps=tuple(steps),
            stopped_early=False,
        )
