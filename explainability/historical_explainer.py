"""
Historical event similarity explanation.

Wraps the Phase 5 `HistoricalMatcher` and turns its nearest analogues into
something an operator can act on: "today's conditions resemble Kerala,
16 Aug 2018 (Heavy, 119 mm) at 84% similarity".

Events are named from what the record actually contains — region, date,
observed category, observed rainfall. A popular nickname the dataset
cannot substantiate ("the 2018 Kerala Flood Event") is never invented;
where the date and region correspond to a well-known event, the reader can
see that from the date itself.

The similarity number is also a confidence input: a prediction whose
conditions closely resemble observed events is better supported than one
in a part of feature space the record has never visited.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class HistoricalExplanationUnavailableError(RuntimeError):
    """Raised when no reference set is available to compare against."""


@dataclass
class HistoricalExplanation:
    """Nearest past analogues for the region-day being explained."""

    matches: list[dict[str, Any]]
    reference_size: int
    notes: list[str] = field(default_factory=list)

    @property
    def best(self) -> dict[str, Any] | None:
        return self.matches[0] if self.matches else None

    @property
    def best_similarity(self) -> float | None:
        return None if self.best is None else float(self.best["similarity"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_events": self.reference_size,
            "closest_match": self.best,
            "matches": self.matches,
            "notes": self.notes,
        }


class HistoricalExplainer:
    """Explains a prediction by the past events it most resembles."""

    def __init__(self, bundle: dict[str, Any]) -> None:
        matcher = bundle.get("matcher")
        if matcher is None:
            raise HistoricalExplanationUnavailableError(
                "The fusion bundle carries no historical matcher; retrain Phase 5."
            )
        self.matcher = matcher
        self.match_features: list[str] = list(bundle.get("match_features", []))

    def explain(
        self,
        row: pd.DataFrame,
        config: Any,
        *,
        top_k: int = 3,
        exclude: tuple[str, str] | None = None,
    ) -> HistoricalExplanation:
        """Return the closest historical analogues for this region-day.

        Args:
            exclude: (region, date) to drop, so replaying a day that is
                itself a reference event does not match itself at 100%.
        """
        if not isinstance(row, pd.DataFrame) or len(row) != 1:
            raise ValueError("HistoricalExplainer.explain expects a single-row DataFrame")

        features = row.iloc[0].to_dict()
        missing = [name for name in self.match_features if features.get(name) is None]
        if missing:
            raise HistoricalExplanationUnavailableError(
                f"Cannot compare with history — missing conditions: {missing}"
            )

        matches = self.matcher.match(features, config, top_k=top_k, exclude=exclude)
        notes: list[str] = []
        if not matches:
            notes.append(
                "No comparable historical event was found; the reference set holds only "
                "observed high-impact days, and none resembled these conditions."
            )
        elif matches[0]["similarity"] < 0.5:
            notes.append(
                "The closest historical analogue is weak, so these conditions are unusual "
                "against the record — treat the similarity signal with caution."
            )
        return HistoricalExplanation(
            matches=matches,
            reference_size=int(getattr(self.matcher, "reference_size", 0)),
            notes=notes,
        )
