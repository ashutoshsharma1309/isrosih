"""
Historical pattern matching.

Compares the current region-day against past high-impact events in the
unified dataset and returns the closest analogues with a similarity
score. Two uses:

1. Operator context — "today looks like Kerala, 16 Aug 2018 (Heavy,
   119 mm)" is far more actionable than a bare probability.
2. Confidence — a prediction whose conditions closely resemble observed
   historical events is better supported than one in a part of feature
   space the record has never visited.

Events are named from what the data actually records (region, date,
observed category, observed rainfall). No event is given a popular
nickname the dataset cannot substantiate.

Similarity is computed on standardized features as

    similarity = exp(-distance / scale)

where `distance` is the Euclidean distance in standardized feature space
and `scale` is the median pairwise distance within the reference set, so
"typical distance apart" lands near 37% rather than at an arbitrary
value. The result is bounded (0, 1].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ai_models.fusion_model.config import FusionConfig

logger = logging.getLogger(__name__)

#: Feature space used for analogue matching — the physically meaningful
#: conditions, deliberately excluding upstream model scores so that a
#: match reflects the weather/cloud state rather than model agreement.
MATCH_FEATURES: tuple[str, ...] = (
    "humidity_pct",
    "pressure_hpa",
    "temperature_c",
    "wind_speed_ms",
    "cloud_cover_pct",
    "rain_sum_3d",
    "cloud_density",
    "cold_top_fraction",
    "spatial_dispersion",
)


class MatcherNotFittedError(RuntimeError):
    """Raised when matching is attempted before fitting a reference set."""


@dataclass
class HistoricalEvent:
    region: str
    date: str
    category: int
    rainfall_mm: float
    similarity: float

    def describe(self, label_names: dict[int, str]) -> str:
        return (
            f"{self.region} {self.date} "
            f"({label_names.get(self.category, self.category)}, {self.rainfall_mm:.0f} mm)"
        )

    def to_dict(self, label_names: dict[int, str]) -> dict[str, Any]:
        return {
            "event": self.describe(label_names),
            "region": self.region,
            "date": self.date,
            "observed_category": self.category,
            "observed_rainfall_mm": round(float(self.rainfall_mm), 1),
            "similarity": round(float(self.similarity), 4),
            "similarity_pct": round(float(self.similarity) * 100, 1),
        }


class HistoricalMatcher:
    """Nearest-analogue lookup over past high-impact region-days."""

    def __init__(self, features: Sequence[str] = MATCH_FEATURES) -> None:
        self.features = list(features)
        self._scaler: StandardScaler | None = None
        self._matrix: np.ndarray | None = None
        self._metadata: pd.DataFrame | None = None
        self._scale: float = 1.0

    # ------------------------------------------------------------------
    def fit(self, frame: pd.DataFrame, config: FusionConfig) -> "HistoricalMatcher":
        """Build the reference set from observed high-impact region-days.

        Only days whose *observed* next-day outcome was heavy or extreme
        become reference events — these are the patterns worth matching.
        """
        missing = set(self.features) - set(frame.columns)
        if missing:
            raise ValueError(f"Frame missing match features: {sorted(missing)}")

        events = frame[frame[config.target_column] > 0].copy()
        if events.empty:
            raise ValueError("No high-impact region-days available to build a reference set")

        self._scaler = StandardScaler()
        self._matrix = self._scaler.fit_transform(events[self.features].astype(float))
        self._metadata = events[["region", "date", config.target_column, "rainfall_mm"]].reset_index(drop=True)
        self._scale = _median_pairwise_distance(self._matrix)
        logger.info(
            "Historical reference set: %d high-impact region-days (distance scale %.3f)",
            len(events), self._scale,
        )
        return self

    # ------------------------------------------------------------------
    def match(
        self,
        features: dict[str, float],
        config: FusionConfig,
        top_k: int = 3,
        exclude: tuple[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the `top_k` most similar historical events.

        Args:
            exclude: a (region, date) pair to drop from the results. When
                replaying a historical day that is itself in the reference
                set, the day matches itself at 100% and would otherwise
                inflate both the analogue list and the confidence score.
        """
        if self._matrix is None or self._scaler is None or self._metadata is None:
            raise MatcherNotFittedError("HistoricalMatcher.fit() must be called before match()")

        missing = [name for name in self.features if features.get(name) is None]
        if missing:
            raise ValueError(f"Cannot match — missing required conditions: {missing}")

        vector = self._scaler.transform(
            pd.DataFrame([{name: float(features[name]) for name in self.features}])
        )
        distances = np.linalg.norm(self._matrix - vector, axis=1)
        similarities = np.exp(-distances / max(self._scale, 1e-9))

        if exclude is not None:
            region, date = exclude
            is_self = (
                (self._metadata["region"].astype(str) == str(region))
                & (self._metadata["date"].astype(str) == str(date))
            ).to_numpy()
            similarities = np.where(is_self, -np.inf, similarities)

        order = [i for i in np.argsort(-similarities) if np.isfinite(similarities[i])]
        order = order[: max(int(top_k), 1)]
        matches = []
        for index in order:
            row = self._metadata.iloc[int(index)]
            matches.append(
                HistoricalEvent(
                    region=str(row["region"]),
                    date=str(row["date"]),
                    category=int(row[config.target_column]),
                    rainfall_mm=float(row["rainfall_mm"]),
                    similarity=float(similarities[int(index)]),
                ).to_dict(config.label_names)
            )
        return matches

    # ------------------------------------------------------------------
    @property
    def reference_size(self) -> int:
        return 0 if self._metadata is None else len(self._metadata)


def _median_pairwise_distance(matrix: np.ndarray, sample_cap: int = 400) -> float:
    """Median pairwise distance, subsampled so the pairwise matrix stays small."""
    if len(matrix) <= 1:
        return 1.0
    if len(matrix) > sample_cap:
        rng = np.random.default_rng(0)
        matrix = matrix[rng.choice(len(matrix), sample_cap, replace=False)]
    differences = matrix[:, None, :] - matrix[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    upper = distances[np.triu_indices_from(distances, k=1)]
    median = float(np.median(upper))
    return median if median > 0 else 1.0
