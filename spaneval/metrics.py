from __future__ import annotations

from dataclasses import dataclass



@dataclass
class Metrics:
    """Container for evaluation metrics with computed properties."""

    correct: float = 0
    incorrect: float = 0
    missed: float = 0
    spurious: float = 0
    possible: float = 0
    actual: float = 0

    @property
    def precision(self) -> float:
        """Calculate precision."""
        if self.actual > 0:
            return self.correct / self.actual
        else:
            return 0.0

    @property
    def recall(self) -> float:
        """Calculate recall."""
        if self.possible > 0:
            return self.correct / self.possible
        else:
            return 0.0

    @property
    def f1(self) -> float:
        """Calculate F1 score."""
        if self.precision + self.recall > 0:
            return 2 * self.precision * self.recall / (self.precision + self.recall)
        else:
            return 0.0

    def set_inferred_possible(self) -> None:
        """Sets the total number of possible true entities = correct + incorrect +
        missed."""
        self.possible = self.correct + self.incorrect + self.missed

    def set_inferred_actual(self) -> None:
        """Sets the total number of actually predicted entities = correct + incorrect +
        spurious."""
        self.actual = self.correct + self.incorrect + self.spurious

    def to_dict(self) -> dict[str, int | float]:
        """Convert metrics to dictionary with calculated scores."""
        return {
            "correct": self.correct,
            "incorrect": self.incorrect,
            "missed": self.missed,
            "spurious": self.spurious,
            "possible": self.possible,
            "actual": self.actual,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }

    def __add__(self, other: Metrics) -> Metrics:
        """Add two metrics together for aggregation."""
        return Metrics(
            correct=self.correct + other.correct,
            incorrect=self.incorrect + other.incorrect,
            missed=self.missed + other.missed,
            spurious=self.spurious + other.spurious,
            possible=self.possible + other.possible,
            actual=self.actual + other.actual,
        )


class DocumentMetrics:
    """Container for evaluation metrics for a single document."""

    def __init__(self, entity_types: list[str]|None=None, overall_metrics:Metrics|None=None, metrics_by_entity_type:dict[str, Metrics]|None=None) -> None:
        
        if entity_types is not None:
            self.entity_types = entity_types
        else:
            self.entity_types = []

        if overall_metrics is None:
            self.overall_metrics = Metrics()
        else:
            self.overall_metrics = overall_metrics
        if metrics_by_entity_type is None:
            self.metrics_by_entity_type = {
                entity_type: Metrics() for entity_type in entity_types
            }
        else:
            self.metrics_by_entity_type = metrics_by_entity_type

    def add_correct(self, value: float, entity_type: str) -> None:
        """Add correct predictions for a specific entity type."""
        self.overall_metrics.correct += value
        self.metrics_by_entity_type[entity_type].correct += value

    def add_incorrect(self, value: float, entity_type: str) -> None:
        """Add incorrect predictions for a specific entity type."""
        self.overall_metrics.incorrect += value
        self.metrics_by_entity_type[entity_type].incorrect += value

    def add_missed(self, value: float, entity_type: str) -> None:
        """Add missed predictions for a specific entity type."""
        self.overall_metrics.missed += value
        self.metrics_by_entity_type[entity_type].missed += value

    def add_spurious(self, value: float, entity_type: str) -> None:
        """Add spurious predictions for a specific entity type."""
        self.overall_metrics.spurious += value
        self.metrics_by_entity_type[entity_type].spurious += value

    def add_possible(self, value: float, entity_type: str) -> None:
        """Add possible predictions for a specific entity type."""
        self.overall_metrics.possible += value
        self.metrics_by_entity_type[entity_type].possible += value

    def add_actual(self, value: float, entity_type: str) -> None:
        """Add actual predictions for a specific entity type."""
        self.overall_metrics.actual += value
        self.metrics_by_entity_type[entity_type].actual += value

    def set_inferred_possible(self) -> None:
        """Set the total number of possible true entities for all entity types."""
        self.overall_metrics.set_inferred_possible()
        for entity_type, metrics in self.metrics_by_entity_type.items():
            metrics.set_inferred_possible()

    def set_inferred_actual(self) -> None:
        """Set the total number of actually predicted entities for all entity types."""
        self.overall_metrics.set_inferred_actual()
        for entity_type, metrics in self.metrics_by_entity_type.items():
            metrics.set_inferred_actual()


    def get_metric(self, metric: str, entity_type: str | None = None) -> float:
        if entity_type is None:
            return getattr(self.overall_metrics, metric)
        else:
            return getattr(self.metrics_by_entity_type[entity_type], metric)

