from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
try:
    from typing import override
except ImportError:  # Python < 3.12
    def override(f):  # type: ignore[misc]
        return f

from .entity import Entity, Match, get_entity_types_from_matches
from .metrics import DocumentMetrics


def _check_strategy_instance(s: object, param: str = "strategy") -> None:
    """Raise a helpful TypeError if s is a strategy class rather than an instance."""
    if isinstance(s, type):
        raise TypeError(
            f"{param} must be a strategy instance, not the class {s.__name__!r}. "
            f"Did you mean {s.__name__}()?"
        )


@dataclass(frozen=True)
class TrueEntityOverlap:
    """Overlap as a fraction of the true entity's length.

    Example: true=[0,4), pred=[2,10) → intersection=2, true_length=4 → 0.5
    """

    def compute(self, true_entity: Entity, pred_entity: Entity) -> float:
        intersection = max(0, min(true_entity.end, pred_entity.end) - max(true_entity.start, pred_entity.start))
        return intersection / true_entity.length


@dataclass(frozen=True)
class PredEntityOverlap:
    """Overlap as a fraction of the predicted entity's length.

    Example: true=[0,4), pred=[2,10) → intersection=2, pred_length=8 → 0.25
    """

    def compute(self, true_entity: Entity, pred_entity: Entity) -> float:
        intersection = max(0, min(true_entity.end, pred_entity.end) - max(true_entity.start, pred_entity.start))
        return intersection / pred_entity.length


@dataclass(frozen=True)
class JaccardOverlap:
    """Overlap as intersection over union of both spans.

    Example: true=[0,4), pred=[2,10) → intersection=2, union=10 → 0.2
    """

    def compute(self, true_entity: Entity, pred_entity: Entity) -> float:
        intersection = max(0, min(true_entity.end, pred_entity.end) - max(true_entity.start, pred_entity.start))
        if intersection == 0:
            return 0.0
        union = max(true_entity.end, pred_entity.end) - min(true_entity.start, pred_entity.start)
        return intersection / union


@dataclass(frozen=True)
class EvaluationStrategy(ABC):
    """Abstract base class for evaluation strategies."""

    def __str__(self) -> str:
        name = getattr(self, "name", None)
        return name if name is not None else type(self).__name__

    @abstractmethod
    def evaluate_matches(self, matches: Iterable[Match]) -> DocumentMetrics:
        pass


@dataclass(frozen=True)
class EntityCountStrategy(EvaluationStrategy):

    @override
    def evaluate_matches(self, matches: Iterable[Match]) -> DocumentMetrics:
        entity_types = get_entity_types_from_matches(matches)
        document_metrics = DocumentMetrics(entity_types)

        true_entities_to_matches: dict[Entity, list[Match]] = {}
        for match in matches:
            true_entity = match.true_entity
            if true_entity is None:
                document_metrics.add_spurious(1, match.pred_entity.entity_type)  # pyright: ignore[reportOptionalMemberAccess]
            else:
                if true_entity not in true_entities_to_matches:
                    true_entities_to_matches[true_entity] = []
                true_entities_to_matches[true_entity].append(match)

        for true_entity, match_list in true_entities_to_matches.items():
            if len(match_list) == 1 and match_list[0].pred_entity is None:
                document_metrics.add_missed(1, true_entity.entity_type)
                continue

            pred_entities = [m.pred_entity for m in match_list if m.pred_entity is not None]
            selected_preds = self.select_pred_entities(true_entity, pred_entities)
            s = self.score(true_entity, selected_preds)
            document_metrics.add_correct(s, true_entity.entity_type)
            document_metrics.add_incorrect(1 - s, true_entity.entity_type)

        document_metrics.set_inferred_possible()
        document_metrics.set_inferred_actual()
        return document_metrics

    def select_pred_entities(self, true_entity: Entity, pred_entities: list[Entity]) -> list[Entity]:
        return pred_entities

    @abstractmethod
    def score(self, true_entity: Entity, pred_entities: list[Entity]) -> float:
        pass


@dataclass(frozen=True)
class Partial(EntityCountStrategy):
    """SemEval Partial: 1.0 for exact match, partial_score for any other overlap; type ignored."""

    partial_score: float = 0.5

    @override
    def select_pred_entities(self, true_entity: Entity, pred_entities: list[Entity]) -> list[Entity]:
        """Return the single predicted entity with the most overlap; ties broken by earliest start.

        Example: true=[2,8), pred_a=[0,5) overlap=3, pred_b=[4,10) overlap=4 → [pred_b]
        """
        best = max(
            pred_entities,
            key=lambda p: (
                min(p.end, true_entity.end) - max(p.start, true_entity.start),
                -p.start,
            ),
        )
        return [best]

    @override
    def score(self, true_entity: Entity, pred_entities: list[Entity]) -> float:
        best_pred = pred_entities[0]
        if best_pred.start == true_entity.start and best_pred.end == true_entity.end:
            return 1.0
        return self.partial_score


@dataclass(frozen=True)
class ProportionalCoverage(EntityCountStrategy):
    require_type_match: bool = False

    @override
    def score(self, true_entity: Entity, pred_entities: list[Entity]) -> float:
        covered_chars = sum(
            max(0, min(true_entity.end, p.end) - max(true_entity.start, p.start))
            for p in pred_entities
            if not self.require_type_match or p.entity_type == true_entity.entity_type
        )
        return covered_chars / true_entity.length


@dataclass(frozen=True)
class MinimumOverlap(EntityCountStrategy):
    threshold: float
    overlap: TrueEntityOverlap | PredEntityOverlap | JaccardOverlap = field(default_factory=TrueEntityOverlap)
    threshold_inclusive: bool = True
    match_score: float = 1.0
    require_type_match: bool = False
    name: str | None = field(default=None, repr=False)

    @override
    def select_pred_entities(self, true_entity: Entity, pred_entities: list[Entity]) -> list[Entity]:
        """Return the single predicted entity with the most overlap; ties broken by type match, then earliest start.

        Example: true=[2,8) NAME, pred_a=[0,6) overlap=4 DATE, pred_b=[3,9) overlap=5 NAME → [pred_b]
        """
        best = max(
            pred_entities,
            key=lambda p: (
                min(p.end, true_entity.end) - max(p.start, true_entity.start),
                p.entity_type == true_entity.entity_type,
                -p.start,
            ),
        )
        return [best]

    @override
    def score(self, true_entity: Entity, pred_entities: list[Entity]) -> float:
        best_pred = pred_entities[0]
        span_value = self.overlap.compute(true_entity, best_pred)
        if self.threshold_inclusive:
            span_passes = span_value >= self.threshold
        else:
            span_passes = span_value > self.threshold
        span_score = self.match_score if span_passes else 0.0
        type_score = 1.0 if (not self.require_type_match or best_pred.entity_type == true_entity.entity_type) else 0.0
        return span_score * type_score


class Strict(MinimumOverlap):
    """SemEval Strict: exact span boundaries and type match required."""

    def __init__(self) -> None:
        super().__init__(threshold=1.0, overlap=JaccardOverlap(), require_type_match=True)


class Exact(MinimumOverlap):
    """SemEval Exact: exact span boundaries; type ignored."""

    def __init__(self) -> None:
        super().__init__(threshold=1.0, overlap=JaccardOverlap())


class EntType(MinimumOverlap):
    """SemEval EntType: any overlap with correct type."""

    def __init__(self) -> None:
        super().__init__(threshold=0.0, overlap=JaccardOverlap(), threshold_inclusive=False, require_type_match=True)


class Contains(MinimumOverlap):
    """Correct if prediction fully contains the true entity span."""

    def __init__(self) -> None:
        super().__init__(threshold=1.0, overlap=TrueEntityOverlap())


class AnyOverlap(MinimumOverlap):
    """Any non-zero overlap scores 1.0; type and boundaries ignored."""

    def __init__(self) -> None:
        super().__init__(threshold=0.0, overlap=JaccardOverlap(), threshold_inclusive=False)


@dataclass(frozen=True)
class TextCoverage(EvaluationStrategy):
    """Strategy based on character-level text coverage."""

    require_type_match: bool = False

    @override
    def evaluate_matches(
        self,
        matches
    ) -> DocumentMetrics:
        """Evaluate the predicted entities against the true entities."""

        entity_types = get_entity_types_from_matches(matches)
        document_metrics = DocumentMetrics(entity_types)

        # map true entities to matches to better handle cases
        # where a true entity is matched to multiple predicted entities
        true_entities_to_matches: dict[Entity, list[Match]] = {}
        for match in matches:
            true_entity = match.true_entity
            if true_entity is not None:
                if true_entity not in true_entities_to_matches:
                    true_entities_to_matches[true_entity] = []
                true_entities_to_matches[true_entity].append(match)

        # now handle all cases where a true entity is matched to at least one predicted entity
        # and possibly multiple predicted entities
        for true_entity, match_list in true_entities_to_matches.items():
            covered_text_len = 0

            for match in match_list:
                pred_entity = match.pred_entity

                if pred_entity is not None:
                    if self.require_type_match and pred_entity.entity_type != true_entity.entity_type:
                        continue
                    overlap_start = max(true_entity.start, pred_entity.start)
                    overlap_end = min(true_entity.end, pred_entity.end)
                    overlap_length = overlap_end - overlap_start
                    covered_text_len += overlap_length

            document_metrics.add_correct(covered_text_len, true_entity.entity_type)
            document_metrics.add_missed(
                true_entity.end - true_entity.start - covered_text_len,
                true_entity.entity_type,
            )
            document_metrics.add_possible(
                true_entity.end - true_entity.start, true_entity.entity_type
            )

        # now handle all cases where a pred entity is matched to at least one true entity
        # and possibly multiple true entities
        pred_entities_to_matches: dict[Entity, list[Match]] = {}
        for match in matches:
            pred_entity = match.pred_entity
            if pred_entity is not None:
                if pred_entity not in pred_entities_to_matches:
                    pred_entities_to_matches[pred_entity] = []
                pred_entities_to_matches[pred_entity].append(match)

        for pred_entity, match_list in pred_entities_to_matches.items():
            same_type_covered = 0
            cross_type_covered = 0
            for match in match_list:
                true_entity = match.true_entity
                if true_entity is not None:
                    overlap_start = max(true_entity.start, pred_entity.start)
                    overlap_end = min(true_entity.end, pred_entity.end)
                    overlap_length = overlap_end - overlap_start
                    if not self.require_type_match or pred_entity.entity_type == true_entity.entity_type:
                        same_type_covered += overlap_length
                        document_metrics.add_actual(overlap_length, true_entity.entity_type)
                    else:
                        cross_type_covered += overlap_length
                        document_metrics.add_actual(overlap_length, pred_entity.entity_type)

            spurious_text_len = pred_entity.end - pred_entity.start - same_type_covered - cross_type_covered
            document_metrics.add_spurious(spurious_text_len, pred_entity.entity_type)
            document_metrics.add_actual(spurious_text_len, pred_entity.entity_type)
            document_metrics.add_incorrect(cross_type_covered, pred_entity.entity_type)

        return document_metrics
