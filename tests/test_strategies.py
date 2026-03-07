import pytest

from spaneval.entity import (  # pyright: ignore[reportImplicitRelativeImport]
    Entity,
    EntityMatcher,
)
from spaneval.strategies import (  # pyright: ignore[reportImplicitRelativeImport]
    TextCoverage,
    EntType,
    Exact,
    EvaluationStrategy,
    JaccardOverlap,
    MinimumOverlap,
    Partial,
    PredEntityOverlap,
    ProportionalCoverage,
    Strict,
    Contains,
    TrueEntityOverlap,
)
from spaneval.metrics import DocumentMetrics


def _eval(strategy: EvaluationStrategy, true: list, pred: list) -> DocumentMetrics:
    """Match entities then score with strategy — mirrors the main pipeline."""
    matches = EntityMatcher().match_entities(true, pred)
    return strategy.evaluate_matches(matches)

def test_strict_evaluation():
    true_entities = [Entity(entity_type="PERSON", start=0, end=4)]
    pred_entities = [Entity(entity_type="PERSON", start=0, end=4)]

    document_metrics = _eval(Strict(), true_entities, pred_entities)

    for metrics in [document_metrics.overall_metrics] + list(
        document_metrics.metrics_by_entity_type.values()
    ):
        assert metrics.correct == 1
        assert metrics.incorrect == 0
        assert metrics.missed == 0
        assert metrics.spurious == 0
        assert metrics.possible == 1
        assert metrics.actual == 1


def test_strict_evaluation_split_predicted_entity():
    true_entities = [Entity(entity_type="PERSON", start=0, end=4)]
    pred_entities = [
        Entity(entity_type="FIRST_NAME", start=0, end=2),
        Entity(entity_type="LAST_NAME", start=2, end=4),
    ]

    document_metrics = _eval(Strict(), true_entities, pred_entities)

    for metrics in [
        document_metrics.overall_metrics,
        document_metrics.metrics_by_entity_type["PERSON"],
    ]:
        assert metrics.correct == 0
        assert metrics.incorrect == 1
        assert metrics.missed == 0
        assert metrics.spurious == 0
        assert metrics.possible == 1
        assert metrics.actual == 1

    for entity_type in ["FIRST_NAME", "LAST_NAME"]:
        metrics = document_metrics.metrics_by_entity_type[entity_type]
        assert metrics.correct == 0
        assert metrics.incorrect == 0
        assert metrics.missed == 0
        assert metrics.spurious == 0
        assert metrics.possible == 0
        assert metrics.actual == 0


def test_strict_evaluation_split_true_entity():
    true_entities = [
        Entity(entity_type="FIRST_NAME", start=0, end=2),
        Entity(entity_type="LAST_NAME", start=2, end=4),
    ]
    pred_entities = [Entity(entity_type="PERSON", start=0, end=4)]

    document_metrics = _eval(Strict(), true_entities, pred_entities)

    for metrics in [document_metrics.overall_metrics]:
        assert metrics.correct == 0
        assert metrics.incorrect == 2
        assert metrics.missed == 0
        assert metrics.spurious == 0
        assert metrics.possible == 2
        assert metrics.actual == 2

    for entity_type in ["FIRST_NAME", "LAST_NAME"]:
        metrics = document_metrics.metrics_by_entity_type[entity_type]
        assert metrics.correct == 0
        assert metrics.incorrect == 1
        assert metrics.missed == 0
        assert metrics.spurious == 0
        assert metrics.possible == 1
        assert metrics.actual == 1

    for entity_type in ["PERSON"]:
        metrics = document_metrics.metrics_by_entity_type[entity_type]
        assert metrics.correct == 0
        assert metrics.incorrect == 0
        assert metrics.missed == 0
        assert metrics.spurious == 0
        assert metrics.possible == 0
        assert metrics.actual == 0

def test_exact_evaluation():
    true_entities = [Entity(entity_type="PERSON", start=0, end=4)]
    pred_entities = [Entity(entity_type="MISC", start=0, end=4)]

    document_metrics = _eval(Exact(), true_entities, pred_entities)

    for entity_type in ["PERSON"]:
        metrics = document_metrics.metrics_by_entity_type[entity_type]
        assert metrics.correct == 1
        assert metrics.incorrect == 0
        assert metrics.missed == 0
        assert metrics.spurious == 0
        assert metrics.possible == 1
        assert metrics.actual == 1

    for entity_type in ["MISC"]:
        metrics = document_metrics.metrics_by_entity_type[entity_type]
        assert metrics.correct == 0
        assert metrics.incorrect == 0
        assert metrics.missed == 0
        assert metrics.spurious == 0
        assert metrics.possible == 0
        assert metrics.actual == 0



def test_exact_evaluation_split_predicted_entity():
    true_entities = [Entity(entity_type="PERSON", start=0, end=4)]
    pred_entities = [
        Entity(entity_type="FIRST_NAME", start=0, end=2),
        Entity(entity_type="LAST_NAME", start=2, end=4),
    ]

    document_metrics = _eval(Exact(), true_entities, pred_entities)

    for metrics in [
        document_metrics.overall_metrics,
        document_metrics.metrics_by_entity_type["PERSON"],
    ]:
        assert metrics.correct == 0
        assert metrics.incorrect == 1
        assert metrics.missed == 0
        assert metrics.spurious == 0
        assert metrics.possible == 1
        assert metrics.actual == 1

    for entity_type in ["FIRST_NAME", "LAST_NAME"]:
        metrics = document_metrics.metrics_by_entity_type[entity_type]
        assert metrics.correct == 0
        assert metrics.incorrect == 0
        assert metrics.missed == 0
        assert metrics.spurious == 0
        assert metrics.possible == 0
        assert metrics.actual == 0


def test_exact_evaluation_split_true_entity():
    true_entities = [
        Entity(entity_type="FIRST_NAME", start=0, end=2),
        Entity(entity_type="LAST_NAME", start=2, end=4),
    ]
    pred_entities = [Entity(entity_type="PERSON", start=0, end=4)]

    document_metrics = _eval(Exact(), true_entities, pred_entities)

    for metrics in [document_metrics.overall_metrics]:
        assert metrics.correct == 0
        assert metrics.incorrect == 2
        assert metrics.missed == 0
        assert metrics.spurious == 0
        assert metrics.possible == 2
        assert metrics.actual == 2

    for entity_type in ["FIRST_NAME", "LAST_NAME"]:
        metrics = document_metrics.metrics_by_entity_type[entity_type]
        assert metrics.correct == 0
        assert metrics.incorrect == 1
        assert metrics.missed == 0
        assert metrics.spurious == 0
        assert metrics.possible == 1
        assert metrics.actual == 1

    for entity_type in ["PERSON"]:
        metrics = document_metrics.metrics_by_entity_type[entity_type]
        assert metrics.correct == 0
        assert metrics.incorrect == 0
        assert metrics.missed == 0
        assert metrics.spurious == 0
        assert metrics.possible == 0
        assert metrics.actual == 0


def test_ent_type_evaluation():
    true_entities = [Entity(entity_type="PERSON", start=0, end=4)]
    pred_entities = [Entity(entity_type="PERSON", start=1, end=5)]

    document_metrics = _eval(EntType(), true_entities, pred_entities)

    for metrics in [document_metrics.overall_metrics] + list(
        document_metrics.metrics_by_entity_type.values()
    ):
        assert metrics.correct == 1
        assert metrics.incorrect == 0
        assert metrics.missed == 0
        assert metrics.spurious == 0
        assert metrics.possible == 1
        assert metrics.actual == 1


def test_ent_type_evaluation_split_predicted_entity():
    # EntType checks only the best-overlap pred's type (not all preds).
    # With two preds of equal overlap and neither matching the true type,
    # the best pred (smallest start) has a non-matching type → incorrect.
    true_entities = [Entity(entity_type="PERSON", start=0, end=4)]
    pred_entities = [
        Entity(entity_type="FIRST_NAME", start=0, end=2),
        Entity(entity_type="LAST_NAME", start=2, end=4),
    ]

    document_metrics = _eval(EntType(), true_entities, pred_entities)

    for metrics in [
        document_metrics.overall_metrics,
        document_metrics.metrics_by_entity_type["PERSON"],
    ]:
        assert metrics.correct == 0
        assert metrics.incorrect == 1
        assert metrics.missed == 0
        assert metrics.spurious == 0
        assert metrics.possible == 1
        assert metrics.actual == 1

    for entity_type in ["FIRST_NAME", "LAST_NAME"]:
        metrics = document_metrics.metrics_by_entity_type[entity_type]
        assert metrics.correct == 0
        assert metrics.incorrect == 0
        assert metrics.missed == 0
        assert metrics.spurious == 0
        assert metrics.possible == 0
        assert metrics.actual == 0


def test_ent_type_evaluation_split_true_entity():
    true_entities = [
        Entity(entity_type="FIRST_NAME", start=0, end=2),
        Entity(entity_type="LAST_NAME", start=2, end=4),
    ]
    pred_entities = [Entity(entity_type="PERSON", start=0, end=4)]

    document_metrics = _eval(EntType(), true_entities, pred_entities)

    for metrics in [document_metrics.overall_metrics]:
        assert metrics.correct == 0
        assert metrics.incorrect == 2
        assert metrics.missed == 0
        assert metrics.spurious == 0
        assert metrics.possible == 2
        assert metrics.actual == 2

    for entity_type in ["FIRST_NAME", "LAST_NAME"]:
        metrics = document_metrics.metrics_by_entity_type[entity_type]
        assert metrics.correct == 0
        assert metrics.incorrect == 1
        assert metrics.missed == 0
        assert metrics.spurious == 0
        assert metrics.possible == 1
        assert metrics.actual == 1

    for entity_type in ["PERSON"]:
        metrics = document_metrics.metrics_by_entity_type[entity_type]
        assert metrics.correct == 0
        assert metrics.incorrect == 0
        assert metrics.missed == 0
        assert metrics.spurious == 0
        assert metrics.possible == 0
        assert metrics.actual == 0


def test_partial_evaluation():
    true_entities = [Entity(entity_type="PERSON", start=0, end=4)]
    pred_entities = [Entity(entity_type="PERSON", start=0, end=3)]

    document_metrics = _eval(Partial(), true_entities, pred_entities)

    for metrics in [document_metrics.overall_metrics] + list(
        document_metrics.metrics_by_entity_type.values()
    ):
        assert metrics.correct == 0.5
        assert metrics.incorrect == 0.5
        assert metrics.missed == 0
        assert metrics.spurious == 0
        assert metrics.possible == 1
        assert metrics.actual == 1


def test_partial_evaluation_split_predicted_entity():
    true_entities = [Entity(entity_type="PERSON", start=0, end=4)]
    pred_entities = [
        Entity(entity_type="FIRST_NAME", start=0, end=2),
        Entity(entity_type="LAST_NAME", start=2, end=4),
    ]

    document_metrics = _eval(Partial(), true_entities, pred_entities)

    for metrics in [
        document_metrics.overall_metrics,
        document_metrics.metrics_by_entity_type["PERSON"],
    ]:
        assert metrics.correct == 0.5
        assert metrics.incorrect == 0.5
        assert metrics.missed == 0
        assert metrics.spurious == 0
        assert metrics.possible == 1
        assert metrics.actual == 1

    for entity_type in ["FIRST_NAME", "LAST_NAME"]:
        metrics = document_metrics.metrics_by_entity_type[entity_type]
        assert metrics.correct == 0
        assert metrics.incorrect == 0
        assert metrics.missed == 0
        assert metrics.spurious == 0
        assert metrics.possible == 0
        assert metrics.actual == 0


def test_partial_evaluation_split_true_entity():
    true_entities = [
        Entity(entity_type="FIRST_NAME", start=0, end=2),
        Entity(entity_type="LAST_NAME", start=2, end=4),
    ]
    pred_entities = [Entity(entity_type="PERSON", start=0, end=4)]

    document_metrics = _eval(Partial(), true_entities, pred_entities)

    for metrics in [document_metrics.overall_metrics]:
        assert metrics.correct == 1
        assert metrics.incorrect == 1
        assert metrics.missed == 0
        assert metrics.spurious == 0
        assert metrics.possible == 2
        assert metrics.actual == 2

    for entity_type in ["FIRST_NAME", "LAST_NAME"]:
        metrics = document_metrics.metrics_by_entity_type[entity_type]
        assert metrics.correct == 0.5
        assert metrics.incorrect == 0.5
        assert metrics.missed == 0
        assert metrics.spurious == 0
        assert metrics.possible == 1
        assert metrics.actual == 1

    for entity_type in ["PERSON"]:
        metrics = document_metrics.metrics_by_entity_type[entity_type]
        assert metrics.correct == 0
        assert metrics.incorrect == 0
        assert metrics.missed == 0
        assert metrics.spurious == 0
        assert metrics.possible == 0
        assert metrics.actual == 0


def test_text_coverage_exact():
    true_entities = [Entity(entity_type="PERSON", start=0, end=4)]
    pred_entities = [Entity(entity_type="PERSON", start=0, end=4)]

    strategy = TextCoverage()
    document_metrics = _eval(strategy, true_entities, pred_entities)

    for metrics in [document_metrics.overall_metrics] + list(
        document_metrics.metrics_by_entity_type.values()
    ):
        assert metrics.correct == 4
        assert metrics.incorrect == 0
        assert metrics.missed == 0
        assert metrics.spurious == 0
        assert metrics.possible == 4
        assert metrics.actual == 4


def test_text_coverage_submatch():
    true_entities = [Entity(entity_type="PERSON", start=0, end=4)]
    pred_entities = [Entity(entity_type="PERSON", start=0, end=3)]

    strategy = TextCoverage()
    document_metrics = _eval(strategy, true_entities, pred_entities)

    for metrics in [document_metrics.overall_metrics] + list(
        document_metrics.metrics_by_entity_type.values()
    ):
        assert metrics.correct == 3
        assert metrics.incorrect == 0
        assert metrics.missed == 1
        assert metrics.spurious == 0
        assert metrics.possible == 4
        assert metrics.actual == 3


def test_text_coverage_overmatch():
    true_entities = [Entity(entity_type="PERSON", start=1, end=4)]
    pred_entities = [Entity(entity_type="PERSON", start=0, end=5)]

    strategy = TextCoverage()
    document_metrics = _eval(strategy, true_entities, pred_entities)

    for metrics in [document_metrics.overall_metrics] + list(
        document_metrics.metrics_by_entity_type.values()
    ):
        assert metrics.correct == 3
        assert metrics.incorrect == 0
        assert metrics.missed == 0
        assert metrics.spurious == 2
        assert metrics.possible == 3
        assert metrics.actual == 5


def test_text_coverage_exact_type_mismatch():
    true_entities = [Entity(entity_type="PERSON", start=0, end=4)]
    pred_entities = [Entity(entity_type="ORGANISATION", start=0, end=4)]

    strategy = TextCoverage()
    document_metrics = _eval(strategy, true_entities, pred_entities)

    for metrics in [document_metrics.overall_metrics] + [
        document_metrics.metrics_by_entity_type["PERSON"]
    ]:
        assert metrics.correct == 4
        assert metrics.incorrect == 0
        assert metrics.missed == 0
        assert metrics.spurious == 0
        assert metrics.possible == 4
        assert metrics.actual == 4

    for entity_type in ["ORGANISATION"]:
        metrics = document_metrics.metrics_by_entity_type[entity_type]
        assert metrics.correct == 0
        assert metrics.incorrect == 0
        assert metrics.missed == 0
        assert metrics.spurious == 0
        assert metrics.possible == 0
        assert metrics.actual == 0


def test_text_coverage_submatch_type_mismatch():
    true_entities = [Entity(entity_type="PERSON", start=0, end=4)]
    pred_entities = [Entity(entity_type="ORGANISATION", start=1, end=3)]

    strategy = TextCoverage()
    document_metrics = _eval(strategy, true_entities, pred_entities)

    for metrics in [document_metrics.overall_metrics] + [
        document_metrics.metrics_by_entity_type["PERSON"]
    ]:
        assert metrics.correct == 2
        assert metrics.incorrect == 0
        assert metrics.missed == 2
        assert metrics.spurious == 0
        assert metrics.possible == 4
        assert metrics.actual == 2

    for entity_type in ["ORGANISATION"]:
        metrics = document_metrics.metrics_by_entity_type[entity_type]
        assert metrics.correct == 0
        assert metrics.incorrect == 0
        assert metrics.missed == 0
        assert metrics.spurious == 0
        assert metrics.possible == 0
        assert metrics.actual == 0


def test_text_coverage_overmatch_type_mismatch():
    true_entities = [Entity(entity_type="PERSON", start=1, end=4)]
    pred_entities = [Entity(entity_type="ORGANISATION", start=0, end=5)]

    strategy = TextCoverage()
    document_metrics = _eval(strategy, true_entities, pred_entities)

    for metrics in [document_metrics.overall_metrics]:
        assert metrics.correct == 3
        assert metrics.incorrect == 0
        assert metrics.missed == 0
        assert metrics.spurious == 2
        assert metrics.possible == 3
        assert metrics.actual == 5

    for entity_type in ["PERSON"]:
        metrics = document_metrics.metrics_by_entity_type[entity_type]
        assert metrics.correct == 3
        assert metrics.incorrect == 0
        assert metrics.missed == 0
        assert metrics.spurious == 0
        assert metrics.possible == 3
        assert metrics.actual == 3

    for entity_type in ["ORGANISATION"]:
        metrics = document_metrics.metrics_by_entity_type[entity_type]
        assert metrics.correct == 0
        assert metrics.incorrect == 0
        assert metrics.missed == 0
        assert metrics.spurious == 2
        assert metrics.possible == 0
        assert metrics.actual == 2


def test_entity_count_missed_entity_in_multi_entity_doc():
    # A true entity with no overlapping predicted entity must be counted as
    # missed, regardless of how many other entities are in the document.
    true_entities = [
        Entity(entity_type="PERSON", start=0, end=5),
        Entity(entity_type="LOCATION", start=10, end=20),  # no overlapping pred
    ]
    pred_entities = [
        Entity(entity_type="PERSON", start=0, end=5),
    ]

    document_metrics = _eval(Strict(), true_entities, pred_entities)

    assert document_metrics.overall_metrics.correct == 1
    assert document_metrics.overall_metrics.missed == 1
    assert document_metrics.overall_metrics.possible == 2
    assert document_metrics.overall_metrics.actual == 1

    assert document_metrics.metrics_by_entity_type["PERSON"].correct == 1
    assert document_metrics.metrics_by_entity_type["PERSON"].missed == 0

    assert document_metrics.metrics_by_entity_type["LOCATION"].correct == 0
    assert document_metrics.metrics_by_entity_type["LOCATION"].missed == 1


def test_text_coverage_true_entity_matched_by_two_pred_entities():
    # When a true entity is covered by multiple predicted entities, character
    # coverage must be the total overlap across all predictions, not summed
    # once per predicted entity.
    true_entities = [Entity(entity_type="PERSON", start=0, end=10)]
    pred_entities = [
        Entity(entity_type="PERSON", start=0, end=5),   # covers first half
        Entity(entity_type="PERSON", start=5, end=10),  # covers second half
    ]

    strategy = TextCoverage()
    document_metrics = _eval(strategy, true_entities, pred_entities)

    for metrics in [document_metrics.overall_metrics] + list(
        document_metrics.metrics_by_entity_type.values()
    ):
        assert metrics.correct == 10
        assert metrics.missed == 0
        assert metrics.spurious == 0
        assert metrics.possible == 10
        assert metrics.actual == 10


# ---------------------------------------------------------------------------
# MinimumOverlap targeted tests
# ---------------------------------------------------------------------------

def test_threshold_span_overlap_criteria_produce_different_scores():
    # For a pred that covers only part of the true span, TrueEntityOverlap,
    # PredEntityOverlap, and JaccardOverlap return different values, which
    # crosses different thresholds and produces different correct/incorrect counts.
    true_entity = Entity(entity_type="X", start=0, end=10)   # length 10
    pred_entity = Entity(entity_type="X", start=0, end=6)    # overlap 6, pred length 6

    # TrueEntityOverlap = 6/10 = 0.6
    # PredEntityOverlap = 6/6  = 1.0
    # JaccardOverlap    = 6/10 = 0.6

    # threshold=0.7 → TrueEntityOverlap fails, PredEntityOverlap passes
    true_overlap_strategy = MinimumOverlap(overlap=TrueEntityOverlap(), threshold=0.7)
    pred_overlap_strategy = MinimumOverlap(overlap=PredEntityOverlap(), threshold=0.7)

    true_metrics = _eval(true_overlap_strategy, [true_entity], [pred_entity])
    pred_metrics = _eval(pred_overlap_strategy, [true_entity], [pred_entity])

    assert true_metrics.overall_metrics.correct == 0.0
    assert pred_metrics.overall_metrics.correct == 1.0


def test_threshold_span_threshold_inclusive_at_boundary():
    # When the overlap value exactly equals the threshold, threshold_inclusive=True
    # must award correct and threshold_inclusive=False must award incorrect.
    true_entity = Entity(entity_type="X", start=0, end=4)
    pred_entity = Entity(entity_type="X", start=0, end=4)  # Jaccard = 1.0

    inclusive = MinimumOverlap(overlap=JaccardOverlap(), threshold=1.0, threshold_inclusive=True)
    exclusive = MinimumOverlap(overlap=JaccardOverlap(), threshold=1.0, threshold_inclusive=False)

    assert _eval(inclusive, [true_entity], [pred_entity]).overall_metrics.correct == 1.0
    assert _eval(exclusive, [true_entity], [pred_entity]).overall_metrics.correct == 0.0


def test_threshold_span_match_score_below_one_returned_when_criteria_met():
    # match_score < 1.0 must be the value returned for a match, not 1.0.
    true_entity = Entity(entity_type="X", start=0, end=4)
    pred_entity = Entity(entity_type="X", start=0, end=4)

    strategy = MinimumOverlap(overlap=JaccardOverlap(), threshold=1.0, match_score=0.75)
    metrics = _eval(strategy, [true_entity], [pred_entity]).overall_metrics

    assert metrics.correct == 0.75
    assert metrics.incorrect == 0.25


def test_threshold_span_require_type_match_gates_type_independently_of_span():
    # With require_type_match=True, a perfect span but wrong type must score 0;
    # a perfect span with correct type must score match_score.
    true_entity = Entity(entity_type="PERSON", start=0, end=4)
    pred_wrong_type = Entity(entity_type="LOCATION", start=0, end=4)
    pred_right_type = Entity(entity_type="PERSON", start=0, end=4)

    strategy = MinimumOverlap(overlap=JaccardOverlap(), threshold=1.0, require_type_match=True)

    wrong_metrics = _eval(strategy, [true_entity], [pred_wrong_type]).overall_metrics
    right_metrics = _eval(strategy, [true_entity], [pred_right_type]).overall_metrics

    assert wrong_metrics.correct == 0.0
    assert right_metrics.correct == 1.0


def test_threshold_span_select_pred_entities_type_tiebreak_over_start():
    # When two preds have equal raw overlap, the one whose type matches the true
    # entity must be preferred over the one with the smaller start position.
    true_entity = Entity(entity_type="PERSON", start=0, end=4)
    # Both preds overlap by 2 chars; pred_b has smaller start but wrong type.
    pred_a = Entity(entity_type="PERSON",   start=2, end=4)  # start=2, type matches
    pred_b = Entity(entity_type="LOCATION", start=0, end=2)  # start=0, type mismatch

    strategy = MinimumOverlap(overlap=JaccardOverlap(), threshold=0.0, threshold_inclusive=False, require_type_match=True)
    metrics = _eval(strategy, [true_entity], [pred_a, pred_b]).overall_metrics

    # pred_a is selected (type match wins the tie); type matches → correct
    assert metrics.correct == 1.0


# ---------------------------------------------------------------------------
# Partial targeted tests
# ---------------------------------------------------------------------------

def test_partial_exact_match_scores_one():
    # Exact span boundaries → full credit regardless of type.
    true_entity = Entity(entity_type="PERSON", start=0, end=4)
    pred_entity = Entity(entity_type="LOCATION", start=0, end=4)

    metrics = _eval(Partial(), [true_entity], [pred_entity]).overall_metrics

    assert metrics.correct == 1.0
    assert metrics.incorrect == 0.0


def test_partial_partial_overlap_scores_partial_score():
    # Any overlap that is not exact → partial_score (default 0.5).
    true_entity = Entity(entity_type="PERSON", start=0, end=4)
    pred_entity = Entity(entity_type="PERSON", start=0, end=3)

    metrics = _eval(Partial(), [true_entity], [pred_entity]).overall_metrics

    assert metrics.correct == 0.5
    assert metrics.incorrect == 0.5


def test_partial_custom_partial_score_is_respected():
    # partial_score parameter controls the fractional credit awarded.
    true_entity = Entity(entity_type="PERSON", start=0, end=4)
    pred_entity = Entity(entity_type="PERSON", start=0, end=3)

    metrics = _eval(Partial(partial_score=0.25), [true_entity], [pred_entity]).overall_metrics

    assert metrics.correct == 0.25
    assert metrics.incorrect == 0.75


def test_partial_select_pred_entities_smallest_start_tiebreak():
    # When two preds have equal raw overlap, the one with the smaller start must
    # be selected (type is not used as a tiebreaker for Partial).
    true_entity = Entity(entity_type="PERSON", start=2, end=6)
    # Both overlap by 2 chars with [2, 6); pred_a starts earlier.
    pred_a = Entity(entity_type="X", start=0, end=4)  # overlap [2,4) = 2, start=0
    pred_b = Entity(entity_type="X", start=4, end=8)  # overlap [4,6) = 2, start=4

    metrics = _eval(Partial(), [true_entity], [pred_a, pred_b]).overall_metrics

    # pred_a selected (start=0 < start=4); span [0,4) ≠ [2,6) → partial_score
    assert metrics.correct == 0.5


# ---------------------------------------------------------------------------
# Overlap criterion unit tests
# ---------------------------------------------------------------------------

def test_true_entity_overlap_exact_match():
    # When spans are identical, intersection equals true entity length → 1.0.
    true = Entity("X", start=0, end=10)
    pred = Entity("X", start=0, end=10)
    assert TrueEntityOverlap().compute(true, pred) == 1.0


def test_true_entity_overlap_pred_subset_of_true():
    # Pred covers 6 of 10 true chars → 0.6.
    true = Entity("X", start=0, end=10)
    pred = Entity("X", start=0, end=6)
    assert TrueEntityOverlap().compute(true, pred) == 0.6


def test_true_entity_overlap_pred_superset_of_true():
    # Pred fully covers the true span (and more) → intersection == true length → 1.0.
    true = Entity("X", start=2, end=6)
    pred = Entity("X", start=0, end=10)
    assert TrueEntityOverlap().compute(true, pred) == 1.0


def test_true_entity_overlap_partial_overlap():
    # Pred overlaps the right half of the true span: intersection 2, true length 4 → 0.5.
    true = Entity("X", start=0, end=4)
    pred = Entity("X", start=2, end=6)
    assert TrueEntityOverlap().compute(true, pred) == 0.5


def test_true_entity_overlap_no_overlap_returns_zero():
    # Defensive case: no overlap → 0.0.
    true = Entity("X", start=0, end=4)
    pred = Entity("X", start=5, end=9)
    assert TrueEntityOverlap().compute(true, pred) == 0.0


def test_pred_entity_overlap_exact_match():
    true = Entity("X", start=0, end=10)
    pred = Entity("X", start=0, end=10)
    assert PredEntityOverlap().compute(true, pred) == 1.0


def test_pred_entity_overlap_pred_subset_of_true():
    # Pred is fully inside the true span → intersection == pred length → 1.0.
    true = Entity("X", start=0, end=10)
    pred = Entity("X", start=2, end=6)
    assert PredEntityOverlap().compute(true, pred) == 1.0


def test_pred_entity_overlap_true_subset_of_pred():
    # True is fully inside pred: intersection 4, pred length 10 → 0.4.
    true = Entity("X", start=3, end=7)
    pred = Entity("X", start=0, end=10)
    assert PredEntityOverlap().compute(true, pred) == 0.4


def test_pred_entity_overlap_partial_overlap():
    # Pred overlaps right side of true: intersection 2, pred length 4 → 0.5.
    true = Entity("X", start=0, end=4)
    pred = Entity("X", start=2, end=6)
    assert PredEntityOverlap().compute(true, pred) == 0.5


def test_pred_entity_overlap_no_overlap_returns_zero():
    true = Entity("X", start=0, end=4)
    pred = Entity("X", start=5, end=9)
    assert PredEntityOverlap().compute(true, pred) == 0.0


def test_jaccard_overlap_exact_match():
    # Identical spans → intersection == union → 1.0.
    true = Entity("X", start=0, end=4)
    pred = Entity("X", start=0, end=4)
    assert JaccardOverlap().compute(true, pred) == 1.0


def test_jaccard_overlap_pred_subset_of_true():
    # Pred [0,6) inside true [0,10): intersection 6, union 10 → 0.6.
    true = Entity("X", start=0, end=10)
    pred = Entity("X", start=0, end=6)
    assert JaccardOverlap().compute(true, pred) == 0.6


def test_jaccard_overlap_shifted_spans():
    # true [0,4), pred [2,6): intersection [2,4)=2, union [0,6)=6 → 1/3.
    true = Entity("X", start=0, end=4)
    pred = Entity("X", start=2, end=6)
    assert abs(JaccardOverlap().compute(true, pred) - 1 / 3) < 1e-9


def test_jaccard_overlap_one_char_overlap():
    # Minimal overlap: true [0,5), pred [4,9): intersection 1, union 9 → 1/9.
    true = Entity("X", start=0, end=5)
    pred = Entity("X", start=4, end=9)
    assert abs(JaccardOverlap().compute(true, pred) - 1 / 9) < 1e-9


def test_jaccard_overlap_no_overlap_returns_zero():
    true = Entity("X", start=0, end=4)
    pred = Entity("X", start=5, end=9)
    assert JaccardOverlap().compute(true, pred) == 0.0


# ---------------------------------------------------------------------------
# ProportionalCoverage tests
# ---------------------------------------------------------------------------

def test_proportional_overlap_span_single_pred_partial_overlap():
    # A pred covering 6 of 10 true chars yields score 0.6.
    true_entity = Entity("NAME", start=0, end=10)
    pred_entity = Entity("NAME", start=0, end=6)

    metrics = _eval(ProportionalCoverage(), [true_entity], [pred_entity]).overall_metrics

    assert metrics.correct == pytest.approx(0.6)
    assert metrics.incorrect == pytest.approx(0.4)
    assert metrics.missed == 0
    assert metrics.spurious == 0


def test_proportional_overlap_span_two_preds_union_covers_full_true_entity():
    # Two non-overlapping preds together covering all 10 chars → score = 1.0.
    true_entity = Entity("NAME", start=0, end=10)
    pred1 = Entity("NAME", start=0, end=5)
    pred2 = Entity("NAME", start=5, end=10)

    metrics = _eval(ProportionalCoverage(), [true_entity], [pred1, pred2]).overall_metrics

    assert metrics.correct == pytest.approx(1.0)
    assert metrics.incorrect == pytest.approx(0.0)
    assert metrics.missed == 0
    assert metrics.spurious == 0


def test_proportional_overlap_span_missed_entity_when_no_pred():
    # A true entity with no overlapping pred is counted as missed.
    true_entity = Entity("NAME", start=0, end=10)

    metrics = _eval(ProportionalCoverage(), [true_entity], []).overall_metrics

    assert metrics.missed == 1
    assert metrics.correct == 0
    assert metrics.spurious == 0


def test_proportional_overlap_span_spurious_pred_when_no_true():
    # A pred with no overlapping true entity is counted as spurious.
    pred_entity = Entity("NAME", start=0, end=10)

    metrics = _eval(ProportionalCoverage(), [], [pred_entity]).overall_metrics

    assert metrics.spurious == 1
    assert metrics.correct == 0
    assert metrics.missed == 0


# ---------------------------------------------------------------------------
# ProportionalCoverage(require_type_match=True) tests
# ---------------------------------------------------------------------------

def test_typed_proportional_overlap_span_wrong_type_pred_does_not_contribute():
    # A pred of the wrong type contributes 0 coverage → score = 0.0.
    true_entity = Entity("NAME", start=0, end=10)
    pred_entity = Entity("ADDRESS", start=0, end=10)

    metrics = _eval(ProportionalCoverage(require_type_match=True), [true_entity], [pred_entity]).overall_metrics

    assert metrics.correct == pytest.approx(0.0)
    assert metrics.incorrect == pytest.approx(1.0)


def test_typed_proportional_overlap_span_all_wrong_type_preds_count_as_incorrect_not_missed():
    # When all preds are wrong-type, the true entity is incorrect (a pred was
    # present and matched), not missed.
    true_entity = Entity("NAME", start=0, end=10)
    pred_entity = Entity("ADDRESS", start=3, end=7)

    metrics = _eval(ProportionalCoverage(require_type_match=True), [true_entity], [pred_entity]).overall_metrics

    assert metrics.incorrect == pytest.approx(1.0)
    assert metrics.missed == 0


def test_typed_proportional_overlap_span_right_type_partial_pred_contributes():
    # A same-type pred covering 6 of 10 chars → score = 0.6.
    true_entity = Entity("NAME", start=0, end=10)
    pred_entity = Entity("NAME", start=0, end=6)

    metrics = _eval(ProportionalCoverage(require_type_match=True), [true_entity], [pred_entity]).overall_metrics

    assert metrics.correct == pytest.approx(0.6)
    assert metrics.incorrect == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Contains tests
# ---------------------------------------------------------------------------

def test_superset_pred_strictly_contains_true_is_correct():
    # Pred [0,10) fully contains true [2,8) → TrueEntityOverlap = 1.0 → correct.
    true_entity = Entity("X", start=2, end=8)
    pred_entity = Entity("X", start=0, end=10)

    metrics = _eval(Contains(), [true_entity], [pred_entity]).overall_metrics

    assert metrics.correct == 1.0
    assert metrics.incorrect == 0.0


def test_superset_pred_strictly_inside_true_is_incorrect():
    # Pred [2,8) is inside true [0,10) → TrueEntityOverlap = 6/10 = 0.6 < 1.0 → incorrect.
    true_entity = Entity("X", start=0, end=10)
    pred_entity = Entity("X", start=2, end=8)

    metrics = _eval(Contains(), [true_entity], [pred_entity]).overall_metrics

    assert metrics.correct == 0.0
    assert metrics.incorrect == 1.0


def test_superset_pred_exactly_equal_to_true_is_correct():
    # Pred span == true span → TrueEntityOverlap = 1.0 → correct.
    true_entity = Entity("X", start=0, end=10)
    pred_entity = Entity("X", start=0, end=10)

    metrics = _eval(Contains(), [true_entity], [pred_entity]).overall_metrics

    assert metrics.correct == 1.0
    assert metrics.incorrect == 0.0


# ---------------------------------------------------------------------------
# MinimumOverlap tests
# ---------------------------------------------------------------------------

def test_minimum_overlap_true_entity_overlap_above_threshold_is_correct():
    # TrueEntityOverlap = 7/10 = 0.7 > threshold 0.5 → correct.
    true_entity = Entity("X", start=0, end=10)
    pred_entity = Entity("X", start=0, end=7)

    metrics = _eval(MinimumOverlap(0.5), [true_entity], [pred_entity]).overall_metrics

    assert metrics.correct == 1.0
    assert metrics.incorrect == 0.0


def test_minimum_overlap_true_entity_overlap_below_threshold_is_incorrect():
    # TrueEntityOverlap = 3/10 = 0.3 < threshold 0.5 → incorrect.
    true_entity = Entity("X", start=0, end=10)
    pred_entity = Entity("X", start=0, end=3)

    metrics = _eval(MinimumOverlap(0.5), [true_entity], [pred_entity]).overall_metrics

    assert metrics.correct == 0.0
    assert metrics.incorrect == 1.0


def test_minimum_overlap_true_entity_overlap_exactly_at_threshold_is_correct():
    # TrueEntityOverlap = 5/10 = 0.5 == threshold 0.5; threshold_inclusive=True → correct.
    true_entity = Entity("X", start=0, end=10)
    pred_entity = Entity("X", start=0, end=5)

    metrics = _eval(MinimumOverlap(0.5), [true_entity], [pred_entity]).overall_metrics

    assert metrics.correct == 1.0
    assert metrics.incorrect == 0.0


# ---------------------------------------------------------------------------
# TypedTextCoverage tests
# ---------------------------------------------------------------------------

def test_typed_text_coverage_wrong_type_pred_does_not_contribute_to_true_entity_coverage():
    # An ADDRESS pred overlapping a NAME true entity contributes 0 to NAME coverage.
    true_entity = Entity("NAME", start=0, end=10)
    pred_entity = Entity("ADDRESS", start=0, end=10)

    metrics = _eval(TextCoverage(require_type_match=True), [true_entity], [pred_entity])

    name = metrics.metrics_by_entity_type["NAME"]
    assert name.correct == 0
    assert name.missed == 10
    assert name.possible == 10


def test_typed_text_coverage_right_type_partial_pred_contributes_proportionally():
    # A NAME pred covering 6 of 10 NAME true chars: correct=6, missed=4.
    true_entity = Entity("NAME", start=0, end=10)
    pred_entity = Entity("NAME", start=0, end=6)

    metrics = _eval(TextCoverage(require_type_match=True), [true_entity], [pred_entity])

    name = metrics.metrics_by_entity_type["NAME"]
    assert name.correct == 6
    assert name.missed == 4
    assert name.possible == 10
    assert name.spurious == 0


def test_typed_text_coverage_cross_type_overlapping_pred_is_not_spurious():
    # An ADDRESS pred fully overlapping a NAME true entity must have spurious=0
    # (it overlapped something, just the wrong type).
    true_entity = Entity("NAME", start=0, end=10)
    pred_entity = Entity("ADDRESS", start=0, end=10)

    metrics = _eval(TextCoverage(require_type_match=True), [true_entity], [pred_entity])

    address = metrics.metrics_by_entity_type["ADDRESS"]
    assert address.spurious == 0
    assert address.actual == 10


def test_typed_text_coverage_non_overlapping_pred_is_fully_spurious():
    # A pred that does not overlap any true entity is fully spurious.
    true_entity = Entity("NAME", start=0, end=5)
    pred_entity = Entity("ADDRESS", start=10, end=20)

    metrics = _eval(TextCoverage(require_type_match=True), [true_entity], [pred_entity])

    address = metrics.metrics_by_entity_type["ADDRESS"]
    assert address.spurious == 10
    assert address.actual == 10


def test_typed_text_coverage_cross_type_chars_counted_as_incorrect():
    # A pred that overlaps a true entity with the wrong type has its overlapping chars
    # counted as incorrect, not spurious: actual = correct + incorrect + spurious.
    true_entity = Entity("NAME", start=0, end=10)
    pred_entity = Entity("ADDRESS", start=0, end=10)

    metrics = _eval(TextCoverage(require_type_match=True), [true_entity], [pred_entity])

    address = metrics.metrics_by_entity_type["ADDRESS"]
    assert address.incorrect == 10
    assert address.spurious == 0
    assert address.actual == address.correct + address.incorrect + address.spurious
