import math
import pytest

from spaneval import evaluate
from spaneval.entity import Entity, Match
from spaneval.results import Goal, Results
from spaneval.strategies import AnyOverlap, EntType, Partial, ProportionalCoverage, Strict, TextCoverage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_type_results():
    """Multi-document results with PERSON and DATE entities.

    PERSON: predicted correctly in all docs (Strict-correct).
    DATE:   predicted with a span off by two characters (Strict-incorrect,
            EntType-correct because there is non-zero overlap and the type matches).
    """
    true = [
        [Entity("PERSON", 0, 10), Entity("DATE", 20, 30)],
        [Entity("PERSON", 5, 15)],
    ]
    pred = [
        [Entity("PERSON", 0, 10), Entity("DATE", 20, 28)],  # DATE span short by 2
        [Entity("PERSON", 5, 15)],
    ]
    return evaluate(true, pred)


@pytest.fixture
def partial_recall_results():
    """Results where PERSON has recall=0.5 under Strict (one of two missed).

    PERSON: 2 true, 1 predicted correctly → recall=0.5, precision=1.0
    DATE:   1 true, 1 predicted correctly → recall=1.0, precision=1.0
    """
    true = [Entity("PERSON", 0, 5), Entity("PERSON", 10, 15), Entity("DATE", 20, 25)]
    pred = [Entity("PERSON", 0, 5), Entity("DATE", 20, 25)]
    return evaluate(true, pred)


# ---------------------------------------------------------------------------
# NERResults aggregation and per-type robustness
# ---------------------------------------------------------------------------

def test_per_type_metrics_when_entity_type_absent_from_some_documents():
    # _type_metrics must not raise KeyError when an entity type
    # appears in one document but is absent from another.
    doc_true = [
        [Entity("PERSON", 0, 5), Entity("DATE", 10, 20)],
        [Entity("PERSON", 0, 5)],  # DATE is absent here
    ]
    doc_pred = [
        [Entity("PERSON", 0, 5)],
        [Entity("PERSON", 0, 5)],
    ]
    results = evaluate(doc_true, doc_pred)

    # Neither of these must raise
    results.report()
    date_m = results._type_metrics(Strict(), "DATE")

    # DATE missed in doc 0; no DATE entities in doc 1 — recall must be 0
    assert date_m.recall == 0.0
    assert date_m.missed == 1


def test_ner_results_smoke(capsys):
    results = evaluate([Entity("PERSON", 1, 4)], [Entity("PERSON", 1, 4)])
    results.report()
    # checks that report runs without error and outputs something
    captured = capsys.readouterr()
    assert captured.out


# ---------------------------------------------------------------------------
# metrics() / metrics_by_type()
# ---------------------------------------------------------------------------

def test_metrics_single_strategy_equals_overall_metric(two_type_results):
    # When a single strategy is given, metrics() must equal the overall metric
    # for that strategy (summing per-type is equivalent to summing per-doc overall).
    overall = two_type_results._overall_metrics(Strict())
    composite = two_type_results.metrics(Strict())

    assert math.isclose(composite.precision, overall.precision)
    assert math.isclose(composite.recall,    overall.recall)
    assert math.isclose(composite.f1,        overall.f1)


def test_metrics_per_type_dict_selects_correct_per_type_metrics(two_type_results):
    # With {"PERSON": Strict(), "DATE": EntType()}, PERSON is scored under Strict
    # (exact span required) and DATE under EntType (any overlap, correct type).
    # PERSON matches exactly in both docs → perfect.
    # DATE span is off by 2 chars → incorrect under Strict but correct under EntType.
    # Composite recall and precision must therefore be 1.0.
    composite = two_type_results.metrics({"PERSON": Strict(), "DATE": EntType()})

    assert math.isclose(composite.recall,    1.0)
    assert math.isclose(composite.precision, 1.0)


def test_metrics_raises_for_uncovered_type_when_no_default(two_type_results):
    # When a type present in the results has no assigned strategy and
    # default_strategy is None, a ValueError naming the missing type must be raised.
    with pytest.raises(ValueError, match="DATE"):
        two_type_results.metrics({"PERSON": Strict()})


def test_metrics_uses_default_strategy_for_uncovered_types(two_type_results):
    # When default_strategy is provided, uncovered types are evaluated under it
    # without raising.
    composite = two_type_results.metrics(
        {"PERSON": Strict()},
        default_strategy=EntType(),
    )
    # DATE under EntType: correct (overlap exists, type matches) → precision=recall=1.0 for DATE
    # PERSON under Strict: also correct → combined recall and precision both 1.0
    assert math.isclose(composite.recall,    1.0)
    assert math.isclose(composite.precision, 1.0)


def test_metrics_by_type_returns_per_type_breakdown(two_type_results):
    # metrics_by_type() must return one Metrics per entity type.
    by_type = two_type_results.metrics_by_type(Strict())

    assert set(by_type.keys()) == {"PERSON", "DATE"}
    # PERSON exact in both docs → recall=precision=1.0
    assert math.isclose(by_type["PERSON"].recall,    1.0)
    assert math.isclose(by_type["PERSON"].precision, 1.0)
    # DATE span off by 2 chars → incorrect under Strict → recall < 1.0
    assert by_type["DATE"].recall < 1.0


def test_metrics_by_type_per_type_dict_applies_correct_strategies(two_type_results):
    # With DATE under EntType, DATE must score perfectly despite the span offset.
    by_type = two_type_results.metrics_by_type({"PERSON": Strict(), "DATE": EntType()})

    assert math.isclose(by_type["PERSON"].recall,    1.0)
    assert math.isclose(by_type["DATE"].recall,      1.0)
    assert math.isclose(by_type["DATE"].precision,   1.0)


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------

def test_goal_raises_typeerror_with_hint_when_strategy_class_passed_instead_of_instance():
    # Goal(strategy=Strict) must raise TypeError with a message suggesting Strict().
    with pytest.raises(TypeError, match="Strict()"):
        Goal(strategy=Strict, recall=0.9, precision=0.8)


def test_score_returns_minimum_of_normalised_scores(partial_recall_results):
    # PERSON under Strict: recall=0.5, precision=1.0
    # DATE under Strict:   recall=1.0, precision=1.0
    # With targets recall=0.8 / precision=0.8 for both:
    #   PERSON recall score    = 0.5 / 0.8 = 0.625  ← bottleneck
    #   PERSON precision score = 1.0 / 0.8 = 1.25
    #   DATE recall score      = 1.0 / 0.8 = 1.25
    #   DATE precision score   = 1.0 / 0.8 = 1.25
    # score must equal 0.5 / 0.8.
    goals = {
        "PERSON": Goal(strategy=Strict(), recall=0.8, precision=0.8),
        "DATE":   Goal(strategy=Strict(), recall=0.8, precision=0.8),
    }
    score = partial_recall_results.score(goals)
    assert math.isclose(score, 0.5 / 0.8)


def test_score_is_uncapped_when_metrics_exceed_targets(two_type_results):
    # PERSON and DATE are both predicted correctly under their respective
    # strategies, so normalised scores exceed 1.0 when targets are below 1.0.
    goals = {
        "PERSON": Goal(strategy=Strict(),  recall=0.5, precision=0.5),
        "DATE":   Goal(strategy=EntType(), recall=0.5, precision=0.5),
    }
    score = two_type_results.score(goals)
    assert score > 1.0


def test_score_returns_one_when_all_goal_types_absent_by_default(two_type_results):
    # When all goal types are absent from the results, score() returns 1.0 by
    # default (require_all_types=False) — nothing to find, goals trivially met.
    assert two_type_results.score(
        {"PHONE": Goal(strategy=Strict(), recall=0.9, precision=0.8)}
    ) == 1.0


def test_score_raises_naming_phantom_type_when_required(two_type_results):
    # With require_all_types=True, score() raises a ValueError that names the
    # phantom type, which helps catch typos in goal type names.
    with pytest.raises(ValueError, match="PHONE"):
        two_type_results.score(
            {"PHONE": Goal(strategy=Strict(), recall=0.9, precision=0.8)},
            require_all_types=True,
        )



# ---------------------------------------------------------------------------
# missed_docs / spurious_docs / incorrect_docs
# ---------------------------------------------------------------------------

@pytest.fixture
def multi_doc_diverse_results():
    """Three documents covering all diagnostic scenarios.

    Doc 0: PERSON exact match — no issues.
    Doc 1: DATE not predicted at all → missed.
    Doc 2: DATE span short by 2 chars (overlaps but imprecise) + spurious PERSON prediction.
    """
    true = [
        [Entity("PERSON", 0, 10)],
        [Entity("DATE", 20, 30)],
        [Entity("DATE", 20, 30)],
    ]
    pred = [
        [Entity("PERSON", 0, 10)],
        [],
        [Entity("DATE", 20, 28), Entity("PERSON", 40, 50)],
    ]
    return evaluate(true, pred)


def test_missed_docs_returns_indices_of_documents_with_unmatched_true_entities(multi_doc_diverse_results):
    # Doc 1: DATE has no prediction → its match has pred_entity=None
    assert multi_doc_diverse_results.missed_docs == [1]


def test_missed_docs_is_empty_when_all_true_entities_have_overlapping_predictions(two_type_results):
    # DATE(20,28) overlaps DATE(20,30) → not missed; PERSON exact → not missed
    assert two_type_results.missed_docs == []


def test_spurious_docs_returns_indices_of_documents_with_unmatched_predictions(multi_doc_diverse_results):
    # Doc 2: PERSON(40,50) has no overlapping true entity → spurious
    assert multi_doc_diverse_results.spurious_docs == [2]


def test_spurious_docs_is_empty_when_all_predictions_overlap_a_true_entity(two_type_results):
    # All predictions in two_type_results overlap a true entity
    assert two_type_results.spurious_docs == []


def test_incorrect_docs_returns_indices_where_prediction_fails_strategy_threshold(multi_doc_diverse_results):
    # Doc 2: DATE(20,28) overlaps DATE(20,30) but Jaccard = 8/10 < 1.0 → incorrect under Strict
    assert multi_doc_diverse_results.incorrect_docs(strategy=Strict()) == [2]


def test_incorrect_docs_is_empty_when_any_overlap_counts_as_full_credit(multi_doc_diverse_results):
    # AnyOverlap: any non-zero Jaccard overlap → threshold exceeded → score=1.0 → incorrect=0.
    # Doc 2's DATE(20,28) overlaps DATE(20,30), so the entity is fully correct; no doc qualifies.
    assert multi_doc_diverse_results.incorrect_docs(strategy=AnyOverlap()) == []


def test_report_warns_when_per_type_dict_mixes_entity_count_and_text_coverage(two_type_results):
    # Mixing EntityCountStrategy (Strict) and TextCoverage in one dict means the Overall row
    # sums entities and characters — incompatible units. A UserWarning must be emitted.
    with pytest.warns(UserWarning, match="mixes EntityCount and TextCoverage"):
        two_type_results.report(strategy={"PERSON": Strict(), "DATE": TextCoverage()})


def test_report_warns_when_default_strategy_is_text_coverage_and_dict_has_entity_count(two_type_results):
    # The incompatible-unit mixing can also arise via default_strategy.
    with pytest.warns(UserWarning, match="mixes EntityCount and TextCoverage"):
        two_type_results.report(strategy={"PERSON": Strict()}, default_strategy=TextCoverage())


def test_report_does_not_warn_when_all_strategies_are_entity_count(two_type_results, recwarn):
    # A pure entity-count dict is fine; no mixed-unit warning should be emitted.
    two_type_results.report(strategy={"PERSON": Strict(), "DATE": AnyOverlap()})
    mixing_warnings = [w for w in recwarn if "mixes EntityCount and TextCoverage" in str(w.message)]
    assert not mixing_warnings


def test_report_does_not_warn_when_all_strategies_are_text_coverage(two_type_results, recwarn):
    # A uniform TextCoverage dict is fine; no mixed-unit warning should be emitted.
    two_type_results.report(strategy={"PERSON": TextCoverage(), "DATE": TextCoverage()})
    mixing_warnings = [w for w in recwarn if "mixes EntityCount and TextCoverage" in str(w.message)]
    assert not mixing_warnings
