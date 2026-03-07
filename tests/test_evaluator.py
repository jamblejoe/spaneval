import pytest

from spaneval import evaluate
from spaneval.entity import Entity
from spaneval.strategies import Strict


def test_evaluate_returns_results():
    results = evaluate([Entity("PERSON", 0, 5)], [Entity("PERSON", 0, 5)])
    assert results is not None


def test_evaluate_single_document_auto_detected():
    # A flat list of entities is treated as a single document.
    results = evaluate([Entity("PERSON", 0, 5)], [Entity("PERSON", 0, 5)])
    assert len(results.matches) == 1


def test_evaluate_multi_document_auto_detected():
    # A list of lists is treated as multiple documents.
    results = evaluate(
        [[Entity("PERSON", 0, 5)], [Entity("ORG", 10, 15)]],
        [[Entity("PERSON", 0, 5)], [Entity("ORG", 10, 15)]],
    )
    assert len(results.matches) == 2


def test_evaluate_raises_on_mismatched_doc_count():
    with pytest.raises(ValueError, match="Mismatch"):
        evaluate(
            [[Entity("PERSON", 0, 5)], [Entity("ORG", 10, 15)]],
            [[Entity("PERSON", 0, 5)]],
        )


def test_evaluate_raises_when_both_ignore_params_specified():
    with pytest.raises(ValueError, match="Cannot specify both"):
        evaluate([], [], ignore_types=["MISC"], ignore_all_types_but=["PERSON"])


def test_evaluate_ignore_types_excludes_entity_type():
    # MISC entities must not appear in results when ignore_types=["MISC"].
    true = [Entity("PERSON", 0, 5), Entity("MISC", 10, 15)]
    pred = [Entity("PERSON", 0, 5), Entity("MISC", 10, 15)]
    results = evaluate(true, pred, ignore_types=["MISC"])
    assert "MISC" not in results.entity_types


def test_evaluate_ignore_all_types_but_includes_only_specified():
    # Only PERSON entities are evaluated; DATE is excluded.
    true = [Entity("PERSON", 0, 5), Entity("DATE", 10, 15)]
    pred = [Entity("PERSON", 0, 5)]
    results = evaluate(true, pred, ignore_all_types_but=["PERSON"])
    assert results.entity_types == ["PERSON"]


def test_evaluate_any_strategy_usable_on_results():
    # Any strategy can be applied at report/goal_score time without pre-registration.
    from spaneval.strategies import Partial
    results = evaluate([Entity("PERSON", 0, 10)], [Entity("PERSON", 0, 8)])
    m = results._type_metrics(Partial(), "PERSON")
    assert m is not None
