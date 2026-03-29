"""Parity tests against nervaluate for the four shared SemEval strategies.

nervaluate is the reference implementation for Strict, Exact, EntType, and
Partial (SemEval 2013 evaluation). These tests verify that spaneval produces
the same precision, recall, and F1 on identical inputs.

Raw counts are NOT compared for Partial: nervaluate tracks a separate
``partial`` counter (integer); spaneval folds partial matches into correct /
incorrect as 0.5 fractions. The final metrics agree; the intermediate tallies
differ by design.

Test inputs are kept free of:
- Overlapping true entities  (spaneval raises ValueError; nervaluate does not)
- Overlapping predicted entities  (spaneval resolves by longest-span; nervaluate
  leaves resolution to the caller)
"""
import pytest
from nervaluate import Evaluator

from spaneval import evaluate
from spaneval.entity import Entity


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_nervaluate(entities: list[Entity]) -> list[dict]:
    """Convert a spaneval Entity list to nervaluate's dict format."""
    return [{"label": e.entity_type, "start": e.start, "end": e.end} for e in entities]


def _spaneval_metrics(true: list[Entity], pred: list[Entity], strategy):
    """Return overall precision/recall/f1 from spaneval for one document."""
    results = evaluate(true, pred)
    m = results.metrics(strategy)
    return m.precision, m.recall, m.f1


def _nervaluate_metrics(
    true: list[Entity], pred: list[Entity], scenario: str, tags: list[str]
) -> tuple[float, float, float]:
    """Return precision/recall/f1 from nervaluate for the given scenario.

    nervaluate crashes with IndexError when any document's entity list is
    empty; those cases are skipped at the call site.
    """
    nv_true = [_to_nervaluate(true)]
    nv_pred = [_to_nervaluate(pred)]
    evaluator = Evaluator(nv_true, nv_pred, tags=tags)
    out = evaluator.evaluate()
    m = out["overall"][scenario]
    return m.precision, m.recall, m.f1


# ---------------------------------------------------------------------------
# Test cases
# Each entry is (description, true_entities, pred_entities, entity_tags)
# ---------------------------------------------------------------------------

CASES: list[tuple[str, list[Entity], list[Entity], list[str]]] = [
    (
        "all correct — exact spans and matching types",
        [Entity("PER", 0, 4), Entity("ORG", 10, 16)],
        [Entity("PER", 0, 4), Entity("ORG", 10, 16)],
        ["PER", "ORG"],
    ),
    (
        "all missed — no predictions",
        [Entity("PER", 0, 4), Entity("ORG", 10, 16)],
        [],
        ["PER", "ORG"],
    ),
    (
        "all spurious — no true entities",
        [],
        [Entity("PER", 0, 4), Entity("ORG", 10, 16)],
        ["PER", "ORG"],
    ),
    (
        "correct span, wrong type",
        [Entity("PER", 0, 4)],
        [Entity("ORG", 0, 4)],
        ["PER", "ORG"],
    ),
    (
        "partial span overlap — predicted span shorter than true",
        [Entity("PER", 0, 8)],
        [Entity("PER", 0, 4)],
        ["PER"],
    ),
    (
        "partial span overlap — predicted span longer than true",
        [Entity("PER", 2, 6)],
        [Entity("PER", 0, 8)],
        ["PER"],
    ),
    (
        "mixed: one correct, one missed, one spurious",
        [Entity("PER", 0, 4), Entity("ORG", 10, 16)],
        [Entity("PER", 0, 4), Entity("LOC", 20, 26)],
        ["PER", "ORG", "LOC"],
    ),
    (
        "single entity, no overlap between true and pred",
        [Entity("PER", 0, 4)],
        [Entity("PER", 10, 14)],
        ["PER"],
    ),
]


# ---------------------------------------------------------------------------
# Strategy mapping
# nervaluate scenario name → spaneval strategy import
# ---------------------------------------------------------------------------

from spaneval.strategies import Strict, Exact, EntType, Partial  # noqa: E402

STRATEGIES = [
    ("strict",   Strict()),
    ("exact",    Exact()),
    ("ent_type", EntType()),
    ("partial",  Partial()),
]


# ---------------------------------------------------------------------------
# Parametrized parity tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario,strategy", STRATEGIES, ids=[s for s, _ in STRATEGIES])
@pytest.mark.parametrize("description,true,pred,tags", CASES, ids=[c[0] for c in CASES])
def test_parity(description, true, pred, tags, scenario, strategy):
    if not true or not pred:
        pytest.skip("nervaluate crashes on empty entity lists (upstream bug)")

    sp_p, sp_r, sp_f1 = _spaneval_metrics(true, pred, strategy)
    nv_p, nv_r, nv_f1 = _nervaluate_metrics(true, pred, scenario, tags)

    assert sp_p == pytest.approx(nv_p, abs=1e-9), f"precision mismatch ({description})"
    assert sp_r == pytest.approx(nv_r, abs=1e-9), f"recall mismatch ({description})"
    assert sp_f1 == pytest.approx(nv_f1, abs=1e-9), f"f1 mismatch ({description})"
