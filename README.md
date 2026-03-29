# spaneval

Pure evaluation library for **span-level entity extraction** — takes ground truth and predicted `(entity_type, start, end)` tuples and returns precision, recall, and F1. Works with any pipeline that produces character-span predictions: LLM extractors, fine-tuned models, rule-based systems. Configurable overlap strategies, per-type precision/recall targets, and a scalar optimization score for automated prompt engineering.

Inspired by and a generalization of [nervaluate](https://github.com/MantisAI/nervaluate).

## Installation

```
pip install spaneval
```

## Relationship to seqeval and nervaluate

**seqeval** works on IOB/BIO token sequences. LLMs output character spans — converting between the two is lossy and inconvenient.

**nervaluate** is the closest prior work and covers the SemEval 2013 strategies well. This library extends the same foundation with configurable overlap thresholds, per-type strategy assignment, and `score()` — a scalar optimization target for automated prompt engineering and hyperparameter search. Precision, recall, and F1 for the four shared strategies (`Strict`, `Exact`, `EntType`, `Partial`) are verified to match nervaluate on identical inputs.

## Quickstart

```python
from spaneval import evaluate, to_entities

true = to_entities([
    {"entity_type": "PERSON", "start":  0, "end": 10},
    {"entity_type": "ORG",    "start": 34, "end": 44},
])
pred = to_entities([
    {"entity_type": "PERSON", "start":  0, "end":  9},  # slightly off boundary
    {"entity_type": "ORG",    "start": 34, "end": 44},  # exact
])

results = evaluate(true, pred)
results.report()
```

`report()` with no arguments shows a ± range across two strategies (Strict / AnyOverlap), giving an instant picture of how boundary precision affects your numbers.

For a guided walkthrough, see the examples:

- `examples/quickstart.py` — five steps from zero-config to per-type strategy assignment
- `examples/goals.py` — two steps from goal definition to automated prompt optimization

## Strategies

Two branches, covered in full in [`docs/`](docs/README.md):

**Entity-count strategies** score each entity as correct, incorrect, missed, or spurious:

| Strategy | Span | Type |
|---|---|---|
| `Strict` | Exact boundaries | Required |
| `Exact` | Exact boundaries | Ignored |
| `EntType` | Any overlap | Required |
| `Partial` | Any overlap; 0.5 credit if boundaries differ | Ignored |
| `ProportionalCoverage` | Fraction of true-entity characters covered | Ignored |
| `AnyOverlap` | Any overlap = full credit | Ignored |
| `Contains` | Prediction must fully contain the true span | Ignored |
| `MinimumOverlap` | Configurable threshold and overlap metric | Configurable |

**TextCoverage** counts characters rather than entities — useful when the goal is text redaction rather than entity classification.

## Per-type strategies and goals

Different entity types can be evaluated under different strategies and held to different targets:

```python
from spaneval.strategies import Strict, ProportionalCoverage

# different strictness per type
results.report(strategy={"PERSON": Strict(), "DATE": ProportionalCoverage()})

# precision/recall targets per type
from spaneval import Goal
goals = {
    "PERSON": Goal(strategy=Strict(),               recall=0.90, precision=0.80),
    "DATE":   Goal(strategy=ProportionalCoverage(),  recall=0.80, precision=0.70),
}
results.report_goals(goals)
```
