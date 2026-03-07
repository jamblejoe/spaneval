# Results: Reporting and Scoring

After evaluation, `Results` exposes three reporting modes and a numeric scoring API for automated optimization.

```python
from spaneval import evaluate, Goal
from spaneval.strategies import Strict, ProportionalCoverage

results = evaluate(true, pred)
```

---

## Exploration report (default)

```python
results.report()
```

Shows precision and recall as midpoint ± half-range across two strategies (Strict / AnyOverlap). This gives an instant sense of how much your numbers depend on the choice of strategy.

```
Strategies: Strict, AnyOverlap

Entity Type    Number    Missed    Spurious    Precision (%)    Recall (%)
        Overall        42         3           5       78 ±  9      82 ±  6
         PERSON        18         1           2       90 ± 10      94 ±  5
           DATE        14         2           3       62 ± 22      71 ± 14
            ORG        10         0           0       88 ±  3      88 ±  3
```

A wide ± means span boundaries are frequently imprecise. A narrow ± means predictions are either clearly right or clearly wrong.

---

## Single-strategy report

```python
results.report(strategy=ProportionalCoverage())
```

Exact precision, recall, and F1 for one strategy.

```
Entity Type    Number    Missed    Spurious    Precision (%)    Recall (%)    F1 (%)
        Overall        42         3           5               81            85        83
         PERSON        18         1           2               92            95        93
           DATE        14         2           3               71            77        74
            ORG        10         0           0               88            88        88
```

---

## Per-type strategy report

```python
results.report(
    strategy={"PERSON": Strict(), "DATE": ProportionalCoverage(), "ORG": ProportionalCoverage()},
)
```

Each entity type is scored under its own strategy. The Overall row is a composite: per-type metrics are summed. If any type present in the results is not covered by the dict and `default_strategy` is not set, a `ValueError` names the missing types.

```python
# Provide a fallback for uncovered types:
results.report(
    strategy={"PERSON": Strict()},
    default_strategy=ProportionalCoverage(),
)
```

---

## Programmatic metric access

Use `metrics()` and `metrics_by_type()` to retrieve `Metrics` objects for downstream use (logging, CI checks, custom aggregation).

### `metrics()` — one number for everything

Returns a **single `Metrics`** aggregated across all entity types. This is the programmatic equivalent of the Overall row in `report()`.

```python
m = results.metrics(Strict())
m.precision  # overall precision across all types
m.recall     # overall recall across all types
m.f1         # overall F1
```

### `metrics_by_type()` — one number per entity type

Returns a **`dict[str, Metrics]`** with one entry per entity type. Use this when you need to inspect or log individual types separately.

```python
by_type = results.metrics_by_type(Strict())
by_type["PERSON"].precision  # PERSON precision only
by_type["DATE"].recall       # DATE recall only
```

Both methods accept a single strategy (applied to all types) or a dict of per-type strategies:

```python
# Same strategy for all types:
results.metrics(Strict())
results.metrics_by_type(Strict())

# Different strategy per type:
results.metrics({"PERSON": Strict(), "DATE": ProportionalCoverage()})
results.metrics_by_type({"PERSON": Strict(), "DATE": ProportionalCoverage()})

# Dict with a fallback for uncovered types:
results.metrics({"PERSON": Strict()}, default_strategy=ProportionalCoverage())
```

Typical use cases:

```python
# Experiment tracking
m = results.metrics(Strict())
mlflow.log_metrics({"precision": m.precision, "recall": m.recall, "f1": m.f1})

# CI regression check
assert m.recall >= 0.90, f"recall dropped to {m.recall:.2f}"

# Compare two runs
before = evaluate(true, pred_v1).metrics(Strict())
after  = evaluate(true, pred_v2).metrics(Strict())
print(f"F1 delta: {after.f1 - before.f1:+.3f}")
```

---

## Goal-based report

```python
goals = {
    "PERSON": Goal(strategy=Strict(),              recall=0.90, precision=0.80),
    "DATE":   Goal(strategy=ProportionalCoverage(), recall=0.80, precision=0.70),
    "ORG":    Goal(strategy=ProportionalCoverage(), recall=0.85, precision=0.70),
}

results.report_goals(goals)
```

Prints a scorecard showing actual vs. target recall and precision for each type. Scores are normalized: 1.0 = exactly on target, > 1.0 = above target (uncapped). The `←` marks the bottleneck — the weakest type/metric combination.

```
Entity Type          Strategy    Recall    R-Target    R-Score    Precision    P-Target    P-Score
        Overall        (goals)                                        0.91 ←
         PERSON        Strict      0.95        0.90       1.06         0.82        0.80       1.02
           DATE  ProportionalC    0.71        0.80       0.89 ←        0.75        0.70       1.07
            ORG  ProportionalC    0.88        0.85       1.04         0.91        0.70       1.30
```

---

## Bottleneck score

```python
score = results.score(goals)  # float, no output printed
```

Returns the same bottleneck value shown in the Overall row of `report_goals()` — the minimum normalized score across all type/metric pairs. Use this in automated loops:

```python
best_score = 0.0
for prompt in candidate_prompts:
    predictions = run_llm(prompt, documents)
    score = evaluate(true, predictions).score(goals)
    if score > best_score:
        best_score = score
        best_prompt = prompt
```

The bottleneck property ensures the optimizer cannot improve the score by sacrificing one entity type for another — it always has to fix the weakest link.

---

## Diagnostic helpers

These do not depend on a strategy — they reflect which entities had no overlapping counterpart at all:

```python
results.missed_docs    # list[int] — doc indices with at least one unmatched true entity
results.spurious_docs  # list[int] — doc indices with at least one unmatched prediction
```

`incorrect_docs` requires a strategy, because it covers the case where a prediction *did* overlap a true entity but failed the strategy's scoring threshold:

```python
results.incorrect_docs(strategy=Strict())
# list[int] — doc indices where at least one overlapping prediction was scored as incorrect
```
