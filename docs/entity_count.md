# Evaluation Strategies

A **strategy** defines what counts as a correct prediction and how much credit to award.
Given a list of true entities and predicted entities, a strategy produces precision, recall, and F1.

All strategies on this page are **entity-count strategies**: they score each true entity, then aggregate:

```
possible = correct + incorrect + missed   # = total true entities
actual   = correct + incorrect + spurious # = total predicted entities

recall    = correct / possible
precision = correct / actual
```

`correct` and `incorrect` can be fractional for some strategies (see [Aggregation](#aggregation));
`missed` and `spurious` are always whole numbers.

## Matching

Before scoring, predicted entities are matched to true entities by any non-zero character overlap.
One predicted entity can overlap multiple true entities and vice versa.
Unmatched true entities count as **missed**; unmatched predicted entities as **spurious**.

## Overview

| [Strategy](#strategies) | A match requires… | Type-sensitive | Score per match | [Aggregation](#aggregation) |
|---|---|---|---|---|
| `Strict` | exact boundaries | yes | 0 or 1 | binary |
| `Exact` | exact boundaries | no | 0 or 1 | binary |
| `EntType` | any overlap | yes | 0 or 1 | binary |
| `Partial` | any overlap | no | 0, 0.5, or 1 | entity-macro |
| `AnyOverlap` | any overlap | no | 0 or 1 | binary |
| `Contains` | prediction contains full true span | no | 0 or 1 | binary |
| `ProportionalCoverage` | any overlap | no | 0–1 (continuous) | entity-macro |
| `MinimumOverlap` | configurable minimum overlap | configurable | 0 or 1 | binary |

## Aggregation

**Binary strategies** (`Strict`, `Exact`, `EntType`, `AnyOverlap`, `Contains`, `MinimumOverlap`):
every entity is either fully correct (1) or not (0).
`correct / possible` is the fraction of true entities found; `correct / actual` is the fraction of predictions that were right.
Precision and recall have their standard meaning.

**Entity-macro strategies** (`Partial`, `ProportionalCoverage`):
a true entity can receive a partial score between 0 and 1.
`correct` becomes a sum of fractional scores, so:

- **recall** = average score per true entity — every entity is weighted equally regardless of span length
- **precision** = average score per predicted entity slot

Every true entity contributes equally to the metric regardless of how long it is.
A 3-character name and a 50-character address each count as one entity.

---

## Strategies

`Strict`, `Exact`, `EntType`, `AnyOverlap`, and `Contains` are convenience subclasses of [`MinimumOverlap`](#minimumoverlap) with fixed parameters:

```python
Strict() = MinimumOverlap(
    threshold=1.0,
    overlap=JaccardOverlap(),
    require_type_match=True,
)

Exact() = MinimumOverlap(
    threshold=1.0,
    overlap=JaccardOverlap(),
)

EntType() = MinimumOverlap(
    threshold=0.0,
    overlap=JaccardOverlap(),
    threshold_inclusive=False,
    require_type_match=True,
)

AnyOverlap() = MinimumOverlap(
    threshold=0.0,
    overlap=JaccardOverlap(),
    threshold_inclusive=False,
)

Contains() = MinimumOverlap(
    threshold=1.0,
    overlap=TrueEntityOverlap(),
)
```

The overlap metric controls how the intersection between spans is measured:

| Metric | Formula | Use when… |
|---|---|---|
| `JaccardOverlap` | intersection / union | both spans should substantially coincide |
| `TrueEntityOverlap` | intersection / true length | the true entity must be substantially covered |
| `PredEntityOverlap` | intersection / pred length | the prediction itself must be substantially precise |

---

### `Strict`

Shorthand for `MinimumOverlap(threshold=1.0, overlap=JaccardOverlap(), require_type_match=True)`.

A prediction is correct if and only if its span boundaries match exactly **and** its type matches.

```python
from spaneval import evaluate
from spaneval.entity import Entity
from spaneval.strategies import Strict

true_entities = [Entity("PERSON", start=0, end=10)]

evaluate(true_entities, [Entity("PERSON", start=0, end=10)]).metrics(Strict())
# correct=1

evaluate(true_entities, [Entity("PERSON", start=0, end=9)]).metrics(Strict())
# incorrect=1  (boundary off by one)

evaluate(true_entities, [Entity("ORG", start=0, end=10)]).metrics(Strict())
# incorrect=1  (type mismatch)
```

**When to use.** The highest bar. Use when exact spans and types matter.

**Watch out.** Very sensitive to tokenization inconsistencies and off-by-one boundary differences. A single trailing space causes a miss.

---

### `Exact`

Shorthand for `MinimumOverlap(threshold=1.0, overlap=JaccardOverlap())`.

A prediction is correct if its span boundaries match exactly. Entity type is ignored.

```python
from spaneval import evaluate
from spaneval.entity import Entity
from spaneval.strategies import Exact

true_entities = [Entity("PERSON", start=0, end=10)]

evaluate(true_entities, [Entity("PERSON", start=0, end=10)]).metrics(Exact())
# correct=1

evaluate(true_entities, [Entity("ORG", start=0, end=10)]).metrics(Exact())
# correct=1  (type ignored)

evaluate(true_entities, [Entity("PERSON", start=0, end=9)]).metrics(Exact())
# incorrect=1  (boundary off by one)
```

**When to use.** Boundary accuracy is the primary concern and type errors are acceptable — e.g. type labels are noisy, or there is only one entity type in the task.

---

### `EntType`

Shorthand for `MinimumOverlap(threshold=0.0, overlap=JaccardOverlap(), threshold_inclusive=False, require_type_match=True)`.

A prediction is correct if it has any character overlap with a true entity **and** the types match.
The amount of overlap is irrelevant — any overlap counts fully.

```python
from spaneval import evaluate
from spaneval.entity import Entity
from spaneval.strategies import EntType

true_entities = [Entity("PERSON", start=0, end=10)]

evaluate(true_entities, [Entity("PERSON", start=8, end=15)]).metrics(EntType())
# correct=1  (1-char overlap, type matches)

evaluate(true_entities, [Entity("ORG", start=8, end=15)]).metrics(EntType())
# incorrect=1  (type mismatch)
```

**When to use.** Type correctness is the primary concern; exact boundaries are not. Useful when the pipeline is good at locating entities but makes type errors.

**Watch out.** A 1-character overlap is treated identically to a perfect boundary match. If span quality also matters, combine with a threshold-based strategy or use `MinimumOverlap(require_type_match=True)`.

---

### `Partial`

Based on the SemEval 2013 shared task evaluation protocol. Any character overlap counts as a match;
the score depends on boundary quality:

- Exact boundaries → **1.0**
- Any overlap, but boundaries differ → **0.5**
- No overlap → missed / spurious

Entity type is ignored.

```python
from spaneval import evaluate
from spaneval.entity import Entity
from spaneval.strategies import Partial

true_entities = [Entity("NAME", start=0, end=10)]

evaluate(true_entities, [Entity("NAME", start=0, end=10)]).metrics(Partial())
# correct=1.0  (exact boundaries)

evaluate(true_entities, [Entity("NAME", start=0, end=7)]).metrics(Partial())
# correct=0.5  (overlap, but boundaries differ)

evaluate(true_entities, [Entity("NAME", start=8, end=15)]).metrics(Partial())
# correct=0.5  (1-char overlap also gives 0.5)
```

**When to use.** When SemEval-compatible numbers are required, or when you want to reward finding the right region without requiring exact boundaries and a flat 0.5 penalty is an acceptable simplification.

**Watch out.** The 0.5 credit is uniform: a 1-character overlap and a 99%-coverage overlap receive the same score.
If the actual degree of overlap matters, use `ProportionalCoverage` instead.

---

### `AnyOverlap`

Shorthand for `MinimumOverlap(threshold=0.0, overlap=JaccardOverlap(), threshold_inclusive=False)`.

Any non-zero character overlap → score = 1.0. Type and boundaries are both ignored.

```python
from spaneval import evaluate
from spaneval.entity import Entity
from spaneval.strategies import AnyOverlap

true_entities = [Entity("NAME", start=0, end=10)]

evaluate(true_entities, [Entity("ORG", start=8, end=15)]).metrics(AnyOverlap())
# correct=1  (1-char overlap; type and boundaries ignored)

evaluate(true_entities, [Entity("NAME", start=10, end=15)]).metrics(AnyOverlap())
# missed=1, spurious=1  (no overlap — adjacent spans don't count)
```

**When to use.** Detection is the only concern — you just want to know whether the entity region was found at all, regardless of precision or type. Provides a loose lower bound on recall.

**Watch out.** Very generous. A model that predicts one span covering the entire document would score
close to 1.0 on recall.

---

### `Contains`

Shorthand for `MinimumOverlap(threshold=1.0, overlap=TrueEntityOverlap())`.

A prediction is correct if it fully contains the true entity span (`pred.start ≤ true.start` and `pred.end ≥ true.end`).

```python
from spaneval import evaluate
from spaneval.entity import Entity
from spaneval.strategies import Contains

true_entities = [Entity("NAME", start=2, end=8)]

evaluate(true_entities, [Entity("NAME", start=0, end=10)]).metrics(Contains())
# correct=1  (superset)

evaluate(true_entities, [Entity("NAME", start=2, end=8)]).metrics(Contains())
# correct=1  (exact match is also a superset)

evaluate(true_entities, [Entity("NAME", start=3, end=8)]).metrics(Contains())
# incorrect=1  (start is clipped)
```

**When to use.** Over-prediction is acceptable, under-prediction is not — e.g. text redaction where missing any part of a sensitive span is a failure.

**Watch out.** Rewards wide, conservative predictions. A model that always predicts the full sentence achieves perfect recall.

---

### `ProportionalCoverage`

Score = fraction of the true entity's characters covered by the prediction(s).
Multiple predictions can jointly cover a true entity; their contributions are summed without double-counting.

```python
from spaneval import evaluate
from spaneval.entity import Entity
from spaneval.strategies import ProportionalCoverage

true_entities = [Entity("NAME", start=0, end=10)]  # 10 chars

evaluate(true_entities, [Entity("NAME", start=0, end=7)]).metrics(ProportionalCoverage())
# correct=0.7  (7/10 chars covered)

evaluate(true_entities, [Entity("NAME", start=0, end=10)]).metrics(ProportionalCoverage())
# correct=1.0  (full coverage)
```

Entity type is ignored. For type-sensitive coverage use `ProportionalCoverage(require_type_match=True)`.

**Aggregation.** Entity-macro: every true entity contributes equally to recall, so a 3-character span
and a 50-character span are weighted the same. This contrasts with [`TextCoverage`](character_count.md),
which weights entities by length.

**When to use.** Partial coverage should be rewarded in proportion to how much of the span was found.
More nuanced than `Partial`, which gives a flat 0.5 for any partial match.

**Watch out.** The score measures coverage of the *true* entity, so recall has a cleaner interpretation
("average fraction of each true entity that was covered") than precision.

---

### `MinimumOverlap`

Configurable binary strategy: specify a minimum overlap ratio and an overlap metric.
Score is 1.0 if the threshold is met, 0.0 otherwise.

```python
# true entity must be at least 80% covered
MinimumOverlap(threshold=0.8)

# prediction must substantially cover the true entity, and type must match
MinimumOverlap(overlap=TrueEntityOverlap(), threshold=0.8, require_type_match=True)
```

```python
from spaneval import evaluate
from spaneval.entity import Entity
from spaneval.strategies import MinimumOverlap

true_entities = [Entity("NAME", start=0, end=10)]

evaluate(true_entities, [Entity("NAME", start=0, end=9)]).metrics(MinimumOverlap(threshold=0.8))
# correct=1  (Jaccard = 9/10 = 0.9 ≥ 0.8)

evaluate(true_entities, [Entity("NAME", start=0, end=7)]).metrics(MinimumOverlap(threshold=0.8))
# incorrect=1  (Jaccard = 7/10 = 0.7 < 0.8)
```

See the [overlap metrics table](#strategies) above for the three available overlap metrics and when to use each.

**When to use.** You have a specific quality threshold in mind and want a clean binary pass/fail.
E.g. "at least 80% of the true span must be covered, type required."

---

## Behaviour notes

**Type matching on best-overlap pred only.** When multiple predicted entities overlap a single true entity, type checking is applied only to the predicted entity with the highest character overlap. This is intentional: the best-overlap pred is the candidate the model most likely intended for that span.

**`incorrect` is always `1 - score`.** Every strategy sets `incorrect = 1 - score` for each matched true entity. This means `incorrect` is non-zero even for a partial match — e.g. 80% coverage under `ProportionalCoverage` gives `correct=0.8, incorrect=0.2`. `incorrect_docs()` therefore returns any document with an imperfect match. To find documents where all overlapping matches were fully correct, use `AnyOverlap`.
