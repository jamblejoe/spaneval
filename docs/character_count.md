# Text Coverage

`TextCoverage` is a **character-level** strategy. Instead of counting entities as correct or incorrect, it counts characters. Every character in a true entity or predicted entity contributes individually to the metrics, so a 50-character address contributes 50 times as much as a 1-character initial.

This is the alternative to the [entity-count strategies](entity_count.md), which treat every entity as one unit regardless of length.

## Metrics

The six metric fields (`correct`, `incorrect`, `missed`, `spurious`, `possible`, `actual`) retain their names but their unit is now **characters**, not entities.

| Field | Meaning |
|---|---|
| `correct` | Characters of true entities covered by a prediction (correct type, if `require_type_match=True`) |
| `incorrect` | Characters of predictions that overlap a true entity of the wrong type (`require_type_match=True` only; always 0 otherwise) |
| `missed` | Characters of true entities not covered by any correct-type prediction |
| `spurious` | Characters of predictions that do not overlap any true entity |
| `possible` | Total characters across all true entities |
| `actual` | Total characters across all predicted entities |

The identity `actual = correct + incorrect + spurious` holds at the overall and per-entity-type level.

```
recall    = correct / possible  =  covered true chars / all true chars
precision = correct / actual    =  covered true chars / all pred chars
```

## `TextCoverage`

```python
from spaneval.strategies import TextCoverage

TextCoverage()
TextCoverage(require_type_match=True)
```

**Parameters.**

| Parameter | Default | Effect |
|---|---|---|
| `require_type_match` | `False` | If `True`, a prediction only covers characters of a true entity with the same type. Wrong-type overlapping characters are counted as `incorrect` rather than `correct`, penalising precision. |

### How characters are counted

Every character of every true entity is either **correctly covered**, **incorrectly covered** (wrong-type prediction, `require_type_match=True` only), or **missed**. Every character of every predicted entity is either **correct**, **incorrect**, or **spurious**. Multiple predictions can jointly cover the same true entity; their contributions are summed.

### Worked example

```python
from spaneval import evaluate
from spaneval.entity import Entity
from spaneval.strategies import TextCoverage

# text = "Jane Brown is a doctor"
#         0         1         2
#         0123456789012345678901

true_entities = [
    Entity("NAME", start=0,  end=10),  # "Jane Brown"
    Entity("ROLE", start=16, end=22),  # "doctor"
]
pred_entities = [
    Entity("NAME", start=0,  end=4),   # "Jane"     — partial NAME hit
    Entity("NAME", start=14, end=22),  # "a doctor" — ROLE found but overshoots
]

m = evaluate(true_entities, pred_entities).metrics(TextCoverage())
# correct=10, missed=6, spurious=2, possible=16, actual=12
# recall    = 10/16 = 0.62  (only "Jane" found; "Brown" missed)
# precision = 10/12 = 0.83  (2 chars of "a " outside any true entity)
```

### Interaction with `require_type_match`

When `require_type_match=True`, a prediction of the wrong type does not count as covering those characters. Its overlapping characters are counted as `incorrect` — they hit the right location but carry the wrong label. The distinction between `incorrect` and `spurious` tells you *why* precision is below 1.0:

| | `spurious` | `incorrect` |
|---|---|---|
| Overlaps a true entity? | No | Yes |
| Correct type? | — | No |
| Interpretation | prediction was noise | prediction found the right span but wrong type |

```python
from spaneval import evaluate
from spaneval.entity import Entity
from spaneval.strategies import TextCoverage

# text = "Jane Brown"
#         0123456789

true_entities = [Entity("NAME", start=0, end=10)]  # "Jane Brown"
pred_entities = [Entity("ORG",  start=0, end=10)]  # same span, wrong type

m = evaluate(true_entities, pred_entities).metrics(TextCoverage())
# correct=10, incorrect=0, missed=0, spurious=0
# recall=1.0, precision=1.0

m = evaluate(true_entities, pred_entities).metrics(TextCoverage(require_type_match=True))
# correct=0, incorrect=10, missed=10, spurious=0
# recall=0.0, precision=0.0
```

With `require_type_match=False`, the `incorrect` field is always 0.

### When to use

`TextCoverage` is well-suited for **text redaction and anonymization** tasks where the goal is measured in characters:

- **Recall** answers: "What fraction of sensitive text did the model find?"
- **Precision** answers: "What fraction of the text the model flagged was actually sensitive?"

Use it when longer entities should carry more weight in the score — finding a 50-character address matters more than finding a 2-character initials.

### Watch out

- **Long entities dominate.** A single long missed entity can substantially lower recall even if all other entities are found. If every entity should count equally, use [`ProportionalCoverage`](entity_count.md#proportionalcoverage) instead.

- **Wide predictions hurt precision.** A prediction that covers a true entity but extends far beyond it — e.g. flagging an entire sentence to ensure a name inside is captured — generates spurious characters that lower precision.

---

## Comparison with `ProportionalCoverage`

Both strategies reward partial coverage, but they weight entities differently:

| | `ProportionalCoverage` | `TextCoverage` |
|---|---|---|
| Unit | entities (entity-macro) | characters (character-macro) |
| Weighting | every entity counts equally | long entities contribute more |
| Precision | average score per predicted entity | correctly covered chars / all pred chars |
| Recall | average fraction of each true entity covered | correctly covered chars / all true chars |

Use `ProportionalCoverage` when all entities should have equal influence on the score regardless of length. Use `TextCoverage` when the amount of sensitive text found matters.
