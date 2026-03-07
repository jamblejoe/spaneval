# Matching

Before any strategy scores entities, the raw true and predicted entity lists are preprocessed and paired into `Match` objects.

## Preprocessing

**True entities** must be non-overlapping. If two true entities share any character, a `ValueError` is raised. They are sorted by start offset.

**Predicted entities** may overlap. Overlapping predictions are resolved automatically: within each group of mutually overlapping predictions, the longest span is kept and the rest are silently dropped (a `UserWarning` is emitted by default; pass `warn_on_overlapping_preds=False` to suppress it). The resolved list is sorted by start offset.

## Matching

A `Match` pairs one true entity with one predicted entity that overlaps it by at least one character. The relationship is many-to-many:

- One predicted entity can overlap several true entities → one `Match` per true entity it touches.

  *Example:* ground truth annotates first and last name separately; the model predicts the full name as one span.
  ```python
  from spaneval.entity import Entity, EntityMatcher

  # text =         "Anna Berg"
  # position       0123456789
  true = [Entity("FIRST_NAME", 0, 4), Entity("LAST_NAME", 5, 9)]
  pred = [Entity("PERSON",     0, 9)]

  matches = EntityMatcher().match_entities(true, pred)
  # Match(true_entity=Entity("FIRST_NAME", 0, 4), pred_entity=Entity("PERSON", 0, 9))
  # Match(true_entity=Entity("LAST_NAME",  5, 9), pred_entity=Entity("PERSON", 0, 9))
  ```

- Several predicted entities can overlap the same true entity → each appears as a separate `Match` against that true entity.

  *Example:* ground truth has the full name as one span; the model predicts first and last name separately.
  ```python
  from spaneval.entity import Entity, EntityMatcher

  # text =         "Anna Berg"
  # position        0123456789
  true = [Entity("PERSON",     0, 9)]
  pred = [Entity("FIRST_NAME", 0, 4), Entity("LAST_NAME", 5, 9)]

  matches = EntityMatcher().match_entities(true, pred)
  # Match(true_entity=Entity("PERSON", 0, 9), pred_entity=Entity("FIRST_NAME", 0, 4))
  # Match(true_entity=Entity("PERSON", 0, 9), pred_entity=Entity("LAST_NAME",  5, 9))
  ```

**Unmatched true entities** (no overlapping prediction) produce `Match(true_entity=..., pred_entity=None)`.

**Unmatched predicted entities** (no overlapping true entity) produce `Match(true_entity=None, pred_entity=...)`.

Both slots being `None` is invalid and raises a `ValueError`.

## What strategies receive

All strategies receive the full list of `Match` objects. How null slots are interpreted — as missed, spurious, or otherwise — is up to the strategy.

**Entity-count strategies** (`Strict`, `Partial`, `ProportionalCoverage`, etc.) iterate over the matches grouped by true entity, scoring each true entity against its matched predictions. Matches with `pred_entity=None` count as missed; matches with `true_entity=None` count as spurious.

**`TextCoverage`** consumes the full match list directly and counts characters, not entities.
