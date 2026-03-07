import warnings

import pytest

from spaneval.entity import Entity, EntityMatcher


def get_match_key(match):
    return match.true_entity.start if match.true_entity else match.pred_entity.start


def test_match_entities_spurious_predicted_entity():
    true_entities = [
        Entity(entity_type="PERSON", start=0, end=5),
        Entity(entity_type="LOCATION", start=10, end=20),
    ]
    pred_entities = [
        Entity(entity_type="ORGANIZATION", start=30, end=40)  # Spurious entity
    ]
    matcher = EntityMatcher()
    matches = matcher.match_entities(true_entities, pred_entities)
    matches = sorted(matches, key=get_match_key)

    assert len(matches) == 3
    assert matches[2].pred_entity == pred_entities[0]
    assert matches[2].true_entity is None


def test_match_entities_missed_true_entity():
    true_entities = [
        Entity(entity_type="PERSON", start=0, end=5),
        Entity(entity_type="LOCATION", start=10, end=20),  # This should be missed
    ]
    pred_entities = [Entity(entity_type="PERSON", start=0, end=5)]
    matcher = EntityMatcher()
    matches = matcher.match_entities(true_entities, pred_entities)
    matches = sorted(matches, key=get_match_key)

    assert len(matches) == 2
    assert matches[1].true_entity == true_entities[1]
    assert matches[1].pred_entity is None


def test_match_entities_with_empty_lists():
    true_entities = []
    pred_entities = []
    matcher = EntityMatcher()
    matches = matcher.match_entities(true_entities, pred_entities)

    assert matches == []


def test_match_entities_multiple_exact_matches():
    true_entities = [
        Entity(entity_type="PERSON", start=0, end=5),
        Entity(entity_type="LOCATION", start=10, end=20),
        Entity(entity_type="ORGANIZATION", start=25, end=30),
    ]
    pred_entities = [
        Entity(entity_type="PERSON", start=0, end=5),
        Entity(entity_type="LOCATION", start=10, end=20),
        Entity(entity_type="ORGANIZATION", start=25, end=30),
    ]
    matcher = EntityMatcher()
    matches = matcher.match_entities(true_entities, pred_entities)
    matches = sorted(matches, key=get_match_key)

    assert len(matches) == 3
    assert matches[0].true_entity == true_entities[0]
    assert matches[0].pred_entity == pred_entities[0]
    assert matches[1].true_entity == true_entities[1]
    assert matches[1].pred_entity == pred_entities[1]
    assert matches[2].true_entity == true_entities[2]
    assert matches[2].pred_entity == pred_entities[2]


def test_match_entities_multiple_partial_matches_different_types():
    true_entities = [
        Entity(entity_type="PERSON", start=0, end=10),
        Entity(entity_type="LOCATION", start=15, end=25),
        Entity(entity_type="ORGANIZATION", start=30, end=40),
    ]
    pred_entities = [
        Entity(entity_type="PERSON", start=5, end=15),  # Partial match with PERSON
        Entity(entity_type="LOCATION", start=20, end=30),  # Partial match with LOCATION
        Entity(
            entity_type="ORGANIZATION", start=35, end=45
        ),  # Partial match with ORGANIZATION
    ]
    matcher = EntityMatcher()
    matches = matcher.match_entities(true_entities, pred_entities)
    matches = sorted(matches, key=get_match_key)

    assert len(matches) == 3
    assert matches[0].true_entity == true_entities[0]
    assert matches[0].pred_entity == pred_entities[0]
    assert matches[1].true_entity == true_entities[1]
    assert matches[1].pred_entity == pred_entities[1]
    assert matches[2].true_entity == true_entities[2]
    assert matches[2].pred_entity == pred_entities[2]


def test_match_entities_multiple_pred_same_true():
    true_entities = [
        Entity(entity_type="PERSON", start=0, end=10),
    ]
    pred_entities = [
        Entity(entity_type="PERSON", start=0, end=5),
        Entity(entity_type="LOCATION", start=7, end=10),
    ]

    matcher = EntityMatcher()
    matches = matcher.match_entities(true_entities, pred_entities)
    matches = sorted(matches, key=get_match_key)

    assert len(matches) == 2
    assert matches[0].true_entity == true_entities[0]
    assert matches[0].pred_entity == pred_entities[0]
    assert matches[1].true_entity == true_entities[0]
    assert matches[1].pred_entity == pred_entities[1]


def test_match_entities_multiple_true_same_pred():
    true_entities = [
        Entity(entity_type="PERSON", start=0, end=5),
        Entity(entity_type="LOCATION", start=7, end=10),
    ]
    pred_entities = [
        Entity(entity_type="PERSON", start=0, end=10),
    ]

    matcher = EntityMatcher()
    matches = matcher.match_entities(true_entities, pred_entities)
    matches = sorted(matches, key=get_match_key)

    assert len(matches) == 2
    assert matches[0].true_entity == true_entities[0]
    assert matches[0].pred_entity == pred_entities[0]
    assert matches[1].true_entity == true_entities[1]
    assert matches[1].pred_entity == pred_entities[0]


def test_match_entities_comprehensive_smoke_test():
    true_entities = [
        Entity(entity_type="PERSON", start=0, end=10),  # Overlaps with two pred_entities
        Entity(entity_type="LOCATION", start=15, end=25),
        Entity(entity_type="ORGANIZATION", start=30, end=35),
        Entity(entity_type="MISC", start=40, end=50),
        Entity(entity_type="PERSON", start=55, end=60),  # Missed
        Entity(entity_type="LOCATION", start=65, end=75),
        Entity(entity_type="ORGANIZATION", start=80, end=90),  # Overlaps with two pred_entities
        Entity(entity_type="MISC", start=95, end=100),  # Missed
        Entity(entity_type="PERSON", start=105, end=110),
        Entity(entity_type="LOCATION", start=115, end=120),
    ]
    pred_entities = [
        Entity(entity_type="PERSON", start=0, end=5),  # Overlaps with one true_entity
        Entity(entity_type="PERSON", start=6, end=10),  # Overlaps with one true_entity
        Entity(entity_type="LOCATION", start=18, end=22),
        Entity(entity_type="ORGANIZATION", start=30, end=35),  # Exact match
        Entity(entity_type="MISC", start=42, end=48),
        Entity(entity_type="PERSON", start=60, end=65),  # Spurious
        Entity(entity_type="LOCATION", start=68, end=72),
        Entity(entity_type="ORGANIZATION", start=80, end=85),  # Overlaps with one true_entity
        Entity(entity_type="ORGANIZATION", start=86, end=90),  # Overlaps with one true_entity
        Entity(entity_type="MISC", start=100, end=105),  # Spurious
        Entity(entity_type="PERSON", start=105, end=110),  # Exact match
        Entity(entity_type="LOCATION", start=115, end=120),  # Exact match
    ]

    matcher = EntityMatcher()
    matches = matcher.match_entities(true_entities, pred_entities)

    true_positive_matches = [m for m in matches if m.true_entity is not None and m.pred_entity is not None]
    spurious_matches = [m for m in matches if m.true_entity is None]
    missed_matches = [m for m in matches if m.pred_entity is None]

    assert len(true_positive_matches) == 10
    assert len(spurious_matches) == 2
    assert len(missed_matches) == 2

    # Check for one true entity matching multiple predicted entities
    true_0_10 = [m for m in true_positive_matches if m.true_entity.start == 0 and m.true_entity.end == 10]
    assert len(true_0_10) == 2
    assert true_0_10[0].pred_entity.start == 0 and true_0_10[0].pred_entity.end == 5
    assert true_0_10[1].pred_entity.start == 6 and true_0_10[1].pred_entity.end == 10

    # Check for multiple true entities matching one predicted entity
    pred_80_90 = [m for m in true_positive_matches if m.pred_entity.start == 80 or m.pred_entity.start == 86]
    assert len(pred_80_90) == 2
    assert pred_80_90[0].true_entity.start == 80 and pred_80_90[0].true_entity.end == 90
    assert pred_80_90[1].true_entity.start == 80 and pred_80_90[1].true_entity.end == 90

    assert spurious_matches[0].pred_entity.start == 60
    assert spurious_matches[1].pred_entity.start == 100
    assert missed_matches[0].true_entity.start == 55
    assert missed_matches[1].true_entity.start == 95


# ---------------------------------------------------------------------------
# Overlapping predicted entity resolution
# ---------------------------------------------------------------------------

def test_overlapping_preds_longest_span_is_kept():
    # Two overlapping preds: [0,15] is longer than [0,10]; longest must be kept
    true_entities = [Entity(entity_type="PERSON", start=0, end=15)]
    pred_entities = [
        Entity(entity_type="PERSON", start=0, end=10),
        Entity(entity_type="PERSON", start=0, end=15),
    ]
    matcher = EntityMatcher(warn_on_overlapping_preds=False)
    matches = matcher.match_entities(true_entities, pred_entities)

    pred_entities_in_matches = [m.pred_entity for m in matches if m.pred_entity is not None]
    assert len(pred_entities_in_matches) == 1
    assert pred_entities_in_matches[0].end == 15


def test_overlapping_preds_emits_warning_by_default():
    pred_entities = [
        Entity(entity_type="PERSON", start=0, end=10),
        Entity(entity_type="PERSON", start=5, end=15),
    ]
    matcher = EntityMatcher()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        matcher.match_entities([], pred_entities)

    assert len(caught) == 1
    assert issubclass(caught[0].category, UserWarning)
    assert "Overlapping predicted entities resolved" in str(caught[0].message)


def test_overlapping_preds_warning_names_kept_and_dropped_entities():
    # Warning message must identify which entity is kept and which are dropped
    pred_entities = [
        Entity(entity_type="ORG", start=0, end=5),
        Entity(entity_type="PERSON", start=0, end=20),
    ]
    matcher = EntityMatcher()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        matcher.match_entities([], pred_entities)

    msg = str(caught[0].message)
    assert "PERSON[0:20]" in msg   # kept (longest, 20 chars)
    assert "ORG[0:5]" in msg       # dropped


def test_overlapping_preds_warning_suppressed_when_disabled():
    pred_entities = [
        Entity(entity_type="PERSON", start=0, end=10),
        Entity(entity_type="PERSON", start=5, end=15),
    ]
    matcher = EntityMatcher(warn_on_overlapping_preds=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        matcher.match_entities([], pred_entities)

    assert len(caught) == 0


def test_overlapping_preds_three_way_group_longest_kept():
    # Three mutually overlapping preds; the longest [0:30] must be kept
    pred_entities = [
        Entity(entity_type="A", start=0, end=10),
        Entity(entity_type="B", start=5, end=20),
        Entity(entity_type="C", start=10, end=30),
    ]
    matcher = EntityMatcher(warn_on_overlapping_preds=False)
    resolved = matcher._sort_and_resolve(pred_entities)

    assert len(resolved) == 1
    assert resolved[0].entity_type == "C"
    assert resolved[0].end == 30


def test_overlapping_preds_exact_duplicate_one_kept():
    # Exact same span predicted twice; one is kept, no crash
    pred_entities = [
        Entity(entity_type="PERSON", start=0, end=10),
        Entity(entity_type="PERSON", start=0, end=10),
    ]
    matcher = EntityMatcher(warn_on_overlapping_preds=False)
    resolved = matcher._sort_and_resolve(pred_entities)

    assert len(resolved) == 1


def test_overlapping_preds_non_overlapping_groups_resolved_independently():
    # Two separate overlap groups: [0,10]/[5,15] and [30,50]/[40,60]
    pred_entities = [
        Entity(entity_type="A", start=0, end=10),
        Entity(entity_type="B", start=5, end=15),
        Entity(entity_type="C", start=30, end=50),
        Entity(entity_type="D", start=40, end=60),
    ]
    matcher = EntityMatcher(warn_on_overlapping_preds=False)
    resolved = matcher._sort_and_resolve(pred_entities)

    # One winner per group
    assert len(resolved) == 2
    # Both groups have tied lengths; max() returns the first maximum, so first entity wins
    assert resolved[0].entity_type == "A"   # [0:10], 10 chars == B[5:15], first wins tie
    assert resolved[1].entity_type == "C"   # [30:50], 20 chars == D[40:60], first wins tie


def test_true_entity_overlap_still_raises():
    # Overlapping true entities are always a data error; must still raise
    true_entities = [
        Entity(entity_type="PERSON", start=0, end=10),
        Entity(entity_type="PERSON", start=5, end=15),
    ]
    matcher = EntityMatcher()
    with pytest.raises(ValueError, match="Overlapping entities detected"):
        matcher.match_entities(true_entities, [])
