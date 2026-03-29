import pytest

from spaneval.entity import Entity, to_entities, to_documents


def test_entity_start_greater_than_end():
    with pytest.raises(
        ValueError, match="Start offset 10 cannot be greater than end offset 5"
    ):
        Entity(entity_type="PERSON", start=10, end=5)


def test_entity_start_negative():
    with pytest.raises(ValueError, match="Start offset cannot be negative: -1"):
        Entity(entity_type="LOCATION", start=-1, end=5)


def test_entity_length():
    entity = Entity(entity_type="ORGANIZATION", start=5, end=15)
    assert entity.length == 10


def test_entity_span():
    entity = Entity(entity_type="PRODUCT", start=3, end=8)
    expected_span = range(3, 8)
    assert entity.span == expected_span


def test_entity_span_with_same_start_and_end():
    entity = Entity(entity_type="TIME", start=5, end=5)
    expected_span = range(5, 5)
    assert entity.span == expected_span


def test_entity_from_dict_valid_keys():
    entity_dict = {
        "entity_type": "EVENT",
        "start": 10,
        "end": 20,
        "original_text": "conference",
        "replacement_text": "meeting",
    }
    entity = Entity.from_dict(entity_dict)
    assert entity.entity_type == "EVENT"
    assert entity.start == 10
    assert entity.end == 20
    assert entity.original_text == "conference"
    assert entity.replacement_text == "meeting"


def test_entity_from_dict_missing_keys():
    entity_dict = {
        "entity_type": "EVENT",
        "start": 10,
        # "end" key is missing
    }
    with pytest.raises(ValueError, match="Missing required keys: {'end'}"):
        Entity.from_dict(entity_dict)


def test_entity_to_dict():
    entity = Entity(
        entity_type="ANIMAL",
        start=5,
        end=15,
        original_text="cat",
        replacement_text="dog",
    )
    expected_dict = {
        "entity_type": "ANIMAL",
        "start": 5,
        "end": 15,
        "original_text": "cat",
        "replacement_text": "dog",
    }
    assert entity.to_dict() == expected_dict


def test_entity_to_dict_with_none_values():
    entity = Entity(
        entity_type="ANIMAL", start=5, end=15, original_text=None, replacement_text=None
    )
    expected_dict = {"entity_type": "ANIMAL", "start": 5, "end": 15}
    assert entity.to_dict() == expected_dict


def test_entity_from_dict_extra_keys_are_ignored():
    # from_dict must silently ignore keys beyond the recognised fields.
    entity_dict = {
        "entity_type": "PERSON",
        "start": 10,
        "end": 20,
        "original_text": "John Doe",
        "replacement_text": "Anonymous",
        "extra_key": "extra_value",
    }
    entity = Entity.from_dict(entity_dict)
    assert entity.entity_type == "PERSON"
    assert entity.start == 10
    assert entity.end == 20
    assert entity.original_text == "John Doe"
    assert entity.replacement_text == "Anonymous"


def test_entity_from_dict_missing_optional_keys_default_to_none():
    # from_dict must set optional fields to None when absent.
    entity_dict = {
        "entity_type": "PERSON",
        "start": 10,
        "end": 20,
    }
    entity = Entity.from_dict(entity_dict)
    assert entity.entity_type == "PERSON"
    assert entity.start == 10
    assert entity.end == 20
    assert entity.original_text is None
    assert entity.replacement_text is None


def test_to_entities_field_names_remaps_all_required_fields():
    # field_names remaps source keys to spaneval's canonical names before parsing.
    dicts = [{"label_": "PERSON", "start_char": 0, "end_char": 10}]
    entities = to_entities(dicts, field_names={"label_": "entity_type", "start_char": "start", "end_char": "end"})
    assert len(entities) == 1
    assert entities[0].entity_type == "PERSON"
    assert entities[0].start == 0
    assert entities[0].end == 10


def test_to_entities_field_names_partial_remap():
    # Only non-standard fields need to be listed; standard fields pass through unchanged.
    dicts = [{"label": "ORG", "start": 5, "end": 15}]
    entities = to_entities(dicts, field_names={"label": "entity_type"})
    assert entities[0].entity_type == "ORG"
    assert entities[0].start == 5
    assert entities[0].end == 15


def test_to_entities_without_field_names_unchanged():
    # Omitting field_names preserves existing behaviour.
    dicts = [{"entity_type": "LOC", "start": 0, "end": 4}]
    entities = to_entities(dicts)
    assert entities[0].entity_type == "LOC"


def test_to_documents_field_names_remaps_across_documents():
    # field_names applies uniformly across all documents.
    dicts = [
        [{"label_": "PERSON", "start_char": 0, "end_char": 4}],
        [{"label_": "ORG", "start_char": 10, "end_char": 16}],
    ]
    documents = to_documents(dicts, field_names={"label_": "entity_type", "start_char": "start", "end_char": "end"})
    assert documents[0][0].entity_type == "PERSON"
    assert documents[1][0].entity_type == "ORG"
