from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Entity:
    """Immutable entity representation with enhanced validation."""

    entity_type: str
    start: int
    end: int
    original_text: str | None = None
    replacement_text: str | None = None

    def __post_init__(self):
        """Validate entity properties."""
        if self.start > self.end:
            raise ValueError(
                f"Start offset {self.start} cannot be greater than end offset {self.end}"
            )
        if self.start < 0:
            raise ValueError(f"Start offset cannot be negative: {self.start}")
        if not self.entity_type.strip():
            raise ValueError("Entity type cannot be empty")

    @property
    def span(self) -> range:
        """Get the span as a range object."""
        return range(self.start, self.end)

    @property
    def length(self) -> int:
        """Get the length of the entity span."""
        return self.end - self.start

    @classmethod
    def from_dict(cls, entity_dict: dict[str, str | int]) -> Entity:
        """Create Entity from dictionary with validation."""
        required_keys = {"entity_type", "start", "end"}
        missing_keys = required_keys - set(entity_dict.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys: {missing_keys}")

        original_text = (
            str(entity_dict.get("original_text"))
            if entity_dict.get("original_text")
            else None
        )
        replacement_text = (
            str(entity_dict.get("replacement_text"))
            if entity_dict.get("replacement_text")
            else None
        )

        return cls(
            entity_type=str(entity_dict["entity_type"]).strip(),
            start=int(entity_dict["start"]),
            end=int(entity_dict["end"]),
            original_text=original_text,
            replacement_text=replacement_text,
        )

    def to_dict(self) -> dict[str, str | int]:
        """Convert the Entity instance to a dictionary."""
        entity_dict = {
            "entity_type": self.entity_type,
            "start": self.start,
            "end": self.end,
        }

        # Add original_text only if it's not None
        if self.original_text is not None:
            entity_dict["original_text"] = self.original_text

        # Add replacement_text only if it's not None
        if self.replacement_text is not None:
            entity_dict["replacement_text"] = self.replacement_text

        return entity_dict

    def overlaps_with(self, other: Entity) -> bool:
        """True if the two half-open intervals share at least one character.

        Example: [0,4) and [3,7) overlap; [0,4) and [4,8) do not.
        """
        return not (self.end <= other.start or other.end <= self.start)



@dataclass(frozen=True)
class Match:
    """Match between a predicted and true entity."""

    true_entity: Entity | None = None
    pred_entity: Entity | None = None

    def __post_init__(self):
        """
        Validate the initialization of a Match object.

        Ensures that at least one of the true_entity or pred_entity is provided
        during the creation of a Match instance.

        Raises:
            ValueError: If both true_entity and pred_entity are None.
        """
        if self.true_entity is None and self.pred_entity is None:
            raise ValueError("Either true_entity or pred_entity must be provided")


class EntityMatcher:
    """Handles entity matching logic."""

    def __init__(self, warn_on_overlapping_preds: bool = True):
        self.warn_on_overlapping_preds = warn_on_overlapping_preds

    def _sort_and_validate(self, entities: list[Entity]) -> list[Entity]:
        """Sort and validate true entities. Raises ValueError on overlap."""
        if len(entities) == 0:
            return entities

        entities = sorted(entities, key=lambda e: e.start)

        last_end = entities[0].end
        for i in range(1, len(entities)):
            e = entities[i]
            if e.start < last_end:
                raise ValueError(
                    f"Overlapping entities detected: {entities[i - 1]} and {e}"
                )
            else:
                last_end = e.end

        return entities

    def _sort_and_resolve(self, entities: list[Entity]) -> list[Entity]:
        """Sort predicted entities and resolve overlaps by keeping the longest span.

        When a group of predicted entities overlap each other, the longest span is
        kept and the rest are discarded. A warning is emitted for each resolved
        group unless warn_on_overlapping_preds=False.
        """
        if len(entities) == 0:
            return entities

        entities = sorted(entities, key=lambda e: e.start)
        result = []
        i = 0
        while i < len(entities):
            # Collect all entities in this overlap group
            group = [entities[i]]
            group_end = entities[i].end
            j = i + 1
            while j < len(entities) and entities[j].start < group_end:
                group.append(entities[j])
                group_end = max(group_end, entities[j].end)
                j += 1

            if len(group) > 1:
                kept = max(group, key=lambda e: e.length)
                dropped = [e for e in group if e is not kept]
                if self.warn_on_overlapping_preds:
                    dropped_str = ", ".join(
                        f"{e.entity_type}[{e.start}:{e.end}] ({e.length} chars)"
                        for e in dropped
                    )
                    warnings.warn(
                        f"Overlapping predicted entities resolved: keeping "
                        f"{kept.entity_type}[{kept.start}:{kept.end}] ({kept.length} chars, longest) "
                        f"over: {dropped_str}. "
                        f"Pass warn_on_overlapping_preds=False to suppress.",
                        UserWarning,
                        stacklevel=3,
                    )
                result.append(kept)
            else:
                result.append(group[0])

            i = j

        return result

    def match_entities(
        self,
        true_entities: list[Entity],
        pred_entities: list[Entity],
    ) -> list[Match]:
        """Matches predicted entities against true entities based on span overlap.

        True entities must be non-overlapping; a ValueError is raised if they are.
        Overlapping predicted entities are resolved automatically by keeping the
        longest span in each overlapping group (with a warning by default).

        Args:
            true_entities: List of true entities.
            pred_entities: List of predicted entities.

        Returns:
            A list of `Match` objects representing the comparison. This includes
            matches for overlapping entities, spurious predictions (no overlap),
            and missed true entities.
        """
        true_entities = self._sort_and_validate(true_entities)
        pred_entities = self._sort_and_resolve(pred_entities)

        matches = []
        matched_true_entities = set()

        true_ptr = 0
        pred_ptr = 0
        while pred_ptr < len(pred_entities) and true_ptr < len(true_entities):
            pred_entity = pred_entities[pred_ptr]
            true_entity = true_entities[true_ptr]

            if pred_entity.end <= true_entity.start:
                # pred_entity is completely before true_entity, so it's a spurious match
                matches.append(Match(pred_entity=pred_entity))
                pred_ptr += 1
            elif true_entity.end <= pred_entity.start:
                # true_entity is completely before pred_entity, so advance true_ptr
                true_ptr += 1
            else:
                # overlap detected
                # find all overlapping true entities for the current predicted entity
                temp_true_ptr = true_ptr
                while temp_true_ptr < len(true_entities) and true_entities[temp_true_ptr].overlaps_with(pred_entity):
                    matched_true_entities.add(true_entities[temp_true_ptr])
                    matches.append(Match(true_entity=true_entities[temp_true_ptr], pred_entity=pred_entity))
                    temp_true_ptr += 1
                pred_ptr += 1

        # add remaining predicted entities as spurious
        while pred_ptr < len(pred_entities):
            matches.append(Match(pred_entity=pred_entities[pred_ptr]))
            pred_ptr += 1

        # handle missed entities
        for true_entity in true_entities:
            if true_entity not in matched_true_entities:
                matches.append(Match(true_entity=true_entity))

        return matches



def get_entity_types_from_matches(matches:Iterable[Match])->list[str]:
    entity_types: set[str] = set()
    for match in matches:
        if match.true_entity is not None:
            entity_types.add(match.true_entity.entity_type)
        if match.pred_entity is not None:
            entity_types.add(match.pred_entity.entity_type)

    return sorted(entity_types)


def _apply_field_names(d: dict, field_names: dict[str, str]) -> dict:
    """Remap dict keys according to field_names (source name → spaneval name)."""
    return {field_names.get(k, k): v for k, v in d.items()}


def to_entities(dicts: list[dict], field_names: dict[str, str] | None = None) -> list[Entity]:
    """Convert a list of dicts to a list of Entity objects. Validates at call site.

    Args:
        dicts: List of dicts with entity data.
        field_names: Optional mapping from source field names to spaneval field names.
            Only non-standard fields need to be listed. For example, to read spaCy
            output: ``field_names={"label_": "entity_type", "start_char": "start",
            "end_char": "end"}``.
    """
    if field_names:
        dicts = [_apply_field_names(d, field_names) for d in dicts]
    return [Entity.from_dict(d) for d in dicts]


def to_documents(dicts: list[list[dict]], field_names: dict[str, str] | None = None) -> list[list[Entity]]:
    """Convert a list of per-document dict lists to a list of per-document Entity lists.

    Args:
        dicts: List of per-document dict lists.
        field_names: Optional mapping from source field names to spaneval field names.
            Only non-standard fields need to be listed. See :func:`to_entities`.
    """
    if field_names:
        dicts = [[_apply_field_names(d, field_names) for d in doc] for doc in dicts]
    return [[Entity.from_dict(d) for d in doc] for doc in dicts]