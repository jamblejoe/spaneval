from collections.abc import Iterable

from .entity import Entity, EntityMatcher
from .results import Results


def evaluate(
    true_entities: list[Entity] | list[list[Entity]],
    pred_entities: list[Entity] | list[list[Entity]],
    ignore_types: Iterable[str] | None = None,
    ignore_all_types_but: Iterable[str] | None = None,
    warn_on_overlapping_preds: bool = True,
) -> Results:
    """Evaluate span extraction performance on a single or multiple documents.

    Accepts either a flat list of entities (single document) or a list of
    lists (multiple documents). Single documents are automatically wrapped.

    Args:
        true_entities: Flat list of entities, or list of per-document entity lists.
        pred_entities: Flat list of entities, or list of per-document entity lists.
        ignore_types: Entity types to exclude from evaluation.
        ignore_all_types_but: If set, evaluate only these entity types.
        warn_on_overlapping_preds: Emit a warning when overlapping predicted
            entities are detected and resolved. Default True.

    Returns:
        Results: Evaluation results.
    """
    if ignore_all_types_but is not None and ignore_types is not None:
        raise ValueError("Cannot specify both ignore_all_types_but and ignore_types. Choose one.")

    if true_entities and isinstance(true_entities[0], list):
        doc_true: list[list[Entity]] = true_entities  # type: ignore[assignment]
        doc_pred: list[list[Entity]] = pred_entities  # type: ignore[assignment]
    else:
        doc_true = [true_entities]  # type: ignore[list-item]
        doc_pred = [pred_entities]  # type: ignore[list-item]

    if len(doc_true) != len(doc_pred):
        raise ValueError(
            f"Mismatch in document count: {len(doc_true)} true docs vs {len(doc_pred)} pred docs"
        )

    ignore_set: set[str] = set(ignore_types) if ignore_types is not None else set()
    keep_only: set[str] | None = set(ignore_all_types_but) if ignore_all_types_but is not None else None
    matcher = EntityMatcher(warn_on_overlapping_preds=warn_on_overlapping_preds)

    doc_matches = []
    for true_doc, pred_doc in zip(doc_true, doc_pred):
        if keep_only is not None:
            true_doc = [e for e in true_doc if e.entity_type in keep_only]
            pred_doc = [e for e in pred_doc if e.entity_type in keep_only]
        elif ignore_set:
            true_doc = [e for e in true_doc if e.entity_type not in ignore_set]
            pred_doc = [e for e in pred_doc if e.entity_type not in ignore_set]
        doc_matches.append(matcher.match_entities(true_doc, pred_doc))

    return Results(matches=doc_matches)
