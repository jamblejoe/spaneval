# %%
from spaneval import evaluate, to_entities
from spaneval.toy_data import toy_true, toy_pred
from spaneval.strategies import Strict, ProportionalCoverage

# =============================================================================
# Step 1 — First look, zero configuration
# =============================================================================
# %%
# You have a ground truth and a set of LLM predictions.
# Load the built-in toy dataset and call evaluate().

results = evaluate(toy_true, toy_pred)
results.report()

# The default bundle — Strict (ceiling) and AnyOverlap (floor) — gives an instant spread:
#   Strict     = exact span + correct type required
#   AnyOverlap = any non-zero overlap = full credit, type ignored
#
# The ± columns show how much your numbers depend on how strictly you measure.
# A wide range means span boundaries are frequently imprecise.
# A narrow range means predictions are either clearly right or clearly wrong.
#
# DATE has a massive ± 50 on precision — boundary precision is the problem there.
# PERSON is much tighter at ± 10.

# =============================================================================
# Step 2 — Representing your data
# =============================================================================
# %%
# toy_true and toy_pred are lists of documents; each document is a list of Entity
# objects. Entity has three required fields: entity_type, start, end (character
# offsets, half-open interval [start, end)).

toy_true[0]
# [
#   Entity(entity_type="PERSON", start= 0, end=10),   # "Anna Weber"
#   Entity(entity_type="ORG",    start=34, end=44),   # "Proxima AG"
#   Entity(entity_type="DATE",   start=48, end=61),   # "12 March 2023"
# ]

# %%
# If your pipeline produces plain dicts, convert them before evaluation using
# to_entities() (one document) or to_documents() (list of documents):

raw_doc = [
    {"entity_type": "PERSON", "start":  0, "end": 10},
    {"entity_type": "ORG",    "start": 34, "end": 44},
]
doc = to_entities(raw_doc)

# If your framework uses different field names (spaCy, HuggingFace, etc.), pass
# field_names to remap them — no pre-processing needed:

# spaCy:       to_entities(dicts, field_names={"label_": "entity_type", "start_char": "start", "end_char": "end"})
# HuggingFace: to_entities(dicts, field_names={"entity_group": "entity_type"})

# Only non-standard fields need to be listed; the rest pass through unchanged.

# For multi-document data use to_documents():
# from spaneval import to_documents
# doc_true = to_documents(raw_true)   # raw_true: list[list[dict]]
# doc_pred = to_documents(raw_pred)
# results  = evaluate(doc_true, doc_pred)

# Validation (start <= end, non-empty type, non-negative offsets) fires at the
# to_entities() / to_documents() call, not deep inside the evaluator.

# =============================================================================
# Step 3 — Understand what the spread means
# =============================================================================
# %%
# Notice DATE has a huge ± on precision and poor recall regardless of strategy.
# That tells you the model is finding DATE spans loosely — boundaries vary a lot.
# PERSON is tighter.
#
# Drill into which documents are causing problems.

# missed and spurious are strategy-independent: determined at matching time.
print("Documents with missed entities:     ", results.missed_docs)
print("Documents with spurious predictions:", results.spurious_docs)

# %%
# incorrect depends on the strategy: a span that overlaps but doesn't exactly match
# is incorrect under Strict, but gets partial credit under ProportionalCoverage.

print("Incorrect under Strict:              ", results.incorrect_docs(strategy=Strict()))
print("Incorrect under ProportionalCoverage: ", results.incorrect_docs(strategy=ProportionalCoverage()))
# incorrect_docs(Strict) is typically a superset of incorrect_docs(ProportionalCoverage)

# Inspect those documents to understand whether the problem is span boundaries,
# wrong types, or missed detections.

# =============================================================================
# Step 4 — Pin a strategy
# =============================================================================
# %%
# Once you understand the spread, pick the strategy that matches what "correct"
# means for your use case. For anonymization, a slightly off span boundary is
# usually acceptable — the text is still anonymized. ProportionalCoverage
# (fraction of true-entity characters covered, type ignored) is typically a
# good choice.

results.report(strategy=ProportionalCoverage())

# You now have exact numbers for one strategy. F1 gives you a single per-type summary.

# =============================================================================
# Step 5 — Assign strategies per entity type
# =============================================================================
# %%
# Different entity types may warrant different strictness. Names must be found
# precisely (wrong span could leave part of a name visible), but dates only need
# to be found at all.

results.report(
    strategy={"PERSON": Strict(), "DATE": ProportionalCoverage(), "ORG": ProportionalCoverage()},
)

# The Overall row is a composite: each type contributes its metrics under its own
# strategy. If any entity type in your data is not covered by the dict and you
# have not set default_strategy, you get an informative error telling you which
# types are missing.

# Continue in examples/goals.py to define targets and automate optimization.
