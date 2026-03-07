# %%
from spaneval import evaluate, Goal
from spaneval.toy_data import toy_true, toy_pred
from spaneval.strategies import Strict, ProportionalCoverage

results = evaluate(toy_true, toy_pred)

# =============================================================================
# Step 1 — Define goals
# =============================================================================
# %%
# You know which strategies to use (see examples/quickstart.py).
# The next question is: what is good enough?
# Express this as per-type precision and recall targets using Goal.
# Both targets are required — 99% recall at 0% precision is not useful anonymization.

goals = {
    "PERSON": Goal(strategy=Strict(),              recall=0.90, precision=0.80),
    "DATE":   Goal(strategy=ProportionalCoverage(), recall=0.80, precision=0.70),
    "ORG":    Goal(strategy=ProportionalCoverage(), recall=0.85, precision=0.70),
}

results.report_goals(goals)

# Scores are uncapped: > 1.0 means above target. The ← marks the bottleneck —
# the weakest type/metric combination. That is what you need to fix first.

# =============================================================================
# Step 2 — Use score() for automated prompt engineering
# =============================================================================
# %%
# score() returns the bottleneck float without printing — clean for use inside a loop.

score = results.score(goals)
print(f"score = {score:.2f}  (1.0 = all targets exactly met; > 1.0 = all exceeded)")

# %%
# Feed this into a prompt optimisation loop.
# Replace candidate_prompts, run_llm, and documents with your own:

best_score = 0.0
for prompt in candidate_prompts:
    predictions = run_llm(prompt, documents)
    r = evaluate(toy_true, predictions)
    score = r.score(goals)
    if score > best_score:
        best_score = score
        best_prompt = prompt

# The bottleneck property ensures the optimizer cannot improve the score by
# sacrificing one entity type for another — it always has to fix the weakest link.
