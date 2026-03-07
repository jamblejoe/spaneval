# CLAUDE.md

## Architecture reference

See `docs/` for the full reference: [matching](docs/matching.md), [entity-count strategies](docs/entity_count.md), [character-count strategies](docs/character_count.md), [results and reporting](docs/results.md).

## Code style

- Do not add `from __future__ import annotations` unless actually needed (forward references, union syntax on older Python, etc.).

## Tests

- Test function names and inline comments must precisely describe what is being tested — the invariant or behaviour under verification.
- Do not reference session context, bug IDs, or implementation details of a fix in test names or comments.

## Branch hygiene

For cross-cutting changes that don't belong on the current feature branch: ask the user first, then stash → switch to `main` or a new branch → fix, test, commit → merge back and pop the stash.
