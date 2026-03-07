"""Toy dataset for exploring the user journey without real data.

5 short documents with ground-truth and intentionally imperfect predicted entities.
Designed to produce a visible spread across Strict / ProportionalOverlap / AnyOverlap:

  PERSON — mostly found, but one miss, one partial boundary, one spurious prediction
  DATE   — frequently missed or with imprecise boundaries → wide ±
  ORG    — well-found, one spurious prediction
"""

from .entity import Entity


def _e(text: str, substr: str, entity_type: str) -> Entity:
    start = text.index(substr)
    return Entity(entity_type=entity_type, start=start, end=start + len(substr))


_T0 = "Anna Weber filed a complaint with Proxima AG on 12 March 2023."
_T1 = "The policy holder Klaus Richter was injured on 3 April 2019."
_T2 = "Sarah Klein submitted a claim on 15 February 2022."
_T3 = "Tom Bauer called the helpdesk of Max Schulz."
_T4 = "Contract with Muster GmbH, signed by Lena Braun, expires soon."

toy_true = [
    [
        _e(_T0, "Anna Weber",    "PERSON"),
        _e(_T0, "Proxima AG",    "ORG"),
        _e(_T0, "12 March 2023", "DATE"),
    ],
    [
        _e(_T1, "Klaus Richter", "PERSON"),
        _e(_T1, "3 April 2019",  "DATE"),
    ],
    [
        _e(_T2, "Sarah Klein",      "PERSON"),
        _e(_T2, "15 February 2022", "DATE"),
    ],
    [
        _e(_T3, "Tom Bauer", "PERSON"),
    ],
    [
        _e(_T4, "Muster GmbH", "ORG"),
        _e(_T4, "Lena Braun",  "PERSON"),
    ],
]

toy_pred = [
    [
        _e(_T0, "Anna Weber", "PERSON"),  # exact
        _e(_T0, "Proxima AG", "ORG"),     # exact
        _e(_T0, "March 2023", "DATE"),    # partial: leading "12 " missing
    ],
    [
        _e(_T1, "Richter", "PERSON"),     # partial: first name missing
        # "3 April 2019" not predicted → missed
    ],
    [
        _e(_T2, "February 2022", "DATE"), # partial: leading "15 " missing
        # "Sarah Klein" not predicted → missed
    ],
    [
        _e(_T3, "Tom Bauer",  "PERSON"),  # exact
        _e(_T3, "Max Schulz", "PERSON"),  # spurious: annotator did not mark this name
        _e(_T3, "helpdesk",   "ORG"),     # spurious: not a ground-truth entity
    ],
    [
        _e(_T4, "Muster GmbH", "ORG"),    # exact
        _e(_T4, "Lena Braun",  "PERSON"), # exact
    ],
]
