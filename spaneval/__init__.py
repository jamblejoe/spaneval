import importlib.metadata

try:
    __version__ = importlib.metadata.version(__name__)
except Exception:
    __version__ = "unknown"

from .entity import Entity, to_entities, to_documents
from .evaluator import evaluate
from .results import Goal, Results
