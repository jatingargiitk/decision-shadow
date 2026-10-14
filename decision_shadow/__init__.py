"""Shadow-test a decision model against the classifier it would replace."""

from .backends import Answer, Backend, Decision, JevBackend
from .questions import choice, noul, score
from .shadow import Shadow

__all__ = ["Answer", "Backend", "Decision", "JevBackend", "Shadow", "choice", "noul", "score"]
__version__ = "0.1.0"
