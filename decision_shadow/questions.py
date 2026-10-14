"""Builders for the three question types the Decisions API accepts."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence


def choice(instructions: str, criteria: Mapping[str, str]) -> Dict[str, Any]:
    """Pick one label. `criteria` maps each label to a plain-English description."""
    if len(criteria) < 2:
        raise ValueError("choice needs at least two labels")
    return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}


def noul(instructions: str, true: Optional[str] = None, false: Optional[str] = None) -> Dict[str, Any]:
    """Probability that a yes/no proposition holds."""
    q: Dict[str, Any] = {"type": "noul", "instructions": instructions}
    if true is not None or false is not None:
        q["criteria"] = {k: v for k, v in (("true", true), ("false", false)) if v is not None}
    return q


def score(instructions: str, levels: Sequence[str]) -> Dict[str, Any]:
    """Ordinal scale, least to most. The API caps it at 10 levels."""
    if not 2 <= len(levels) <= 10:
        raise ValueError("score needs between 2 and 10 levels")
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}
