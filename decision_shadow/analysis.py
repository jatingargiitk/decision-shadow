"""Read a decision log and answer: is the model ready to decide this site?"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from .backends import Answer

SATURATED = 0.99


def load(path: Union[str, Path]) -> List[Dict[str, Any]]:
    out = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


@dataclass
class SiteReport:
    site: str
    n: int = 0
    errors: int = 0
    compared: int = 0
    agreed: int = 0
    fallbacks: int = 0
    model_ms: List[int] = field(default_factory=list)
    legacy_ms: List[int] = field(default_factory=list)
    cost: float = 0.0
    answers: int = 0
    saturated: int = 0
    snapshots: Counter = field(default_factory=Counter)
    disagreements: Counter = field(default_factory=Counter)

    @property
    def agreement(self) -> Optional[float]:
        return self.agreed / self.compared if self.compared else None

    @property
    def saturation(self) -> Optional[float]:
        return self.saturated / self.answers if self.answers else None

    def warnings(self) -> List[str]:
        w = []
        if len(self.snapshots) > 1:
            w.append(f"model snapshot changed mid-log: {', '.join(sorted(self.snapshots))}"
                     " — thresholds tuned on one may not hold on the other")
        if self.saturation is not None and self.answers >= 10 and self.saturation > 0.9:
            w.append(f"{self.saturation:.0%} of answers are at or past {SATURATED} certainty"
                     " — probabilities this peaked can't separate easy from hard cases")
        if self.n and self.errors / self.n > 0.05:
            w.append(f"{self.errors / self.n:.0%} of model calls failed")
        if self.compared and self.compared < 30:
            w.append(f"only {self.compared} compared decisions — too few to trust the agreement rate")
        return w


def report(records: Iterable[Mapping[str, Any]]) -> Dict[str, SiteReport]:
    sites: Dict[str, SiteReport] = {}
    for r in records:
        s = sites.setdefault(r.get("site") or "?", SiteReport(site=r.get("site") or "?"))
        s.n += 1
        if r.get("error"):
            s.errors += 1
        if r.get("used") == "fallback":
            s.fallbacks += 1
        if r.get("agree") is not None:
            s.compared += 1
            if r["agree"]:
                s.agreed += 1
            else:
                s.disagreements[(_key(r.get("legacy_value")), _key(r.get("model_value")))] += 1
        if not r.get("error") and r.get("model_ms") is not None:
            s.model_ms.append(int(r["model_ms"]))
        if r.get("legacy_ms") is not None:
            s.legacy_ms.append(int(r["legacy_ms"]))
        s.cost += float(r.get("cost") or 0.0)
        if r.get("model"):
            s.snapshots[r["model"]] += 1
        for raw in (r.get("answers") or {}).values():
            c = Answer.from_wire(raw).certainty
            if c is not None:
                s.answers += 1
                if c >= SATURATED:
                    s.saturated += 1
    return sites


@dataclass
class Cut:
    threshold: float
    coverage: float
    accuracy: Optional[float]
    n: int


def tune(
    records: Iterable[Mapping[str, Any]],
    site: str,
    question: str,
    labels: Optional[Mapping[str, Any]] = None,
    thresholds: Iterable[float] = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99),
) -> Tuple[List[Cut], int, str]:
    """Coverage and accuracy of the model's decision above each certainty threshold.

    Accuracy is measured against `labels` (record id -> correct value) when
    given, otherwise against the legacy classifier's answer.
    """
    rows: List[Tuple[float, bool]] = []
    for r in records:
        if r.get("site") != site or r.get("error"):
            continue
        raw = (r.get("answers") or {}).get(question)
        if not raw:
            continue
        cert = Answer.from_wire(raw).certainty
        if cert is None:
            continue
        if labels is not None:
            if r.get("id") not in labels:
                continue
            truth = labels[r["id"]]
        else:
            truth = r.get("legacy_value")
            if truth is None:
                continue
        rows.append((cert, _key(r.get("model_value")) == _key(truth)))

    cuts = []
    for t in thresholds:
        above = [ok for c, ok in rows if c >= t]
        cuts.append(Cut(
            threshold=t,
            coverage=len(above) / len(rows) if rows else 0.0,
            accuracy=sum(above) / len(above) if above else None,
            n=len(above),
        ))
    return cuts, len(rows), "labels" if labels is not None else "legacy classifier"


def recommend(cuts: List[Cut], target: float, min_n: int = 20) -> Optional[Cut]:
    """Lowest threshold whose accuracy meets `target` on at least `min_n` decisions."""
    for c in cuts:
        if c.accuracy is not None and c.n >= min_n and c.accuracy >= target:
            return c
    return None


def percentile(xs: List[int], p: float) -> Optional[int]:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]


def _key(v: Any) -> str:
    return json.dumps(v, sort_keys=True, default=str)
