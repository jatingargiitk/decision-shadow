"""Decision backends. A backend takes (state, questions) and returns a Decision."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol


@dataclass
class Answer:
    type: str
    choice: Optional[str] = None
    score: Optional[float] = None
    noul: Optional[float] = None
    probabilities: Dict[str, float] = field(default_factory=dict)
    confidence: Optional[float] = None

    @property
    def value(self) -> Any:
        """The label for a choice, the weighted level for a score, P(yes) for a noul."""
        return {"choice": self.choice, "score": self.score, "noul": self.noul}.get(self.type)

    @property
    def certainty(self) -> Optional[float]:
        """One number in [0, 1] to threshold on, whatever the question type."""
        if self.type == "noul" and self.noul is not None:
            return max(self.noul, 1.0 - self.noul)
        if self.confidence is not None:
            return self.confidence
        return max(self.probabilities.values()) if self.probabilities else None

    @classmethod
    def from_wire(cls, raw: Mapping[str, Any]) -> "Answer":
        probs = raw.get("probabilities") or {}
        return cls(
            type=str(raw.get("type") or ""),
            choice=raw.get("choice"),
            score=_float(raw.get("score")),
            noul=_float(raw.get("noul")),
            probabilities={str(k): float(v) for k, v in probs.items()},
            confidence=_float(raw.get("confidence")),
        )

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"type": self.type}
        for k in ("choice", "score", "noul", "confidence"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        if self.probabilities:
            d["probabilities"] = self.probabilities
        return d


@dataclass
class Decision:
    answers: Dict[str, Answer] = field(default_factory=dict)
    model: Optional[str] = None
    cost: Optional[float] = None
    ms: int = 0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def __getitem__(self, key: str) -> Answer:
        return self.answers[key]


class Backend(Protocol):
    def decide(self, state: Any, questions: Mapping[str, Any]) -> Decision: ...


OPENROUTER_URL = "https://openrouter.ai/api/alpha/decisions"
TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"


class JevBackend:
    """TypeSafe Jev over OpenRouter's Decisions API, or TypeSafe's own endpoint.

    Never raises: failures come back as a Decision with `error` set, so the
    caller can fall back to its existing classifier.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        url: str = OPENROUTER_URL,
        timeout: float = 8.0,
        retries: int = 1,
        headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        typesafe = url.startswith(TYPESAFE_URL)
        self.model = model or os.getenv("JEV_MODEL") or ("jev-latest" if typesafe else "~typesafe/jev-latest")
        self.api_key = api_key or os.getenv("TYPESAFE_API_KEY" if typesafe else "OPENROUTER_API_KEY")
        self.url = url
        self.timeout = timeout
        self.retries = retries
        self.headers = dict(headers or {})

    def decide(self, state: Any, questions: Mapping[str, Any]) -> Decision:
        t0 = time.monotonic()
        if not self.api_key:
            return Decision(error="no API key", ms=0)
        body = json.dumps({"model": self.model, "state": state, "questions": dict(questions)}, default=str)
        error = "no attempt"
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(
                self.url,
                data=body.encode(),
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    **self.headers,
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read())
                return _parse(data, int((time.monotonic() - t0) * 1000))
            except urllib.error.HTTPError as e:
                detail = e.read()[:200].decode(errors="replace")
                error = f"{e.code}: {detail}"
                if e.code != 429 and e.code < 500:
                    break
                if attempt < self.retries:
                    time.sleep(_retry_after(e.headers.get("Retry-After"), attempt))
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                if attempt < self.retries:
                    time.sleep(_retry_after(None, attempt))
        return Decision(error=error, ms=int((time.monotonic() - t0) * 1000))


def _parse(data: Mapping[str, Any], ms: int) -> Decision:
    raw = data.get("answers")
    if not isinstance(raw, dict):
        return Decision(error="response had no answers", ms=ms)
    return Decision(
        answers={k: Answer.from_wire(v) for k, v in raw.items() if isinstance(v, dict)},
        model=data.get("model"),
        cost=_float((data.get("usage") or {}).get("cost")),
        ms=ms,
    )


def _retry_after(header: Optional[str], attempt: int) -> float:
    try:
        return min(float(header), 10.0) if header else 0.5 * (2 ** attempt)
    except ValueError:
        return 0.5 * (2 ** attempt)


def _float(v: Any) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None
