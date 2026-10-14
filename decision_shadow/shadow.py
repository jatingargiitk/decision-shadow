"""Run a decision model next to an existing classifier before trusting it.

Modes, per Shadow instance (default from $DECISION_SHADOW_MODE):
  off     the wrapped function runs alone; no backend calls.
  shadow  the wrapped function decides; the backend runs alongside, both logged.
  on      the backend decides; any failure, or `decide` returning None,
          falls back to the wrapped function.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, TypeVar, Union

from .backends import Backend, Decision, JevBackend

MODES = ("off", "shadow", "on")
F = TypeVar("F", bound=Callable[..., Any])


def _env_mode() -> str:
    m = (os.getenv("DECISION_SHADOW_MODE") or "off").strip().lower()
    return m if m in MODES else "off"


class Shadow:
    def __init__(
        self,
        backend: Optional[Backend] = None,
        log: Union[str, Path, None] = None,
        mode: Optional[str] = None,
        log_state: bool = False,
        workers: int = 4,
    ) -> None:
        if mode is not None and mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self._mode = mode
        self.backend = backend or JevBackend()
        self.log_path = Path(log or os.getenv("DECISION_SHADOW_LOG") or "decisions.jsonl")
        self.log_state = log_state
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="decision-shadow")
        self._lock = threading.Lock()
        self._pending: List[Future] = []

    @property
    def mode(self) -> str:
        return self._mode or _env_mode()

    def site(
        self,
        name: str,
        *,
        state: Callable[..., Any],
        questions: Mapping[str, Any],
        decide: Callable[[Decision], Any],
        same: Callable[[Any, Any], bool] = lambda a, b: a == b,
    ) -> Callable[[F], F]:
        """Wrap an existing classifier as one decision site.

        `state` builds the backend's state from the function's arguments.
        `decide` turns the backend's answers into a value shaped like the
        function's return value, or None to say "not confident, fall back".
        `same` compares the two for the agreement column.
        """

        def wrap(fn: F) -> F:
            @functools.wraps(fn)
            def inner(*args: Any, **kwargs: Any) -> Any:
                mode = self.mode
                if mode == "off":
                    return fn(*args, **kwargs)
                st = state(*args, **kwargs)
                if mode == "shadow":
                    fut = self._pool.submit(self._call, st, questions, decide)
                    t0 = time.monotonic()
                    legacy = fn(*args, **kwargs)
                    legacy_ms = int((time.monotonic() - t0) * 1000)
                    self._track(fut, lambda d, v: self._write(
                        name, mode, "legacy", st, d, v, legacy, legacy_ms, same))
                    return legacy
                d, value = self._call(st, questions, decide)
                if value is not None:
                    self._write(name, mode, "model", st, d, value, None, None, same)
                    return value
                t0 = time.monotonic()
                legacy = fn(*args, **kwargs)
                self._write(name, mode, "fallback", st, d, value, legacy,
                            int((time.monotonic() - t0) * 1000), same)
                return legacy

            return inner  # type: ignore[return-value]

        return wrap

    def flush(self, timeout: Optional[float] = None) -> None:
        """Wait for shadow calls still in flight to finish and be logged."""
        with self._lock:
            pending, self._pending = self._pending, []
        for f in pending:
            try:
                f.result(timeout=timeout)
            except Exception:
                pass

    def _call(self, st: Any, questions: Mapping[str, Any], decide: Callable[[Decision], Any]):
        d = self.backend.decide(st, questions)
        if not d.ok:
            return d, None
        try:
            return d, decide(d)
        except Exception as e:
            d.error = f"decide raised {type(e).__name__}: {e}"
            return d, None

    def _track(self, fut: Future, done: Callable[[Decision, Any], None]) -> None:
        logged: Future = Future()

        def cb(f: Future) -> None:
            try:
                d, v = f.result()
            except Exception as e:
                d, v = Decision(error=f"{type(e).__name__}: {e}"), None
            try:
                done(d, v)
            finally:
                logged.set_result(None)

        with self._lock:
            self._pending = [p for p in self._pending if not p.done()]
            self._pending.append(logged)
        fut.add_done_callback(cb)

    def _write(self, site: str, mode: str, used: str, st: Any, d: Decision, value: Any,
               legacy: Any, legacy_ms: Optional[int], same: Callable[[Any, Any], bool]) -> None:
        agree = None
        if value is not None and legacy is not None:
            try:
                agree = bool(same(legacy, value))
            except Exception:
                agree = None
        rec: Dict[str, Any] = {
            "id": uuid.uuid4().hex[:12],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "site": site,
            "mode": mode,
            "used": used,
            "state_sha": hashlib.sha256(json.dumps(st, sort_keys=True, default=str).encode()).hexdigest()[:16],
            "model_value": value,
            "legacy_value": legacy,
            "agree": agree,
            "answers": {k: a.to_dict() for k, a in d.answers.items()},
            "model": d.model,
            "model_ms": d.ms,
            "legacy_ms": legacy_ms,
            "cost": d.cost,
            "error": d.error,
        }
        if self.log_state:
            rec["state"] = st
        line = json.dumps(rec, default=str) + "\n"
        with self._lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a") as f:
                f.write(line)
