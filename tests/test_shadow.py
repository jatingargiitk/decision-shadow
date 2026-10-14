import io
import json
import time
import unittest
import urllib.error
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from decision_shadow import Answer, Decision, JevBackend, Shadow, choice, noul, score
from decision_shadow import analysis


class FakeBackend:
    def __init__(self, label="billing", confidence=0.9, error=None, delay=0.0):
        self.label, self.confidence, self.error, self.delay = label, confidence, error, delay
        self.calls = 0

    def decide(self, state, questions):
        self.calls += 1
        time.sleep(self.delay)
        if self.error:
            return Decision(error=self.error, ms=1)
        return Decision(
            answers={"team": Answer(type="choice", choice=self.label, confidence=self.confidence,
                                    probabilities={self.label: self.confidence})},
            model="typesafe/jev-1.13-20260917", cost=0.00001, ms=5,
        )


Q = {"team": choice("Which team owns this ticket?", {"billing": "Money", "technical": "Bugs"})}


def confident(d, floor=0.8):
    a = d["team"]
    return a.choice if a.certainty >= floor else None


class ShadowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.log = Path(self.tmp.name) / "d.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def wrap(self, backend, mode, legacy="billing"):
        sh = Shadow(backend=backend, log=self.log, mode=mode)

        @sh.site("tickets", state=lambda t: {"ticket": t}, questions=Q, decide=confident)
        def route(text):
            route.calls += 1
            return legacy

        route.calls = 0
        return sh, route

    def records(self):
        return analysis.load(self.log) if self.log.exists() else []

    def test_off_never_calls_backend(self):
        b = FakeBackend()
        sh, route = self.wrap(b, "off")
        self.assertEqual(route("x"), "billing")
        sh.flush()
        self.assertEqual(b.calls, 0)
        self.assertEqual(self.records(), [])

    def test_shadow_returns_legacy_and_logs_both(self):
        b = FakeBackend(label="technical")
        sh, route = self.wrap(b, "shadow")
        self.assertEqual(route("refund please"), "billing")
        sh.flush()
        [r] = self.records()
        self.assertEqual(r["used"], "legacy")
        self.assertEqual(r["legacy_value"], "billing")
        self.assertEqual(r["model_value"], "technical")
        self.assertIs(r["agree"], False)
        self.assertNotIn("state", r)

    def test_shadow_runs_model_concurrently(self):
        b = FakeBackend(delay=0.3)
        sh, route = self.wrap(b, "shadow")
        t0 = time.monotonic()
        route("x")
        self.assertLess(time.monotonic() - t0, 0.2)
        sh.flush()
        self.assertEqual(len(self.records()), 1)

    def test_on_uses_model_without_calling_legacy(self):
        sh, route = self.wrap(FakeBackend(label="technical"), "on")
        self.assertEqual(route("x"), "technical")
        self.assertEqual(route.calls, 0)
        self.assertEqual(self.records()[0]["used"], "model")

    def test_on_falls_back_when_unsure(self):
        sh, route = self.wrap(FakeBackend(label="technical", confidence=0.4), "on")
        self.assertEqual(route("x"), "billing")
        r = self.records()[0]
        self.assertEqual(r["used"], "fallback")
        self.assertIsNone(r["model_value"])

    def test_on_falls_back_on_backend_error(self):
        sh, route = self.wrap(FakeBackend(error="503: down"), "on")
        self.assertEqual(route("x"), "billing")
        self.assertEqual(self.records()[0]["error"], "503: down")

    def test_decide_exception_is_a_fallback_not_a_crash(self):
        sh = Shadow(backend=FakeBackend(), log=self.log, mode="on")

        @sh.site("t", state=lambda t: t, questions=Q, decide=lambda d: d["missing"].choice)
        def route(text):
            return "billing"

        self.assertEqual(route("x"), "billing")
        self.assertIn("decide raised KeyError", self.records()[0]["error"])

    def test_mode_from_env(self):
        b = FakeBackend()
        sh = Shadow(backend=b, log=self.log)
        with mock.patch.dict("os.environ", {"DECISION_SHADOW_MODE": "shadow"}):
            self.assertEqual(sh.mode, "shadow")
        with mock.patch.dict("os.environ", {"DECISION_SHADOW_MODE": "bogus"}):
            self.assertEqual(sh.mode, "off")


class QuestionTests(unittest.TestCase):
    def test_noul_nests_descriptions_under_criteria(self):
        q = noul("Refund?", true="Wants money back", false="Does not")
        self.assertEqual(q["criteria"], {"true": "Wants money back", "false": "Does not"})
        self.assertNotIn("criteria", noul("Refund?"))

    def test_limits(self):
        with self.assertRaises(ValueError):
            choice("x", {"only": "one"})
        with self.assertRaises(ValueError):
            score("x", [str(i) for i in range(11)])

    def test_certainty(self):
        self.assertEqual(Answer(type="noul", noul=0.1).certainty, 0.9)
        self.assertEqual(Answer(type="choice", probabilities={"a": 0.7, "b": 0.3}).certainty, 0.7)
        self.assertEqual(Answer(type="score", confidence=0.77, probabilities={"1": 0.85}).certainty, 0.77)


class JevBackendTests(unittest.TestCase):
    RESPONSE = {
        "model": "typesafe/jev-1.13-20260917",
        "answers": {"severity": {"type": "score", "score": 1.15,
                                 "probabilities": {"0": 0, "1": 0.85, "2": 0.15}, "confidence": 0.77}},
        "usage": {"input_tokens": 354, "output_tokens": 17, "cost": 0.000014868},
    }

    def test_parses_response(self):
        resp = mock.MagicMock()
        resp.__enter__.return_value.read.return_value = json.dumps(self.RESPONSE).encode()
        with mock.patch("urllib.request.urlopen", return_value=resp) as op:
            d = JevBackend(api_key="k").decide("state", {"severity": score("How bad?", ["a", "b", "c"])})
        self.assertTrue(d.ok)
        self.assertEqual(d["severity"].value, 1.15)
        self.assertEqual(d.model, "typesafe/jev-1.13-20260917")
        self.assertAlmostEqual(d.cost, 0.000014868)
        body = json.loads(op.call_args[0][0].data)
        self.assertEqual(body["model"], "~typesafe/jev-latest")

    def test_no_key_is_an_error_not_a_call(self):
        with mock.patch.dict("os.environ", {}, clear=True), mock.patch("urllib.request.urlopen") as op:
            d = JevBackend().decide("s", {})
        self.assertEqual(d.error, "no API key")
        op.assert_not_called()

    def test_client_errors_are_not_retried(self):
        err = urllib.error.HTTPError("u", 400, "bad", {}, io.BytesIO(b"bad question"))
        with mock.patch("urllib.request.urlopen", side_effect=err) as op:
            d = JevBackend(api_key="k", retries=3).decide("s", {})
        self.assertEqual(op.call_count, 1)
        self.assertTrue(d.error.startswith("400"))

    def test_rate_limit_is_retried(self):
        err = urllib.error.HTTPError("u", 429, "slow", {"Retry-After": "0"}, io.BytesIO(b""))
        ok = mock.MagicMock()
        ok.__enter__.return_value.read.return_value = json.dumps(self.RESPONSE).encode()
        with mock.patch("urllib.request.urlopen", side_effect=[err, ok]) as op:
            d = JevBackend(api_key="k", retries=1).decide("s", {})
        self.assertEqual(op.call_count, 2)
        self.assertTrue(d.ok)


if __name__ == "__main__":
    unittest.main()
