import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

from decision_shadow import analysis
from decision_shadow.cli import main


def rec(i, legacy, model, conf, site="tickets", snapshot="typesafe/jev-1.13-20260917", **kw):
    r = {
        "id": f"r{i}", "site": site, "used": "legacy",
        "legacy_value": legacy, "model_value": model, "agree": legacy == model,
        "answers": {"team": {"type": "choice", "choice": model, "confidence": conf,
                             "probabilities": {model: conf}}},
        "model": snapshot, "model_ms": 100 + i, "legacy_ms": 900 + i, "cost": 0.00001, "error": None,
    }
    r.update(kw)
    return r


class ReportTests(unittest.TestCase):
    def test_agreement_latency_and_disagreements(self):
        rs = [rec(i, "billing", "billing", 0.95) for i in range(8)]
        rs += [rec(8, "billing", "technical", 0.6), rec(9, "billing", "technical", 0.55)]
        s = analysis.report(rs)["tickets"]
        self.assertEqual(s.n, 10)
        self.assertAlmostEqual(s.agreement, 0.8)
        self.assertEqual(s.disagreements.most_common(1)[0][1], 2)
        self.assertEqual(analysis.percentile(s.model_ms, 0.5), 104)

    def test_warns_on_saturation_and_snapshot_change(self):
        rs = [rec(i, "a", "a", 1.0) for i in range(10)]
        rs.append(rec(10, "a", "a", 1.0, snapshot="typesafe/jev-1.14-20261001"))
        w = " ".join(analysis.report(rs)["tickets"].warnings())
        self.assertIn("snapshot changed", w)
        self.assertIn("certainty", w)

    def test_errors_do_not_count_toward_latency(self):
        rs = [rec(0, "a", None, 0.9, error="timeout", agree=None, answers={})]
        s = analysis.report(rs)["tickets"]
        self.assertEqual(s.errors, 1)
        self.assertEqual(s.model_ms, [])


class TuneTests(unittest.TestCase):
    def rows(self):
        hi = [rec(i, "billing", "billing", 0.95) for i in range(30)]
        lo = [rec(100 + i, "billing", "technical" if i % 2 else "billing", 0.6) for i in range(20)]
        return hi + lo

    def test_higher_threshold_trades_coverage_for_accuracy(self):
        cuts, total, against = analysis.tune(self.rows(), "tickets", "team")
        self.assertEqual(total, 50)
        self.assertEqual(against, "legacy classifier")
        by_t = {c.threshold: c for c in cuts}
        self.assertAlmostEqual(by_t[0.5].accuracy, 0.8)
        self.assertEqual(by_t[0.9].accuracy, 1.0)
        self.assertAlmostEqual(by_t[0.9].coverage, 0.6)
        self.assertEqual(analysis.recommend(cuts, 0.95).threshold, 0.7)

    def test_labels_override_legacy(self):
        labels = {f"r{i}": "technical" for i in range(30)}
        cuts, total, against = analysis.tune(self.rows(), "tickets", "team", labels)
        self.assertEqual(total, 30)
        self.assertEqual(against, "labels")
        self.assertEqual(cuts[0].accuracy, 0.0)

    def test_recommend_needs_enough_decisions(self):
        cuts, _, _ = analysis.tune(self.rows()[:5], "tickets", "team")
        self.assertIsNone(analysis.recommend(cuts, 0.95, min_n=20))


class CliTests(unittest.TestCase):
    def test_report_and_tune_run(self):
        with TemporaryDirectory() as d:
            log = Path(d) / "d.jsonl"
            log.write_text("".join(json.dumps(rec(i, "a", "a", 0.9)) + "\n" for i in range(25)) + "not json\n")
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(main(["report", str(log)]), 0)
                self.assertEqual(main(["tune", str(log), "--site", "tickets", "--question", "team"]), 0)
            self.assertIn("agreement   100.0% of 25", out.getvalue())
            self.assertIn("recommended: act when certainty >= 0.50", out.getvalue())


if __name__ == "__main__":
    unittest.main()
