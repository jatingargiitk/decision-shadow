from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import analysis


def _fmt(v: Optional[float], pct: bool = True) -> str:
    if v is None:
        return "-"
    return f"{v:.1%}" if pct else f"{v}"


def cmd_report(args: argparse.Namespace) -> int:
    records = analysis.load(args.log)
    sites = analysis.report(records)
    if args.json:
        print(json.dumps({k: {
            "n": s.n, "errors": s.errors, "compared": s.compared, "agreement": s.agreement,
            "fallbacks": s.fallbacks, "cost": s.cost, "saturation": s.saturation,
            "model_ms_p50": analysis.percentile(s.model_ms, 0.5),
            "model_ms_p95": analysis.percentile(s.model_ms, 0.95),
            "legacy_ms_p50": analysis.percentile(s.legacy_ms, 0.5),
            "snapshots": dict(s.snapshots), "warnings": s.warnings(),
        } for k, s in sites.items()}, indent=2))
        return 0
    if not sites:
        print(f"no records in {args.log}")
        return 1
    for s in sorted(sites.values(), key=lambda s: -s.n):
        print(f"\n{s.site}")
        print(f"  decisions   {s.n}  (errors {s.errors}, fallbacks {s.fallbacks})")
        print(f"  agreement   {_fmt(s.agreement)} of {s.compared} compared")
        print(f"  latency     model p50 {_fmt(analysis.percentile(s.model_ms, .5), False)} ms"
              f" / p95 {_fmt(analysis.percentile(s.model_ms, .95), False)} ms"
              f"   legacy p50 {_fmt(analysis.percentile(s.legacy_ms, .5), False)} ms")
        print(f"  cost        ${s.cost:.6f}")
        print(f"  saturation  {_fmt(s.saturation)} of {s.answers} answers at >= {analysis.SATURATED}")
        if s.disagreements:
            print("  top disagreements (legacy -> model)")
            for (a, b), n in s.disagreements.most_common(5):
                print(f"    {n:>4}  {a} -> {b}")
        for w in s.warnings():
            print(f"  ! {w}")
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    records = analysis.load(args.log)
    labels = None
    if args.labels:
        labels = {}
        for r in analysis.load(args.labels):
            labels[r["id"]] = r["label"]
    cuts, total, against = analysis.tune(records, args.site, args.question, labels)
    if not total:
        print(f"no usable decisions for site={args.site} question={args.question}")
        return 1
    print(f"{args.site}.{args.question}: {total} decisions, accuracy against {against}\n")
    print("  threshold  coverage  accuracy      n")
    for c in cuts:
        print(f"  {c.threshold:>9.2f}  {c.coverage:>8.1%}  {_fmt(c.accuracy):>8}  {c.n:>5}")
    best = analysis.recommend(cuts, args.target, args.min_n)
    if best:
        print(f"\nrecommended: act when certainty >= {best.threshold:.2f}"
              f" — {best.coverage:.0%} of traffic at {best.accuracy:.1%} accuracy")
    else:
        print(f"\nno threshold reaches {args.target:.0%} accuracy on >= {args.min_n} decisions")
    if labels is None:
        print("note: no labels given, so 'accuracy' means agreement with the classifier being replaced")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="decision-shadow")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("report", help="per-site agreement, latency, cost and warnings")
    r.add_argument("log")
    r.add_argument("--json", action="store_true")
    r.set_defaults(fn=cmd_report)

    t = sub.add_parser("tune", help="sweep certainty thresholds for one question")
    t.add_argument("log")
    t.add_argument("--site", required=True)
    t.add_argument("--question", required=True)
    t.add_argument("--labels", help="JSONL of {\"id\": ..., \"label\": ...}")
    t.add_argument("--target", type=float, default=0.95)
    t.add_argument("--min-n", type=int, default=20)
    t.set_defaults(fn=cmd_tune)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
