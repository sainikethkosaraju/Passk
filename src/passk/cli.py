"""passk report | compare | cost | gate | import-openai"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from passk.compare import compare
from passk.cost import PriceBook, cost_summary
from passk.importers.openai_evals import import_openai
from passk.logs import read_logs
from passk.metrics import aggregate
from passk.report import build_report, render_html


def _report(args: argparse.Namespace) -> int:
    table = read_logs(*args.logs)
    rep = build_report(table, PriceBook.load(args.prices) if args.prices else None)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(rep, indent=2, sort_keys=True))
        (out / "report.html").write_text(render_html(rep))
    if args.otel:
        from passk.otel import trials_to_spans, write_jsonl
        write_jsonl(trials_to_spans(table), args.otel)
    print(json.dumps(rep, indent=2, sort_keys=True) if args.json else _summary(rep))
    return 0


def _summary(rep: dict) -> str:
    lines = [f"models: {', '.join(rep['models'])}  trials: {rep['trials']}  tasks: {rep['tasks_count']}"]
    for k, v in rep["pass_hat_k"].items():
        lines.append(f"pass^{k} = {v['value']:.3f}  [{v['lo']:.3f}, {v['hi']:.3f}]"
                     f"   pass@{k} = {rep['pass_at_k'][k]['value']:.3f}")
    c = rep["cost"]
    if "total_usd" in c:
        per = c["usd_per_success"]
        lines.append(f"shadow cost ${c['total_usd']:.4f}; per success "
                     f"{'n/a' if per is None else f'${per:.4f}'}")
    lines += [f"flaky: {rep['flags']['flaky']}", f"zero-success: {rep['flags']['zero_success']}"]
    lines += [f"warning: {w}" for w in rep["warnings"]]
    return "\n".join(lines)


def _compare(args: argparse.Namespace) -> int:
    c = compare(read_logs(args.a), read_logs(args.b), k=args.k, min_trials=args.min_trials)
    print(json.dumps(c.__dict__, indent=2, default=str))
    return 0


def _cost(args: argparse.Namespace) -> int:
    book = PriceBook.load(args.prices) if args.prices else None
    print(json.dumps(cost_summary(read_logs(*args.logs), book), indent=2))
    return 0


def _gate(args: argparse.Namespace) -> int:
    base, cand = read_logs(args.baseline), read_logs(args.candidate)
    a = aggregate(list(base.counts().values()), args.k)["value"]
    b = aggregate(list(cand.counts().values()), args.k)["value"]
    drop = a - b
    verdict = compare(base, cand, k=args.k, min_trials=args.min_trials)
    fail = drop > args.max_drop
    print(f"pass^{args.k}: baseline {a:.3f} -> candidate {b:.3f} (drop {drop:+.3f}, "
          f"max {args.max_drop}); paired verdict: {verdict.verdict}")
    if fail:
        print("GATE FAILED: reliability dropped by more than the threshold", file=sys.stderr)
    return 1 if fail else 0


def _import(args: argparse.Namespace) -> int:
    res = import_openai(args.export_dir, args.out)
    print(f"converted {res.items} items; {len(res.not_converted)} notes in import_report.md")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="passk")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report"); r.add_argument("logs", nargs="+")
    r.add_argument("--out"); r.add_argument("--json", action="store_true")
    r.add_argument("--prices"); r.add_argument("--otel", help="write GenAI spans JSONL here")
    r.set_defaults(fn=_report)
    c = sub.add_parser("compare"); c.add_argument("a"); c.add_argument("b")
    c.add_argument("-k", type=int, default=1); c.add_argument("--min-trials", type=int, default=5)
    c.set_defaults(fn=_compare)
    co = sub.add_parser("cost"); co.add_argument("logs", nargs="+"); co.add_argument("--prices")
    co.set_defaults(fn=_cost)
    g = sub.add_parser("gate"); g.add_argument("--baseline", required=True)
    g.add_argument("--candidate", required=True)
    g.add_argument("--max-drop", type=float, default=0.05)
    g.add_argument("-k", type=int, default=1); g.add_argument("--min-trials", type=int, default=5)
    g.set_defaults(fn=_gate)
    i = sub.add_parser("import-openai"); i.add_argument("export_dir"); i.add_argument("--out", default="imported")
    i.set_defaults(fn=_import)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
