"""passk report | compare | gate | cost | power | import-openai | update-prices"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from passk import stats
from passk.compare import MIN_TRIALS, Comparison, compare, compare_ks
from passk.cost import PriceBook, build_snapshot, cost_summary
from passk.importers.openai_evals import import_openai
from passk.logs import read_logs
from passk.metrics import aggregate
from passk.report import build_report, render_html


def _book(args: argparse.Namespace) -> PriceBook | None:
    book = PriceBook.load(args.prices) if getattr(args, "prices", None) else None
    aliases = dict(a.split("=", 1) for a in (getattr(args, "alias", None) or []))
    if aliases:
        book = book or PriceBook.load()
        book.aliases.update(aliases)
    return book


def _report(args: argparse.Namespace) -> int:
    table = read_logs(*args.logs)
    rep = build_report(table, _book(args))
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
    lines = []
    for m, s in rep["models"].items():
        lines.append(f"{m}: {s['trials']} trials, {s['tasks_count']} tasks, {s['errors']} errored")
        for k, v in s["pass_hat_k"].items():
            lines.append(f"  pass^{k} = {v['value']:.3f}  [{v['lo']:.3f}, {v['hi']:.3f}]   "
                         f"pass@{k} = {s['pass_at_k'][k]['value']:.3f}")
        c = s["cost"]
        if c.get("total_usd") is not None:
            per = c.get("usd_per_success")
            lines.append(f"  shadow cost ${c['total_usd']:.4f}; per success "
                         f"{'n/a' if per is None else f'${per:.5f}'}; missing prices: {c['missing_prices'] or 'none'}")
        lines.append(f"  flaky: {s['flags']['flaky']}  zero-success: {s['flags']['zero_success']}")
    lines += [f"warning: {w}" for w in rep["warnings"]]
    return "\n".join(lines)


def _fmt_cmp(c: Comparison) -> str:
    return (f"pass^{c.k}: delta {c.delta:+.3f} [{c.lo:+.3f}, {c.hi:+.3f}] over {c.tasks} tasks; "
            f"p={c.p_value:.4f}{'' if c.p_adjusted is None else f' (Holm {c.p_adjusted:.4f})'}; "
            f"wins/losses/ties {c.wins}/{c.losses}/{c.ties}; d_z={c.effect_size:.2f}; "
            f"MDE={c.mde:.3f}; verdict: {c.verdict} ({c.reason})")


def _compare(args: argparse.Namespace) -> int:
    a, b = read_logs(args.a), read_logs(args.b)
    ks = args.k or [1]
    res = compare_ks(a, b, ks, min_trials=args.min_trials, alpha=args.alpha) if len(ks) > 1 else \
        [compare(a, b, ks[0], min_trials=args.min_trials, alpha=args.alpha)]
    if args.json:
        print(json.dumps([c.__dict__ for c in res], indent=2, default=str))
    else:
        print("\n".join(_fmt_cmp(c) for c in res))
    return 0


def _cost(args: argparse.Namespace) -> int:
    print(json.dumps(cost_summary(read_logs(*args.logs), _book(args)), indent=2))
    return 0


def _gate(args: argparse.Namespace) -> int:
    """Exit 0 pass, 1 regression, 2 not enough data to decide (when --strict)."""
    base, cand = read_logs(args.baseline), read_logs(args.candidate)
    a = aggregate(list(base.counts().values()), args.k, reps=200)["value"]
    b = aggregate(list(cand.counts().values()), args.k, reps=200)["value"]
    c = compare(base, cand, args.k, min_trials=args.min_trials, alpha=args.alpha)
    drop = a - b
    if args.mode == "point":
        fail, why = drop > args.max_drop, f"drop {drop:+.3f} vs max {args.max_drop}"
    elif args.mode == "significant":
        fail = c.verdict == "regressed" and drop > args.max_drop
        why = f"verdict {c.verdict} and drop {drop:+.3f} vs max {args.max_drop}"
    else:  # noninferior: must PROVE the candidate is within margin
        if c.verdict == "inconclusive" and "refusing" in c.reason:
            print("GATE INCONCLUSIVE: " + c.reason, file=sys.stderr)
            return 2 if args.strict else 0
        fail, why = c.lo <= -args.max_drop, f"lower bound {c.lo:+.3f} vs margin -{args.max_drop}"
    status = "FAIL" if fail else "PASS"
    if args.format == "markdown":
        print(f"### Passk gate: **{status}**\n\n| | baseline | candidate |\n|---|---|---|\n"
              f"| pass^{args.k} | {a:.3f} | {b:.3f} |\n\n{_fmt_cmp(c)}\n\nMode `{args.mode}`: {why}")
    elif args.format == "json":
        print(json.dumps({"status": status, "mode": args.mode, "baseline": a, "candidate": b,
                          "comparison": c.__dict__}, default=str))
    else:
        print(f"{status}: pass^{args.k} {a:.3f} -> {b:.3f} ({why}); {c.verdict}")
    return 1 if fail else 0


def _power(args: argparse.Namespace) -> int:
    if args.logs:
        a, b = read_logs(args.logs[0]), read_logs(args.logs[1])
        c = compare(a, b, args.k)
        print(f"observed: {c.tasks} tasks, MDE at 80% power = {c.mde:.3f}")
        sdv = c.mde * (c.tasks ** 0.5) / (stats._N.inv_cdf(0.975) + stats._N.inv_cdf(0.8)) if c.tasks else 0
    else:
        sdv = args.sd
    if args.delta:
        print(f"tasks needed to detect delta={args.delta} (sd of paired diffs {sdv:.3f}): "
              f"{stats.required_tasks(args.delta, sdv)}")
    return 0


def _update_prices(args: argparse.Namespace) -> int:
    out = build_snapshot(args.source, args.out, args.include)
    print(f"wrote {out}")
    return 0


def _import(args: argparse.Namespace) -> int:
    res = import_openai(args.export_dir, args.out)
    print(f"converted {res.items} items; {len(res.not_converted)} notes in import_report.md")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="passk")
    sub = p.add_subparsers(dest="cmd", required=True)
    price_opts = argparse.ArgumentParser(add_help=False)
    price_opts.add_argument("--prices"); price_opts.add_argument("--alias", action="append",
                                                                 metavar="LOGMODEL=PRICEKEY")
    r = sub.add_parser("report", parents=[price_opts]); r.add_argument("logs", nargs="+")
    r.add_argument("--out"); r.add_argument("--json", action="store_true")
    r.add_argument("--otel", help="write GenAI spans JSONL here"); r.set_defaults(fn=_report)
    c = sub.add_parser("compare"); c.add_argument("a"); c.add_argument("b")
    c.add_argument("-k", type=int, action="append"); c.add_argument("--min-trials", type=int, default=MIN_TRIALS)
    c.add_argument("--alpha", type=float, default=0.05); c.add_argument("--json", action="store_true")
    c.set_defaults(fn=_compare)
    co = sub.add_parser("cost", parents=[price_opts]); co.add_argument("logs", nargs="+")
    co.set_defaults(fn=_cost)
    g = sub.add_parser("gate"); g.add_argument("--baseline", required=True)
    g.add_argument("--candidate", required=True); g.add_argument("--max-drop", type=float, default=0.05)
    g.add_argument("-k", type=int, default=1); g.add_argument("--min-trials", type=int, default=MIN_TRIALS)
    g.add_argument("--alpha", type=float, default=0.05)
    g.add_argument("--mode", choices=["point", "significant", "noninferior"], default="significant")
    g.add_argument("--format", choices=["text", "markdown", "json"], default="text")
    g.add_argument("--strict", action="store_true", help="exit 2 when data is insufficient")
    g.set_defaults(fn=_gate)
    pw = sub.add_parser("power"); pw.add_argument("logs", nargs="*"); pw.add_argument("-k", type=int, default=1)
    pw.add_argument("--delta", type=float); pw.add_argument("--sd", type=float, default=0.3)
    pw.set_defaults(fn=_power)
    up = sub.add_parser("update-prices"); up.add_argument("source", nargs="?",
        default="https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json")
    up.add_argument("--out", default="data/prices"); up.add_argument("--include", default=None)
    up.set_defaults(fn=lambda a: _update_prices(_with_default_include(a)))
    i = sub.add_parser("import-openai"); i.add_argument("export_dir"); i.add_argument("--out", default="imported")
    i.set_defaults(fn=_import)
    args = p.parse_args(argv)
    return args.fn(args)


def _with_default_include(a: argparse.Namespace) -> argparse.Namespace:
    from passk.cost import DEFAULT_INCLUDE
    a.include = a.include or DEFAULT_INCLUDE
    return a


if __name__ == "__main__":
    sys.exit(main())
