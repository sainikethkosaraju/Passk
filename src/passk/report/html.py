"""Static, dependency-free HTML report (inline SVG, no external assets)."""

from __future__ import annotations

from html import escape
from typing import Any


def _curve(report: dict[str, Any]) -> str:
    pts = [(int(k), v["value"]) for k, v in report["pass_hat_k"].items() if v["value"] is not None]
    if not pts:
        return "<p>No data.</p>"
    w, h, pad = 360, 180, 30
    xmax = max(k for k, _ in pts) or 1
    def sx(k): return pad + (k - 1) * (w - 2 * pad) / max(1, xmax - 1)
    def sy(v): return h - pad - v * (h - 2 * pad)
    path = " ".join(f"{sx(k):.1f},{sy(v):.1f}" for k, v in pts)
    dots = "".join(f'<circle cx="{sx(k):.1f}" cy="{sy(v):.1f}" r="3"/>' for k, v in pts)
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="pass^k curve">'
            f'<polyline fill="none" stroke="currentColor" points="{path}"/>{dots}'
            f'<text x="{pad}" y="{h - 8}" font-size="10">k = 1..{xmax}</text>'
            f'<text x="2" y="{pad}" font-size="10">1.0</text></svg>')


def render_html(report: dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><td>{escape(t['id'])}</td><td>{t['successes']}/{t['n']}</td>"
        f"<td>{t['wilson_lo']:.2f}-{t['wilson_hi']:.2f}</td></tr>"
        for t in report["tasks"]
    )
    k_rows = "".join(
        f"<tr><td>{k}</td><td>{_f(v['value'])}</td><td>{_f(v['lo'])}-{_f(v['hi'])}</td>"
        f"<td>{_f(report['pass_at_k'][k]['value'])}</td></tr>"
        for k, v in report["pass_hat_k"].items()
    )
    cost = report["cost"]
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Passk report</title>
<style>body{{font:15px system-ui;max-width:760px;margin:2rem auto;padding:0 1rem}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #8884;padding:4px 8px;text-align:left}}</style>
<h1>Passk report</h1>
<p>Models: {escape(", ".join(report["models"]))} &middot; trials: {report["trials"]} &middot; tasks: {report["tasks_count"]}</p>
<h2>Reliability</h2>{_curve(report)}
<table><tr><th>k</th><th>pass^k</th><th>95% CI (bootstrap over tasks)</th><th>pass@k</th></tr>{k_rows}</table>
<h2>Cost (shadow, snapshot {escape(str(cost.get("price_snapshot", "n/a")))})</h2>
<p>Total ${_f(cost.get("total_usd"), 4)}; per success ${_f(cost.get("usd_per_success"), 4)}.
Missing prices: {escape(", ".join(cost.get("missing_prices", [])) or "none")}</p>
<h2>Latency</h2><p>median {_f(report["latency"]["median_s"])}s, p95 {_f(report["latency"]["p95_s"])}s</p>
<h2>Flags</h2><p>Flaky: {escape(", ".join(report["flags"]["flaky"]) or "none")}.
Zero-success: {escape(", ".join(report["flags"]["zero_success"]) or "none")}.</p>
<h2>Per-task</h2><table><tr><th>Task</th><th>Successes</th><th>Wilson 95%</th></tr>{rows}</table>
<p>{escape("; ".join(report["warnings"]))}</p></html>"""


def _f(x: Any, d: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{d}f}"
