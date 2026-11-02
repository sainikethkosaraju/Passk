"""Self-contained static HTML report (inline SVG, no external assets, light/dark aware)."""

from __future__ import annotations

from html import escape
from typing import Any

COLORS = ["#2563eb", "#d97706", "#059669", "#9333ea", "#dc2626", "#0891b2"]
CSS = """
:root{--bg:#fff;--fg:#111827;--mut:#6b7280;--line:#d1d5db;--ok:#059669;--bad:#dc2626;--card:#f9fafb}
@media (prefers-color-scheme:dark){:root{--bg:#0b0f17;--fg:#e5e7eb;--mut:#9ca3af;--line:#374151;--card:#111827}}
body{font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg);max-width:880px;margin:2rem auto;padding:0 16px}
h1,h2,h3{line-height:1.2}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
td,th{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin:12px 0}
svg{max-width:100%;height:auto}svg text{fill:var(--fg);font-size:11px}.mut{color:var(--mut)}
.cell{display:inline-block;width:11px;height:11px;margin-right:2px;border-radius:2px}
.ok{background:var(--ok)}.bad{background:var(--bad)}
"""


def _f(x: Any, d: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{d}f}"


def _curve(models: dict[str, Any]) -> str:
    w, h, pl, pb = 440, 220, 36, 26
    xmax = max((int(k) for s in models.values() for k in s["pass_hat_k"]), default=1)
    sx = lambda k: pl + (k - 1) * (w - pl - 12) / max(1, xmax - 1)  # noqa: E731
    sy = lambda v: h - pb - v * (h - pb - 12)  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="pass^k versus k with 95% intervals">']
    for g in (0, 0.25, 0.5, 0.75, 1):
        out.append(f'<line x1="{pl}" x2="{w - 12}" y1="{sy(g):.1f}" y2="{sy(g):.1f}" stroke="var(--line)"/>'
                   f'<text x="2" y="{sy(g) + 4:.1f}">{g:.2f}</text>')
    for k in range(1, xmax + 1):
        out.append(f'<text x="{sx(k) - 3:.1f}" y="{h - 8}">{k}</text>')
    for i, (m, s) in enumerate(models.items()):
        col = COLORS[i % len(COLORS)]
        pts = [(int(k), v) for k, v in s["pass_hat_k"].items() if v["value"] is not None]
        if not pts:
            continue
        band = [f"{sx(k):.1f},{sy(v['hi']):.1f}" for k, v in pts] + \
               [f"{sx(k):.1f},{sy(v['lo']):.1f}" for k, v in reversed(pts)]
        out.append(f'<polygon points="{" ".join(band)}" fill="{col}" opacity="0.15"/>')
        out.append(f'<polyline fill="none" stroke="{col}" stroke-width="2" points="'
                   + " ".join(f"{sx(k):.1f},{sy(v['value']):.1f}" for k, v in pts) + '"/>')
        k, v = pts[-1]
        out.append(f'<text x="{sx(k) - 4:.1f}" y="{sy(v["value"]) - 6:.1f}" text-anchor="end" '
                   f'style="fill:{col}">{escape(m.split("/")[-1])}</text>')
    out.append("</svg>")
    return "".join(out)


def _frontier(rep: dict[str, Any]) -> str:
    pts = rep["frontier"]["points"]
    if len(pts) < 1:
        return '<p class="mut">No cost data.</p>'
    w, h, pl, pb = 440, 220, 44, 26
    cmax = max(p["usd_per_success"] for p in pts) * 1.15 or 1
    sx = lambda c: pl + c / cmax * (w - pl - 14)  # noqa: E731
    sy = lambda r: h - pb - r * (h - pb - 12)  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="cost versus reliability">',
           f'<text x="{pl}" y="{h - 6}">USD per success (shadow) →</text><text x="2" y="12">pass^{rep["frontier"]["k"]}</text>']
    nd = set(rep["frontier"]["non_dominated"])
    for p in pts:
        on = p["model"] in nd
        out.append(f'<circle cx="{sx(p["usd_per_success"]):.1f}" cy="{sy(p["pass_hat_k"]):.1f}" r="{6 if on else 4}" '
                   f'fill="{"var(--ok)" if on else "var(--mut)"}"/>'
                   f'<text x="{sx(p["usd_per_success"]) + 8:.1f}" y="{sy(p["pass_hat_k"]) + 4:.1f}">'
                   f'{escape(p["model"].split("/")[-1])}</text>')
    out.append("</svg>")
    return "".join(out)


def render_html(report: dict[str, Any]) -> str:
    body = []
    for m, s in report["models"].items():
        rows = "".join(
            f"<tr><td>{k}</td><td>{_f(v['value'])}</td><td>{_f(v['lo'])} – {_f(v['hi'])}</td>"
            f"<td>{_f(s['pass_at_k'][k]['value'])}</td><td>{v['tasks']}</td></tr>"
            for k, v in s["pass_hat_k"].items())
        strips = "".join(
            f"<tr><td>{escape(t['id'])}</td><td>" +
            "".join(f'<span class="cell {"ok" if x else "bad"}" title="{"pass" if x else "fail"}"></span>'
                    for x in t["trials"]) +
            f"</td><td>{t['successes']}/{t['n']}</td><td>{_f(t['jeffreys_lo'], 2)}–{_f(t['jeffreys_hi'], 2)}</td></tr>"
            for t in s["tasks"])
        c = s["cost"]
        ci = c.get("usd_per_success_ci")
        body.append(f"""<div class="card"><h2>{escape(m)}</h2>
<p>{s['trials']} trials over {s['tasks_count']} tasks ({s['errors']} errored). Latency median {_f(s['latency']['median_s'])}s,
p95 {_f(s['latency']['p95_s'])}s. Tokens in/out/cached: {s['tokens']['input']}/{s['tokens']['output']}/{s['tokens']['cache_read']}.</p>
<table><tr><th>k</th><th>pass^k</th><th>95% CI (two-stage bootstrap)</th><th>pass@k</th><th>tasks</th></tr>{rows}</table>
<p>Shadow cost ${_f(c.get('total_usd'), 4)}; per success ${_f(c.get('usd_per_success'), 4)}
{f"(95% CI {_f(ci[0], 4)}–{_f(ci[1], 4)})" if ci else ""}. Price snapshot {escape(str(c.get('price_snapshot', 'n/a')))}
({escape(', '.join(f'{a}: {b}' for a, b in c.get('price_resolution', {}).items()) or 'n/a')}).
Missing prices: {escape(', '.join(c.get('missing_prices', [])) or 'none')}.</p>
<p>Flaky: {escape(', '.join(s['flags']['flaky']) or 'none')}. Zero-success: {escape(', '.join(s['flags']['zero_success']) or 'none')}.</p>
<h3>Trials per task</h3><table><tr><th>Task</th><th>Trials (green = pass)</th><th>Passes</th><th>Jeffreys 95%</th></tr>{strips}</table></div>""")
    return f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Passk report</title><style>{CSS}</style><h1>Passk report</h1>
<div class="card"><h2>Reliability by k</h2>{_curve(report["models"])}
<p class="mut">pass^k = probability that k independent trials all succeed. Bands are 95% intervals from a two-stage bootstrap (tasks, then trials).</p></div>
<div class="card"><h2>Cost versus reliability</h2>{_frontier(report)}
<p class="mut">Green = non-dominated (Pareto frontier). Cost is shadow cost from a dated list-price snapshot.</p></div>
{"".join(body)}<p class="mut">{escape("; ".join(report["warnings"]))}</p></html>"""
