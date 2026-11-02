# passk

Reliability and cost layer for [Inspect AI](https://inspect.aisi.org.uk/) eval logs: how often does an agent succeed **every** time, at what cost, and did a change make it better or worse?

```bash
pip install -e .                              # no runtime dependencies
passk report examples/logs/sample_a.json      # pass^k, pass@k, intervals, shadow cost, flaky tasks
passk gate --baseline examples/logs/sample_a.json --candidate examples/logs/sample_b_regressed.json --max-drop 0.05   # exits 1
```

## How it works
```
Inspect logs (.json/.eval) -> logs.py (trial table) -> metrics.py (pass^k, pass@k, Wilson, bootstrap)
                                                    -> cost.py (dated price snapshot)
                                                    -> compare.py / flaky.py -> report/ (JSON + HTML) -> cli.py gate
```
`pass^k` = C(c,k)/C(n,k), `pass@k` = 1-C(n-c,k)/C(n,k); see [docs/method.md](docs/method.md).

## Commands
| Command | What |
| --- | --- |
| `passk report LOGS [--out DIR] [--otel spans.jsonl]` | per-task + overall metrics, JSON and static HTML |
| `passk compare A B [-k N]` | paired verdict: regressed / improved / inconclusive (refuses below 5 trials/task) |
| `passk cost LOGS` | shadow cost and cost per success |
| `passk gate --baseline A --candidate B --max-drop 0.05` | CI gate, exit 1 on regression |
| `passk import-openai EXPORT_DIR --out DIR` | OpenAI Evals export -> Inspect task + `import_report.md` |

Python API: `passk.metrics.pass_hat_k(n, c, k)`, `pass_at_k(n, c, k)`, `passk.cost.price(usage, model)`.
Also: `passk.graders` (state, tool-call, model grader with Unknown + calibration), `passk.cassette`, `passk.pacing`, `passk.otel`.

## Verified claims
| Claim | Evidence |
| --- | --- |
| Estimators equal exact enumeration; pass^1 = pass@1; pass^k non-increasing | [tests/test_metrics.py](tests/test_metrics.py) |
| Intervals reach nominal coverage (seeded simulation) | `test_wilson_coverage`, `test_bootstrap_coverage_over_tasks` |
| Gate catches injected regression, passes noise | [tests/test_compare_cli.py](tests/test_compare_cli.py), data from [examples/make_sample_logs.py](examples/make_sample_logs.py) |
| Cost matches hand-computed values (cached + long-context) | [tests/test_logs_cost_flaky.py](tests/test_logs_cost_flaky.py) |
| Report schema locked | `test_report_json_and_html_golden` |

## Limitations
- Sample logs are **synthetic** (Inspect schema); a live Inspect run was not possible in the build sandbox. Reader built against the observed Inspect 0.3.287 sample keys ([docs/inspect-logs.md](docs/inspect-logs.md)).
- The bundled price snapshot is **illustrative**, not authoritative. Replace before publishing cost numbers.
- The OpenAI export layout is reconstructed from public API shapes ([docs/importer.md](docs/importer.md)).
- OTLP export is optional and untested against a live collector; OpenTelemetry GenAI conventions are still in Development.
- `collect.yml` is untested live (requires your repo secrets).

## Configuration
Copy `.env.example` to `.env` (git-ignored). Passk itself needs no keys; they are only for the scheduled collector (store as repository secrets).

## What I would do differently
Run a real Inspect eval to validate the reader end-to-end, and ship a price-snapshot updater.

## AI assistance
Written with Claude Code; every claim above is backed by a test in this repo.

MIT licensed. See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), [CHANGELOG.md](CHANGELOG.md).
