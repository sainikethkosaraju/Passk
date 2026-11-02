# Changelog
## 0.3.0
- Validated against real Inspect logs; added zstd `.eval` reader (the previous reader could not open real `.eval` files).
- New statistics core (Jeffreys/Bayes, BCa, two-stage smoothed bootstrap, exact randomisation, Holm, MDE) with Monte-Carlo calibration; fixed undercoverage found by the simulation.
- compare/gate rewritten (modes, markdown/json output, exit 2); real LiteLLM price snapshots with provenance, resolver, `update-prices`, Pareto frontier.
- Inspect hook + record/replay model provider; real OpenTelemetry export; importer for two OpenAI layouts producing runnable tasks; report schema v2.
- Landscape teardown: Inspect already ships `pass_k`; positioning corrected.
## 0.2.0
- Graders, OTel adapter, first importer. ## 0.1.0 - estimators, reports, CLI.
