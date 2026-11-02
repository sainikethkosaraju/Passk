# Landscape teardown (blueprint task T0.3)
Method: Inspect's installed source was read directly (v0.3.287). Everything else comes from **one
web search session on 2026-11-02**; those rows are leads to verify, not conclusions. No claim here
is endorsed by, or compatible with, any listed project.

| Capability | Who already has it | Evidence | Passk's position |
| --- | --- | --- | --- |
| pass^k / pass@k point estimators | **Inspect itself**: reducers `pass_k` (`C(c,k)/C(n,k)`) and `pass_at` | `inspect_ai/scorer/_reducer/reducer.py` | **Gap closed.** Passk uses the same estimator and *cross-validates against Inspect's reducers* (`tests/test_inspect_integration.py`). It does not claim novelty here. |
| Intervals across tasks and trials, paired version comparison, exact tests, power/MDE | Not found in Inspect (it reports per-run metrics and stderr) | source read | **Gap confirmed (partly).** Two-stage smoothed bootstrap, Jeffreys/Bayes extrapolation, paired sign-flip, Holm, MDE; calibration in [coverage.md](coverage.md). |
| CI gate on eval results | `evalgate-ci` (Wilson lower bound; McNemar paired test), `cigate` (bias-corrected judge pass rate) per [search results](https://pypi.org/project/evalgate-ci/) | web, unverified | **Overlap.** Passk's gate is on *pass^k*, uses a paired cluster bootstrap + exact randomisation test, has a non-inferiority mode and reports MDE. Whether that matters to users is untested. |
| Cost per successful task with CI, Pareto frontier | Articles recommend it; no Inspect-native tool found | web, unverified | Gap plausible, not proven. Prices: real LiteLLM snapshot, resolution recorded per model. |
| Deterministic record/replay inside Inspect | Inspect has response caching (`cache=`) keyed on inputs, not trial index | not verified in source this session | Passk's provider keys on the **epoch** so repeated trials of one prompt are preserved. Verify against `cache` semantics before claiming superiority. |
| OpenAI Evals export -> runnable Inspect task with loss report | not found | web | Gap plausible. Import formats are reconstructed from public API shapes, not a vendor spec. |

**Positioning after the teardown:** Passk is *not* "pass^k for Inspect" (Inspect has it). It is the
statistics, cost and regression-gating layer around Inspect logs, calibrated by simulation. If a
reviewer asks "why not evalgate-ci?", the answer is the metric (pass^k), the interval method
(measured coverage) and the Inspect-native record/replay, not the existence of a gate.
