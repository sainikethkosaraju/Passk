# Method notes
- **pass^k** = C(c,k)/C(n,k): chance that k trials drawn without replacement from the n recorded
  trials all succeed. Exact; verified against brute-force enumeration (`tests/test_metrics.py`).
- **pass@k** = 1 - C(n-c,k)/C(n,k) (Chen et al. 2021).
- Per-task rates get Wilson intervals; overall metrics are means over tasks (tasks with n<k are
  excluded and counted) with a percentile bootstrap **over tasks**. Trials within a task are not
  independent of the task, so bootstrapping trials would understate uncertainty.
- Errored samples count as failures.
- **compare** pairs by task id, bootstraps the mean difference in pass^k, and refuses to call
  a verdict if either side has fewer than `--min-trials` (default 5) trials per task.
- **Cost** is *shadow cost*: tokens x list price from a dated snapshot. Free-tier spend is $0.
  The bundled snapshot is ILLUSTRATIVE; replace it with a dated LiteLLM cost-map snapshot.
  Long-context tier is applied per sample total (approximation).
# Limits
- Success = all scorers correct; partial credit is not modelled.
- Interval coverage is checked by seeded simulation, not proven for small task counts.
- Sample logs in `examples/logs` are synthetic (`examples/make_sample_logs.py`).
