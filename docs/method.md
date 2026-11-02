# Method
- **pass^k** = C(c,k)/C(n,k): P(k trials drawn without replacement from n recorded trials all succeed); unbiased for p^k (property-tested). **pass@k** = 1 - C(n-c,k)/C(n,k).
- **Unit of analysis is the task.** Headline = mean over tasks. Tasks with n < k are excluded and counted.
- **Interval**: two-stage bootstrap. Resample tasks; for each drawn task draw p ~ Beta(c+1/2, n-c+1/2) (Jeffreys) then n new Bernoulli trials; recompute. Plug-in p = c/n was tried first and undercovers at high k because tasks at 0/n or n/n contribute no within-task variance ([coverage.md](coverage.md)).
- **Bayes**: per task p ~ Beta(1/2+c, 1/2+n-c); E[p^k] closed form via gamma functions; defined for k > n.
- **Paired comparison**: per-task difference of pass^k; CI by paired two-stage bootstrap; p by exact sign-flip randomisation (<= 16 nonzero pairs, else Monte Carlo); verdict requires both; Holm across several k; MDE = (z_{1-a/2} + z_power) * sd / sqrt(tasks).
- **Gate modes**: `point` (drop > margin), `significant` (regressed AND drop > margin), `noninferior` (fail unless the lower CI bound is above -margin; needs enough data, exit 2 with `--strict` otherwise).
- **Cost**: shadow cost = tokens x list price from a dated snapshot; model names are resolved exact -> alias -> prefix-stripped -> median across providers, and the route is recorded.
