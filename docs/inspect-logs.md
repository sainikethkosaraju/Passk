# Inspect log fields used (T0.5)
Observed on a real `inspect-ai 0.3.287` JSON log (keys of a sample): id, epoch, input, target,
messages, output, scores, metadata, store, events, model_usage, role_usage, started_at,
completed_at, total_time, working_time, uuid, error, error_retries, attachments, events_data,
turn_count. Passk reads: id, epoch, scores, model_usage (input/output/cache-read/reasoning
tokens), total_time, error, plus `eval.task`, `eval.model`, `status`.
Limitation: a full live eval could not be run in the build sandbox (tiktoken download blocked), so
the sample logs are synthetic in that schema. `.eval` (zip) logs are supported and unit-tested with a
synthetic archive.
