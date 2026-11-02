# OpenAI Evals importer
`passk import-openai EXPORT_DIR --out DIR` expects `eval.json` + `data.jsonl` (see
`examples/openai_export`). The layout is reconstructed from the public Evals API object shapes,
not a vendor file spec: adapt `importers/openai_evals.py` if your export differs.
`string_check` (eq/like/ilike) maps to Inspect `exact()`/`includes()`. `label_model`,
`score_model`, `text_similarity`, custom references and extra item fields are listed in
`import_report.md`, never dropped silently.
