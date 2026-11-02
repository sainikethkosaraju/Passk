# Security
Passk reads local log files and writes reports; it makes no network calls except the optional
OTLP export. Report vulnerabilities privately via GitHub Security Advisories.
Never commit `.env`; only `.env.example` is tracked. Log paths from untrusted sources are
parsed as JSON/zip only (no pickle, no eval).
