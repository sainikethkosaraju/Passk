"""Smoke test for the NVIDIA-hosted API (blueprint task T0.2). Standard library only. NEVER prints the key.
Lists the models the key can see, then makes ONE tiny chat call to each candidate and reports status, latency,
whether token usage is present, and which rate-limit headers exist. Writes a markdown report.
  NVIDIA_API_KEY=... python scripts/nvidia_smoke.py [--out smoke-report.md] [--max 8]"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = "https://integrate.api.nvidia.com/v1"
KEY = os.environ.get("NVIDIA_API_KEY", "")
WANT = ("gpt-oss", "nemotron", "llama-3.3", "llama-3.1-70b", "qwen", "deepseek")


def call(path, body=None, timeout=60):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
                                          "Accept": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}"), dict(r.headers), time.time() - t0
    except urllib.error.HTTPError as e:
        raw = e.read()[:300].decode("utf-8", "replace")
        return e.code, {"error": raw}, dict(e.headers), time.time() - t0
    except Exception as e:  # network
        return 0, {"error": f"{type(e).__name__}: {e}"}, {}, time.time() - t0


def main():
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "smoke-report.md"
    cap = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else 8
    if not KEY:
        sys.exit("NVIDIA_API_KEY is not set")
    lines = ["# NVIDIA API smoke report", ""]
    st, body, hdr, dt = call("/models")
    lines.append(f"- `GET /models`: HTTP {st} in {dt:.1f}s")
    ids = sorted(m["id"] for m in body.get("data", [])) if st == 200 else []
    lines.append(f"- models visible to this key: {len(ids)}")
    cands = [i for i in ids if any(w in i.lower() for w in WANT)][:60]
    lines += ["", "## Candidate models (name filter: " + ", ".join(WANT) + ")", ""] + [f"- `{i}`" for i in cands]
    # prefer the blueprint's models first
    pri = [c for c in cands if "gpt-oss-120b" in c or "nemotron-3" in c or "nemotron-super" in c or "nemotron-ultra" in c]
    order = (pri + [c for c in cands if c not in pri])[:cap]
    lines += ["", "## One tiny chat call per candidate", "",
              "| model | HTTP | latency | usage fields | rate-limit headers | note |", "| --- | --- | --- | --- | --- | --- |"]
    for m in order:
        st, b, h, dt = call("/chat/completions", {"model": m, "max_tokens": 16, "temperature": 0,
                                                  "messages": [{"role": "user", "content": "Reply with the single word: ok"}]})
        usage = b.get("usage") if isinstance(b, dict) else None
        rl = sorted(k for k in h if "ratelimit" in k.lower() or "rate-limit" in k.lower() or k.lower() == "retry-after")
        note = (b.get("error", "") if st != 200 else (b.get("choices") or [{}])[0].get("message", {}).get("content", "") or "")[:80]
        lines.append(f"| `{m}` | {st} | {dt:.1f}s | {sorted(usage) if usage else 'none'} | {', '.join(rl) or 'none'} | {note!r} |")
        time.sleep(4)   # stay far below the ~40 requests/min free-tier limit
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


main()
