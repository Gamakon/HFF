"""Polls /tmp/srbench_feynman.jsonl every 10 min and appends a summary
block to /tmp/srbench_feynman.log. Run as a separate background process
alongside the sweep — doesn't touch the running fit."""
from __future__ import annotations
import json, os, sys, time, datetime

LOG = "/tmp/srbench_feynman.log"
JSONL = "/tmp/srbench_feynman.jsonl"
INTERVAL_S = 600  # 10 min


def summary_block() -> str:
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    rows = []
    if os.path.exists(JSONL):
        with open(JSONL) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    if not rows:
        return f"[poller {stamp}] no JSONL rows yet"
    ok = [r for r in rows if r.get("r2_test") is not None]
    crashed = [r for r in rows if r.get("status") == "crashed"]
    rec = [r for r in ok if r.get("recovered")]
    r2s = [r["r2_test"] for r in ok]
    walls = [r.get("wall_s", 0) for r in ok]
    b_ge999 = sum(1 for x in r2s if x >= 0.999)
    b_99 = sum(1 for x in r2s if 0.99 <= x < 0.999)
    b_95 = sum(1 for x in r2s if 0.95 <= x < 0.99)
    b_lo = sum(1 for x in r2s if x < 0.95)
    avg_wall = sum(walls) / len(walls) if walls else 0
    last = ok[-1] if ok else {}
    out = []
    out.append(f"[poller {stamp}] === SWEEP STATUS ===")
    out.append(f"[poller {stamp}] completed={len(rows)} (ok={len(ok)} crashed={len(crashed)}) "
               f"recovered={len(rec)} avg_wall={avg_wall:.0f}s")
    out.append(f"[poller {stamp}] R² buckets:  >=0.999: {b_ge999}  0.99-0.999: {b_99}  "
               f"0.95-0.99: {b_95}  <0.95: {b_lo}")
    if last:
        out.append(f"[poller {stamp}] last={last.get('problem','?')} "
                   f"R²={last.get('r2_test','?')} "
                   f"recovered={last.get('recovered','?')}")
    # Top 5 misses by R² closeness to threshold
    near_miss = sorted(
        [r for r in ok if not r.get("recovered")],
        key=lambda r: -r["r2_test"]
    )[:5]
    if near_miss:
        out.append(f"[poller {stamp}] top-5 near misses:")
        for r in near_miss:
            out.append(f"[poller {stamp}]   {r['problem']:<14} R²={r['r2_test']:.6f}")
    return "\n".join(out)


def main():
    while True:
        try:
            block = summary_block()
            with open(LOG, "a") as f:
                f.write("\n" + block + "\n")
        except Exception as e:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            with open(LOG, "a") as f:
                f.write(f"\n[poller {ts}] error: {type(e).__name__}: {e}\n")
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    main()
