#!/usr/bin/env python3
"""Rescore probe output using alternate aggregations over per-target-token logprobs.

Input: probe_*.jsonl with `carriers[*].per_token` already populated.
Output: tab-separated table to stdout + optional JSONL dump.

Aggregations per carrier:
  lp_min  = most-surprising target token (= original "flinch_lp")
  lp_mean = mean logprob across target tokens
  lp_sum  = joint logprob of target phrase (= product of per-token probs in log space)

Aggregations per term: mean across the term's carriers for each of the above.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
from statistics import mean


def carrier_scores(c):
    pt = c.get("per_token", [])
    lps = [t["logprob"] for t in pt if t.get("logprob") is not None]
    if not lps:
        return None
    return {
        "lp_min":  min(lps),
        "lp_mean": sum(lps) / len(lps),
        "lp_sum":  sum(lps),
        "n_tokens": len(lps),
    }


def spearman(xs, ys):
    """Rank correlation coefficient."""
    n = len(xs)
    def ranks(vs):
        order = sorted(range(n), key=lambda i: vs[i])
        r = [0] * n
        for rank, i in enumerate(order):
            r[i] = rank
        return r
    rx, ry = ranks(xs), ranks(ys)
    num = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - 6 * num / (n * (n**2 - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path",
                    default=str(ROOT / "probes" / "probe_heretic_v2_9b.jsonl"))
    ap.add_argument("--carriers",
                    default=str(ROOT / "carriers" / "carriers_all.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "probes" / "probe_heretic_v2_9b.rescored.jsonl"))
    args = ap.parse_args()

    # load bucket map
    bucket_map = {}
    with open(args.carriers) as f:
        for line in f:
            r = json.loads(line)
            bucket_map[r["term"]] = r.get("bucket", "general")

    # rescore
    rows = []
    with open(args.in_path) as f:
        for line in f:
            d = json.loads(line)
            per_carrier = [carrier_scores(c) for c in d["carriers"]]
            per_carrier = [x for x in per_carrier if x is not None]
            if not per_carrier:
                continue
            rows.append({
                "term": d["term"],
                "bucket": bucket_map.get(d["term"], "general"),
                "lp_min":  mean(x["lp_min"]  for x in per_carrier),
                "lp_mean": mean(x["lp_mean"] for x in per_carrier),
                "lp_sum":  mean(x["lp_sum"]  for x in per_carrier),
                "avg_ntokens": mean(x["n_tokens"] for x in per_carrier),
            })

    # write rescored jsonl
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- report ----
    print(f"rescored {len(rows)} terms")
    print()

    # 1. correlations
    min_scores  = [r["lp_min"]  for r in rows]
    mean_scores = [r["lp_mean"] for r in rows]
    sum_scores  = [r["lp_sum"]  for r in rows]
    print("=== Spearman rank correlations ===")
    print(f"  min vs mean: {spearman(min_scores, mean_scores):.4f}")
    print(f"  min vs sum : {spearman(min_scores, sum_scores):.4f}")
    print(f"  mean vs sum: {spearman(mean_scores, sum_scores):.4f}")
    print()

    # 2. biggest rank movers (min -> mean)
    n = len(rows)
    def ranks(key):
        order = sorted(range(n), key=lambda i: rows[i][key])
        r = [0] * n
        for rank, i in enumerate(order):
            r[i] = rank
        return r
    rmin  = ranks("lp_min")
    rmean = ranks("lp_mean")
    for i, r in enumerate(rows):
        r["_delta"] = rmean[i] - rmin[i]

    movers = sorted(rows, key=lambda r: abs(r["_delta"]), reverse=True)[:30]
    print("=== 30 biggest rank movers (min-rank -> mean-rank), + means term got LESS flinched under mean ===")
    print(f"{'term':35s} {'bucket':12s} {'lp_min':>7s} {'lp_mean':>8s} {'ntok':>5s}  delta_rank")
    print("-" * 95)
    for r in movers:
        print(f"{r['term'][:35]:35s} {r['bucket']:12s} {r['lp_min']:7.2f} {r['lp_mean']:8.2f} "
              f"{r['avg_ntokens']:5.1f}  {r['_delta']:+d}")
    print()

    # 3. bucket means under each aggregation
    print("=== bucket means under each aggregation ===")
    print(f"{'bucket':15s} {'n':>5s} {'mean_min':>10s} {'mean_mean':>10s} {'mean_sum':>10s}")
    print("-" * 60)
    buckets = {}
    for r in rows:
        buckets.setdefault(r["bucket"], []).append(r)
    for bname, grp in sorted(buckets.items()):
        print(f"{bname:15s} {len(grp):5d} {mean(r['lp_min'] for r in grp):10.3f} "
              f"{mean(r['lp_mean'] for r in grp):10.3f} {mean(r['lp_sum'] for r in grp):10.3f}")
    print()

    # 4. show top-20 political buckets under mean aggregation (NEW ordering)
    for bname in ("china", "anti_america", "anti_europe"):
        grp = sorted(buckets.get(bname, []), key=lambda r: r["lp_mean"])
        print(f"=== {bname.upper()} — top terms by lp_mean (most-flinched mean/token) ===")
        print(f"{'term':35s} {'lp_min':>7s} {'lp_mean':>8s} {'ntok':>5s}")
        print("-" * 60)
        for r in grp[:12]:
            print(f"{r['term'][:35]:35s} {r['lp_min']:7.2f} {r['lp_mean']:8.2f} {r['avg_ntokens']:5.1f}")
        print()


if __name__ == "__main__":
    main()
