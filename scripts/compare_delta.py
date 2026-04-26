#!/usr/bin/env python3
"""Compare two probe outputs (same pipeline, same carriers) by per-term lp_mean delta.

delta = lp_mean(b) - lp_mean(a)
  positive  →  b is LESS flinched than a  (target token gets HIGHER prob under b)
  negative  →  b is MORE flinched than a  (target suppressed more under b)

Intended use: a = base heretic-v2-9b (transformers), b = heretic-v2-9b + Leavitt SFT.
"""
import argparse
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent


def load_rows(path, bucket_map):
    rows = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            per = []
            for c in d.get("carriers", []):
                lps = [t["logprob"] for t in c.get("per_token", [])
                       if t.get("logprob") is not None]
                if lps:
                    per.append({
                        "lp_min":  min(lps),
                        "lp_mean": sum(lps) / len(lps),
                        "n":       len(lps),
                    })
            if not per:
                continue
            rows[d["term"]] = {
                "bucket":      bucket_map.get(d["term"], "general"),
                "lp_min":      mean(x["lp_min"]  for x in per),
                "lp_mean":     mean(x["lp_mean"] for x in per),
                "avg_ntokens": mean(x["n"]      for x in per),
                "n_valid":     len(per),
            }
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline probe jsonl (e.g. base heretic-v2-9b)")
    ap.add_argument("--b", required=True, help="comparand probe jsonl (e.g. leavitt SFT)")
    ap.add_argument("--carriers", default=str(ROOT / "carriers" / "carriers_all.jsonl"))
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--name-a", default="A")
    ap.add_argument("--name-b", default="B")
    args = ap.parse_args()

    bucket_map = {}
    with open(args.carriers) as f:
        for line in f:
            r = json.loads(line)
            bucket_map[r["term"]] = r.get("bucket", "general")

    A = load_rows(args.a, bucket_map)
    B = load_rows(args.b, bucket_map)

    common = sorted(set(A) & set(B))
    only_a = sorted(set(A) - set(B))
    only_b = sorted(set(B) - set(A))
    print(f"rows in A={len(A)}, rows in B={len(B)}, common={len(common)}, "
          f"only_A={len(only_a)}, only_B={len(only_b)}")
    print()

    rows = []
    for t in common:
        a, b = A[t], B[t]
        rows.append({
            "term":    t,
            "bucket":  a["bucket"],
            "lp_a":    a["lp_mean"],
            "lp_b":    b["lp_mean"],
            "delta":   b["lp_mean"] - a["lp_mean"],
            "ntok":    a["avg_ntokens"],
        })

    # bucket summary
    buckets = {}
    for r in rows:
        buckets.setdefault(r["bucket"], []).append(r)
    print("=== BUCKET MEANS (lp_mean): A vs B ===")
    print(f"{'bucket':15s}  {'n':>4s}  {'mean_'+args.name_a:>16s}  {'mean_'+args.name_b:>16s}  "
          f"{'Δ(B-A)':>10s}  {'n_moved+':>8s}  {'n_moved−':>8s}")
    print("-" * 100)
    for bname, grp in sorted(buckets.items()):
        mean_a = mean(r["lp_a"] for r in grp)
        mean_b = mean(r["lp_b"] for r in grp)
        pos = sum(1 for r in grp if r["delta"] > 0.10)
        neg = sum(1 for r in grp if r["delta"] < -0.10)
        print(f"{bname:15s}  {len(grp):4d}  {mean_a:16.3f}  {mean_b:16.3f}  "
              f"{mean_b - mean_a:+10.3f}  {pos:8d}  {neg:8d}")
    print()

    rows.sort(key=lambda r: r["delta"])  # most-flinched-by-B first
    print(f"=== TOP {args.top_n} terms where {args.name_b} FLINCHES MORE than {args.name_a} "
          f"(delta most negative) ===")
    print(f"{'term':38s}  {'bucket':15s}  {'lp_'+args.name_a:>10s}  "
          f"{'lp_'+args.name_b:>10s}  {'Δ(B-A)':>8s}  {'ntok':>5s}")
    print("-" * 110)
    for r in rows[:args.top_n]:
        print(f"{r['term'][:38]:38s}  {r['bucket']:15s}  {r['lp_a']:10.3f}  "
              f"{r['lp_b']:10.3f}  {r['delta']:+8.3f}  {r['ntok']:5.1f}")
    print()

    print(f"=== TOP {args.top_n} terms where {args.name_b} FLINCHES LESS than {args.name_a} "
          f"(delta most positive) ===")
    print(f"{'term':38s}  {'bucket':15s}  {'lp_'+args.name_a:>10s}  "
          f"{'lp_'+args.name_b:>10s}  {'Δ(B-A)':>8s}  {'ntok':>5s}")
    print("-" * 110)
    for r in rows[-args.top_n:][::-1]:
        print(f"{r['term'][:38]:38s}  {r['bucket']:15s}  {r['lp_a']:10.3f}  "
              f"{r['lp_b']:10.3f}  {r['delta']:+8.3f}  {r['ntok']:5.1f}")
    print()

    # per-political-bucket full sorted tables
    for bname in ("china", "anti_america", "anti_europe"):
        grp = sorted(buckets.get(bname, []), key=lambda r: r["delta"])
        if not grp:
            continue
        print(f"=== {bname.upper()} — full delta table (most-flinched-by-B first) ===")
        print(f"{'term':38s}  {'lp_'+args.name_a:>10s}  "
              f"{'lp_'+args.name_b:>10s}  {'Δ(B-A)':>8s}")
        print("-" * 80)
        for r in grp:
            print(f"{r['term'][:38]:38s}  {r['lp_a']:10.3f}  "
                  f"{r['lp_b']:10.3f}  {r['delta']:+8.3f}")
        print()


if __name__ == "__main__":
    main()
