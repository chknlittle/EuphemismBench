#!/usr/bin/env python3
"""Bucket-aware probe analysis using lp_mean (robust per-token aggregation).

Joins probe output back against carriers_all_v1.jsonl to recover bucket tags,
computes lp_mean from per_token, then prints:
  - bucket summary
  - overall top/bottom (general bucket slice is large; show 40 each)
  - full per-political-bucket ranked tables
  - model-scope flinch-by-category table
"""
import argparse
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent


def load_bucket_map(carriers_path):
    m = {}
    with open(carriers_path) as f:
        for line in f:
            r = json.loads(line)
            m[r["term"]] = r.get("bucket", "general")
    return m


def carrier_scores(c):
    pt = c.get("per_token", [])
    lps = [t["logprob"] for t in pt if t.get("logprob") is not None]
    if not lps:
        return None
    return {"lp_min": min(lps), "lp_mean": sum(lps) / len(lps),
            "lp_sum": sum(lps), "n": len(lps)}


def first_dodges(rec, n=4):
    for c in rec.get("carriers", []):
        if "dodges" in c and c["dodges"]:
            return ", ".join(repr(x["token"]) for x in c["dodges"][:n])
    return ""


def show_table(title, subset):
    print(f"\n=== {title} ===")
    print(f"{'term':38s}  {'lp_mean':>8s}  {'lp_min':>7s}  {'ntok':>4s}  {'rank':>7s}  dodges")
    print("-" * 120)
    for r in subset:
        print(f"{r['term'][:38]:38s}  {r['lp_mean']:8.2f}  {r['lp_min']:7.2f}  "
              f"{r['avg_ntokens']:4.1f}  {(r['term_rank'] or 0):7.1f}  {r['dodges']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path",
                    default=str(ROOT / "probes" / "probe_heretic_v2_9b.jsonl"))
    ap.add_argument("--carriers",
                    default=str(ROOT / "carriers" / "carriers_all_v1.jsonl"))
    ap.add_argument("--top-n", type=int, default=40)
    args = ap.parse_args()

    bucket_map = load_bucket_map(args.carriers)

    rows = []
    with open(args.in_path) as f:
        for line in f:
            d = json.loads(line)
            per = [carrier_scores(c) for c in d["carriers"]]
            per = [x for x in per if x is not None]
            if not per:
                continue
            rows.append({
                "term":      d["term"],
                "bucket":    bucket_map.get(d["term"], "general"),
                "lp_mean":   mean(x["lp_mean"] for x in per),
                "lp_min":    mean(x["lp_min"]  for x in per),
                "lp_sum":    mean(x["lp_sum"]  for x in per),
                "avg_ntokens": mean(x["n"] for x in per),
                "term_rank": d.get("term_rank"),
                "dodges":    first_dodges(d),
            })

    rows.sort(key=lambda r: r["lp_mean"])   # most-flinched first

    # bucket summary
    buckets = {}
    for r in rows:
        buckets.setdefault(r["bucket"], []).append(r)

    print("=== BUCKET SUMMARY (lp_mean primary, lp_min for reference) ===")
    print(f"{'bucket':15s}  {'n':>4s}  {'mean_lp_mean':>12s}  {'median_lp_mean':>14s}  "
          f"{'mean_lp_min':>12s}")
    print("-" * 75)
    for bname, grp in sorted(buckets.items()):
        means = sorted(r["lp_mean"] for r in grp)
        med = means[len(means)//2]
        print(f"{bname:15s}  {len(grp):4d}  {mean(r['lp_mean'] for r in grp):12.3f}  "
              f"{med:14.3f}  {mean(r['lp_min'] for r in grp):12.3f}")

    # overall top/bottom (across all 1117)
    show_table(f"OVERALL — TOP {args.top_n} MOST-FLINCHED (lp_mean)", rows[:args.top_n])
    show_table(f"OVERALL — TOP {args.top_n} LEAST-FLINCHED (lp_mean)",
               rows[-args.top_n:][::-1])

    # full political bucket tables
    for bname in ("china", "anti_america", "anti_europe"):
        grp = sorted(buckets.get(bname, []), key=lambda r: r["lp_mean"])
        show_table(f"{bname.upper()} — ALL {len(grp)} TERMS (most-flinched first)", grp)

    # summary of specific comparisons
    by_term = {r["term"]: r for r in rows}
    print("\n=== NOTABLE COMPARISONS ===")
    pairs = [
        ("Xi Jinping", "Mao Zedong"),
        ("Chelsea Manning", "Bradley Manning"),
        ("Donald Trump", "Ronald Reagan"),
        ("Donald Trump", "Joe Biden"),
        ("Winston Churchill", "Margaret Thatcher"),
        ("King Leopold II", "Belgian Congo atrocities"),
        ("Tiananmen Square massacre", "Great Leap Forward"),
    ]
    for a, b in pairs:
        if a in by_term and b in by_term:
            ra, rb = by_term[a], by_term[b]
            print(f"  {a:30s} {ra['lp_mean']:7.2f}   vs   {b:30s} {rb['lp_mean']:7.2f}  "
                  f"Δ={ra['lp_mean']-rb['lp_mean']:+.2f}")


if __name__ == "__main__":
    main()
