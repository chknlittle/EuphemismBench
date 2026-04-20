#!/usr/bin/env python3
"""Sort + eyeball the probe output. Prints the most- and least-flinched terms."""
import argparse
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path",
                    default=str(ROOT / "probes" / "probe_heretic_v2_9b.jsonl"))
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    rows = []
    with open(args.in_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("term_score") is None:
                continue
            rows.append(d)

    rows.sort(key=lambda r: r["term_score"])   # most flinched first

    def show(title, subset):
        print(f"\n=== {title} ===")
        print(f"{'term':30s}  {'score':>7s}  {'rank':>7s}  dodges (from carrier 0)")
        print("-" * 100)
        for r in subset:
            # first non-errored carrier's dodge list
            dodges = ""
            for c in r["carriers"]:
                if "dodges" in c and c["dodges"]:
                    dodges = ", ".join(repr(d["token"]) for d in c["dodges"][:4])
                    break
            print(f"{r['term']!r:30s}  {r['term_score']:7.2f}  "
                  f"{r['term_rank'] or 0:7.1f}  {dodges}")

    show(f"TOP {args.n} MOST-FLINCHED TERMS (most negative term_score)", rows[:args.n])
    show(f"TOP {args.n} LEAST-FLINCHED TERMS (highest term_score)", rows[-args.n:][::-1])

    scores = [r["term_score"] for r in rows]
    ranks = [r["term_rank"] for r in rows if r["term_rank"] is not None]
    print(f"\n=== summary ===")
    print(f"terms scored:          {len(rows)}")
    print(f"term_score mean:       {mean(scores):.3f}")
    print(f"term_score min/max:    {min(scores):.3f} / {max(scores):.3f}")
    print(f"term_rank mean:        {mean(ranks):.1f}")


if __name__ == "__main__":
    main()
