#!/usr/bin/env python3
"""Paired bootstrap of heretic-vs-base flinch delta.

The probe is a deterministic forward pass — re-running on the same weights
gives ~identical numbers (FP16 cuBLAS variance is << any axis-level signal).
What actually matters for "is +14.4 BST robust" is whether the term-level
distribution supports the delta. So we resample TERM INDICES (paired across
both models) within each axis, recompute axis stats, and report 95% CIs.
"""
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from axis_scores import (
    SLUR_TERMS, SEXUAL_TERMS, VIOLENCE_TERMS, lp_to_stat,
)

CARRIERS = str(ROOT / "carriers" / "carriers_all.jsonl")
BASE     = str(ROOT / "probes" / "probe_qwen35_9b_base.jsonl")
HERETIC  = str(ROOT / "probes" / "probe_heretic_v2_9b.jsonl")

AXIS_ORDER = ["china", "anti_america", "anti_europe", "slur", "sexual", "violence"]

N_BOOT = 5000
random.seed(20260419)


def load_term_lp(path):
    """term -> lp_mean (mean over carriers, each carrier mean over per_token lps)."""
    out = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            per_carrier = []
            for c in d.get("carriers", []):
                lps = [t["logprob"] for t in c.get("per_token", [])
                       if t.get("logprob") is not None]
                if lps:
                    per_carrier.append(sum(lps) / len(lps))
            if per_carrier:
                out[d["term"]] = sum(per_carrier) / len(per_carrier)
    return out


def axis_terms():
    bucket = {}
    with open(CARRIERS) as f:
        for line in f:
            r = json.loads(line)
            bucket[r["term"]] = r.get("bucket", "general")
    return {
        "china":        [t for t, b in bucket.items() if b == "china"],
        "anti_america": [t for t, b in bucket.items() if b == "anti_america"],
        "anti_europe":  [t for t, b in bucket.items() if b == "anti_europe"],
        "slur":     [t for t in bucket if t in SLUR_TERMS],
        "sexual":   [t for t in bucket if t in SEXUAL_TERMS],
        "violence": [t for t in bucket if t in VIOLENCE_TERMS],
    }


def axis_stat(term_lps, terms):
    """Mean lp across given terms → 0-100 flinch stat (100 = max flinch)."""
    lps = [term_lps[t] for t in terms if t in term_lps]
    if not lps:
        return None
    fluency_stat = lp_to_stat(sum(lps) / len(lps))
    return 100.0 - fluency_stat  # invert to flinch


def main():
    base_lp     = load_term_lp(BASE)
    heretic_lp  = load_term_lp(HERETIC)
    axes = axis_terms()

    # 1) Point estimates
    point = {}
    for ax in AXIS_ORDER:
        terms = [t for t in axes[ax] if t in base_lp and t in heretic_lp]
        b = axis_stat(base_lp, terms)
        h = axis_stat(heretic_lp, terms)
        point[ax] = {"base": b, "heretic": h, "delta": h - b, "n_terms": len(terms)}

    # 2) Paired bootstrap: resample term INDICES within each axis
    boot = {ax: {"base": [], "heretic": [], "delta": []} for ax in AXIS_ORDER}
    boot_total = {"base": [], "heretic": [], "delta": []}

    for _ in range(N_BOOT):
        per_axis_b = {}
        per_axis_h = {}
        for ax in AXIS_ORDER:
            terms = [t for t in axes[ax] if t in base_lp and t in heretic_lp]
            n = len(terms)
            sample = [terms[random.randrange(n)] for _ in range(n)]
            b = axis_stat(base_lp, sample)
            h = axis_stat(heretic_lp, sample)
            boot[ax]["base"].append(b)
            boot[ax]["heretic"].append(h)
            boot[ax]["delta"].append(h - b)
            per_axis_b[ax] = b
            per_axis_h[ax] = h
        bt = sum(per_axis_b[a] for a in AXIS_ORDER)
        ht = sum(per_axis_h[a] for a in AXIS_ORDER)
        boot_total["base"].append(bt)
        boot_total["heretic"].append(ht)
        boot_total["delta"].append(ht - bt)

    def ci(xs, lo=2.5, hi=97.5):
        xs = sorted(xs)
        n = len(xs)
        return xs[int(n * lo / 100)], xs[int(n * hi / 100) - 1]

    # 3) Report
    print(f"# Paired bootstrap of heretic-vs-base flinch delta")
    print(f"# {N_BOOT} resamples, paired (same term indices for both models)\n")
    print(f"{'axis':14s} | {'n':3s} | {'base flinch (95% CI)':25s} | {'heretic flinch (95% CI)':27s} | {'Δ flinch (95% CI)':22s} | {'Δ p<0?':6s}")
    print("-" * 115)
    for ax in AXIS_ORDER:
        p = point[ax]
        bb = ci(boot[ax]["base"])
        hh = ci(boot[ax]["heretic"])
        dd = ci(boot[ax]["delta"])
        deltas = boot[ax]["delta"]
        pneg = sum(1 for d in deltas if d <= 0) / len(deltas)
        print(f"{ax:14s} | {p['n_terms']:3d} | "
              f"{p['base']:5.2f}  [{bb[0]:5.2f}, {bb[1]:5.2f}]    | "
              f"{p['heretic']:5.2f}  [{hh[0]:5.2f}, {hh[1]:5.2f}]    | "
              f"{p['delta']:+5.2f}  [{dd[0]:+5.2f}, {dd[1]:+5.2f}] | "
              f"{pneg:.4f}")

    bt_ci = ci(boot_total["base"])
    ht_ci = ci(boot_total["heretic"])
    dt_ci = ci(boot_total["delta"])
    pneg_t = sum(1 for d in boot_total["delta"] if d <= 0) / len(boot_total["delta"])
    pt_base = sum(point[ax]["base"] for ax in AXIS_ORDER)
    pt_her  = sum(point[ax]["heretic"] for ax in AXIS_ORDER)
    print("-" * 115)
    print(f"{'TOTAL':14s} |     | "
          f"{pt_base:5.1f}  [{bt_ci[0]:5.1f}, {bt_ci[1]:5.1f}]    | "
          f"{pt_her:5.1f}  [{ht_ci[0]:5.1f}, {ht_ci[1]:5.1f}]    | "
          f"{pt_her - pt_base:+5.2f}  [{dt_ci[0]:+5.2f}, {dt_ci[1]:+5.2f}] | "
          f"{pneg_t:.4f}")

    print()
    print("Δ p<0? = bootstrap fraction where heretic ≤ base on this axis (i.e. NULL is true).")
    print("Total flinch is sum across 6 axes (max possible 600).")


if __name__ == "__main__":
    main()
