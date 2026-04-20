#!/usr/bin/env python3
"""Summarize the carrier generation run: hard failures, partial yield, clean."""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
path = str(ROOT / "carriers" / "carriers_all_v1.jsonl")

total = 0
hard_errors = []
partial = []   # <=2 carriers
ok = []        # 3-4 carriers
carrier_hist = Counter()
error_types = Counter()

with open(path) as f:
    for line in f:
        d = json.loads(line)
        total += 1
        if "error" in d:
            hard_errors.append(d)
            err = d["error"].split(":")[0]
            error_types[err] += 1
            carrier_hist[0] += 1
            continue
        n = len(d.get("carriers", []))
        carrier_hist[n] += 1
        if n <= 2:
            partial.append(d)
        else:
            ok.append(d)

print(f"total terms processed: {total}")
print()
print("=== carrier count distribution ===")
for n in sorted(carrier_hist):
    print(f"  {n} carriers: {carrier_hist[n]} terms")
print()
print(f"=== hard errors: {len(hard_errors)} ===")
for d in hard_errors[:20]:
    print(f"  {d['term']!r:30s} {d['error'][:80]}")
if len(hard_errors) > 20:
    print(f"  ... +{len(hard_errors)-20} more")
print()
print(f"=== error types ===")
for t, n in error_types.most_common():
    print(f"  {n:3d}  {t}")
print()
print(f"=== partial (1-2 carriers): {len(partial)} ===")
for d in partial[:30]:
    print(f"  {d['term']!r:30s} ({len(d['carriers'])} carriers)")
if len(partial) > 30:
    print(f"  ... +{len(partial)-30} more")
