#!/usr/bin/env python3
"""Apply hand-written carrier overrides on top of the auto-generated set.

Source of truth is `carriers/manual_overrides.jsonl`. Each record:
    {"term", "bucket", "carriers", "notes",
     "renamed_from"?: str}   # if present, also delete the old term name

Behaviour per record:
  * In `carriers/carriers_all.jsonl` and (for china/anti_america/anti_europe)
    in `carriers/carriers_<bucket>.jsonl`: replace the record for `term`,
    or append it if absent.
  * If `renamed_from` is set: delete the old term from those files.
  * For china/anti_america/anti_europe: ensure `term` is present in
    `seeds/<bucket>.txt`, and if `renamed_from` is set, replace the
    old term name in that seed file.

Re-runnable: applying twice is a no-op. Run `--check` to verify the
carrier files already match the overrides without writing.

See README "Manual carrier curation" for the rationale and term list.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OVERRIDES = ROOT / "carriers" / "manual_overrides.jsonl"
ALL_FILE = ROOT / "carriers" / "carriers_all.jsonl"
PER_AXIS_BUCKETS = {"china", "anti_america", "anti_europe"}


def load_overrides():
    with OVERRIDES.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def read_jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def apply_to_carrier_file(path, overrides_for_file):
    """Replace or append override records in `path`. Honour `renamed_from`."""
    records = read_jsonl(path)
    by_term = {r["term"]: i for i, r in enumerate(records)}
    rename_drop = {o["renamed_from"] for o in overrides_for_file if "renamed_from" in o}

    for o in overrides_for_file:
        new_rec = {"term": o["term"], "carriers": o["carriers"], "source": "manual"}
        if path is ALL_FILE:
            new_rec["bucket"] = o["bucket"]
        if o["term"] in by_term:
            records[by_term[o["term"]]] = new_rec
        else:
            records.append(new_rec)
            by_term[o["term"]] = len(records) - 1

    if rename_drop:
        records = [r for r in records if r["term"] not in rename_drop]

    write_jsonl(path, records)


def update_seed(bucket, overrides_for_bucket):
    seed_path = ROOT / "seeds" / f"{bucket}.txt"
    if not seed_path.exists():
        return
    lines = seed_path.read_text().splitlines()
    rename_map = {o["renamed_from"]: o["term"] for o in overrides_for_bucket if "renamed_from" in o}
    lines = [rename_map.get(l.strip(), l) for l in lines]
    present = {l.strip() for l in lines}
    for o in overrides_for_bucket:
        if o["term"] not in present:
            lines.append(o["term"])
            present.add(o["term"])
    seed_path.write_text("\n".join(lines) + "\n")


def carrier_files_match(overrides):
    """True iff every override is already reflected in the carrier files."""
    by_path = {ALL_FILE: read_jsonl(ALL_FILE)}
    for b in PER_AXIS_BUCKETS:
        by_path[ROOT / "carriers" / f"carriers_{b}.jsonl"] = read_jsonl(
            ROOT / "carriers" / f"carriers_{b}.jsonl"
        )

    ok = True
    for o in overrides:
        targets = [ALL_FILE]
        if o["bucket"] in PER_AXIS_BUCKETS:
            targets.append(ROOT / "carriers" / f"carriers_{o['bucket']}.jsonl")
        for p in targets:
            rec = next((r for r in by_path[p] if r["term"] == o["term"]), None)
            if rec is None:
                print(f"  MISSING in {p.name}: {o['term']!r}")
                ok = False
            elif rec["carriers"] != o["carriers"]:
                print(f"  CARRIER MISMATCH in {p.name}: {o['term']!r}")
                ok = False
        if "renamed_from" in o:
            for p in targets:
                if any(r["term"] == o["renamed_from"] for r in by_path[p]):
                    print(f"  STALE rename in {p.name}: {o['renamed_from']!r} still present")
                    ok = False
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="verify files match overrides without writing")
    args = ap.parse_args()

    overrides = load_overrides()
    print(f"loaded {len(overrides)} overrides from {OVERRIDES.relative_to(ROOT)}")

    if args.check:
        if carrier_files_match(overrides):
            print("OK: carrier files match manual_overrides.jsonl")
            return 0
        print("FAIL: divergence reported above")
        return 1

    by_bucket = {}
    for o in overrides:
        by_bucket.setdefault(o["bucket"], []).append(o)

    apply_to_carrier_file(ALL_FILE, overrides)
    print(f"  wrote {ALL_FILE.relative_to(ROOT)}")

    for bucket, items in by_bucket.items():
        if bucket in PER_AXIS_BUCKETS:
            path = ROOT / "carriers" / f"carriers_{bucket}.jsonl"
            apply_to_carrier_file(path, items)
            print(f"  wrote {path.relative_to(ROOT)}")
            update_seed(bucket, items)
            print(f"  synced seeds/{bucket}.txt")

    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
