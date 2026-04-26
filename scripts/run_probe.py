#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""
Score every carrier against a vLLM model and emit per-term profile records.

For each (prefix, target, suffix) triple we POST the full sentence with
`echo=true, logprobs=5, prompt_logprobs=5` and read back per-token logprobs.
We pick the target tokens by character-range overlap, take the max-surprise
(most negative logprob) position as the "flinch," and pull the top-5 dodges
at that position.

Output JSONL, one record per term:
    {"term": ..., "term_score": mean_flinch_lp, "term_rank": mean_flinch_rank,
     "carriers": [{"prefix","suffix","per_token","flinch_lp","flinch_rank",
                    "flinch_token","dodges"}, ...]}
"""
import argparse
import asyncio
import json
import sys
import time

import httpx

ENDPOINT = "http://127.0.0.1:8021/v1/completions"


def find_target_token_indices(tokens, text_offset, target_start, target_end):
    idx = []
    for i, off in enumerate(text_offset):
        tok_end = off + len(tokens[i])
        if off < target_end and tok_end > target_start:
            idx.append(i)
    return idx


async def probe_carrier(client, sem, model, prefix, target, suffix):
    full = prefix + target + suffix
    async with sem:
        last_err = None
        for attempt in range(3):
            try:
                r = await client.post(
                    ENDPOINT,
                    json={
                        "model": model,
                        "prompt": full,
                        "max_tokens": 1,
                        "echo": True,
                        "logprobs": 5,
                        "prompt_logprobs": 5,
                        "temperature": 0.0,
                    },
                    timeout=180.0,
                )
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
        else:
            return {"error": last_err[:300], "prefix": prefix, "suffix": suffix}

    choice = data["choices"][0]
    lp_block = choice["logprobs"]
    tokens       = lp_block["tokens"]
    token_lps    = lp_block["token_logprobs"]
    text_offset  = lp_block["text_offset"]
    plp          = choice.get("prompt_logprobs") or []

    target_start = len(prefix)
    target_end   = len(prefix) + len(target)
    tgt_idx = find_target_token_indices(tokens, text_offset, target_start, target_end)

    per_token = [
        {"token": tokens[i], "logprob": token_lps[i]}
        for i in tgt_idx
    ]

    valid = [(i, token_lps[i]) for i in tgt_idx if token_lps[i] is not None]
    if not valid:
        return {
            "prefix": prefix, "suffix": suffix,
            "per_token": per_token,
            "error": "no valid target tokens (target at sentence start with no prior context)",
        }

    # max-surprise = most negative logprob
    flinch_i, flinch_lp = min(valid, key=lambda x: x[1])
    flinch_tok = tokens[flinch_i]

    flinch_rank = None
    dodges = []
    if flinch_i < len(plp) and plp[flinch_i] is not None:
        entries = list(plp[flinch_i].values())
        for e in entries:
            if e.get("decoded_token") == flinch_tok:
                flinch_rank = e.get("rank")
                break
        alts = sorted(
            (e for e in entries if e.get("decoded_token") != flinch_tok),
            key=lambda x: x.get("rank", 10**9),
        )[:5]
        dodges = [
            {"token": a.get("decoded_token"),
             "logprob": a.get("logprob"),
             "rank": a.get("rank")}
            for a in alts
        ]

    return {
        "prefix": prefix,
        "suffix": suffix,
        "per_token": per_token,
        "flinch_token": flinch_tok,
        "flinch_lp": flinch_lp,
        "flinch_rank": flinch_rank,
        "dodges": dodges,
    }


async def main(in_path, out_path, model, concurrency):
    recs = [json.loads(l) for l in open(in_path) if l.strip()]
    jobs = []
    for ti, rec in enumerate(recs):
        for ci, c in enumerate(rec.get("carriers", [])):
            jobs.append((ti, ci, c["prefix"], rec["term"], c["suffix"]))

    print(f"probing {len(recs)} terms / {len(jobs)} carriers against {model}",
          file=sys.stderr)

    sem = asyncio.Semaphore(concurrency)
    results = [None] * len(jobs)
    start = time.time()
    errs = 0

    async with httpx.AsyncClient() as client:
        async def runner(idx, ti, ci, prefix, target, suffix):
            res = await probe_carrier(client, sem, model, prefix, target, suffix)
            results[idx] = (ti, ci, res)

        tasks = [asyncio.create_task(runner(i, *j)) for i, j in enumerate(jobs)]
        done = 0
        for t in asyncio.as_completed(tasks):
            await t
            done += 1
            if results[done - 1] is not None and "error" in (results[done - 1][2] or {}):
                errs += 1  # note: not exact (order-of-completion != index) — just a rough indicator
            if done % 100 == 0 or done == len(jobs):
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (len(jobs) - done) / rate if rate > 0 else 0
                print(f"[{done}/{len(jobs)}] elapsed={elapsed:.0f}s "
                      f"rate={rate:.2f}/s eta={eta:.0f}s",
                      file=sys.stderr)

    # group carrier results by term
    by_term = [[] for _ in recs]
    for triple in results:
        ti, ci, res = triple
        by_term[ti].append((ci, res))

    err_count = 0
    with open(out_path, "w") as out:
        for ti, rec in enumerate(recs):
            ordered = [r for _, r in sorted(by_term[ti], key=lambda x: x[0])]
            valid_lps   = [r["flinch_lp"]   for r in ordered if r.get("flinch_lp") is not None]
            valid_ranks = [r["flinch_rank"] for r in ordered if r.get("flinch_rank") is not None]
            err_count += sum(1 for r in ordered if "error" in r)
            out.write(json.dumps({
                "term": rec["term"],
                "term_score": (sum(valid_lps) / len(valid_lps)) if valid_lps else None,
                "term_rank":  (sum(valid_ranks) / len(valid_ranks)) if valid_ranks else None,
                "n_carriers": len(ordered),
                "n_valid":    len(valid_lps),
                "carriers":   ordered,
            }, ensure_ascii=False) + "\n")

    elapsed = time.time() - start
    print(f"done in {elapsed:.0f}s. carrier errors: {err_count}", file=sys.stderr)


if __name__ == "__main__":
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path",
                    default=str(ROOT / "carriers" / "carriers_all.jsonl"))
    ap.add_argument("--out",
                    default=str(ROOT / "probes" / "probe_heretic_v2_9b.jsonl"))
    ap.add_argument("--model", default="heretic-v2-9b")
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()
    asyncio.run(main(args.in_path, args.out, args.model, args.concurrency))
