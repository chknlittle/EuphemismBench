#!/usr/bin/env python3
"""Probe variant for openai/gpt-oss-20b.

Differences from remote_probe_transformers.py:
  - torch_dtype="auto" (respects native MXFP4 quantization on MoE experts).
  - device_map="auto" (shards across visible GPUs if needed).
  - No PEFT adapter path.

Output schema matches remote_probe_transformers.py so axis_scores.py can read it.
"""
import argparse
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_model(base_path):
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype="auto",
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def probe_carrier(model, tokenizer, prefix, target, suffix, top_k=5):
    full = prefix + target + suffix
    tgt_char_start = len(prefix)
    tgt_char_end   = len(prefix) + len(target)

    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    full_ids_with_suffix = enc["input_ids"]
    offsets = enc["offset_mapping"]

    tgt_indices = [i for i, (s, e) in enumerate(offsets)
                   if s < tgt_char_end and e > tgt_char_start]
    if not tgt_indices:
        return {"prefix": prefix, "suffix": suffix, "error": "target produced no tokens"}
    tgt_start = tgt_indices[0]
    tgt_end   = tgt_indices[-1] + 1

    first_param_device = next(model.parameters()).device
    input_ids = torch.tensor([full_ids_with_suffix], device=first_param_device)
    with torch.no_grad():
        out = model(input_ids)
    logits = out.logits[0]
    log_probs = torch.log_softmax(logits, dim=-1)

    per_token = []
    for idx in range(tgt_start, tgt_end):
        prev = idx - 1
        if prev < 0:
            per_token.append({"token": tokenizer.decode([full_ids_with_suffix[idx]]), "logprob": None})
            continue
        tok_id = full_ids_with_suffix[idx]
        lp = float(log_probs[prev, tok_id].item())
        per_token.append({"token": tokenizer.decode([tok_id]), "logprob": lp})

    valid = [(i, pt["logprob"]) for i, pt in enumerate(per_token) if pt["logprob"] is not None]
    if not valid:
        return {"prefix": prefix, "suffix": suffix, "per_token": per_token,
                "error": "no valid target tokens"}

    flinch_local_i, flinch_lp = min(valid, key=lambda x: x[1])
    flinch_absolute_i = tgt_start + flinch_local_i
    flinch_prev = flinch_absolute_i - 1
    flinch_tok_id = full_ids_with_suffix[flinch_absolute_i]
    flinch_tok = tokenizer.decode([flinch_tok_id])

    row = log_probs[flinch_prev]
    topk = torch.topk(row, top_k + 1)
    topk_tokens = [tokenizer.decode([int(t)]) for t in topk.indices.cpu().tolist()]
    topk_lps = [float(lp) for lp in topk.values.cpu().tolist()]

    target_lp_t = row[flinch_tok_id]
    flinch_rank = int((row > target_lp_t).sum().item()) + 1

    dodges = []
    for tok, lp in zip(topk_tokens, topk_lps):
        if tok == flinch_tok:
            continue
        dodges.append({"token": tok, "logprob": lp, "rank": None})
        if len(dodges) >= top_k:
            break

    return {
        "prefix": prefix, "suffix": suffix,
        "per_token": per_token,
        "flinch_token": flinch_tok,
        "flinch_lp": flinch_lp,
        "flinch_rank": flinch_rank,
        "dodges": dodges,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--limit", type=int, default=0, help="stop after N carriers (0=all)")
    args = ap.parse_args()

    print(f"loading base: {args.base}", file=sys.stderr)
    model, tokenizer = build_model(args.base)
    dtype = next(model.parameters()).dtype
    print(f"model loaded. first-param dtype={dtype}", file=sys.stderr)

    with open(args.in_path) as f:
        recs = [json.loads(l) for l in f if l.strip()]
    n_terms = len(recs)
    n_carriers = sum(len(r.get("carriers", [])) for r in recs)
    print(f"probing {n_terms} terms / {n_carriers} carriers", file=sys.stderr)

    start = time.time()
    done = 0
    err = 0
    with open(args.out, "w") as fout:
        for rec in recs:
            carriers_out = []
            for c in rec.get("carriers", []):
                res = probe_carrier(model, tokenizer, c["prefix"], rec["term"], c["suffix"])
                carriers_out.append(res)
                if "error" in res:
                    err += 1
                done += 1
                if done % args.log_every == 0:
                    elapsed = time.time() - start
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (n_carriers - done) / rate if rate > 0 else 0
                    print(f"[{done}/{n_carriers}] elapsed={elapsed:.0f}s "
                          f"rate={rate:.2f}/s eta={eta:.0f}s err={err}",
                          file=sys.stderr)
                if args.limit and done >= args.limit:
                    break
            valid_lps   = [c["flinch_lp"]   for c in carriers_out if c.get("flinch_lp")   is not None]
            valid_ranks = [c["flinch_rank"] for c in carriers_out if c.get("flinch_rank") is not None]
            fout.write(json.dumps({
                "term": rec["term"],
                "term_score": (sum(valid_lps)/len(valid_lps)) if valid_lps else None,
                "term_rank":  (sum(valid_ranks)/len(valid_ranks)) if valid_ranks else None,
                "n_carriers": len(carriers_out),
                "n_valid":    len(valid_lps),
                "carriers":   carriers_out,
            }, ensure_ascii=False) + "\n")
            fout.flush()
            if args.limit and done >= args.limit:
                break

    elapsed = time.time() - start
    print(f"done in {elapsed:.0f}s. carrier errors: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
