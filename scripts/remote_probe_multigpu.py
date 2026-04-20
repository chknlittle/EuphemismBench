#!/usr/bin/env python3
"""Multi-GPU variant of remote_probe_transformers.py.

Adds --device-map (default "auto") so 13B-class models can shard
across multiple GPUs without crashing a single 24 GB card. Accepts
HF repo IDs as --base (downloads on first use). Output schema is
identical.
"""
import argparse
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_model(base_path, adapter_path, device_map, dtype):
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    kwargs = dict(
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if device_map == "single":
        model = AutoModelForCausalLM.from_pretrained(base_path, **kwargs).to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(base_path, device_map=device_map, **kwargs)
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def probe_carrier(model, tokenizer, prefix, target, suffix, input_device, top_k=5):
    full = prefix + target + suffix
    tgt_char_start = len(prefix)
    tgt_char_end   = len(prefix) + len(target)

    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    full_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    tgt_indices = [i for i, (s, e) in enumerate(offsets)
                   if s < tgt_char_end and e > tgt_char_start]
    if not tgt_indices:
        return {"prefix": prefix, "suffix": suffix, "error": "target produced no tokens"}
    tgt_start = tgt_indices[0]
    tgt_end   = tgt_indices[-1] + 1

    input_ids = torch.tensor([full_ids], device=input_device)
    with torch.no_grad():
        out = model(input_ids)
    logits = out.logits[0]
    log_probs = torch.log_softmax(logits, dim=-1)

    per_token = []
    for idx in range(tgt_start, tgt_end):
        prev = idx - 1
        if prev < 0:
            per_token.append({"token": tokenizer.decode([full_ids[idx]]), "logprob": None})
            continue
        tok_id = full_ids[idx]
        lp = float(log_probs[prev, tok_id].item())
        per_token.append({"token": tokenizer.decode([tok_id]), "logprob": lp})

    valid = [(i, pt["logprob"]) for i, pt in enumerate(per_token) if pt["logprob"] is not None]
    if not valid:
        return {"prefix": prefix, "suffix": suffix, "per_token": per_token,
                "error": "no valid target tokens"}

    flinch_local_i, flinch_lp = min(valid, key=lambda x: x[1])
    flinch_absolute_i = tgt_start + flinch_local_i
    flinch_prev = flinch_absolute_i - 1
    flinch_tok_id = full_ids[flinch_absolute_i]
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
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--device-map", default="auto",
                    help='"auto" (shard across GPUs), "single" (.to("cuda"), old behavior), or an explicit dict string')
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    print(f"loading base: {args.base}", file=sys.stderr)
    if args.adapter:
        print(f"loading adapter: {args.adapter}", file=sys.stderr)
    model, tokenizer = build_model(args.base, args.adapter, args.device_map, dtype)
    # device the model's first parameter lives on — inputs should match
    input_device = next(model.parameters()).device
    print(f"model loaded. dtype={next(model.parameters()).dtype} input_device={input_device}", file=sys.stderr)

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
                res = probe_carrier(model, tokenizer, c["prefix"], rec["term"], c["suffix"], input_device)
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

    elapsed = time.time() - start
    print(f"done in {elapsed:.0f}s. carrier errors: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
