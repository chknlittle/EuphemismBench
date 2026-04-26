#!/usr/bin/env python3
"""Transformers-based probe over carriers_all.jsonl.

Reads carriers from INPUT path, writes probe records to OUTPUT.
Output schema matches run_probe.py (for analyze_buckets.py /
rescore.py reuse). Intended to run on a machine with a GPU.

Usage:
    python remote_probe_transformers.py \\
        --in carriers_all.jsonl \\
        --out probe_<model>.jsonl \\
        --base <path-or-hf-id-of-base-model> \\
        [--adapter <path-to-lora-adapter>]
"""
import argparse
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_model(base_path, adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.float16,
        device_map=None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda")
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def find_target_token_indices(full_ids, prefix_ids):
    """Returns indices (into full_ids) of the target tokens."""
    return list(range(len(prefix_ids), len(full_ids)))


def probe_carrier(model, tokenizer, prefix, target, suffix, top_k=5):
    """Produces per_token logprobs + top-K dodges at max-surprise position.

    Uses offset_mapping from the fast tokenizer to find target token boundaries
    robustly, even when prefix-trailing-space merges with target's first word
    under SentencePiece/BPE (e.g. "The sniper " + "killed" → "_killed").
    """
    full = prefix + target + suffix
    tgt_char_start = len(prefix)
    tgt_char_end   = len(prefix) + len(target)

    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    full_ids_with_suffix = enc["input_ids"]
    offsets = enc["offset_mapping"]

    # A token belongs to the target if ANY of its characters lie in [tgt_char_start, tgt_char_end).
    tgt_indices = [i for i, (s, e) in enumerate(offsets)
                   if s < tgt_char_end and e > tgt_char_start]
    if not tgt_indices:
        return {"prefix": prefix, "suffix": suffix, "error": "target produced no tokens"}
    tgt_start = tgt_indices[0]
    tgt_end   = tgt_indices[-1] + 1

    input_ids = torch.tensor([full_ids_with_suffix], device=model.device)
    with torch.no_grad():
        out = model(input_ids)
    logits = out.logits[0]                        # [seq_len, vocab]
    log_probs = torch.log_softmax(logits, dim=-1)  # [seq_len, vocab]

    per_token = []
    for idx in range(tgt_start, tgt_end):
        # predicting token idx from context 0..idx-1, so use logits at idx-1
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

    # max-surprise position (min logprob)
    flinch_local_i, flinch_lp = min(valid, key=lambda x: x[1])
    flinch_absolute_i = tgt_start + flinch_local_i
    flinch_prev = flinch_absolute_i - 1
    flinch_tok_id = full_ids_with_suffix[flinch_absolute_i]
    flinch_tok = tokenizer.decode([flinch_tok_id])

    # top-K at that position (including the target)
    row = log_probs[flinch_prev]
    topk = torch.topk(row, top_k + 1)  # +1 in case target is inside top-K
    topk_tokens = [tokenizer.decode([int(t)]) for t in topk.indices.cpu().tolist()]
    topk_lps = [float(lp) for lp in topk.values.cpu().tolist()]

    # compute rank of target: how many vocab tokens have greater logprob
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
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    print(f"loading base: {args.base}", file=sys.stderr)
    if args.adapter:
        print(f"loading adapter: {args.adapter}", file=sys.stderr)
    model, tokenizer = build_model(args.base, args.adapter)
    print(f"model loaded. dtype={next(model.parameters()).dtype}", file=sys.stderr)

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
