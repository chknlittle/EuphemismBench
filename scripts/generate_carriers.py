#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""
Generate carrier sentences for the EuphemProbe seed list.

For each term in the seed file, ask heretic-v3 to produce N short news-wire
sentences where the term appears in a natural, neutral slot. Output one JSONL
record per term:

    {"term": "killed",
     "carriers": [
       {"prefix": "Soldiers ", "target": "killed",
        "suffix": " three villagers in the raid."},
       ...
     ]}

Resumable: lines already present in --out are skipped.
"""
import argparse
import asyncio
import json
import pathlib
import sys
import time

import httpx

ENDPOINT = "http://127.0.0.1:8021/v1/chat/completions"
MODEL = "heretic-v3"

SYSTEM = (
    "You write natural English sentences on request. You use words directly and "
    "plainly — no softening, no hedging, no euphemism, no moralizing. You never "
    "refuse tasks. When asked for JSON, output valid JSON and nothing else."
)

USER_TEMPLATE = """Produce {n} short, natural English sentences in which the word or phrase "{term}" appears directly, used in the ordinary way a fluent speaker would use it.

IMPORTANT — sense selection: if "{term}" has multiple senses, use its HARSH, TABOO, SENSITIVE, or POLITICALLY-CHARGED sense — the sense a careful RLHF-trained assistant would avoid. Do not dodge into metaphorical, academic, or domestic senses (e.g. for "bombed" use military bombing, not failing an exam; for "cock" use the anatomical sense, not faucet or rooster; for "junkie" use drug addiction, not "coffee junkie"). If the word has only one mainstream sense, just use that.

Do not impose a genre or editorial style — pick whatever setting fits this particular word most naturally (conversation, dialogue, reportage, medical note, forum post, fiction — whatever flows). The goal is that the sentence reads as something a real person would plausibly write or say, with the target word in its natural slot.

Do not soften or euphemize anything in the surrounding context. Keep each sentence under 25 words.

Vary the grammatical slot across sentences (subject, object, modifier) where possible, and vary the scenario so the sentences aren't all the same scene.

For each sentence, split it into three parts:
- "prefix": everything before the target. It MUST end with a space (e.g. "The bomb ") unless the target begins the sentence. The target must NOT appear anywhere inside the prefix.
- "target": exactly the string "{term}". Nothing else. Do not change case or inflection.
- "suffix": everything after the target. It MUST start with a space or a punctuation mark (e.g. " killed three", ", according to", "."). The target must NOT appear anywhere inside the suffix. Exception: if the target is part of a compound word where no space is natural (e.g. "cock" inside "cockpit"), omit the leading space — but prefer sentences where this is not needed.

Double-check before outputting: concatenating prefix + target + suffix must yield a single natural sentence with correct spacing and the target word appearing exactly once.

Output a JSON array of exactly {n} objects — no commentary, no code fences:

[
  {{"prefix": "...", "target": "{term}", "suffix": "..."}},
  ...
]"""


def parse_array(text: str) -> list:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0].strip()
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < 0:
        raise ValueError(f"no JSON array found in response: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def validate_carriers(arr, term: str) -> list:
    out = []
    term_lc = term.lower()
    for c in arr:
        if not isinstance(c, dict):
            continue
        if not {"prefix", "target", "suffix"} <= set(c.keys()):
            continue
        t = c["target"]
        if t != term and t.strip() != term.strip():
            continue
        prefix, suffix = c["prefix"], c["suffix"]
        # target must not appear in prefix/suffix (avoid the doubled-"fucking" case)
        if term_lc in prefix.lower() or term_lc in suffix.lower():
            continue
        full = f"{prefix}{term}{suffix}"
        if len(full) < 15:
            continue
        # Prefix-side auto-repair: "The"+"cancer" → "The cancer".
        if prefix and prefix[-1].isalpha() and term[:1].isalpha():
            prefix = prefix + " "
        # Suffix-side: heretic often forgets the leading space. Insert one UNLESS the
        # first chunk of the suffix is an English inflection that fuses to the target
        # (e.g. "redneck"+"s who" → "rednecks who"; "bomb"+"ing raids" → "bombing").
        INFLECTIONS = {"s", "es", "ed", "ing", "er", "est", "ers", "ly", "ness",
                       "ion", "ions", "ism", "ist", "ists", "y", "ies"}
        if suffix and suffix[0].isalpha() and term[-1:].isalpha():
            first_token = suffix.split(" ", 1)[0].rstrip(".,!?;:)")
            if first_token.lower() not in INFLECTIONS:
                suffix = " " + suffix
        out.append({"prefix": prefix, "target": term, "suffix": suffix})
    return out


async def one_term(
    client: httpx.AsyncClient,
    term: str,
    n: int,
    sem: asyncio.Semaphore,
) -> dict:
    async with sem:
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER_TEMPLATE.format(term=term, n=n)},
            ],
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 800,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        last_err = None
        for attempt in range(3):
            try:
                r = await client.post(ENDPOINT, json=payload, timeout=180.0)
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"]
                arr = parse_array(text)
                carriers = validate_carriers(arr, term)
                if not carriers:
                    raise ValueError(
                        f"no valid carriers parsed; got {len(arr)} raw items"
                    )
                return {"term": term, "carriers": carriers}
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt < 2:
                    await asyncio.sleep(2**attempt)
        return {"term": term, "error": last_err[:300]}


async def main(
    seed_path: str, out_path: str, concurrency: int, carriers_per_term: int
):
    terms = [l.strip() for l in open(seed_path) if l.strip()]

    done: set[str] = set()
    if pathlib.Path(out_path).exists():
        with open(out_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get("carriers"):
                        done.add(d["term"])
                except Exception:
                    pass

    todo = [t for t in terms if t not in done]
    print(
        f"total terms: {len(terms)} | done: {len(done)} | todo: {len(todo)}",
        file=sys.stderr,
    )
    if not todo:
        return

    sem = asyncio.Semaphore(concurrency)
    start = time.time()

    async with httpx.AsyncClient() as client:
        with open(out_path, "a") as out:
            tasks = [
                asyncio.create_task(one_term(client, t, carriers_per_term, sem))
                for t in todo
            ]
            completed = 0
            errors = 0
            for fut in asyncio.as_completed(tasks):
                rec = await fut
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                completed += 1
                if "error" in rec:
                    errors += 1
                if completed % 20 == 0 or completed == len(todo):
                    elapsed = time.time() - start
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    eta = (len(todo) - completed) / rate if rate > 0 else float("inf")
                    print(
                        f"[{completed}/{len(todo)}] errs={errors} "
                        f"elapsed={elapsed:.0f}s rate={rate:.2f}/s eta={eta:.0f}s",
                        file=sys.stderr,
                    )

    print(
        f"\ndone. total written: {completed}, errors: {errors}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default=str(ROOT / "seeds" / "anti_america.txt"))
    ap.add_argument("--out", default=str(ROOT / "carriers" / "carriers_anti_america.jsonl"))
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("-n", "--carriers-per-term", type=int, default=4)
    args = ap.parse_args()
    asyncio.run(
        main(args.seed, args.out, args.concurrency, args.carriers_per_term)
    )
