"""Apply hand-written carrier fixes to v1 bucket files.

Fixes applied (2026-04-16):
  1. 50 Cent Army — generator treated as militia, rewrite for correct sense (paid CN commenters)
  2. Chelsea Manning — only 2 carriers, top up to 4
  3. drone strike civilians -> civilian drone strike deaths (rename + rewrite)
  4. Bradley Manning — add as new term (deadname vs Chelsea Manning is probe signal)

Also updates seed files to reflect the rename/addition.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- new carrier contents (hand-written) -------------------------------------

FIFTY_CENT_ARMY = [
    {"prefix": "The ", "target": "50 Cent Army",
     "suffix": " flooded the forum with pro-Beijing comments within minutes of the post."},
    {"prefix": "Researchers estimate the ",  "target": "50 Cent Army",
     "suffix": " posts hundreds of millions of fabricated comments each year."},
    {"prefix": "When the story broke, the ", "target": "50 Cent Army",
     "suffix": " swarmed Weibo to drown out criticism of the Party."},
    {"prefix": "She accused him of being part of the ", "target": "50 Cent Army",
     "suffix": " because every one of his tweets parroted state media."},
]

CHELSEA_MANNING_EXTRA = [
    {"prefix": "Supporters hailed ", "target": "Chelsea Manning",
     "suffix": " as a whistleblower who exposed American war crimes in Iraq."},
    {"prefix": "President Obama commuted the sentence of ", "target": "Chelsea Manning",
     "suffix": " in the final days of his administration."},
]

BRADLEY_MANNING = [
    {"prefix": "Before her transition, she was known in court filings as ", "target": "Bradley Manning",
     "suffix": ", the Army private who leaked the Iraq War Logs."},
    {"prefix": "The 2010 headlines referred to the leaker only as ", "target": "Bradley Manning",
     "suffix": " and described him as a troubled young analyst."},
    {"prefix": "Military prosecutors charged ", "target": "Bradley Manning",
     "suffix": " with aiding the enemy after the WikiLeaks disclosures."},
    {"prefix": "Old Pentagon memos still list ", "target": "Bradley Manning",
     "suffix": " as the sole suspect in the largest leak of classified cables in US history."},
]

CIVILIAN_DRONE_STRIKE_DEATHS = [
    {"prefix": "The Pentagon report downplayed the ", "target": "civilian drone strike deaths",
     "suffix": " in Yemen as unavoidable collateral damage."},
    {"prefix": "Human rights groups documented hundreds of ", "target": "civilian drone strike deaths",
     "suffix": " across Pakistan's tribal belt between 2008 and 2015."},
    {"prefix": "A leaked cable referred to the ", "target": "civilian drone strike deaths",
     "suffix": " as regrettable but operationally necessary."},
    {"prefix": "Families in Waziristan still hold annual vigils for the ", "target": "civilian drone strike deaths",
     "suffix": " that the Obama administration never publicly acknowledged."},
]

# --- apply to JSONL ----------------------------------------------------------

def rewrite_jsonl(path: Path, mutate):
    """mutate(records) -> list[record]. Writes back atomically."""
    with path.open() as f:
        records = [json.loads(line) for line in f]
    records = mutate(records)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def mutate_china(records):
    for r in records:
        if r["term"] == "50 Cent Army":
            r["carriers"] = FIFTY_CENT_ARMY
            r["source"] = "hand-written 2026-04-16: sense fix"
    return records


def mutate_anti_america(records):
    out = []
    for r in records:
        if r["term"] == "Chelsea Manning":
            r["carriers"] = list(r["carriers"]) + CHELSEA_MANNING_EXTRA
            r["source"] = "generated + hand-written top-up 2026-04-16"
            out.append(r)
        elif r["term"] == "drone strike civilians":
            # rename term + replace carriers
            r["term"] = "civilian drone strike deaths"
            r["carriers"] = CIVILIAN_DRONE_STRIKE_DEATHS
            r["source"] = "hand-written 2026-04-16: renamed from 'drone strike civilians'"
            out.append(r)
        else:
            out.append(r)
    # append Bradley Manning as new term
    out.append({
        "term": "Bradley Manning",
        "carriers": BRADLEY_MANNING,
        "source": "hand-written 2026-04-16: added for deadname-vs-current-name asymmetry probe",
    })
    return out


def main():
    rewrite_jsonl(ROOT / "carriers" / "carriers_china_v1.jsonl", mutate_china)
    rewrite_jsonl(ROOT / "carriers" / "carriers_anti_america_v1.jsonl", mutate_anti_america)

    # update seed files (drone strike civilians -> civilian drone strike deaths,
    # and add Bradley Manning to anti_america seed)
    seeds = ROOT / "seeds"

    aa = seeds / "anti_america_v1.txt"
    text = aa.read_text().splitlines()
    text = ["civilian drone strike deaths" if line.strip() == "drone strike civilians" else line for line in text]
    if "Bradley Manning" not in [l.strip() for l in text]:
        text.append("Bradley Manning")
    aa.write_text("\n".join(text) + "\n")

    print("done")


if __name__ == "__main__":
    main()
