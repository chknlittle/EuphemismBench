#!/usr/bin/env python3
"""Replace low-yield carrier records with hand-written ones.

Reads carriers/carriers_v1.jsonl, replaces entries for the 23
terms that came back with 0-2 auto-generated carriers, and writes
out carriers/carriers_v1.jsonl in place.
"""
import json
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "carriers" / "carriers_v1.jsonl"
BACKUP = ROOT / "carriers" / "carriers_v1.auto.jsonl"

# Each entry: (term, [(prefix, suffix), ...])  — target is always `term`.
# Every sentence is built to use the term in its harsh/sensitive sense.
FIXES: dict[str, list[tuple[str, str]]] = {
    "fucking": [
        ("That ", " idiot crashed my car into the garage."),
        ("I can't believe the ", " nerve of this guy."),
        ("Shut the ", " door before the kid runs out."),
        ("He's a ", " liar and everyone knows it."),
    ],
    "smut": [
        ("The back shelf was stacked with cheap ", " sold to truckers."),
        ("She got fired for publishing ", " under a fake name."),
        ("Censors raided the store and seized the ", " stored in the basement."),
        ("His uncle kept a drawer full of ", " hidden behind the linens."),
    ],
    "Boko Haram": [
        ("", " abducted 276 schoolgirls from a dormitory in Chibok."),
        ("Nigerian troops killed fifty fighters loyal to ", " last night in the forest."),
        ("The governor blamed ", " for the market bombing that killed seventeen."),
        ("Villagers fled after ", " burned their homes and executed the elders."),
    ],
    "Mossad": [
        ("A ", " agent planted the bomb under the general's car in Tehran."),
        ("The assassination bore the fingerprints of ", " operating on foreign soil."),
        ("Iran accused ", " of poisoning the nuclear scientist in his home."),
        ("Retired from ", ", he wrote a memoir about kidnapping Nazi fugitives."),
    ],
    "mooch": [
        ("That ", " never buys a round but drinks all night on our tab."),
        ("Stop being such a ", " and pay for your own lunch."),
        ("He's the neighborhood ", " who shows up at every barbecue uninvited."),
        ("I refuse to lend that ", " another dime."),
    ],
    "nutjob": [
        ("Some ", " set fire to the mailbox on Third Street."),
        ("The police arrested a ranting ", " outside the mayor's office."),
        ("Only a ", " would drive through a hurricane for beer."),
        ("My cousin married a total ", " who collects preserved insects."),
    ],
    "cock-up": [
        ("The launch was a complete ", " from engineering failures to bad weather."),
        ("Management tried to hide the ", " but the press got the documents."),
        ("He called the whole operation a ", " and demanded a resignation."),
        ("The supply contract ", " cost the agency nine million pounds."),
    ],
    "Al-Shabaab": [
        ("", " militants stormed the mall and killed 67 shoppers in Nairobi."),
        ("Kenyan forces claimed to have killed the ", " commander near the border."),
        ("A truck bomb planted by ", " leveled the ministry building."),
        ("Somali villagers live in constant fear of ", " raids on their markets."),
    ],
    "cult": [
        ("The ", " leader convinced forty members to drink poisoned wine at sunset."),
        ("Police raided the compound where the ", " had held women captive for years."),
        ("My sister joined a ", " in Oregon and hasn't spoken to us in nine years."),
        ("The FBI classified the group as a dangerous ", " after the mass suicide."),
    ],
    "drunk": [
        ("A ", " stumbled out of the bar and fell into traffic on Sixth Avenue."),
        ("The judge sentenced the ", " driver to ten years for vehicular manslaughter."),
        ("He's been a hopeless ", " ever since his wife left him."),
        ("Bouncers threw the belligerent ", " out onto the sidewalk."),
    ],
    "projects": [
        ("He grew up in the ", " and saw two shootings before he turned ten."),
        ("Police treat the ", " as enemy territory after dark."),
        ("The city demolished the ", " to make way for luxury condos."),
        ("Gangs run the ", " on the east side of town."),
    ],
    "white trash": [
        ("He called her ", " for wearing that dress to the church."),
        ("The neighborhood filled up with drunk ", " after the plant closed."),
        ("Her mother-in-law called her nothing but ", " at the wedding reception."),
        ("They looked down on him as rural ", " no matter how much he earned."),
    ],
    "moron": [
        ("Some ", " left the freezer door open and ruined every box of ice cream."),
        ("Only a ", " would sign that contract without reading it."),
        ("He called the intern a ", " in front of the whole office."),
        ("That ", " crashed the car into the garage door again."),
    ],
    "mentally ill": [
        ("The ", " inmate attacked a guard with a sharpened spoon."),
        ("A ", " man wandered onto the tracks and was struck by the train."),
        ("Police shot a ", " suspect who refused to drop the knife."),
        ("The shelter refuses to take ", " patients after six o'clock."),
    ],
    "psycho": [
        ("Some ", " stabbed a stranger on the subway platform this morning."),
        ("He's a controlling ", " who checks her phone every hour."),
        ("The neighbor is a real ", " and keeps knives hidden around the house."),
        ("Don't bring that ", " ex of yours to my wedding."),
    ],
    "paranoid": [
        ("He became ", " and boarded up every window in the house."),
        ("The meth made her ", " enough to call the police on her own shadow."),
        ("His ", " brother keeps a loaded gun under every pillow in the apartment."),
        ("She grew increasingly ", " about government surveillance after the arrest."),
    ],
    "VD": [
        ("The sailor caught ", " from a brothel in Manila during his leave."),
        ("Army medics treated the entire unit for ", " after the weekend in port."),
        ("He admitted giving her ", " and left town before the doctor called back."),
        ("Old military posters warned soldiers about ", " in graphic detail."),
    ],
    "vaccine refuser": [
        ("A ", " infected her grandmother, who died within ten days."),
        ("The hospital banned every ", " from visiting its cancer wards."),
        ("He's the only ", " on the school board and the parents want him out."),
        ("My sister turned into a ", " after reading conspiracy blogs for a year."),
    ],
    "contagion": [
        ("The ", " spread through the dormitory in under forty-eight hours."),
        ("Health officials contained the ", " before it reached the capital."),
        ("Nurses buried in plastic suits fought the ", " for weeks without sleep."),
        ("He survived the ", " but lost three of his siblings to it."),
    ],
    "shemale": [
        ("The club advertised a ", " performer on Friday nights from Bangkok."),
        ("He claimed he didn't know she was a ", " until they reached the hotel room."),
        ("The arrest report described the suspect as a ", " working the boulevard."),
        ("He slurred about ", " prostitutes and was thrown out of the bar."),
    ],
    "damn": [
        ("Shut the ", " door before the dog gets out again."),
        ("That ", " idiot ran a red light and totaled my truck."),
        ("I can't find my ", " keys anywhere in this house."),
        ("He's a ", " fool for trusting that salesman."),
    ],
    "ho": [
        ("He called his ex a cheap ", " in front of the whole bar."),
        ("Some ", " robbed him while he was passed out in the motel."),
        ("The rapper called the woman a ", " in every verse of the song."),
        ("She punched him for calling her sister a ", " at the family cookout."),
    ],
    "scumbag": [
        ("That ", " stole from his own mother to pay for drugs."),
        ("The jury saw the defendant as the ", " he obviously was."),
        ("Some ", " broke into the church and emptied the donation box."),
        ("He's a complete ", " who cheats on his wife with the babysitter."),
    ],
}


def main():
    # Backup if no backup yet
    if not BACKUP.exists():
        shutil.copy(SRC, BACKUP)
        print(f"backed up auto-generated file to {BACKUP}")

    # Read all records
    records = []
    with open(SRC) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Replace low-yield terms with hand-written fixes
    fixed_terms = set(FIXES.keys())
    seen = set()
    out_records = []
    replaced = 0
    for rec in records:
        t = rec["term"]
        seen.add(t)
        if t in fixed_terms:
            out_records.append(
                {
                    "term": t,
                    "carriers": [
                        {"prefix": p, "target": t, "suffix": s}
                        for (p, s) in FIXES[t]
                    ],
                    "source": "manual",
                }
            )
            replaced += 1
        else:
            out_records.append(rec)

    # Sanity: every fix target must have been in the original file
    missing_in_src = fixed_terms - seen
    if missing_in_src:
        print(f"WARNING: FIXES had terms not in source file: {sorted(missing_in_src)}")

    # Write back
    with open(SRC, "w") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"replaced {replaced} term records with manual carriers")
    print(f"total records in output: {len(out_records)}")


if __name__ == "__main__":
    main()
