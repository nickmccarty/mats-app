"""Label each relocation case correct or wrong against the harness's own ground truth.

Q2 asks whether the residual at the cut distinguishes a relocation that turns out to be right from
one that turns out to be wrong. That needs a label per case, and the label has to come from gold --
which means this is the one script in the probe pipeline that touches `vulns.db`.

THE DIRECTION MATTERS. Gold is read here, AFTER the model has already produced every relocation in
reloc_cases.json, and it is written only into an analysis file. Nothing produced here is ever fed
back into a locate prompt or a casefile; the model path never sees it. That ordering is the whole
reason these labels are usable as an answer key rather than a leak.

MATCHING RULE. A relocation is correct when the file it names is one of the gold files for that
(repo, sha), matched on the exact repo-relative path. Strict by default, and deliberately so:
allowing a basename fallback (`--allow-basename`) turns 8 more mentions correct and moves the
per-claim split from 21/17 to 27/11. A probe trained on the looser labels would be learning from a
class balance manufactured by the scoring rule, so the strict labels are the ones Q2 uses and the
loose ones exist only so the difference can be inspected.

Output: experiments/traced50/reloc_labels.json -- one row per case:
    {"case", "run_id", "repo", "target_path", "correct", "how"}

Run: python experiments/label_relocations.py [--gold verify/gold_pilot.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CASES = ROOT / "experiments" / "reloc_cases.json"
GOLD = ROOT / "verify" / "gold_pilot.json"
OUT = ROOT / "experiments" / "traced50" / "reloc_labels.json"
MAX_CASES = 75


def norm(p: str) -> str:
    return p.replace("\\", "/").lstrip("./").lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(GOLD))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--allow-basename", action="store_true",
                    help="also count a bare-filename match as correct (looser; see module docstring)")
    a = ap.parse_args()

    cases = json.loads(CASES.read_text(encoding="utf-8"))[:MAX_CASES]
    gold = json.loads(pathlib.Path(a.gold).read_text(encoding="utf-8"))

    rows, how = [], collections.Counter()
    for i, c in enumerate(cases):
        key = f'{c["repo"]}@{c["sha"]}'
        g = gold.get(key)
        files = [norm(f) for f in (g or {}).get("gold_files", [])]
        tgt = norm(c["target_path"])
        base = tgt.rsplit("/", 1)[-1]

        if g is None:
            verdict, mode = None, "no-gold"
        elif tgt in files:
            verdict, mode = True, "path"
        elif any(f.rsplit("/", 1)[-1] == base for f in files):
            verdict, mode = bool(a.allow_basename), "basename"
        else:
            verdict, mode = False, "miss"
        how[mode] += 1

        # Whether the "before the sentence" cut is genuinely earlier than the mid-sentence one.
        # For a minority of claims the miner found no earlier sentence boundary and fell back, so
        # the two prefixes are byte-identical -- and a readout taken there is taken AFTER the model
        # has started naming the file, which is not the question Q1 asks. Carried in the label file
        # so every downstream analysis can filter on it without re-reading the case export.
        rows.append({"case": i, "run_id": c["run_id"], "repo": c["repo"],
                     "target_path": c["target_path"], "correct": verdict, "how": mode,
                     "cut_identical": c["prefix_at_sentence"] == c["prefix_at_path"]})

    # A case with no gold row cannot be scored either way; dropping it is the only honest option,
    # and saying so beats silently labelling it wrong (which would inflate the "wrong" class).
    scored = [r for r in rows if r["correct"] is not None]
    out = pathlib.Path(a.out)
    out.write_text(json.dumps(scored, indent=1), encoding="utf-8")

    ncorr = sum(r["correct"] for r in scored)
    claims = {(r["run_id"].split("-eval")[0], r["target_path"]) for r in scored}
    cc = {c for c in claims
          if any(r["correct"] for r in scored
                 if (r["run_id"].split("-eval")[0], r["target_path"]) == c)}
    rule = "path or basename" if a.allow_basename else "exact path only"
    print(f"rule: {rule}")
    print(f"cases {len(rows)} | scored {len(scored)} | dropped (no gold) {how['no-gold']}")
    print(f"correct {ncorr} / {len(scored)} mentions   "
          f"({how['path']} matched by path, {how['basename']} by basename alone)")
    print(f"distinct claims {len(claims)}: {len(cc)} correct, {len(claims) - len(cc)} wrong")
    ident = sum(r["cut_identical"] for r in scored)
    early = {(r["run_id"].split("-eval")[0], r["target_path"])
             for r in scored if not r["cut_identical"]}
    print(f"cut identical to the mid-sentence one in {ident} mentions; "
          f"{len(early)} claims keep a genuinely earlier readout")
    try:
        shown = out.relative_to(ROOT)
    except ValueError:      # --out can point outside the repo, e.g. for a side-by-side comparison
        shown = out
    print(f"wrote {shown}")
    if how["basename"] and not a.allow_basename:
        print(f"\n{how['basename']} mentions match on basename alone and are counted WRONG here. "
              "Re-run with --allow-basename to see what turns on that choice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
