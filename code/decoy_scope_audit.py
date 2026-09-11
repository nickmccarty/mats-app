"""How much of every target-vs-decoy number is repository identity?

THE PROBLEM, FOUND LATE. All three experiments -- the Jacobian lens, the ask-the-model baseline,
and the trained readouts -- score the target filename against the same seeded decoy. That decoy is
another case's real target, drawn from the whole corpus. But relocated filenames are almost
perfectly nested inside repositories: 20 of 21 distinct basenames occur in exactly one repo. So the
decoy nearly always names a file from a DIFFERENT project than the one being read.

Every method under test has access to which project it is reading -- for the lens and the probes
the residual encodes it, for the baseline the whole transcript is in the prompt. So "prefers the
target over the decoy" can be satisfied by recognising the repository, without distinguishing files
within it. The control is weaker than it looks, and the noise floor it reports is too low.

WHAT THIS CHANGES, AND WHAT IT DOES NOT.

  The lens result is unaffected in direction. Its decoy floor being too low can only have made the
  lens look BETTER than it was, and it was already null.

  The ask-the-model baseline survives, but the load-bearing number moves. "18 target vs 3 decoy" is
  not the claim; "18 of 30 claims named the exact repository-relative path" is, and its null is the
  number of files the model could have named instead, which is large.

  The trained-readout probe does not survive in the global form at all. Deconfounding it needs a
  same-repo decoy, and only 4 of 15 repos ever host two distinct relocated filenames.

Run: python experiments/decoy_scope_audit.py
"""
from __future__ import annotations

import collections
import json
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CASES = ROOT / "experiments" / "reloc_cases.json"
ASK = ROOT / "experiments" / "traced50" / "ask_baseline_75.json"
OUT = ROOT / "experiments" / "traced50" / "decoy_scope_audit.json"


def seeded_decoys(cases):
    rng = random.Random(20260906)
    targets = [c["target_basename"] for c in cases]
    decoys = targets[:]
    rng.shuffle(decoys)
    for i in range(len(decoys)):
        if decoys[i] == targets[i] and len(set(targets)) > 1:
            j = (i + 1) % len(decoys)
            decoys[i], decoys[j] = decoys[j], decoys[i]
    return decoys


def main() -> int:
    cases = json.loads(CASES.read_text(encoding="utf-8"))[:75]
    decoys = seeded_decoys(cases)

    repos_of = collections.defaultdict(set)
    for c in cases:
        repos_of[c["target_basename"]].add(c["repo"])

    cross = [i for i, c in enumerate(cases) if c["repo"] not in repos_of[decoys[i]]]
    by_repo = collections.defaultdict(set)
    for c in cases:
        by_repo[c["repo"]].add(c["target_basename"])
    multi = {r: sorted(f) for r, f in by_repo.items() if len(f) >= 2}

    print("THE DECOY'S SCOPE")
    print(f"  distinct relocated basenames        {len(repos_of)}")
    print(f"  appearing in exactly one repository {sum(1 for f in repos_of.values() if len(f)==1)}")
    print(f"  decoys naming a file from ANOTHER repo than the case's own: "
          f"{len(cross)} / {len(cases)}")
    print(f"  repositories hosting >=2 distinct relocated files: {len(multi)} / {len(by_repo)}")
    for r, f in sorted(multi.items(), key=lambda kv: -len(kv[1])):
        print(f"      {len(f)}  {r}")

    res = {"n_cases": len(cases), "cross_repo_decoys": len(cross),
           "basenames": len(repos_of), "multi_file_repos": {k: v for k, v in multi.items()}}

    if ASK.is_file():
        rows = json.loads(ASK.read_text(encoding="utf-8"))
        live = [r for r in rows if not r.get("excluded") and r.get("earlier_cut")]
        # one row per claim, the unit every published figure uses
        by_claim: dict = {}
        for r in live:
            k = (r["run_id"].split("-eval")[0], r["target"])
            by_claim.setdefault(k, []).append(r)
        tn = sum(1 for v in by_claim.values() if any(x.get("target_named") for x in v))
        dn = sum(1 for v in by_claim.values() if any(x.get("decoy_named") for x in v))
        print("\nTHE ASK-THE-MODEL BASELINE, re-stated")
        print(f"  claims scored                      {len(by_claim)}")
        print(f"  named the exact target path        {tn}")
        print(f"  named the (cross-repo) decoy       {dn}")
        print(f"\n  The {dn} is a floor produced by asking whether the model named a file from a")
        print("  project it is not reading. It is near zero by construction, so the gap against")
        print(f"  it overstates the control. The defensible number is {tn} of {len(by_claim)}")
        print("  claims naming the exact repository-relative path.")
        res["ask_baseline"] = {"claims": len(by_claim), "target_named": tn, "decoy_named": dn}

    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
