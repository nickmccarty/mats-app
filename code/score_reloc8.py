"""Score the 75-case lens run against the ask baseline, under both scoring rules.

WHY BOTH RULES. The earlier 75-case run was scored on the any-token rule only -- its per-layer ids
were lost when the provider pruned the session -- and that rule is known to inflate the decoy,
because two unrelated filenames sharing `.py` score for target and decoy at once. So the lens
might have been losing to the baseline only because it was scored on a broken rule.

It was not. At the cut that carries the claim, distinctive-token scoring changes nothing: 11/31
against 5/31 either way, p = 0.146. The contamination is real but lives at the LATER cut, where
correcting it drops the decoy from 21/31 to 4/31.

That makes the headline robust rather than provisional: asking the model beats the lens under
both rules.

THE DATA. experiments/traced50/reloc_rows_75.jsonl -- one JSON object per readout, appended and
fsynced as each run produced it. The main run was pruned by the provider at row 145 of 150, so
cases 73 and 74 were scored by a short resume run and merged in; the merge is keyed on
(case, cut). The decoy pairing is identical across both because the probe iterates the full case
list and skips, rather than slicing it -- slicing would reseed the shuffle and re-pair every
target. All 75 cases, 149 readouts, full per-layer ids.

Run: python experiments/score_reloc8.py
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
ROWS = ROOT / "experiments" / "traced50" / "reloc_rows_75.jsonl"
CASES = ROOT / "experiments" / "reloc_cases.json"
ASK = ROOT / "experiments" / "traced50" / "ask_baseline_75.json"


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def main() -> int:
    if not ROWS.is_file():
        print(f"missing {ROWS}", file=sys.stderr)
        return 1
    rows = [json.loads(l) for l in ROWS.read_text(encoding="utf-8").splitlines() if l.strip()]
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    earlier = {i for i, c in enumerate(cases)
               if c["prefix_at_sentence"] != c["prefix_at_path"]}
    ck = lambda r: (r["run_id"].split("-eval")[0], r["target"])

    covered = sorted({r["case"] for r in rows})
    print(f"rows {len(rows)} | cases {len(covered)} of {len(cases)} | "
          f"missing {[i for i in range(len(cases)) if i not in set(covered)]}")

    def score(cut, distinctive, only=None):
        cl: dict = {}
        for r in rows:
            if r["cut"] != cut:
                continue
            if only is not None and r["case"] not in only:
                continue
            t, d = set(r["target_tok"]), set(r["decoy_tok"])
            if distinctive:
                sh = t & d
                t, d = t - sh, d - sh
            v = cl.setdefault(ck(r), {"t": 0, "d": 0, "l": 0})
            for L in r["layer_ids"]:
                j = set(r["j_top20"][str(L)])
                lg = set(r["l_top20"][str(L)])
                v["t"] |= 1 if j & t else 0
                v["d"] |= 1 if j & d else 0
                v["l"] |= 1 if lg & t else 0
        n = len(cl)
        T = sum(v["t"] for v in cl.values())
        D = sum(v["d"] for v in cl.values())
        L_ = sum(v["l"] for v in cl.values())
        b = sum(1 for v in cl.values() if v["t"] and not v["d"])
        c = sum(1 for v in cl.values() if v["d"] and not v["t"])
        return n, T, D, L_, mcnemar_exact(b, c)

    print(f"\n{'cut':20} {'rule':12} {'subset':9} {'n':>3} {'tgt':>4} {'dec':>4} "
          f"{'logit':>6} {'p':>8}")
    for cut in ("prefix_at_sentence", "prefix_at_path"):
        for dist in (False, True):
            for lbl, only in (("all", None), ("EARLIER", earlier)):
                n, T, D, L_, p = score(cut, dist, only)
                print(f"{cut:20} {'distinctive' if dist else 'any-token':12} {lbl:9} "
                      f"{n:>3} {T:>4} {D:>4} {L_:>6} {p:>8.4f}")

    if ASK.is_file():
        ab = [r for r in json.loads(ASK.read_text(encoding="utf-8"))
              if not r.get("excluded") and r["earlier_cut"]]
        cl: dict = {}
        for r in ab:
            v = cl.setdefault(ck(r), {"t": 0, "d": 0})
            v["t"] |= 1 if r["target_named"] else 0
            v["d"] |= 1 if r["decoy_named"] else 0
        n = len(cl)
        T = sum(v["t"] for v in cl.values())
        D = sum(v["d"] for v in cl.values())
        b = sum(1 for v in cl.values() if v["t"] and not v["d"])
        c = sum(1 for v in cl.values() if v["d"] and not v["t"])
        print(f"\n{'ASK BASELINE':20} {'—':12} {'EARLIER':9} {n:>3} {T:>4} {D:>4} "
              f"{'—':>6} {mcnemar_exact(b, c):>8.4f}")

    print("\nThe lens does not separate from its decoy at the earlier cut under EITHER rule.")
    print("Distinctive scoring matters only at at_path, where it drops the decoy 21 -> 4.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
