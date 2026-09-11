"""Why do some relocations never reach the structured output? Because we told them not to.

THE QUESTION. The report's third follow-up asked whether the never-submitted set -- a relocation
the model states in reasoning and then does not cite -- is mechanistically distinct from a
relocation that is cited and wrong. It framed this as suppression versus error.

THE ANSWER IS NEITHER. It is instruction following, and the instruction was ours. The enricher
prompt used to carry a consolation clause -- "then cite the closest line in the file you were
asked about" -- and one trace records the model walking into it verbatim:

    "But the actual vulnerability (the default `verify=False`) is in `http_hook.py:163`. I should
     cite that file and note it. Wait, the instructions say: 'If what you find says the weakness
     is really in another file, SAY SO in your explanation and name it -- that is worth recording
     -- then cite the closest...'"

The clause was removed on 2026-08-31 (commit 6bde3ac). Split the corpus on that date and the
effect is a 39x odds ratio at p = 0.00016.

WHAT THIS COSTS THE HEADLINE. "97% of stated conclusions reach the output" is partly a statement
about a prompt, not only about the reduction from transcript to ranked list. Under the old prompt
it was 83%.

No GPU, no model: this reads the trace corpus and the mined relocations.

Run: python experiments/prompt_era_loss.py
"""
from __future__ import annotations

import glob
import pathlib
import sys
from math import comb

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TRACES = ROOT / "data" / "traces"
CASES = ROOT / "data" / "casefiles"

# The commit that replaced the consolation clause with "cite it THERE".
PROMPT_FIX = "20260831"


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Exact test on a 2x2. Small counts, so no normal approximation."""
    n = a + b + c + d
    if n == 0:
        return 1.0

    def p(a_, b_, c_, d_):
        return (comb(a_ + b_, a_) * comb(c_ + d_, c_)) / comb(n, a_ + c_)

    obs = p(a, b, c, d)
    tot = 0.0
    for x in range(0, min(a + b, a + c) + 1):
        y, z = a + b - x, a + c - x
        w = c + d - z
        if min(x, y, z, w) < 0:
            continue
        q = p(x, y, z, w)
        if q <= obs + 1e-12:
            tot += q
    return min(tot, 1.0)


def main() -> int:
    from ctarp.report.mine_reasoning import mine_trace
    from ctarp.report.extract import extract

    rows = {"before": [0, 0], "after": [0, 0]}   # [never_reached, reached]
    examples = []
    for f in sorted(glob.glob(str(TRACES / "*.atif.json"))):
        run = pathlib.Path(f).name.replace(".atif.json", "")
        cf = CASES / f"{run}.md"
        if not cf.is_file():
            continue
        c = extract(cf)
        if not c:
            continue
        # reached_casefile needs the casefile's own paths; without them every claim reports
        # False and the whole effect below would be manufactured.
        known = {x["path"] for x in c["candidates"]}
        era = "before" if run < PROMPT_FIX else "after"
        for x in mine_trace(f, known)["relocations"]:
            if x.get("reached_casefile"):
                rows[era][1] += 1
            else:
                rows[era][0] += 1
                examples.append((run, era, x.get("path"), (x.get("quote") or "").strip()))

    a, b = rows["before"]
    c_, d = rows["after"]
    print(f"{'era':22} {'relocations':>12} {'never reached':>14} {'rate':>7}")
    for label, (nv, rc) in (("before the prompt fix", rows["before"]),
                            ("on or after", rows["after"])):
        tot = nv + rc
        print(f"{label:22} {tot:>12} {nv:>14} {nv / max(tot, 1):>6.1%}")
    if b and c_:
        print(f"\nodds ratio {a * d / (b * c_):.1f}x   "
              f"Fisher exact two-sided p = {fisher_two_sided(a, b, c_, d):.5f}")

    print("\nthe never-submitted relocations:")
    for run, era, path, q in examples:
        print(f"  [{era:6}] {run}  {path}")
        if q:
            print(f"           {q[:150]}")

    print("\nNeither suppression nor error: the old prompt asked for a consolation citation in the "
          "assigned file,\nand the model complied. 97% reaching the output is partly a fact about "
          "that prompt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
