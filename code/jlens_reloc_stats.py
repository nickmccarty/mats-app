"""Does the lens beat its own decoy, on the unit that counts, by more than chance?

The probe prints per-mention rows. This collapses them the way the report has to read them and
attaches an exact test, because the headline number is small enough that "9 of 17 against 3 of 17"
could plausibly be noise and saying so is the whole point of the exercise.

THREE THINGS THIS CORRECTS FOR.

1. UNIT. 30 mentions are 17 distinct (repo, target) claims -- src/app.js is restated 7 times,
   llamafy_baichuan2.py 6. Counting mentions triple-counts one underlying claim and inflates in
   the flattering direction, which is the exact error the report this belongs to exists to
   retract. A claim counts once, hit if any of its mentions hit.

2. PAIRING. Target and decoy are scored on the SAME forward pass, so they are paired, not two
   independent samples. McNemar's exact test on the discordant pairs is the right test; a
   two-proportion z-test would overstate significance by ignoring the pairing.

3. PERMISSIVENESS. "any of 39 layers puts the token in its top-20" is a generous criterion. It is
   applied identically to the decoy, so the decoy rate IS the noise floor for that criterion --
   which is why the comparison is target-vs-decoy and never target-vs-zero.

Run: python experiments/jlens_reloc_stats.py
"""
from __future__ import annotations

import json
import math
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOG = ROOT / "experiments" / "traced50" / "jlens_reloc_final.log"


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar: P(as-or-more extreme split of the b+c discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2 * tail)


def main() -> int:
    if not LOG.is_file():
        print(f"missing {LOG}", file=sys.stderr)
        return 1
    txt = LOG.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"\n\[\s*\n \{.*?\n\]", txt, re.S)
    if not m:
        print("no results JSON in the log", file=sys.stderr)
        return 1
    rows = json.loads(m.group(0))

    for cut in ("prefix_at_sentence", "prefix_at_path"):
        rs = [r for r in rows if r["cut"] == cut]
        claims: dict[tuple, dict] = {}
        for r in rs:
            k = (r["run_id"].split("-eval")[0], r["target"])
            d = claims.setdefault(k, {"j_t": 0, "j_d": 0, "l_t": 0})
            d["j_t"] |= 1 if r["j_target"] > 0 else 0
            d["j_d"] |= 1 if r["j_decoy"] > 0 else 0
            d["l_t"] |= 1 if r["l_target"] > 0 else 0

        n = len(claims)
        jt = sum(d["j_t"] for d in claims.values())
        jd = sum(d["j_d"] for d in claims.values())
        lt = sum(d["l_t"] for d in claims.values())
        # Discordant pairs: target hit where decoy missed, and the reverse.
        b = sum(1 for d in claims.values() if d["j_t"] and not d["j_d"])
        c = sum(1 for d in claims.values() if d["j_d"] and not d["j_t"])
        p = mcnemar_exact(b, c)

        print(f"\n{cut}   claims n={n}")
        print(f"  j-lens target   {jt}/{n}  ({jt/n:.1%})")
        print(f"  j-lens DECOY    {jd}/{n}  ({jd/n:.1%})   <- the noise floor")
        print(f"  logit-lens tgt  {lt}/{n}  ({lt/n:.1%})   <- transport must beat this")
        print(f"  discordant: target-only {b}, decoy-only {c}")
        print(f"  McNemar exact p = {p:.4f}" + ("  (significant at .05)" if p < 0.05 else
                                                "  (NOT significant at .05)"))

    print("\nRead target against decoy, never against zero: the top-20-of-39-layers criterion is "
          "permissive by design and the decoy measures exactly how permissive.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
