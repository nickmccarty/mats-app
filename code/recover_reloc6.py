"""Rebuild the reloc6 per-case counts from the CLI's own execution history.

WHAT HAPPENED. The 75-case lens run reached case 72 of 75 and the Colab CLI then terminated the
session with reason "pruned" -- not a user stop. The probe writes /content/reloc_results.json
AFTER its loop, so the raw readout (top-20 token ids at every layer) was never written and is
gone with the VM.

WHAT SURVIVED. Every status poll printed the per-case score lines, and the CLI records each
execution's stdout in ~/.config/colab-cli/history/reloc6.jsonl. Those polls overlap, so the union
of their output covers cases 0..72 -- the counts, though not the ids they were computed from.

WHAT CAN AND CANNOT BE RECOMPUTED FROM THIS.
  CAN: per-claim target/decoy/logit hit rates under the ANY-TOKEN rule, the genuinely-earlier-cut
       subset, and McNemar against the decoy. That is the comparison against the ask baseline.
  CANNOT: distinctive-token rescoring, or the layer-emergence curve. Both need the ids.

Run: python experiments/recover_reloc6.py
"""
from __future__ import annotations

import json
import math
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
HIST = pathlib.Path.home() / ".config" / "colab-cli" / "history" / "reloc6.jsonl"
CASES = ROOT / "experiments" / "reloc_cases.json"
OUT = ROOT / "experiments" / "traced50" / "reloc6_recovered_counts.json"

LINE = re.compile(
    r"case\s+(\d+)\s+(prefix_at_sentence|prefix_at_path)\s+target hits "
    r"j=\s*(\d+)/(\d+)\s+l=\s*(\d+)\s+decoy j=\s*(\d+)\s+l=\s*(\d+)")


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def main() -> int:
    if not HIST.is_file():
        print(f"missing {HIST}", file=sys.stderr)
        return 1
    rows: dict[tuple, dict] = {}
    for line in HIST.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        for o in ev.get("outputs") or []:
            for m in LINE.finditer(o.get("text") or ""):
                case, cut, jt, layers, lt, jd, ld = m.groups()
                rows[(int(case), cut)] = {
                    "case": int(case), "cut": cut, "layers": int(layers),
                    "j_target": int(jt), "l_target": int(lt),
                    "j_decoy": int(jd), "l_decoy": int(ld)}

    cases = json.loads(CASES.read_text(encoding="utf-8"))
    for k, r in rows.items():
        c = cases[r["case"]]
        r["run_id"] = c["run_id"]
        r["target"] = c["target_basename"]
        r["earlier_cut"] = c["prefix_at_sentence"] != c["prefix_at_path"]

    got = sorted({c for c, _ in rows})
    print(f"recovered {len(rows)} readouts covering cases {min(got)}..{max(got)} "
          f"({len(got)} distinct of {len(cases)})")
    missing = [i for i in range(len(cases)) if i not in set(got)]
    if missing:
        print(f"NOT recovered: cases {missing} — the run was pruned before they were reached "
              f"or their poll output was never captured")

    OUT.write_text(json.dumps(sorted(rows.values(), key=lambda r: (r["case"], r["cut"])), indent=1),
                   encoding="utf-8")

    def per_claim(rs):
        cl: dict = {}
        for r in rs:
            k = (r["run_id"].split("-eval")[0], r["target"])
            d = cl.setdefault(k, {"t": 0, "d": 0, "l": 0})
            d["t"] |= 1 if r["j_target"] > 0 else 0
            d["d"] |= 1 if r["j_decoy"] > 0 else 0
            d["l"] |= 1 if r["l_target"] > 0 else 0
        return cl

    print("\n=== LENS, any-token rule (the only rule these counts support) ===")
    for cut in ("prefix_at_sentence", "prefix_at_path"):
        for label, pred in (("all recovered", lambda r: True),
                            ("GENUINELY EARLIER", lambda r: r["earlier_cut"])):
            rs = [r for r in rows.values() if r["cut"] == cut and pred(r)]
            if not rs:
                continue
            cl = per_claim(rs)
            n = len(cl)
            t = sum(v["t"] for v in cl.values())
            d = sum(v["d"] for v in cl.values())
            l = sum(v["l"] for v in cl.values())
            b = sum(1 for v in cl.values() if v["t"] and not v["d"])
            c = sum(1 for v in cl.values() if v["d"] and not v["t"])
            print(f"  {cut:20} {label:18} n={n:>3}  target {t:>3}  decoy {d:>3}  "
                  f"logit {l:>3}  p={mcnemar_exact(b, c):.4f}")

    print("\nask baseline, same question, genuinely earlier cut: target 18/30, decoy 3/30, "
          "p = 0.0003")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
