"""Where in the stack does the relocated file become decodable — and what happens under honest scoring?

Consumes the raw readout the second run recorded (top-20 token ids at every layer, both lenses,
plus the target and decoy token sets). The first run stored only how many layers hit, which is why
this analysis needed a second A100 hour; it will not need a third, because everything below is a
re-score of ids already on disk.

TWO THINGS THIS PRODUCES.

1. THE EMERGENCE CURVE. For each layer, the fraction of claims whose target token is in the
   top 20 there — against the decoy, and against the plain logit lens. A lens result is a claim
   about where a computation has settled, and a hit count summed over layers cannot make it.

2. SYMMETRIC DISTINCTIVE-TOKEN SCORING. The first run counted a hit when ANY token of a filename
   entered the top 20, so `.py` — shared by two unrelated files — scored for the target and the
   decoy at once. Dropping shared tokens from BOTH sides is the fix. The one-sided version
   (void the decoy, keep the target) moved p from 0.109 to 0.0078 and was not reportable, because
   it corrects only the side that hurts the hypothesis.

Read target against decoy, never against zero.

Run: python experiments/jlens_layer_curve.py
"""
from __future__ import annotations

import collections
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "experiments" / "traced50" / "reloc_results.json"
CUTS = ("prefix_at_sentence", "prefix_at_path")


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def claim_key(r: dict) -> tuple:
    return (r["run_id"].split("-eval")[0], r["target"])


def score(rows: list, cut: str, distinctive: bool) -> dict:
    """Per-claim hit/miss for target and decoy, optionally on distinctive tokens only."""
    claims: dict = {}
    for r in rows:
        if r["cut"] != cut:
            continue
        tgt, dec = set(r["target_tok"]), set(r["decoy_tok"])
        if distinctive:
            # SYMMETRIC. Whatever the two names share cannot distinguish them, so it is removed
            # from both — not just from the decoy.
            shared = tgt & dec
            tgt, dec = tgt - shared, dec - shared
        d = claims.setdefault(claim_key(r), {"t": 0, "d": 0, "l": 0})
        for layer in r["layer_ids"]:
            j = set(r["j_top20"][str(layer)])
            l_ = set(r["l_top20"][str(layer)])
            d["t"] |= 1 if j & tgt else 0
            d["d"] |= 1 if j & dec else 0
            d["l"] |= 1 if l_ & tgt else 0
    n = len(claims)
    t = sum(v["t"] for v in claims.values())
    dd = sum(v["d"] for v in claims.values())
    ll = sum(v["l"] for v in claims.values())
    b = sum(1 for v in claims.values() if v["t"] and not v["d"])
    c = sum(1 for v in claims.values() if v["d"] and not v["t"])
    return {"n": n, "target": t, "decoy": dd, "logit": ll, "p": mcnemar_exact(b, c)}


def curve(rows: list, cut: str, distinctive: bool = True) -> dict:
    """Per layer: fraction of CLAIMS with the token in top-20, for j-target, j-decoy, l-target."""
    per: dict = collections.defaultdict(lambda: collections.defaultdict(
        lambda: {"t": 0, "d": 0, "l": 0}))
    layers: list = []
    for r in rows:
        if r["cut"] != cut:
            continue
        layers = r["layer_ids"]
        tgt, dec = set(r["target_tok"]), set(r["decoy_tok"])
        if distinctive:
            shared = tgt & dec
            tgt, dec = tgt - shared, dec - shared
        k = claim_key(r)
        for layer in layers:
            j = set(r["j_top20"][str(layer)])
            l_ = set(r["l_top20"][str(layer)])
            cell = per[layer][k]
            cell["t"] |= 1 if j & tgt else 0
            cell["d"] |= 1 if j & dec else 0
            cell["l"] |= 1 if l_ & tgt else 0
    out = {"layers": layers, "t": [], "d": [], "l": []}
    for layer in layers:
        cs = per[layer]
        n = max(len(cs), 1)
        out["t"].append(round(sum(v["t"] for v in cs.values()) / n, 4))
        out["d"].append(round(sum(v["d"] for v in cs.values()) / n, 4))
        out["l"].append(round(sum(v["l"] for v in cs.values()) / n, 4))
    return out


def main() -> int:
    if not RAW.is_file():
        print(f"missing {RAW}\n  colab download -s reloc4 /content/reloc_results.json "
              f"{RAW}", file=sys.stderr)
        return 1
    rows = json.loads(RAW.read_text(encoding="utf-8"))
    print(f"rows: {len(rows)}   layers: {rows[0]['layers']}")

    print("\n=== PER-CLAIM, two scoring rules ===")
    for cut in CUTS:
        a = score(rows, cut, distinctive=False)
        b = score(rows, cut, distinctive=True)
        print(f"\n{cut}   n={a['n']} claims")
        print(f"  any token        target {a['target']}/{a['n']}  decoy {a['decoy']}/{a['n']}  "
              f"logit {a['logit']}/{a['n']}   p={a['p']:.4f}")
        print(f"  DISTINCTIVE only target {b['target']}/{b['n']}  decoy {b['decoy']}/{b['n']}  "
              f"logit {b['logit']}/{b['n']}   p={b['p']:.4f}   <- symmetric, reportable")

    print("\n=== EMERGENCE CURVES (distinctive tokens, fraction of claims per layer) ===")
    data = {}
    for cut in CUTS:
        c = curve(rows, cut)
        data[cut] = c
        peak = max(range(len(c["t"])), key=lambda i: c["t"][i]) if c["t"] else 0
        first = next((i for i, v in enumerate(c["t"]) if v > 0), None)
        print(f"\n{cut}")
        print(f"  layers {c['layers'][0]}..{c['layers'][-1]}")
        print(f"  target peaks at layer {c['layers'][peak]} ({c['t'][peak]:.0%} of claims)")
        if first is not None:
            print(f"  target first nonzero at layer {c['layers'][first]}")
        print(f"  target {[f'{v:.2f}' for v in c['t']]}")
        print(f"  decoy  {[f'{v:.2f}' for v in c['d']]}")
        print(f"  logit  {[f'{v:.2f}' for v in c['l']]}")

    out = ROOT / "experiments" / "traced50" / "layer_curve.json"
    out.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"\nwrote {out.relative_to(ROOT)} for the figure")
    return 0


if __name__ == "__main__":
    sys.exit(main())
