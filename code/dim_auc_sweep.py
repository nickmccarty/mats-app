"""Single-dimension discrimination, and whether it survives being tested on a different claim.

THE EXPERIMENT THIS MIRRORS. A standard move in the reasoning-interpretability literature: score
every unit by how well its activation separates the event of interest, find units with high ROC-AUC
and large Cohen's d, then report that they fail to generalise and conclude polysemanticity.

THE PART THAT USUALLY GOES UNMEASURED. "Fails to generalise" is normally shown by plotting one
unit's activations over a fresh sequence and observing it firing in the wrong places. That is a
demonstration, not a measurement, and it cannot distinguish polysemanticity from the far more
boring explanation: with 2,048 dimensions and ~35 claims, the best-looking dimension is the best of
2,048 draws from noise, and its in-sample AUC is a selection artifact with no signal to lose.

So this reports three numbers per layer, and only the gap between them means anything:

  IN-SAMPLE      the best dimension's AUC on the claims used to pick it. Always high. Worthless.
  HELD-OUT       that same dimension, re-scored on claims it was not selected on.
  SHUFFLED       the identical procedure with labels permuted -- how high IN-SAMPLE goes when there
                 is provably nothing to find.

If in-sample sits near shuffled, there was never a signal and polysemanticity explains nothing.

Run: python experiments/dim_auc_sweep.py [--perms 200]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESID = ROOT / "experiments" / "traced50" / "resid.npz"
META = ROOT / "experiments" / "traced50" / "resid_meta.json"
LABELS = ROOT / "experiments" / "traced50" / "reloc_labels.json"
OUT = ROOT / "experiments" / "traced50" / "dim_auc_sweep.json"


def auc_cols(X, y):
    """ROC-AUC of every column at once, by rank. Returns (n_dims,)."""
    import numpy as np
    order = np.argsort(X, axis=0)
    ranks = np.empty_like(order, dtype=np.float64)
    n = X.shape[0]
    rows = np.arange(1, n + 1)[:, None]
    np.put_along_axis(ranks, order, np.broadcast_to(rows, X.shape), axis=0)
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return np.full(X.shape[1], 0.5)
    return (ranks[pos].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def cohens_d(X, y):
    import numpy as np
    a, b = X[y == 1], X[y == 0]
    if len(a) < 2 or len(b) < 2:
        return np.zeros(X.shape[1])
    s = np.sqrt(((len(a) - 1) * a.var(0, ddof=1) + (len(b) - 1) * b.var(0, ddof=1))
                / max(1, len(a) + len(b) - 2))
    return (a.mean(0) - b.mean(0)) / (s + 1e-9)


def main() -> int:
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=200)
    ap.add_argument("--layers", type=int, default=8)
    a = ap.parse_args()

    if not RESID.is_file():
        print(f"missing {RESID}", file=sys.stderr)
        return 1
    z = np.load(RESID)
    meta = json.loads(META.read_text(encoding="utf-8"))
    labels = {r["case"]: r for r in json.loads(LABELS.read_text(encoding="utf-8"))}

    def design(layer):
        X, y, g = [], [], []
        for m in meta:
            if m.get("is_control") or m["cut"] != "prefix_at_sentence":
                continue
            lab = labels.get(m["case"])
            if lab is None or lab.get("cut_identical"):
                continue
            X.append(z[m["key"]][layer].astype(np.float32))
            y.append(1 if lab["correct"] else 0)
            g.append(f'{m["run_id"].split("-eval")[0]}::{lab["target_path"]}')
        return np.stack(X), np.array(y), np.array(g)

    n_layers = meta[0]["n_layers"]
    probe_layers = sorted({int(round(v)) for v in np.linspace(0, n_layers - 1, a.layers)})
    rng = np.random.default_rng(20260911)

    def best_dim_gap(X, y, g, tr):
        """Pick the best dim on the training claims, score it on the held-out ones.

        The split is passed IN rather than drawn here, so the real labels and every permutation are
        evaluated on the identical partition. Re-drawing it per permutation would compare one
        specific split against an average over splits, and the difference between those two things
        would land in the reported gap.
        """
        te = ~tr
        if len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2:
            return None
        atr = auc_cols(X[tr], y[tr])
        j = int(np.argmax(np.abs(atr - 0.5)))
        flip = atr[j] < 0.5
        col = -X[:, j:j + 1] if flip else X[:, j:j + 1]
        return float(auc_cols(col[tr], y[tr])[0]), float(auc_cols(col[te], y[te])[0]), j

    res = []
    print(f"{'layer':>6} {'claims':>7} {'in-sample':>10} {'held-out':>9} "
          f"{'shuffled':>9} {'|d|':>6}")
    for L in probe_layers:
        X, y, g = design(L)
        uniq = np.unique(g)
        if len(uniq) < 6:
            continue
        half = rng.permutation(uniq)[: len(uniq) // 2]
        tr = np.isin(g, half)
        got = best_dim_gap(X, y, g, tr)
        if got is None:
            continue
        ins, held, j = got
        null = []
        for _ in range(a.perms):
            yy = rng.permutation(y)
            r = best_dim_gap(X, yy, g, tr)
            if r:
                null.append(r[0])
        nm = float(np.mean(null)) if null else float("nan")
        d = float(np.abs(cohens_d(X, y))[j])
        res.append({"layer": int(L), "dim": j, "in_sample": ins, "held_out": held,
                    "shuffled_in_sample": nm, "cohens_d": d, "n_claims": int(len(np.unique(g)))})
        print(f"{L:>6} {len(np.unique(g)):>7} {ins:>10.3f} {held:>9.3f} {nm:>9.3f} {d:>6.2f}")

    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    if res:
        mi = float(np.mean([r["in_sample"] for r in res]))
        mh = float(np.mean([r["held_out"] for r in res]))
        ms = float(np.mean([r["shuffled_in_sample"] for r in res]))
        print(f"\nmean in-sample {mi:.3f} | held-out {mh:.3f} | shuffled in-sample {ms:.3f}")
        print("Compare in-sample against SHUFFLED, not against 0.5. If they match, the "
              "'best dimension' is the best of 2,048 noise draws and there is no signal for\n"
              "polysemanticity to be hiding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
