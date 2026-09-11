"""Q4: is the RELOCATION EVENT decodable, before the sentence that carries it?

The earlier probes asked what the model was about to say. This asks whether it was about to change
its mind at all -- the question the backtracking literature asks of reasoning traces, with the
difference that a relocation has a gold answer and a "Wait" does not.

  positives   the residual at the sentence boundary before a relocation
  negatives   the residual at a sentence boundary in the SAME run, in a step with no relocation,
              chosen to match the positive's prefix length

WHAT WOULD MAKE THIS RESULT FAKE, and what is done about each:

  LENGTH. Relocations arrive late in long traces. A classifier could learn "long context" and score
  well while knowing nothing. Controls are length-matched at export (median gap 14 characters), and
  this script reports the AUC of a classifier given ONLY the token count. That number is the floor
  the real probe has to clear.

  RUN IDENTITY. Positives and their matched negatives share a run, a repository and a prompt. Folds
  are leave-one-RUN-out, so a run never appears on both sides -- otherwise the probe could memorise
  the run and read the label off it.

  DIMENSIONALITY. 2,048 dimensions against ~28 runs. PCA inside each fold, and every reported AUC
  sits beside the permutation null for the identical procedure.

Run: python experiments/probe_event.py [--perms 500]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
T50 = ROOT / "experiments" / "traced50"
LABELS = T50 / "reloc_labels.json"
OUT = T50 / "probe_event.json"


def load(np):
    """Positives from the relocation pass, negatives from the control pass."""
    zp = np.load(T50 / "resid.npz")
    mp = json.loads((T50 / "resid_meta.json").read_text(encoding="utf-8"))
    zc_p, mc_p = T50 / "resid_ctrl.npz", T50 / "resid_ctrl_meta.json"
    if not zc_p.is_file() or not mc_p.is_file():
        print("missing the control extraction -- run colab_run_controls.py first", file=sys.stderr)
        return None
    zc = np.load(zc_p)
    mc = json.loads(mc_p.read_text(encoding="utf-8"))
    labels = {r["case"]: r for r in json.loads(LABELS.read_text(encoding="utf-8"))}
    return zp, mp, zc, mc, labels


def final_char_matched(cases_reloc, cases_ctrl):
    """Cases whose relocation prefix and control prefix end on the SAME character.

    Layer 0 is the embedding output, before any block has run, and the first version of this probe
    scored 0.894 there -- which cannot be a fact about computation. It was a fact about punctuation:
    the two exports cut on different conventions, so relocation prefixes end on `.` (45 of 75) and
    control prefixes end on `.` never, ending instead on a space (28 of 74). The classifier was
    reading the last token's identity.

    Requiring the pair to share a final character removes the shortcut without re-extracting. It is
    a patch, not the fix -- the fix is to cut both exports identically, which needs another
    extraction -- and it costs most of the remaining sample.
    """
    rel = {i: c["prefix_at_sentence"][-1:] for i, c in enumerate(cases_reloc)}
    ok = set()
    for c in cases_ctrl:
        i = c.get("matched_case")
        if i in rel and c["prefix_at_sentence"][-1:] == rel[i]:
            ok.add(i)
    return ok


def length_balanced(np, mp, mc, labels, max_gap):
    """Cases whose relocation prompt and control prompt are within `max_gap` relative token length.

    Export matched the reasoning PREFIX to a median of 14 characters, which is tight -- but the
    prompt also carries the conversation, and controls come from earlier steps (20.6 messages
    against 27.3). The residue is real: over all pairs, token count alone separates the classes at
    AUC 0.62, so a probe scoring 0.62 would be reporting prompt length.

    Capping the relative gap at 30% leaves 25 pairs and drops length-only AUC to 0.498. That is the
    subset on which an activation result means what it says; the full set is reported beside it so
    the cost of the restriction is visible rather than hidden.
    """
    pos = {}
    for m in mp:
        if m["cut"] != "prefix_at_sentence" or m.get("is_control"):
            continue
        lab = labels.get(m["case"])
        if lab is None or lab.get("cut_identical"):
            continue
        pos[m["case"]] = m["tokens"]
    ok = set()
    for m in mc:
        c = m.get("matched_case")
        if c in pos:
            hi = max(pos[c], m["tokens"])
            if hi and abs(pos[c] - m["tokens"]) / hi <= max_gap:
                ok.add(c)
    return ok


def design(np, zp, mp, zc, mc, labels, layer, only=None):
    """One positive per usable relocation readout, and ONLY the controls matched to them.

    The controls were exported 1:1 against all 75 cases, but the positives are filtered down to the
    cuts that are genuinely earlier than the naming sentence. Keeping every control would leave
    negatives whose partner is not in the design -- which quietly unbalances the classes and, worse,
    breaks the length matching the whole control rests on, because the dropped positives are not a
    random subset.
    """
    keep = set()
    X, y, g, tok = [], [], [], []
    for m in mp:
        if m["cut"] != "prefix_at_sentence" or m.get("is_control"):
            continue
        lab = labels.get(m["case"])
        if lab is None or lab.get("cut_identical"):
            continue
        if only is not None and m["case"] not in only:
            continue
        keep.add(m["case"])
        X.append(zp[m["key"]][layer].astype(np.float32))
        y.append(1); g.append(m["run_id"]); tok.append(m["tokens"])
    for m in mc:
        if m.get("matched_case") not in keep:
            continue
        X.append(zc[m["key"]][layer].astype(np.float32))
        y.append(0); g.append(m["run_id"]); tok.append(m["tokens"])
    return np.stack(X), np.array(y), np.array(g), np.array(tok, dtype=np.float64)


def auc(truth, pred, np):
    truth = np.asarray(truth); pred = np.asarray(pred)
    if len(set(truth.tolist())) < 2:
        return 0.5
    order = np.argsort(pred); r = np.empty(len(pred)); r[order] = np.arange(1, len(pred) + 1)
    pos = truth == 1; n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cv_auc(np, X, yy, groups, n_comp=10, C=0.05):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    preds, truth = [], []
    for u in np.unique(groups):
        te = groups == u; tr = ~te
        if len(set(yy[tr].tolist())) < 2:
            continue
        k = min(n_comp, int(tr.sum()) - 1, X.shape[1])
        pipe = make_pipeline(StandardScaler(), PCA(n_components=k),
                             LogisticRegression(C=C, max_iter=2000))
        pipe.fit(X[tr], yy[tr])
        preds += list(pipe.predict_proba(X[te])[:, 1]); truth += list(yy[te])
    return auc(truth, preds, np)


def main() -> int:
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=500)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--max-token-gap", type=float, default=0.30,
                    help="keep only pairs whose prompts are within this relative token length; "
                         "1.0 keeps everything and reintroduces the length shortcut")
    a = ap.parse_args()

    got = load(np)
    if got is None:
        return 1
    zp, mp, zc, mc, labels = got
    n_layers = mp[0]["n_layers"]
    probe_layers = sorted({int(round(v)) for v in np.linspace(0, n_layers - 1, a.layers)})
    rng = np.random.default_rng(20260911)

    # THE FLOOR, CHECKED BEFORE ANYTHING ELSE. If token count alone separates the classes, the
    # matching failed and no number below is about activations.
    cr = json.loads((ROOT / "experiments" / "reloc_cases.json").read_text(encoding="utf-8"))[:75]
    cc = json.loads((ROOT / "experiments" / "control_cases.json").read_text(encoding="utf-8"))
    lb = length_balanced(np, mp, mc, labels, a.max_token_gap)
    fc = final_char_matched(cr, cc)
    both = lb & fc

    for label, only in (("all pairs", None),
                        (f"within {a.max_token_gap:.0%} token length", lb),
                        ("final character matched", fc),
                        ("both restrictions", both)):
        X, y, g, tok = design(np, zp, mp, zc, mc, labels, n_layers // 2, only)
        la = auc(y, tok, np)
        print(f"{label:<34} positives {int(y.sum()):>3} negatives {int((y==0).sum()):>3} "
              f"runs {len(set(g)):>3} | length-only AUC {la:.3f}")

    only = both
    X, y, g, tok = design(np, zp, mp, zc, mc, labels, n_layers // 2, only)
    if len(set(y.tolist())) < 2 or int(y.sum()) < 6:
        print(f"\nBOTH restrictions leave {int(y.sum())} pairs -- too few to score. The event "
              "question needs a re-extraction with both exports cut on one convention.")
        return 0
    len_auc = auc(y, tok, np)
    print("\nscoring the subset that is BOTH length-balanced and final-character matched.")
    print(f"the length shortcut on this subset is worth {len_auc:.3f}.")
    print("WATCH LAYER 0: it is the embedding output, before any block has run. A high AUC there "
          "is\nstill a property of the last token, not of anything the model computed.")

    res = {"length_only_auc": len_auc, "max_token_gap": a.max_token_gap,
           "n_positives": int(y.sum()), "layers": []}
    print(f"\n{'layer':>6} {'AUC':>6} {'null mean':>10} {'null p95':>9} {'p':>7}")

    def paired_shuffle(y, g):
        """Swap labels WITHIN each run rather than across the whole set.

        Positives and their matched negatives were deliberately drawn from the same run, so a
        global shuffle destroys the pairing and produces runs that are all-positive or
        all-negative -- which leave-one-run-out scoring handles differently from the real data.
        Permuting inside each run keeps every run's class balance and tests the only thing at
        issue: whether the label is attached to the right readout.
        """
        yy = y.copy()
        for u in np.unique(g):
            m = g == u
            yy[m] = rng.permutation(y[m])
        return yy

    for L in probe_layers:
        X, y, g, _ = design(np, zp, mp, zc, mc, labels, L, only)
        real = cv_auc(np, X, y, g)
        null = np.array([cv_auc(np, X, paired_shuffle(y, g), g) for _ in range(a.perms)])
        p = (float((null >= real).sum()) + 1) / (a.perms + 1)
        res["layers"].append({"layer": int(L), "auc": real, "null_mean": float(null.mean()),
                              "null_p95": float(np.quantile(null, .95)), "p": p})
        print(f"{L:>6} {real:>6.3f} {null.mean():>10.3f} {np.quantile(null,.95):>9.3f} {p:>7.3f}")
    # MULTIPLE COMPARISONS. Eight layers are eight tests, and the smallest of eight p-values is not
    # the p-value of the finding. Benjamini-Hochberg at q=0.05 is the least conservative correction
    # anyone would accept here, so it is the one reported -- and it is applied to our own result
    # rather than left for a reader to apply.
    if res["layers"]:
        ps = sorted((r["p"], r["layer"]) for r in res["layers"])
        m, q = len(ps), 0.05
        survivors = [(p, L) for i, (p, L) in enumerate(ps, 1) if p <= i / m * q]
        res["bh_q05_survivors"] = [L for _, L in survivors]
        res["bh_threshold_smallest"] = q / m
        print(f"\n  {m} layers tested. Benjamini-Hochberg at q=0.05 needs the smallest p below "
              f"{q/m:.5f}.")
        print(f"  smallest p is {ps[0][0]:.3f} at layer {ps[0][1]}; "
              f"surviving layers: {res['bh_q05_survivors'] or 'none'}")
        if not survivors:
            print("  Nothing survives correction. The late-layer pattern is suggestive and"
                  " underpowered,\n  not a result.")

    if res["layers"]:
        best = max(res["layers"], key=lambda r: r["auc"])
        print(f"\n  Best layer {best['layer']}: AUC {best['auc']:.3f} against a null 95th "
              f"percentile of {best['null_p95']:.3f}.")
        print(f"  At {res['n_positives']} matched pairs nothing below AUC {best['null_p95']:.3f} "
              "was detectable.")

    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print("Read each AUC against its null column. An AUC below the null's 95th percentile is "
          "not a finding, however far it sits above 0.5.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
