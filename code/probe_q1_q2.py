"""Follow-up Q1 and Q2: is the answer linearly decodable, and do right and wrong relocations differ?

Q1  IS THE NAMED FILE DECODABLE FROM THE RESIDUAL AT THE CUT?
    A readout trained on this task, rather than the published Jacobian lens the pilot used. Ridge
    from the residual h to the mean unembedding vector of the target's tokens, then score the
    target against the SAME seeded decoy the lens and the ask-baseline were scored against. A
    paired sign test over claims.

Q2  DO RIGHT AND WRONG RELOCATIONS LOOK DIFFERENT AT THAT POINT?
    Logistic regression on h predicting whether the relocated file is in the fix commit. 21 correct
    claims against 17 wrong, which is about as balanced as this corpus gets.

THREE THINGS THAT KEEP THIS HONEST, because n is small enough that a careless probe finds
whatever it is asked to find:

  GROUPED FOLDS. Splits are leave-one-CLAIM-out, never leave-one-readout-out. The same claim
  appears as several mentions and at two cut points; a random split puts near-duplicates on both
  sides and reports memorisation as generalisation.

  A PERMUTATION NULL. 4,096 dimensions against ~38 claims will separate shuffled labels too. The
  reported p is the fraction of label shuffles whose cross-validated score beats the real one, so
  the null is measured on this data rather than assumed.

  DIMENSIONALITY HELD DOWN. PCA to a small number of components fitted INSIDE each fold. Fitting
  it on all the data first would leak the test fold into the projection, which is the classic way
  to publish a probe that cannot be reproduced.

Run: python experiments/probe_q1_q2.py [--layers 8] [--perms 500]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESID = ROOT / "experiments" / "traced50" / "resid.npz"
META = ROOT / "experiments" / "traced50" / "resid_meta.json"
EMB = ROOT / "experiments" / "traced50" / "tok_emb.npz"
EMBN = ROOT / "experiments" / "traced50" / "tok_emb_names.json"
CASES = ROOT / "experiments" / "reloc_cases.json"
LABELS = ROOT / "experiments" / "traced50" / "reloc_labels.json"
OUT = ROOT / "experiments" / "traced50" / "probe_q1_q2.json"


def load_all(labels_path=None):
    import numpy as np
    if not RESID.is_file():
        print(f"missing {RESID} -- run the extraction first", file=sys.stderr)
        return None
    z = np.load(RESID)
    meta = json.loads(META.read_text(encoding="utf-8"))
    lp = pathlib.Path(labels_path) if labels_path else LABELS
    labels = {r["case"]: r for r in json.loads(lp.read_text(encoding="utf-8"))}
    emb = np.load(EMB) if EMB.is_file() else None
    embn = json.loads(EMBN.read_text(encoding="utf-8")) if EMBN.is_file() else None

    # The decoy pairing, derived locally. If the Colab artifact is present too, the two must agree
    # -- a mismatch would mean the lens, the baseline and these probes were scored against
    # different controls, which would make the comparison table meaningless.
    cases = json.loads(CASES.read_text(encoding="utf-8"))[:75]
    local = seeded_decoys(cases)
    if embn is not None and embn.get("decoys") != local:
        print("FATAL: locally derived decoys disagree with tok_emb_names.json", file=sys.stderr)
        return None
    if embn is None:
        embn = {"decoys": local, "names": {}}
    return z, meta, labels, emb, embn


def seeded_decoys(cases):
    """Reproduce the decoy pairing the lens run and the ask-baseline were scored against.

    Derived here rather than read from the Colab artifact, so the control survives a lost session:
    it is a seeded shuffle of the target basenames, and the seed is the one both earlier
    experiments used. Any drift between this and `tok_emb_names.json` would silently re-pair every
    case, so `main` cross-checks the two whenever both are present.
    """
    import random
    rng = random.Random(20260906)
    targets = [c["target_basename"] for c in cases]
    decoys = targets[:]
    rng.shuffle(decoys)
    for i in range(len(decoys)):
        if decoys[i] == targets[i] and len(set(targets)) > 1:
            j = (i + 1) % len(decoys)
            decoys[i], decoys[j] = decoys[j], decoys[i]
    return decoys


def same_repo_decoys(cases):
    """A decoy drawn from the SAME repository as the target it is paired against.

    WHY THE SEEDED GLOBAL DECOY IS NOT ENOUGH HERE. 20 of the 21 distinct relocated filenames occur
    in exactly one repository, and 71 of 75 global decoys name a file that never appears in the
    case's own repo. "Is the readout nearer the target than the decoy" is then answerable from
    repository identity, which every residual carries trivially -- the whole context is that repo's
    source. A probe can score far above chance while knowing only which project it is reading.

    Restricting the decoy to a different file from the same repository removes that shortcut
    entirely: repo identity now favours both candidates equally, so anything above chance has to
    come from distinguishing files WITHIN the project.

    It costs most of the sample. Only 4 of 15 repos ever host two distinct relocated filenames, so
    this scores ~21 of 75 cases. That is the honest exchange rate: a small clean number instead of
    a large confounded one.
    """
    import random
    rng = random.Random(20260906)
    by_repo: dict[str, list[str]] = {}
    for c in cases:
        by_repo.setdefault(c["repo"], [])
        if c["target_basename"] not in by_repo[c["repo"]]:
            by_repo[c["repo"]].append(c["target_basename"])
    out = []
    for c in cases:
        alts = [f for f in by_repo.get(c["repo"], []) if f != c["target_basename"]]
        out.append(rng.choice(sorted(alts)) if alts else None)
    return out


def claim_of(m, labels):
    """The unit of analysis: a distinct (task, file) assertion, not a mention and not a readout."""
    lab = labels.get(m["case"])
    path = lab["target_path"] if lab else m["target"]
    return (m["run_id"].split("-eval")[0], path)


def usable(m, labels, drop_identical=True):
    """A readout at the cut the question is actually about.

    Only `prefix_at_sentence`, and -- by default -- only where that prefix is genuinely earlier than
    the mid-sentence one. For 12 of 75 mentions the miner found no earlier sentence boundary and
    fell back, making the two byte-identical; a readout there is taken after the model has begun
    naming the file, which is the thing Q1 is supposed to be blind to.
    """
    if m["cut"] != "prefix_at_sentence":
        return False
    lab = labels.get(m["case"])
    if lab is None:
        return False
    return not (drop_identical and lab.get("cut_identical"))


def q2_probe(z, meta, labels, layer, perms, rng, drop_identical=True):
    """Right vs wrong, leave-one-claim-out, with a permutation null."""
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, g = [], [], []
    for m in meta:
        if not usable(m, labels, drop_identical):
            continue
        X.append(z[m["key"]][layer].astype(np.float32))
        y.append(1 if labels[m["case"]]["correct"] else 0)
        g.append(claim_of(m, labels))
    if not X:
        return None
    X = np.stack(X); y = np.array(y)
    groups = np.array([hash(k) % (10**9) for k in g])
    uniq = np.unique(groups)
    if len(uniq) < 6 or len(set(y)) < 2:
        return None

    def cv_score(yy):
        # leave-one-claim-out; a fold whose training labels collapse to one class is skipped
        # rather than scored, because predicting the majority there is not evidence.
        preds, truth = [], []
        for u in uniq:
            te = groups == u
            tr = ~te
            if len(set(yy[tr])) < 2:
                continue
            n_comp = min(10, tr.sum() - 1, X.shape[1])
            pipe = make_pipeline(StandardScaler(), PCA(n_components=n_comp),
                                 LogisticRegression(C=0.05, max_iter=2000))
            pipe.fit(X[tr], yy[tr])
            preds += list(pipe.predict_proba(X[te])[:, 1])
            truth += list(yy[te])
        if len(set(truth)) < 2:
            return 0.5
        # AUC by rank, no sklearn.metrics dependency on tie handling
        order = np.argsort(preds)
        ranks = np.empty(len(preds)); ranks[order] = np.arange(1, len(preds) + 1)
        pos = np.array(truth) == 1
        n1, n0 = pos.sum(), (~pos).sum()
        return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

    real = cv_score(y)

    # PERMUTE AT THE CLAIM LEVEL, not the readout level.
    #
    # A claim's readouts all carry one label -- it is one assertion, mentioned several times. A
    # readout-level shuffle produces claims whose own readouts disagree, which real data never
    # contains and which leave-one-claim-out scoring cannot get right by construction. That makes
    # the null harder than the thing it is the null FOR, and every p computed against it is
    # optimistic. Shuffling the labels over claims and broadcasting back preserves the structure
    # and tests exactly the hypothesis of interest: that the assignment of outcome to claim is
    # arbitrary.
    claim_ids = np.unique(groups)
    claim_label = np.array([y[groups == u][0] for u in claim_ids])
    mixed = [int(u) for u in claim_ids if len(set(y[groups == u].tolist())) > 1]
    if mixed:   # would mean the label is not a property of the claim, and the grouping is wrong
        print(f"  WARNING: {len(mixed)} claims carry disagreeing labels; "
              "claim-level permutation is not valid here", file=sys.stderr)
    null = []
    for _ in range(perms):
        perm = rng.permutation(claim_label)
        yy = np.empty_like(y)
        for u, lab in zip(claim_ids, perm):
            yy[groups == u] = lab
        null.append(cv_score(yy))
    null = np.array(null)
    p = float((null >= real).sum() + 1) / (perms + 1)
    # The null's 95th percentile IS the detection threshold at this sample size: an AUC below it
    # could not have been called significant however large the underlying effect. Reporting it
    # converts "we found nothing" into "we could not have found anything smaller than this",
    # which is the only honest way to read a null at 35 claims.
    return {"layer": int(layer), "n_readouts": int(len(y)), "n_claims": int(len(uniq)),
            "pos": int(y.sum()), "auc": float(real), "null_mean": float(null.mean()),
            "null_p95": float(np.quantile(null, 0.95)),
            "detectable_gap": float(np.quantile(null, 0.95) - null.mean()), "p": p}


def q1_probe(z, meta, labels, emb, embn, layer, rng, drop_identical=True):
    """Target vs decoy, from a ridge readout trained leave-one-claim-out."""
    import numpy as np
    from sklearn.linear_model import Ridge

    names, decoys = embn["names"], embn["decoys"]

    def vec(name):
        ids = names.get(name) or []
        rows = [emb[str(t)].astype(np.float32) for t in ids if str(t) in emb]
        return np.mean(rows, axis=0) if rows else None

    X, Yt, Yd, g = [], [], [], []
    dropped_same = 0
    for m in meta:
        if not usable(m, labels, drop_identical):
            continue
        t = vec(m["target"])
        d = vec(decoys[m["case"]]) if m["case"] < len(decoys) else None
        if t is None or d is None:
            continue
        # Only 21 distinct basenames span the 75 cases, so a decoy can be the same filename as the
        # target it is paired against. Such a readout cannot favour either and would silently
        # inflate the denominator; drop it and count it.
        if m["case"] < len(decoys) and decoys[m["case"]] == m["target"]:
            dropped_same += 1
            continue
        X.append(z[m["key"]][layer].astype(np.float32))
        Yt.append(t); Yd.append(d)
        g.append(claim_of(m, labels))
    if len(X) < 8:
        return None
    X = np.stack(X); Yt = np.stack(Yt); Yd = np.stack(Yd)
    groups = np.array([hash(k) % (10**9) for k in g])
    uniq = np.unique(groups)

    votes: dict = {}
    for u in uniq:
        te = groups == u
        tr = ~te
        if tr.sum() < 4:
            continue
        r = Ridge(alpha=1e4).fit(X[tr], Yt[tr])
        pred = r.predict(X[te])
        # index the original list: claim keys are tuples, and np.asarray would turn them into rows
        gg = [g[i] for i in np.where(te)[0]]
        for k, row in enumerate(pred):
            nrm = np.linalg.norm(row) + 1e-9
            st = float(row @ Yt[te][k] / (nrm * (np.linalg.norm(Yt[te][k]) + 1e-9)))
            sd = float(row @ Yd[te][k] / (nrm * (np.linalg.norm(Yd[te][k]) + 1e-9)))
            if st != sd:
                votes.setdefault(gg[k], []).append(st > sd)

    # Per CLAIM, like everything else. Scoring 62 readouts as independent trials was how the first
    # version of this reported p = 0.0 for what is at most 35 observations.
    hits = miss = tied = 0
    for v in votes.values():
        up = sum(v)
        if up * 2 > len(v): hits += 1
        elif up * 2 < len(v): miss += 1
        else: tied += 1
    n = hits + miss
    if n == 0:
        return None
    # exact binomial, two-sided, against the 50% a coin would give
    from math import comb
    k = max(hits, miss)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2**n
    return {"layer": int(layer), "n_claims_scored": n, "target_wins": hits, "decoy_wins": miss,
            "tied_claims": tied, "dropped_identical_decoy": dropped_same,
            "decoy_scope": "global -- CONFOUNDED by repository identity, see decoy_scope_audit.py",
            "rate": hits / n, "p": float(min(1.0, 2 * tail))}


def q1_centroid(z, meta, labels, embn, layer, drop_identical=True):
    """Q1 without the unembedding matrix: do readouts that precede the SAME filename cluster?

    For a held-out readout, compare its residual against the mean residual of other readouts whose
    target was the same filename, and against the mean for the decoy filename. Closer to the target
    centroid is a hit. Same seeded decoy pairing as everything else, so the number sits on the same
    axis as the lens and the ask-baseline.

    THE CONFOUND THIS HAS TO DISMISS. Readouts sharing a filename often share a repository and a
    run, so a naive centroid would be a run-identity detector wearing a filename's name. Every
    readout contributing to a centroid is therefore required to come from a DIFFERENT run than the
    readout being scored; a case whose target filename appears in no other run is unscoreable and
    is counted, not quietly dropped.
    """
    import numpy as np
    decoys = embn["decoys"]

    rows = [m for m in meta if usable(m, labels, drop_identical)
            and m["case"] < len(decoys) and decoys[m["case"]]
            and decoys[m["case"]] != m["target"]]
    if len(rows) < 8:
        return None
    H = {m["key"]: z[m["key"]][layer].astype(np.float32) for m in rows}

    def centroid(name, exclude_run):
        v = [H[m["key"]] for m in rows
             if m["target"] == name and m["run_id"] != exclude_run]
        return np.mean(v, axis=0) if v else None

    def cos(a, b):
        return float(a @ b / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)))

    per_claim: dict = {}
    unscoreable = 0
    for m in rows:
        ct = centroid(m["target"], m["run_id"])
        cd = centroid(decoys[m["case"]], m["run_id"])
        if ct is None or cd is None:
            unscoreable += 1
            continue
        h = H[m["key"]]
        st, sd = cos(h, ct), cos(h, cd)
        if st == sd:
            continue
        per_claim.setdefault(claim_of(m, labels), []).append(st > sd)

    # COLLAPSE TO CLAIMS BEFORE TESTING. One assertion restated seven times is one observation, not
    # seven; an exact binomial over readouts would treat those seven correlated outcomes as
    # independent trials and shrink p by orders of magnitude. A claim counts once, by majority of
    # its readouts, and a claim that splits evenly is dropped rather than broken arbitrarily.
    hits = miss = split = 0
    for votes in per_claim.values():
        up = sum(votes)
        if up * 2 > len(votes):
            hits += 1
        elif up * 2 < len(votes):
            miss += 1
        else:
            split += 1
    n = hits + miss
    if n == 0:
        return None
    from math import comb
    k = max(hits, miss)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2**n
    return {"layer": int(layer), "n_claims_scored": n, "target_wins": hits, "decoy_wins": miss,
            "tied_claims": split, "n_readouts": sum(len(v) for v in per_claim.values()),
            "unscoreable_no_other_run": unscoreable, "rate": hits / n,
            "p": float(min(1.0, 2 * tail))}


def main() -> int:
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=8, help="evenly spaced layers to test")
    ap.add_argument("--perms", type=int, default=500)
    ap.add_argument("--labels", default=None,
                    help="alternate label file, e.g. the --allow-basename set, to test whether "
                         "any conclusion turns on the gold matching rule")
    ap.add_argument("--only", choices=("q1", "q2", "both"), default="both",
                    help="Q2's permutation null is the expensive half; --only q1 re-runs the "
                         "cheap readout probes without paying for it again")
    ap.add_argument("--keep-identical-cuts", action="store_true",
                    help="also use readouts whose 'before the sentence' cut is byte-identical to "
                         "the mid-sentence one (they are taken after naming has begun)")
    a = ap.parse_args()
    drop = not a.keep_identical_cuts

    got = load_all(a.labels)
    if got is None:
        return 1
    z, meta, labels, emb, embn = got
    n_layers = meta[0]["n_layers"]
    probe_layers = sorted({int(round(x)) for x in
                           np.linspace(0, n_layers - 1, a.layers)})
    rng = np.random.default_rng(20260911)
    n_use = sum(usable(m, labels, drop) for m in meta)
    print(f"readouts {len(meta)} | layers {n_layers} | d_model {meta[0]['d_model']}")
    print(f"usable at the earlier cut: {n_use}"
          f"{'' if drop else '  (INCLUDING identical cuts -- not the headline condition)'}")
    print(f"testing layers {probe_layers}\n")

    res = {"q1": [], "q2": []}

    if a.only in ("q2", "both"):
        print("Q2  right vs wrong relocation, leave-one-claim-out, permutation null")
        print(f"{'layer':>6} {'claims':>7} {'correct':>8} {'AUC':>6} {'null':>6} "
              f"{'p95':>6} {'p':>7}")
        for L in probe_layers:
            r = q2_probe(z, meta, labels, L, a.perms, rng, drop)
            if r:
                res["q2"].append(r)
                print(f"{r['layer']:>6} {r['n_claims']:>7} {r['pos']:>8} "
                      f"{r['auc']:>6.3f} {r['null_mean']:>6.3f} {r['null_p95']:>6.3f} "
                      f"{r['p']:>7.3f}")
        if res["q2"]:
            best = max(res["q2"], key=lambda r: r["auc"])
            print(f"\n  Best layer {best['layer']}: AUC {best['auc']:.3f}, and the null's 95th "
                  f"percentile is {best['null_p95']:.3f}.")
            print(f"  At {best['n_claims']} claims this design could not have detected anything "
                  f"below AUC {best['null_p95']:.3f};")
            print("  the null result bounds the effect, it does not exclude one.")

    if a.only == "q2":
        pass
    elif emb is not None and embn.get("names"):
        print("\nQ1  target vs decoy from a trained ridge readout, leave-one-claim-out")
        print("    (global decoy -- carries the same repository confound as Q1b below)")
        print(f"{'layer':>6} {'claims':>7} {'target':>7} {'decoy':>6} {'rate':>6} {'p':>7}")
        for L in probe_layers:
            r = q1_probe(z, meta, labels, emb, embn, L, rng, drop)
            if r:
                res["q1"].append(r)
                print(f"{r['layer']:>6} {r['n_claims_scored']:>7} {r['target_wins']:>7} "
                      f"{r['decoy_wins']:>6} {r['rate']:>6.1%} {r['p']:>7.4f}")
        print("    Layer 40 is the final block, immediately before the unembedding, so a map"
              " trained\n    there is close to reading the model's own next-token distribution.")
    else:
        print("\nQ1 (unembedding form) skipped: tok_emb.npz not present "
              "(run experiments/dump_token_embeddings.py on the session)")

    if a.only in ("q1", "both"):
        cases = json.loads(CASES.read_text(encoding="utf-8"))[:75]
        scopes = [("global decoy  (CONFOUNDED -- see below)", embn["decoys"]),
                  ("same-repo decoy (the honest one)", same_repo_decoys(cases))]
        for name, dec in scopes:
            print(f"\nQ1b  nearest-centroid, {name}")
            print(f"{'layer':>6} {'claims':>7} {'target':>7} {'decoy':>6} {'rate':>6} "
                  f"{'p':>8} {'tied':>5}")
            key = "q1_centroid" if dec is embn["decoys"] else "q1_centroid_same_repo"
            res[key] = []
            for L in probe_layers:
                r = q1_centroid(z, meta, labels, {"decoys": dec}, L, drop)
                if r:
                    res[key].append(r)
                    print(f"{r['layer']:>6} {r['n_claims_scored']:>7} {r['target_wins']:>7} "
                          f"{r['decoy_wins']:>6} {r['rate']:>6.1%} {r['p']:>8.4f} "
                          f"{r['tied_claims']:>5}")
        print("\n  The global decoy almost always names a file from a DIFFERENT repository (71 of")
        print("  75 cases), and the residual encodes which repository it is reading. Read the")
        print("  same-repo rows; the global rows are printed to show the size of that shortcut.")

    # MERGE, don't overwrite. --only q1 and --only q2 are run separately because Q2's permutation
    # null costs half an hour; clobbering would mean the file never holds both halves at once.
    if OUT.is_file():
        try:
            prev = json.loads(OUT.read_text(encoding="utf-8"))
            for k, v in prev.items():
                if v and not res.get(k):
                    res[k] = v
        except json.JSONDecodeError:
            pass
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print("Read every row against its null, not against zero: at this n a probe separates "
          "shuffled labels too.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
