"""Emit the Colab notebook that replicates follow-up Q1 and Q2 from the raw residual arrays.

Same standard as verify_findings.ipynb: nothing is asserted, every number is recomputed in front of
the reader from files they can inspect. The difference is that these two questions are PROBES, and
a probe on 38 claims at 4,096 dimensions will report a result whether or not one exists. So the
notebook spends as much of its length on the null as on the finding, and computes the naive
random-split score next to the grouped one so the reader can see the inflation instead of taking
my word that grouping matters.

Run: python verify/make_probe_notebook.py
Then upload verify/probe_q1_q2.ipynb with the five artifacts listed in its first cell.
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "probe_q1_q2.ipynb"


def md(t: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip().splitlines(True)}


def code(t: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.strip().splitlines(True)}


CELLS = [
    md(r"""
# Is the answer in there? Two probes and their nulls

The lens pilot asked whether a **published** readout — a Jacobian lens — recovers the file the
model is about to name, at the cut before it names it. It does not, at the sample size available.

That leaves a question the pilot could not answer, because it recorded the lens's *verdict* (top-20
token ids) and not the lens's *input*. Two questions, in fact:

**Q1 — is the answer linearly decodable at all?** Not by the published lens, but by any linear
readout trained on this task. A negative lens result is consistent with two very different worlds:
the information is not there, or it is there and that particular lens does not read it.

**Q2 — do right and wrong relocations look different?** When the model relocates a vulnerability to
a file, sometimes that file is in the fix commit and sometimes it is not. If the residual at the
cut separates the two, there is something to build a gate on. If it does not, the model's internal
state at that moment does not know it is about to be wrong.

Both need the residual vectors themselves, so they were re-extracted with `output_hidden_states=True`
— no lens involved.

**Upload these next to the notebook:**

| file | what it is |
|---|---|
| `resid.npz` | residual stream at the last position, every layer, fp16, one array per readout |
| `resid_meta.json` | what each array is: case, cut point, run, target file, repo, sha |
| `reloc_labels.json` | per case, whether the relocated file is in the fix commit (gold from `vulns.db`) |
| `tok_emb.npz` | unembedding rows for every target and decoy filename token |
| `tok_emb_names.json` | filename → token ids, and the seeded decoy pairing |
| `reloc_cases_75.json` | the corrected case export — repo, sha, target, and both cut points |

No GPU and no model. The forward passes already happened.

---

### Why most of this notebook is about the null

There are ~38 distinct claims and 2,048 dimensions per layer. A linear classifier in that regime
separates **random** labels nearly perfectly. Any number reported here is meaningless without the
matching number from shuffled labels, so every cell computes both.

Three specific precautions, each demonstrated below rather than promised:

1. **Leave-one-claim-out, not leave-one-row-out.** One claim appears as several mentions and at two
   cut points. A random split puts near-duplicates of the test row in the training set. Cell 4
   computes the naive split too — the gap between the two numbers *is* the artefact.
2. **A permutation null.** The reported p is the fraction of label shuffles whose cross-validated
   score beats the real one. The null is measured on this data, not assumed to be 0.5.
3. **PCA fitted inside each fold.** Fitting it once on everything leaks the test fold into the
   projection.
"""),

    code(r"""
import json, pathlib, numpy as np

def need(name):
    p = pathlib.Path(name)
    if not p.exists():
        raise SystemExit(f"missing {name} — upload it next to this notebook")
    return p

Z      = np.load(need("resid.npz"))
META   = json.loads(need("resid_meta.json").read_text(encoding="utf-8"))
LABELS = {r["case"]: r for r in json.loads(need("reloc_labels.json").read_text(encoding="utf-8"))}
EMB    = np.load(need("tok_emb.npz"))
NAMES  = json.loads(need("tok_emb_names.json").read_text(encoding="utf-8"))

N_LAYERS = META[0]["n_layers"]; D = META[0]["d_model"]
print(f"{len(META)} readouts | {N_LAYERS} layers | d_model {D}")
print(f"{len(LABELS)} cases carry a gold label")
cuts = {}
for m in META: cuts[m["cut"]] = cuts.get(m["cut"], 0) + 1
print("cut points:", cuts)

# The harness keeps BerriAI/litellm as an untouched holdout: no number from it may enter an
# aggregate. Asserted here rather than promised, because a silent leak would invalidate everything
# below and nothing else in this notebook would notice.
held = sorted({r["repo"] for r in LABELS.values() if "litellm" in r["repo"].lower()})
print(f"\nholdout repos present: {held or 'none'}  "
      f"{'FAIL — holdout leaked into the probe corpus' if held else 'PASS'}")
import collections as _c
print("\nrepos:", dict(_c.Counter(r["repo"] for r in LABELS.values()).most_common(5)), "...")
"""),

    md(r"""
## 1. The unit of analysis

The pilot's first version of this analysis counted **mentions** and got a result that shrank when
counted by **claim**. One claim was restated seven times; seven correlated rows were being treated
as seven independent observations.

A claim here is `(task, relocated file)`. Everything downstream groups by it.

### And the cut has to be genuinely earlier

Each case has two cut points: `prefix_at_sentence`, before the clause that names the file, and
`prefix_at_path`, mid-sentence just after the model has typed the opening backtick. For **12 of 75**
mentions the miner found no earlier sentence boundary and fell back, so the two prefixes are
byte-identical — and a readout there is taken *after* naming has begun, which is precisely what Q1
is meant to be blind to.

This was one of the three corrections in the original pilot, and it is the one that changed the
conclusion. Those readouts are excluded below. Set `DROP_IDENTICAL = False` to put them back and
watch the numbers improve for the wrong reason.

### One more choice that had to be made explicit

Q2's labels come from matching the relocated file against the gold files for that `(repo, sha)`.
Matching on the **exact repo-relative path** gives 51 of 75 mentions correct, and **21 correct / 17
wrong** across the 38 claims. Also accepting a **bare filename** match gives 59 of 75, and **27 /
11**.

The looser rule would hand the classifier a much easier class balance, manufactured by the scoring
rule rather than by the model. Everything below uses the strict labels. `label_relocations.py
--allow-basename` regenerates the loose set if you want to check what turns on it.
"""),

    code(r"""
def claim_of(m):
    lab = LABELS.get(m["case"])
    path = lab["target_path"] if lab else m["target"]
    return f'{m["run_id"].split("-eval")[0]}::{path}'

AT = "prefix_at_sentence"   # the cut BEFORE the model names the file
DROP_IDENTICAL = True       # see the note below

def usable(m):
    if m["cut"] != AT or m["case"] not in LABELS: return False
    return not (DROP_IDENTICAL and LABELS[m["case"]].get("cut_identical"))

rows = [m for m in META if usable(m)]
claims = sorted({claim_of(m) for m in rows})
ident = sum(1 for m in META if m["cut"] == AT and m["case"] in LABELS
            and LABELS[m["case"]].get("cut_identical"))
print(f"{len(rows)} readouts at '{AT}' → {len(claims)} distinct claims")
print(f"({ident} readouts excluded: their 'before the sentence' cut is byte-identical to the "
      "mid-sentence one, so they sit after naming has begun)")

by = {}
for m in rows: by.setdefault(claim_of(m), []).append(m)
print("readouts per claim:", sorted((len(v) for v in by.values()), reverse=True))

lab_by_claim = {c: LABELS[v[0]["case"]]["correct"] for c, v in by.items()}
ncorr = sum(lab_by_claim.values())
print(f"\ncorrect relocations: {ncorr} / {len(claims)} claims   "
      f"(wrong: {len(claims)-ncorr})")
"""),

    md(r"""
## 2. Building the design matrix

One row per readout, at one layer. `X` is the residual, `y` is whether the relocation turned out to
be right, `g` is the claim it belongs to.
"""),

    code(r"""
def design(layer):
    X, y, g = [], [], []
    for m in META:
        if not usable(m): continue
        X.append(Z[m["key"]][layer].astype(np.float32))
        y.append(1 if LABELS[m["case"]]["correct"] else 0)
        g.append(claim_of(m))
    return np.stack(X), np.array(y), np.array(g)

X, y, g = design(N_LAYERS // 2)
print("X", X.shape, "| positives", int(y.sum()), "| groups", len(set(g)))
print(f"\n{X.shape[1]} dimensions and {len(set(g))} independent units. "
      "That ratio is why the null matters more than the number.")
"""),

    md(r"""
## 3. Scoring, with the fold rule as a parameter

`cv_auc` takes the grouping as an argument so the next cell can run the *same* classifier under the
honest rule and the naive one and print both.

A fold whose training labels collapse to one class is skipped rather than scored — predicting the
majority there is not evidence either way.
"""),

    code(r"""
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def auc(truth, pred):
    truth = np.asarray(truth); pred = np.asarray(pred)
    if len(set(truth)) < 2: return 0.5
    order = np.argsort(pred); r = np.empty(len(pred)); r[order] = np.arange(1, len(pred)+1)
    pos = truth == 1; n1, n0 = pos.sum(), (~pos).sum()
    return (r[pos].sum() - n1*(n1+1)/2) / (n1*n0)

def cv_auc(X, yy, groups, n_comp=10, C=0.05):
    preds, truth = [], []
    for u in np.unique(groups):
        te = groups == u; tr = ~te
        if len(set(yy[tr])) < 2: continue
        k = min(n_comp, int(tr.sum()) - 1, X.shape[1])
        pipe = make_pipeline(StandardScaler(), PCA(n_components=k),
                             LogisticRegression(C=C, max_iter=2000))
        pipe.fit(X[tr], yy[tr])
        preds += list(pipe.predict_proba(X[te])[:, 1]); truth += list(yy[te])
    return auc(truth, preds)

print(f"grouped by claim : AUC {cv_auc(X, y, g):.3f}")
"""),

    md(r"""
## 4. What the naive split buys you

Same data, same classifier, same regularisation. The only change is that folds are drawn over
**rows** instead of over claims, so a mention of a claim can sit in training while another mention
of the same claim sits in test.

If the two numbers differ, the difference is leakage, not signal.
"""),

    code(r"""
naive = np.arange(len(y))            # every row its own "group" = leave-one-row-out
print(f"leave-one-ROW-out (naive)  : AUC {cv_auc(X, y, naive):.3f}   <- inflated by duplicates")
print(f"leave-one-CLAIM-out (used) : AUC {cv_auc(X, y, g):.3f}")
print("\nThe first number is the one a careless version of this analysis would have reported.")
"""),

    md(r"""
## 5. Q2 — right vs wrong, against a permutation null

500 label shuffles per layer. `p` is the fraction of shuffles that scored at least as high as the
real labels, which is the only interpretation of these AUCs that survives the dimensionality.
"""),

    code(r"""
rng = np.random.default_rng(20260911)
PERMS = 500
probe_layers = sorted({int(round(v)) for v in np.linspace(0, N_LAYERS-1, 8)})

q2 = []
print(f"{'layer':>6} {'AUC':>6} {'null mean':>10} {'null p95':>9} {'p':>7}")
for L in probe_layers:
    X, y, g = design(L)
    real = cv_auc(X, y, g)
    null = np.array([cv_auc(X, rng.permutation(y), g) for _ in range(PERMS)])
    p = ((null >= real).sum() + 1) / (PERMS + 1)
    q2.append({"layer": L, "auc": real, "null_mean": null.mean(),
               "null_p95": np.quantile(null, .95), "p": p})
    print(f"{L:>6} {real:>6.3f} {null.mean():>10.3f} {np.quantile(null,.95):>9.3f} {p:>7.3f}")
"""),

    code(r"""
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 3.6))
L = [r["layer"] for r in q2]
ax.fill_between(L, [r["null_mean"] for r in q2], [r["null_p95"] for r in q2],
                color="#bbb", alpha=.5, label="permutation null (mean to 95th pct)")
ax.plot(L, [r["auc"] for r in q2], "o-", color="#b5462f", label="real labels")
ax.axhline(.5, color="#888", lw=.8, ls=":")
ax.set_xlabel("layer"); ax.set_ylabel("cross-validated AUC")
ax.set_title("Q2: does the residual know the relocation is wrong?")
ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.show()
print("Read the curve against the grey band, not against 0.5.")
"""),

    md(r"""
## 6. Q1 — target vs decoy, from a readout trained on the task

A ridge regression from the residual to the mean unembedding vector of the target filename's
tokens, trained leave-one-claim-out. For each held-out readout, the prediction is scored by cosine
against the target and against **the same seeded decoy** the Jacobian lens and the ask-the-model
baseline were scored against — so all three experiments answer the identical question and their
numbers sit on one axis.

The decoy is another case's real target filename, so a readout that has learned "filenames look
like this" and nothing more scores 50%.

**One wrinkle the cell handles explicitly.** The 75 cases use only 21 distinct basenames — plenty
of these repositories have their own `utils.py` — so a decoy drawn from that pool is sometimes the
*same filename* as the target it was paired against. Those readouts cannot favour either side and
are dropped rather than counted; the count is printed, because leaving them in would pad the
denominator with guaranteed non-evidence.
"""),

    code(r"""
from sklearn.linear_model import Ridge

names, decoys = NAMES["names"], NAMES["decoys"]
def vec(name):
    ids = names.get(name) or []
    r = [EMB[str(t)].astype(np.float32) for t in ids if str(t) in EMB]
    return np.mean(r, axis=0) if r else None

def q1(layer, alpha=1e4):
    X, Yt, Yd, g = [], [], [], []
    same = 0
    for m in META:
        if not usable(m): continue
        t = vec(m["target"]); d = vec(decoys[m["case"]]) if m["case"] < len(decoys) else None
        if t is None or d is None: continue
        if m["case"] < len(decoys) and decoys[m["case"]] == m["target"]:
            same += 1; continue     # decoy IS the target filename -- no signal either way
        X.append(Z[m["key"]][layer].astype(np.float32)); Yt.append(t); Yd.append(d)
        g.append(claim_of(m))
    if same: globals()["_dropped_same"] = same
    X, Yt, Yd, g = np.stack(X), np.stack(Yt), np.stack(Yd), np.array(g)
    hit = mis = 0
    for u in np.unique(g):
        te = g == u; tr = ~te
        if tr.sum() < 4: continue
        pred = Ridge(alpha=alpha).fit(X[tr], Yt[tr]).predict(X[te])
        for k, row in enumerate(pred):
            cos = lambda a, b: float(a @ b / ((np.linalg.norm(a)+1e-9)*(np.linalg.norm(b)+1e-9)))
            st, sd = cos(row, Yt[te][k]), cos(row, Yd[te][k])
            hit += st > sd; mis += sd > st
    return hit, mis

from math import comb
print(f"{'layer':>6} {'target':>7} {'decoy':>6} {'rate':>7} {'p (exact binomial)':>20}")
q1rows = []
for L in probe_layers:
    h, m = q1(L); n = h + m
    if not n: continue
    k = max(h, m)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k, n+1)) / 2**n)
    q1rows.append({"layer": L, "hit": h, "miss": m, "rate": h/n, "p": p})
    print(f"{L:>6} {h:>7} {m:>6} {h/n:>6.1%} {p:>20.4f}")
print(f"\nreadouts dropped because the decoy was the same filename as the target: "
      f"{globals().get('_dropped_same', 0)}")
"""),

    md(r"""
## 6b. The control that turned out not to control for much

Every number above, and both published numbers in the table below, score the target against a
**seeded global decoy** — another case's real target filename, drawn from the whole corpus.

That control is weaker than it looks, and the next cell measures how much weaker.

Relocated filenames are almost perfectly nested inside repositories. If nearly every decoy names a
file from a *different project* than the one being read, then "prefers the target over the decoy"
can be satisfied by recognising the repository — which every method under test has free access to,
because the whole context is that project's source code.

The fix is a decoy drawn from **the same repository** as the target. Repository identity then
favours both candidates equally, and anything above chance has to come from telling files apart
*within* a project. It costs most of the sample, and the cell reports exactly how much.
"""),

    code(r"""
import collections, random

cases = json.loads(_p.Path("reloc_cases_75.json").read_text(encoding="utf-8"))[:75] \
        if _p.Path("reloc_cases_75.json").exists() else None

# Derive the decoy pairing from the seed rather than trusting the uploaded artifact, so this cell
# runs whether or not tok_emb_names.json is present -- and so a drift between the two is visible.
def seeded_decoys(cs):
    rng = random.Random(20260906)
    t = [c["target_basename"] for c in cs]; d = t[:]
    rng.shuffle(d)
    for i in range(len(d)):
        if d[i] == t[i] and len(set(t)) > 1:
            j = (i + 1) % len(d); d[i], d[j] = d[j], d[i]
    return d

if cases is not None:
    decoys = seeded_decoys(cases)
    if "NAMES" in dir() and NAMES.get("decoys") and NAMES["decoys"] != decoys:
        print("WARNING: uploaded decoy pairing disagrees with the seed — the experiments were "
              "not scored against the same control")

if cases is None:
    print("reloc_cases.json not uploaded — skipping the decoy-scope audit.")
else:
    repos_of = collections.defaultdict(set)
    for c in cases: repos_of[c["target_basename"]].add(c["repo"])
    cross = [i for i, c in enumerate(cases) if c["repo"] not in repos_of[decoys[i]]]
    by_repo = collections.defaultdict(set)
    for c in cases: by_repo[c["repo"]].add(c["target_basename"])
    multi = {r: sorted(f) for r, f in by_repo.items() if len(f) >= 2}

    print(f"distinct relocated basenames            {len(repos_of)}")
    print(f"  occurring in exactly one repository   "
          f"{sum(1 for f in repos_of.values() if len(f)==1)}")
    print(f"decoys naming a file from ANOTHER repo  {len(cross)} / {len(cases)}")
    print(f"repos hosting >=2 distinct relocations  {len(multi)} / {len(by_repo)}")
    for r, f in sorted(multi.items(), key=lambda kv: -len(kv[1])):
        print(f"      {len(f)}  {r}")
    print("\nA decoy from another project is near-unreachable by construction, so the floor it")
    print("reports is too low and the gap above it is too flattering.")
"""),

    md(r"""
## 7. The three experiments on one axis

Holding the decoy pairing fixed across all three makes the table internally consistent: each row is
"how often does this method prefer the file the model actually named over an unrelated file", at
the same cut on the same claims.

After §6b, read the *decoy* column as a floor that is too low rather than as a noise estimate. What
survives unchanged is the **target** column — for the baseline, the number of claims where the
model named the exact repository-relative path, out of the many files it could have named instead.
The lens is unaffected in direction: a floor that is too low could only have flattered it, and it
was null anyway.
"""),

    code(r"""
best = max(q1rows, key=lambda r: r["rate"]) if q1rows else None
print(f"{'method':<38} {'n':>4} {'target':>7} {'decoy':>6} {'p':>9}")
print("-" * 68)
print(f"{'ask the model outright':<38} {30:>4} {18:>7} {3:>6} {0.0003:>9.4f}")
print(f"{'Jacobian lens (distinctive tokens)':<38} {31:>4} {11:>7} {5:>6} {0.146:>9.3f}")
if best:
    n = best["hit"] + best["miss"]
    print(f"{'trained ridge probe (best layer)':<38} {n:>4} {best['hit']:>7} "
          f"{best['miss']:>6} {best['p']:>9.4f}")
print()
print("The first two rows are reproduced from verify_findings.ipynb; the third is computed above.")
print("Note the first row is not an interpretability result -- it is the model answering a")
print("question. It is here as the ceiling the readouts are being measured against.")
"""),

    md(r"""
## 8. The best single dimension, and what it is worth

A standard move when looking for a behaviour in activations: score every unit by how well it
separates the event, find units with high ROC-AUC and large Cohen's *d*, observe that they fire in
the wrong places on held-out data, and conclude polysemanticity.

The step that usually goes unmeasured is what that procedure returns when there is **provably
nothing to find**. With 2,048 dimensions and 35 claims, the best-looking dimension is the best of
2,048 draws from noise, and its in-sample AUC is a selection artifact.

So this cell computes three numbers per layer, and only the gaps between them carry information:

- **in-sample** — the best dimension's AUC on the claims used to pick it
- **held-out** — that same dimension, re-scored on claims it was not selected on
- **shuffled** — the identical procedure with labels permuted

If in-sample sits near shuffled, polysemanticity is not the explanation; there was never a signal.
"""),

    code(r"""
def auc_cols(X, y):
    order = np.argsort(X, axis=0)
    ranks = np.empty_like(order, dtype=np.float64)
    rows_ = np.arange(1, X.shape[0] + 1)[:, None]
    np.put_along_axis(ranks, order, np.broadcast_to(rows_, X.shape), axis=0)
    pos = y == 1; n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0: return np.full(X.shape[1], .5)
    return (ranks[pos].sum(axis=0) - n1*(n1+1)/2) / (n1*n0)

def best_dim(X, y, g, r):
    uniq = np.unique(g); pick = r.permutation(uniq); half = pick[:len(pick)//2]
    tr = np.isin(g, half); te = ~tr
    if len(set(y[tr])) < 2 or len(set(y[te])) < 2: return None
    atr = auc_cols(X[tr], y[tr]); j = int(np.argmax(np.abs(atr - .5)))
    col = (-X[:, j:j+1]) if atr[j] < .5 else X[:, j:j+1]
    return float(auc_cols(col[tr], y[tr])[0]), float(auc_cols(col[te], y[te])[0])

r2 = np.random.default_rng(20260911)
print(f"{'layer':>6} {'in-sample':>10} {'held-out':>9} {'shuffled':>9}")
sweep = []
for L in probe_layers:
    X, y, g = design(L)
    got = best_dim(X, y, g, r2)
    if not got: continue
    ins, held = got
    null = [v[0] for v in (best_dim(X, r2.permutation(y), g, r2) for _ in range(200)) if v]
    sweep.append((L, ins, held, float(np.mean(null))))
    print(f"{L:>6} {ins:>10.3f} {held:>9.3f} {np.mean(null):>9.3f}")
print("\nCompare in-sample against SHUFFLED, not against 0.5.")
"""),

    md(r"""
## 9. Is the *event* decodable? (needs the control extraction)

Everything above asks what the model was about to **say**. This asks whether it was about to
**change its mind at all** — the question the backtracking literature asks of reasoning traces,
except that a relocation has a gold answer and a "Wait" does not.

- **positives** — the residual at the sentence boundary before a relocation
- **negatives** — a sentence boundary in the *same run*, in a step containing no relocation,
  chosen to match the positive's prefix length

The matching is what makes it a control, and it took three attempts to become one. Each failure
produced a confident-looking number first:

| shortcut | what it was worth | fix |
|---|---|---|
| prompt length | AUC **0.62** from token count alone | cap the within-pair token gap at 30% |
| punctuation | AUC **0.894 at layer 0** | cut both exports on one convention |
| leftover punctuation skew | ~**0.68** from the last token | stratify on final character, then match length |

The second one is the instructive one. **Layer 0 is the embedding output — no transformer block has
run.** A separation there cannot be a fact about computation; it can only be a fact about the last
token. It was: the relocation export ends its prefix on the punctuation mark, the control export
kept the trailing space, so relocations ended on `.` 45 times in 75 and controls never did. One
character made the classes linearly separable before the model did anything.

The cell below therefore prints the token-count-only AUC for each candidate subset *before* it
reports anything about activations, and you should read the layer-0 row of the results table as a
diagnostic rather than as a finding. Folds are leave-one-**run**-out, because a positive and its
matched negative share a run and would otherwise sit on opposite sides of a split.
"""),

    code(r"""
import pathlib as _p
if not _p.Path("resid_ctrl.npz").exists():
    print("resid_ctrl.npz not uploaded — skipping the event probe.")
else:
    ZC = np.load("resid_ctrl.npz")
    MC = json.loads(_p.Path("resid_ctrl_meta.json").read_text(encoding="utf-8"))

    ctl_by_case = {m["matched_case"]: m for m in MC}

    def design_event(layer, only=None):
        # only the controls whose matched positive survived the filters; an unmatched negative
        # breaks the pairing the length and punctuation matching both depend on
        keep, X, y, g, tok = set(), [], [], [], []
        for m in META:
            if not usable(m): continue
            if only is not None and m["case"] not in only: continue
            keep.add(m["case"])
            X.append(Z[m["key"]][layer].astype(np.float32))
            y.append(1); g.append(m["run_id"]); tok.append(m["tokens"])
        for m in MC:
            if m.get("matched_case") not in keep: continue
            X.append(ZC[m["key"]][layer].astype(np.float32))
            y.append(0); g.append(m["run_id"]); tok.append(m["tokens"])
        return np.stack(X), np.array(y), np.array(g), np.array(tok, dtype=float)

    # the two shortcuts, as subsets
    pos_tok = {m["case"]: m["tokens"] for m in META if usable(m)}
    lb = {c for c, t in pos_tok.items() if c in ctl_by_case
          and abs(t - ctl_by_case[c]["tokens"]) / max(t, ctl_by_case[c]["tokens"]) <= 0.30}
    fc = {c for c in pos_tok if c in ctl_by_case and cases is not None
          and cases[c]["prefix_at_sentence"][-1:]
          == ctl_by_case[c]["prefix_at_sentence"][-1:]} if cases else set(pos_tok)
    both = lb & fc

    for name, sub in (("all pairs", None), ("within 30% token length", lb),
                      ("final character matched", fc), ("both restrictions", both)):
        X, y, g, tok = design_event(N_LAYERS // 2, sub)
        print(f"{name:<28} pairs {int(y.sum()):>3} | length-only AUC {auc(y, tok):.3f}")

    X, y, g, tok = design_event(N_LAYERS // 2, both)
    if int(y.sum()) < 6:
        print(f"\nBoth restrictions leave {int(y.sum())} pairs — too few to score.")
    else:
        print(f"\nscoring the doubly-restricted subset (length shortcut worth {auc(y,tok):.3f})")
        print("WATCH LAYER 0 — it is the embedding output, before any block has run.\n")
        print(f"{'layer':>6} {'AUC':>6} {'null mean':>10} {'null p95':>9} {'p':>7}")
        ev = []
        for L in probe_layers:
            X, y, g, _ = design_event(L, both)
            def pshuf(y, g):
                yy = y.copy()
                for u in np.unique(g): yy[g == u] = r2.permutation(y[g == u])
                return yy
            real = cv_auc(X, y, g)
            null = np.array([cv_auc(X, pshuf(y, g), g) for _ in range(300)])
            p = ((null >= real).sum() + 1) / 301
            ev.append({"layer": L, "auc": real, "null": null.mean(), "p": p})
            print(f"{L:>6} {real:>6.3f} {null.mean():>10.3f} "
                  f"{np.quantile(null,.95):>9.3f} {p:>7.3f}")
"""),

    md(r"""
## 10. What this does and does not establish

Whatever the cells above print, the honest reading is bounded by n. ~38 claims is a pilot, and a
pilot's job is to size the real experiment, not to settle the question.

Specifically:

- **Q1 is not answerable on this corpus at all**, and that is the most useful thing the follow-up
  produced. Deconfounding the decoy leaves a handful of scoreable claims, because relocated
  filenames are nested inside repositories. A corpus built to answer it would need within-repository
  relocation pairs by construction — which is a design requirement discovered by measurement rather
  than guessed at.
- A **null** result for Q1 at this n does not show the information is absent. It shows that a linear
  readout of this form, at this sample size, does not recover it — which is a statement about the
  readout and the power, jointly.
- A **positive** result for Q2 that clears its permutation null is worth following, but it is one
  corpus, one model, and one cut point. The next step is a held-out repo set, not a stronger claim.
- The ask-the-model baseline outperforming both readouts is itself the most interesting line in the
  table, and it is the one that needs the least statistical defence.

The reason to run it this way — grouped folds, permutation nulls, the naive number printed beside
the honest one — is that the failure mode here is not being wrong. It is being unfalsifiably right.
"""),
]


def main() -> int:
    nb = {"cells": CELLS, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5}
    OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
