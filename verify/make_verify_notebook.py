"""Emit a Colab notebook that re-derives every claimed number from the raw artifacts.

The point is that nothing here is taken on trust. Each cell loads a file, recomputes a figure from
scratch, and prints it beside the value that was claimed, with an explicit PASS/FAIL. Where a
number was corrected, both the old and the new value are computed so the correction is visible
rather than asserted.

No GPU. No model. The expensive part -- reading a Jacobian lens at 39 layers over 60 forward
passes -- already happened, and its RAW output is in reloc_results.json: the top-20 token ids at
every layer for both lenses, plus the token sets for each target and decoy. Every scoring rule in
the report is a re-score of that file, which is exactly why it was recorded that way.

Run: python verify/make_verify_notebook.py
Then open verify/verify_findings.ipynb in Colab and Run all: it fetches its own inputs.
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / "verify_findings.ipynb"


def md(t: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip().splitlines(True)}


def code(t: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.strip().splitlines(True)}


CELLS = [
    md(r"""
# Verify the findings

Every number in the report and on the site, recomputed here from raw files. Nothing is asserted;
each cell prints what it computed next to what was claimed and marks it PASS or FAIL.

**Nothing to upload.** The cell below fetches these from `verify/` in the repository;
drop your own copies beside the notebook to override them.

| file | what it is |
|---|---|
| `reloc_cases.json` | first export: 30 relocation cases, cut from the visible message |
| `reloc_results.json` | first export's raw lens readout — top-20 ids at all 39 layers, both lenses |
| `gold_pilot.json` | ground truth, taken from the harness's own `vulns.db` |
| `reloc_cases_75.json` | **corrected export**: 75 cases, cut from `reasoning_content` |
| `reloc_rows_75.jsonl` | **the lens run that matters** — raw per-layer ids, 149 rows, all 75 cases |
| `ask_baseline_75.json` | the model asked outright, same cases, same decoys |
| `relocation_reach.json` | every mined relocation: run, path, whether it reached the casefile, prompt era |

No GPU and no model needed. The lens has already been read; this re-scores its output.

**Three corrections are reproduced below rather than hidden.** In each case the original number
was computed on a unit or a subset that turned out not to be what it claimed:

1. **mention vs claim** — 30 mentions are 17 distinct claims; one is restated 7 times.
2. **any token vs distinctive tokens** — two unrelated filenames share `.py`, which scored a hit
   for the target and the decoy at once.
3. **all cuts vs genuinely earlier cuts** — for 7 of 17 claims `prefix_at_sentence` is
   byte-identical to `prefix_at_path`, so the "before the sentence" measurement was partly taken
   after the sentence had started.

The third is the one that changes the conclusion.
"""),

    code(r"""
import json, math, collections, pathlib, urllib.request

# Inputs come from the repository unless they are already sitting next to the notebook. A cold
# Colab runtime starts empty, and "upload these seven files first" is the step that gets skipped —
# so Run-all works from a fresh runtime with no setup. Drop the files in beside the notebook and
# they win, which keeps this usable offline and against modified inputs.
RAW = "https://raw.githubusercontent.com/nickmccarty/mats-app/main/verify/"

def fetch(name):
    p = pathlib.Path(name)
    if not p.exists():
        print(f"  fetching {name}")
        urllib.request.urlretrieve(RAW + name, p)
    return p

def load(name):
    return json.loads(fetch(name).read_text(encoding="utf-8"))

cases = load("reloc_cases.json")
rows  = load("reloc_results.json")
gold  = load("gold_pilot.json")

print(f"cases  {len(cases)} mentions")
print(f"rows   {len(rows)} readouts  ({len(rows)//2} cases x 2 cut points)")
print(f"gold   {len(gold)} tasks")
print(f"layers {rows[0]['layers']}")

CHECKS = []
def check(label, got, claimed):
    ok = (got == claimed)
    CHECKS.append((label, got, claimed, ok))
    print(f"{'PASS' if ok else 'FAIL'}  {label:52} computed={got!s:14} claimed={claimed!s}")
    return ok
"""),

    md(r"""
## 1. The unit: 30 mentions are 17 claims

A "mention" is one occurrence of a relocation in a transcript. A "claim" is a distinct
(task, file) assertion. Counting mentions treats one file named seven times as seven pieces of
evidence.
"""),

    code(r"""
claim_key = lambda r: (r["run_id"].split("-eval")[0], r["target"])
claims = {claim_key(r) for r in rows}
per_name = collections.Counter(c["target_basename"] for c in cases)

check("mentions", len(cases), 30)
check("distinct claims", len(claims), 17)
check("most-repeated file appears N times", per_name.most_common(1)[0][1], 7)
print("\nrepetition per file:")
for n, k in sorted(((v, k) for k, v in per_name.items()), reverse=True):
    print(f"  {n:>2}x  {k}")
"""),

    md(r"""
## 2. Ground truth comes from the harness's own database

Not from anything computed here. `gold_pilot.json` is a straight extract of `vulns.db` for the
nine tasks in the pilot: gold files, gold line ranges, the fix commit and the advisory id.
"""),

    code(r"""
key_of = lambda c: f'{c["repo"]}@{c["sha"]}'
missing = [key_of(c) for c in cases if key_of(c) not in gold]
check("cases with gold present", len(cases) - len(missing), 30)

# the case the demo video is built on
demo = next(c for c in cases if "pyload" in c["repo"] and "cnl_blueprint" in c["target_path"])
g = gold[key_of(demo)]
print(f"\ndemo task: {demo['repo']} @ {demo['sha'][:9]}")
# `.get`, not `[...]`: this line only prints provenance, and the gold export has been regenerated
# once since without a `cwe` field. A missing display value stopped the whole verification run
# partway through, which is the one thing a verification notebook must not do quietly.
print(f"  advisory   {g.get('ghsa', '?')}  ({g.get('cwe', 'CWE not in this export')})")
print(f"  fix commit {g['fix_sha']}")
print(f"  gold files {g['gold_files']}")
print(f"  gold lines {g['gold_ranges']}")
check("demo gold file is the relocated file",
      demo["target_path"] in g["gold_files"], True)
"""),

    md(r"""
## 3. Scoring the lens, three ways

A readout "hits" when a token of the filename appears in the top 20 at any layer. Two knobs
change the answer, and both are reproduced:

- **any token** vs **distinctive only** — drop tokens the target and decoy share, from *both*
  sides. A one-sided version (void the decoy, keep the target) is also computed, to show why it
  is not licensed: it corrects only the side that hurts the hypothesis.
- **all claims** vs **genuinely earlier cut** — exclude claims where `prefix_at_sentence` equals
  `prefix_at_path`.
"""),

    code(r"""
# Two-sided exact test on the discordant pairs. Target and decoy share a forward pass, so they
# are paired; a two-proportion z-test would treat them as independent and overstate significance.
def mcnemar_exact(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)

# which cases have a cut that is genuinely earlier than the at_path cut
earlier_cases = {i for i, c in enumerate(cases)
                 if c["prefix_at_sentence"] != c["prefix_at_path"]}
print(f"cases with a genuinely earlier cut: {len(earlier_cases)} of {len(cases)}")

def score(cut, distinctive=True, only=None, one_sided=False):
    out = {}
    for r in rows:
        if r["cut"] != cut: continue
        if only is not None and r["case"] not in only: continue
        tgt, dec = set(r["target_tok"]), set(r["decoy_tok"])
        shared = tgt & dec
        if distinctive:
            tgt = tgt - shared
            dec = dec - shared
        d = out.setdefault(claim_key(r), {"t": 0, "d": 0, "l": 0})
        for L in r["layer_ids"]:
            j  = set(r["j_top20"][str(L)])
            lg = set(r["l_top20"][str(L)])
            d["t"] |= 1 if j & tgt else 0
            hit_d = 1 if j & dec else 0
            if one_sided and shared: hit_d = 0     # void decoy on shared-token rows only
            d["d"] |= hit_d
            d["l"] |= 1 if lg & tgt else 0
    n = len(out)
    t = sum(v["t"] for v in out.values()); dd = sum(v["d"] for v in out.values())
    l = sum(v["l"] for v in out.values())
    b = sum(1 for v in out.values() if v["t"] and not v["d"])
    c = sum(1 for v in out.values() if v["d"] and not v["t"])
    return {"n": n, "target": t, "decoy": dd, "logit": l, "p": mcnemar_exact(b, c)}

print(f"\n{'scoring':46} {'n':>3} {'tgt':>4} {'dec':>4} {'logit':>6} {'p':>8}")
for label, kw in [
    ("at_sentence, any token, all claims",      dict(cut="prefix_at_sentence", distinctive=False)),
    ("at_sentence, distinctive, all claims",    dict(cut="prefix_at_sentence")),
    ("at_sentence, distinctive, ONE-SIDED",     dict(cut="prefix_at_sentence", one_sided=True)),
    ("at_sentence, distinctive, EARLIER ONLY",  dict(cut="prefix_at_sentence", only=earlier_cases)),
    ("at_path,     distinctive, all claims",    dict(cut="prefix_at_path")),
    ("at_path,     distinctive, EARLIER ONLY",  dict(cut="prefix_at_path", only=earlier_cases)),
]:
    s = score(**kw)
    print(f"{label:46} {s['n']:>3} {s['target']:>4} {s['decoy']:>4} {s['logit']:>6} {s['p']:>8.4f}")
"""),

    md(r"""
### What those rows say

The first row is the number that appeared in the report and on the site. The fourth is the same
measurement restricted to claims whose "before the sentence" cut is actually before the sentence.

**The correction is the fourth row, and it removes the result.**
"""),

    code(r"""
s_all     = score("prefix_at_sentence")
s_earlier = score("prefix_at_sentence", only=earlier_cases)
p_all     = score("prefix_at_path")

check("at_sentence all-claims target",   f"{s_all['target']}/{s_all['n']}",         "9/17")
check("at_sentence all-claims decoy",    f"{s_all['decoy']}/{s_all['n']}",          "3/17")
check("at_sentence all-claims p",        round(s_all['p'], 4),                       0.1094)
check("at_sentence EARLIER target",      f"{s_earlier['target']}/{s_earlier['n']}", "4/12")
check("at_sentence EARLIER decoy",       f"{s_earlier['decoy']}/{s_earlier['n']}",  "2/12")
check("at_sentence EARLIER p",           round(s_earlier['p'], 4),                   0.6875)
check("at_path target",                  f"{p_all['target']}/{p_all['n']}",         "16/17")
check("at_path decoy (distinctive)",     f"{p_all['decoy']}/{p_all['n']}",          "2/17")
"""),

    md(r"""
## 4. The decoy is silent unless it shares a token

The claim was that name *length* drives spurious decoy hits. It does not — that hypothesis is
computed and rejected below. What drives them is a shared token, always `.py`.
"""),

    code(r"""
share, nshare = [], []
for r in rows:
    (share if (set(r["target_tok"]) & set(r["decoy_tok"])) else nshare).append(r)

def decoy_stats(rs):
    fires = sum(1 for r in rs if r["j_decoy"] > 0)
    mean  = sum(r["j_decoy"] for r in rs) / max(len(rs), 1)
    return len(rs), fires / max(len(rs), 1), mean

for label, rs in (("shares a token", share), ("shares nothing", nshare)):
    n, rate, mean = decoy_stats(rs)
    print(f"{label:18} n={n:>2}  decoy fires {rate:.0%} of rows  mean {mean:.2f} layers")

check("rows where decoy shares a token", len(share), 6)
check("shares-nothing decoy mean layers", round(decoy_stats(nshare)[2], 2), 0.11)
check("shares-a-token decoy mean layers", round(decoy_stats(share)[2], 2), 8.17)

# the rejected hypothesis: token count vs hits, over all 120 (name, hits) observations
ntok = {}
for r in rows:
    ntok[r["target"]] = len(r["target_tok"]); ntok[r["decoy"]] = len(r["decoy_tok"])
xs, ys = [], []
for r in rows:
    xs += [ntok[r["target"]], ntok[r["decoy"]]]
    ys += [r["j_target"], r["j_decoy"]]
mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
cov = sum((a-mx)*(b-my) for a, b in zip(xs, ys))
den = (sum((a-mx)**2 for a in xs) ** .5) * (sum((b-my)**2 for b in ys) ** .5)
print(f"\nPearson r, token count vs hits, n={len(xs)}: {cov/den:.3f}")
check("length hypothesis r (rejected, ~0.18)", round(cov/den, 3), 0.183)
"""),

    md(r"""
## 5. Transport against the plain logit lens

Per readout rather than per claim. This one survived every correction.
"""),

    code(r"""
for cut in ("prefix_at_sentence", "prefix_at_path"):
    rs = [r for r in rows if r["cut"] == cut]
    j = sum(r["j_target"] for r in rs); l = sum(r["l_target"] for r in rs)
    worse = sum(1 for r in rs if r["j_target"] < r["l_target"])
    print(f"{cut:20} j={j:>4}  logit={l:>4}  ratio={j/max(l,1):.2f}x  "
          f"rows where transport is WORSE: {worse}")

worse_total = sum(1 for r in rows if r["j_target"] < r["l_target"])
check("readouts where transport is worse", worse_total, 0)
check("at_sentence layer-hit ratio",
      round(sum(r["j_target"] for r in rows if r["cut"]=="prefix_at_sentence") /
            max(sum(r["l_target"] for r in rows if r["cut"]=="prefix_at_sentence"), 1), 1), 4.2)
"""),

    md(r"""
## 6. Emergence by layer

Where the target enters the top 20, as a fraction of claims, distinctive tokens only.
"""),

    code(r"""
def curve(cut, only=None):
    per = collections.defaultdict(lambda: collections.defaultdict(lambda: {"t":0,"d":0,"l":0}))
    layers = []
    for r in rows:
        if r["cut"] != cut: continue
        if only is not None and r["case"] not in only: continue
        layers = r["layer_ids"]
        tgt, dec = set(r["target_tok"]), set(r["decoy_tok"])
        sh = tgt & dec; tgt, dec = tgt-sh, dec-sh
        for L in layers:
            j = set(r["j_top20"][str(L)]); lg = set(r["l_top20"][str(L)])
            cell = per[L][claim_key(r)]
            cell["t"] |= 1 if j & tgt else 0
            cell["d"] |= 1 if j & dec else 0
            cell["l"] |= 1 if lg & tgt else 0
    out = {k: [] for k in "tdl"}
    for L in layers:
        cs = per[L]; n = max(len(cs), 1)
        for k in "tdl":
            out[k].append(round(sum(v[k] for v in cs.values())/n, 3))
    return layers, out

layers, c = curve("prefix_at_path")
first = next((L for L, v in zip(layers, c["t"]) if v > 0), None)
print("at_path, target by layer:", c["t"])
print("at_path, logit  by layer:", c["l"])
check("first layer where target appears (at_path)", first, 20)
check("logit lens peak is at the final layer", c["l"].index(max(c["l"])), 38)

try:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, cut in zip(axes, ("prefix_at_sentence", "prefix_at_path")):
        L, cc = curve(cut)
        ax.plot(L, cc["t"], lw=2.4, label="j-lens target")
        ax.plot(L, cc["l"], lw=1.6, label="logit lens target")
        ax.plot(L, cc["d"], lw=1.4, ls="--", label="j-lens decoy")
        ax.set_title(cut); ax.set_xlabel("layer"); ax.grid(alpha=.25)
    axes[0].set_ylabel("fraction of claims"); axes[0].legend()
    plt.tight_layout(); plt.show()
except ImportError:
    print("(matplotlib not available — numbers above are the figure)")
"""),

    md(r"""
## 7. The "just ask it" baseline

Optional: only runs if `ask_baseline.json` was uploaded. It asks the model, at the same cut,
which file it is about to name — the cheap comparison the lens has to beat.
"""),

    code(r"""
try:
    ab = load("ask_baseline.json")
except SystemExit:
    print("ask_baseline.json not uploaded — skipping")
    ab = None

if ab:
    scored = [r for r in ab if not r.get("excluded")]
    print(f"{len(ab)} cases, {len(ab)-len(scored)} excluded (model returned no answer)")
    for label, rs in (("all cases", scored),
                      ("genuinely earlier cut", [r for r in scored if r["earlier_cut"]])):
        cl = {}
        for r in rs:
            d = cl.setdefault((r["run_id"].split("-eval")[0], r["target"]), {"t":0,"d":0})
            d["t"] |= 1 if r["target_named"] else 0
            d["d"] |= 1 if r["decoy_named"] else 0
        if not cl: continue
        n = len(cl)
        t = sum(v["t"] for v in cl.values()); d_ = sum(v["d"] for v in cl.values())
        b = sum(1 for v in cl.values() if v["t"] and not v["d"])
        c = sum(1 for v in cl.values() if v["d"] and not v["t"])
        print(f"  {label:24} target {t}/{n}  decoy {d_}/{n}  p={mcnemar_exact(b,c):.4f}")
    print("\nlens on the same subset: target 4/12, decoy 2/12, p = 0.6875")
"""),


    md(r"""
## 9. The headline result: asking beats the lens

The three cells above verify the FIRST export (30 cases, 17 claims), whose cut was taken from the
visible message. Everything below is the SECOND export (75 cases, 34 claims), cut from
`reasoning_content` — the field the miner actually found the claim in.

Two more files are needed for this section:

| file | what it is |
|---|---|
| `reloc_cases_75.json` | the corrected export: 75 mentions, 34 claims, 15 repositories |
| `ask_baseline_75.json` | the model asked outright, at the same cut, on those cases |
| `reloc_rows_75.jsonl` | the lens run's raw readout — top-20 ids at all 39 layers, one row per readout |
| *(optional)* `reloc6_recovered_counts.json` | an earlier, lost run's counts, kept for provenance |

**Two 75-case lens runs, and only the second is usable.** The first reached case 72 and the
compute provider pruned the session; because the probe wrote its readout only after the loop,
nothing survived but per-case counts scraped from the CLI history. The probe was then changed to
append each readout as it is produced, and the second run was pruned too — at row 145 of 150 — but
its rows were already on disk. `reloc_rows_75.jsonl` is that file: 145 rows, 73 of 75 cases, full
per-layer ids. Cases 73 and 74 are missing, not excluded.

**The objection this section exists to test.** The lens loses to the baseline. The obvious
rejoinder is that it loses only because it was scored on the any-token rule, which is known to
inflate its decoy — two unrelated filenames sharing `.py` score for target and decoy at once. Both
rules are therefore computed below. They give the same answer at the cut that matters.
"""),

    code(r"""
try:
    cases75 = load("reloc_cases_75.json")
    ask     = load("ask_baseline_75.json")
    lens_rows = [json.loads(l) for l
                 in fetch("reloc_rows_75.jsonl").read_text(encoding="utf-8").splitlines()
                 if l.strip()]
except (SystemExit, FileNotFoundError) as e:
    print(e); cases75 = ask = lens_rows = None

if cases75:
    ck = lambda r: (r["run_id"].split("-eval")[0], r["target"])
    # computed out of the f-string: nested same-type quotes inside one are a syntax error
    # before Python 3.12, and this notebook has to run wherever it is opened.
    claims75 = {(c["run_id"].split("-eval")[0], c["target_basename"]) for c in cases75}
    repos75 = {c["repo"] for c in cases75}
    print(f"cases {len(cases75)} mentions, {len(claims75)} claims, {len(repos75)} repos")
    check("second export: mentions", len(cases75), 75)
    check("second export: every claim came from reasoning_content",
          all(c.get("source_field") == "reasoning_content" for c in cases75), True)
    check("holdout absent (BerriAI/litellm)",
          any("litellm" in c["repo"].lower() for c in cases75), False)

    def per_claim_bool(rs, tkey, dkey):
        cl = {}
        for r in rs:
            d = cl.setdefault(ck(r), {"t": 0, "d": 0})
            d["t"] |= 1 if r[tkey] else 0
            d["d"] |= 1 if r[dkey] else 0
        return cl

    # --- baseline: asking the model, genuinely earlier cut only
    scored = [r for r in ask if not r.get("excluded")]
    ab = per_claim_bool([r for r in scored if r["earlier_cut"]], "target_named", "decoy_named")
    n = len(ab); t = sum(v["t"] for v in ab.values()); d = sum(v["d"] for v in ab.values())
    b = sum(1 for v in ab.values() if v["t"] and not v["d"])
    c = sum(1 for v in ab.values() if v["d"] and not v["t"])
    print()
    print(f"ASK   earlier cut: target {t}/{n}  decoy {d}/{n}  p={mcnemar_exact(b,c):.4f}")
    check("ask baseline target (earlier cut)", f"{t}/{n}", "18/30")
    check("ask baseline decoy  (earlier cut)", f"{d}/{n}", "3/30")
    check("ask baseline p      (earlier cut)", round(mcnemar_exact(b,c), 4), 0.0003)

    # --- lens: scored from the RAW per-layer ids, under BOTH rules.
    # This is the check the section exists for. The lens loses to the baseline; the obvious
    # rejoinder is that it loses only to the permissive any-token rule, which inflates its decoy.
    # Both rules are computed so that rejoinder can be settled rather than argued.
    earlier75 = {i for i, c in enumerate(cases75)
                 if c["prefix_at_sentence"] != c["prefix_at_path"]}

    def lens_score(cut, distinctive, only=None):
        cl = {}
        for r in lens_rows:
            if r["cut"] != cut: continue
            if only is not None and r["case"] not in only: continue
            t_, d_ = set(r["target_tok"]), set(r["decoy_tok"])
            if distinctive:
                sh = t_ & d_
                t_, d_ = t_ - sh, d_ - sh
            v = cl.setdefault(ck(r), {"t": 0, "d": 0})
            for L in r["layer_ids"]:
                j = set(r["j_top20"][str(L)])
                v["t"] |= 1 if j & t_ else 0
                v["d"] |= 1 if j & d_ else 0
        n_ = len(cl)
        T = sum(v["t"] for v in cl.values()); D = sum(v["d"] for v in cl.values())
        bb = sum(1 for v in cl.values() if v["t"] and not v["d"])
        cc = sum(1 for v in cl.values() if v["d"] and not v["t"])
        return n_, T, D, mcnemar_exact(bb, cc)

    print()
    for rule in (False, True):
        n2, t2, d2, p2 = lens_score("prefix_at_sentence", rule, earlier75)
        nm = "distinctive" if rule else "any-token"
        print(f"LENS  earlier cut, {nm:11}: target {t2}/{n2}  decoy {d2}/{n2}  p={p2:.4f}")
        check(f"lens target (earlier, {nm})", f"{t2}/{n2}", "11/31")
        check(f"lens decoy  (earlier, {nm})", f"{d2}/{n2}", "5/31")
        check(f"lens p      (earlier, {nm})", round(p2, 3), 0.146)

    # the shared-token contamination is real -- but only at the LATER cut
    _, _, d_any, _ = lens_score("prefix_at_path", False, earlier75)
    _, _, d_dst, _ = lens_score("prefix_at_path", True, earlier75)
    print(f"at_path decoy: any-token {d_any}/31 -> distinctive {d_dst}/31")
    check("at_path decoy, any-token", f"{d_any}/31", "21/31")
    check("at_path decoy, distinctive", f"{d_dst}/31", "4/31")

    covered = sorted({r["case"] for r in lens_rows})
    print()
    print(f"lens rows cover {len(covered)} of {len(cases75)} cases "
          f"(missing: {[i for i in range(len(cases75)) if i not in set(covered)]})")
"""),


    md(r"""
## 10. Why some stated conclusions never reach the output

The report's third follow-up asked whether the never-submitted set is a *suppression* story or an
*error* story. It is neither. The enricher prompt used to carry a consolation clause — "then cite
the closest line in the file you were asked about" — and the model complied with it. The clause
was removed on 2026-08-31.

`relocation_reach.json` is every mined relocation with the run it came from, whether it reached
the casefile, and which side of that date it falls on.
"""),

    code(r"""
try:
    reach = load("relocation_reach.json")
except SystemExit as e:
    print(e); reach = None

if reach:
    from math import comb
    def fisher(a, b, c, d):
        n = a + b + c + d
        f = lambda a_, b_, c_, d_: (comb(a_+b_, a_) * comb(c_+d_, c_)) / comb(n, a_+c_)
        obs = f(a, b, c, d); tot = 0.0
        for x in range(0, min(a+b, a+c) + 1):
            y, z = a+b-x, a+c-x; w = c+d-z
            if min(x, y, z, w) < 0: continue
            q = f(x, y, z, w)
            if q <= obs + 1e-12: tot += q
        return min(tot, 1.0)

    era = {"before": [0, 0], "after": [0, 0]}   # [never reached, reached]
    for r in reach:
        era[r["era"]][1 if r["reached_casefile"] else 0] += 1
    a, b = era["before"]; c, d = era["after"]

    print(f"{'era':24} {'relocations':>12} {'never reached':>14} {'rate':>7}")
    for lbl, (nv, rc) in (("before the prompt fix", era["before"]), ("on or after", era["after"])):
        print(f"{lbl:24} {nv+rc:>12} {nv:>14} {nv/max(nv+rc,1):>6.1%}")
    p_ = fisher(a, b, c, d)
    print(f"odds ratio {a*d/max(b*c,1):.1f}x   Fisher exact two-sided p = {p_:.5f}")

    check("relocations mined", len(reach), 218)
    check("never reached, before fix", f"{a}/{a+b}", "5/29")
    check("never reached, after fix",  f"{c}/{c+d}", "1/189")
    check("Fisher exact p", round(p_, 5), 0.00016)

    # the trace that records the mechanism in the model's own words
    hit = [r for r in reach if not r["reached_casefile"] and "I should cite that file" in r["quote"]]
    check("a trace shows the model redirected by the instruction", len(hit) >= 1, True)
    if hit:
        print()
        print(hit[0]["run_id"], "->", hit[0]["path"])
        print(" ", hit[0]["quote"][-200:])
"""),

    md(r"""
## 11. Verdict
"""),

    code(r"""
bad = [c for c in CHECKS if not c[3]]
print(f"{len(CHECKS) - len(bad)} of {len(CHECKS)} checks reproduce the claimed value.\n")
if bad:
    print("MISMATCHES — the claimed number is not what this notebook computes:")
    for label, got, claimed, _ in bad:
        print(f"  {label:52} computed={got}  claimed={claimed}")
else:
    print("Every claimed number was recomputed from the raw files and matches.")
    print()
    print("The conclusion those numbers support:")
    print("  - ASKING the model which file it is about to name recovers it in 18 of 30")
    print("    claims against a decoy floor of 3 (p = 0.0003). The information is there.")
    print("  - the LENS, same question and same claims, gets 11 of 31 against a floor of")
    print("    5 (p = 0.15). It does not separate from its decoy. The lens does not read it,")
    print("    and this holds under BOTH scoring rules -- so it is not an artefact of the")
    print("    permissive any-token rule. That rule does inflate the decoy, but only at the")
    print("    LATER cut, where correcting it drops the decoy from 21/31 to 4/31.")
    print("  - the comparison favours the baseline by construction: it lets the model")
    print("    generate before answering, while the lens reads one activation.")
    print("  - what survives about the transport: never worse than the logit lens on any")
    print("    of the 60 readouts of the first export, and it recovers the file from layer")
    print("    20 at the LATER cut, where the model has already begun the sentence.")
    print("  - the six relocations that never reach the output are not suppression and not")
    print("    error: 5 of 29 under a prompt carrying a consolation clause, 1 of 189 after it")
    print("    was removed (p = 0.00016). The model followed an instruction we wrote.")
"""),
]

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}, "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 0}

OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {OUT}  ({OUT.stat().st_size:,} bytes, {len(CELLS)} cells)")
