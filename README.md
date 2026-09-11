# Chain-of-thought faithfulness, with an answer key

**Working draft — not for redistribution.**

Agentic harnesses for vulnerability localization emit long reasoning traces and then reduce them to
a ranked list of files, discarding the reasoning. Whether that discarded text carries information
the ranked list does not is a chain-of-thought faithfulness question, and it is normally
unanswerable: the reasoning and the output are both model artifacts, with nothing independent to
check them against. Here there is a third object. Every task sits at the commit before a real fix,
so the fix commit names the true files whatever the model said.

## Start here

| | |
|---|---|
| **[The write-up](https://nickmccarty.me/mats-app/)** | The whole thing as a web page. Generated from `report/nanda.md`, same source as the PDF. |
| **[The PDF](report/ctarp-faithfulness.pdf)** | The same text, typeset, with the figures placed. |
| **[Verify every number](https://colab.research.google.com/github/nickmccarty/mats-app/blob/main/verify/verify_findings.ipynb)** | Colab. Re-derives each published figure from the raw files and prints computed against claimed. No GPU, ~2s, **40 of 40 pass**. |
| **[The follow-up probes](https://colab.research.google.com/github/nickmccarty/mats-app/blob/main/verify/probe_q1_q2.ipynb)** | Colab. Four activation-level questions, each against a measured permutation null. No GPU, ~15 min. |
| **[Reproduce the lens pilot](https://colab.research.google.com/github/nickmccarty/mats-app/blob/main/notebooks/jlens_reloc_replication.ipynb)** | Colab. Weights, extraction, scoring. Needs an A100 and ~2 hours. |
| **[The trajectories](site/traces/trajectory.html)** | All 32 runs behind the claims, searchable. Opens from `file://`. |

The notebooks fetch their own inputs from this repository, so there is nothing to upload. Files
placed beside a notebook override the fetched ones.

## What it found

**The headline did not survive de-duplication.** Counted per mention, mined relocations name a true
gold file 81.4% of the time, which looks like it beats the 49.8% calibration they are measured
against. Counted per distinct claim — the unit that calibration uses — precision is **40.5% on 37
claims**, below it. Three tasks are half the population; one claim is restated 55 times.

**A cheap baseline beats the instrument.** At the token before the model writes the relocated file,
measured only where that cut is genuinely earlier than the naming sentence:

| | claims | target | decoy | p |
|---|---|---|---|---|
| **ask the model outright** | 30 | **18** | 3 | **0.0003** |
| Jacobian lens, any-token scoring | 31 | 11 | 5 | 0.146 |
| Jacobian lens, distinctive tokens | 31 | 11 | 5 | 0.146 |

![Asking the model against the Jacobian lens, each with its decoy floor](report/figures/baseline.png)

*Both methods with their decoy floors drawn across the bars rather than beside them: a hit rate is
unreadable without the floor it has to clear.*

The information is there and the lens does not read it, which is a result about the instrument
rather than about the model. The comparison is unfair to the lens — it reads one activation while
the baseline generates up to 2,400 tokens first — so it bounds what the lens missed rather than
showing the answer sits in the residual stream.

**Every activation-level follow-up returned a null or an artifact.** A trained readout recovers the
file in 20 of 24 claims at its best layer, but it is identifying the *repository*; deconfounded it
scores two. Right-versus-wrong relocation is null at a detection threshold of AUC 0.676. Whether a
relocation is *coming* is suggestive at late layers and survives no correction for multiple
comparisons.

![Both probes against the permutation nulls they have to clear](report/figures/nullband.png)

*Each panel draws its own permutation null as a band, mean to 95th percentile, for the identical
procedure; a filled dot is a layer above its null. Reading either curve against 0.5 instead of
against its band gives the wrong answer in both panels.*

![The best single dimension: real labels, shuffled labels, and held-out claims](report/figures/noise.png)

*Rank every unit by ROC-AUC, take the best, check whether it generalizes — run here with the step
usually skipped, the same procedure on shuffled labels. Mean in-sample AUC is 0.847 with real
labels and 0.852 with random ones. Held-out drops to 0.349.*

**Five measurements looked publishable and were wrong**, each in the direction the hypothesis
wanted: a repository detector reading as a filename probe; a delimiter detector separating classes
at the embedding layer, before any transformer block had run; a prompt-length shortcut; a
per-mention count inflating a per-claim p-value; a permutation null built at the wrong unit. The
mistaken value is reported beside the corrected one throughout, and the verification notebook
recomputes both.

**Why some stated conclusions never reach the output.** Six relocations are stated in reasoning and
never cited. That is not suppression and not error: 5 of 29 under a prompt carrying a consolation
clause (*"then cite the closest line in the file you were asked about"*), 1 of 189 after the clause
was removed. Fisher exact p = 0.00016, odds ratio 39. One trace records the model resolving to cite
the right file and then being redirected mid-sentence.

## What this does not show

- **n is small.** 37 distinct claims in the mining result, ~30–35 in the pilot, 23 matched pairs in
  the event probe. Every null here is weak: the Q2 probe could not have detected anything below
  AUC 0.676, which is a large effect. The nulls bound the effects; they do not exclude them.
- **One model, one corpus, one cut point.** Qwen3.6-35B-A3B on 15 repositories.
- **The decoy controls for filename plausibility, not repository identity.** 71 of 75 decoys name a
  file from a different project, so every target-vs-decoy figure reports a floor that is too low.
  It does not change the lens result, which was null against an over-generous floor.

![Distinct relocated files per repository](report/figures/confound.png)

*Only the four repositories above the dashed line can supply a same-repository decoy at all.*

- **"Does the residual identify the file rather than the project?" is not answerable here.**
  Relocated filenames are nested inside repositories, and a within-repository decoy leaves two
  scoreable claims. A successor corpus has to be built for that question rather than filtered into
  it.
- **The baseline is not an interpretability result.** It lets the model generate before answering.
- **Gold is fix-commit files**, which under-credits tests, callers and configuration that a real fix
  touches but the reference patch does not.
- **`BerriAI/litellm` is a holdout** and contributes nothing here. The probe notebook asserts this
  rather than promising it.

## Layout

```
index.html      the write-up as a web page, generated from report/nanda.md
SUBMISSION.md   the same text as plain Markdown
report/         the PDF, its MyST source, references, and all eight figures
verify/         the two checking notebooks and the exact inputs they read
code/           the probes, the analyses, and the Colab session helpers
data/           inputs and the raw readouts downloaded from Colab
site/           the longer harness write-up and the trajectory viewer
```

`index.html` and `SUBMISSION.md` are generated from `report/nanda.md`. Editing them by hand
reintroduces the drift they exist to prevent.

Worth knowing where to look:

- `code/probe_q1_q2.py` — the follow-up probes: grouped folds, permutation nulls, per layer
- `code/probe_event.py` — the event probe, which reports the length-only AUC before any result
- `code/dim_auc_sweep.py` — the best single dimension, against shuffled labels
- `code/decoy_scope_audit.py` — reproduces the published 30/18/3, then re-states what it controls for
- `code/label_relocations.py` — the only script that reads gold, and it reads it *after* every
  relocation already exists
- `report/figures/make_probe_figs.py` — the D3 source for the three probe figures
- `data/trajectories/` — full ATIF trace, tool trace and casefile for every source run

## Reproducing the GPU work

The Colab links above cover what a reader needs. The extraction behind them, if you want to re-run
it: model `Qwen/Qwen3.6-35B-A3B` (~67 GB bf16, offloads to system RAM on a 40 GB A100), lens
`camilablank/workspace-lenses`, 75 cases × 2 cut points × 2 lenses.

```
python code/export_reloc_cases.py       # trajectories -> self-contained probe cases
python code/export_control_cuts.py      # matched non-relocation controls
colab exec -s <s> -f code/colab/colab_prefetch.py      # detached; 135 GB of weights
colab exec -s <s> -f code/colab/colab_run_extract.py   # detached; ~45 min per condition
python code/label_relocations.py        # gold -> labels. 51/75 mentions, 21/17 claims
python code/probe_q1_q2.py              # both probes, per layer, with nulls
```

Four things cost runs here, each of them silently:

| trap | what it looks like | what to do |
|---|---|---|
| Colab prunes long sessions | the run vanishes; `colab stop` says "not found" | write incrementally and pull as it grows, so a prune costs a tail |
| `colab exec` is a blocking HTTP call | `ConnectionError` from the CLI's own transport | detach anything slow, poll with a tiny status script |
| Git Bash rewrites POSIX paths | `Download failed: C:/Program Files/Git/content/...` | `MSYS_NO_PATHCONV=1`, and never send its stderr to `/dev/null` |
| a crashed run leaves the model resident | the next load OOMs asking for ~96% of `max_memory` | the extractor prints free VRAM and refuses below 34 GiB |

Two independent lens runs over the same cases produced identical per-case counts, so the readout is
deterministic. The prompt baseline is `temperature=0` but is a generation, so exact reproduction is
not guaranteed.

---

Nicholas McCarty · Upskilled Consulting
