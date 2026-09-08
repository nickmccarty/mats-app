# MATS application artifacts — confidential working draft

Assembled 2026-09-08. **Not for redistribution.** The findings are provisional: the pilot rests
on ~30 distinct claims and its lens arm is not statistically significant. Two scoring rules have
already been corrected. The lens run covers all 75 cases: the main run was pruned by the provider
at row 145 of 150 and the remaining two cases were scored by a resume run and merged.

**Every number is independently checkable.** `verify/verify_findings.ipynb` runs in Colab with no
GPU and no model, re-derives each figure from the raw files beside it, and prints computed against
claimed with PASS/FAIL. It currently reports 35 of 35.

The lens is shown under **both** scoring rules because the obvious objection is that it loses only
to the permissive any-token rule, which inflates its decoy. It does not: at this cut both rules
give 11 of 31 against a floor of 5. That rule does inflate the decoy — but only at the later cut,
where correcting it drops the decoy from 21 of 31 to 4.

Everything below was produced by `ctarp`, a local multi-model vulnerability-localisation harness.
The application material is the faithfulness draft plus the Jacobian-lens pilot it now contains.

---

## Start here

| | |
|---|---|
| `report/ctarp-faithfulness.pdf` | **The draft.** 9pp. Confidential banner, no open-access stamp. |
| `site/index.html` | The web version. Open from `file://`; §8 "Asking the activations" is the pilot. |
| `site/deck.html` | 10-slide deck. Slides 9–10 are the pilot and what its control caught. |
| `notebooks/jlens_reloc_replication.ipynb` | Reproduces the pilot end to end on a Colab A100. |

## What the pilot found

**The cheap baseline beats the lens.** At the cut before the model writes the relocated file —
measured only where that cut is genuinely earlier than the naming sentence:

| | claims | target | decoy | p |
|---|---|---|---|---|
| **ask the model outright** | 30 | **18** | 3 | **0.0003** |
| Jacobian lens, any-token scoring | 31 | 11 | 5 | 0.146 |
| Jacobian lens, distinctive tokens | 31 | 11 | 5 | 0.146 |

The information is there and the lens does not read it. That is a statement about the instrument,
not the model — and it makes the faithfulness question well-posed rather than answering it.

The comparison is deliberately unfair to the lens: it reads one activation, while the baseline
lets the model generate up to 2,400 tokens first. A baseline that strong losing would have settled
the matter; winning only bounds what the lens failed to find.

**What survives about the transport:** against the plain logit lens it is never worse on any of
the 60 readouts of the first export, and at the *later* cut it recovers the file from layer 20 of
39 while the logit lens arrives only at layer 38. That is a property of the transport, not
evidence about the earlier cut.

**Two corrections are reproduced rather than hidden.** Earlier figures counted claims whose
"before the sentence" prefix was byte-identical to the mid-sentence one, and scored a hit when any
token of a filename entered the top 20 — so two unrelated files sharing `.py` scored for target
and decoy at once. `verify/verify_findings.ipynb` recomputes both the old and corrected values.

## Layout

```
report/       the draft, its MyST source, and the four figures
notebooks/    the Colab replication notebook and the script that generates it
code/         the probe, the analyses, and the Colab session helpers
data/         inputs and the raw readouts downloaded from Colab
site/         the web version, openable from file://
```

### `report/`

- `ctarp-faithfulness.pdf` — the draft as circulated
- `nanda.md` — MyST source (built with `myst build --pdf`; needs the typst CLI)
- `figures/emergence.png` — where the file becomes decodable, by layer
- `figures/transport.png` — Jacobian transport against plain logit lens, per readout
- `figures/control.png` — what the decoy caught
- `figures/shared.png` — the decoy's noise floor, split by shared tokens

### `code/`

- `export_reloc_cases.py` — turns 35B trajectories into self-contained probe cases
- `jlens_reloc_probe.py` — the probe. Records **top-20 token ids at every layer**, not a hit count
- `jlens_layer_curve.py` — emergence curves and symmetric distinctive-token scoring
- `jlens_reloc_stats.py` — per-claim collapse and paired McNemar test
- `colab/` — session helpers. `colab exec` is a blocking HTTP call on a shared kernel, so the
  download and the probe both run detached with a tiny status script polling them

### `data/`

- `reloc_cases.json` — **the corrected export**: 75 mentions, 34 claims, 15 repositories, cut from
  `reasoning_content` (the field the miner finds claims in)
- `reloc_cases_v1_30.json` — the superseded export: 30 mentions, 17 claims, cut from `message`.
  Every number published before 2026-09-07 was computed on this, and its cut sits in the wrong text
- `reloc_results.json` — **the raw readout** for the v1 export, 735 KB: top-20 ids at all 39 layers
  for both lenses. Every scoring question on v1 is a re-analysis of this file
- `reloc6_recovered_counts.json` — per-case counts for the 75-case run, scraped from the CLI's
  execution history after the provider pruned the session at case 72 of 75. Counts only; the raw
  ids were never written, so this cannot be re-scored on distinctive tokens
- `ask_baseline_75.json` / `ask_baseline_30.json` — the model asked outright, both case sets
- `layer_curve.json` — the emergence curves, derived from the v1 raw readout
- `trajectories/` — the full ATIF trace, tool trace and casefile for all 29 runs behind the cases
- `logs/` — the per-run probe logs and the baseline log

## Method notes worth knowing before reading the numbers

**Read target against decoy, never against zero.** Every readout is scored a second time against
another case's target — same filename distribution, wrong answer. The criterion ("any token in a
top-20, at any layer") is permissive by design and the decoy measures exactly how permissive.

**Per claim, not per mention.** The 30 mentions are 17 distinct (repo, target) claims; one is
restated 7 times. Counting mentions inflates in the flattering direction, which is the error the
main report exists to retract.

**Distinctive tokens, symmetrically.** Two unrelated filenames sharing `.py` scored a hit for the
target and the decoy at once. Shared tokens are now dropped from both sides. Dropping them from
the decoy alone moves p from 0.109 to 0.008 and is not licensed — it corrects only the side that
hurts the hypothesis.

**Only the genuinely earlier cut counts.** `prefix_at_sentence` falls back to the whole step when a
claim has no earlier sentence boundary, making it byte-identical to `prefix_at_path` for a minority
of claims. Measuring "before the sentence" on those is measuring after it. On the v1 export,
excluding them moved the lens from 9/17 (p = 0.109) to 4/12 (p = 0.6875).

**BerriAI/litellm is a holdout** in the wider corpus and contributes nothing here; the pilot's 15
repositories do not include it.

## Reproducing

```
colab new -s reloc --gpu A100
colab install -s reloc "git+https://github.com/anthropics/jacobian-lens.git" accelerate
colab upload -s reloc data/reloc_cases.json /content/reloc_cases.json
# then run notebooks/jlens_reloc_replication.ipynb, or the detached CLI path in code/colab/
```

The model is `Qwen/Qwen3.6-35B-A3B` (~67 GB bf16, offloads to system RAM on a 40 GB A100); the lens
is `camilablank/workspace-lenses`. Two independent runs produced identical per-case counts.
