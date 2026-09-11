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
| `verify/verify_findings.ipynb` | Recomputes every claimed number from raw files. No GPU. |
| `verify/probe_q1_q2.ipynb` | The two follow-up probes, each against its permutation null. No GPU. |

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

**A third correction, found while building the follow-up probes, and it reaches this table.** The
decoy in every row is another case's real target, drawn from the whole corpus. But relocated
filenames are almost perfectly nested inside repositories: **20 of 21 distinct basenames occur in
exactly one repo, and 71 of 75 decoys name a file from a different project than the one being
read.** A method can prefer the target over that decoy by recognising the repository — which every
method here has free access to, because the entire context is that project's source.

So the *decoy* column is a floor that is too low, not a noise estimate. What survives is the
*target* column: **18 of 30 claims where the model named the exact repository-relative path**, out
of the many files it could have named instead. The lens is unaffected in direction — a floor that
is too low could only have flattered it, and it was null regardless.

`code/decoy_scope_audit.py` reproduces the published 30/18/3 from the stored replies and then
re-states it. The repair for a future corpus is concrete: relocation pairs must be drawn from
**within** a repository, so that repo identity favours both candidates equally.

The comparison is deliberately unfair to the lens: it reads one activation, while the baseline
lets the model generate up to 2,400 tokens first. A baseline that strong losing would have settled
the matter; winning only bounds what the lens failed to find.

**Why some stated conclusions never reach the output.** The six relocations that are stated in
reasoning but never cited are not a suppression story and not an error story: 5 of 29 under a
prompt carrying a consolation clause (*"then cite the closest line in the file you were asked
about"*), 1 of 189 after that clause was removed — Fisher exact p = 0.00016, odds ratio 39. One
trace records the model resolving to cite the right file and then being redirected by the
instruction mid-sentence. The 97% figure is therefore partly a fact about a prompt, not only about
the reduction from transcript to ranked list.

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
verify/       the two checking notebooks and the exact inputs they re-derive from
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
- `extract_residuals.py` — saves the residual stream itself, for the follow-up probes. No lens
- `dump_token_embeddings.py` — the candidate filenames in the same space, sliced from the cached
  safetensors without loading the model
- `label_relocations.py` — the only script in the probe pipeline that reads gold, and it reads it
  *after* every relocation already exists. Nothing it writes re-enters a locate prompt
- `probe_q1_q2.py` — both follow-up probes: grouped folds, permutation nulls, per-layer
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

Five things can be reproduced, in increasing order of cost. Only the second and fourth need a GPU;
the fifth reuses the fourth's extraction.

### 1. Check every number — no GPU, no model, ~2 seconds

```
# upload to Colab (or run locally with python >= 3.10):
#   verify/verify_findings.ipynb
#   verify/reloc_cases.json          verify/reloc_results.json
#   verify/reloc_cases_75.json       verify/reloc_rows_75.jsonl
#   verify/ask_baseline_75.json      verify/gold_pilot.json
# then: Run all
```

Each cell recomputes a figure from the raw files and prints it beside the claimed value with
PASS/FAIL. It currently reports **35 of 35**. Where a number was corrected during the work, both
the original and the corrected value are computed, so a correction is visible as arithmetic rather
than asserted in prose.

This is the honest entry point: it checks the arithmetic without trusting the pipeline that
produced it.

### 2. Re-run the lens — Colab A100, ~15 min weights + ~90 min probe

```
colab new -s reloc --gpu A100
colab install -s reloc "git+https://github.com/anthropics/jacobian-lens.git" accelerate
colab upload -s reloc data/reloc_cases.json      /content/reloc_cases.json
colab upload -s reloc code/jlens_reloc_probe.py  /content/jlens_reloc_probe.py

# weights first, DETACHED -- 135 GB, and a foreground exec outruns the CLI's read timeout
colab exec -s reloc -f code/colab/colab_prefetch.py
colab exec -s reloc -f code/colab/colab_prefetch_status.py     # until PREFETCH DONE

# then the probe, also detached
colab exec -s reloc -f code/colab/colab_run_probe.py
colab exec -s reloc -f code/colab/colab_probe_status.py        # poll

# PULL ROWS AS THEY LAND -- do not wait for the end
MSYS_NO_PATHCONV=1 colab download -s reloc /content/reloc_rows.jsonl reloc_rows.jsonl

colab stop -s reloc
python code/score_reloc8.py
```

Model `Qwen/Qwen3.6-35B-A3B` (~67 GB bf16; offloads to system RAM on a 40 GB A100), lens
`camilablank/workspace-lenses`. 75 cases × 2 cut points × 2 lenses = 150 readouts.

**Five things that cost runs here. Each one failed silently.**

| trap | what it looks like | what to do |
|---|---|---|
| Colab prunes long sessions | run vanishes mid-probe; `colab stop` says "not found" | the probe appends each readout to `/content/reloc_rows.jsonl` and fsyncs, so pull that file every few minutes — a prune then costs only the tail |
| `colab exec` is a blocking HTTP call on one shared kernel | `ConnectionError` from the CLI's own transport, which reads like a network fault | detach anything slow (`colab_prefetch.py`, `colab_run_probe.py`) and poll with a tiny status script |
| Git Bash rewrites POSIX paths | `Download failed: C:/Program Files/Git/content/...` | `MSYS_NO_PATHCONV=1` on every `colab download`, and never redirect its stderr to `/dev/null` |
| a crashed run leaves the model on the GPU | next load OOMs asking for ~96% of `max_memory` | `free_gpu()` in the probe prints free VRAM and refuses below 34 GiB. Do **not** run `code/colab/colab_restart_kernel.py` — it is disarmed because `os._exit` wedges the CLI session |
| `jlens.apply` defaults to `max_seq_len=512`, right-truncating | 0 hits for target *and* decoy — reads like a clean null | the probe sets 4096 and `tok.truncation_side = "left"`, and prints the decoded prompt tail for the first cases |

To resume a partial run, set `ONLY = "73,74"` in `code/colab/colab_run_probe.py`. The probe still
iterates the whole case list and skips, rather than slicing it — slicing would reseed the decoy
shuffle and re-pair every target, so the new rows would not merge with the old.

### 3. Re-run the prompt baseline — local, no Colab

Needs the 35B served on `:8083` (any OpenAI-compatible endpoint).

```
python code/ask_baseline.py --n 75 --out ask_baseline_75.json
```

It replays each case to the same cut, appends *"which file are you about to name? Reply with
exactly one repository-relative path"*, and scores the reply against the target and the same
decoy the lens was scored against. `max_tokens` is 2400 because this model reasons before
answering and a smaller budget returns an empty string — which scores as a miss and would
manufacture a negative. Empty replies are excluded and counted, not scored as misses.

### 4. Re-run the follow-up probes — Colab A100, ~15 min weights + ~100 min extraction

Two questions the lens run could not answer, because it saved the readout's *verdict* (top-20 token
ids) and not the readout's *input*:

- **Q1** — is the file linearly decodable at the cut by a readout **trained on this task**, rather
  than by the published lens? A negative lens result is consistent with "the information is absent"
  and with "that lens does not read it", and those are different conclusions.
- **Q2** — does the residual at that moment distinguish a relocation that turns out to be **right**
  from one that turns out to be **wrong**? If so, there is something to build a gate on.

No lens is involved: `output_hidden_states=True` returns the residual stream directly.

```
colab new -s resid --gpu A100
colab install -s resid accelerate safetensors
colab upload -s resid data/reloc_cases.json        /content/reloc_cases.json
colab upload -s resid code/extract_residuals.py    /content/extract_residuals.py

colab exec -s resid -f code/colab/colab_prefetch.py            # detached, 135 GB
colab exec -s resid -f code/colab/colab_prefetch_status.py     # until PREFETCH DONE

colab exec -s resid -f code/colab/colab_run_extract.py         # detached
colab exec -s resid -f code/colab/colab_extract_status.py      # poll

# pull as it grows -- the npz is rewritten every 4 readouts for exactly this reason
MSYS_NO_PATHCONV=1 colab download -s resid /content/resid.npz       resid.npz
MSYS_NO_PATHCONV=1 colab download -s resid /content/resid_meta.json resid_meta.json

# the candidate filenames in the same space, sliced from the cached safetensors -- no model load
colab upload -s resid code/dump_token_embeddings.py /content/dump_token_embeddings.py
colab exec   -s resid -f code/dump_token_embeddings.py
MSYS_NO_PATHCONV=1 colab download -s resid /content/tok_emb.npz       tok_emb.npz
MSYS_NO_PATHCONV=1 colab download -s resid /content/tok_emb_names.json tok_emb_names.json
colab stop -s resid

python code/label_relocations.py     # gold -> reloc_labels.json, 51/75 mentions, 21/17 claims
python code/probe_q1_q2.py           # both probes, per layer, with permutation nulls
```

`colab.exe` and every analysis script here need **numpy, sklearn, matplotlib**; run them with the
interpreter that has them, not a bare `python`.

**Why most of the probe code is null-estimation.** There are 38 distinct claims and 4,096
dimensions per layer. A linear classifier in that regime separates *random* labels nearly
perfectly, so an AUC on its own means nothing. Three precautions, each demonstrated in the notebook
rather than promised:

| precaution | what it prevents |
|---|---|
| folds are **leave-one-claim-out**, never leave-one-row-out | one claim appears as several mentions and at two cut points; a random split puts near-duplicates of the test row in training. The notebook prints the naive number beside the honest one — the gap *is* the artefact |
| a **permutation null**, 500 shuffles per layer | the reported p is the fraction of label shuffles that beat the real labels, so the null is measured on this data rather than assumed to be 0.5 |
| PCA fitted **inside** each fold | fitting once on everything leaks the test fold into the projection |

One more choice worth knowing about: Q2's labels come from matching the relocated file against gold.
Matching on the exact repo-relative path gives **21 correct / 17 wrong** across 38 claims; also
accepting a bare-filename match gives **27 / 11**. The looser rule hands the classifier a class
balance manufactured by the scoring rule, so the strict labels are the ones used.
`python code/label_relocations.py --allow-basename` regenerates the loose set, and
`code/probe_q1_q2.py --labels <file>` re-runs against it.

### 5. The event probe and the single-dimension sweep

Two further questions the same residuals support.

**Is the *event* decodable — was the model about to change its mind at all?** This is the question
the backtracking literature asks of reasoning traces, and it needs negatives. `export_control_cuts.py`
emits one matched control per relocation: a sentence boundary in the **same run**, in a step
containing **no** relocation, chosen to match the positive's prefix length. Matching is what makes
it a control — relocations arrive late in long traces, so unmatched negatives would let a
classifier score well by detecting context length alone.

```
python code/export_control_cuts.py            # 74 of 75 matched, median length gap 14 chars
colab upload -s resid data/control_cases.json /content/control_cases.json
colab exec   -s resid -f code/colab/colab_run_controls.py      # refuses if the first pass is live
colab exec   -s resid -f code/colab/colab_controls_status.py
MSYS_NO_PATHCONV=1 colab download -s resid /content/resid_ctrl.npz resid_ctrl.npz
python code/probe_event.py
```

`probe_event.py` reports the AUC of a classifier given **only the token count** before it reports
anything about activations. That is the floor the real probe has to clear, and folds are
leave-one-**run**-out because a positive and its matched negative share a run.

**What is the best single dimension worth?** `dim_auc_sweep.py` runs the standard recipe — score
every unit by ROC-AUC, take the best, note its Cohen's *d*, check whether it generalises — and adds
the step that is usually skipped: the same procedure on **shuffled labels**.

```
python code/dim_auc_sweep.py
```

With 2,048 dimensions and 35 claims, the best-looking dimension is the best of 2,048 draws from
noise. Reading its in-sample AUC against 0.5 rather than against the shuffled baseline is how a
selection artifact becomes a finding about polysemanticity.

Everything above is also packaged as `verify/probe_q1_q2.ipynb` — upload it with `resid.npz`,
`resid_meta.json`, `reloc_labels.json`, `tok_emb.npz`, `tok_emb_names.json` and (for §9)
`resid_ctrl.npz` and `resid_ctrl_meta.json`, then Run all. No GPU; the forward passes have already
happened.

### What "reproduce" means here

Two independent lens runs over the same cases produced identical per-case counts, so the readout
is deterministic. The prompt baseline is `temperature=0` but is a generation, so exact
reproduction is not guaranteed; the claim rests on 30 claims at p = 0.0003, not on any single
answer.
