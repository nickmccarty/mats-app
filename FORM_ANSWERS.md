# Application form — drafts

Drafted from the work. **Four of these I cannot answer for you** and they are marked
**YOU MUST WRITE THIS**. One of them — the LLM-usage question — would be self-defeating to hand to
an LLM, so what is below it is raw material rather than an answer: specific things that actually
happened, for you to select from and write in your own words.

---

## What question did you try to answer?

Agentic harnesses for vulnerability localization produce long reasoning traces and then reduce them
to a ranked list of files, discarding the reasoning. I asked whether that discarded text carries
information the ranked list does not — a chain-of-thought faithfulness question — and then whether
the answer is recoverable from activations before the model writes it.

Concretely, three questions:

1. Do conclusions stated in reasoning fail to reach the structured output?
2. At the token before the model names a file it was not asked about, is that file decodable from
   the residual stream?
3. Do activations distinguish a relocation that turns out to be correct from one that is wrong?

---

## Why is this question interesting / why did you choose it?

Faithfulness tests usually cannot check a model's stated target against an objective answer,
because the reasoning and the output are both model artifacts. The standard workaround is to
perturb the reasoning and see whether the answer moves, which measures self-consistency more than
faithfulness.

Vulnerability localization supplies a third object. Every task sits at the commit before a real
fix, and the fix commit — written by the project's maintainers, not by an annotator reading model
output — names the true files. So a claim the model makes mid-reasoning can be scored against
something neither the transcript nor the output produced.

I also had the corpus already: 504 recorded trajectories from a harness I built, with full ATIF
traces, so the reasoning text and the structured output were both recoverable per run.

---

## What conclusions have you reached?

**Claims I believe are supported:**

1. **Mining assertions from transcripts inflates when counted per mention.** Relocations name a
   true gold file 81.4% per mention and 40.5% per distinct claim (n = 37). The per-claim figure is
   *below* the 49.8% three-pass-agreement baseline it was reported as beating. Three tasks are half
   the population; one claim is restated 55 times.
2. **The reduction from transcript to ranked list is not where stated conclusions are lost.**
   212 of 218 relocations reach the structured output.
3. **The six exceptions are instruction-following, not suppression or error.** 5 of 29 under a
   prompt carrying a consolation clause, 1 of 189 after it was removed. Fisher exact p = 0.00016,
   odds ratio 39. One trace shows the model resolving to cite the correct file and being redirected
   mid-sentence.
4. **The published Jacobian lens does not recover the relocated file at the cut before it is
   named**, while asking the model outright does: 11 of 31 versus 18 of 30 claims.

**Claims I disproved or retracted, including my own:**

5. **My own headline (81.4%) was wrong** — wrong unit of analysis.
6. **A trained readout appearing to recover the file (20 of 24 claims at its best layer,
   p = 0.0015) is a repository detector.** 20 of 21 relocated filenames occur in exactly one repo,
   and 71 of 75 decoys name a file from a different project. Deconfounded it scores two claims.
7. **Activations do not distinguish correct from incorrect relocations** at this sample size —
   null against a measured permutation null, detection threshold AUC 0.676 at 35 claims.
8. **Whether a relocation is about to happen is suggestive at late layers and does not survive
   correction.** Smallest p = 0.033; Benjamini–Hochberg at q = 0.05 needs 0.00625.
9. **At n ≈ 35 and 2,048 dimensions, the standard "best single unit" analysis produces its own
   conclusion from noise.** Mean in-sample AUC 0.847 with real labels, 0.852 with shuffled ones.

---

## Technical setup

**Corpus.** 594 tasks from real advisories (GHSA/CVE), each a repository at the parent of a known
fix commit. Produced 1,288 casefiles and 504 recorded trajectories. Gold = files and line ranges
named by the fix commit diff. `BerriAI/litellm` is held out and contributes nothing.

**Models.** Antares-1B (locate, ranks candidate files, run 3× per task for agreement);
Qwen3.6-35B-A3B (plan/judge, and the model all probes are run on); FastContext-4B (enrich, explores
inside a file and cites lines). All served locally; the 35B extraction ran on a rented A100 with
weights offloaded to system RAM.

**What is quantified.**

| quantity | definition | measurement |
|---|---|---|
| relocation | a statement in reasoning naming a file other than the one the run was asked about | mined with four filters (position in step, supersession, real-extension, hedge detection); removed 33 of 133 raw candidates on a held-out sample |
| precision of a relocation | does the named file appear in the fix commit | exact repo-relative path match; a looser basename rule is reported alongside |
| reached the output | does the claim appear in the structured casefile | string match against the casefile's candidate paths |
| decodable before stated | does a readout at the cut prefer the target over a decoy | top-20 token ids at every layer; decoy = another case's real target, seeded `random.Random(20260906)` |
| separability of right vs wrong | AUC of a classifier on the residual | logistic regression, PCA inside each fold, leave-one-claim-out, claim-level permutation null |

**Cut points.** Each case replays a real trajectory to two points: `at_path` (mid-sentence, after
the opening backtick) and `at_sentence` (before the naming clause exists). Only claims where these
differ are counted — for a minority the miner found no earlier boundary and they are byte-identical.

**Metrics.** McNemar exact for paired target-vs-decoy; Fisher exact for the prompt-era 2×2;
permutation nulls (300–500 shuffles) for every probe AUC, reported as the null's mean and 95th
percentile rather than against 0.5; Benjamini–Hochberg across layers.

**Reproduction.** `verify/verify_findings.ipynb` re-derives every published number from the raw
files and prints computed against claimed — 40 of 40 pass, no GPU, about two seconds. It fetches
its own inputs, so it runs from a cold Colab runtime with nothing uploaded.

---

## Strongest evidence against these hypotheses

**Against "the reasoning contains information the output loses":** 212 of 218 relocations reach the
output. The hypothesis that motivated the project is false in this corpus, and the six exceptions
turned out to be caused by my own prompt.

**Against "the answer is decodable before it is stated":** every activation-level probe returned a
null or an artifact. The one that looked strongest (20 of 24, p = 0.0015, rising monotonically with
depth) collapsed to two scoreable claims once the decoy was restricted to the same repository.

**Against my own method:** the shuffled-label control reaches the same in-sample AUC as the real
labels (0.852 vs 0.847), which means the single-dimension analysis has no discriminating power at
this n regardless of what it appears to show.

**Against the pilot's decoy control:** it turns out to control for filename plausibility but not
repository identity, so every target-vs-decoy floor in the report is too low.

---

## Biggest limitations

1. **n is far too small.** 37 claims mined, ~30–35 in the pilot, 23 matched pairs in the event
   probe. The Q2 probe could not have detected any effect below AUC 0.676. **Addressable:** 218
   relocations exist; I extracted 75, which is what one A100 session covered. This was a compute
   and time constraint, not a corpus one.
2. **The decoy does not control for repository identity.** 71 of 75 decoys name a file from another
   project. **Partly addressable now** — I found it and reported it — but a within-repository decoy
   leaves two scoreable claims, so the question needs a corpus *built* for it. Not fixable by
   filtering.
3. **One model, one corpus, one cut point.** Nothing here establishes generality.
4. **The baseline is not an interpretability result.** It lets the model generate up to 2,400
   tokens; the lens reads one activation. It bounds what the lens missed, it does not show the
   answer is in the residual stream.
5. **Gold is fix-commit files**, which under-credits tests, callers and configuration a real fix
   touches but the reference patch does not.
6. **The prompt-era finding is observational.** The clause was removed for unrelated reasons and I
   compared eras after the fact. **Addressable and not done:** rerun both prompts on the same tasks.
7. **Three of the four follow-up questions were added or reframed after seeing early results.**
   The denominators differ between analyses for stated reasons, but the sequence was not
   pre-registered.

---

## How did you use LLMs? — **YOU MUST WRITE THIS**

Handing this particular question to an LLM would be the wrong move, so what follows is **material,
not an answer**. All of it is factual; pick what is true of how you worked and write it yourself.

**Division of labour, honestly stated.** Most of the code and prose was LLM-generated under your
direction; the research direction, the decisions about what to keep, and every rejection were
yours. Say the real proportion. A reviewer can usually tell, and the answer to this question is
itself a test.

**Specific mistakes caught — by you:**

- I diagnosed an italics-rendering bug as a Google Docs paste problem **twice** and reflowed a file
  to fix it. You said it persisted a third time. The actual cause was a CSS selector I had written:
  `p > em:only-child` — `:only-child` counts element children and ignores text nodes, so any
  paragraph containing a single italicised word matched. 18 of 39 italics were rendering as blocks.
- You asked why the denominator kept changing between results (37, 30, 31, 24, 35). Each was
  correct, but checking it surfaced that "20 of 24" was the *best layer* and I had taken both
  halves of the fraction from it without saying so.
- You rejected three headline phrasings and a subtitle.
- You pointed out an empty cell in a card grid; it was a two-column auto-fit grid with three items,
  drawing its borders by showing a background through the gaps.
- You remembered that pages 1–3 are meant to be an executive summary. There wasn't one.

**Specific mistakes caught by controls rather than by anyone's judgement:**

- A probe scored AUC 0.894 at layer 0 — the embedding output, before any transformer block runs.
  No mechanism can explain that. The two exports were cutting on different punctuation.
- A permutation null was shuffling labels at the readout level when a claim's readouts share one
  label, which made the null harder than the thing it was a null for.
- A per-mention binomial reported p = 0.0 on at most 35 independent observations.

**Mistakes that reached a published number before being caught:** five, each flattering the
hypothesis. They are listed in the write-up with the wrong value beside the corrected one.

**Process points worth making:** every published number is recomputed by a notebook that runs in
two seconds; figures draw their own permutation nulls so "is this above chance" is a visual
question; the extraction refuses to start below 34 GiB free VRAM; the script that reads gold does so
only after every model claim already exists.

---

## Prior experience with mechanistic interpretability — **YOU MUST WRITE THIS**

I have no basis to answer this.

## 1–3 pieces of evidence you could do good research — **YOU MUST WRITE THIS**

~100 words, and explicitly not about this project. Unusual backgrounds are invited.

## Why Neel's stream specifically — **YOU MUST WRITE THIS**

## Likelihood of joining (Sept 28 – Oct 30) — **YOU MUST WRITE THIS**

## Anything else I should know (optional)

Candidate, if you want one: the project began as a harness-engineering exercise and the
faithfulness question emerged from it. Eight measurements were wrong before they were checked, and
that is documented in the write-up rather than tidied away — including a server flag named for the
behaviour it did not produce, and a parser that reported reading 1,288 of 1,288 casefiles while
finding candidates in 437.
