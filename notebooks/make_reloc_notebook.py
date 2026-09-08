"""Emit a self-contained Colab notebook that replicates the jlens relocation probe.

The notebook is the honest version of the seven attempts it took to get this measurement: every
trap that produced a plausible-looking wrong answer is a guard in the code, with the reason next
to it. In particular it keeps the decoy control, the logit-lens baseline, the per-claim collapse,
and the paired significance test, because the per-mention numbers look considerably better than
the truth and the whole point is not to quote them.

Run: python notebooks/make_reloc_notebook.py
Then upload notebooks/jlens_reloc_replication.ipynb and experiments/reloc_cases.json to Colab (A100).
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "jlens_reloc_replication.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip().splitlines(True)}


CELLS = [
    md(r"""
# Is the relocated file already in the activations before the model says it?

Each case is a real trajectory in which `qwen3.6-35b`, examining file A, wrote that the
vulnerability is *actually* in file B. We replay the conversation to the moment **before** B is
written and apply the Jacobian lens there. If B is already what the activations are disposed to
produce, the sentence is a readout of a conclusion the forward pass had already reached, rather
than the reasoning that produced it.

**Two cut points, testing different strengths of claim**

| cut | prompt ends with | what it tests |
|---|---|---|
| `at_path` | ``The vulnerability is present in ` `` | weak: the model is mid-sentence, partly next-token prediction |
| `at_sentence` | the previous sentence, complete | strong: nothing on the page points at B yet |

**The control is the point.** A lens that ranks any plausible source filename highly would "find"
B every time. Every readout is therefore scored against a **decoy** — another case's target,
shuffled in. Same filename distribution, wrong answer. If target and decoy score alike, the lens
is reading *a file is coming*, not *which one*. Read target against decoy, never against zero.

**Requires an A100 (40 GB).** The model is ~67 GB in bf16 and offloads to system RAM.
"""),

    md(r"""
## 0. Traps this notebook has guards for

Seven attempts produced this measurement. Four of the failures returned *plausible* wrong answers
rather than errors, which is why each guard below carries its reason:

- `jlens.apply(max_seq_len=...)` **defaults to 512**, and the tokenizer truncates from the
  **right** — so a 6,000-token transcript became its own first 512 tokens and the probe read the
  middle of the system prompt. That scored **0 for target and decoy across 39 layers**, which
  looks exactly like a clean null result. Fixed by a real window plus `truncation_side="left"`,
  and the notebook prints the decoded tail so you can see what the model actually read.
- `jlens.unembed` calls `lm_head` **outside** the dispatched forward, so if accelerate offloads
  the head it is a meta tensor and every call dies. Fixed by pinning head/norm/embeddings.
- A raised exception keeps the previous model **resident on the GPU**; the next load then OOMs
  while asking for ~96% of its `max_memory`. Fixed by a free-VRAM preflight that refuses rather
  than discovering it 20 minutes in.
"""),

    code(r"""
# Deps. jlens is not on PyPI.
!pip install -q "git+https://github.com/anthropics/jacobian-lens.git" accelerate
"""),

    code(r"""
# The cases. Upload reloc_cases.json (produced by experiments/export_reloc_cases.py).
import os

CASES_JSON = "/content/reloc_cases.json"
if not os.path.exists(CASES_JSON):
    from google.colab import files
    up = files.upload()
    name = next(iter(up))
    if name != os.path.basename(CASES_JSON):
        os.rename(name, CASES_JSON)
print("cases file:", CASES_JSON, os.path.getsize(CASES_JSON), "bytes")
"""),

    code(r"""
MODEL = "Qwen/Qwen3.6-35B-A3B"
LENS_REPO = "camilablank/workspace-lenses"
LENS_FILE = "qwen3.6-35b-a3b/j-lens/lens.pt"
MAX_CASES = 30

# Keep the TAIL of the transcript: the lens is applied at the final position and the claim forms
# near the end. Pair this with truncation_side = "left" below or the window keeps the wrong end.
MAX_PROMPT_CHARS = 16000
MAX_SEQ_LEN = 4096

import json
cases = json.load(open(CASES_JSON, encoding="utf-8"))[:MAX_CASES]
print(f"cases: {len(cases)}")
print("distinct claims:",
      len({(c["run_id"].split("-eval")[0], c["target_basename"]) for c in cases}))
"""),

    code(r"""
# PREFLIGHT: check the card BEFORE asking for it.
# A previous cell that raised keeps its model alive through sys.last_traceback and accelerate's
# hooks. Three runs OOM'd inside from_pretrained fighting a predecessor that no longer existed
# logically but still held 35 GiB. If this refuses: Runtime > Restart session, then re-run.
import gc, sys, torch

def free_gpu():
    for attr in ("last_traceback", "last_value", "last_type"):
        if getattr(sys, attr, None) is not None:
            setattr(sys, attr, None)
    try:
        ns = get_ipython().user_ns
        for k in ("_", "__", "___", "_i", "_ii", "_iii"):
            ns.pop(k, None)
        ns.get("Out", {}).clear()
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    free = torch.cuda.mem_get_info()[0] / 2**30
    print(f"GPU free: {free:.1f} GiB")
    return free

assert free_gpu() > 34.0, "card not empty - Runtime > Restart session before loading"
"""),

    code(r"""
# Load. ~67 GB bf16 does not fit a 40 GB A100, so accelerate offloads most blocks -- fine, their
# hooks materialise them during the forward pass. But jlens.unembed calls lm_head DIRECTLY,
# outside any dispatched forward, so lm_head / model.norm / embed_tokens must be pinned to the
# GPU or every apply() dies with "Cannot copy out of meta tensor; no data!".
import transformers, jlens
from accelerate import infer_auto_device_map, init_empty_weights

cfg = transformers.AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
with init_empty_weights():
    skel = transformers.AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
dmap = infer_auto_device_map(
    skel, max_memory={0: "32GiB", "cpu": "76GiB"}, dtype=torch.bfloat16,
    no_split_module_classes=getattr(skel, "_no_split_modules", None) or [])
for k in list(dmap):
    if k in ("lm_head", "model.norm", "model.embed_tokens"):
        dmap[k] = 0
del skel

hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16, device_map=dmap, trust_remote_code=True)
tok = transformers.AutoTokenizer.from_pretrained(MODEL)

# TRUNCATE FROM THE LEFT. jlens.encode passes truncation=True, which keeps the FIRST max_seq_len
# tokens by default. With the claim at the end of the transcript, the default silently probes the
# wrong end of the prompt and scores zero for target AND decoy.
tok.truncation_side = "left"

model = jlens.from_hf(hf, tok)
lens = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE)
print("lm_head on:", hf.lm_head.weight.device)
print("model + lens ready")
"""),

    code(r"""
# Decoys: every other case's target, shuffled. Matched difficulty, wrong answer.
import random

def toks_of(tok, s):
    out = set()
    for v in (s, " " + s, s.split(".")[0], " " + s.split(".")[0]):
        try:
            out.update(tok(v, add_special_tokens=False)["input_ids"])
        except Exception:
            pass
    return out

rng = random.Random(20260906)
targets = [c["target_basename"] for c in cases]
decoys = targets[:]
rng.shuffle(decoys)
for i in range(len(decoys)):
    if decoys[i] == targets[i] and len(set(targets)) > 1:
        j = (i + 1) % len(decoys)
        decoys[i], decoys[j] = decoys[j], decoys[i]
print("example pairing:", targets[0], "vs decoy", decoys[0])
"""),

    code(r"""
# The probe. ~10-40 min on an A100: 2 cuts x 2 lens modes per case, through offloaded layers.
results = []
for i, c in enumerate(cases):
    for cut in ("prefix_at_sentence", "prefix_at_path"):
        convo = "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in c["messages"])
        prompt = (convo + "<|assistant|>\n" + c[cut])[-MAX_PROMPT_CHARS:]
        try:
            jl, _, ids = lens.apply(model, prompt, positions=[-1], max_seq_len=MAX_SEQ_LEN)
            ll, _, _ = lens.apply(model, prompt, positions=[-1], max_seq_len=MAX_SEQ_LEN,
                                  use_jacobian=False)
            # SHOW WHAT THE MODEL ACTUALLY READ. Zero for target AND decoy is far more often an
            # empty window than a real null; printing the tail keeps the two distinguishable.
            if i < 2:
                seq = ids[0] if getattr(ids, "ndim", 1) > 1 else ids
                print(f"  [window {len(seq)} tok] ...{tok.decode(seq[-32:])!r}")
        except Exception as e:
            print(f"  case {i} {cut}: FAILED {type(e).__name__}: {e}")
            continue

        tgt, dec = toks_of(tok, c["target_basename"]), toks_of(tok, decoys[i])
        layers = sorted(jl)
        # RECORD THE RAW READOUT, NOT JUST THE VERDICT. The first version of this experiment
        # stored how many layers hit and never which, so two later questions -- where in the stack
        # the file becomes decodable, and what the numbers look like once a shared `.py` is
        # discounted -- each cost another A100 hour. Top-20 ids per layer is ~700 KB for the whole
        # run and turns every later scoring question into a re-analysis.
        row = {"case": i, "cut": cut, "run_id": c["run_id"], "target": c["target_basename"],
               "decoy": decoys[i], "j_target": 0, "j_decoy": 0, "l_target": 0, "l_decoy": 0,
               "layer_ids": layers, "j_top20": {}, "l_top20": {},
               "target_tok": sorted(tgt), "decoy_tok": sorted(dec)}
        for layer in layers:
            jt_ids = jl[layer][0].topk(20).indices.tolist()
            lt_ids = ll[layer][0].topk(20).indices.tolist()
            row["j_top20"][str(layer)] = jt_ids
            row["l_top20"][str(layer)] = lt_ids
            jt, lt = set(jt_ids), set(lt_ids)
            row["j_target"] += 1 if jt & tgt else 0
            row["j_decoy"] += 1 if jt & dec else 0
            row["l_target"] += 1 if lt & tgt else 0
            row["l_decoy"] += 1 if lt & dec else 0
        row["layers"] = len(layers)
        results.append(row)
        print(f"  case {i:>2} {cut:18} target j={row['j_target']:>3}/{row['layers']} "
              f"l={row['l_target']:>3}   decoy j={row['j_decoy']:>3} l={row['l_decoy']:>3}")

json.dump(results, open("/content/reloc_results.json", "w"), indent=1)
print("saved /content/reloc_results.json")
"""),

    code(r"""
# PER CLAIM, NOT PER MENTION, on DISTINCTIVE tokens, with a paired test.
#
# Three corrections, each of which changes the answer:
#
# 1. UNIT. The 30 mentions are 17 distinct (repo, target) claims -- one restated 7 times, another
#    6. Counting mentions triple-counts one underlying claim and inflates in the flattering
#    direction.
# 2. TOKENS. A hit on ANY token of a filename counts `.py`, which two unrelated files share, for
#    the target and the decoy at once. Dropping shared tokens from BOTH sides is the fix. Doing it
#    one-sided -- voiding the decoy but not the target -- corrects only the side that hurts the
#    hypothesis, and moves p from 0.109 to 0.008 on nothing.
# 3. PAIRING. Target and decoy are scored on the SAME forward pass, so McNemar's exact test on the
#    discordant pairs, not a two-proportion z-test.
import math

def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)

def per_claim(rs, distinctive):
    claims = {}
    for r in rs:
        tgt, dec = set(r["target_tok"]), set(r["decoy_tok"])
        if distinctive:
            shared = tgt & dec
            tgt, dec = tgt - shared, dec - shared
        d = claims.setdefault((r["run_id"].split("-eval")[0], r["target"]),
                              {"t": 0, "d": 0, "l": 0})
        for layer in r["layer_ids"]:
            j = set(r["j_top20"][str(layer)])
            l_ = set(r["l_top20"][str(layer)])
            d["t"] |= 1 if j & tgt else 0
            d["d"] |= 1 if j & dec else 0
            d["l"] |= 1 if l_ & tgt else 0
    return claims

for cut in ("prefix_at_sentence", "prefix_at_path"):
    rs = [r for r in results if r["cut"] == cut]
    print(f"\n{cut}")
    for label, distinctive in (("any token ", False), ("DISTINCTIVE", True)):
        claims = per_claim(rs, distinctive)
        n = len(claims)
        jt = sum(v["t"] for v in claims.values())
        jd = sum(v["d"] for v in claims.values())
        lt = sum(v["l"] for v in claims.values())
        b = sum(1 for v in claims.values() if v["t"] and not v["d"])
        cc = sum(1 for v in claims.values() if v["d"] and not v["t"])
        p = mcnemar_exact(b, cc)
        print(f"  {label}  target {jt}/{n}   DECOY {jd}/{n} (floor)   "
              f"logit {lt}/{n} (transport must beat this)   p={p:.4f}"
              + ("" if p < 0.05 else "  NOT significant"))
"""),

    code(r"""
# THE EMERGENCE CURVE -- where in the stack the file becomes decodable.
# Only possible because the loop above stored top-20 ids per layer rather than a hit count.
import collections

def curve(cut, distinctive=True):
    per = collections.defaultdict(lambda: collections.defaultdict(
        lambda: {"t": 0, "d": 0, "l": 0}))
    layers = []
    for r in (x for x in results if x["cut"] == cut):
        layers = r["layer_ids"]
        tgt, dec = set(r["target_tok"]), set(r["decoy_tok"])
        if distinctive:
            shared = tgt & dec
            tgt, dec = tgt - shared, dec - shared
        k = (r["run_id"].split("-eval")[0], r["target"])
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
        for key in ("t", "d", "l"):
            out[key].append(round(sum(v[key] for v in cs.values()) / n, 3))
    return out

for cut in ("prefix_at_sentence", "prefix_at_path"):
    c = curve(cut)
    first = next((i for i, v in enumerate(c["t"]) if v > 0), None)
    print(f"\n{cut}   layers {c['layers'][0]}..{c['layers'][-1]}")
    print(f"  j-lens target first nonzero at layer {first}, peak {max(c['t']):.0%}")
    print(f"  target {c['t']}")
    print(f"  decoy  {c['d']}")
    print(f"  logit  {c['l']}   <- note where this arrives")
"""),

    md(r"""
## How to read the output

**Control first.** A target rate only means something against its decoy. The
top-20-of-39-layers criterion is permissive by design, and the decoy measures exactly how
permissive — it is applied identically to both.

**Then the baseline.** The Jacobian transport has to beat the plain logit lens, or it is not
doing the work the method claims.

**Then per-claim, with the p-value.** Per-mention rates run considerably higher than per-claim
rates and are the wrong unit.

### What this run produced (n = 17 claims, 9 repos, two independent runs, identical counts)

| | `at_sentence` (strong claim) | `at_path` (weak claim) |
|---|---|---|
| j-lens target | 9/17 (52.9%) | 16/17 (94.1%) |
| j-lens decoy, any token | 3/17 (17.6%) | 5/17 (29.4%) |
| j-lens decoy, distinctive | 3/17 (17.6%) | **2/17 (11.8%)** |
| logit-lens target | 6/17 (35.3%) | 14/17 (82.4%) |
| McNemar exact, distinctive | **p = 0.109 — not significant** | **p = 0.0001** |

**The hit rate is not the finding.** The emergence curve is: at `at_path` the transport recovers
the target from **layer 20 of 39** and reaches 53% of claims by layer 30, while the plain logit
lens stays at or below 6% until **layer 38** — the final layer, where it is decoding the token the
model is about to emit. Roughly eighteen layers of lead, with the decoy flat below 6% throughout.
Per readout the transport is also never the worse of the two on any of the 60.

**The significant cut is the uninteresting one.** At `at_path` the model has already written
``present in ` ``, so much of the effect is next-token prediction. At `at_sentence`, where the
claim would be meaningful, the direction is right but n = 17 gives p = 0.11 — **suggestive and
underpowered, not established.** More claims is the only fix; the per-mention framing (12/30 vs
3/30) would look better and would be the wrong unit.

**Correcting the scoring defect does not rescue it, and we expected it to.** The decoy control
exposed that a shared `.py` scored for target and decoy at once. A one-sided fix — voiding the
contaminated decoy hits only — moved `at_sentence` to p = 0.008. Scored symmetrically, with shared
tokens dropped from *both* sides, it is unchanged: those decoy hits fired on the decoy's own
distinctive tokens. The defect was real and it was not hiding an effect.
"""),
]

nb = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "A100", "machine_shape": "hm"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size} bytes, {len(CELLS)} cells)")
