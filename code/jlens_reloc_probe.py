"""Is the relocated file already in the activations before the model writes it?

THE EXPERIMENT. Each case is a real trajectory in which qwen3.6-35b, examining file A, wrote that
the vulnerability is actually in file B. We replay the conversation up to the moment BEFORE B is
written and apply the Jacobian lens at that position. If B's tokens are already what the
activations are disposed to produce, the sentence is a readout of a conclusion the forward pass
had already reached rather than the reasoning that produced it.

That is a chain-of-thought faithfulness question with an answer key: the fix commit says whether B
was right.

TWO CUT POINTS, testing different strengths of claim:
  at_path      stops immediately before the path string. The model has written "the vulnerability
               is present in `" and is about to name it.
  at_sentence  stops before the whole clause. Nothing on the page points at B yet. 22 of 30 cases
               have a genuinely earlier sentence cut.

THE CONTROL IS THE POINT. A lens that ranks any plausible source filename highly would "find" B
every time. So every readout is scored against a DECOY: another case's target basename, shuffled
in. Same distribution of filenames, wrong answer. If target and decoy score alike, the lens is
reading "a file is coming", not which one.

And logit lens (use_jacobian=False) runs on every position, because the transport has to beat
decoding the residual stream directly or it is not doing the work.

Usage (Colab A100 high-RAM; ~67 GB in bf16, offloads to system RAM):
    colab new -s reloc --gpu A100
    colab upload -s reloc experiments/reloc_cases.json /content/reloc_cases.json
    colab exec -s reloc -f experiments/jlens_reloc_probe.py
    colab stop -s reloc
"""

MODEL = "Qwen/Qwen3.6-35B-A3B"
LENS_REPO = "camilablank/workspace-lenses"
LENS_FILE = "qwen3.6-35b-a3b/j-lens/lens.pt"
CASES_JSON = "/content/reloc_cases.json"
# 75 after the export was fixed to read reasoning_content, the field the miner
# actually found the claims in. The old 30 were the subset where the path also
# appeared in the visible message, so their cut point was in the wrong text.
MAX_CASES = 75
# The conversation prefixes are whole enrichment transcripts and can be very long. Keep the tail:
# the lens is applied at the final position, and the claim is formed near the end.
MAX_PROMPT_CHARS = 16000
# jlens.apply defaults to 512, which silently discarded the part of the prompt the
# experiment is about.
#
# Two runs then OOM'd and I blamed this number. Wrong: both died inside from_pretrained,
# before a single case ran, and the requested allocation tracked max_memory (36 GiB -> 34.69,
# 28 GiB -> 26.94) rather than sequence length. It was the allocator warmup fighting a model
# the PREVIOUS run had left resident -- see free_gpu(). 4096 is a real window, and the probe
# reads the LAST token under left-side truncation, so the claim and 4,000 tokens of its
# context are what survive.
MAX_SEQ_LEN = 4096

import json
import os
import random
import sys


def toks_of(tok, s):
    """Token ids that spell a filename, both bare and space-prefixed."""
    out = set()
    for v in (s, " " + s, s.split(".")[0], " " + s.split(".")[0]):
        try:
            out.update(tok(v, add_special_tokens=False)["input_ids"])
        except Exception:
            pass
    return out


def free_gpu(torch) -> float:
    """Release whatever the previous exec left on the card, and report what is actually free.

    colab exec reuses ONE kernel. When a run raises, sys.last_traceback keeps its frames alive,
    those frames hold `hf`, and `hf` holds 35 GiB of weights -- so the next run's allocator
    warmup asks for ~96% of its max_memory budget on a card that is already full and dies inside
    from_pretrained. Two runs were lost to this and both looked like "the window is too big".
    """
    import gc  # gc is not needed at module scope; sys is imported there and must NOT be
    # re-imported here -- a function-local import shadows the module binding for the whole
    # function, which is exactly what broke the per-row append with UnboundLocalError.

    for attr in ("last_traceback", "last_value", "last_type"):
        if getattr(sys, attr, None) is not None:
            setattr(sys, attr, None)
    try:  # IPython parks recent results in _, __, ___ and Out[]
        ns = get_ipython().user_ns  # noqa: F821
        for k in ("_", "__", "___", "_i", "_ii", "_iii"):
            ns.pop(k, None)
        ns.get("Out", {}).clear()
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    free = torch.cuda.mem_get_info()[0] / 2**30
    print(f"GPU free after cleanup: {free:.1f} GiB", flush=True)
    return free


def main() -> int:
    import torch
    import transformers
    import jlens

    # CHECK THE CARD BEFORE ASKING FOR IT. This is the line whose absence cost two runs.
    if free_gpu(torch) < 34.0:
        print("REFUSING: the card is not empty. A previous run is still resident; restart the "
              "Colab runtime rather than reading this run's numbers.", flush=True)
        return 2

    cases = json.load(open(CASES_JSON, encoding="utf-8"))[:MAX_CASES]
    print(f"cases: {len(cases)}", flush=True)

    # PIN THE HEAD AND THE FINAL NORM TO THE GPU. 67 GB in bf16 does not fit a 40 GB A100, so
    # accelerate offloads most blocks -- fine, because their hooks materialise them during the
    # forward pass. But jlens.unembed calls self._lm_head(self._final_norm(x)) DIRECTLY, outside
    # any dispatched forward, so if those two land on the offload device they are meta tensors
    # with nothing to materialise them and every apply() dies with "Cannot copy out of meta
    # tensor; no data!" -- which is exactly what the first A100 run produced, on all 30 cases.
    print(f"loading {MODEL} (bf16, offloading, head pinned) ...", flush=True)
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
    print("lm_head on:", hf.lm_head.weight.device, flush=True)
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    # TRUNCATE FROM THE LEFT. jlens.encode calls the tokenizer with truncation=True, which keeps
    # the FIRST max_seq_len tokens by default -- so a 6,000-token transcript was cut to its first
    # 512 and positions=[-1] probed the middle of the system prompt, thousands of tokens from the
    # claim. That produced 0 hits for target AND decoy across 39 layers and 60 runs, which reads
    # like a null result and is actually a prompt that never contained the thing being asked about.
    tok.truncation_side = "left"
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE)
    print("model + lens ready", flush=True)

    # Decoys: every other case's target, shuffled. Matched difficulty, wrong answer.
    rng = random.Random(20260906)
    targets = [c["target_basename"] for c in cases]
    decoys = targets[:]
    rng.shuffle(decoys)
    for i in range(len(decoys)):
        if decoys[i] == targets[i] and len(set(targets)) > 1:
            j = (i + 1) % len(decoys)
            decoys[i], decoys[j] = decoys[j], decoys[i]

    # RESUME A PARTIAL RUN. Set ONLY_CASES="73,74" to score just those indices. The loop still
    # runs over the FULL case list and skips, rather than slicing the list first, because the
    # decoy shuffle above is seeded over all 75 -- slicing would re-pair every target with a
    # different decoy and the new rows would not be comparable with the old ones.
    only = os.environ.get("ONLY_CASES", "").strip()
    only_set = {int(x) for x in only.split(",") if x.strip().isdigit()} if only else None
    if only_set:
        print(f"ONLY_CASES set: scoring {sorted(only_set)} of {len(cases)}", flush=True)

    results = []
    for i, c in enumerate(cases):
        if only_set is not None and i not in only_set:
            continue
        for cut in ("prefix_at_sentence", "prefix_at_path"):
            convo = "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in c["messages"])
            prompt = (convo + "<|assistant|>\n" + c[cut])[-MAX_PROMPT_CHARS:]

            try:
                # max_seq_len defaults to 512 in jlens.apply. The claim sits at the END of a
                # multi-thousand-token transcript, so the window has to reach it.
                jl, _, ids = lens.apply(model, prompt, positions=[-1], max_seq_len=MAX_SEQ_LEN)
                ll, _, _ = lens.apply(model, prompt, positions=[-1], max_seq_len=MAX_SEQ_LEN,
                                      use_jacobian=False)
                # SHOW WHAT THE MODEL ACTUALLY READ. A run that scores 0 for target AND decoy
                # is far more often a prompt that never contained the claim than a real null --
                # that exact signature already appeared once here and twice elsewhere today, and
                # each time it was empty input, not an absent effect. Printing the tail makes
                # the two distinguishable without another A100 hour.
                if i < 2:
                    seq = ids[0] if getattr(ids, "ndim", 1) > 1 else ids
                    print(f"    [window {len(seq)} tok] ...{tok.decode(seq[-32:])!r}", flush=True)
            except Exception as e:
                print(f"  case {i} {cut}: FAILED {type(e).__name__}: {e}", flush=True)
                continue

            tgt = toks_of(tok, c["target_basename"])
            dec = toks_of(tok, decoys[i])
            layers = sorted(jl)
            row = {"case": i, "cut": cut, "run_id": c["run_id"],
                   "target": c["target_basename"], "decoy": decoys[i],
                   "j_target": 0, "j_decoy": 0, "l_target": 0, "l_decoy": 0,
                   # RECORD THE RAW READOUT, NOT JUST THE VERDICT. The first run stored how many
                   # layers hit and never which, so two questions it could have answered needed a
                   # second A100 hour: where in the stack the file becomes decodable, and what the
                   # numbers look like under a scoring rule that ignores a shared `.py`. Top-20
                   # ids per layer is ~700 KB for the whole run and makes every later scoring
                   # question a re-analysis instead of a re-run.
                   "layer_ids": layers,
                   "j_top20": {}, "l_top20": {},
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
            # APPEND EACH ROW AS IT LANDS. The previous run wrote its results only after the loop
            # and the provider pruned the session at case 72 of 75 -- so a run that had completed
            # 96% of its work left nothing behind, and its per-layer ids are unrecoverable. One
            # JSON object per line, flushed and fsynced, costs milliseconds and means a prune now
            # costs only the tail. Read it back with [json.loads(l) for l in open(path)].
            with open("/content/reloc_rows.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            print(f"  case {i:>2} {cut:18} target hits j={row['j_target']:>3}/{row['layers']} "
                  f"l={row['l_target']:>3}  decoy j={row['j_decoy']:>3} l={row['l_decoy']:>3}",
                  flush=True)

    # PER CLAIM, NOT PER MENTION. The 30 cases are 12 distinct (repo, target) claims -- one
    # restated 7 times, another 6, another 5. Reporting 30 rows would imply 30 independent
    # observations and inflate in the flattering direction, which is the exact error the report
    # this experiment belongs to exists to retract. A claim counts once, hit if ANY mention hits.
    def per_claim(rs, field):
        byclaim = {}
        for r in rs:
            k = (r["run_id"].split("-eval")[0], r["target"])
            byclaim[k] = byclaim.get(k, 0) or (1 if r[field] > 0 else 0)
        return sum(byclaim.values()), len(byclaim)

    print("\n=== SUMMARY ===", flush=True)
    for cut in ("prefix_at_sentence", "prefix_at_path"):
        rs = [r for r in results if r["cut"] == cut]
        if not rs:
            continue
        n = len(rs)
        L = sum(r["layers"] for r in rs) or 1
        print(f"{cut}  (n={n})")
        print(f"  j-lens     target {sum(r['j_target'] for r in rs)/L:.3f} of layers   "
              f"decoy {sum(r['j_decoy'] for r in rs)/L:.3f}")
        print(f"  logit-lens target {sum(r['l_target'] for r in rs)/L:.3f} of layers   "
              f"decoy {sum(r['l_decoy'] for r in rs)/L:.3f}")
        hit = sum(1 for r in rs if r["j_target"] > 0)
        dhit = sum(1 for r in rs if r["j_decoy"] > 0)
        print(f"  mentions with ANY j-lens target hit: {hit}/{n}   decoy: {dhit}/{n}")
        th, tc = per_claim(rs, "j_target")
        dh, _ = per_claim(rs, "j_decoy")
        lh, _ = per_claim(rs, "l_target")
        print(f"  CLAIMS (the unit that counts): j-lens target {th}/{tc}  decoy {dh}/{tc}  "
              f"logit-lens target {lh}/{tc}")
    # The raw readout goes to a FILE, not the log. With top-20 ids per layer this is ~700 KB;
    # printing it would bury the summary the poller greps for, and the poller is how the run is
    # watched. Download it with `colab download -s <session> /content/reloc_results.json`.
    out = "/content/reloc_results.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh)
    # `os` is imported at module scope. A second `import os` HERE made os function-local for the
    # whole of main(), so the os.fsync in the per-row append above raised UnboundLocalError on
    # case 0 -- the module-level import was shadowed by an import that runs 200 lines later.
    print(f"\nraw readout: {out}  ({os.path.getsize(out):,} bytes, "
          f"{len(results)} rows x {results[0]['layers'] if results else 0} layers)", flush=True)
    print("\nREAD THE CONTROL FIRST. If decoy tracks target, the lens is reading 'a filename is "
          "coming', not which one, and nothing here is about relocation.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
