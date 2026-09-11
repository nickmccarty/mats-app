"""Save the residual stream at the cut, for the probe experiments (follow-up Q1 and Q2).

WHAT THIS IS FOR. The lens pilot asked whether a PUBLISHED readout (a Jacobian lens) recovers the
relocated file. It does not. Q1 and Q2 ask a different question: is the answer linearly decodable
at all, by a readout TRAINED on this task -- and do the activations distinguish a relocation that
turns out to be right from one that turns out to be wrong.

Both need the residual vectors themselves. The lens probe recorded top-20 token ids, which is the
readout's verdict and not its input, so neither question could be asked of that file.

NO LENS HERE. `output_hidden_states=True` returns the residual stream at every layer directly, so
this script needs the model and nothing else.

WHAT IT SAVES. For each case and each cut point, the hidden state at the LAST position -- the
token the claim is about -- at every layer, in fp16. 150 readouts x 40 layers x 4096 dims is about
49 MB, which is small enough to keep whole and re-analyse offline forever.

WRITTEN INCREMENTALLY, for the reason the last three runs were nearly lost: the provider prunes
long sessions without warning, and a file written only at the end is a file that does not exist.
The npz is rewritten every few readouts and should be pulled down as it grows.

Usage (Colab A100; ~67 GB bf16, offloads to system RAM):
    colab upload -s <s> experiments/reloc_cases.json /content/reloc_cases.json
    colab upload -s <s> experiments/extract_residuals.py /content/extract_residuals.py
    colab exec   -s <s> -f experiments/colab_run_extract.py
    MSYS_NO_PATHCONV=1 colab download -s <s> /content/resid.npz resid.npz
"""

import json
import os
import sys

MODEL = "Qwen/Qwen3.6-35B-A3B"
MAX_PROMPT_CHARS = 16000
MAX_SEQ_LEN = 4096
SAVE_EVERY = 4          # readouts between rewrites of the npz

# Overridable so the SAME code path reads the relocation cases and the matched control cuts. A
# second copy of this file that differed only in three constants is a second copy that drifts --
# and the whole point of the controls is that they are processed identically to the positives.
CASES_JSON = os.environ.get("RESID_CASES", "/content/reloc_cases.json")
OUT = os.environ.get("RESID_OUT", "/content/resid.npz")
META_OUT = os.environ.get("RESID_META", "/content/resid_meta.json")
MAX_CASES = int(os.environ.get("RESID_MAX", "75"))
# controls have one meaningful cut, not two; both fields hold the same prefix there
CUTS = tuple(os.environ.get("RESID_CUTS", "prefix_at_sentence,prefix_at_path").split(","))


def main() -> int:
    import numpy as np
    import torch
    import transformers

    # Same precondition as the lens probe: a crashed predecessor leaves 35 GiB resident and the
    # next load dies inside from_pretrained asking for ~96% of its budget.
    free = torch.cuda.mem_get_info()[0] / 2**30
    print(f"GPU free: {free:.1f} GiB", flush=True)
    if free < 34.0:
        print("REFUSING: card not empty; restart the runtime.", flush=True)
        return 2

    cases = json.load(open(CASES_JSON, encoding="utf-8"))[:MAX_CASES]
    print(f"cases: {len(cases)}", flush=True)

    # PREFLIGHT, BEFORE THE 12-MINUTE MODEL LOAD. A missing key is a one-second check that,
    # discovered after from_pretrained, costs a quarter of an hour of A100 time and reads as a
    # crash rather than as a malformed input. This exists because the control export shipped
    # without `messages` and did exactly that.
    need = ("run_id", "repo", "sha", "messages", "target_basename", "target_path") + tuple(CUTS)
    bad = [(i, k) for i, c in enumerate(cases) for k in need if k not in c]
    if bad:
        miss = sorted({k for _, k in bad})
        print(f"REFUSING: {len(bad)} missing fields across {len({i for i, _ in bad})} cases; "
              f"absent keys: {miss}", flush=True)
        return 2
    print(f"preflight ok: every case carries {', '.join(need)}", flush=True)

    from accelerate import infer_auto_device_map, init_empty_weights
    cfg = transformers.AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    with init_empty_weights():
        skel = transformers.AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    dmap = infer_auto_device_map(
        skel, max_memory={0: "32GiB", "cpu": "76GiB"}, dtype=torch.bfloat16,
        no_split_module_classes=getattr(skel, "_no_split_modules", None) or [])
    # embed_tokens and the final norm are read outside dispatched forwards elsewhere in this
    # codebase; pinning them costs nothing and removes a class of meta-tensor failure.
    for k in list(dmap):
        if k in ("lm_head", "model.norm", "model.embed_tokens"):
            dmap[k] = 0
    del skel
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=dmap, trust_remote_code=True)
    hf.eval()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    # LEFT TRUNCATION. The claim is at the END of the transcript; the default keeps the first
    # max_seq_len tokens and would hand every probe the middle of a system prompt.
    tok.truncation_side = "left"
    print("model ready", flush=True)

    store: dict = {}
    meta = []
    n = 0
    for i, c in enumerate(cases):
        for cut in CUTS:
            convo = "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in c["messages"])
            prompt = (convo + "<|assistant|>\n" + c[cut])[-MAX_PROMPT_CHARS:]
            enc = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
            ids = enc["input_ids"].to(hf.device)
            try:
                with torch.no_grad():
                    out = hf(ids, output_hidden_states=True, use_cache=False)
            except Exception as e:
                print(f"  case {i} {cut}: FAILED {type(e).__name__}: {e}", flush=True)
                continue
            # hidden_states is (embeddings, block_1, ..., block_L); take the final position of each
            hs = out.hidden_states
            arr = np.stack([h[0, -1, :].float().cpu().numpy().astype(np.float16) for h in hs])
            key = f"{i}|{cut}"
            store[key] = arr
            meta.append({"key": key, "case": i, "cut": cut, "run_id": c["run_id"],
                         "target": c["target_basename"], "target_path": c["target_path"],
                         "repo": c["repo"], "sha": c["sha"],
                         "is_control": bool(c.get("is_control")),
                         "matched_case": c.get("matched_case"),
                         "n_layers": int(arr.shape[0]), "d_model": int(arr.shape[1]),
                         "tokens": int(ids.shape[1])})
            n += 1
            del out, hs
            if n % SAVE_EVERY == 0:
                np.savez_compressed(OUT, **store)
                with open(META_OUT, "w", encoding="utf-8") as fh:
                    json.dump(meta, fh)
                os.sync() if hasattr(os, "sync") else None
            print(f"  [{n:>3}] case {i:>2} {cut:18} layers={arr.shape[0]} tok={ids.shape[1]}",
                  flush=True)

    np.savez_compressed(OUT, **store)
    with open(META_OUT, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    print(f"\nwrote {OUT}: {n} readouts, {os.path.getsize(OUT):,} bytes", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
