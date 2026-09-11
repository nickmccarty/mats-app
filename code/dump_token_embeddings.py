"""Save the unembedding rows for every target and decoy filename token.

WHY SEPARATELY FROM THE RESIDUALS. The extraction saves what the model was computing; deciding
WHICH file that computation points at needs the candidate filenames represented in the same space.
That is the unembedding matrix, and only the rows for tokens actually used are needed -- 75 of the
248,320 this model has.

WHY IT DOES NOT LOAD THE MODEL. The weights are already on the VM from the prefetch, so the rows
can be sliced straight out of the safetensors shard that holds `lm_head.weight` (or the tied
`model.embed_tokens.weight`). That takes seconds and no GPU, and it works on a session whose model
is long gone -- unlike reaching into a live process, which only works if this runs inside the
extraction itself.

Output:
    /content/tok_emb.npz         one fp16 row per token id
    /content/tok_emb_names.json  filename -> token ids, and the seeded decoy pairing

Run: colab exec -s <session> -f experiments/dump_token_embeddings.py
"""

MODEL = "Qwen/Qwen3.6-35B-A3B"
CASES_JSON = "/content/reloc_cases.json"
OUT = "/content/tok_emb.npz"

import json
import random
import sys


def toks_of(tok, s):
    """Token ids that spell a filename, bare and space-prefixed -- the same rule the lens probe
    used, so both experiments score the same candidate tokens."""
    out = set()
    for v in (s, " " + s, s.split(".")[0], " " + s.split(".")[0]):
        try:
            out.update(tok(v, add_special_tokens=False)["input_ids"])
        except Exception:
            pass
    return out


def main() -> int:
    import numpy as np
    import transformers
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    cases = json.load(open(CASES_JSON, encoding="utf-8"))[:75]
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)

    # the seeded decoy pairing the lens and the baseline were both scored against
    rng = random.Random(20260906)
    targets = [c["target_basename"] for c in cases]
    decoys = targets[:]
    rng.shuffle(decoys)
    for i in range(len(decoys)):
        if decoys[i] == targets[i] and len(set(targets)) > 1:
            j = (i + 1) % len(decoys)
            decoys[i], decoys[j] = decoys[j], decoys[i]

    names = {}
    for i, c in enumerate(cases):
        names[c["target_basename"]] = sorted(toks_of(tok, c["target_basename"]))
        names[decoys[i]] = sorted(toks_of(tok, decoys[i]))
    need = sorted({t for ids in names.values() for t in ids})
    print(f"{len(names)} filenames -> {len(need)} distinct tokens", flush=True)

    # 75 cases collapse to a small set of distinct basenames (many repos have their own utils.py),
    # so a decoy drawn from the pool can be byte-identical to the target it is paired against. Those
    # cases carry no target-vs-decoy signal in either direction and the scorer must drop them rather
    # than count them. Reported here so the number is visible before any probe runs on it.
    same = [i for i, c in enumerate(cases) if decoys[i] == c["target_basename"]]
    print(f"decoy identical to target in {len(same)} of {len(cases)} cases: {same}", flush=True)

    # Find which shard holds the output embedding. Qwen ties lm_head to embed_tokens in some
    # configs, so try both names rather than assuming.
    idx = json.load(open(hf_hub_download(MODEL, "model.safetensors.index.json"), encoding="utf-8"))
    wmap = idx["weight_map"]
    key = next((k for k in ("lm_head.weight", "model.embed_tokens.weight") if k in wmap), None)
    if key is None:
        print("no lm_head/embed_tokens in the index", flush=True)
        return 2
    shard = hf_hub_download(MODEL, wmap[key])
    print(f"{key} lives in {wmap[key]}", flush=True)

    # framework="pt", not "np": these weights are bfloat16, which numpy has no dtype for, so the
    # np backend raises "data type 'bfloat16' not understood" on the first row. torch reads it and
    # converts. Importing torch here costs nothing extra -- it is already loaded on this VM.
    import torch  # noqa: F401  (imported for the safetensors pt backend)
    with safe_open(shard, framework="pt") as f:
        W = f.get_slice(key)
        vocab, d = W.get_shape()
        print(f"{key}: {vocab} x {d}", flush=True)
        rows = {}
        for t in need:
            rows[str(t)] = W[t:t + 1, :].float().numpy().astype(np.float16)[0]

    np.savez_compressed(OUT, **rows)
    with open("/content/tok_emb_names.json", "w", encoding="utf-8") as fh:
        json.dump({"names": names, "decoys": decoys, "key": key}, fh)
    print(f"wrote {OUT}: {len(rows)} token rows, d_model={d}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
