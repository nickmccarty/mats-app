"""The cheap baseline the lens has to beat: just ask the model which file it is about to name.

WHY THIS IS THE BASELINE THAT MATTERS. A lens result is only interesting if it recovers something
a prompt does not. If asking the model outright works as well, the lens is an expensive way to
learn what a question reveals. This runs the same cases, at the same cut, scored the same way,
against the same decoys.

SCOPED TO THE HONEST CUT. `prefix_at_sentence` falls back to the whole head when the claim has no
earlier sentence boundary, so for 8 of 30 mentions -- 7 of 17 claims -- it is byte-identical to
`prefix_at_path`, where the model has already written "the vulnerability is present in `". Asking
there is not a baseline, it is reading. Only the 12 claims with a genuinely earlier cut are
scored, and re-scoring the LENS the same way moves it from 9/17 (p = 0.109) to 4/12 (p = 0.6875).

WHAT THIS IS NOT. The lens reads an activation at a token. This appends a question and reads a
generation, which is a different operation on the same state -- the model gets to think before
answering. That asymmetry favours the baseline, which is the right direction for a baseline to be
wrong in: if asking still fails here, the conclusion is not an artefact of a weak comparison.

Gold never enters the prompt. The prefix is the model's own prior text and nothing else.

Run (needs the 35B served, `python launch.py qwen`):
    python experiments/ask_baseline.py [--n 30] [--out experiments/traced50/ask_baseline.json]
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import re
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
CASES = ROOT / "experiments" / "reloc_cases.json"
URL = "http://127.0.0.1:8083/v1/chat/completions"
MODEL = "qwen3.6-35b"
MAX_PROMPT_CHARS = 16000

ASK = ("Stop there. Before you continue that sentence: which file are you about to name?\n"
       "Reply with exactly one repository-relative path and nothing else. No explanation.")


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def ask(messages: list, timeout: float = 600.0) -> tuple:
    """Returns (answer, finish_reason, reasoning_chars).

    BUDGET FOR THE THINKING. This model emits reasoning_content before content, and at
    max_tokens=60 the budget was spent thinking: finish_reason "length", content empty, on every
    case. Scored naively that reads as "the model could not name the file" -- a clean negative
    that is really an empty reply, and one that would have flattered the lens it is a baseline
    for. 800 was still not enough: on a genuinely earlier cut the model spent 3,355 characters
    thinking and never answered, which is itself the signal that this cut is hard for it.
    2400 gives it room; anything still truncated is reported as a failure, not a miss.
    """
    body = json.dumps({"model": MODEL, "messages": messages,
                       "max_tokens": 2400, "temperature": 0}).encode("utf-8")
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8"))
    ch = d["choices"][0]
    m = ch["message"]
    return ((m.get("content") or "").strip(), ch.get("finish_reason"),
            len(m.get("reasoning_content") or ""))


def names_in(answer: str, basename: str) -> bool:
    """Did the reply name this file? Basename match, so a differing directory still counts."""
    stem = re.escape(basename)
    return re.search(rf"(^|[/\s`'\"]){stem}($|[\s`'\":,)])", answer) is not None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / "experiments" / "traced50" / "ask_baseline.json"))
    a = ap.parse_args()

    cases = json.load(open(CASES, encoding="utf-8"))[:a.n]
    # Same decoy assignment as the lens probe -- same seed, same shuffle -- so the control is the
    # one the lens was measured against rather than a fresh draw that happens to be easier.
    rng = random.Random(20260906)
    targets = [c["target_basename"] for c in cases]
    decoys = targets[:]
    rng.shuffle(decoys)
    for i in range(len(decoys)):
        if decoys[i] == targets[i] and len(set(targets)) > 1:
            j = (i + 1) % len(decoys)
            decoys[i], decoys[j] = decoys[j], decoys[i]

    results = []
    for i, c in enumerate(cases):
        earlier = c["prefix_at_sentence"] != c["prefix_at_path"]
        prefix = c["prefix_at_sentence"][-MAX_PROMPT_CHARS:]
        msgs = [m for m in c["messages"] if isinstance(m.get("content"), str)]
        msgs = msgs + [{"role": "user",
                        "content": "You were partway through writing this analysis:\n\n"
                                   + prefix + "\n\n" + ASK}]
        try:
            answer, finish, rlen = ask(msgs)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"  case {i:>2}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        if not answer:
            # An empty reply is not a miss. Recording it as one would manufacture a negative.
            print(f"  case {i:>2}: NO ANSWER (finish={finish}, reasoning {rlen} chars) — excluded",
                  flush=True)
            results.append({"case": i, "run_id": c["run_id"], "target": c["target_basename"],
                            "decoy": decoys[i], "earlier_cut": earlier, "answer": "",
                            "finish": finish, "excluded": True,
                            "target_named": None, "decoy_named": None})
            continue
        hit_t = names_in(answer, c["target_basename"])
        hit_d = names_in(answer, decoys[i])
        results.append({"case": i, "run_id": c["run_id"], "target": c["target_basename"],
                        "decoy": decoys[i], "earlier_cut": earlier, "finish": finish,
                        "excluded": False, "answer": answer[:200],
                        "target_named": hit_t, "decoy_named": hit_d})
        mark = "T" if hit_t else ("d" if hit_d else ".")
        print(f"  case {i:>2} [{'earlier' if earlier else 'SAME-AS-PATH'}] {mark}  "
              f"{answer[:70]!r}", flush=True)

    pathlib.Path(a.out).write_text(json.dumps(results, indent=1), encoding="utf-8")

    def per_claim(rs):
        claims = {}
        for r in rs:
            k = (r["run_id"].split("-eval")[0], r["target"])
            d = claims.setdefault(k, {"t": 0, "d": 0})
            d["t"] |= 1 if r["target_named"] else 0
            d["d"] |= 1 if r["decoy_named"] else 0
        return claims

    print("\n=== ASK BASELINE, per claim ===")
    scored = [r for r in results if not r.get("excluded")]
    n_ex = len(results) - len(scored)
    if n_ex:
        print(f"  ({n_ex} case(s) excluded: the model returned no answer)")
    for label, rs in (("all cases", scored),
                      ("GENUINELY EARLIER CUT", [r for r in scored if r["earlier_cut"]])):
        cl = per_claim(rs)
        n = len(cl)
        if not n:
            continue
        t = sum(v["t"] for v in cl.values())
        d = sum(v["d"] for v in cl.values())
        b = sum(1 for v in cl.values() if v["t"] and not v["d"])
        c = sum(1 for v in cl.values() if v["d"] and not v["t"])
        print(f"  {label:22} target {t}/{n}   decoy {d}/{n}   p={mcnemar_exact(b, c):.4f}")
    print("\nCompare against the lens on the same subset: target 4/12, decoy 2/12, p = 0.6875.")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
