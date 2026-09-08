"""Export the 35B's relocations as self-contained probe cases for the Jacobian lens.

THE QUESTION. A relocation is the model writing "the vulnerability is actually in X" about a file
it was not sent to examine. The lens reads out what an activation is DISPOSED to make the model
say. So: at the token immediately BEFORE the model writes X, is X already there?

If it is, the sentence is a readout of a conclusion the forward pass had already reached, not the
reasoning that produced it. That is a chain-of-thought faithfulness claim with an answer key --
the fix commit says whether X was actually right.

WHY THE 35B AND NOT THE 4B WE HAVE LOCALLY. Measured over 718 traces:

  qwen3.6-35b        66 traces, 27 with relocations, 73 relocations   1.11 per trace
  fastcontext (old) 432 traces, 55 with relocations, 129 relocations  0.30 per trace
  qwen3.8-flash      60 traces,  7 with relocations,  14 relocations  0.23 per trace
  qwen3.5-4b         28 traces,  0 with relocations,   0              0

qwen3.5-4b has a published lens and does not relocate -- it narrates a procedure (205 "let me"
against 8 "actually" over 35k characters). The 35B relocates four times as often as anything else
AND camilablank/workspace-lenses publishes a lens for it. It is the only model that satisfies both
halves.

WHAT THIS EMITS. One case per relocation: the conversation prefix up to the moment of the claim,
truncated immediately before the path is written, plus the target path, the file the model was
actually sent to examine, and whether the claim ever reached the casefile. Everything the probe
needs, so the Colab session loads one JSON and never touches this repo.

Run: python experiments/export_reloc_cases.py
"""
from __future__ import annotations

import glob
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TRACES = ROOT / "data" / "traces"
CASES = ROOT / "data" / "casefiles"
OUT = ROOT / "experiments" / "reloc_cases.json"
TARGET_MODEL = "qwen3.6-35b"


def casefile_meta(run: str) -> dict:
    p = CASES / f"{run}.md"
    if not p.is_file():
        return {}
    t = p.read_text(encoding="utf-8", errors="replace")
    fm = {}
    m = re.match(r"---\n(.*?)\n---\n", t, re.S)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                fm[k.strip()] = v.strip()
    return fm


def main() -> int:
    from ctarp.report.mine_reasoning import mine_trace

    cases = []
    for f in sorted(glob.glob(str(TRACES / "*.atif.json"))):
        run = pathlib.Path(f).name.replace(".atif.json", "")
        fm = casefile_meta(run)
        if fm.get("enrich_model") != TARGET_MODEL:
            continue
        # PASS THE CASEFILE'S PATHS. Without them _in_casefile has nothing to match
        # against and every claim reports reached_casefile=False -- which inverted the
        # corpus figure (97.4% of relocations DO reach the output) and would have been
        # read as 30 lost claims.
        known = None
        cf = CASES / f'{run}.md'
        if cf.is_file():
            from ctarp.report.extract import extract
            c = extract(cf)
            if c:
                known = {y['path'] for y in c['candidates']}
        r = mine_trace(f, known)
        if not r["relocations"]:
            continue

        d = json.loads(pathlib.Path(f).read_text(encoding="utf-8", errors="replace"))
        for rec in (d if isinstance(d, list) else [d]):
            steps = rec.get("steps") or []
            for x in r["relocations"]:
                # Locate the step this claim came from. step_id is a string in the trace.
                idx = next((i for i, s in enumerate(steps)
                            if str(s.get("step_id")) == str(x.get("step"))), None)
                if idx is None:
                    continue
                # THE SAME FIELD THE MINER READ. mine_reasoning takes reasoning_content first and
                # only falls back to the agent message; the 35B runs on llama.cpp with --jinja,
                # which populates reasoning_content, so that is where its relocations live. This
                # used to read `message` alone and then look for the claimed path in it -- absent
                # 56 times out of 86, which silently discarded 65% of the corpus as "no prefix
                # found" rather than as a field mismatch. Selecting text the same way the claim
                # was found is the only way the two can agree.
                st = steps[idx]
                text = st.get("reasoning_content") or ""
                field = "reasoning_content"
                if not text and str(st.get("source") or "").lower() == "agent":
                    m = st.get("message")
                    if isinstance(m, str):
                        text, field = m, "message"
                if not isinstance(text, str) or not text:
                    continue

                # TRUNCATE IMMEDIATELY BEFORE THE PATH IS WRITTEN. Probing after the model has
                # already written X would answer nothing -- of course X is present once it is on
                # the page. The claim under test is that X is decodable BEFORE it is stated.
                base = x["path"].rsplit("/", 1)[-1]
                cut = text.find(x["path"])
                if cut < 0:
                    cut = text.find(base)
                if cut <= 0:
                    continue

                # TWO CUTS, BECAUSE THEY TEST DIFFERENT CLAIMS.
                #
                # at_path stops immediately before the path string. The first version cut before
                # the BASENAME, which left "The vulnerability is present in `src/" on the page --
                # asking whether "logging" follows "src/" is next-token prediction, not evidence
                # about what the forward pass had settled on.
                #
                # at_sentence stops before the whole clause that carries the claim. That is the
                # real question: before the model began articulating the relocation at all, was
                # the target already what its activations were disposed to say?
                prefix_path = text[:cut]
                head = text[:cut]
                b = max(head.rfind(". "), head.rfind("\n"),
                        head.rfind("! "), head.rfind("? "))
                prefix_sentence = head[:b + 1] if b > 0 else head

                # The conversation up to this step, verbatim, so the probe replays the same
                # forward pass the trace recorded rather than an approximation of it.
                msgs = []
                for s in steps[:idx]:
                    src = str(s.get("source") or "").lower()
                    role = {"system": "system", "user": "user",
                            "agent": "assistant", "tool": "tool"}.get(src)
                    if role and isinstance(s.get("message"), str):
                        msgs.append({"role": role, "content": s["message"]})

                cases.append({
                    "run_id": run,
                    # which field the claim was read from, so a later analysis can tell a
                    # reasoning-trace claim from one written in the visible message
                    "source_field": field,
                    "repo": fm.get("repo", ""),
                    "sha": fm.get("sha", ""),
                    "step": x.get("step"),
                    "target_path": x["path"],
                    "target_basename": base,
                    "reached_casefile": bool(x.get("reached_casefile")),
                    "acted_on": bool(x.get("acted_on")),
                    "quote": x.get("quote", ""),
                    "messages": msgs,
                    "prefix_at_path": prefix_path,
                    "prefix_at_sentence": prefix_sentence,
                })

    OUT.write_text(json.dumps(cases, indent=1), encoding="utf-8")
    reached = sum(1 for c in cases if c["reached_casefile"])
    print(f"cases exported: {len(cases)}  -> {OUT.relative_to(ROOT)}")
    print(f"  reached the casefile:     {reached}")
    print(f"  never reached the output: {len(cases) - reached}   <- the interesting subset")
    print(f"  distinct runs:            {len({c['run_id'] for c in cases})}")
    if cases:
        c = cases[0]
        print(f"\n  example: {c['run_id']} step {c['step']}")
        print(f"    target: {c['target_path']}")
        print(f"    at_path ends:     ...{c['prefix_at_path'][-70:]!r}")
        print(f"    at_sentence ends: ...{c['prefix_at_sentence'][-70:]!r}")
        print(f"    prior messages: {len(c['messages'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
