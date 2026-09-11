"""Export matched NON-relocation cut points, so "about to relocate" has something to be measured against.

WHY THIS EXISTS. Every readout taken so far sits at a relocation: the moment before the model names
a file it was not sent to examine. That supports questions of the form "is the TARGET decodable",
but it cannot support the question the backtracking literature actually asks -- is the EVENT
decodable? Does the residual, at some sentence boundary in a reasoning trace, carry "I am about to
abandon the file I was asked about"?

Answering that needs negatives, and the negatives have to be matched, because the obvious confounds
all point the same way:

  POSITION. A relocation usually arrives late in a long trace. Sampling controls uniformly would
  make the classifier a context-length detector. Controls here are chosen to minimise the gap in
  prefix length against the relocation they are matched to.

  TRAJECTORY. A control drawn from a different run differs in repository, task, and prompt era.
  Controls come from the SAME run as the relocation they match.

  CONTAMINATION. A boundary two sentences before a relocation is arguably already "about to
  relocate". Controls are taken only from steps that contain no relocation at all, so a positive
  label never leaks into a negative.

The matching is deliberately conservative: it costs sample size (a run with no clean step
contributes nothing) and it removes the three explanations a reviewer would reach for first.

Output: experiments/control_cases.json, same shape as reloc_cases.json so the extractor reads both
with one code path, plus "is_control": true and the matching diagnostics.

Run: python experiments/export_control_cuts.py
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
RELOC = ROOT / "experiments" / "reloc_cases.json"
OUT = ROOT / "experiments" / "control_cases.json"
TARGET_MODEL = "qwen3.6-35b"
MIN_PREFIX = 200        # a boundary this early carries no reasoning to read
MAX_CASES = 75          # match the relocation export


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


def boundaries(text: str) -> list[int]:
    """Every sentence end, as a cut index, on EXACTLY the relocation export's convention.

    That export computes `b = max(head.rfind(". "), head.rfind("\\n"), ...)` and cuts `head[:b+1]`,
    so the prefix ends on the punctuation mark itself and never on the trailing space.

    Matching `m.end()` here instead kept the space, and the one-character difference was enough to
    make the two classes linearly separable at the EMBEDDING layer: relocation prefixes ended on
    `.` 45 times out of 75 and control prefixes never did, ending on a space 28 times instead. The
    first event probe scored AUC 0.894 at layer 0 -- before any transformer block had run -- which
    is the signature of a delimiter detector, not of a model computing anything.
    """
    out = []
    for m in re.finditer(r"(?:\. |\n|! |\? )", text):
        i = m.end() - 1 if m.group().endswith(" ") else m.end()
        if MIN_PREFIX <= i < len(text) - 40:
            out.append(i)
    return out


def step_text(st: dict) -> str:
    """The same field the miner reads: reasoning first, visible message only as a fallback."""
    t = st.get("reasoning_content") or ""
    if not t and str(st.get("source") or "").lower() == "agent":
        m = st.get("message")
        if isinstance(m, str):
            t = m
    return t if isinstance(t, str) else ""


def main() -> int:
    from ctarp.report.mine_reasoning import mine_trace

    reloc = json.loads(RELOC.read_text(encoding="utf-8"))[:MAX_CASES]
    want: dict[str, list[dict]] = {}
    for i, c in enumerate(reloc):
        p = c["prefix_at_sentence"]
        want.setdefault(c["run_id"], []).append(
            {"case": i, "len": len(p), "last": p[-1:]})

    controls, no_clean_step = [], []
    for f in sorted(glob.glob(str(TRACES / "*.atif.json"))):
        run = pathlib.Path(f).name.replace(".atif.json", "")
        if run not in want:
            continue
        fm = casefile_meta(run)
        if fm.get("enrich_model") != TARGET_MODEL:
            continue

        known = None
        cf = CASES / f"{run}.md"
        if cf.is_file():
            from ctarp.report.extract import extract
            c = extract(cf)
            if c:
                known = {y["path"] for y in c["candidates"]}
        r = mine_trace(f, known)
        reloc_steps = {str(x.get("step")) for x in r["relocations"]}
        # every path this run ever relocated to -- a step naming one is not a clean negative even
        # if the miner did not register a relocation there
        reloc_paths = {x["path"] for x in r["relocations"]}
        reloc_bases = {p.rsplit("/", 1)[-1] for p in reloc_paths}

        d = json.loads(pathlib.Path(f).read_text(encoding="utf-8", errors="replace"))
        for rec in (d if isinstance(d, list) else [d]):
            steps = rec.get("steps") or []

            # candidate negatives: boundaries in steps that carry no relocation at all
            cands = []
            for idx, st in enumerate(steps):
                if str(st.get("step_id")) in reloc_steps:
                    continue
                text = step_text(st)
                if not text:
                    continue
                if any(p in text for p in reloc_paths) or any(b in text for b in reloc_bases):
                    continue
                for b in boundaries(text):
                    cands.append((idx, text, b))
            if not cands:
                no_clean_step.append(run)
                continue

            used = set()
            for tgt in want[run]:
                # STRATIFY ON THE FINAL CHARACTER FIRST, then take the nearest by length.
                #
                # Cutting both exports on one convention was necessary but not sufficient: usable
                # relocations end on `.` 71% of the time and controls only 35%, so a classifier
                # reading the last token alone still scores around 0.68. Requiring the pair to end
                # on the same character makes that shortcut worth exactly chance. Length matching
                # then runs inside the stratum.
                free = [c for c in cands if (c[0], c[2]) not in used]
                same = [c for c in free if c[1][:c[2]][-1:] == tgt["last"]]
                pool = same or free
                best = min(pool, key=lambda c: abs(c[2] - tgt["len"]), default=None)
                if best is None:
                    continue
                idx, text, b = best
                used.add((idx, b))

                msgs = []
                for s in steps[:idx]:
                    src = str(s.get("source") or "").lower()
                    role = {"system": "system", "user": "user",
                            "agent": "assistant", "tool": "tool"}.get(src)
                    if role and isinstance(s.get("message"), str):
                        msgs.append({"role": role, "content": s["message"]})

                controls.append({
                    "run_id": run,
                    "is_control": True,
                    "matched_case": tgt["case"],
                    "repo": fm.get("repo", ""),
                    "sha": fm.get("sha", ""),
                    "step": steps[idx].get("step_id"),
                    # no relocation here; the fields exist so the extractor needs no special case
                    "target_path": "",
                    "target_basename": "",
                    # the conversation up to this step, same construction as the relocation export.
                    # Omitting this silently produced a control whose prompt was the bare reasoning
                    # prefix with no conversation in front of it -- a different forward pass from
                    # the positives it is supposed to be matched against.
                    "messages": msgs,
                    "prefix_at_sentence": text[:b],
                    "prefix_at_path": text[:b],
                    "prefix_len": b,
                    "matched_len": tgt["len"],
                })

    OUT.write_text(json.dumps(controls, indent=1), encoding="utf-8")
    print(f"controls exported: {len(controls)}  -> {OUT.relative_to(ROOT)}")
    print(f"  matched relocations: {len({c['matched_case'] for c in controls})} of {len(reloc)}")
    print(f"  distinct runs:       {len({c['run_id'] for c in controls})}")
    if no_clean_step:
        print(f"  runs with no relocation-free step: {sorted(set(no_clean_step))}")
    # The two shortcuts this export exists to close, reported so a reader can see they are closed.
    same_last = sum(1 for c in controls
                    if c["prefix_at_sentence"][-1:] == reloc[c["matched_case"]]
                    ["prefix_at_sentence"][-1:])
    print(f"  final character matched to its relocation: {same_last} / {len(controls)}")
    if controls:
        gaps = sorted(abs(c["prefix_len"] - c["matched_len"]) for c in controls)
        mid = gaps[len(gaps) // 2]
        print(f"\n  prefix-length gap to the matched relocation: "
              f"median {mid}, worst {gaps[-1]} characters")
        print("  (a large gap would mean the probe could separate the classes on length alone)")
        c = controls[0]
        print(f"\n  example: {c['run_id']} step {c['step']}")
        print(f"    ends: ...{c['prefix_at_sentence'][-70:]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
