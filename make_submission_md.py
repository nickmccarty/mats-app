"""Render report/nanda.md as plain Markdown for pasting into a Google Doc.

WHY A CONVERTER AND NOT A HAND-WRITTEN COPY. The report is MyST: it carries frontmatter, admonition
blocks and `:::{figure}` directives that Google Docs renders as literal colons, and it is still
being edited. A second hand-maintained copy would drift from it within a day -- which is the exact
failure the handover folder already hit once. This regenerates from the source, so the submission
text cannot disagree with the report it came from.

WHAT IT CHANGES:
  - frontmatter is dropped and replaced by a title block
  - `:::{warning} X` becomes a quoted callout
  - `:::{figure} path` becomes an explicit INSERT FIGURE marker naming the file to place, with the
    caption beneath it, because a Doc cannot resolve a relative image path
  - a figure index and an artifact index are appended, so the reader knows what exists and where

Run: python make_submission_md.py
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "report" / "nanda.md"
OUT = pathlib.Path.home() / "Desktop" / "mats-app" / "SUBMISSION.md"

FIG_CAPTIONS: dict[str, str] = {}


def convert(text: str) -> str:
    # drop YAML frontmatter
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)

    out, i, fig_n = [], 0, 0
    lines = text.splitlines()
    while i < len(lines):
        line = lines[i]

        m = re.match(r"^:::\{(\w+)\}\s*(.*)$", line)
        if m:
            kind, title = m.group(1), m.group(2).strip()
            body: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith(":::"):
                body.append(lines[i])
                i += 1
            i += 1  # closing :::

            if kind == "figure":
                fig_n += 1
                path = title
                name = pathlib.Path(path).name
                cap = [b for b in body
                       if b.strip() and not b.strip().startswith((":label:", ":width:", ":alt:"))]
                caption = " ".join(x.strip() for x in cap).strip()
                FIG_CAPTIONS[name] = caption
                out += ["", f"> **[ INSERT FIGURE {fig_n} — `report/figures/{name}` ]**", ""]
                out += [f"*Figure {fig_n}. {caption}*", ""]
            else:
                out.append("")
                if title:
                    out.append(f"> **{title}**")
                    out.append(">")
                for b in body:
                    out.append(f"> {b}".rstrip())
                out.append("")
            continue

        out.append(line)
        i += 1
    return "\n".join(out)


HEADER = """# Chain-of-Thought Faithfulness With Ground Truth

**Nicholas McCarty** · Upskilled Consulting
Working draft — please do not redistribute, post, or cite.

---

"""

FOOTER_TMPL = """

---

# Appendix A — Figures to insert

All five are in `mats-app/report/figures/`. They are captured at 2x and are readable at full
document width; none of them carry information that is also in the prose, so dropping one loses
something.

| # | file | what it shows |
|---|---|---|
{rows}

# Appendix B — What is in the handover folder

`mats-app/` (164 files) accompanies this document.

| path | what it is |
|---|---|
| `report/ctarp-faithfulness.pdf` | this document, typeset, 10pp |
| `verify/verify_findings.ipynb` | **re-derives every number below from the raw files.** Colab, no GPU, no model, ~2s. Prints computed against claimed with PASS/FAIL; currently 35 of 35 |
| `verify/*.json`, `verify/reloc_rows_75.jsonl` | the exact inputs that notebook reads |
| `data/reloc_rows_75.jsonl` | the lens readout: top-20 token ids at all 39 layers, both lenses, 149 readouts across all 75 cases |
| `data/ask_baseline_75.json` | the prompt baseline: what the model answered, per case |
| `data/reloc_cases.json` | the 75 cases — conversation prefix, both cut points, target path |
| `data/reloc_cases_v1_30.json` | the superseded 30-case export, kept so old figures can be traced |
| `data/trajectories/` | full ATIF trace, tool trace and casefile for all 29 source runs |
| `site/index.html`, `site/deck.html` | the web version and a ten-slide deck, both openable from `file://` |
| `site/relocation.mp4` | 49s, silent: one case end to end, from the question to the fix commit |
| `code/` | the probe, the analyses, and the Colab session helpers |

# Appendix C — How to check the numbers

```
# no GPU, no model, about two seconds
upload verify/verify_findings.ipynb to Colab
upload the six files in verify/ beside it
run all cells
```

Each cell recomputes a figure from the raw files and prints it next to the claimed value. Where a
number was corrected during the work, both the old and the new value are computed, so the
correction is visible as arithmetic rather than asserted in prose.
"""


def main() -> int:
    if not SRC.is_file():
        print(f"missing {SRC}", file=sys.stderr)
        return 1
    body = convert(SRC.read_text(encoding="utf-8"))

    order = ["baseline.png", "emergence.png", "transport.png", "shared.png", "control.png"]
    rows = []
    for n, name in enumerate([f for f in order if f in FIG_CAPTIONS], start=1):
        cap = FIG_CAPTIONS[name]
        short = cap.split(".")[0][:110]
        rows.append(f"| {n} | `{name}` | {short} |")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(HEADER + body + FOOTER_TMPL.format(rows="\n".join(rows)), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size:,} bytes)")
    print(f"figures referenced: {len(FIG_CAPTIONS)}")
    for k in FIG_CAPTIONS:
        print(f"  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
