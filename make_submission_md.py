"""Render report/nanda.md as plain Markdown for pasting into a Google Doc.

THIS IS THE SUBMISSION ARTIFACT. The application form takes a link to a Doc, and Google Docs
converts pasted Markdown on the way in -- headings, bold, lists and tables all survive. What does
not survive is everything that makes the source a MyST document: YAML frontmatter pastes as visible
garbage, `:::{warning}` pastes as literal colons, `@citekey` pastes as `@citekey`, and a relative
image path resolves to nothing.

So this converts rather than copies, and it regenerates from report/nanda.md so the submitted text
cannot disagree with the PDF built from the same file. A hand-maintained second copy drifted within
a day the one time it existed.

WHAT IT CHANGES:
  - frontmatter is dropped, but `parts.abstract` is LIFTED OUT of it first and emitted as the
    abstract. Dropping the block wholesale silently deleted the abstract when it moved into
    frontmatter to satisfy the PDF template.
  - `:::{warning} X` becomes a quoted callout
  - `![caption](path)` becomes an explicit INSERT FIGURE marker naming the file to place, with the
    caption beneath it, because a Doc cannot resolve a relative image path
  - `@citekey` becomes a readable label, with a references appendix built from references.bib
  - the figure and artifact appendices are BUILT FROM THE FOLDER, not hand-listed, so they cannot
    describe files that no longer exist

Run: python make_submission_md.py
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "report" / "nanda.md"
BIB = ROOT / "report" / "references.bib"
DEST = pathlib.Path.home() / "Desktop" / "mats-app"
OUT = DEST / "SUBMISSION.md"

FIGURES: list[tuple[int, str, str]] = []   # (n, filename, caption)
CITED: list[str] = []


def insertable(name: str) -> str:
    """The filename a reader can actually place in a Google Doc.

    The probe figures are authored as SVG so typst embeds them as vectors in the PDF, but Docs
    accepts PNG, JPEG and GIF only -- an SVG cannot be inserted at all. A 2x raster is generated
    alongside each one, so the marker names that instead. If the raster is missing the marker keeps
    the SVG name rather than inventing a file, and the mismatch is visible.
    """
    if not name.lower().endswith(".svg"):
        return name
    png = name[:-4] + ".png"
    return png if (ROOT / "report" / "images" / "diagrams" / png).is_file() else name


def load_bib() -> dict:
    if not BIB.is_file():
        return {}
    out = {}
    for e in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", BIB.read_text(encoding="utf-8"), re.S):
        key, fields = e.group(1).strip(), e.group(2)
        au = re.search(r"author\s*=\s*\{(.+?)\}(?=,\s*\n)", fields, re.S)
        yr = re.search(r"year\s*=\s*\{(\d{4})\}", fields)
        ti = re.search(r"title\s*=\s*\{(.+?)\}(?=,\s*\n)", fields, re.S)
        jo = re.search(r"(?:journal|booktitle|institution|howpublished)\s*=\s*\{(.+?)\}(?=,?\s*\n)",
                       fields, re.S)
        name = "?"
        if au:
            parts = re.split(r"\s+and\s+", " ".join(au.group(1).split()))
            name = parts[0].split(",")[0].strip("{} ")
            if len(parts) > 1:
                name += " et al."
        out[key] = {
            "label": f"{name} ({yr.group(1)})" if yr else name,
            "title": " ".join(ti.group(1).split()) if ti else "",
            "where": " ".join(jo.group(1).split()) if jo else "",
            "year": yr.group(1) if yr else "",
        }
    return out


BIBD = load_bib()


def frontmatter(text: str) -> tuple[dict, str]:
    m = re.match(r"\A---\n(.*?)\n---\n", text, re.S)
    if not m:
        return {}, text
    raw, body = m.group(1), text[m.end():]
    meta = {}
    for key in ("title", "subtitle"):
        km = re.search(rf'^{key}:\s*"?(.+?)"?\s*$', raw, re.M)
        if km:
            meta[key] = km.group(1)
    am = re.search(r"^parts:\n\s+abstract:\s*\|\n(.*?)(?=\n[a-z_]+:|\Z)", raw, re.S | re.M)
    if am:
        meta["abstract"] = "\n".join(
            re.sub(r"^ {4}", "", ln) for ln in am.group(1).split("\n")).strip()
    return meta, body


def cite(m: re.Match) -> str:
    key = m.group(1)
    if key not in BIBD:
        return m.group(0)
    if key not in CITED:
        CITED.append(key)
    return f"[{BIBD[key]['label']}]"


def convert(text: str) -> str:
    text = re.sub(r"(?<![\w`])@([A-Za-z][\w:+-]*\d{4}\w*)", cite, text)

    out: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        # :::{directive} ... :::
        m = re.match(r"^:::\{(\w+)\}\s*(.*)$", line)
        if m:
            kind, title = m.group(1), m.group(2).strip()
            body: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith(":::"):
                body.append(lines[i])
                i += 1
            i += 1

            # `:::{figure} path` still carries the five original PNG figures; the three probe
            # figures use plain `![](...)`. Both have to produce a marker, or a figure silently
            # becomes a blockquote of its own caption.
            if kind == "figure":
                name = insertable(pathlib.Path(title).name)
                cap = " ".join(
                    b.strip() for b in body
                    if b.strip() and not b.strip().startswith((":label:", ":width:", ":alt:",
                                                               ":align:", ":name:")))
                n = len(FIGURES) + 1
                FIGURES.append((n, name, cap))
                out += ["", f"> **[ INSERT FIGURE {n} — `report/figures/{name}` ]**", "",
                        f"*Figure {n}. {cap}*", ""]
                continue

            out.append("")
            if title:
                out += [f"> **{title}**", ">"]
            for b in body:
                out.append(f"> {b}".rstrip())
            out.append("")
            continue

        # ![caption](path) -- a Doc cannot resolve the path, so name the file explicitly
        im = re.match(r"^!\[(.*?)\]\((.+?)\)\s*$", line)
        if im:
            alt, path = im.group(1).strip(), im.group(2).strip()
            name = insertable(pathlib.Path(path).name)
            # the source puts the full caption in the italic paragraph after the image
            cap_lines, j = [], i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].lstrip().startswith("*") and \
                    not lines[j].lstrip().startswith("**"):
                while j < len(lines) and lines[j].strip():
                    cap_lines.append(lines[j].strip())
                    j += 1
                i = j
            caption = " ".join(cap_lines).strip("* ").strip() or alt
            n = len(FIGURES) + 1
            FIGURES.append((n, name, alt or caption))
            out += ["", f"> **[ INSERT FIGURE {n} — `report/figures/{name}` ]**", "",
                    f"*Figure {n}. {caption}*", ""]
            i += 1
            continue

        out.append(line)
        i += 1
    return "\n".join(out)


def unwrap(text: str) -> str:
    """Reflow each paragraph onto a single line.

    The source is hard-wrapped at ~95 characters, which is right for a file under version control
    and wrong for this one. Markdown treats a single newline inside a paragraph as a space, but
    Google Docs' paste converter does not reliably: it breaks at the wrap, and around inline
    emphasis sitting near one it breaks on BOTH sides, so `*relocations*` lands alone on its own
    line and the sentence reads as three fragments.

    Joining each paragraph into one long line changes no Markdown semantics and removes the
    ambiguity entirely. Structural lines keep their own breaks, because joining them would destroy
    them: table rows, headings, fences, list items, and blockquote lines are all line-oriented.
    """
    out: list[str] = []
    buf: list[str] = []
    in_fence = False

    def flush():
        if buf:
            out.append(" ".join(x.strip() for x in buf))
            buf.clear()

    for line in text.splitlines():
        s = line.strip()
        if s.startswith("```"):
            flush(); in_fence = not in_fence; out.append(line); continue
        if in_fence:
            out.append(line); continue

        # Blank, table row, heading, blockquote and rule are line-oriented: emit as they are.
        if not s or s.startswith(("|", "#", ">", "---", "***")):
            flush(); out.append(line); continue

        # A LIST ITEM STARTS A BLOCK; IT DOES NOT END ONE. The first version treated the marker
        # line like a heading and flushed after it, which left each item's continuation as a
        # separate unindented paragraph -- and an unindented paragraph after a bullet terminates
        # the list. Buffering the marker line means the continuation lines join onto it.
        if re.match(r"^([-*+]|\d+\.)\s", s):
            flush(); buf.append(s); continue

        buf.append(line)
    flush()
    return "\n".join(out)


def artifact_rows() -> str:
    """Describe what is actually in the bundle. A hand-written table here listed a slide deck and a
    video for a day after both were deleted."""
    known = [
        ("report/ctarp-faithfulness.pdf", "this document, typeset"),
        ("index.html", "the same write-up as a web page, openable from `file://`"),
        ("verify/verify_findings.ipynb",
         "**re-derives every number below from the raw files.** Colab, no GPU, no model"),
        ("verify/probe_q1_q2.ipynb",
         "the four follow-up probes, each against a measured permutation null"),
        ("verify/resid.npz", "the residual stream: 150 readouts, 41 layers, fp16"),
        ("verify/resid_ctrl.npz", "the matched non-relocation controls, 74 readouts"),
        ("data/reloc_rows_75.jsonl",
         "the lens readout: top-20 token ids at every layer, both lenses"),
        ("data/ask_baseline_75.json", "the prompt baseline: what the model answered, per case"),
        ("data/reloc_cases.json", "the 75 cases — conversation prefix, both cut points, target"),
        ("data/trajectories/", "full ATIF trace, tool trace and casefile for every source run"),
        ("site/traces/trajectory.html", "all 32 trajectories, searchable, openable from `file://`"),
        ("code/", "the probes, the analyses, and the Colab session helpers"),
    ]
    rows = []
    for path, desc in known:
        p = DEST / path
        if p.exists():
            rows.append(f"| `{path}` | {desc} |")
    return "\n".join(rows)


def main() -> int:
    if not SRC.is_file():
        print(f"missing {SRC}", file=sys.stderr)
        return 1
    meta, body = frontmatter(SRC.read_text(encoding="utf-8"))

    parts = [
        f"# {meta.get('title', 'Write-up')}", "",
        "**Nicholas McCarty** · Upskilled Consulting  ",
        "Working draft — please do not redistribute, post, or cite.", "",
    ]
    if meta.get("subtitle"):
        parts += [f"*{meta['subtitle']}*", ""]
    parts += ["---", ""]
    if meta.get("abstract"):
        parts += ["## Abstract", "", convert(meta["abstract"]), ""]
    parts += [convert(body)]

    if FIGURES:
        rows = "\n".join(f"| {n} | `{f}` | {c} |" for n, f, c in FIGURES)
        parts += ["", "---", "", "# Appendix A — Figures to insert", "",
                  f"All {len(FIGURES)} are in `mats-app/report/figures/`. The three probe figures "
                  "are SVG and stay sharp at any size; the rest are 2x PNG. None of them duplicates "
                  "information that is also in the prose.", "",
                  "| # | file | what it shows |", "|---|---|---|", rows, ""]

    parts += ["", "# Appendix B — What is in the handover folder", "",
              "`mats-app/` accompanies this document and is also at "
              "<https://github.com/nickmccarty/mats-app>.", "",
              "| path | what it is |", "|---|---|", artifact_rows(), ""]

    if CITED:
        refs = "\n".join(
            f"{i}. **{BIBD[k]['label']}** {BIBD[k]['title']}."
            + (f" *{BIBD[k]['where']}*." if BIBD[k]["where"] else "")
            for i, k in enumerate(CITED, 1))
        parts += ["", "# Appendix C — References", "", refs, ""]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(unwrap("\n".join(parts)), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size:,} bytes)")
    print(f"  abstract: {'lifted from frontmatter' if meta.get('abstract') else 'MISSING'}")
    print(f"  figures:  {len(FIGURES)} marker(s)")
    print(f"  citations: {len(CITED)} resolved of {len(BIBD)} in the bibliography")
    left = [m for m in re.findall(r"@[A-Za-z][\w:+-]*\d{4}\w*", OUT.read_text(encoding='utf-8'))]
    if left:
        print(f"  WARNING: {len(left)} unresolved citation key(s): {sorted(set(left))[:4]}")
    if ":::" in OUT.read_text(encoding="utf-8"):
        print("  WARNING: a ::: directive survived conversion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
