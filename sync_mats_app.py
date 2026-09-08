"""Refresh ~/Desktop/mats-app from this repo. Run it after ANY rebuild.

WHY THIS EXISTS. mats-app is a snapshot, and a snapshot with no refresh path silently goes stale:
the annotated abstract was corrected here, the site was rebuilt, and the copy in mats-app kept
showing the old two-paragraph version because nothing re-copied it. That is not a mistake you
notice by looking at the repo -- it is only visible in the folder you actually hand over.

Nothing here builds. Build first, then sync:

    python site/build_site.py                       # site/ + figures + pdf copy
    npx mystmd build --pdf   (in report/)           # the pdf itself, needs the typst CLI

Run: python sync_mats_app.py [--dest PATH] [--check]
  --check  report what would change and exit non-zero if anything is stale (no writes)
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent
DEST = pathlib.Path.home() / "Desktop" / "mats-app"
FIGS = ("control.png", "emergence.png", "transport.png", "shared.png", "baseline.png")

# (source, destination-relative) -- every file the handover folder carries.
PAIRS: list[tuple[pathlib.Path, str]] = [
    (ROOT / "report" / "ctarp-faithfulness.pdf", "report/ctarp-faithfulness.pdf"),
    (ROOT / "report" / "nanda.md", "report/nanda.md"),
    # the Google-Docs-ready rendering, generated from nanda.md so it cannot drift
    (ROOT / "make_submission_md.py", "make_submission_md.py"),
    (ROOT / "notebooks" / "jlens_reloc_replication.ipynb",
     "notebooks/jlens_reloc_replication.ipynb"),
    (ROOT / "notebooks" / "make_reloc_notebook.py", "notebooks/make_reloc_notebook.py"),
    (ROOT / "experiments" / "jlens_reloc_probe.py", "code/jlens_reloc_probe.py"),
    (ROOT / "experiments" / "jlens_layer_curve.py", "code/jlens_layer_curve.py"),
    (ROOT / "experiments" / "jlens_reloc_stats.py", "code/jlens_reloc_stats.py"),
    (ROOT / "experiments" / "export_reloc_cases.py", "code/export_reloc_cases.py"),
    (ROOT / "experiments" / "colab_prefetch.py", "code/colab/colab_prefetch.py"),
    (ROOT / "experiments" / "colab_prefetch_status.py", "code/colab/colab_prefetch_status.py"),
    (ROOT / "experiments" / "colab_run_probe.py", "code/colab/colab_run_probe.py"),
    (ROOT / "experiments" / "colab_restart_kernel.py", "code/colab/colab_restart_kernel.py"),
    (ROOT / "experiments" / "reloc_cases.json", "data/reloc_cases.json"),
    # The superseded 30-case export, kept for provenance: it is the set every number published
    # before 2026-09-07 was computed on, and it was cut from the wrong field (`message` rather
    # than `reasoning_content`). Downstream analysis needs to be able to tell the two apart.
    (ROOT / "experiments" / "reloc_cases_v1_30.json", "data/reloc_cases_v1_30.json"),
    (ROOT / "experiments" / "traced50" / "reloc_results.json", "data/reloc_results.json"),
    (ROOT / "experiments" / "traced50" / "layer_curve.json", "data/layer_curve.json"),
    (ROOT / "experiments" / "traced50" / "jlens_reloc_final.log", "data/logs/run1_counts_only.log"),
    (ROOT / "experiments" / "traced50" / "jlens_reloc_run2.log", "data/logs/run2_per_layer.log"),
    # The 75-case run left no log of its own: the provider pruned the session at case 72, before
    # the probe wrote anything. What survives is the per-case counts scraped back out of the
    # CLI's execution history, plus the script that did the scraping.
    (ROOT / "experiments" / "traced50" / "reloc6_recovered_counts.json",
     "data/reloc6_recovered_counts.json"),
    (ROOT / "experiments" / "recover_reloc6.py", "code/recover_reloc6.py"),
    # The successful 75-case run: rows appended and pulled incrementally, so the prune at row 145
    # of 150 cost only the tail. Full per-layer ids, so it can be re-scored under any rule.
    (ROOT / "experiments" / "traced50" / "reloc_rows_75.jsonl", "data/reloc_rows_75.jsonl"),
    (ROOT / "experiments" / "score_reloc8.py", "code/score_reloc8.py"),
    (ROOT / "verify" / "reloc_rows_75.jsonl", "verify/reloc_rows_75.jsonl"),
    (ROOT / "verify" / "reloc_cases_75.json", "verify/reloc_cases_75.json"),
    (ROOT / "verify" / "ask_baseline_75.json", "verify/ask_baseline_75.json"),
    # the prompt baseline the lens has to beat, on both case sets
    (ROOT / "experiments" / "ask_baseline.py", "code/ask_baseline.py"),
    (ROOT / "experiments" / "traced50" / "ask_baseline.json", "data/ask_baseline_30.json"),
    (ROOT / "experiments" / "traced50" / "ask_baseline_75.json", "data/ask_baseline_75.json"),
    (ROOT / "experiments" / "traced50" / "ask_baseline_75.log", "data/logs/ask_baseline_75.log"),
    # the independent verification bundle: notebook plus the exact inputs it re-derives from
    (ROOT / "verify" / "verify_findings.ipynb", "verify/verify_findings.ipynb"),
    (ROOT / "verify" / "make_verify_notebook.py", "verify/make_verify_notebook.py"),
    (ROOT / "verify" / "gold_pilot.json", "verify/gold_pilot.json"),
    (ROOT / "site" / "index.html", "site/index.html"),
    (ROOT / "site" / "deck.html", "site/deck.html"),
    (ROOT / "site" / "evidence.html", "site/evidence.html"),
    (ROOT / "site" / "ctarp-faithfulness.pdf", "site/ctarp-faithfulness.pdf"),
    (ROOT / "site" / "relocation.mp4", "site/relocation.mp4"),
    (ROOT / "demo" / "cards.html", "demo/cards.html"),
    (ROOT / "demo" / "build_cut.sh", "demo/build_cut.sh"),
    (ROOT / "demo" / "relocation.mp4", "demo/relocation.mp4"),
]
for f in FIGS:
    PAIRS.append((ROOT / "report" / "images" / "diagrams" / f, f"report/figures/{f}"))
    PAIRS.append((ROOT / "site" / f, f"site/{f}"))


# Files a previous sync placed that are no longer part of the bundle. The sync only ever copies,
# so a renamed artifact leaves its predecessor sitting beside it -- and reloc8_rows.jsonl (145
# rows, the pruned run) next to reloc_rows_75.jsonl (149 rows, complete) is exactly the pair a
# reader could score the wrong one of. Named explicitly rather than inferred, so nothing outside
# this list is ever deleted.
SUPERSEDED = (
    "data/reloc8_rows.jsonl",
    "verify/reloc8_rows.jsonl",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(DEST))
    ap.add_argument("--check", action="store_true", help="report staleness, write nothing")
    a = ap.parse_args()
    dest = pathlib.Path(a.dest)

    stale, missing, copied = [], [], 0
    for src, rel in PAIRS:
        dst = dest / rel
        if not src.is_file():
            missing.append(str(src.relative_to(ROOT)))
            continue
        same = dst.is_file() and dst.read_bytes() == src.read_bytes()
        if same:
            continue
        stale.append(rel)
        if not a.check:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1

    # The traces directory is the trajectory viewer's data; copied wholesale because the viewer
    # is cache-busted against it and a partial copy renders an empty pane.
    tsrc, tdst = ROOT / "site" / "traces", dest / "site" / "traces"
    if tsrc.is_dir() and not a.check:
        tdst.mkdir(parents=True, exist_ok=True)
        for f in tsrc.iterdir():
            if f.is_file() and (not (tdst / f.name).is_file()
                                or (tdst / f.name).read_bytes() != f.read_bytes()):
                shutil.copy2(f, tdst / f.name)

    # THE SOURCE TRAJECTORIES BEHIND EVERY LENS CASE. Without these the bundle carries the
    # measurement and not the thing measured: reloc_cases.json holds a cut-down prefix, and any
    # later question -- was the claim superseded, what did the model do after, which tool call
    # preceded it -- has to go back to the full trace and its casefile. Driven off the case file
    # so it stays in step with whatever export produced it.
    cj = ROOT / "experiments" / "reloc_cases.json"
    if cj.is_file():
        import json as _json
        runs = sorted({c["run_id"] for c in _json.loads(cj.read_text(encoding="utf-8"))})
        copied_tr = 0
        for run in runs:
            for src, rel in ((ROOT / "data" / "traces" / f"{run}.atif.json",
                              f"data/trajectories/{run}.atif.json"),
                             (ROOT / "data" / "traces" / f"{run}.trace.json",
                              f"data/trajectories/{run}.trace.json"),
                             (ROOT / "data" / "casefiles" / f"{run}.md",
                              f"data/trajectories/{run}.md")):
                if not src.is_file():
                    continue
                d = dest / rel
                if d.is_file() and d.read_bytes() == src.read_bytes():
                    continue
                stale.append(rel)
                if not a.check:
                    d.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, d)
                    copied_tr += 1
        if copied_tr:
            print(f"  trajectories           {copied_tr} file(s) for {len(runs)} run(s)")
            copied += copied_tr

    for rel in SUPERSEDED:
        f = dest / rel
        if f.is_file():
            if a.check:
                stale.append(f"{rel} (superseded, would be removed)")
            else:
                f.unlink()
                print(f"  removed superseded    {rel}")

    if missing:
        print("MISSING (build these first):")
        for m in missing:
            print("  ", m)
    if a.check:
        print(f"stale: {len(stale)}")
        for s in stale:
            print("  ", s)
        return 1 if (stale or missing) else 0
    print(f"synced {copied} file(s) to {dest}" if copied else f"{dest} already current")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
