"""Emit the probe figures as one self-contained D3 page.

WHY THESE AND NOT THE USUAL ONES. The standard interpretability figure -- a layer-by-layer curve of
some metric -- is unreadable here, because at 23-35 units the metric's null is not 0.5 and is not
even constant across layers. Every chart on this page therefore draws the permutation null as a
band behind the measurement, so "is this above chance" is a visual question rather than an act of
faith. That is the one thing a TransformerLens-style plot never shows and the only thing that makes
these numbers interpretable.

Three figures:
  fig-null-band   per-layer AUC against the null band it has to clear, for both probes
  fig-noise       what the best single dimension is worth: real vs shuffled vs held-out
  fig-confound    why the decoy failed -- filenames nested inside repositories

Run: python report/diagrams/make_probe_figs.py
Then screenshot each .fig to report/images/diagrams/.
"""
from __future__ import annotations

import collections
import json
import pathlib
import random

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
T50 = ROOT / "experiments" / "traced50"
OUT = ROOT / "report" / "diagrams" / "probes.html"


def seeded_decoys(cases):
    rng = random.Random(20260906)
    t = [c["target_basename"] for c in cases]
    d = t[:]
    rng.shuffle(d)
    for i in range(len(d)):
        if d[i] == t[i] and len(set(t)) > 1:
            j = (i + 1) % len(d)
            d[i], d[j] = d[j], d[i]
    return d


def main() -> int:
    q = json.loads((T50 / "probe_q1_q2.json").read_text(encoding="utf-8"))
    ev = json.loads((T50 / "probe_event.json").read_text(encoding="utf-8"))
    sweep = json.loads((T50 / "dim_auc_sweep.json").read_text(encoding="utf-8"))
    cases = json.loads((ROOT / "experiments" / "reloc_cases.json").read_text(encoding="utf-8"))[:75]

    decoys = seeded_decoys(cases)
    repos_of = collections.defaultdict(set)
    for c in cases:
        repos_of[c["target_basename"]].add(c["repo"])
    by_repo = collections.defaultdict(set)
    for c in cases:
        by_repo[c["repo"]].add(c["target_basename"])
    confound = {
        # Two different owners publish `setup-steamcmd`; keying the bars on the bare name collapsed
        # them into one and produced a bar that was two colours at once.
        "repos": sorted(({"repo": (r.split("/")[-1]
                                   if sum(1 for o in by_repo if o.split("/")[-1] == r.split("/")[-1]) == 1
                                   else r),
                          "files": len(f)} for r, f in by_repo.items()),
                        key=lambda x: -x["files"]),
        "cross": sum(1 for i, c in enumerate(cases) if c["repo"] not in repos_of[decoys[i]]),
        "n": len(cases),
        "single": sum(1 for f in repos_of.values() if len(f) == 1),
        "names": len(repos_of),
    }

    data = {
        "q2": q.get("q2", []),
        "event": ev.get("layers", []),
        "event_meta": {k: ev.get(k) for k in
                       ("length_only_auc", "n_positives", "bh_threshold_smallest",
                        "bh_q05_survivors")},
        "sweep": sweep,
        "confound": confound,
    }

    OUT.write_text(PAGE.replace("__DATA__", json.dumps(data)), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  q2 layers {len(data['q2'])} | event layers {len(data['event'])} | "
          f"sweep layers {len(data['sweep'])} | repos {len(confound['repos'])}")
    return 0


PAGE = r"""<!doctype html>
<meta charset="utf-8">
<title>CTARP probe figures</title>
<script src="d3.min.js"></script>
<style>
  :root{
    --ink:#14171a; --mid:#5b6570; --faint:#c8cdd3; --rule:#e3e7ea; --bg:#ffffff;
    --real:#b5462f;     /* the measurement */
    --null:#9aa5b1;     /* the null band */
    --alt:#2f6f8f;      /* the second series */
    --warn:#8a6d1f;
  }
  html,body{background:var(--bg);color:var(--ink);margin:0;
    font:14px/1.5 "Source Sans 3","Segoe UI",system-ui,sans-serif}
  .fig{width:940px;margin:28px auto;padding:22px 26px 26px;background:var(--bg)}
  h2{font:600 17px/1.3 "Source Serif 4",Georgia,serif;margin:0 0 2px}
  .sub{color:var(--mid);font-size:12.5px;margin:0 0 16px;max-width:70ch}
  .axis text{fill:var(--mid);font-size:11px}
  .axis line,.axis path{stroke:var(--rule)}
  .lbl{font-size:11px;fill:var(--mid)}
  .key{font-size:11.5px;fill:var(--ink)}
  .note{font-size:11.5px;color:var(--mid);margin-top:10px;max-width:78ch}
  .tag{font:600 10.5px/1 ui-monospace,monospace;letter-spacing:.04em;
    text-transform:uppercase;fill:var(--mid)}
  b{color:var(--ink)}
</style>
<div class="fig" id="fig-null-band">
  <h2>Every AUC, against the null it has to clear</h2>
  <p class="sub">At 23&ndash;35 independent units a linear probe separates <i>shuffled</i> labels too,
  and the null is neither 0.5 nor constant across layers. The grey band is the permutation null
  (mean to 95th percentile) for the identical procedure; the line is the measurement. A point inside
  the band is not a result, however far above 0.5 it sits.</p>
  <svg width="880" height="300"></svg>
  <p class="note" id="nb-note"></p>
</div>

<div class="fig" id="fig-noise">
  <h2>What the best single dimension is worth</h2>
  <p class="sub">The familiar recipe: rank every unit by ROC-AUC, take the best, note its effect
  size, observe that it fails to generalise, conclude polysemanticity. The step usually skipped is
  the same procedure on <i>shuffled</i> labels &mdash; drawn here in grey. With 2,048 dimensions and
  35 claims the best-looking unit is the best of 2,048 draws from noise.</p>
  <svg width="880" height="300"></svg>
  <p class="note" id="ns-note"></p>
</div>

<div class="fig" id="fig-confound">
  <h2>Why the decoy controlled for less than it looked</h2>
  <p class="sub">Each bar is one repository, its height the number of distinct files the model ever
  relocated to inside it. A decoy drawn from the whole corpus almost always names a file from a
  <i>different</i> project &mdash; and the residual encodes which project it is reading, because the
  context is that project's source.</p>
  <svg width="880" height="260"></svg>
  <p class="note" id="cf-note"></p>
</div>

<script>
const D = __DATA__;
const F = d3.format(".3f");

/* ---------- small multiples: one panel per probe, each with ITS OWN null ----------
   Overlaying two null bands on one axis made them indistinguishable, which defeats the
   point of drawing the null at all. Each probe gets its own panel and its own band. */
function panel(svg, s, ox, W, H){
  const m={t:20,r:16,b:34,l:44};
  const x=d3.scaleLinear().domain(d3.extent(s.rows,r=>r.layer)).range([ox+m.l,ox+W-m.r]);
  const y=d3.scaleLinear().domain([0.25,0.80]).range([H-m.b,m.t]);

  svg.append("text").attr("class","tag").attr("x",ox+m.l).attr("y",m.t-6)
     .attr("fill",s.color).text(s.name);
  svg.append("g").attr("class","axis").attr("transform",`translate(0,${H-m.b})`)
     .call(d3.axisBottom(x).ticks(6).tickFormat(d=>d));
  svg.append("g").attr("class","axis").attr("transform",`translate(${ox+m.l},0)`)
     .call(d3.axisLeft(y).ticks(5));
  svg.append("text").attr("class","lbl").attr("x",ox+W-m.r).attr("y",H-6)
     .attr("text-anchor","end").text("layer");

  const area=d3.area().x(d=>x(d.layer)).y0(d=>y(d.null_mean)).y1(d=>y(d.null_p95))
               .curve(d3.curveMonotoneX);
  svg.append("path").datum(s.rows).attr("d",area).attr("fill","var(--null)").attr("opacity",.38);

  svg.append("line").attr("x1",ox+m.l).attr("x2",ox+W-m.r).attr("y1",y(0.5)).attr("y2",y(0.5))
     .attr("stroke","var(--faint)").attr("stroke-dasharray","2 3");
  svg.append("text").attr("class","lbl").attr("x",ox+m.l+3).attr("y",y(0.5)-4).text("0.5");

  const line=d3.line().x(d=>x(d.layer)).y(d=>y(d.auc)).curve(d3.curveMonotoneX);
  svg.append("path").datum(s.rows).attr("d",line).attr("fill","none")
     .attr("stroke",s.color).attr("stroke-width",2);
  svg.selectAll(null).data(s.rows).join("circle")
     .attr("cx",d=>x(d.layer)).attr("cy",d=>y(d.auc))
     .attr("r",d=>d.auc>d.null_p95?4.5:3)
     .attr("fill",d=>d.auc>d.null_p95?s.color:"var(--bg)")
     .attr("stroke",s.color).attr("stroke-width",1.5);

  // call out layer 0 explicitly: it is the diagnostic, not a data point
  const l0=s.rows[0];
  svg.append("text").attr("class","lbl").attr("x",x(l0.layer)+2).attr("y",y(l0.auc)-10)
     .attr("fill","var(--mid)").text("layer 0");
}
{
  const svg=d3.select("#fig-null-band svg").attr("height",330);
  panel(svg,{name:"event probe — is a relocation coming?",rows:D.event,color:"var(--real)"},0,440,300);
  panel(svg,{name:"right vs wrong relocation",rows:D.q2,color:"var(--alt)"},440,440,300);
  svg.append("text").attr("class","lbl").attr("transform","rotate(-90)")
     .attr("x",-26).attr("y",13).attr("text-anchor","end").text("cross-validated AUC");
  svg.append("text").attr("class","key").attr("x",440).attr("y",322)
     .attr("text-anchor","middle").attr("fill","var(--mid)")
     .text("shaded = permutation null (mean → 95th pct) · filled dot = above its null");
}
{
  const e=D.event_meta, best=D.event.reduce((a,b)=>b.auc>a.auc?b:a);
  d3.select("#nb-note").html(
    `<b>Event probe</b> (${e.n_positives} matched pairs, length shortcut worth
     ${F(e.length_only_auc)}): layer 0 &mdash; the embedding output, before any block has run &mdash;
     sits at ${F(D.event[0].auc)} inside its null, which is the diagnostic passing. The peak is
     ${F(best.auc)} at layer ${best.layer}. But eight layers are eight tests: Benjamini&ndash;Hochberg
     at q=0.05 needs the smallest p below ${F(e.bh_threshold_smallest)}, and
     <b>${(e.bh_q05_survivors&&e.bh_q05_survivors.length)?"":"nothing survives"}</b>.
     <b>Right vs wrong</b> stays inside its null at every layer.`);
}

/* ---------- figure 2: real vs shuffled vs held-out ---------- */
{
  const svg=d3.select("#fig-noise svg"), W=880,H=300,m={t:18,r:150,b:34,l:46};
  const x=d3.scaleLinear().domain(d3.extent(D.sweep,d=>d.layer)).range([m.l,W-m.r]);
  const y=d3.scaleLinear().domain([0.1,1.0]).range([H-m.b,m.t]);
  svg.append("g").attr("class","axis").attr("transform",`translate(0,${H-m.b})`)
     .call(d3.axisBottom(x).ticks(8));
  svg.append("g").attr("class","axis").attr("transform",`translate(${m.l},0)`)
     .call(d3.axisLeft(y).ticks(5));
  svg.append("text").attr("class","lbl").attr("x",W-m.r).attr("y",H-6)
     .attr("text-anchor","end").text("layer");
  svg.append("text").attr("class","lbl").attr("transform","rotate(-90)")
     .attr("x",-m.t-6).attr("y",14).attr("text-anchor","end").text("AUC of the best dimension");
  const series=[
    {k:"in_sample",name:"real labels, in-sample",c:"var(--real)"},
    {k:"shuffled_in_sample",name:"SHUFFLED labels, in-sample",c:"var(--null)"},
    {k:"held_out",name:"real labels, held-out claims",c:"var(--alt)"}
  ];
  series.forEach(s=>{
    const line=d3.line().x(d=>x(d.layer)).y(d=>y(d[s.k])).curve(d3.curveMonotoneX);
    svg.append("path").datum(D.sweep).attr("d",line).attr("fill","none")
       .attr("stroke",s.c).attr("stroke-width",s.k==="shuffled_in_sample"?2.5:2)
       .attr("stroke-dasharray",s.k==="shuffled_in_sample"?"6 3":null);
    svg.selectAll(null).data(D.sweep).join("circle")
       .attr("cx",d=>x(d.layer)).attr("cy",d=>y(d[s.k])).attr("r",3).attr("fill",s.c);
    const last=D.sweep[D.sweep.length-1];
    svg.append("text").attr("class","key").attr("x",x(last.layer)+9).attr("y",y(last[s.k])+4)
       .attr("fill",s.c).text(s.name);
  });
  svg.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(0.5)).attr("y2",y(0.5))
     .attr("stroke","var(--faint)").attr("stroke-dasharray","2 3");
  const mi=d3.mean(D.sweep,d=>d.in_sample), ms=d3.mean(D.sweep,d=>d.shuffled_in_sample),
        mh=d3.mean(D.sweep,d=>d.held_out), md=d3.max(D.sweep,d=>d.cohens_d);
  d3.select("#ns-note").html(
    `Mean in-sample AUC is <b>${F(mi)}</b> with real labels and <b>${F(ms)}</b> with random ones &mdash;
     the same number. Held-out drops to <b>${F(mh)}</b>, and Cohen's <i>d</i> reaches
     <b>${md.toFixed(2)}</b> on labels that carry no information. "High AUC, large effect, fails to
     generalise, therefore polysemanticity" is a conclusion this data reaches with nothing present.`);
}

/* ---------- figure 3: the nesting that broke the decoy ---------- */
{
  // generous bottom margin: two repositories need their owner prefix to disambiguate, and the
  // rotated labels were being clipped at the old height
  const svg=d3.select("#fig-confound svg").attr("height",310), W=880,H=310,m={t:18,r:24,b:124,l:46};
  const r=D.confound.repos;
  const x=d3.scaleBand().domain(r.map(d=>d.repo)).range([m.l,W-m.r]).padding(0.28);
  const y=d3.scaleLinear().domain([0,d3.max(r,d=>d.files)]).nice().range([H-m.b,m.t]);
  svg.append("g").attr("class","axis").attr("transform",`translate(0,${H-m.b})`)
     .call(d3.axisBottom(x)).selectAll("text")
     .attr("transform","rotate(-38)").attr("text-anchor","end").attr("dx","-.5em").attr("dy",".2em");
  svg.append("g").attr("class","axis").attr("transform",`translate(${m.l},0)`)
     .call(d3.axisLeft(y).ticks(4));
  svg.append("text").attr("class","lbl").attr("transform","rotate(-90)")
     .attr("x",-m.t-6).attr("y",14).attr("text-anchor","end").text("distinct files relocated to");
  svg.selectAll(null).data(r).join("rect")
     .attr("x",d=>x(d.repo)).attr("y",d=>y(d.files))
     .attr("width",x.bandwidth()).attr("height",d=>y(0)-y(d.files))
     .attr("fill",d=>d.files>=2?"var(--real)":"var(--faint)");
  // the line a within-repo decoy needs to clear
  svg.append("line").attr("x1",m.l).attr("x2",W-m.r).attr("y1",y(2)).attr("y2",y(2))
     .attr("stroke","var(--warn)").attr("stroke-dasharray","4 3");
  svg.append("text").attr("class","tag").attr("x",W-m.r).attr("y",y(2)-7)
     .attr("text-anchor","end").attr("fill","var(--warn)")
     .text("2 files — the minimum for a within-repository decoy");
  const c=D.confound;
  d3.select("#cf-note").html(
    `<b>${c.single} of ${c.names}</b> distinct relocated filenames occur in exactly one repository,
     and <b>${c.cross} of ${c.n}</b> seeded decoys name a file from a different project than the one
     being read. Only the ${r.filter(d=>d.files>=2).length} highlighted repositories can supply a
     same-repository decoy at all &mdash; which is why the deconfounded probe scores two claims.`);
}
</script>
"""

if __name__ == "__main__":
    raise SystemExit(main())
