#!/usr/bin/env bash
# Assemble the landing-page video from still frames, with crossfades.
#
# WHY STILLS AND NOT SCREEN CAPTURE. Three gdigrab takes were thrown away: the first two were
# force-killed so the MP4 index was never written ("moov atom not found"), and the third was
# captured at 1520x950 on a 3840x2400 physical desktop -- a zoomed corner of the screen with the
# operator's own chat window in frame. Browser screenshots have none of those failure modes: exact
# viewport, no desktop furniture, no container to corrupt. The UI frames below are the real
# completed run (20260907-111923-eval001), not a mock-up.
#
# EVERY CLAIM ON THE CARDS IS CHECKED against data/vulns.db and the casefile:
#   gold      src/pyload/webui/app/blueprints/cnl_blueprint.py  [[23, 34]]
#   fix       f4e2d12416ba2dfac7b036d5c8d6dab5461b9840  (GHSA-x698-5hjm-w2m5)
#   reasoning "lines 22-36, the local_check decorator"        <- contains gold
#   emitted   cnl_blueprint.py:43  (add_cors)                 <- outside gold
#
# Run: bash demo/build_cut.sh
set -euo pipefail
cd "$(dirname "$0")"

F=frames
OUT=relocation.mp4
FPS=30
X=0.8   # crossfade seconds

# shot list: file, seconds on screen
SHOTS=(
  "$F/c1.png:5.0"
  "$F/ui_question.png:5.5"
  "$F/c2.png:4.5"
  "$F/ui_citation.png:7.0"
  "$F/c3.png:6.0"
  "$F/c4.png:7.0"
  "$F/ui_evidence.png:5.0"
  "$F/c5.png:7.5"
  "$F/c6.png:8.0"
)

rm -rf .seg && mkdir -p .seg
i=0
for s in "${SHOTS[@]}"; do
  img="${s%%:*}"; dur="${s##*:}"
  # A still becomes a clip. yuv420p and even dimensions keep it playable everywhere.
  ffmpeg -v error -loop 1 -i "$img" -t "$dur" -r $FPS \
    -vf "scale=1536:880:force_original_aspect_ratio=decrease,pad=1536:880:(ow-iw)/2:(oh-ih)/2:color=0x121619,format=yuv420p" \
    -c:v libx264 -preset veryfast -crf 20 ".seg/$(printf %02d $i).mp4"
  i=$((i+1))
done

# Chain the clips with xfade. Each xfade shortens the timeline by X, so the offset for clip n is
# (sum of durations so far) - n*X. Computing it wrong is the classic way to get a video that
# freezes on shot three.
n=${#SHOTS[@]}
inputs=(); for f in .seg/*.mp4; do inputs+=(-i "$f"); done

filter=""; prev="0:v"; acc=0
for ((k=1;k<n;k++)); do
  d="${SHOTS[$((k-1))]##*:}"
  acc=$(python -c "print(round($acc + $d, 3))")
  off=$(python -c "print(round($acc - $k*$X, 3))")
  lbl="v$k"
  filter+="[$prev][$k:v]xfade=transition=fade:duration=$X:offset=$off[$lbl];"
  prev="$lbl"
done
filter="${filter%;}"

ffmpeg -v error -y "${inputs[@]}" -filter_complex "$filter" -map "[$prev]" \
  -c:v libx264 -preset slow -crf 19 -pix_fmt yuv420p -movflags +faststart "$OUT"

rm -rf .seg
ffprobe -v error -show_entries format=duration:stream=width,height -of default=nw=1 "$OUT"
echo "wrote demo/$OUT"
