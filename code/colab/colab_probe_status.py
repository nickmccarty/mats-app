"""Tail the detached probe. Short by design, so the poll never outruns the exec read timeout.

Prints the milestones that distinguish the failure modes this experiment has actually hit --
free VRAM before load, the load itself, the decoded prompt tail, per-case scores -- rather than
the whole log, which is thousands of lines of transformers noise.
"""

import os
import re

LOG = "/content/probe.log"
FLAG = "/content/probe.done"

KEEP = re.compile(
    r"GPU free after cleanup|REFUSING|lm_head on:|model \+ lens ready|\[window \d+ tok\]|"
    r"OutOfMemoryError|Traceback|Error:|FAILED|case +\d+ +prefix_|=== SUMMARY ===|"
    r"j-lens|logit-lens|mentions with|CLAIMS|READ THE CONTROL"
)

state = open(FLAG).read().strip() if os.path.exists(FLAG) else ""
lines = open(LOG, errors="replace").read().splitlines() if os.path.exists(LOG) else []
hits = [l for l in lines if KEEP.search(l)]

print(f"PROBE {'DONE ' + state if state else 'RUNNING'}  ({len(lines)} log lines)")
for l in hits[-40:]:
    print(l)
