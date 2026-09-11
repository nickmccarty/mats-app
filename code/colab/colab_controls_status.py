"""Tail the detached control extraction. Short, so the poll never outruns the exec read timeout."""

import os
import re

LOG = "/content/extract_ctrl.log"
FLAG = "/content/extract_ctrl.done"
OUT = "/content/resid_ctrl.npz"

KEEP = re.compile(r"GPU free|REFUSING|model ready|FAILED|Traceback|Error|wrote /content|^\s*\[")

state = open(FLAG).read().strip() if os.path.exists(FLAG) else ""
lines = open(LOG, errors="replace").read().splitlines() if os.path.exists(LOG) else []
hits = [l for l in lines if KEEP.search(l)]
size = os.path.getsize(OUT) if os.path.exists(OUT) else 0

print(f"CONTROLS {'DONE ' + state if state else 'RUNNING'}  ({len(lines)} log lines, "
      f"npz {size:,} bytes)")
for l in hits[-12:]:
    print(l)
