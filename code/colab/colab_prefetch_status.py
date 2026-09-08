"""How far along is the detached prefetch? Short, so it always beats the exec read timeout.

Prints one of:
  PREFETCH DONE    - weights and lens are cached; run the probe now
  PREFETCH FAILED  - the detached job raised; the tail of its log says why
  PREFETCH RUNNING - still going, with GB on disk so progress is visible rather than inferred
"""

import os
import pathlib

CACHE = pathlib.Path(os.environ.get("HF_HOME", "/root/.cache/huggingface")) / "hub"
LOG = "/content/prefetch.log"
FLAG = "/content/prefetch.done"


def gb() -> float:
    total = 0
    for root, _dirs, files in os.walk(CACHE):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 2**30


size = gb()
state = pathlib.Path(FLAG).read_text().strip() if os.path.exists(FLAG) else ""
tail = ""
if os.path.exists(LOG):
    tail = "".join(open(LOG, errors="replace").readlines()[-4:]).strip()

if state == "ok":
    print(f"PREFETCH DONE  cache={size:.1f} GB")
elif state == "fail":
    print(f"PREFETCH FAILED  cache={size:.1f} GB\n{tail}")
else:
    print(f"PREFETCH RUNNING  cache={size:.1f} GB (target ~68 GB)\n{tail}")
