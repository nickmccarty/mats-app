"""Pull the model and the lens onto the VM in the background, so no download runs inside exec.

WHY. `colab exec` is a blocking HTTP call with a read timeout on the client. On a cached VM the
probe loads in a couple of minutes and that is fine. On a FRESH VM the same script has to fetch
67 GB first, the call outruns the timeout, and the CLI raises ConnectionError/ReadTimeout from
its own transport (exec_command -> start_kernel -> fetch -> post -> send) -- which looks like a
model or network failure and is actually the client giving up on a call that was always going to
take a quarter of an hour.

So the download is detached here: Popen it, return immediately, and poll with colab_prefetch_status.
The probe then runs against a warm cache and finishes inside the timeout, exactly as it did on the
session that already had the weights.

The lens is fetched in the same pass. It is only ~1 GB against the model's 67, but a download
inside the run is a download inside the run regardless of size, and this removes the last one.

Run: colab exec -s <session> -f experiments/colab_prefetch.py
Then: colab exec -s <session> -f experiments/colab_prefetch_status.py   (until it says DONE)
"""

import subprocess
import sys

MODEL = "Qwen/Qwen3.6-35B-A3B"
LENS_REPO = "camilablank/workspace-lenses"
LENS_FILE = "qwen3.6-35b-a3b/j-lens/lens.pt"
LOG = "/content/prefetch.log"
FLAG = "/content/prefetch.done"

CODE = f"""
import traceback
from huggingface_hub import snapshot_download, hf_hub_download
try:
    print("lens ...", flush=True)
    hf_hub_download({LENS_REPO!r}, filename={LENS_FILE!r})
    print("model ...", flush=True)
    snapshot_download({MODEL!r}, max_workers=8)
    open({FLAG!r}, "w").write("ok")
    print("PREFETCH OK", flush=True)
except Exception:
    traceback.print_exc()
    open({FLAG!r}, "w").write("fail")
    print("PREFETCH FAILED", flush=True)
"""

log = open(LOG, "w")
p = subprocess.Popen([sys.executable, "-u", "-c", CODE], stdout=log, stderr=subprocess.STDOUT)
print(f"prefetch detached, pid={p.pid}, log={LOG}, flag={FLAG}", flush=True)
