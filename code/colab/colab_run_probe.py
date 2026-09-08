"""Run the probe detached on the VM, for the same reason the download is detached.

Attempt 7 died in the Colab CLI's HTTP read timeout while a fresh VM pulled 67 GB inside a
blocking exec. The probe is now at risk of the same thing for a different reason: it makes 120
apply() calls (30 cases x 2 cuts x jacobian/logit), and where the run that finished used a
512-token window, this one uses 4096 through a model whose layers mostly stream from system RAM.
An 8x longer forward turns a few minutes into potentially forty, which is well past the timeout.

Detaching costs nothing and removes the failure mode: the probe writes to /content/probe.log on
the VM, and colab_probe_status.py tails it. The run is then bounded by the A100, not by an HTTP
client's patience.

Expects /content/jlens_reloc_probe.py to exist (colab upload it first).

Run: colab exec -s <session> -f experiments/colab_run_probe.py
Then: colab exec -s <session> -f experiments/colab_probe_status.py
"""

import os
import subprocess
import sys

PROBE = "/content/jlens_reloc_probe.py"
LOG = "/content/probe.log"
FLAG = "/content/probe.done"

# WHICH CASES TO SCORE. Empty means all of them. Set it to e.g. "73,74" to resume a run that was
# cut short -- the probe still iterates the full case list and skips, so the seeded decoy pairing
# is identical to a complete run and the new rows merge with the old ones.
#
# This is a CONSTANT, not an environment variable: `colab exec` runs this file on the VM, where
# the local shell's environment does not exist, so an env lookup here would always read "".
ONLY = "73,74"

WRAP = f"""
import os, runpy, sys, traceback
os.environ["ONLY_CASES"] = {ONLY!r}
sys.argv = [{PROBE!r}]
try:
    runpy.run_path({PROBE!r}, run_name="__main__")
    open({FLAG!r}, "w").write("ok")
except SystemExit as e:
    open({FLAG!r}, "w").write("ok" if not e.code else f"exit:{{e.code}}")
except Exception:
    traceback.print_exc()
    open({FLAG!r}, "w").write("fail")
print("PROBE WRAPPER FINISHED", flush=True)
"""

log = open(LOG, "w")
p = subprocess.Popen([sys.executable, "-u", "-c", WRAP], stdout=log, stderr=subprocess.STDOUT)
print(f"probe detached, pid={p.pid}, log={LOG}, flag={FLAG}", flush=True)
