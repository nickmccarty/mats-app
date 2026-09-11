"""Launch the residual extraction detached, for the same reason the probe is detached.

`colab exec` is a blocking HTTP call with a client-side read timeout; a run that takes ninety
minutes outruns it and surfaces as a ConnectionError from the CLI's own transport. Detaching makes
the run's lifetime independent of the call's.

Expects /content/extract_residuals.py to exist (colab upload it first).

Run: colab exec -s <session> -f experiments/colab_run_extract.py
Then: colab exec -s <session> -f experiments/colab_extract_status.py
"""

import subprocess
import sys

SCRIPT = "/content/extract_residuals.py"
LOG = "/content/extract.log"
FLAG = "/content/extract.done"

WRAP = f"""
import runpy, sys, traceback
sys.argv = [{SCRIPT!r}]
try:
    runpy.run_path({SCRIPT!r}, run_name="__main__")
    open({FLAG!r}, "w").write("ok")
except SystemExit as e:
    open({FLAG!r}, "w").write("ok" if not e.code else f"exit:{{e.code}}")
except Exception:
    traceback.print_exc()
    open({FLAG!r}, "w").write("fail")
print("EXTRACT WRAPPER FINISHED", flush=True)
"""

log = open(LOG, "w")
p = subprocess.Popen([sys.executable, "-u", "-c", WRAP], stdout=log, stderr=subprocess.STDOUT)
print(f"extract detached, pid={p.pid}, log={LOG}, flag={FLAG}", flush=True)
