"""Launch the CONTROL residual extraction detached, after the relocation pass has finished.

Same extractor, same model, same prompt construction -- only the cases file and the output paths
differ, and they differ through environment variables rather than through a second copy of the
script. The controls have to be processed identically to the positives or they are not controls.

One cut per control, not two: a control has no relocation, so `prefix_at_sentence` and
`prefix_at_path` hold the same string and extracting both would just duplicate 74 forward passes.

REFUSES TO START while the first extraction is still running. Two 35B models on one 40 GB A100 is
an OOM, and the failure arrives inside `from_pretrained` as a request for ~96% of the budget, which
reads like a configuration error rather than a collision.

Run: colab exec -s <session> -f experiments/colab_run_controls.py
Then: colab exec -s <session> -f experiments/colab_controls_status.py
"""

import os
import subprocess
import sys

SCRIPT = "/content/extract_residuals.py"
LOG = "/content/extract_ctrl.log"
FLAG = "/content/extract_ctrl.done"
FIRST_FLAG = "/content/extract.done"

if os.path.exists(FIRST_FLAG):
    print(f"first pass finished: {open(FIRST_FLAG).read().strip()!r}", flush=True)
else:
    # A fresh session after a prune has no first pass to wait for. The real constraint is free
    # VRAM, and the extractor checks that itself and refuses below 34 GiB -- so this is a note,
    # not a gate. Gating on the flag here would make a pruned run unrecoverable.
    print(f"note: {FIRST_FLAG} absent (fresh session?). The extractor's own free-VRAM check "
          "is the guard against a collision.", flush=True)

ENV = dict(os.environ,
           RESID_CASES="/content/control_cases.json",
           RESID_OUT="/content/resid_ctrl.npz",
           RESID_META="/content/resid_ctrl_meta.json",
           RESID_CUTS="prefix_at_sentence",
           RESID_MAX="75")

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
print("CONTROL WRAPPER FINISHED", flush=True)
"""

# Clear the previous attempt's flag. Leaving it means the status poller reads "DONE fail" from the
# run that already ended and reports the new one as broken before it has done anything.
if os.path.exists(FLAG):
    print(f"clearing stale flag: {open(FLAG).read().strip()!r}", flush=True)
    os.remove(FLAG)

log = open(LOG, "w")
p = subprocess.Popen([sys.executable, "-u", "-c", WRAP], stdout=log,
                     stderr=subprocess.STDOUT, env=ENV)
print(f"controls detached, pid={p.pid}, log={LOG}, flag={FLAG}", flush=True)
