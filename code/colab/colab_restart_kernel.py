"""DO NOT RUN THIS. It wedges the CLI session, twice observed, and it is almost never needed.

WHAT IT USED TO DO. `os._exit(0)` inside the Colab kernel, to drop a model that a raised exception
had left resident on the GPU.

WHY IT IS DISARMED. The kernel dies before it can acknowledge the exec RPC, so the CLI's session
state stays BUSY on this script forever: every later `colab exec` blocks client-side, and the only
way out is `colab stop` + `colab new`, which also discards the 135 GB weight cache. That has now
happened twice -- on reloc2 and again on reloc7, the second time when the restart was not even
needed, because the thing holding memory was a DETACHED subprocess that had already exited.

WHAT TO DO INSTEAD:
  - a detached probe that crashed has already released its memory; just relaunch it
  - a crashed in-kernel run: `free_gpu()` in jlens_reloc_probe.py clears sys.last_traceback and
    IPython's Out[] refs, and REFUSES to load if the card is still occupied, which is the check
    that matters
  - genuinely stuck kernel: `colab stop` + `colab new` is slower but leaves the CLI consistent

Run: nothing happens. That is deliberate.
"""

print("colab_restart_kernel.py is DISARMED — it wedges the CLI session. See the docstring.")
print("A detached probe that exited has already freed the GPU; relaunch it instead.")
