"""How many pinned bytes would the serve path ask for, at context 131072?

Constructed with an empty warm list on purpose: ExpertBank.__init__ calls
load_warm(), so a bank built with the planner's blocks would try to pin them and
die the same way the serve path does.
"""
import os
import sys

ROOT = r"C:\Projects\quantization\Sabah-MoE-Accelerator"
sys.path.insert(0, ROOT)

MODEL = r"C:\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
CONTEXT = int(sys.argv[1]) if len(sys.argv) > 1 else 131072
GRANTED = 5.034e9          # measured on this rig with tools/probe_pinned.py

from sabah.core.model_inspector import inspect_model
from sabah.hardware.profile import HardwareProfile, qualify
from sabah.planner.planner import plan as build_plan
from sabah.runtime.expert_bank import ExpertBank
from sabah.tools.cli import HW_PATH

profile = inspect_model(MODEL)
hw = HardwareProfile.load(HW_PATH()) if os.path.exists(HW_PATH()) else qualify(profile.path)
projection = build_plan(profile, hw, context=CONTEXT, concurrency=1)

bank = ExpertBank(profile, mode="mmap", warm_blocks=[])     # no pinning yet

blocks = list(projection.warm_blocks or [])
want = 0
allocs = 0
per = 0
for b in blocks:
    if b not in bank.blocks:
        continue
    for r in bank.ROLES:
        t = bank.desc[(b, r)]
        p = int(t["per_expert_bytes"])
        want += bank.n_experts * p
        allocs += bank.n_experts
        per = p

print("context              : %d" % CONTEXT)
print("n_experts (per layer): %d" % bank.n_experts)
print("roles                : %d  (blocks: %d)"
      % (len(bank.ROLES), len([b for b in blocks if b in bank.blocks])))
print("bytes per expert     : %d" % per)
print("pinned allocs wanted : %d" % allocs)
print("pinned bytes wanted  : %.3f GB" % (want / 1e9))
print("pinned bytes granted : %.3f GB (measured, tools/probe_pinned.py)" % (GRANTED / 1e9))
print("VERDICT              : %s" % (
    "serve cannot start: wants %.0f%% of what this driver grants"
    % (100.0 * want / GRANTED) if want > GRANTED else "fits"))
bank.close()
