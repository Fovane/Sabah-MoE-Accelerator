"""How much pinned host memory does this rig actually grant a process?"""
import sys

ROOT = r"C:\Projects\quantization\Sabah-MoE-Accelerator"
sys.path.insert(0, ROOT)

from sabah.runtime import rt as _rt

rt = _rt
PER = 704000          # the size the serve path asks for, per expert
ptrs = []
try:
    for i in range(200000):
        p = rt.lib().sabah_host_alloc(PER)
        if not p:
            print("alloc #%d returned NULL after %.2f GB pinned"
                  % (i, len(ptrs) * PER / 1e9))
            break
        ptrs.append(p)
except Exception as e:                     # noqa: BLE001
    print("alloc #%d raised: %s" % (len(ptrs), e))

print("granted %d buffers x %d B = %.3f GB pinned" % (len(ptrs), PER, len(ptrs) * PER / 1e9))
