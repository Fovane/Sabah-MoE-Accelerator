"""Gate A: reproducible integration.

1. Check out pristine llama.cpp 96ffdc41c into a fresh worktree, apply patches
   0001 and 0002, and compare every patched source with the tested tree.
2. Build it from scratch with the documented CUDA configuration.
3. Run the freshly built llama-sabah-diag and the tested build on development
   prompt dev_rc4 (native CUDA path, 1 step) and compare the argmax.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V5 = os.path.abspath(os.path.join(HERE, "..", ".."))
TESTED = os.environ.get("SABAH_LLAMA_TREE", "D:/sabah_scaling/llama-sabah-clean")
FRESH = os.environ.get("SABAH_LLAMA_FRESH", "D:/sabah_v1build/llama")
BASE = "96ffdc41ceb055e1c2d3d96667ae6d9f0ccb710b"
PATCHES = [os.path.join(V5, "patches", "llama.cpp", p) for p in
           ("0001-sabah-native-mul-mat-id.patch", "0002-sabah-diag-capture-tool.patch")]
FILES = ["ggml/src/ggml-backend.cpp", "ggml/src/ggml-cuda/ggml-cuda.cu", "tools/CMakeLists.txt",
         "tools/sabah-diag/sabah-diag.cpp", "tools/sabah-diag/CMakeLists.txt"]
MSVC = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"


def sh(cmd, cwd=None, env=None):
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, errors="replace")


def norm(p):
    return open(p, "rb").read().replace(b"\r\n", b"\n")


def main(out):
    res = dict(base=BASE, fresh_tree=FRESH, steps={})
    if os.path.exists(FRESH):
        sh(["git", "worktree", "remove", "--force", FRESH], cwd=TESTED)
        shutil.rmtree(FRESH, ignore_errors=True)
    os.makedirs(os.path.dirname(FRESH), exist_ok=True)
    r = sh(["git", "worktree", "add", "--detach", FRESH, BASE], cwd=TESTED)
    res["steps"]["checkout"] = r.returncode == 0
    for p in PATCHES:
        r = sh(["git", "apply", p], cwd=FRESH)
        res["steps"]["apply_" + os.path.basename(p)] = r.returncode == 0
    res["identical_sources"] = {f: norm(os.path.join(FRESH, f)) == norm(os.path.join(TESTED, f)) for f in FILES}
    env = dict(os.environ, PATH=MSVC + os.pathsep + os.environ["PATH"])
    r = sh(["cmake", "-S", ".", "-B", "build", "-DGGML_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_NATIVE=OFF", "-DCMAKE_CUDA_ARCHITECTURES=89"], cwd=FRESH, env=env)
    res["steps"]["configure"] = r.returncode == 0
    r = sh(["cmake", "--build", "build", "--config", "Release", "--target", "llama-sabah-diag", "llama-server",
            "-j", "8"], cwd=FRESH, env=env)
    res["steps"]["build"] = r.returncode == 0
    if r.returncode != 0:
        res["build_tail"] = r.stdout[-3000:]
    # smoke: same development prompt through both builds
    sys.path.insert(0, HERE)
    import run_path
    mf = os.path.join(os.path.dirname(out), "dev_rc4.jsonl")
    open(mf, "w", encoding="utf-8").write(json.dumps({"id": "dev_rc4", "text": "Reply with exactly: SABAH_OK"}) + "\n")
    got = {}
    for tag, bindir in (("tested", os.path.join(TESTED, "build", "bin", "Release")),
                        ("fresh", os.path.join(FRESH, "build", "bin", "Release"))):
        run_path.DIAG = os.path.join(bindir, "llama-sabah-diag.exe")
        d = os.path.join(os.path.dirname(out), "R18_smoke_" + tag)
        rec = run_path.run("cuda", d, mf, 1)
        a = os.path.join(d, "dev_rc4", "argmax.i32")
        import numpy as np
        got[tag] = dict(rc=rec["rc"], argmax=np.fromfile(a, np.int32).tolist() if os.path.exists(a) else None)
    res["smoke"] = got
    res["ok"] = (all(res["steps"].values()) and all(res["identical_sources"].values()) and
                 got["tested"]["rc"] == 0 and got["fresh"]["rc"] == 0 and
                 got["tested"]["argmax"] is not None and got["tested"]["argmax"] == got["fresh"]["argmax"])
    res["summary"] = "sources identical %d/%d; build %s; smoke argmax equal %s" % (
        sum(res["identical_sources"].values()), len(FILES), res["steps"].get("build"),
        got["tested"]["argmax"] == got["fresh"]["argmax"])
    json.dump(res, open(out, "w"), indent=1)
    print(res["summary"], "ok" if res["ok"] else "FAIL")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "D:/sabah_v1val/R18_build.json")
