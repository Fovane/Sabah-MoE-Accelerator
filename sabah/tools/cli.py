"""
Sabah Accelerator — command line.

    sabah inspect  <model.gguf>     what the model is, and whether Sabah supports it
    sabah qualify  [model.gguf]     measure this machine, save a hardware profile
    sabah plan     <model.gguf>     build an execution plan (projection)
    sabah doctor                    environment check with actionable fixes
    sabah status                    show the saved profiles
    sabah requalify [model.gguf]    force re-measurement
    sabah selftest [model.gguf]     prove the runtime decodes and executes
                                    experts exactly (needs the CUDA runtime)
    sabah bench    [model.gguf]     MEASURED per-block throughput and the
                                   forced-resident vs streamed A/B
    sabah benchmark <model.gguf>   full-model reference provenance benchmark
    sabah serve    <model.gguf>    local OpenAI-compatible API; choose
                                   --backend sabah or --backend reference
    sabah calibrate                 re-measure the planner hit curve from a
                                    routing trace

Projections are always labelled. A measured number and a projected number are
never printed in the same style.
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import platform

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sabah.core.model_inspector import (inspect_model, summarize as model_summary,
                                        UnsupportedModel, ARCHS)
from sabah.hardware.profile import (HardwareProfile, qualify,
                                    summarize as hw_summary, QUALIFY_EXE)
from sabah.planner.planner import plan as build_plan, summarize as plan_summary


def state_dir() -> str:
    d = os.environ.get("SABAH_HOME")
    if not d:
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        d = os.path.join(base, "Sabah")
    os.makedirs(d, exist_ok=True)
    return d


HW_PATH = lambda: os.path.join(state_dir(), "hardware_profile.json")
MODEL_CACHE = lambda h: os.path.join(state_dir(), "model_%s.json" % h)


def _banner(title):
    print("=" * 74)
    print("Sabah Accelerator — %s" % title)
    print("=" * 74)


def cmd_inspect(args):
    _banner("model inspection")
    try:
        prof = inspect_model(args.model)
    except UnsupportedModel as e:
        print("cannot inspect: %s" % e)
        return 2
    print(model_summary(prof))
    if args.json:
        print()
        print(prof.to_json())
    return 0 if prof.supported else 3


def cmd_qualify(args):
    _banner("hardware qualification")
    if not os.path.exists(QUALIFY_EXE):
        print("note: the CUDA qualifier is not built, so H2D bandwidth cannot")
        print("      be measured. Build it with:")
        print("        nvcc -O3 -o %s %s"
              % (QUALIFY_EXE, QUALIFY_EXE.replace(".exe", ".cu")))
        print()
    hw = qualify(args.model or "", verbose=True)
    print()
    print(hw_summary(hw))
    hw.save(HW_PATH())
    print()
    print("saved: %s" % HW_PATH())
    return 0


def _load_or_qualify(model_path):
    p = HW_PATH()
    if os.path.exists(p):
        try:
            return HardwareProfile.load(p), False
        except Exception as e:
            print("hardware profile unusable (%s); re-measuring" % e)
    return qualify(model_path, verbose=True), True


def cmd_plan(args):
    _banner("execution plan")
    try:
        model = inspect_model(args.model)
    except UnsupportedModel as e:
        print("cannot inspect: %s" % e)
        return 2
    if not model.supported:
        print(model_summary(model))
        print()
        print("Sabah acceleration not available for this artifact.")
        print("Recommended mode: reference runtime.")
        return 3

    hw, fresh = _load_or_qualify(args.model)
    if fresh:
        hw.save(HW_PATH())
    print()
    p = build_plan(model, hw, context=args.context,
                   concurrency=args.concurrency, workload=args.workload)
    print(plan_summary(p))
    print()
    if p.recommended:
        print("PROJECTED acceleration %.2f-%.2fx. This is a projection from the"
              % (p.projected_speedup_low, p.projected_speedup_high))
        print("model and the measured machine, NOT a benchmark result.")
        print("Run `sabah bench` on this machine to replace it with a")
        print("measured figure.")
    else:
        print("Sabah acceleration not recommended on this hardware.")
        print("Reason: %s" % (p.bottleneck or "insufficient resources"))
        print("Recommended mode: reference runtime.")
    if args.json:
        print()
        print(p.to_json())
    return 0


def cmd_selftest(args):
    _banner("runtime self-test")
    from sabah.runtime import rt
    if not rt.available():
        print("The CUDA runtime is not built, so exact-execution cannot be")
        print("verified on this machine. Build it with:")
        print("    cd sabah/runtime/cuda")
        print("    nvcc -O3 -arch=sm_<cc> -shared -o sabah_rt.dll sabah_rt.cu")
        print("where <cc> is this GPU's compute capability without the dot")
        print("(nvidia-smi --query-gpu=compute_cap --format=csv,noheader).")
        return 2
    rc = 0
    from sabah.tests import test_quant_kernels, test_moe_block
    model = args.model or None
    print(">> quant decode vs gguf-py reference")
    rc |= test_quant_kernels.main(model) or 0
    print()
    print(">> exact expert execution vs CPU reference")
    rc |= test_moe_block.main(model) or 0
    print()
    print("selftest: %s" % ("PASSED" if rc == 0 else "FAILED"))
    return rc


def cmd_bench(args):
    from sabah.tools import bench_moe
    argv = ["--block", str(args.block), "--tokens", str(args.tokens),
            "--slots", args.slots, "--bank-mode", args.bank_mode]
    if args.model:
        argv += ["--model", args.model]
    if args.trace:
        argv += ["--trace", args.trace]
    if args.json:
        argv += ["--json-out", args.json]
    return bench_moe.main(argv)


def cmd_calibrate(args):
    from sabah.tools import calib_check
    argv = []
    if args.model:
        argv += ["--model", args.model]
    if args.trace:
        argv += ["--trace", args.trace]
    if args.json:
        argv += ["--json-out", args.json]
    return calib_check.main(argv)


def cmd_benchmark(args):
    from sabah.tools import benchmark
    argv = [args.model, "--prompt", args.prompt, "--tokens", str(args.tokens),
            "--context", str(args.context)]
    if args.threads:
        argv += ["--threads", str(args.threads)]
    if args.llama_cli:
        argv += ["--llama-cli", args.llama_cli]
    if args.json:
        argv += ["--json-out", args.json]
    if args.dry_run:
        argv += ["--dry-run"]
    return benchmark.main(argv)


def cmd_serve(args):
    from sabah.server import openai_proxy
    backend = args.backend or ("reference" if args.allow_reference else None)
    if backend is None:
        print("serve: choose --backend sabah (Sabah executes every expert MUL_MAT_ID) "
              "or --backend reference --allow-reference (stock llama.cpp)")
        return 2
    try:
        return openai_proxy.main([
            args.model, "--host", args.host, "--port", str(args.port),
            "--backend-port", str(args.backend_port), "--context", str(args.context),
            "--n-gpu-layers", str(args.n_gpu_layers),
            "--backend", backend, "--hot-bytes", str(args.hot_bytes),
            *( ["--llama-server", args.llama_server] if args.llama_server else [] ),
            *( ["--allow-reference"] if args.allow_reference else [] ),
            *( ["--quiet"] if args.quiet else [] ),
        ])
    except (RuntimeError, ValueError, FileNotFoundError) as e:
        print("serve: %s" % e)
        return 2


def cmd_doctor(args):
    _banner("doctor")
    ok = True

    def check(label, good, detail=""):
        nonlocal ok
        print("  [%s] %-42s %s" % ("ok" if good else "!!", label, detail))
        if not good:
            ok = False

    print("environment")
    check("python %s" % platform.python_version(), sys.version_info >= (3, 8))
    try:
        import numpy  # noqa
        check("numpy", True, numpy.__version__)
    except Exception:
        check("numpy", False, "pip install numpy")
    try:
        sys.path.insert(0, r"D:/llama-glm53/gguf-py")
        import gguf  # noqa
        check("gguf-py importable", True)
    except Exception as e:
        check("gguf-py importable", False, "set SABAH_GGUF_PY (%s)" % e)

    print("cuda")
    import shutil
    check("nvidia-smi on PATH", shutil.which("nvidia-smi") is not None)
    check("H2D qualifier built", os.path.exists(QUALIFY_EXE),
          "" if os.path.exists(QUALIFY_EXE) else "nvcc -O3 -o mgpu_qualify.exe mgpu_qualify.cu")
    from sabah.runtime import rt as _rt
    _rtlib = _rt.library_path()
    check("expert runtime built", os.path.exists(_rtlib),
          _rtlib if os.path.exists(_rtlib)
          else "nvcc -O3 -arch=sm_<cc> -shared -o sabah_rt.dll sabah_rt.cu")

    print("state")
    check("state directory writable", os.access(state_dir(), os.W_OK), state_dir())
    check("hardware profile present", os.path.exists(HW_PATH()),
          "" if os.path.exists(HW_PATH()) else "run `sabah qualify`")

    print("supported architectures")
    for a, spec in sorted(ARCHS.items()):
        print("  - %-12s %s" % (a, spec.note))

    print()
    print("doctor: %s" % ("all checks passed" if ok else "issues found above"))
    return 0 if ok else 1


def cmd_status(args):
    _banner("status")
    p = HW_PATH()
    if not os.path.exists(p):
        print("no hardware profile yet — run `sabah qualify`")
        return 1
    hw = HardwareProfile.load(p)
    print(hw_summary(hw))
    print()
    print("profile: %s" % p)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sabah",
                                 description="Sabah Accelerator")
    sub = ap.add_subparsers(dest="cmd")

    a = sub.add_parser("inspect", help="inspect a model artifact")
    a.add_argument("model"); a.add_argument("--json", action="store_true")
    a.set_defaults(fn=cmd_inspect)

    a = sub.add_parser("qualify", help="measure this machine")
    a.add_argument("model", nargs="?", default="")
    a.set_defaults(fn=cmd_qualify)

    a = sub.add_parser("requalify", help="force re-measurement")
    a.add_argument("model", nargs="?", default="")
    a.set_defaults(fn=cmd_qualify)

    a = sub.add_parser("plan", help="build an execution plan")
    a.add_argument("model")
    a.add_argument("--context", type=int, default=8192)
    a.add_argument("--concurrency", type=int, default=1)
    a.add_argument("--workload", default="auto",
                   choices=["auto", "general", "coding", "math", "json", "longctx"])
    a.add_argument("--json", action="store_true")
    a.set_defaults(fn=cmd_plan)

    a = sub.add_parser("selftest", help="prove exact expert execution")
    a.add_argument("model", nargs="?", default="")
    a.set_defaults(fn=cmd_selftest)

    a = sub.add_parser("bench", help="MEASURED per-block throughput")
    a.add_argument("model", nargs="?", default="")
    a.add_argument("--trace", default="")
    a.add_argument("--block", type=int, default=0)
    a.add_argument("--tokens", type=int, default=1000)
    a.add_argument("--slots", default="16,64,256,512")
    a.add_argument("--bank-mode", default="mmap", choices=["mmap", "ram"])
    a.add_argument("--json", default="")
    a.set_defaults(fn=cmd_bench)

    a = sub.add_parser("benchmark", help="measure the full-model reference path")
    a.add_argument("model")
    a.add_argument("--prompt", default="Reply with exactly: SABAH_OK")
    a.add_argument("--tokens", type=int, default=16)
    a.add_argument("--context", type=int, default=4096)
    a.add_argument("--threads", type=int, default=0)
    a.add_argument("--llama-cli", default="")
    a.add_argument("--json", default="")
    a.add_argument("--dry-run", action="store_true")
    a.set_defaults(fn=cmd_benchmark)

    a = sub.add_parser("serve", help="serve a local OpenAI-compatible API")
    a.add_argument("model")
    a.add_argument("--host", default="127.0.0.1")
    a.add_argument("--port", type=int, default=8080)
    a.add_argument("--backend-port", type=int, default=18080)
    a.add_argument("--context", type=int, default=4096)
    a.add_argument("--n-gpu-layers", type=int, default=99)
    a.add_argument("--backend", choices=["sabah", "reference"], default=None)
    a.add_argument("--hot-bytes", type=int, default=1 << 30,
                   help="VRAM budget for Sabah's expert hot tier")
    a.add_argument("--llama-server", default="")
    a.add_argument("--allow-reference", action="store_true",
                   help="required with --backend reference: stock llama.cpp expert execution")
    a.add_argument("--quiet", action="store_true")
    a.set_defaults(fn=cmd_serve)

    a = sub.add_parser("calibrate", help="re-measure the planner hit curve")
    a.add_argument("model", nargs="?", default="")
    a.add_argument("--trace", default="")
    a.add_argument("--json", default="")
    a.set_defaults(fn=cmd_calibrate)

    a = sub.add_parser("doctor", help="environment check")
    a.set_defaults(fn=cmd_doctor)

    a = sub.add_parser("status", help="show saved profiles")
    a.set_defaults(fn=cmd_status)

    args = ap.parse_args(argv)
    if not getattr(args, "fn", None):
        ap.print_help()
        return 1
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
