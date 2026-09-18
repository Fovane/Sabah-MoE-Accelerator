"""Local OpenAI-compatible server for Sabah.

Two backends, both a patched llama.cpp server behind this proxy:

``reference``  stock llama.cpp expert execution. The graph, weights and
               placement are the same as ``sabah``'s; only who executes the
               expert ``MUL_MAT_ID`` differs.
``sabah``      the same graph with every expert ``MUL_MAT_ID`` executed by
               Sabah's native runtime (exact routed experts, VRAM hot tier).

The backend actually running is reported by ``/health`` and is never inferred
from the planner's projection.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from sabah.core.model_inspector import inspect_model
from sabah.planner.planner import ExecClass, plan as build_plan
from sabah.tools.cli import HW_PATH, _load_or_qualify


# The pinned, patched llama.cpp build (see patches/llama.cpp/README.md). An
# unpinned development checkout is deliberately not a default candidate.
PINNED_LLAMA_BIN = os.environ.get(
    "SABAH_LLAMA_BIN", r"D:\sabah_scaling\llama-sabah-clean\build\bin\Release")
RUNTIME_LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "runtime", "cuda", "sabah_rt.dll" if os.name == "nt" else "libsabah_rt.so")
CORRECTNESS = ("RC4: structural PASS, MUL_MAT_ID op-exact vs float64 PASS; "
               "see docs/RC4_NUMERICAL_EQUIVALENCE_REPORT.md")
VERSION = "0.9.0-rc4"


def find_llama_server(explicit: str = "") -> str:
    candidates = [
        explicit,
        os.environ.get("SABAH_LLAMA_SERVER", ""),
        os.path.join(PINNED_LLAMA_BIN, "llama-server.exe" if os.name == "nt" else "llama-server"),
        shutil.which("llama-server") or "",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        "llama-server not found; pass --llama-server or set SABAH_LLAMA_SERVER")


def find_llama_cli(explicit: str = "") -> str:
    candidates = [explicit, os.environ.get("SABAH_LLAMA_CLI", ""),
                  os.path.join(PINNED_LLAMA_BIN, "llama-cli.exe" if os.name == "nt" else "llama-cli"),
                  shutil.which("llama-cli") or ""]
    for path in candidates:
        if path and os.path.isfile(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        "llama-cli not found; pass --llama-cli or set SABAH_LLAMA_CLI")


def read_runtime_status(path: str | None):
    """Live Sabah runtime counters written by sabah_rt (SABAH_LLAMA_STATUS_FILE)."""
    if not path or not os.path.exists(path):
        return None
    for _ in range(3):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            time.sleep(0.05)       # the runtime rewrites the file; retry a torn read
    return None


class _State:
    def __init__(self, model: str, profile, hw, execution: str, backend: str,
                 server_process: subprocess.Popen | None = None, status_file: str | None = None):
        self.model = os.path.abspath(model)
        self.model_name = os.path.basename(model)
        self.profile = profile
        self.hw = hw
        self.execution = execution
        self.backend = backend
        self.process = server_process
        self.status_file = status_file
        self.started_at = time.time()

    def health(self) -> dict:
        return {
            "status": "ok" if self.process is None or self.process.poll() is None else "failed",
            "service": "sabah",
            "version": VERSION,
            "execution": self.execution,
            "backend": self.backend,
            "architecture": self.profile.architecture,
            "model": self.model_name,
            "model_path": self.model,
            "correctness": CORRECTNESS,
            "measured_acceleration": "see docs/RC4_NUMERICAL_EQUIVALENCE_REPORT.md",
            # counters from the runtime itself: proof that Sabah, not stock
            # llama.cpp, is executing expert MUL_MAT_IDs (null for reference)
            "sabah_runtime": read_runtime_status(getattr(self, "status_file", None)),
        }


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "Sabah/" + VERSION

    @property
    def state(self) -> _State:
        return self.server.sabah_state  # type: ignore[attr-defined]

    @property
    def backend_port(self) -> int:
        return self.server.backend_port  # type: ignore[attr-defined]

    def log_message(self, fmt, *args):
        if getattr(self.server, "quiet", False):  # type: ignore[attr-defined]
            return
        super().log_message(fmt, *args)

    def _json(self, status: int, payload: dict):
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):  # noqa: N802
        path = urlsplit(self.path).path
        if path in ("/health", "/v1/health"):
            return self._json(200, self.state.health())
        if path == "/v1/models":
            return self._proxy("GET")
        self._json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def do_POST(self):  # noqa: N802
        path = urlsplit(self.path).path
        if path in ("/v1/chat/completions", "/v1/completions"):
            return self._proxy("POST")
        self._json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def _proxy(self, method: str):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if method == "POST" else None
        conn = http.client.HTTPConnection("127.0.0.1", self.backend_port, timeout=600)
        headers = {"Accept": self.headers.get("Accept", "application/json")}
        if body is not None:
            headers["Content-Type"] = self.headers.get("Content-Type", "application/json")
        try:
            conn.request(method, self.path, body=body, headers=headers)
            resp = conn.getresponse()
            stream = self.headers.get("Accept", "") == "text/event-stream"
            content_type = resp.getheader("Content-Type", "application/json")
            if stream or content_type.startswith("text/event-stream"):
                self.send_response(resp.status)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(("%X\r\n" % len(chunk)).encode("ascii"))
                    self.wfile.write(chunk)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                return
            data = resp.read()
            self.send_response(resp.status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (ConnectionError, OSError, socket.timeout) as exc:
            self._json(502, {"error": {"message": "reference backend unavailable: %s" % exc,
                                        "type": "server_error"}})
        finally:
            conn.close()


class SabahHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, state, backend_port, quiet=False):
        super().__init__(address, ProxyHandler)
        self.sabah_state = state
        self.backend_port = backend_port
        self.quiet = quiet


def _wait_backend(port: int, process: subprocess.Popen, timeout: float = 900.0):
    """Wait until llama-server has LOADED the model: it listens early and
    answers /health with 503 while loading."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError("llama-server exited during startup (%s)" % process.returncode)
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            conn.request("GET", "/health")
            if conn.getresponse().status == 200:
                return
        except OSError:
            pass
        time.sleep(1.0)
    raise TimeoutError("timed out waiting for llama-server on port %d" % port)


def backend_launch(backend: str, exe: str, model: str, backend_port: int, context: int,
                   n_gpu_layers: int, hot_bytes: int, parallel: int = 4,
                   validate: bool = False) -> tuple:
    """Command and environment for a backend. Both backends get identical
    graphs and op placement: experts stay in host memory (--cpu-moe) and every
    expert MUL_MAT_ID is sent to the GPU (GGML_OP_OFFLOAD_MIN_BATCH=1), so the
    only difference is who executes it."""
    cmd = [exe, "-m", model, "--host", "127.0.0.1", "--port", str(backend_port),
           "--ctx-size", str(context), "--n-gpu-layers", str(n_gpu_layers),
           "--cpu-moe", "--no-webui", "--parallel", str(parallel)]
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SABAH_LLAMA", "SABAH_RT_LIB", "GGML_OP_OFFLOAD"))}
    env["GGML_OP_OFFLOAD_MIN_BATCH"] = "1"
    if backend == "sabah":
        if not os.path.exists(RUNTIME_LIB):
            raise FileNotFoundError("Sabah runtime not built: %s" % RUNTIME_LIB)
        import tempfile
        status = os.path.join(tempfile.gettempdir(), "sabah_runtime_status_%d.json" % backend_port)
        if os.path.exists(status):
            os.remove(status)
        env.update(SABAH_LLAMA="1", SABAH_RT_LIB=RUNTIME_LIB,
                   SABAH_LLAMA_HOT_BYTES=str(hot_bytes), SABAH_LLAMA_STATUS_FILE=status)
        if validate:
            # self-validating mode: float64 recomputation of 2 sampled outputs
            # per expert op, and byte verification of every fetch (+ 1/64 hits)
            env.update(SABAH_LLAMA_SELFCHECK="2", SABAH_LLAMA_VERIFY_BYTES="fetch")
    elif backend != "reference":
        raise ValueError("backend must be 'sabah' or 'reference'")
    return cmd, env


def serve(model: str, host: str = "127.0.0.1", port: int = 8080,
          backend_port: int = 18080, context: int = 4096,
          llama_server: str = "", n_gpu_layers: int = 99,
          backend: str = "sabah", hot_bytes: int = 1 << 30, parallel: int = 4,
          validate: bool = False, allow_reference: bool = False, quiet: bool = False) -> int:
    profile = inspect_model(model)
    if not profile.supported:
        raise ValueError("unsupported model; Sabah refuses to serve it")
    hw, fresh = _load_or_qualify(model)
    if fresh:
        hw.save(HW_PATH())
    projection = build_plan(profile, hw, context=context, concurrency=1)
    if backend == "reference" and not allow_reference:
        raise RuntimeError("backend=reference serves stock llama.cpp expert execution; "
                           "pass --allow-reference to confirm that is intended")
    exe = find_llama_server(llama_server)
    cmd, env = backend_launch(backend, exe, model, backend_port, context, n_gpu_layers, hot_bytes,
                              parallel, validate and backend == "sabah")
    execution = "SABAH_NATIVE_MUL_MAT_ID" if backend == "sabah" else "REFERENCE"
    if not quiet:
        print("execution  : %s" % execution)
        print("planner    : %s (%s)" % (projection.exec_class.value,
                                      "PROJECTED" if projection.recommended else "fallback"))
        print("backend    : %s" % exe)
        print("API        : http://%s:%d/v1" % (host, port))
    proc = subprocess.Popen(cmd, cwd=os.path.dirname(exe), env=env,
                            stdout=subprocess.DEVNULL if quiet else None,
                            stderr=subprocess.STDOUT if quiet else None)
    try:
        _wait_backend(backend_port, proc)
        state = _State(model, profile, hw, execution,
                       "llama.cpp+sabah" if backend == "sabah" else "llama.cpp", proc,
                       status_file=env.get("SABAH_LLAMA_STATUS_FILE"))
        httpd = SabahHTTPServer((host, port), state, backend_port, quiet)
        try:
            httpd.serve_forever()
        finally:
            httpd.server_close()
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sabah serve")
    ap.add_argument("model")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--backend-port", type=int, default=18080)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--llama-server", default="")
    ap.add_argument("--n-gpu-layers", type=int, default=99)
    ap.add_argument("--backend", choices=["sabah", "reference"], default="sabah")
    ap.add_argument("--hot-bytes", type=int, default=1 << 30)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--validate", action="store_true",
                    help="sabah backend: in-runtime float64 self-check and byte verification")
    ap.add_argument("--allow-reference", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    return serve(**vars(args))


if __name__ == "__main__":
    raise SystemExit(main())
