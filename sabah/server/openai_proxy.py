"""Local OpenAI-compatible server for Sabah.

The full-model Sabah backend is deliberately not faked here.  Until the
attention/KV integration is available, the server runs the installed
llama.cpp reference engine and exposes its mode as ``REFERENCE`` in health and
logs.  This makes the user flow usable while keeping the exact-routing claim
honest.  The proxy boundary is also the integration seam for the future
residency backend.
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


def find_llama_server(explicit: str = "") -> str:
    candidates = [
        explicit,
        os.environ.get("SABAH_LLAMA_SERVER", ""),
        shutil.which("llama-server") or "",
        r"D:\llama-glm53\build\bin\Release\llama-server.exe",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        "llama-server not found; pass --llama-server or set SABAH_LLAMA_SERVER")


def find_llama_cli(explicit: str = "") -> str:
    candidates = [explicit, os.environ.get("SABAH_LLAMA_CLI", ""),
                  shutil.which("llama-cli") or "",
                  r"D:\llama-glm53\build\bin\Release\llama-cli.exe"]
    for path in candidates:
        if path and os.path.isfile(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        "llama-cli not found; pass --llama-cli or set SABAH_LLAMA_CLI")


class _State:
    def __init__(self, model: str, profile, hw, execution: str, backend: str,
                 server_process: subprocess.Popen | None = None):
        self.model = os.path.abspath(model)
        self.model_name = os.path.basename(model)
        self.profile = profile
        self.hw = hw
        self.execution = execution
        self.backend = backend
        self.process = server_process
        self.started_at = time.time()

    def health(self) -> dict:
        return {
            "status": "ok" if self.process is None or self.process.poll() is None else "failed",
            "service": "sabah",
            "version": "0.9.0-rc1",
            "execution": self.execution,
            "backend": self.backend,
            "architecture": self.profile.architecture,
            "model": self.model_name,
            "model_path": self.model,
            "correctness": "BLOCK_PASS_FULL_MODEL_UNVALIDATED",
            "measured_acceleration": "unavailable",
        }


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "Sabah/0.9.0-rc1"

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


def _wait_backend(port: int, process: subprocess.Popen, timeout: float = 120.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError("llama-server exited during startup (%s)" % process.returncode)
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError("timed out waiting for llama-server on port %d" % port)


def serve(model: str, host: str = "127.0.0.1", port: int = 8080,
          backend_port: int = 18080, context: int = 4096,
          llama_server: str = "", n_gpu_layers: int = 0,
          allow_reference: bool = False, quiet: bool = False) -> int:
    profile = inspect_model(model)
    if not profile.supported:
        raise ValueError("unsupported model; Sabah refuses to serve it")
    hw, fresh = _load_or_qualify(model)
    if fresh:
        hw.save(HW_PATH())
    projection = build_plan(profile, hw, context=context, concurrency=1)

    # The full-model residency integration is not complete.  Never label a
    # llama.cpp process as GPU_HOT_TIER just because the planner projected it.
    if not allow_reference:
        raise RuntimeError(
            "full-model Sabah backend is not ready; use --allow-reference to "
            "serve through the exact llama.cpp reference engine")
    exe = find_llama_server(llama_server)
    cmd = [exe, "-m", model, "--host", "127.0.0.1", "--port", str(backend_port),
           "--ctx-size", str(context), "--n-gpu-layers", str(n_gpu_layers),
           "--cpu-moe", "--no-webui"]
    if not quiet:
        print("execution  : REFERENCE")
        print("planner    : %s (%s)" % (projection.exec_class.value,
                                      "PROJECTED" if projection.recommended else "fallback"))
        print("backend    : %s" % exe)
        print("API        : http://%s:%d/v1" % (host, port))
    proc = subprocess.Popen(cmd, cwd=os.path.dirname(exe),
                            stdout=subprocess.DEVNULL if quiet else None,
                            stderr=subprocess.STDOUT if quiet else None)
    try:
        _wait_backend(backend_port, proc)
        state = _State(model, profile, hw, "REFERENCE", "llama.cpp", proc)
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
    ap.add_argument("--n-gpu-layers", type=int, default=0)
    ap.add_argument("--allow-reference", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    return serve(**vars(args))


if __name__ == "__main__":
    raise SystemExit(main())

