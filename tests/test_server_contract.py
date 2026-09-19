from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

from sabah.server.openai_proxy import SabahHTTPServer, ProxyHandler
from sabah.tools.benchmark import _parse_tok_s


class _Backend(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = json.dumps({"object": "list", "data": []}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


def test_server_health_and_models_proxy():
    backend = ThreadingHTTPServer(("127.0.0.1", 0), _Backend)
    bt = threading.Thread(target=backend.serve_forever, daemon=True)
    bt.start()
    state = type("State", (), {"health": lambda self: {"status": "ok"}})()
    server = SabahHTTPServer(("127.0.0.1", 0), state,
                             backend.server_address[1], quiet=True)
    st = threading.Thread(target=server.serve_forever, daemon=True)
    st.start()
    try:
        base = "http://127.0.0.1:%d" % server.server_address[1]
        with urlopen(base + "/health") as r:
            assert json.load(r)["status"] == "ok"
        with urlopen(base + "/v1/models") as r:
            assert json.load(r)["object"] == "list"
    finally:
        server.shutdown()
        server.server_close()
        backend.shutdown()
        backend.server_close()


def test_reference_parser_accepts_llama_simple_io_line():
    assert _parse_tok_s("[ Prompt: 2.0 t/s | Generation: 7.5 t/s ]") == 7.5



def test_backends_share_placement_and_differ_only_in_executor(monkeypatch, tmp_path):
    from sabah.server import openai_proxy
    lib = tmp_path / "sabah_rt.dll"
    lib.write_bytes(b"")
    monkeypatch.setattr(openai_proxy, "RUNTIME_LIB", str(lib))
    monkeypatch.setenv("SABAH_LLAMA", "stale")            # must not leak into reference
    monkeypatch.setenv("GGML_OP_OFFLOAD_MIN_BATCH", "32")  # must be overridden
    args = ("llama-server", "m.gguf", 18080, 1024, 99, 1 << 30)
    ref_cmd, ref_env = openai_proxy.backend_launch("reference", *args)
    sab_cmd, sab_env = openai_proxy.backend_launch("sabah", *args)
    assert ref_cmd == sab_cmd and "--cpu-moe" in ref_cmd
    # every expert MUL_MAT_ID must reach the GPU path, or Sabah is bypassed
    assert ref_env["GGML_OP_OFFLOAD_MIN_BATCH"] == sab_env["GGML_OP_OFFLOAD_MIN_BATCH"] == "1"
    assert "SABAH_LLAMA" not in ref_env
    assert sab_env["SABAH_LLAMA"] == "1" and sab_env["SABAH_RT_LIB"] == str(lib)
    diff = {k for k in set(ref_env) | set(sab_env) if ref_env.get(k) != sab_env.get(k)}
    assert diff == {"SABAH_LLAMA", "SABAH_RT_LIB", "SABAH_LLAMA_HOT_BYTES", "SABAH_LLAMA_STATUS_FILE"}


class _SlowBackend(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        import time
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        time.sleep(1.5)
        body = json.dumps({"ok": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


def _post_through_proxy(backend_timeout):
    backend = ThreadingHTTPServer(("127.0.0.1", 0), _SlowBackend)
    threading.Thread(target=backend.serve_forever, daemon=True).start()
    state = type("State", (), {"health": lambda self: {"status": "ok"}})()
    server = SabahHTTPServer(("127.0.0.1", 0), state, backend.server_address[1], quiet=True,
                             backend_timeout=backend_timeout)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        req = Request("http://127.0.0.1:%d/v1/chat/completions" % server.server_address[1],
                      data=b"{}", headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=30) as r:
                return r.status
        except Exception as e:  # HTTPError carries the status
            return getattr(e, "code", None)
    finally:
        server.shutdown(); server.server_close(); backend.shutdown(); backend.server_close()


def test_proxy_waits_for_a_slow_backend_by_default():
    # v1 validation: a fixed 600 s proxy timeout turned slow-but-correct
    # generations into 502s while the backend kept working
    assert _post_through_proxy(None) == 200


def test_proxy_timeout_is_explicit_opt_in():
    assert _post_through_proxy(0.5) == 502
