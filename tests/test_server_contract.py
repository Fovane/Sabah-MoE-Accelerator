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

