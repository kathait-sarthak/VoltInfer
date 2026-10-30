import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from engine.proxy import VoltInferProxyServer


class MockLLMHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    requests = []
    response_body = b""
    response_headers = {}
    stream_chunks = None
    first_chunk_sent = threading.Event()
    continue_stream = threading.Event()

    def do_GET(self):
        body = b'{"object":"list","data":[{"id":"test-model"}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        type(self).requests.append(
            {
                "path": self.path,
                "body": self.rfile.read(length),
                "authorization": self.headers.get("Authorization"),
                "openai-beta": self.headers.get("OpenAI-Beta"),
            }
        )
        if type(self).stream_chunks is not None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for index, chunk in enumerate(type(self).stream_chunks):
                self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
                if index == 0:
                    type(self).first_chunk_sent.set()
                    type(self).continue_stream.wait(timeout=3)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return
        body = type(self).response_body or json.dumps(
            {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": "real upstream"}}],
                "usage": {"completion_tokens": 4},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for name, value in type(self).response_headers.items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _post(url, payload, headers=None):
    request_headers = {"Content-Type": "application/json"}
    request_headers.update(headers or {})
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers=request_headers,
    )
    return urlopen(request, timeout=5)


def test_proxy_forwards_real_response_and_records_private_metrics(tmp_path):
    MockLLMHandler.requests = []
    MockLLMHandler.response_body = b'{"id":"chatcmpl-fixed","usage":{"completion_tokens":4}}'
    MockLLMHandler.response_headers = {"X-RateLimit-Remaining": "7"}
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), MockLLMHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()

    proxy = VoltInferProxyServer(
        host="127.0.0.1",
        port=_free_port(),
        upstream_url=f"http://127.0.0.1:{upstream.server_port}/v1",
        upstream_api_key="test-key",
        learning_db_path=str(tmp_path / "learning.sqlite3"),
    )
    proxy.start()
    base_url = f"http://127.0.0.1:{proxy.port}"
    try:
        for _ in range(50):
            try:
                response = _post(
                    f"{base_url}/v1/chat/completions",
                    {"model": "test-model", "messages": [{"role": "user", "content": "secret prompt"}]},
                    headers={"OpenAI-Beta": "responses=2025-01-01"},
                )
                break
            except OSError:
                time.sleep(0.02)
        else:
            raise AssertionError("VoltInfer proxy did not start")

        with response:
            request_id = response.headers["X-VoltInfer-Request-ID"]
            assert response.headers["X-RateLimit-Remaining"] == "7"
            assert response.read() == MockLLMHandler.response_body

        captured = MockLLMHandler.requests[-1]
        assert captured["path"] == "/v1/chat/completions"
        assert captured["authorization"] == "Bearer test-key"
        assert captured["openai-beta"] == "responses=2025-01-01"
        assert json.loads(captured["body"])["messages"][0]["content"] == "secret prompt"

        feedback = _post(
            f"{base_url}/v1/feedback",
            {"request_id": request_id, "rating": 5},
        )
        with feedback:
            assert json.loads(feedback.read())["status"] == "feedback_recorded"

        with urlopen(f"{base_url}/v1/learning/status", timeout=5) as response:
            learning_status = json.loads(response.read())
        assert learning_status["request_count"] == 1
        assert learning_status["feedback_count"] == 1
        assert learning_status["prompt_storage"] is False
        assert learning_status["credential_storage"] is False

        with urlopen(f"{base_url}/v1/models", timeout=5) as response:
            assert json.loads(response.read())["data"][0]["id"] == "test-model"
    finally:
        proxy.stop()
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)


def test_adaptive_request_uses_profile_and_does_not_store_prompt(tmp_path):
    MockLLMHandler.requests = []
    MockLLMHandler.response_body = b'{"id":"chatcmpl-adaptive","usage":{"completion_tokens":4}}'
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), MockLLMHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()

    proxy = VoltInferProxyServer(
        host="127.0.0.1",
        port=_free_port(),
        upstream_url=f"http://127.0.0.1:{upstream.server_port}/v1",
        adaptive=False,
        learning_db_path=str(tmp_path / "learning.sqlite3"),
    )
    proxy.start()
    try:
        with _post(
            f"http://127.0.0.1:{proxy.port}/v1/chat/completions",
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "not stored"}],
                "max_tokens": 80,
                "voltinfer": {"adaptive": True, "profile": "balanced"},
            },
        ) as response:
            assert response.headers["X-VoltInfer-Profile"] == "balanced"
            request_id = response.headers["X-VoltInfer-Request-ID"]
            response.read()

        sent = json.loads(MockLLMHandler.requests[-1]["body"])
        assert sent["max_tokens"] == 60
        assert "voltinfer" not in sent

        feedback = _post(
            f"http://127.0.0.1:{proxy.port}/v1/feedback",
            {"request_id": request_id, "rating": 4},
        )
        feedback.close()
        assert len(MockLLMHandler.requests) == 1
    finally:
        proxy.stop()
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)


def test_proxy_returns_bad_gateway_when_upstream_is_unavailable(tmp_path):
    proxy = VoltInferProxyServer(
        host="127.0.0.1",
        port=_free_port(),
        upstream_url="http://127.0.0.1:1/v1",
        learning_db_path=str(tmp_path / "learning.sqlite3"),
    )
    proxy.start()
    try:
        try:
            _post(
                f"http://127.0.0.1:{proxy.port}/v1/chat/completions",
                {"model": "test-model", "messages": [{"role": "user", "content": "hello"}]},
            )
        except HTTPError as error:
            assert error.code == 502
            assert "configured LLM upstream" in error.read().decode()
        else:
            raise AssertionError("unavailable upstream should return HTTP 502")
    finally:
        proxy.stop()


def test_proxy_rejects_an_upstream_url_that_loops_back_to_itself():
    with pytest.raises(ValueError, match="must not point back"):
        VoltInferProxyServer(
            host="127.0.0.1",
            port=8000,
            upstream_url="http://localhost:8000/v1",
        )


def test_proxy_streams_server_sent_events_incrementally(tmp_path):
    MockLLMHandler.requests = []
    MockLLMHandler.stream_chunks = [
        b'data: {"id":"stream-1","choices":[{"delta":{"content":"hello"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    MockLLMHandler.first_chunk_sent = threading.Event()
    MockLLMHandler.continue_stream = threading.Event()
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), MockLLMHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    proxy = VoltInferProxyServer(
        host="127.0.0.1",
        port=_free_port(),
        upstream_url=f"http://127.0.0.1:{upstream.server_port}/v1",
        learning_db_path=str(tmp_path / "learning.sqlite3"),
    )
    proxy.start()
    try:
        response = _post(
            f"http://127.0.0.1:{proxy.port}/v1/chat/completions",
            {"model": "test-model", "stream": True},
        )
        first_line = response.readline()
        assert b'data: {"id":"stream-1"' in first_line
        assert MockLLMHandler.first_chunk_sent.is_set()
        MockLLMHandler.continue_stream.set()
        rest = response.read()
        assert b"data: [DONE]" in rest
        response.close()
    finally:
        MockLLMHandler.continue_stream.set()
        MockLLMHandler.stream_chunks = None
        proxy.stop()
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)
