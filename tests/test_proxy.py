"""
Unit tests for VoltInfer HTTP Proxy Server and API Endpoints.
"""

import json
import time
import socket
import urllib.request
import urllib.error
import pytest
from engine.proxy import VoltInferProxyServer


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def proxy_server():
    port = get_free_port()
    server = VoltInferProxyServer(host="127.0.0.1", port=port)
    server.start(block=False)
    time.sleep(0.3)
    base_url = f"http://127.0.0.1:{port}"
    yield base_url
    server.stop()


def test_get_telemetry(proxy_server):
    url = f"{proxy_server}/v1/telemetry"
    with urllib.request.urlopen(url) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "online"
        assert "telemetry" in data
        assert "temperature_c" in data["telemetry"]
        assert "thermal_headroom" in data["telemetry"]


def test_get_model_info(proxy_server):
    url = f"{proxy_server}/v1/model/info"
    with urllib.request.urlopen(url) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "online"
        assert "model" in data
        assert "backend_type" in data["model"]


def test_get_config(proxy_server):
    url = f"{proxy_server}/v1/config"
    with urllib.request.urlopen(url) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert "target_max_temp" in data
        assert "max_layers" in data


def test_get_dashboard_assets(proxy_server):
    # Root dashboard HTML
    with urllib.request.urlopen(f"{proxy_server}/") as resp:
        assert resp.status == 200
        assert "text/html" in resp.headers.get("Content-Type", "")
        content = resp.read().decode("utf-8")
        assert "VoltInfer" in content

    # CSS
    with urllib.request.urlopen(f"{proxy_server}/styles.css") as resp:
        assert resp.status == 200
        assert "text/css" in resp.headers.get("Content-Type", "")

    # JS
    with urllib.request.urlopen(f"{proxy_server}/app.js") as resp:
        assert resp.status == 200
        assert "javascript" in resp.headers.get("Content-Type", "")


def test_post_chat_completions(proxy_server):
    url = f"{proxy_server}/v1/chat/completions"
    payload = {
        "model": "volt-infer-adaptive-v1",
        "messages": [
            {"role": "user", "content": "What is closed-loop hardware feedback?"}
        ],
        "max_tokens": 50
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["object"] == "chat.completion"
        assert len(data["choices"]) > 0
        assert "volt_infer_telemetry" in data
        assert "executed_layers" in data["volt_infer_telemetry"]
        assert "energy_saved_pct" in data["volt_infer_telemetry"]


def test_post_standard_completions(proxy_server):
    url = f"{proxy_server}/v1/completions"
    payload = {
        "model": "volt-infer-adaptive-v1",
        "prompt": "Explain silicon thermal throttling",
        "max_tokens": 40
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["object"] == "text_completion"
        assert len(data["choices"]) > 0
        assert "volt_infer_telemetry" in data


def test_post_telemetry_simulation(proxy_server):
    url = f"{proxy_server}/v1/telemetry/simulate"
    
    # 1. Simulate THERMAL_THROTTLING
    req = urllib.request.Request(
        url,
        data=json.dumps({"profile": "THERMAL_THROTTLING"}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["status"] == "simulation_updated"
        assert data["telemetry"]["simulation_mode"] == "THERMAL_THROTTLING"

    # 2. Clear simulation
    req = urllib.request.Request(
        url,
        data=json.dumps({}).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["telemetry"]["simulation_mode"] == "REALTIME"


def test_options_cors(proxy_server):
    req = urllib.request.Request(f"{proxy_server}/v1/chat/completions", method="OPTIONS")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 204
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"
