"""
VoltInfer HTTP Proxy Server
Exposes OpenAI-compatible API endpoints (/v1/chat/completions, /v1/completions),
real-time telemetry endpoints (/v1/telemetry), hardware stress control (/v1/telemetry/simulate),
and serves the interactive visual Web Dashboard.
"""

import json
import os
import time
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
from typing import Dict, Any, Optional

try:
    from .telemetry import TelemetryMonitor
    from .layer_skipper import DynamicLayerModulator
    from .model_loader import RealModelBackend
except ImportError:
    from telemetry import TelemetryMonitor
    from layer_skipper import DynamicLayerModulator
    from model_loader import RealModelBackend

logger = logging.getLogger("VoltInfer.Proxy")

# Path to dashboard static assets
DASHBOARD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dashboard"))


class VoltInferHTTPRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for VoltInfer OpenAI-compatible proxy server."""

    # Server components injected by VoltInferProxyServer
    telemetry_monitor: TelemetryMonitor = None
    layer_modulator: DynamicLayerModulator = None

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default HTTP noise, log cleanly."""
        logger.info(f"{self.address_string()} - {format % args}")

    def _send_json(self, data: Dict[str, Any], status_code: int = 200) -> None:
        """Send JSON HTTP response with CORS headers."""
        body = json.dumps(data, indent=2).encode('utf-8')
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, filepath: str, mime_type: str) -> None:
        """Send static file response."""
        if not os.path.exists(filepath):
            self._send_json({"error": f"File not found: {os.path.basename(filepath)}"}, 404)
            return
        
        with open(filepath, "rb") as f:
            content = f.read()

        self.send_response(200)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight requests."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    # ------------------------------------------------------------------
    # GET Endpoints
    # ------------------------------------------------------------------

    def do_GET(self) -> None:
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        # Telemetry API Endpoint
        if path == "/v1/telemetry":
            state = self.telemetry_monitor.get_state()
            self._send_json({
                "status": "online",
                "telemetry": state.to_dict()
            })
            return

        # Model Info Endpoint
        if path == "/v1/model/info":
            info = self.layer_modulator.model_backend.get_info()
            self._send_json({
                "status": "online",
                "model": info
            })
            return

        # Server Config Endpoint
        if path == "/v1/config":
            self._send_json({
                "target_max_temp": self.layer_modulator.target_max_temp,
                "max_layers": self.layer_modulator.MAX_LAYERS,
                "min_layers": self.layer_modulator.MIN_LAYERS,
                "max_kv_heads": self.layer_modulator.MAX_KV_HEADS,
                "min_kv_heads": self.layer_modulator.MIN_KV_HEADS
            })
            return

        # Serve Web Dashboard Static Assets
        if path == "/" or path == "/index.html":
            index_path = os.path.join(DASHBOARD_DIR, "index.html")
            self._send_file(index_path, "text/html; charset=utf-8")
            return
        elif path == "/styles.css":
            css_path = os.path.join(DASHBOARD_DIR, "styles.css")
            self._send_file(css_path, "text/css; charset=utf-8")
            return
        elif path == "/app.js":
            js_path = os.path.join(DASHBOARD_DIR, "app.js")
            self._send_file(js_path, "application/javascript; charset=utf-8")
            return

        self._send_json({"error": "Endpoint not found"}, 404)

    # ------------------------------------------------------------------
    # POST Endpoints
    # ------------------------------------------------------------------

    def do_POST(self) -> None:
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        # Read JSON Body
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            body = json.loads(post_data.decode("utf-8")) if post_data else {}
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON format in request body"}, 400)
            return

        # 1. OpenAI Compatible Chat Completion Endpoint
        if path in ("/v1/chat/completions", "/chat/completions"):
            self._handle_chat_completions(body)
            return

        # 2. OpenAI Compatible Standard Completion Endpoint
        if path in ("/v1/completions", "/completions", "/generate"):
            self._handle_standard_completions(body)
            return

        # 3. Telemetry Simulation Override Endpoint
        if path in ("/v1/telemetry/simulate", "/simulate"):
            self._handle_telemetry_simulation(body)
            return

        self._send_json({"error": f"POST Endpoint '{path}' not found"}, 404)

    # ------------------------------------------------------------------
    # Request Handlers
    # ------------------------------------------------------------------

    def _handle_chat_completions(self, body: Dict[str, Any]) -> None:
        """Process OpenAI-compatible chat completion request."""
        messages = body.get("messages", [])
        if not messages:
            prompt = body.get("prompt", "Hello VoltInfer")
        else:
            # Extract last user message or concat conversation
            user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
            prompt = user_msgs[-1] if user_msgs else messages[-1].get("content", "Hello")

        max_tokens = body.get("max_tokens", 120)
        state = self.telemetry_monitor.get_state()

        result = self.layer_modulator.execute_inference(prompt, state, max_tokens=max_tokens)
        res_dict = result.to_dict()

        # Format as standard OpenAI ChatCompletion payload
        response_payload = {
            "id": f"chatcmpl-volt-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "volt-infer-adaptive-v1"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.generated_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.generated_tokens,
                "total_tokens": result.total_tokens
            },
            # Extra VoltInfer Telemetry Metadata Block
            "volt_infer_telemetry": {
                "execution_mode": result.mode,
                "executed_layers": result.executed_layers,
                "total_layers": result.total_layers,
                "quantization": result.quantization,
                "active_kv_heads": result.active_kv_heads,
                "early_exit": result.early_exit,
                "prompt_entropy": result.prompt_entropy,
                "hardware_pressure": result.hardware_pressure,
                "latency_sec": result.latency_sec,
                "baseline_latency_sec": result.baseline_latency_sec,
                "power_draw_w": result.power_w,
                "energy_joules": result.energy_joules,
                "joules_saved": result.joules_saved,
                "energy_saved_pct": result.energy_saved_pct,
                "tokens_per_sec": result.tokens_per_sec,
                "silicon_temp_c": state.temperature_c,
                "thermal_headroom_pct": round(state.thermal_headroom * 100, 1)
            }
        }
        self._send_json(response_payload)

    def _handle_standard_completions(self, body: Dict[str, Any]) -> None:
        """Process standard prompt completion request."""
        prompt = body.get("prompt", "Hello")
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else "Hello"

        max_tokens = body.get("max_tokens", 120)
        state = self.telemetry_monitor.get_state()

        result = self.layer_modulator.execute_inference(prompt, state, max_tokens=max_tokens)
        res_dict = result.to_dict()

        response_payload = {
            "id": f"cmpl-volt-{int(time.time() * 1000)}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": body.get("model", "volt-infer-adaptive-v1"),
            "choices": [
                {
                    "text": result.generated_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.generated_tokens,
                "total_tokens": result.total_tokens
            },
            "volt_infer_telemetry": res_dict
        }
        self._send_json(response_payload)

    def _handle_telemetry_simulation(self, body: Dict[str, Any]) -> None:
        """Handle synthetic hardware profile injection."""
        profile = body.get("profile")
        temp_c = body.get("temperature_c")
        gpu_util = body.get("gpu_utilization")

        if profile:
            self.telemetry_monitor.set_power_profile(profile)
        elif temp_c is not None:
            self.telemetry_monitor.set_simulated_temperature(float(temp_c))
        elif gpu_util is not None:
            self.telemetry_monitor.set_simulated_gpu_load(float(gpu_util))
        else:
            self.telemetry_monitor.clear_simulation()

        current_state = self.telemetry_monitor.get_state().to_dict()
        self._send_json({
            "status": "simulation_updated",
            "telemetry": current_state
        })


class VoltInferProxyServer:
    """
    Main Proxy Server launcher managing TelemetryMonitor, DynamicLayerModulator,
    RealModelBackend, and ThreadingHTTPServer.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8000, model_path: Optional[str] = None):
        self.host = host
        self.port = port
        self.telemetry_monitor = TelemetryMonitor(sample_interval=0.15)
        
        # Initialize model backend (Real weights or simulation fallback)
        path = model_path or os.environ.get("VOLT_MODEL_PATH")
        self.model_backend = RealModelBackend(path)
        self.layer_modulator = DynamicLayerModulator(model_backend=self.model_backend)
        self.server: Optional[ThreadingHTTPServer] = None

    def start(self, block: bool = False) -> None:
        """Start telemetry monitor and HTTP proxy server."""
        self.telemetry_monitor.start()

        # Bind shared instances to RequestHandler
        VoltInferHTTPRequestHandler.telemetry_monitor = self.telemetry_monitor
        VoltInferHTTPRequestHandler.layer_modulator = self.layer_modulator

        self.server = ThreadingHTTPServer((self.host, self.port), VoltInferHTTPRequestHandler)
        logger.info(f"VoltInfer Proxy Server listening on http://{self.host}:{self.port}")
        if self.model_backend.is_loaded:
            logger.info(f"Active Model: '{self.model_backend.model_name}' ({self.model_backend.backend_type})")
        else:
            logger.info("Active Model: VoltInfer Adaptive Engine (Simulated Prototype)")

        if block:
            try:
                self.server.serve_forever()
            except KeyboardInterrupt:
                self.stop()
        else:
            import threading
            server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            server_thread.start()

    def stop(self) -> None:
        """Shutdown proxy server and background threads cleanly."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        self.telemetry_monitor.stop()
        logger.info("VoltInfer Proxy Server shut down successfully.")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    
    parser = argparse.ArgumentParser(description="VoltInfer Proxy Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--model-path", "-m", type=str, default=None, help="Path to GGUF or PyTorch model file (e.g. E:/qwen4b.gguf)")
    args = parser.parse_args()

    proxy = VoltInferProxyServer(host=args.host, port=args.port, model_path=args.model_path)
    print(f"Starting VoltInfer Proxy Server on http://{args.host}:{args.port}...")
    proxy.start(block=False)
    time.sleep(1.0)
    print("Proxy is running live. Press Ctrl+C to terminate.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        proxy.stop()
