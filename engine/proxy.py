"""
VoltInfer HTTP Proxy Server
Exposes OpenAI-compatible API endpoints (/v1/chat/completions, /v1/completions),
real-time telemetry endpoints (/v1/telemetry), hardware stress control (/v1/telemetry/simulate),
and serves the interactive visual Web Dashboard.
"""

import json
import os
import sqlite3
import time
import uuid
import logging
import http.client
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
from typing import Dict, Any, Optional

try:
    from .learning import LearningStore, PROFILE_CAPS
    from .telemetry import TelemetryMonitor
    from .layer_skipper import DynamicLayerModulator
    from .model_loader import RealModelBackend
    from .upstream import OpenAICompatibleUpstream
except ImportError:
    from learning import LearningStore, PROFILE_CAPS
    from telemetry import TelemetryMonitor
    from layer_skipper import DynamicLayerModulator
    from model_loader import RealModelBackend
    from upstream import OpenAICompatibleUpstream

logger = logging.getLogger("VoltInfer.Proxy")

# Path to dashboard static assets
DASHBOARD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dashboard"))


class VoltInferHTTPRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for VoltInfer OpenAI-compatible proxy server."""

    def __init__(self, request, client_address, server):
        self.telemetry_monitor = server.telemetry_monitor
        self.layer_modulator = server.layer_modulator
        self.upstream = server.upstream
        self.learning_store = server.learning_store
        self.adaptive_enabled = server.adaptive_enabled
        super().__init__(request, client_address, server)

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
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization, OpenAI-Organization, OpenAI-Project, OpenAI-Beta, Idempotency-Key",
        )
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
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization, OpenAI-Organization, OpenAI-Project, OpenAI-Beta, Idempotency-Key",
        )
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
            if self.upstream:
                info = {
                    "upstream_configured": True,
                    "backend_type": "OPENAI_COMPATIBLE_UPSTREAM",
                    "upstream_url": self.upstream.base_url,
                    "adaptive_profiles": list(PROFILE_CAPS),
                }
            else:
                info = self.layer_modulator.model_backend.get_info()
            self._send_json({
                "status": "online",
                "model": info
            })
            return

        if path == "/v1/learning/status":
            if self.learning_store:
                try:
                    self._send_json(self.learning_store.get_status())
                except sqlite3.Error as error:
                    logger.error("Could not read local learning status: %s", error)
                    self._send_json({"error": "Could not read local learning data"}, 503)
            else:
                self._send_json({
                    "enabled": False,
                    "prompt_storage": False,
                    "credential_storage": False,
                })
            return

        # Server Config Endpoint
        if path == "/v1/config":
            self._send_json({
                "target_max_temp": self.layer_modulator.target_max_temp,
                "max_layers": self.layer_modulator.MAX_LAYERS,
                "min_layers": self.layer_modulator.MIN_LAYERS,
                "max_kv_heads": self.layer_modulator.MAX_KV_HEADS,
                "min_kv_heads": self.layer_modulator.MIN_KV_HEADS,
                "upstream_configured": self.upstream is not None,
                "adaptive_enabled": self.adaptive_enabled,
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

        if self.upstream and path.startswith("/v1/"):
            self._forward_upstream("GET", b"")
            return

        self._send_json({"error": "Endpoint not found"}, 404)

    # ------------------------------------------------------------------
    # POST Endpoints
    # ------------------------------------------------------------------

    def do_POST(self) -> None:
        parsed_url = urlparse(self.path)
        path = parsed_url.path

        # Read JSON Body
        try:
            content_length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            self._send_json({"error": "Invalid Content-Length"}, 400)
            return
        if content_length < 0 or content_length > 16 * 1024 * 1024:
            self._send_json({"error": "Request body must be no larger than 16 MiB"}, 413)
            return
        post_data = self.rfile.read(content_length) if content_length > 0 else b"{}"

        if path == "/v1/feedback":
            self._handle_feedback(post_data)
            return

        if self.upstream and (
            path.startswith("/v1/")
            or path in ("/chat/completions", "/completions")
        ) and path != "/v1/telemetry/simulate":
            self._forward_upstream("POST", post_data)
            return

        try:
            body = json.loads(post_data.decode("utf-8")) if post_data else {}
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON format in request body"}, 400)
            return
        if not isinstance(body, dict):
            self._send_json({"error": "Request body must be a JSON object"}, 400)
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

    def _handle_feedback(self, raw_body: bytes) -> None:
        if not self.learning_store:
            self._send_json({"error": "Local learning is disabled"}, 503)
            return
        try:
            body = json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json({"error": "Invalid JSON format in request body"}, 400)
            return
        if not isinstance(body, dict):
            self._send_json({"error": "Request body must be a JSON object"}, 400)
            return
        request_id = body.get("request_id")
        if not isinstance(request_id, str) or not request_id or len(request_id) > 256:
            self._send_json({"error": "request_id is required and must be at most 256 characters"}, 400)
            return
        try:
            self.learning_store.add_feedback(request_id, body.get("rating"))
        except ValueError as error:
            self._send_json({"error": str(error)}, 400)
            return
        except LookupError as error:
            self._send_json({"error": str(error)}, 404)
            return
        except sqlite3.Error as error:
            logger.error("Could not save local feedback: %s", error)
            self._send_json({"error": "Could not save feedback locally"}, 503)
            return
        self._send_json({"status": "feedback_recorded", "request_id": request_id})

    def _forward_upstream(self, method: str, raw_body: bytes) -> None:
        request_id = uuid.uuid4().hex
        route = urlparse(self.path).path
        inference_route = route in (
            "/v1/chat/completions",
            "/v1/completions",
            "/chat/completions",
            "/completions",
        )
        state = self.telemetry_monitor.get_state()
        pressure = self.layer_modulator.compute_hardware_pressure(state)
        pressure_band = LearningStore.pressure_band(pressure)
        profile = "passthrough"
        profile_source = "disabled"
        requested_max_tokens = None
        applied_max_tokens = None
        model = "unknown"
        outgoing_body = raw_body

        if inference_route:
            try:
                request_data = json.loads(raw_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                request_data = None
            if isinstance(request_data, dict):
                model_value = request_data.get("model")
                if isinstance(model_value, str):
                    model = model_value
                extension = request_data.get("voltinfer")
                extension = extension if isinstance(extension, dict) else {}
                adaptive = self.adaptive_enabled or extension.get("adaptive") is True
                if adaptive:
                    requested_profile = extension.get("profile")
                    if requested_profile is not None and (
                        not isinstance(requested_profile, str)
                        or requested_profile not in PROFILE_CAPS
                    ):
                        self._send_json(
                            {"error": "profile must be one of: full, balanced, efficient"},
                            400,
                        )
                        return
                    try:
                        profile, profile_source, _ = self.learning_store.choose_profile(
                            model, pressure_band, requested_profile
                        ) if self.learning_store else (
                            requested_profile or {"low": "full", "medium": "balanced", "high": "efficient"}[pressure_band],
                            "requested" if requested_profile else "hardware_policy",
                            0,
                        )
                    except ValueError as error:
                        self._send_json({"error": str(error)}, 400)
                        return
                    except sqlite3.Error as error:
                        logger.error("Could not select an adaptive profile: %s", error)
                        self._send_json({"error": "Could not read local learning data"}, 503)
                        return
                    token_key = (
                        "max_completion_tokens"
                        if "max_completion_tokens" in request_data
                        else "max_tokens"
                    )
                    token_limit = request_data.get(token_key)
                    if isinstance(token_limit, int) and not isinstance(token_limit, bool) and token_limit > 0:
                        requested_max_tokens = token_limit
                        applied_max_tokens = max(
                            1, int(token_limit * PROFILE_CAPS[profile])
                        )
                        request_data[token_key] = applied_max_tokens
                    else:
                        profile = "passthrough"
                        profile_source = "no_token_limit"
                    request_data.pop("voltinfer", None)
                    outgoing_body = json.dumps(
                        request_data, separators=(",", ":"), ensure_ascii=False
                    ).encode("utf-8")

        started = time.monotonic()
        try:
            connection, upstream_response = self.upstream.open(
                method, self.path, outgoing_body, dict(self.headers.items())
            )
        except (http.client.HTTPException, OSError, TimeoutError, ValueError) as error:
            logger.warning("Upstream connection failed: %s", error)
            if inference_route and self.learning_store:
                try:
                    self.learning_store.record_request(
                        request_id=request_id,
                        model=model,
                        route=route,
                        latency_ms=(time.monotonic() - started) * 1000,
                        status=502,
                        pressure_band=pressure_band,
                        profile=profile,
                        requested_max_tokens=requested_max_tokens,
                        applied_max_tokens=applied_max_tokens,
                        completion_tokens=None,
                    )
                except sqlite3.Error as storage_error:
                    logger.error("Could not record local inference metrics: %s", storage_error)
            self._send_json({"error": "Could not connect to the configured LLM upstream"}, 502)
            return

        status = upstream_response.status
        content_length = upstream_response.getheader("Content-Length")
        transfer_encoding = upstream_response.getheader("Transfer-Encoding", "")
        connection_header = upstream_response.getheader("Connection", "")
        excluded_headers = {
            "connection",
            "date",
            "keep-alive",
            "server",
            "transfer-encoding",
            "content-length",
            "upgrade",
            "proxy-authenticate",
            "proxy-authorization",
        }
        excluded_headers.update(
            value.strip().lower() for value in connection_header.split(",")
        )
        self.send_response(status)
        for header, value in upstream_response.getheaders():
            if header.lower() not in excluded_headers and not header.lower().startswith(
                "access-control-"
            ):
                self.send_header(header, value)
        if content_length and transfer_encoding.lower() != "chunked":
            self.send_header("Content-Length", content_length)
        self.send_header("X-VoltInfer-Request-ID", request_id)
        self.send_header(
            "Access-Control-Expose-Headers",
            "X-VoltInfer-Request-ID, X-VoltInfer-Profile, X-VoltInfer-Profile-Source",
        )
        if inference_route and profile != "passthrough":
            self.send_header("X-VoltInfer-Profile", profile)
            self.send_header("X-VoltInfer-Profile-Source", profile_source)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        captured = bytearray()
        completed = False
        try:
            while True:
                chunk = upstream_response.read1(64 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
                if len(captured) < 2 * 1024 * 1024:
                    captured.extend(chunk[: 2 * 1024 * 1024 - len(captured)])
            completed = True
        except (http.client.HTTPException, OSError, TimeoutError) as error:
            logger.warning("Upstream response stream interrupted: %s", error)
        finally:
            upstream_response.close()
            connection.close()

        if inference_route and self.learning_store and completed:
            completion_tokens = None
            try:
                response_data = json.loads(captured.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                response_data = {}
            usage = response_data.get("usage", {}) if isinstance(response_data, dict) else {}
            if isinstance(usage, dict):
                token_count = usage.get("completion_tokens")
                if isinstance(token_count, int) and not isinstance(token_count, bool):
                    completion_tokens = token_count
            try:
                self.learning_store.record_request(
                    request_id=request_id,
                    model=model,
                    route=route,
                    latency_ms=(time.monotonic() - started) * 1000,
                    status=status,
                    pressure_band=pressure_band,
                    profile=profile,
                    requested_max_tokens=requested_max_tokens,
                    applied_max_tokens=applied_max_tokens,
                    completion_tokens=completion_tokens,
                )
            except sqlite3.Error as storage_error:
                logger.error("Could not record local inference metrics: %s", storage_error)

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

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        model_path: Optional[str] = None,
        upstream_url: Optional[str] = None,
        upstream_api_key: Optional[str] = None,
        adaptive: bool = False,
        learning_enabled: bool = True,
        learning_db_path: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.telemetry_monitor = TelemetryMonitor(sample_interval=0.15)

        configured_upstream = upstream_url or os.environ.get("VOLTINFER_UPSTREAM_URL")
        self.upstream = (
            OpenAICompatibleUpstream(
                configured_upstream,
                api_key=upstream_api_key or os.environ.get("VOLTINFER_UPSTREAM_API_KEY"),
            )
            if configured_upstream
            else None
        )
        if self.upstream and self._upstream_points_to_listener(
            self.upstream.scheme,
            self.upstream.host,
            self.upstream.port,
            host,
            port,
        ):
            raise ValueError("upstream URL must not point back to the VoltInfer listener")
        # A configured upstream owns model loading; do not load local weights as well.
        path = None if self.upstream else model_path or os.environ.get("VOLT_MODEL_PATH")
        self.model_backend = RealModelBackend(path)
        self.layer_modulator = DynamicLayerModulator(model_backend=self.model_backend)
        self.adaptive_enabled = adaptive
        self.learning_store = (
            LearningStore(learning_db_path)
            if learning_enabled and os.environ.get("VOLTINFER_DISABLE_LEARNING") != "1"
            else None
        )
        self.server: Optional[ThreadingHTTPServer] = None

    @staticmethod
    def _upstream_points_to_listener(
        upstream_scheme: str,
        upstream_host: str,
        upstream_port: Optional[int],
        listener_host: str,
        listener_port: int,
    ) -> bool:
        normalized_upstream_port = upstream_port or (
            443 if upstream_scheme == "https" else 80
        )
        if normalized_upstream_port != listener_port:
            return False
        loopback_hosts = {"127.0.0.1", "::1", "localhost"}
        upstream_is_local = upstream_host.lower() in loopback_hosts
        listener_is_local = listener_host.lower() in loopback_hosts | {"0.0.0.0", "::"}
        return upstream_host.lower() == listener_host.lower() or (
            upstream_is_local and listener_is_local
        )

    def start(self, block: bool = False) -> None:
        """Start telemetry monitor and HTTP proxy server."""
        self.server = ThreadingHTTPServer((self.host, self.port), VoltInferHTTPRequestHandler)
        self.server.telemetry_monitor = self.telemetry_monitor
        self.server.layer_modulator = self.layer_modulator
        self.server.upstream = self.upstream
        self.server.learning_store = self.learning_store
        self.server.adaptive_enabled = self.adaptive_enabled
        self.telemetry_monitor.start()
        logger.info(f"VoltInfer Proxy Server listening on http://{self.host}:{self.port}")
        if self.upstream:
            logger.info("Forwarding OpenAI-compatible requests to %s", self.upstream.base_url)
        elif self.model_backend.is_loaded:
            logger.info(f"Active Model: '{self.model_backend.model_name}' ({self.model_backend.backend_type})")
        else:
            logger.warning("No LLM upstream configured; inference endpoints are in demo simulation mode.")

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
