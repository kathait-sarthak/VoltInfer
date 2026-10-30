# VoltInfer ⚡

### A local OpenAI-compatible LLM proxy with private, feedback-guided adaptation

[![CI](https://github.com/kathait-sarthak/VoltInfer/actions/workflows/ci.yml/badge.svg)](https://github.com/kathait-sarthak/VoltInfer/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![API: OpenAI Compatible](https://img.shields.io/badge/API-OpenAI%20Compatible-green.svg)](https://platform.openai.com/docs/api-reference)
[![Software: Desktop GUI + CLI](https://img.shields.io/badge/Software-Desktop%20GUI%20%2B%20CLI-purple.svg)](#-desktop-control-center-gui)

> VoltInfer sits between an OpenAI-compatible client and an LLM server. It forwards requests and responses, reports local hardware telemetry, and can optionally learn which output-token profile users prefer. It does not currently rewrite hidden transformer layers or quantize arbitrary models.

---

## 🌟 Executive Summary & Product Vision

VoltInfer provides a local proxy at `http://127.0.0.1:8000`. Configure an OpenAI-compatible application to use that address, and set VoltInfer's upstream to a local server such as Ollama or LM Studio. Requests are passed through unchanged unless adaptive profiles are explicitly enabled.

Adaptive profiles only adjust a request's output-token limit (`full`, `balanced`, or `efficient`). Local feedback can influence which profile is selected. Prompts and credentials are not written to the learning database. Models hosted by cloud APIs or local runtimes keep control of their internal layers, quantization, and KV cache; VoltInfer cannot change those internals through a generic API.

```
┌────────────────────────┐               ┌───────────────────────────┐               ┌─────────────────────────┐
│ OpenAI-compatible      │ ────────────► │ VoltInfer Proxy           │ ────────────► │ Configured LLM server   │
│ application            │ ◄──────────── │ Telemetry + local learning│ ◄──────────── │ (Ollama, LM Studio, etc.)│
└────────────────────────┘               └───────────────────────────┘               └─────────────────────────┘
```

---

## ⚡ Key Features

- **Hardware telemetry (`engine/telemetry.py`)**: Asynchronous CPU, memory, and battery sampling where available. Temperature comes from a CPU sensor where supported; otherwise temperature, GPU load, VRAM, and power are clearly marked estimates. Includes synthetic test profiles.
- **OpenAI-compatible forwarding (`engine/upstream.py`)**: Forwards API calls, authorization, error responses, and streaming responses to the configured upstream. Supports `http` and `https`.
- **Private local learning (`engine/learning.py`)**: Stores request timing, token counts, profile selection, and optional 1–5 ratings in a local SQLite database. It never stores prompt text or API keys.
- **Opt-in output profiles**: `full` retains the requested token limit, `balanced` caps it at 75%, and `efficient` caps it at 50%. The profile is applied only when adaptive mode is requested or enabled. User feedback can guide profile selection after there are enough ratings.
- **Demo and experimental local inference (`engine/layer_skipper.py`, `engine/model_loader.py`)**: Without an upstream or supported local runtime, inference uses synthetic demonstration responses. Optional local model loading is experimental; it does not apply the advertised layer, quantization, or KV-cache changes.
- **Local model inspection (`engine/model_loader.py`)**: Includes experimental GGUF metadata parsing and optional runtime integrations; metadata inspection alone does not load or accelerate a model.
- **OpenAI-compatible local API (`engine/proxy.py`)**: Exposes the configured upstream API plus VoltInfer endpoints such as `/v1/telemetry`, `/v1/learning/status`, and `/v1/model/info`.
- **Desktop Software Control Center (`voltinfer_gui.py` / `run_voltinfer_gui.bat`)**: Tkinter desktop control center with hardware gauges, upstream configuration, server controls, and a dashboard launcher.
- **Unified CLI Tool (`voltinfer`)**: Console commands (`voltinfer start`, `voltinfer bench`, `voltinfer status`, `voltinfer app`).
- **Interactive Dashboard (`dashboard/`)**: Local visualizer served at `http://127.0.0.1:8000`.
- **Automated Test Suite (`tests/`)**: Unit and integration tests include forwarding against a mock LLM service, adaptive request behavior, and local feedback storage.

---

## 📊 Benchmarking

The existing benchmark command exercises the demo policy and uses modeled energy estimates. It is useful for testing policy code, but its output is **not a measured energy saving or real-model speedup**. For a meaningful performance comparison, benchmark the same model, prompts, and hardware directly against the upstream runtime; report measured wall time and power separately.

---

## 🚀 Quick Start & Software Usage

### Install on Windows

Download **`VoltInfer-Setup.exe`** from the [latest GitHub release](https://github.com/kathait-sarthak/VoltInfer/releases/latest) and run it. The installer adds Start Menu shortcuts and does not include or download an LLM model. If the installer is not attached to the latest release yet, the portable `voltinfer-windows-x64.exe` is also available on that release page.

To run the portable executable without installing it, download `voltinfer-windows-x64.exe` from that release, open PowerShell in the download folder, and run:

```powershell
.\voltinfer-windows-x64.exe start --upstream-url http://localhost:11434/v1
```

The installed app is available from the Start Menu. The installer and portable executable support the `start`, `bench`, `status`, and `app` commands. New tagged releases build both the portable executable and Windows installer. Configure `VOLTINFER_UPSTREAM_API_KEY` only when connecting to a trusted upstream that requires authentication.

### Option A: One-Click Desktop Software (Windows)

Launch **VoltInfer Desktop** from the Start Menu, enter an OpenAI-compatible upstream URL (for example `http://localhost:11434/v1`), then start the proxy. The `.bat` launchers are for running from a source checkout.

### Option B: Install as Python Package & CLI

```bash
git clone https://github.com/kathait-sarthak/VoltInfer.git
cd VoltInfer
pip install -e .
```

Now you can use the `voltinfer` command anywhere:

```bash
# Start a proxy for an Ollama OpenAI-compatible endpoint
voltinfer start --upstream-url http://localhost:11434/v1

# Optionally enable adaptive output profiles
voltinfer start --upstream-url http://localhost:11434/v1 --adaptive

# Or launch the native Desktop Control Center GUI
voltinfer app

# Run demo-policy benchmarks (modeled estimates)
voltinfer bench

# Check live hardware telemetry of running server
voltinfer status
```

### Option C: Run with Python Directly

```bash
python -m engine.cli start --open-browser
```

---

## 🧪 Automated Tests

Run the test suite with pytest:

```bash
python -m pytest -v
```

Tests cover the proxy, adaptive profiles, telemetry, and local feedback storage. No model download is needed for the test suite.

---

## 📡 API Reference

### `POST /v1/chat/completions` (OpenAI Compatible)

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:4b",
    "messages": [{"role": "user", "content": "Explain energy efficiency in transformer architectures."}],
    "max_tokens": 120
  }'
```

The upstream response body is passed through unchanged. Set your OpenAI-compatible client base URL to `http://127.0.0.1:8000/v1`.

To opt a single request into an output profile, add VoltInfer's local-only extension:

```json
{
  "model": "qwen3:4b",
  "messages": [{"role": "user", "content": "Summarize this document."}],
  "max_tokens": 400,
  "voltinfer": {"adaptive": true, "profile": "balanced"}
}
```

VoltInfer removes the extension before forwarding. `balanced` caps `max_tokens` at 75% of the requested limit. The upstream response includes `X-VoltInfer-Request-ID`; submit a rating to train future profile selection:

```bash
curl http://127.0.0.1:8000/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{"request_id": "ID_FROM_RESPONSE_HEADER", "rating": 5}'
```

`rating` must be from 1 to 5. Adaptive profiles are not applied unless enabled at startup or requested on that individual call. They only affect output-token limits; they do not alter model layers or quantization.

### `GET /v1/telemetry`

```json
{
  "status": "online",
  "telemetry": {
    "cpu_utilization": 28.5,
    "gpu_utilization": 35.0,
    "temperature_c": 46.2,
    "thermal_headroom": 0.825,
    "power_draw_w": 65.4,
    "battery_percent": 85.0
  }
}
```

### `GET /v1/learning/status`

Shows aggregate local request counts, latency, and optional ratings. The SQLite database stores no prompts or API keys. On Windows it is under `%LOCALAPPDATA%\VoltInfer`; set `VOLTINFER_DATA_DIR` to change the storage directory. Set `VOLTINFER_DISABLE_LEARNING=1` or start with `--no-learning` to disable storage.

### `GET /v1/model/info`

```json
{
  "status": "online",
  "model": {
    "upstream_configured": true,
    "backend_type": "OPENAI_COMPATIBLE_UPSTREAM",
    "upstream_url": "http://localhost:11434/v1",
    "adaptive_profiles": ["full", "balanced", "efficient"]
  }
}
```

---

## 💼 Business & Product Roadmap

1. **Runtime-specific controls**: Add explicit adapters for supported local runtimes before claiming layer or quantization control.
2. **Measured efficiency**: Validate power and quality trade-offs on real hardware and publish repeatable benchmarks.
3. **Installers and platform support**: Expand signed installers beyond the current Windows x64 build.

---

## 🛠️ Project Structure

```
VoltInfer/
├── .github/workflows/
│   └── ci.yml              # Automated GitHub Actions multi-platform CI
├── engine/
│   ├── __init__.py         # Package initialization & warning suppression
│   ├── cli.py              # Unified CLI entry point ('voltinfer')
│   ├── telemetry.py        # Asynchronous hardware sampler & stress simulator
│   ├── model_loader.py     # Experimental GGUF metadata parsing and optional loaders
│   ├── upstream.py         # OpenAI-compatible upstream request forwarding
│   ├── learning.py         # Local-only metrics and feedback-guided profiles
│   ├── layer_skipper.py    # Demo policy and synthetic inference responses
│   └── proxy.py            # OpenAI-compatible ThreadingHTTPServer & static file host
├── bench/
│   ├── __init__.py
│   └── run_benchmark.py    # Side-by-side benchmark runner & JSON report exporter
├── dashboard/
│   ├── index.html          # Glassmorphism visual web dashboard
│   ├── styles.css          # Dark mode design system & micro-delta styles
│   └── app.js              # Real-time telemetry streaming & layer visualizer
├── tests/
│   ├── test_telemetry.py   # Telemetry & hardware stress unit tests
│   ├── test_layer_skipper.py # Entropy & layer modulation unit tests
│   ├── test_model_loader.py# GGUF parser & model backend unit tests
│   ├── test_proxy.py       # API endpoints & proxy integration tests
│   └── test_benchmark.py   # Automated benchmark verification tests
├── voltinfer_gui.py        # Desktop Software Control Center GUI (tkinter)
├── installer.iss           # Windows installer definition
├── run_voltinfer.bat       # One-click Windows runner (Server + Browser)
├── run_voltinfer_gui.bat   # One-click Windows runner (Desktop Control Center)
├── pyproject.toml          # PEP 517/518 build configuration & metadata
├── setup.py                # Setuptools installation configuration
├── requirements.txt
└── README.md
```

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for details.

Developed by **Sarthak Kathait** ([@kathait-sarthak](https://github.com/kathait-sarthak)).
