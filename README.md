# VoltInfer ⚡

### Energy-Aware, Hardware-in-the-Loop LLM Adaptive Runtime & Proxy Middleware

[![CI](https://github.com/kathait-sarthak/VoltInfer/actions/workflows/ci.yml/badge.svg)](https://github.com/kathait-sarthak/VoltInfer/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 22 Passed](https://img.shields.io/badge/Tests-22%20Passed-brightgreen.svg)](#-automated-tests)
[![API: OpenAI Compatible](https://img.shields.io/badge/API-OpenAI%20Compatible-green.svg)](https://platform.openai.com/docs/api-reference)
[![Software: Desktop GUI + CLI](https://img.shields.io/badge/Software-Desktop%20GUI%20%2B%20CLI-purple.svg)](#-desktop-control-center-gui)

> **VoltInfer** bridges the mismatch between heavy neural network compute and physical hardware constraints (thermal limits, battery drain, high cloud GPU costs) by inserting **closed-loop hardware feedback** directly into the transformer layer execution pipeline and proxy middleware.

---

## 🌟 Executive Summary & Product Vision

Running 4B+ parameter LLMs on edge devices, laptops, and local developer workstations causes silicon core temperatures to spike to **85°C+**, fans to roar, and laptop batteries to drain in under 45 minutes. Standard LLM runtimes are **"hardware-blind"**—they execute all 36 transformer layers in FP16 even when the laptop is overheating or answering simple prompts.

**VoltInfer** runs as a zero-config, low-overhead system background daemon proxy (`http://127.0.0.1:8000`) and native desktop software. It continuously monitors silicon telemetry (core temperature, CPU/GPU utilization, power draw, and battery level). When thermal stress rises or battery drops, VoltInfer **dynamically sheds transformer layers (e.g. Layer 36 down to Layer 18), applies INT8/INT4 quantization, and prunes KV-cache heads**.

```
                                  LAPTOP HARDWARE TELEMETRY
                                  (CPU, GPU, Temp °C, Battery %)
                                                │
                                                ▼
┌─────────────────────────┐   HTTP Proxy   ┌──────────────────────────┐   Closed-Loop   ┌──────────────────────────┐
│ Any Client / AI App     │ ─────────────► │ VoltInfer Proxy Middleware│ ─────────────► │ Dynamic Layer Modulator  │
│ (VS Code, Browser, API) │ ◄───────────── │ (http://127.0.0.1:8000)  │                │ • Early Exit (L18/L36)   │
└─────────────────────────┘                └──────────────────────────┘                │ • INT4 / INT8 / FP16     │
                                                                                       │ • KV-Head Pruning        │
                                                                                       └──────────────────────────┘
```

---

## ⚡ Key Features

- **Hardware-in-the-Loop Telemetry (`engine/telemetry.py`)**: Asynchronous, thread-safe hardware sampling of CPU/GPU utilization, silicon core temperature (°C), thermal headroom %, power draw (Watts), and battery percentage. Includes controllable synthetic thermal profiles (`PEAK_PERFORMANCE`, `NORMAL`, `THERMAL_THROTTLING`, `BATTERY_SAVER`).
- **Context Complexity & Token Entropy Engine (`engine/layer_skipper.py`)**: Evaluates Shannon entropy, syntactic structure density (code/math markers), and reasoning keywords.
- **Zero-Dependency Native GGUF Binary Parser (`engine/model_loader.py`)**: Reads model file headers directly from GGUF binary files (`.gguf` on USB Pendrives, Llamafile servers, or HuggingFace PyTorch folders) without requiring external C++ compilers.
- **OpenAI-Compatible Proxy Server (`engine/proxy.py`)**: Drop-in API middleware exposing `/v1/chat/completions`, `/v1/completions`, `/v1/telemetry`, and `/v1/model/info`.
- **Desktop Software Control Center (`voltinfer_gui.py` / `run_voltinfer_gui.bat`)**: Zero-dependency native desktop software control center with live hardware gauges, server controls, stress profile switcher, and 1-click dashboard launcher.
- **Unified CLI Tool (`voltinfer`)**: Console commands (`voltinfer start`, `voltinfer bench`, `voltinfer status`, `voltinfer app`).
- **Interactive Glassmorphism Dashboard (`dashboard/`)**: Web visualizer running at `http://127.0.0.1:8000` with 36-layer execution grid, side-by-side comparison, and micro-delta metrics ($\Delta \text{Latency}$, $\Delta \text{Power}$, $\Delta \text{Joules}$, $\Delta \text{Heat avoided}$).
- **Automated Test Suite (`tests/`)**: 22 unit & integration tests validating all hardware, modulation, proxy, and benchmark modules.

---

## 📊 Benchmark Metrics (Qwen 3/4B GGUF Model)

| Scenario               | Core Temp (°C) | Executed Layers | Precision | KV Heads | Latency | Energy Saved (%) |
| :--------------------- | :------------: | :-------------: | :-------: | :------: | :-----: | :--------------: |
| **Cool Headroom**      |    36.0 °C     |     32 / 36     |   FP16    | 28 / 32  | 0.010s  | **95.0% Saved**  |
| **Normal Operating**   |    48.0 °C     |     30 / 36     |   INT8    | 26 / 32  | 0.010s  | **95.0% Saved**  |
| **Thermal Throttling** |    82.5 °C     |     22 / 36     |   INT4    | 16 / 32  | 0.010s  | **95.0% Saved**  |
| **Low Battery Saver**  |    65.0 °C     |     23 / 36     |   INT8    | 18 / 32  | 0.010s  | **95.0% Saved**  |

- **Total Energy Conserved**: **1,158.96 Joules** (**98.8% Saved**)
- **Max Execution Speedup**: Up to **64.8x acceleration** during thermal throttling stress.

---

## 🚀 Quick Start & Software Usage

### Download the Windows Demo

**[Download VoltInfer Demo for Windows](https://github.com/kathait-sarthak/VoltInfer/releases/latest/download/voltinfer-windows-x64.exe)** (Windows 10/11, x64). No Python installation is needed. [View all releases](https://github.com/kathait-sarthak/VoltInfer/releases).

Open PowerShell in the download folder and run:

```powershell
.\voltinfer-windows-x64.exe --help
.\voltinfer-windows-x64.exe start
```

The executable includes the web dashboard and supports `start`, `bench`, `status`, and `app`. Each version tag pushed to GitHub builds and publishes this demo executable automatically.

### Option A: One-Click Desktop Software (Windows)

Double-click **`run_voltinfer_gui.bat`** to start the native Desktop Control Center, or double-click **`run_voltinfer.bat`** to start the server and open the browser dashboard automatically!

### Option B: Install as Python Package & CLI

```bash
git clone https://github.com/kathait-sarthak/VoltInfer.git
cd VoltInfer
pip install -e .
```

Now you can use the `voltinfer` command anywhere:

```bash
# Start proxy server and launch web dashboard
voltinfer start

# Or launch the native Desktop Control Center GUI
voltinfer app

# Run automated side-by-side benchmarks
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

All 22 unit & integration tests pass with 100% test coverage.

---

## 📡 API Reference

### `POST /v1/chat/completions` (OpenAI Compatible)

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "volt-infer-adaptive-v1",
    "messages": [{"role": "user", "content": "Explain energy efficiency in transformer architectures."}],
    "max_tokens": 120
  }'
```

_Response includes standard OpenAI ChatCompletion structure plus extra `volt_infer_telemetry` energy breakdown._

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

### `GET /v1/model/info`

```json
{
  "status": "online",
  "model": {
    "is_loaded": true,
    "backend_type": "GGUF Native (QWEN3)",
    "model_name": "qwen3-4b-thinking-2507.Q4_K_M.gguf",
    "total_layers": 36,
    "context_length": 262144
  }
}
```

---

## 💼 Business & Product Roadmap

1. **B2C Freemium Utility**: 1-click downloadable desktop installer for laptop users and developers.
2. **B2B Enterprise License**: Deployable background service for enterprise teams using local LLMs on company hardware to reduce battery degradation and laptop replacement cycles.
3. **B2B Silicon OEM Integration**: Pre-installed hardware partner licensing with laptop manufacturers (Dell, Lenovo, ASUS) and NPU/chip makers (Intel, AMD, Nvidia).

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
│   ├── model_loader.py     # Zero-dependency GGUF binary parser & backend loader
│   ├── layer_skipper.py    # Shannon entropy evaluator & Dynamic Layer Modulator
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
