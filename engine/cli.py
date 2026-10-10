"""
VoltInfer Unified CLI (Command Line Interface)
Provides CLI entry points for starting the proxy, running benchmarks,
checking status, and launching the desktop control panel software.
"""

import sys
import os
import time
import argparse
import webbrowser
import urllib.request
import json

from engine.proxy import VoltInferProxyServer
from bench.run_benchmark import run_benchmark_suite


def start_server_command(args: argparse.Namespace) -> None:
    """Start the VoltInfer proxy server and optionally launch browser dashboard."""
    print("=====================================================================")
    print("   ⚡ VoltInfer: Energy-Aware Adaptive Runtime Proxy ⚡")
    print("=====================================================================")
    
    server = VoltInferProxyServer(host=args.host, port=args.port, model_path=args.model_path)
    server.start(block=False)
    
    dashboard_url = f"http://{args.host}:{args.port}"
    print(f"\n[+] Proxy listening on: {dashboard_url}")
    print(f"[+] OpenAI Endpoint:   {dashboard_url}/v1/chat/completions")
    print(f"[+] Web Dashboard:     {dashboard_url}/\n")
    
    if args.open_browser:
        time.sleep(0.5)
        print("[+] Opening Web Dashboard in default browser...")
        webbrowser.open(dashboard_url)
    
    print("[*] Proxy is running live. Press Ctrl+C to terminate.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[*] Shutting down VoltInfer proxy...")
        server.stop()
        print("[+] Done.")


def benchmark_command(args: argparse.Namespace) -> None:
    """Run the side-by-side benchmark suite."""
    run_benchmark_suite(model_path=args.model_path)


def status_command(args: argparse.Namespace) -> None:
    """Check running proxy server status and hardware telemetry."""
    url = f"http://{args.host}:{args.port}/v1/telemetry"
    try:
        with urllib.request.urlopen(url, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            tel = data.get("telemetry", {})
            print("=====================================================================")
            print(f"   ⚡ VoltInfer Status: ONLINE ({args.host}:{args.port})")
            print("=====================================================================")
            print(f"  • Silicon Temperature : {tel.get('temperature_c')}°C")
            print(f"  • Thermal Headroom    : {tel.get('thermal_headroom_pct')}%")
            print(f"  • Power Draw          : {tel.get('power_draw_w')} Watts")
            print(f"  • CPU Utilization     : {tel.get('cpu_utilization')}%")
            print(f"  • GPU Utilization     : {tel.get('gpu_utilization')}%")
            print(f"  • RAM Utilization     : {tel.get('ram_utilization')}% ({tel.get('ram_used_gb')}/{tel.get('ram_total_gb')} GB)")
            print(f"  • Mode                : {tel.get('simulation_mode')}")
            print("=====================================================================")
    except Exception as e:
        print(f"[-] Could not reach VoltInfer proxy at http://{args.host}:{args.port}")
        print(f"[-] Error: {e}")
        sys.exit(1)


def app_command(args: argparse.Namespace) -> None:
    """Launch the native desktop software control panel."""
    from voltinfer_gui import launch_gui
    launch_gui()


def main():
    parser = argparse.ArgumentParser(
        prog="voltinfer",
        description="VoltInfer ⚡ Energy-Aware Hardware-in-the-Loop LLM Adaptive Runtime"
    )
    subparsers = parser.add_subparsers(dest="command", help="VoltInfer commands")

    # voltinfer start
    start_parser = subparsers.add_parser("start", help="Start the VoltInfer proxy server & web dashboard")
    start_parser.add_argument("--host", default="127.0.0.1", help="Host IP to bind (default: 127.0.0.1)")
    start_parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    start_parser.add_argument("--model-path", "-m", default=None, help="Path to local GGUF or PyTorch model weights")
    start_parser.add_argument("--open-browser", "-b", action="store_true", default=True, help="Auto-open browser dashboard (default: True)")
    start_parser.add_argument("--no-browser", dest="open_browser", action="store_false", help="Do not open browser")

    # voltinfer bench
    bench_parser = subparsers.add_parser("bench", help="Run automated side-by-side benchmark suite")
    bench_parser.add_argument("--model-path", "-m", default=None, help="Path to GGUF model file")

    # voltinfer status
    status_parser = subparsers.add_parser("status", help="Query live hardware telemetry from active proxy")
    status_parser.add_argument("--host", default="127.0.0.1", help="Host IP (default: 127.0.0.1)")
    status_parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")

    # voltinfer app
    subparsers.add_parser("app", help="Launch native desktop software GUI control panel")

    args = parser.parse_args()

    if args.command == "start" or args.command is None:
        if args.command is None:
            # Default command is start with browser
            args.host = "127.0.0.1"
            args.port = 8000
            args.model_path = None
            args.open_browser = True
        start_server_command(args)
    elif args.command == "bench":
        benchmark_command(args)
    elif args.command == "status":
        status_command(args)
    elif args.command == "app":
        app_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
