"""
VoltInfer Desktop Control Center (GUI Software)
Zero-dependency desktop control panel for VoltInfer using Python's standard tkinter.
Monitors silicon hardware telemetry in real-time, launches/stops the proxy daemon,
injects stress profiles, and opens the Web Dashboard with one click.
"""

import sys
import os
import time
import threading
import webbrowser
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional

from engine.proxy import VoltInferProxyServer
from engine.telemetry import TelemetryMonitor


class VoltInferDesktopApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("VoltInfer ⚡ Desktop Control Center")
        self.root.geometry("780x640")
        self.root.minsize(700, 580)
        self.root.configure(bg="#0b0f19")

        # Core engine state
        self.proxy_server: Optional[VoltInferProxyServer] = None
        self.server_thread: Optional[threading.Thread] = None
        self.selected_model_path: Optional[str] = None
        self.is_running = False

        # Independent local telemetry sampler for UI gauges
        self.gui_telemetry = TelemetryMonitor(sample_interval=0.2)
        self.gui_telemetry.start()

        self._setup_styles()
        self._build_ui()
        self._start_telemetry_timer()

    def _setup_styles(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TProgressbar", thickness=10, background="#00f0ff", troughcolor="#161f30")

    def _build_ui(self):
        # 1. Header Banner
        header = tk.Frame(self.root, bg="#111827", padx=20, pady=14, relief="flat")
        header.pack(fill="x")

        title_frame = tk.Frame(header, bg="#111827")
        title_frame.pack(side="left")

        title_lbl = tk.Label(
            title_frame, text="⚡ VoltInfer", font=("Segoe UI", 18, "bold"),
            fg="#00f0ff", bg="#111827"
        )
        title_lbl.pack(side="left")

        sub_lbl = tk.Label(
            title_frame, text=" | Energy-Aware LLM Runtime & Proxy Daemon",
            font=("Segoe UI", 10), fg="#9ca3af", bg="#111827"
        )
        sub_lbl.pack(side="left", padx=8)

        self.status_badge = tk.Label(
            header, text="● SERVER STOPPED", font=("Segoe UI", 9, "bold"),
            fg="#f87171", bg="#1f2937", padx=12, pady=4, relief="groove"
        )
        self.status_badge.pack(side="right")

        # 2. Main Body Container
        body = tk.Frame(self.root, bg="#0b0f19", padx=20, pady=15)
        body.pack(fill="both", expand=True)

        # Telemetry Card Frame
        telemetry_frame = tk.LabelFrame(
            body, text=" 📊 Silicon Hardware Telemetry ", font=("Segoe UI", 10, "bold"),
            fg="#00f0ff", bg="#131b2e", padx=15, pady=12, relief="ridge"
        )
        telemetry_frame.pack(fill="x", pady=(0, 15))

        grid = tk.Frame(telemetry_frame, bg="#131b2e")
        grid.pack(fill="x")
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(2, weight=1)
        grid.columnconfigure(3, weight=1)

        # Gauge 1: Temperature
        g1 = tk.Frame(grid, bg="#1a233a", padx=10, pady=8, relief="groove")
        g1.grid(row=0, column=0, padx=5, sticky="ew")
        tk.Label(g1, text="CORE TEMP", font=("Segoe UI", 8, "bold"), fg="#9ca3af", bg="#1a233a").pack()
        self.lbl_temp = tk.Label(g1, text="-- °C", font=("Segoe UI", 16, "bold"), fg="#00ff9d", bg="#1a233a")
        self.lbl_temp.pack()
        self.lbl_headroom = tk.Label(g1, text="-- Headroom", font=("Segoe UI", 8), fg="#6ee7b7", bg="#1a233a")
        self.lbl_headroom.pack()

        # Gauge 2: Power Draw
        g2 = tk.Frame(grid, bg="#1a233a", padx=10, pady=8, relief="groove")
        g2.grid(row=0, column=1, padx=5, sticky="ew")
        tk.Label(g2, text="POWER DRAW", font=("Segoe UI", 8, "bold"), fg="#9ca3af", bg="#1a233a").pack()
        self.lbl_power = tk.Label(g2, text="-- W", font=("Segoe UI", 16, "bold"), fg="#fbbf24", bg="#1a233a")
        self.lbl_power.pack()
        self.lbl_power_mode = tk.Label(g2, text="AC Powered", font=("Segoe UI", 8), fg="#9ca3af", bg="#1a233a")
        self.lbl_power_mode.pack()

        # Gauge 3: CPU / GPU Load
        g3 = tk.Frame(grid, bg="#1a233a", padx=10, pady=8, relief="groove")
        g3.grid(row=0, column=2, padx=5, sticky="ew")
        tk.Label(g3, text="CPU / GPU LOAD", font=("Segoe UI", 8, "bold"), fg="#9ca3af", bg="#1a233a").pack()
        self.lbl_load = tk.Label(g3, text="--% / --%", font=("Segoe UI", 14, "bold"), fg="#38bdf8", bg="#1a233a")
        self.lbl_load.pack()
        self.lbl_ram = tk.Label(g3, text="RAM: -- GB", font=("Segoe UI", 8), fg="#9ca3af", bg="#1a233a")
        self.lbl_ram.pack()

        # Gauge 4: Battery & Status
        g4 = tk.Frame(grid, bg="#1a233a", padx=10, pady=8, relief="groove")
        g4.grid(row=0, column=3, padx=5, sticky="ew")
        tk.Label(g4, text="BATTERY / PROFILE", font=("Segoe UI", 8, "bold"), fg="#9ca3af", bg="#1a233a").pack()
        self.lbl_battery = tk.Label(g4, text="--%", font=("Segoe UI", 16, "bold"), fg="#c084fc", bg="#1a233a")
        self.lbl_battery.pack()
        self.lbl_profile_mode = tk.Label(g4, text="REALTIME", font=("Segoe UI", 8, "bold"), fg="#a78bfa", bg="#1a233a")
        self.lbl_profile_mode.pack()

        # 3. Model & Profile Settings
        cfg_frame = tk.LabelFrame(
            body, text=" ⚙ Configuration & Hardware Simulation Profile ",
            font=("Segoe UI", 10, "bold"), fg="#00f0ff", bg="#131b2e", padx=15, pady=10, relief="ridge"
        )
        cfg_frame.pack(fill="x", pady=(0, 15))

        # Model Picker Row
        row1 = tk.Frame(cfg_frame, bg="#131b2e")
        row1.pack(fill="x", pady=4)
        tk.Label(row1, text="GGUF / Model Weights:", font=("Segoe UI", 9), fg="#e5e7eb", bg="#131b2e").pack(side="left")
        
        self.entry_model = tk.Entry(row1, font=("Segoe UI", 9), bg="#1f2937", fg="#f3f4f6", insertbackground="#00f0ff")
        self.entry_model.pack(side="left", fill="x", expand=True, padx=8)
        self.entry_model.insert(0, "Adaptive Simulation (or browse .gguf)")

        btn_browse = tk.Button(
            row1, text="📁 Browse", font=("Segoe UI", 8, "bold"),
            bg="#374151", fg="#f3f4f6", activebackground="#4b5563",
            relief="flat", padx=10, pady=2, command=self._browse_model
        )
        btn_browse.pack(side="right")

        # Profile Injection Row
        row2 = tk.Frame(cfg_frame, bg="#131b2e")
        row2.pack(fill="x", pady=4)
        tk.Label(row2, text="Hardware Stress Profile:", font=("Segoe UI", 9), fg="#e5e7eb", bg="#131b2e").pack(side="left")

        self.profile_var = tk.StringVar(value="REALTIME")
        profiles = ["REALTIME", "PEAK_PERFORMANCE", "NORMAL", "THERMAL_THROTTLING", "BATTERY_SAVER"]
        profile_menu = ttk.Combobox(row2, textvariable=self.profile_var, values=profiles, state="readonly", width=22)
        profile_menu.pack(side="left", padx=8)
        profile_menu.bind("<<ComboboxSelected>>", self._on_profile_change)

        btn_apply = tk.Button(
            row2, text="⚡ Apply Profile", font=("Segoe UI", 8, "bold"),
            bg="#2563eb", fg="#ffffff", activebackground="#1d4ed8",
            relief="flat", padx=10, pady=2, command=self._apply_profile
        )
        btn_apply.pack(side="left")

        # 4. Controls Row (Start, Stop, Browser, Bench)
        ctrl_frame = tk.Frame(body, bg="#0b0f19")
        ctrl_frame.pack(fill="x", pady=(0, 15))

        self.btn_start = tk.Button(
            ctrl_frame, text="▶ Start Proxy Server", font=("Segoe UI", 10, "bold"),
            bg="#059669", fg="#ffffff", activebackground="#047857",
            relief="flat", padx=18, pady=8, command=self.start_server
        )
        self.btn_start.pack(side="left", padx=(0, 8))

        self.btn_stop = tk.Button(
            ctrl_frame, text="⏹ Stop Server", font=("Segoe UI", 10, "bold"),
            bg="#dc2626", fg="#ffffff", activebackground="#b91c1c",
            relief="flat", padx=18, pady=8, state="disabled", command=self.stop_server
        )
        self.btn_stop.pack(side="left", padx=(0, 8))

        self.btn_dashboard = tk.Button(
            ctrl_frame, text="🌐 Open Web Dashboard", font=("Segoe UI", 10, "bold"),
            bg="#0284c7", fg="#ffffff", activebackground="#0369a1",
            relief="flat", padx=18, pady=8, command=self.open_dashboard
        )
        self.btn_dashboard.pack(side="left", padx=(0, 8))

        self.btn_bench = tk.Button(
            ctrl_frame, text="📊 Run Benchmark", font=("Segoe UI", 10, "bold"),
            bg="#4b5563", fg="#ffffff", activebackground="#374151",
            relief="flat", padx=18, pady=8, command=self.run_benchmark_async
        )
        self.btn_bench.pack(side="right")

        # 5. Output / Server Console Log
        log_frame = tk.LabelFrame(
            body, text=" 📜 Daemon Activity Logs ", font=("Segoe UI", 10, "bold"),
            fg="#00f0ff", bg="#131b2e", padx=10, pady=8, relief="ridge"
        )
        log_frame.pack(fill="both", expand=True)

        self.log_text = tk.Text(
            log_frame, bg="#090d16", fg="#d1d5db", font=("Consolas", 9),
            relief="flat", insertbackground="#00f0ff"
        )
        self.log_text.pack(fill="both", expand=True)
        self._log("VoltInfer Desktop Software initialized.")
        self._log("System ready. Click 'Start Proxy Server' to launch runtime.")

    def _log(self, msg: str):
        self.log_text.insert("end", f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.log_text.see("end")

    def _browse_model(self):
        filename = filedialog.askopenfilename(
            title="Select GGUF / Model Weights",
            filetypes=[("GGUF Model Files", "*.gguf"), ("All Files", "*.*")]
        )
        if filename:
            self.selected_model_path = filename
            self.entry_model.delete(0, "end")
            self.entry_model.insert(0, filename)
            self._log(f"Selected model weights: {os.path.basename(filename)}")

    def _on_profile_change(self, event=None):
        self._apply_profile()

    def _apply_profile(self):
        profile = self.profile_var.get()
        self.gui_telemetry.set_power_profile(profile)
        if self.proxy_server:
            self.proxy_server.telemetry_monitor.set_power_profile(profile)
        self._log(f"Applied hardware profile: {profile}")

    def start_server(self):
        if self.is_running:
            return
        
        path_str = self.entry_model.get().strip()
        model_path = path_str if os.path.exists(path_str) else None

        self.proxy_server = VoltInferProxyServer(host="127.0.0.1", port=8000, model_path=model_path)
        self.proxy_server.start(block=False)
        self.is_running = True

        self.status_badge.config(text="● RUNNING (Port 8000)", fg="#34d399", bg="#064e3b")
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")

        self._log("VoltInfer Proxy Server listening on http://127.0.0.1:8000")
        self._log("OpenAI endpoints: /v1/chat/completions, /v1/telemetry")

    def stop_server(self):
        if not self.is_running or not self.proxy_server:
            return

        self.proxy_server.stop()
        self.proxy_server = None
        self.is_running = False

        self.status_badge.config(text="● SERVER STOPPED", fg="#f87171", bg="#1f2937")
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self._log("VoltInfer Proxy Server stopped.")

    def open_dashboard(self):
        url = "http://127.0.0.1:8000"
        if not self.is_running:
            # Auto-start if stopped
            self.start_server()
        webbrowser.open(url)
        self._log(f"Opened Web Dashboard in default browser: {url}")

    def run_benchmark_async(self):
        def _runner():
            self._log("Launching automated benchmark suite...")
            from bench.run_benchmark import run_benchmark_suite
            path_str = self.entry_model.get().strip()
            model_path = path_str if os.path.exists(path_str) else None
            results = run_benchmark_suite(model_path=model_path)
            self._log(f"Benchmark completed successfully! Evaluated {len(results)} inference scenarios.")
            self._log("Detailed results saved to 'bench_report.json'.")

        threading.Thread(target=_runner, daemon=True).start()

    def _start_telemetry_timer(self):
        self._update_telemetry_ui()
        self.root.after(400, self._start_telemetry_timer)

    def _update_telemetry_ui(self):
        monitor = self.proxy_server.telemetry_monitor if (self.proxy_server and self.is_running) else self.gui_telemetry
        state = monitor.get_state()

        # 1. Temp
        self.lbl_temp.config(text=f"{state.temperature_c:.1f} °C")
        headroom_pct = state.thermal_headroom * 100.0
        self.lbl_headroom.config(text=f"{headroom_pct:.1f}% Headroom")
        if state.temperature_c >= 80:
            self.lbl_temp.config(fg="#ef4444")
        elif state.temperature_c >= 65:
            self.lbl_temp.config(fg="#f59e0b")
        else:
            self.lbl_temp.config(fg="#10b981")

        # 2. Power
        self.lbl_power.config(text=f"{state.power_draw_w:.1f} W")
        plugged_str = "AC Power" if state.power_plugged else "Battery Power"
        self.lbl_power_mode.config(text=plugged_str)

        # 3. CPU/GPU Load
        self.lbl_load.config(text=f"CPU {state.cpu_utilization:.0f}% | GPU {state.gpu_utilization:.0f}%")
        self.lbl_ram.config(text=f"RAM: {state.ram_used_gb:.1f}/{state.ram_total_gb:.1f} GB ({state.ram_utilization:.0f}%)")

        # 4. Battery
        bat_str = f"{state.battery_percent:.0f}%" if state.battery_percent is not None else "100% (AC)"
        self.lbl_battery.config(text=bat_str)
        self.lbl_profile_mode.config(text=state.simulation_mode)

    def on_closing(self):
        if self.proxy_server:
            self.proxy_server.stop()
        self.gui_telemetry.stop()
        self.root.destroy()


def launch_gui():
    root = tk.Tk()
    app = VoltInferDesktopApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
