"""
VoltInfer Telemetry & Hardware Reader Module
Samples real-time hardware telemetry (CPU, GPU, RAM, VRAM, thermal headroom, battery state)
and provides asynchronous thread-safe state access with synthetic hardware pressure controls.
"""

from dataclasses import dataclass, asdict
import threading
import time
import math
import logging
from typing import Dict, Any, Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger("VoltInfer.Telemetry")

# Thermal boundaries
T_IDLE = 38.0   # °C at idle
T_MAX = 85.0    # °C thermal throttling threshold
T_CRITICAL = 95.0 # °C emergency shutdown threshold


@dataclass
class HardwareState:
    """Unified container for real-time hardware metrics."""
    cpu_utilization: float        # Percentage (0-100)
    gpu_utilization: float        # Percentage (0-100)
    ram_used_gb: float            # GB used
    ram_total_gb: float           # Total GB
    ram_utilization: float        # Percentage (0-100)
    vram_used_gb: float           # GB used
    vram_total_gb: float          # Total GB
    vram_utilization: float       # Percentage (0-100)
    temperature_c: float          # Core temperature in °C
    thermal_headroom: float       # Normalized 0.0 (throttled) to 1.0 (cool)
    power_draw_w: float           # Current system/GPU power draw in Watts
    battery_percent: Optional[float]  # Percentage (0-100) or None if AC/Desktop
    power_plugged: Optional[bool]     # True if plugged into AC, False if battery
    timestamp: float              # UNIX timestamp
    is_simulated: bool            # True if synthetic/overridden
    simulation_mode: str          # REALTIME, SIMULATED, HYBRID

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to serializable dictionary."""
        d = asdict(self)
        d['thermal_headroom_pct'] = round(self.thermal_headroom * 100, 1)
        return d


class TelemetryMonitor:
    """
    Asynchronous hardware telemetry sampler.
    Runs a dedicated polling loop in a daemon thread.
    Allows dynamic simulation overrides for thermal/power benchmarking.
    """

    def __init__(self, sample_interval: float = 0.2):
        self.sample_interval = sample_interval
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Simulation override parameters
        self._sim_mode = "REALTIME"  # REALTIME, SIMULATED, HYBRID
        self._sim_temperature: Optional[float] = None
        self._sim_gpu_util: Optional[float] = None
        self._sim_cpu_util: Optional[float] = None
        self._sim_battery_pct: Optional[float] = None
        self._sim_plugged: Optional[bool] = None

        # Thermal simulation state (smooth thermal inertia)
        self._smoothed_temp = T_IDLE
        self._smoothed_power = 15.0

        # Initial state
        self._current_state = self._read_hardware()

    def start(self) -> None:
        """Start background sampling thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="VoltInfer-Telemetry")
        self._thread.start()
        logger.info("Hardware Telemetry Monitor started.")

    def stop(self) -> None:
        """Stop background sampling thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("Hardware Telemetry Monitor stopped.")

    def get_state(self) -> HardwareState:
        """Thread-safe getter for latest hardware state."""
        with self._lock:
            return self._current_state

    # ------------------------------------------------------------------
    # Simulation & Control Interfaces
    # ------------------------------------------------------------------

    def set_simulated_temperature(self, temp_c: float) -> None:
        """Inject a synthetic thermal temperature (°C)."""
        with self._lock:
            self._sim_temperature = max(20.0, min(100.0, temp_c))
            self._smoothed_temp = self._sim_temperature
            self._sim_mode = "HYBRID" if self._sim_gpu_util is None else "SIMULATED"
            self._current_state = self._read_hardware()

    def set_simulated_gpu_load(self, gpu_util_pct: float) -> None:
        """Inject synthetic GPU load percentage (0-100)."""
        with self._lock:
            self._sim_gpu_util = max(0.0, min(100.0, gpu_util_pct))
            self._sim_mode = "HYBRID" if self._sim_temperature is None else "SIMULATED"
            self._current_state = self._read_hardware()

    def set_power_profile(self, profile: str) -> None:
        """
        Preset simulation profiles:
        - NORMAL: Ambient room temperature (42°C), normal AC power
        - THERMAL_THROTTLING: High heat stress (82°C), low headroom
        - BATTERY_SAVER: On battery (25%), throttled profile
        - PEAK_PERFORMANCE: Cold/Max Headroom (35°C), 100% power
        """
        profile = profile.upper()
        with self._lock:
            if profile == "THERMAL_THROTTLING":
                self._sim_temperature = 82.5
                self._smoothed_temp = 82.5
                self._sim_gpu_util = 95.0
                self._sim_cpu_util = 88.0
                self._sim_battery_pct = 40.0
                self._sim_plugged = True
                self._sim_mode = "THERMAL_THROTTLING"
            elif profile == "BATTERY_SAVER":
                self._sim_temperature = 65.0
                self._smoothed_temp = 65.0
                self._sim_gpu_util = 50.0
                self._sim_cpu_util = 45.0
                self._sim_battery_pct = 18.0
                self._sim_plugged = False
                self._sim_mode = "BATTERY_SAVER"
            elif profile == "PEAK_PERFORMANCE":
                self._sim_temperature = 36.0
                self._smoothed_temp = 36.0
                self._sim_gpu_util = 25.0
                self._sim_cpu_util = 20.0
                self._sim_battery_pct = 100.0
                self._sim_plugged = True
                self._sim_mode = "PEAK_PERFORMANCE"
            elif profile == "NORMAL":
                self._sim_temperature = 48.0
                self._smoothed_temp = 48.0
                self._sim_gpu_util = 35.0
                self._sim_cpu_util = 30.0
                self._sim_battery_pct = 85.0
                self._sim_plugged = True
                self._sim_mode = "NORMAL"
            elif profile == "REALTIME":
                self.clear_simulation()
                return
            self._current_state = self._read_hardware()

    def clear_simulation(self) -> None:
        """Reset back to pure realtime hardware sampling."""
        with self._lock:
            self._sim_temperature = None
            self._sim_gpu_util = None
            self._sim_cpu_util = None
            self._sim_battery_pct = None
            self._sim_plugged = None
            self._sim_mode = "REALTIME"
            self._current_state = self._read_hardware()

    # ------------------------------------------------------------------
    # Core Telemetry Reader Logic
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        """Main polling loop running in daemon thread."""
        while not self._stop_event.is_set():
            try:
                new_state = self._read_hardware()
                with self._lock:
                    self._current_state = new_state
            except Exception as e:
                logger.error(f"Telemetry read error: {e}")
            time.sleep(self.sample_interval)

    def _read_hardware(self) -> HardwareState:
        """Gather combined system and simulated metrics."""
        now = time.time()

        # 1. CPU & RAM Metrics
        if HAS_PSUTIL:
            cpu_util = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            ram_used = round(mem.used / (1024 ** 3), 2)
            ram_total = round(mem.total / (1024 ** 3), 2)
            ram_util = mem.percent

            # Battery
            battery = psutil.sensors_battery()
            battery_pct = battery.percent if battery else None
            power_plugged = battery.power_plugged if battery else True
        else:
            cpu_util = 25.0
            ram_used = 8.0
            ram_total = 16.0
            ram_util = 50.0
            battery_pct = None
            power_plugged = True

        # Override CPU if simulated
        if self._sim_cpu_util is not None:
            cpu_util = self._sim_cpu_util

        # Override Battery if simulated
        if self._sim_battery_pct is not None:
            battery_pct = self._sim_battery_pct
        if self._sim_plugged is not None:
            power_plugged = self._sim_plugged

        # 2. Simulated / Estimated GPU & VRAM Metrics
        if self._sim_gpu_util is not None:
            gpu_util = self._sim_gpu_util
        else:
            # Estimate GPU load correlated with CPU load + slight random noise
            gpu_util = max(5.0, min(99.0, cpu_util * 1.1 + math.sin(now) * 3.0))

        vram_total = 16.0 # Standard 16GB VRAM pool simulation
        vram_used = round(3.5 + (gpu_util / 100.0) * 8.5, 2)
        vram_util = round((vram_used / vram_total) * 100.0, 1)

        # 3. Thermal Dynamics Calculation
        if self._sim_temperature is not None:
            target_temp = self._sim_temperature
        else:
            # Thermal accumulation model based on CPU+GPU load
            combined_load = (cpu_util * 0.4 + gpu_util * 0.6) / 100.0
            target_temp = T_IDLE + combined_load * (T_MAX - T_IDLE)

        # Apply smooth thermal inertia (heat dissipation delay)
        alpha = 0.15 # Smoothing factor
        self._smoothed_temp += alpha * (target_temp - self._smoothed_temp)
        current_temp = round(self._smoothed_temp, 1)

        # Thermal headroom formula: 1.0 (Cool / 38°C) to 0.0 (Hot / 85°C+)
        thermal_headroom = max(0.0, min(1.0, (T_MAX - current_temp) / (T_MAX - T_IDLE)))
        thermal_headroom = round(thermal_headroom, 3)

        # 4. Power Draw Model (Watts)
        # Power = Base Idle (15W) + Dynamic CPU Power (up to 45W) + Dynamic GPU Power (up to 120W) + Thermal Leakage
        thermal_leakage = max(0.0, (current_temp - 60.0) * 0.8) # Increased leakage at high temps
        cpu_power = (cpu_util / 100.0) * 45.0
        gpu_power = (gpu_util / 100.0) * 120.0
        target_power = 15.0 + cpu_power + gpu_power + thermal_leakage

        self._smoothed_power += 0.2 * (target_power - self._smoothed_power)
        power_draw_w = round(self._smoothed_power, 2)

        is_sim = (self._sim_mode != "REALTIME")

        return HardwareState(
            cpu_utilization=round(cpu_util, 1),
            gpu_utilization=round(gpu_util, 1),
            ram_used_gb=ram_used,
            ram_total_gb=ram_total,
            ram_utilization=round(ram_util, 1),
            vram_used_gb=vram_used,
            vram_total_gb=vram_total,
            vram_utilization=vram_util,
            temperature_c=current_temp,
            thermal_headroom=thermal_headroom,
            power_draw_w=power_draw_w,
            battery_percent=battery_pct,
            power_plugged=power_plugged,
            timestamp=now,
            is_simulated=is_sim,
            simulation_mode=self._sim_mode
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing TelemetryMonitor...")
    monitor = TelemetryMonitor(sample_interval=0.1)
    monitor.start()
    time.sleep(0.5)

    print("Initial Realtime Hardware State:")
    print(monitor.get_state().to_dict())

    print("\nSimulating Thermal Throttling (82.5°C)...")
    monitor.set_power_profile("THERMAL_THROTTLING")
    time.sleep(0.5)
    print(monitor.get_state().to_dict())

    print("\nResetting to Realtime...")
    monitor.clear_simulation()
    time.sleep(0.3)
    print(monitor.get_state().to_dict())

    monitor.stop()
    print("Telemetry test completed successfully!")
