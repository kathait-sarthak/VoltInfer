"""
Unit tests for VoltInfer Telemetry and Hardware Reader.
"""

import time
import pytest
from engine.telemetry import TelemetryMonitor, HardwareState, T_IDLE, T_MAX


def test_hardware_state_to_dict():
    state = HardwareState(
        cpu_utilization=25.0,
        gpu_utilization=30.0,
        ram_used_gb=4.0,
        ram_total_gb=16.0,
        ram_utilization=25.0,
        vram_used_gb=2.0,
        vram_total_gb=8.0,
        vram_utilization=25.0,
        temperature_c=45.0,
        thermal_headroom=0.85,
        power_draw_w=45.0,
        battery_percent=80.0,
        power_plugged=True,
        timestamp=time.time(),
        is_simulated=False,
        simulation_mode="REALTIME"
    )
    d = state.to_dict()
    assert d["cpu_utilization"] == 25.0
    assert d["temperature_c"] == 45.0
    assert d["thermal_headroom_pct"] == 85.0
    assert d["simulation_mode"] == "REALTIME"
    assert "power_draw_w" in d


def test_telemetry_monitor_sampling():
    monitor = TelemetryMonitor(sample_interval=0.05)
    monitor.start()
    time.sleep(0.15)
    
    state = monitor.get_state()
    assert isinstance(state, HardwareState)
    assert state.temperature_c >= 0.0
    assert 0.0 <= state.thermal_headroom <= 1.0
    assert state.cpu_utilization >= 0.0
    assert state.ram_total_gb > 0.0

    monitor.stop()


def test_telemetry_simulation_profiles():
    monitor = TelemetryMonitor(sample_interval=0.05)
    
    # 1. Thermal Throttling Profile
    monitor.set_power_profile("THERMAL_THROTTLING")
    state = monitor.get_state()
    assert state.is_simulated is True
    assert state.temperature_c >= 80.0
    assert state.thermal_headroom < 0.2
    assert state.simulation_mode == "THERMAL_THROTTLING"

    # 2. Battery Saver Profile
    monitor.set_power_profile("BATTERY_SAVER")
    state = monitor.get_state()
    assert state.is_simulated is True
    assert state.battery_percent <= 20.0
    assert state.power_plugged is False
    assert state.simulation_mode == "BATTERY_SAVER"

    # 3. Peak Performance Profile
    monitor.set_power_profile("PEAK_PERFORMANCE")
    state = monitor.get_state()
    assert state.temperature_c <= 40.0
    assert state.power_plugged is True

    # 4. Clear simulation
    monitor.clear_simulation()
    state = monitor.get_state()
    assert state.is_simulated is False
    assert state.simulation_mode == "REALTIME"


def test_custom_simulation_temperature():
    monitor = TelemetryMonitor(sample_interval=0.05)
    monitor.set_simulated_temperature(77.5)
    state = monitor.get_state()
    assert state.is_simulated is True
    assert state.temperature_c == 77.5
    assert state.simulation_mode in ("HYBRID", "SIMULATED")

    monitor.set_simulated_gpu_load(95.0)
    state = monitor.get_state()
    assert state.gpu_utilization == 95.0
