"""
Unit tests for VoltInfer Dynamic Layer Modulator and Entropy Evaluator.
"""

import time
import pytest
from engine.layer_skipper import DynamicLayerModulator, EntropyEvaluator, InferenceResult
from engine.telemetry import HardwareState


def make_mock_state(temp_c: float = 42.0, headroom: float = 0.9, cpu: float = 20.0, gpu: float = 20.0, battery: float = 90.0, plugged: bool = True) -> HardwareState:
    return HardwareState(
        cpu_utilization=cpu,
        gpu_utilization=gpu,
        ram_used_gb=4.0,
        ram_total_gb=16.0,
        ram_utilization=25.0,
        vram_used_gb=2.0,
        vram_total_gb=8.0,
        vram_utilization=25.0,
        temperature_c=temp_c,
        thermal_headroom=headroom,
        power_draw_w=40.0,
        battery_percent=battery,
        power_plugged=plugged,
        timestamp=time.time(),
        is_simulated=False,
        simulation_mode="REALTIME"
    )


def test_entropy_evaluator():
    # Empty prompt
    assert EntropyEvaluator.evaluate("") >= 0.05
    
    # Simple prompt
    simple_score = EntropyEvaluator.evaluate("hello hi thanks what is capital of france")
    
    # Complex math/code prompt
    complex_prompt = (
        "def quicksort(arr): return arr if len(arr) <= 1 else "
        "quicksort([x for x in arr[1:] if x < arr[0]]) + [arr[0]] + "
        "quicksort([x for x in arr[1:] if x >= arr[0]])\n"
        "Please prove algorithm correctness and derive quantum matrix eigenvalues with gradient descent."
    )
    complex_score = EntropyEvaluator.evaluate(complex_prompt)
    
    assert 0.05 <= simple_score <= 1.0
    assert 0.05 <= complex_score <= 1.0
    assert complex_score > simple_score


def test_hardware_pressure():
    modulator = DynamicLayerModulator()
    
    # Cool state
    cool_state = make_mock_state(temp_c=36.0, headroom=0.95, cpu=10.0, gpu=10.0)
    p_cool = modulator.compute_hardware_pressure(cool_state)
    
    # Severe thermal throttled state
    hot_state = make_mock_state(temp_c=84.0, headroom=0.02, cpu=90.0, gpu=95.0)
    p_hot = modulator.compute_hardware_pressure(hot_state)
    
    assert 0.0 <= p_cool < p_hot <= 1.0
    assert p_hot > 0.6


def test_modulation_profile_layer_shedding():
    modulator = DynamicLayerModulator()
    
    cool_state = make_mock_state(temp_c=36.0, headroom=0.95)
    hot_state = make_mock_state(temp_c=84.0, headroom=0.02, cpu=95.0, gpu=95.0)
    
    simple_entropy = 0.15
    layers_cool, quant_cool, heads_cool, early_exit_cool, _ = modulator.determine_modulation_profile(simple_entropy, cool_state)
    layers_hot, quant_hot, heads_hot, early_exit_hot, _ = modulator.determine_modulation_profile(simple_entropy, hot_state)
    
    # Severe thermal pressure should execute fewer layers than cool state
    assert layers_hot < layers_cool
    assert early_exit_hot is True
    # Thermal stress triggers quantization and KV head pruning
    assert quant_hot in ("INT4", "INT8")
    assert heads_hot <= heads_cool


def test_execute_inference():
    modulator = DynamicLayerModulator()
    state = make_mock_state(temp_c=45.0, headroom=0.85)
    
    result = modulator.execute_inference("Explain renewable solar energy", state, max_tokens=40)
    assert isinstance(result, InferenceResult)
    assert len(result.generated_text) > 0
    assert result.total_layers == 32
    assert result.executed_layers <= 32
    assert result.energy_joules > 0
    assert result.joules_saved >= 0
    assert result.energy_saved_pct >= 0
    assert result.tokens_per_sec > 0
    
    res_dict = result.to_dict()
    assert "prompt_tokens" in res_dict
    assert "generated_tokens" in res_dict
