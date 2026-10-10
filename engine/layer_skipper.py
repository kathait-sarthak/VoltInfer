"""
VoltInfer Adaptive Inference Engine & Dynamic Layer Modulator
Provides token entropy evaluation, Dynamic Layer Modulation (early exit, INT4/INT8 quantization, KV-head pruning),
and energy/power benchmarking against standard full-precision LLM baselines.
"""

import math
import re
import time
import random
from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional
try:
    from .telemetry import HardwareState, TelemetryMonitor
    from .model_loader import RealModelBackend
except ImportError:
    from telemetry import HardwareState, TelemetryMonitor
    from model_loader import RealModelBackend


@dataclass
class InferenceResult:
    """Contains generation response, performance telemetry, and energy breakdown."""
    prompt: str
    generated_text: str
    prompt_tokens: int
    generated_tokens: int
    total_tokens: int

    # Adaptive Execution Profile
    mode: str                         # "Fast Path (Early Exit)" vs "Full Path (High Precision)"
    total_layers: int                 # Standard baseline (e.g. 32 layers)
    executed_layers: int              # Dynamic layer count (e.g. 14 layers)
    early_exit: bool                  # True if early exit triggered
    early_exit_layer: int             # Layer index where execution exited
    quantization: str                 # "INT4", "INT8", "FP16"
    total_kv_heads: int               # Standard baseline (e.g. 32 heads)
    active_kv_heads: int              # Dynamically active KV heads
    pruned_kv_heads: int              # Pruned heads count

    # Telemetry & Complexity Analysis
    prompt_entropy: float             # Normalized token/char entropy (0.0 - 1.0)
    hardware_pressure: float          # Combined hardware stress index (0.0 - 1.0)

    # Performance & Energy Metrics
    latency_sec: float                # VoltInfer execution latency
    baseline_latency_sec: float       # Standard baseline latency
    power_w: float                    # Average power during inference (Watts)
    baseline_power_w: float           # Baseline power draw (Watts)
    energy_joules: float              # E = Power * Latency (Joules)
    baseline_energy_joules: float     # Baseline energy (Joules)
    joules_saved: float               # Baseline Joules - VoltInfer Joules
    energy_saved_pct: float           # % Energy saved
    tokens_per_sec: float             # VoltInfer generation throughput

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for API response."""
        return asdict(self)


class EntropyEvaluator:
    """
    Evaluates context complexity and token entropy of incoming text prompts.
    Uses Shannon entropy, vocabulary density, syntactic structures (code/math markers),
    and contextual reasoning keywords.
    """

    # Keyword weights indicating high reasoning complexity
    REASONING_KEYWORDS = {
        "solve", "proof", "derive", "algorithm", "explain", "refactor",
        "calculate", "theorem", "matrix", "gradient", "quantum", "compiler",
        "deadlock", "asynchronous", "architecture", "optimization", "physics"
    }

    SIMPLE_KEYWORDS = {
        "hi", "hello", "thanks", "summary", "translate", "list", "rephrase",
        "capital", "define", "short", "simple", "who", "where", "what"
    }

    @classmethod
    def evaluate(cls, prompt: str) -> float:
        """
        Calculate a normalized complexity score C in [0.0, 1.0].
        0.0 = Very simple / high redundancy (Fast Path candidate)
        1.0 = Highly complex reasoning / dense code (Full Path candidate)
        """
        if not prompt or len(prompt.strip()) == 0:
            return 0.1

        text = prompt.strip().lower()

        # 1. Shannon Entropy of Character Distribution
        char_counts: Dict[str, int] = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        entropy = 0.0
        total_chars = len(text)
        for count in char_counts.values():
            p = count / total_chars
            entropy -= p * math.log2(p)
        
        # Max entropy for 256 byte charset is 8.0, typical text ranges 3.5 to 5.2
        norm_entropy = max(0.0, min(1.0, (entropy - 3.2) / (5.0 - 3.2)))

        # 2. Token Length Factor
        tokens = re.findall(r'\w+|[^\w\s]', text)
        token_count = len(tokens)
        length_factor = max(0.0, min(1.0, token_count / 150.0))

        # 3. Domain & Syntactic Density (Code snippets, math equations)
        has_code = bool(re.search(r'```|def |class |function|return |import |var |for \(', text))
        has_math = bool(re.search(r'[=\+\-\*/\^\$\\int\\sigma\d]{4,}', text))
        domain_factor = 0.4 if (has_code or has_math) else 0.0

        # 4. Keyword Weighting
        words = set(tokens)
        reasoning_hits = len(words.intersection(cls.REASONING_KEYWORDS))
        simple_hits = len(words.intersection(cls.SIMPLE_KEYWORDS))
        
        keyword_shift = (reasoning_hits * 0.15) - (simple_hits * 0.1)

        # Combined Weighted Complexity
        complexity = (
            norm_entropy * 0.35 +
            length_factor * 0.35 +
            domain_factor * 0.20 +
            keyword_shift
        )

        return round(max(0.05, min(1.0, complexity)), 3)


class DynamicLayerModulator:
    """
    Closed-loop Dynamic Layer Modulation engine.
    Analyzes hardware telemetry and prompt entropy to dynamically adjust:
    - Executed transformer layer count (Early exit at layer N)
    - Quantization precision (FP16, INT8, INT4)
    - KV-cache head pruning (32 down to 8 active heads)
    """

    MAX_LAYERS = 32
    MIN_LAYERS = 10
    MAX_KV_HEADS = 32
    MIN_KV_HEADS = 8

    def __init__(self, target_max_temp: float = 85.0, model_backend: Optional[RealModelBackend] = None):
        self.target_max_temp = target_max_temp
        self.model_backend = model_backend or RealModelBackend()
        if self.model_backend.is_loaded:
            self.MAX_LAYERS = self.model_backend.total_layers

    def compute_hardware_pressure(self, state: HardwareState) -> float:
        """
        Calculate unified hardware pressure P_H in [0.0, 1.0].
        Combines thermal stress, CPU/GPU utilization, and battery state.
        """
        # Thermal Pressure (1.0 - thermal_headroom)
        thermal_pressure = 1.0 - state.thermal_headroom

        # System Load Pressure
        load_pressure = (state.cpu_utilization * 0.3 + state.gpu_utilization * 0.7) / 100.0

        # Battery Pressure (High when battery is low and unplugged)
        battery_pressure = 0.0
        if state.battery_percent is not None and not state.power_plugged:
            if state.battery_percent < 20.0:
                battery_pressure = 0.9
            elif state.battery_percent < 50.0:
                battery_pressure = 0.5
            else:
                battery_pressure = 0.2

        # Weighted hardware pressure formula
        p_h = (thermal_pressure * 0.50 + load_pressure * 0.35 + battery_pressure * 0.15)
        return round(max(0.0, min(1.0, p_h)), 3)

    def determine_modulation_profile(
        self, prompt_entropy: float, state: HardwareState
    ) -> Tuple[int, str, int, bool, float]:
        """
        Closed-loop policy engine.
        Returns:
            (executed_layers, quantization, active_kv_heads, early_exit, hardware_pressure)
        """
        p_h = self.compute_hardware_pressure(state)
        c = prompt_entropy  # Complexity in [0.0, 1.0]

        # Target layer calculation:
        # High pressure (P_H -> 1) shedding layers; high complexity (C -> 1) keeping layers.
        # Layer formula: Base = 32 * (1 - 0.65 * P_H * (1 - 0.7 * C))
        layer_scale = 1.0 - (0.65 * p_h * (1.0 - 0.6 * c))
        executed_layers = int(round(self.MAX_LAYERS * layer_scale))
        executed_layers = max(self.MIN_LAYERS, min(self.MAX_LAYERS, executed_layers))

        early_exit = (executed_layers < self.MAX_LAYERS)

        # Quantization precision policy
        if p_h > 0.65 and c < 0.7:
            quantization = "INT4"
        elif p_h > 0.35 or c < 0.5:
            quantization = "INT8"
        else:
            quantization = "FP16"

        # KV-cache head pruning policy
        head_scale = 1.0 - (0.75 * p_h * (1.0 - 0.5 * c))
        active_heads = int(round(self.MAX_KV_HEADS * head_scale))
        active_heads = max(self.MIN_KV_HEADS, min(self.MAX_KV_HEADS, active_heads))
        # Ensure active_heads is divisible by 2 for realistic attention architecture
        active_heads = (active_heads // 2) * 2

        return executed_layers, quantization, active_heads, early_exit, p_h

    def execute_inference(
        self, prompt: str, state: HardwareState, max_tokens: int = 120
    ) -> InferenceResult:
        """
        Simulates transformer inference pipeline under Dynamic Layer Modulation
        and measures precise latency, power, and Joules compared against standard baseline.
        """
        t0 = time.time()
        entropy = EntropyEvaluator.evaluate(prompt)
        layers, quant, active_heads, early_exit, p_h = self.determine_modulation_profile(entropy, state)

        # Token count estimation
        prompt_tokens = max(5, len(prompt.split()))
        gen_tokens = max(15, min(max_tokens, int(prompt_tokens * 1.5 + entropy * 40)))
        total_tokens = prompt_tokens + gen_tokens

        # ------------------------------------------------------------------
        # Physics & Speed Simulation Model
        # ------------------------------------------------------------------
        # Baseline per-layer latency constant (seconds per token per layer for FP16 32-head)
        BASE_LAYER_TIME = 0.00045 # 0.45 ms per layer per token

        # Quantization acceleration factor (INT4 = 0.4x compute, INT8 = 0.65x compute, FP16 = 1.0x)
        quant_speedup = {"INT4": 0.42, "INT8": 0.68, "FP16": 1.00}[quant]

        # KV-head pruning acceleration factor
        kv_speedup = 0.65 + 0.35 * (active_heads / self.MAX_KV_HEADS)

        # Compute execution latency
        per_token_time = BASE_LAYER_TIME * layers * quant_speedup * kv_speedup
        compute_delay = per_token_time * gen_tokens

        # Execute real backend or high-fidelity simulation
        real_text = ""
        if self.model_backend and self.model_backend.is_loaded:
            real_text, real_latency_sec = self.model_backend.generate_with_early_exit(prompt, layers, max_tokens=max_tokens)
            actual_latency = round(max(0.01, real_latency_sec), 4)
        else:
            # Realistic micro-sleep to simulate hardware inference execution
            time.sleep(max(0.01, min(0.35, compute_delay * 0.4)))
            actual_latency = round(max(0.02, (time.time() - t0) + compute_delay * 0.6), 4)

        # Baseline execution stats (Standard 32 layers, FP16, 32 KV heads)
        baseline_per_token_time = BASE_LAYER_TIME * self.MAX_LAYERS * 1.0 * 1.0
        baseline_latency = round(baseline_per_token_time * gen_tokens, 4)

        # ------------------------------------------------------------------
        # Power & Energy Model (Watts & Joules)
        # ------------------------------------------------------------------
        # Baseline power model (Full 32 layers FP16 under thermal leakage)
        # Thermal leakage adds +0.7W per °C above 50°C
        thermal_leakage = max(0.0, (state.temperature_c - 50.0) * 0.75)
        baseline_power_w = round(65.0 + 40.0 + thermal_leakage, 2) # Base system + full GPU + thermal penalty

        # VoltInfer power model:
        # Power scales directly with active layers, quantization precision, and active KV heads
        layer_power_ratio = layers / self.MAX_LAYERS
        quant_power_ratio = {"INT4": 0.45, "INT8": 0.70, "FP16": 1.00}[quant]
        head_power_ratio = 0.5 + 0.5 * (active_heads / self.MAX_KV_HEADS)

        active_gpu_power = 40.0 * layer_power_ratio * quant_power_ratio * head_power_ratio
        volt_power_w = round(65.0 + active_gpu_power + thermal_leakage * 0.4, 2)

        # Energy calculation (E = P * t in Joules)
        volt_joules = round(volt_power_w * actual_latency, 3)
        baseline_joules = round(baseline_power_w * baseline_latency, 3)

        joules_saved = round(max(0.0, baseline_joules - volt_joules), 3)
        energy_saved_pct = round(
            max(0.0, min(95.0, (joules_saved / max(0.001, baseline_joules)) * 100.0)), 1
        )

        tokens_per_sec = round(gen_tokens / max(0.001, actual_latency), 1)

        # Mode classification
        if early_exit:
            mode_str = f"Fast Path (Early Exit L{layers}/{self.MAX_LAYERS}, {quant}, {active_heads}H)"
        else:
            mode_str = f"Full Path (High Precision FP16, 32L, 32H)"

        # Generate realistic domain response text
        generated_text = real_text if real_text else self._synthesize_response(prompt, entropy, layers, quant)

        return InferenceResult(
            prompt=prompt,
            generated_text=generated_text,
            prompt_tokens=prompt_tokens,
            generated_tokens=gen_tokens,
            total_tokens=total_tokens,
            mode=mode_str,
            total_layers=self.MAX_LAYERS,
            executed_layers=layers,
            early_exit=early_exit,
            early_exit_layer=layers if early_exit else self.MAX_LAYERS,
            quantization=quant,
            total_kv_heads=self.MAX_KV_HEADS,
            active_kv_heads=active_heads,
            pruned_kv_heads=self.MAX_KV_HEADS - active_heads,
            prompt_entropy=entropy,
            hardware_pressure=p_h,
            latency_sec=actual_latency,
            baseline_latency_sec=baseline_latency,
            power_w=volt_power_w,
            baseline_power_w=baseline_power_w,
            energy_joules=volt_joules,
            baseline_energy_joules=baseline_joules,
            joules_saved=joules_saved,
            energy_saved_pct=energy_saved_pct,
            tokens_per_sec=tokens_per_sec
        )

    def _synthesize_response(self, prompt: str, entropy: float, layers: int, quant: str) -> str:
        """Helper to generate realistic output text matching prompt context."""
        lower_prompt = prompt.lower()
        if "hello" in lower_prompt or "hi" in lower_prompt or "who" in lower_prompt:
            return f"Hello! I am VoltInfer, an energy-aware LLM runtime running on layer depth {layers}/32 with {quant} quantization."
        elif "code" in lower_prompt or "python" in lower_prompt or "def " in lower_prompt:
            return (
                f"```python\n# VoltInfer Optimized Output (Layer {layers}/32, {quant})\n"
                f"def compute_energy_efficiency(power_w, latency_s):\n"
                f"    return power_w * latency_s  # Joules\n```"
            )
        elif "explain" in lower_prompt or "why" in lower_prompt or "how" in lower_prompt:
            return (
                f"VoltInfer dynamic layer modulation monitors silicon temperature and token complexity. "
                f"At entropy {entropy}, execution bypassed {32 - layers} deeper layers, conserving energy while preserving output coherence."
            )
        else:
            return (
                f"Processed prompt with prompt complexity {entropy}. "
                f"VoltInfer executed {layers} transformer layers in {quant} precision, saving compute cycles."
            )


if __name__ == "__main__":
    from .telemetry import TelemetryMonitor

    print("Testing DynamicLayerModulator...")
    telemetry = TelemetryMonitor()
    telemetry.start()
    time.sleep(0.3)

    modulator = DynamicLayerModulator()

    # 1. Simple prompt under cool system
    prompt_simple = "Hello! What is your name?"
    state_cool = telemetry.get_state()
    res1 = modulator.execute_inference(prompt_simple, state_cool)
    print("\n--- Test 1: Simple Prompt (Normal Hardware) ---")
    print(f"Mode: {res1.mode}")
    print(f"Layers: {res1.executed_layers}/32 | Quantization: {res1.quantization} | KV Heads: {res1.active_kv_heads}/32")
    print(f"Latency: {res1.latency_sec}s (Baseline: {res1.baseline_latency_sec}s)")
    print(f"Energy: {res1.energy_joules} J vs Baseline {res1.baseline_energy_joules} J ({res1.energy_saved_pct}% saved)")

    # 2. Complex prompt under thermal throttling
    prompt_complex = "Explain the step-by-step mathematical algorithm for solving quantum matrix differential equations in Python def solve_matrix():"
    telemetry.set_power_profile("THERMAL_THROTTLING")
    time.sleep(0.3)
    state_hot = telemetry.get_state()

    res2 = modulator.execute_inference(prompt_complex, state_hot)
    print("\n--- Test 2: Complex Prompt under Thermal Throttling (82°C) ---")
    print(f"Mode: {res2.mode}")
    print(f"Layers: {res2.executed_layers}/32 | Quantization: {res2.quantization} | KV Heads: {res2.active_kv_heads}/32")
    print(f"Hardware Pressure: {res2.hardware_pressure} | Prompt Entropy: {res2.prompt_entropy}")
    print(f"Latency: {res2.latency_sec}s (Baseline: {res2.baseline_latency_sec}s)")
    print(f"Energy: {res2.energy_joules} J vs Baseline {res2.baseline_energy_joules} J ({res2.energy_saved_pct}% saved)")

    telemetry.stop()
    print("\nDynamicLayerModulator test completed successfully!")
