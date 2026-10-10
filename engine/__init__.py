"""
VoltInfer Engine Package
Energy-Aware, Hardware-in-the-Loop LLM Inference Engine & Proxy
"""

def __getattr__(name):
    if name in ("TelemetryMonitor", "HardwareState"):
        from .telemetry import TelemetryMonitor, HardwareState
        return locals()[name]
    elif name in ("DynamicLayerModulator", "EntropyEvaluator", "InferenceResult"):
        from .layer_skipper import DynamicLayerModulator, EntropyEvaluator, InferenceResult
        return locals()[name]
    elif name == "VoltInferProxyServer":
        from .proxy import VoltInferProxyServer
        return VoltInferProxyServer
    elif name == "RealModelBackend":
        from .model_loader import RealModelBackend
        return RealModelBackend
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "TelemetryMonitor",
    "HardwareState",
    "DynamicLayerModulator",
    "EntropyEvaluator",
    "InferenceResult",
    "VoltInferProxyServer",
    "RealModelBackend",
]
