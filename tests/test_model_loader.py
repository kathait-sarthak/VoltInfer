"""
Unit tests for VoltInfer Model Loader & Backend Adapter.
"""

import os
import struct
import tempfile
import pytest
from engine.model_loader import RealModelBackend, parse_gguf_metadata


def test_real_model_backend_simulation_fallback():
    backend = RealModelBackend(model_path=None)
    assert backend.is_loaded is False
    assert backend.backend_type == "SIMULATED"
    assert backend.total_layers == 36
    
    info = backend.get_info()
    assert info["is_loaded"] is False
    assert info["backend_type"] == "SIMULATED"
    assert info["total_layers"] == 36
    assert "Simulated Engine" in info["model_path"]


def test_real_model_backend_generate_with_early_exit():
    backend = RealModelBackend()
    text, elapsed = backend.generate_with_early_exit(
        prompt="Tell me a joke",
        target_layers=20,
        max_tokens=30
    )
    assert isinstance(text, str)
    assert elapsed >= 0.0


def test_parse_gguf_metadata_fallback():
    # Non-existent file returns fallback defaults
    meta = parse_gguf_metadata("non_existent_model.gguf")
    assert meta["architecture"] == "qwen3"
    assert meta["total_layers"] == 36


def test_parse_gguf_metadata_valid_header():
    # Create temporary dummy file with GGUF magic bytes
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
        # Write GGUF magic b'GGUF', version 3, 0 tensors, 0 metadata kv
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))  # version
        f.write(struct.pack("<Q", 0))  # tensor count
        f.write(struct.pack("<Q", 0))  # kv count
        temp_path = f.name

    try:
        meta = parse_gguf_metadata(temp_path)
        assert meta["architecture"] == "qwen3"
        assert meta["name"] == os.path.basename(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
