"""
Unit tests for VoltInfer Benchmark Suite.
"""

import os
import pytest
from bench.run_benchmark import TEST_PROMPTS, run_benchmark_suite


def test_benchmark_prompts_configured():
    assert len(TEST_PROMPTS) >= 4
    for item in TEST_PROMPTS:
        assert "id" in item
        assert "prompt" in item
        assert "category" in item


def test_run_benchmark_suite():
    results = run_benchmark_suite()
    assert isinstance(results, list)
    assert len(results) > 0
    
    first = results[0]
    assert "scenario" in first
    assert "latency_sec" in first
    assert "energy_saved_pct" in first
    assert "executed_layers" in first
    assert "joules_saved" in first
