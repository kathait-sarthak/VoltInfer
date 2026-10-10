/**
 * VoltInfer Web Dashboard Application Engine
 * Handles live telemetry streaming, layer visualizer rendering, hardware stress simulations,
 * and OpenAI-compatible proxy requests.
 */

const API_BASE = window.location.origin;

// State management
let currentTelemetry = null;
let lastInferenceResult = null;
let pollTimer = null;

// Initialize on document ready
document.addEventListener("DOMContentLoaded", () => {
  renderLayerGrid(32, 32, "FP16", 32, false);
  startTelemetryPolling();
});

/**
 * Start high-frequency background polling of /v1/telemetry
 */
function startTelemetryPolling() {
  fetchTelemetry();
  pollTimer = setInterval(fetchTelemetry, 500);
}

async function fetchTelemetry() {
  try {
    const res = await fetch(`${API_BASE}/v1/telemetry`);
    if (!res.ok) throw new Error("Telemetry offline");
    const data = await res.json();
    currentTelemetry = data.telemetry;
    updateTelemetryUI(data.telemetry);

    // Fetch Model info
    const modelRes = await fetch(`${API_BASE}/v1/model/info`);
    if (modelRes.ok) {
      const modelData = await modelRes.json();
      const info = modelData.model;
      const modelLabel = info.is_loaded ? `${info.model_name} (${info.backend_type})` : `${currentTelemetry.simulation_mode} MON`;
      document.getElementById("mode-text").innerText = modelLabel;

      // Update Banner Bar
      document.getElementById("model-path-display").innerText = info.model_path;
      document.getElementById("model-backend-display").innerText = info.is_loaded ? `${info.backend_type} (Real Weights)` : "VoltInfer Hardware Engine (Adaptive)";
      document.getElementById("model-layers-display").innerText = `${info.total_layers} Layers`;
    }
  } catch (err) {
    document.getElementById("server-status-text").innerText = "Proxy Offline";
    document.querySelector(".status-indicator").classList.remove("online");
  }
}

/**
 * Update telemetry cards, progress bars, and values
 */
function updateTelemetryUI(state) {
  // Server Status
  document.getElementById("server-status-text").innerText = "Proxy Active (Port 8000)";
  document.querySelector(".status-indicator").classList.add("online");

  // Mode badge
  document.getElementById("mode-text").innerText = `${state.simulation_mode} MON`;

  // 1. Core Temperature & Headroom
  const tempVal = state.temperature_c.toFixed(1);
  document.getElementById("temp-value").innerText = tempVal;
  
  const headroomPct = (state.thermal_headroom * 100).toFixed(1);
  document.getElementById("thermal-headroom-badge").innerText = `${headroomPct}% Headroom`;

  const tempBar = document.getElementById("temp-bar");
  const tempRatio = Math.max(0, Math.min(100, ((state.temperature_c - 38) / (85 - 38)) * 100));
  tempBar.style.width = `${tempRatio}%`;

  const thermalStatusEl = document.getElementById("thermal-status");
  if (state.temperature_c >= 80) {
    thermalStatusEl.innerText = "Severe Throttling";
    thermalStatusEl.style.color = "var(--accent-red)";
  } else if (state.temperature_c >= 65) {
    thermalStatusEl.innerText = "Thermal Pressure";
    thermalStatusEl.style.color = "var(--accent-yellow)";
  } else {
    thermalStatusEl.innerText = "Cool Headroom";
    thermalStatusEl.style.color = "var(--accent-green)";
  }

  // 2. Power Draw
  document.getElementById("power-value").innerText = state.power_draw_w.toFixed(1);
  const powerRatio = Math.max(0, Math.min(100, (state.power_draw_w / 180.0) * 100));
  document.getElementById("power-bar").style.width = `${powerRatio}%`;

  const thermalLeakage = Math.max(0, (state.temperature_c - 50.0) * 0.75).toFixed(1);
  document.getElementById("leakage-value").innerText = `+${thermalLeakage} W`;

  // 3. System Load (CPU & GPU)
  document.getElementById("cpu-value").innerText = `${state.cpu_utilization}%`;
  document.getElementById("cpu-bar").style.width = `${state.cpu_utilization}%`;

  document.getElementById("gpu-value").innerText = `${state.gpu_utilization}%`;
  document.getElementById("gpu-bar").style.width = `${state.gpu_utilization}%`;

  // 4. Memory Pools & Battery
  document.getElementById("ram-value").innerText = `${state.ram_used_gb} / ${state.ram_total_gb} GB`;
  document.getElementById("ram-bar").style.width = `${state.ram_utilization}%`;

  document.getElementById("vram-value").innerText = `${state.vram_used_gb} / ${state.vram_total_gb} GB`;
  document.getElementById("vram-bar").style.width = `${state.vram_utilization}%`;

  const batBadge = document.getElementById("battery-badge");
  if (state.battery_percent !== null && state.battery_percent !== undefined) {
    const src = state.power_plugged ? "AC Charging" : "Battery";
    batBadge.innerText = `${state.battery_percent}% (${src})`;
  } else {
    batBadge.innerText = "AC Connected";
  }

  // Update Dynamic Layer Grid preview if no active prompt result yet
  if (!lastInferenceResult) {
    updateEstimatedLayerGrid(state);
  }
}

/**
 * Predict active execution profile based on current telemetry state
 */
function updateEstimatedLayerGrid(state) {
  const p_h = 0.50 * (1.0 - state.thermal_headroom) + 0.35 * (state.gpu_utilization / 100.0);
  const layers = Math.max(10, Math.min(32, Math.round(32 * (1.0 - 0.55 * p_h))));
  const quant = p_h > 0.65 ? "INT4" : (p_h > 0.35 ? "INT8" : "FP16");
  const kv_heads = Math.max(8, Math.round(32 * (1.0 - 0.6 * p_h)));
  const early_exit = layers < 32;

  renderLayerGrid(layers, 32, quant, kv_heads, early_exit);
}

/**
 * Render 32 Transformer Layer Execution Blocks
 */
function renderLayerGrid(activeLayers, totalLayers, quant, activeHeads, earlyExit) {
  const container = document.getElementById("layers-container");
  container.innerHTML = "";

  // Update Stats Pills
  document.getElementById("active-layers-pill").innerHTML = `<i class="fa-solid fa-bars-staggered"></i> Active Layers: <strong>${activeLayers} / ${totalLayers}</strong>`;
  document.getElementById("quant-pill").innerHTML = `<i class="fa-solid fa-microchip"></i> Precision: <strong>${quant}</strong>`;
  document.getElementById("kv-pill").innerHTML = `<i class="fa-solid fa-sliders"></i> Active KV Heads: <strong>${activeHeads} / 32</strong>`;

  const pathPill = document.getElementById("path-mode-pill");
  if (earlyExit) {
    pathPill.className = "pill pill-green";
    pathPill.innerHTML = `<i class="fa-solid fa-bolt"></i> <strong>Fast Path (Early Exit Layer ${activeLayers})</strong>`;
  } else {
    pathPill.className = "pill";
    pathPill.innerHTML = `<i class="fa-solid fa-circle-check"></i> <strong>Full Depth Path (FP16 32L)</strong>`;
  }

  for (let i = 1; i <= totalLayers; i++) {
    const block = document.createElement("div");
    block.className = "layer-block";

    const isExecuted = i <= activeLayers;
    const isExitGate = (i === activeLayers && earlyExit);

    if (isExitGate) {
      block.classList.add("early-exit-gate");
      block.innerHTML = `
        <span class="layer-num">L${i}</span>
        <span class="layer-type"><i class="fa-solid fa-right-from-bracket"></i> EXIT</span>
      `;
    } else if (isExecuted) {
      const clsMap = { "INT4": "active-int4", "INT8": "active-int8", "FP16": "active-fp16" };
      block.classList.add(clsMap[quant] || "active-fp16");
      block.innerHTML = `
        <span class="layer-num">L${i}</span>
        <span class="layer-type">${quant}</span>
      `;
    } else {
      block.classList.add("skipped-layer");
      block.innerHTML = `
        <span class="layer-num">L${i}</span>
        <span class="layer-type">SKIPPED</span>
      `;
    }

    container.appendChild(block);
  }
}

/**
 * Control API - Set Hardware Power Profile
 */
async function setProfile(profileName) {
  try {
    const res = await fetch(`${API_BASE}/v1/telemetry/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile: profileName })
    });
    const data = await res.json();
    currentTelemetry = data.telemetry;
    updateTelemetryUI(data.telemetry);
  } catch (err) {
    console.error("Profile set failed:", err);
  }
}

/**
 * Control API - Manual Temperature Slider Override
 */
async function updateTempSlider(val) {
  document.getElementById("slider-temp-val").innerText = `${parseFloat(val).toFixed(1)} °C`;
  try {
    const res = await fetch(`${API_BASE}/v1/telemetry/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ temperature_c: parseFloat(val) })
    });
    const data = await res.json();
    currentTelemetry = data.telemetry;
    updateTelemetryUI(data.telemetry);
  } catch (err) {
    console.error("Temp slider update failed:", err);
  }
}

/**
 * Control API - Reset to Pure Hardware Telemetry
 */
async function resetTelemetry() {
  try {
    const res = await fetch(`${API_BASE}/v1/telemetry/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({})
    });
    const data = await res.json();
    lastInferenceResult = null;
    currentTelemetry = data.telemetry;
    updateTelemetryUI(data.telemetry);
  } catch (err) {
    console.error("Reset telemetry failed:", err);
  }
}

/**
 * Submit Prompt Request to OpenAI-compatible Endpoint
 */
async function submitPrompt() {
  const promptInput = document.getElementById("prompt-input").value;
  if (!promptInput.trim()) return;

  const btn = document.getElementById("btn-submit-prompt");
  btn.disabled = true;
  btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Processing VoltInfer Proxy...`;

  try {
    const response = await fetch(`${API_BASE}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "volt-infer-adaptive-v1",
        messages: [{ role: "user", content: promptInput }],
        max_tokens: 120
      })
    });

    const data = await response.json();
    const result = data.volt_infer_telemetry;
    const textOutput = data.choices[0].message.content;

    lastInferenceResult = result;

    // Render exact executed layers grid
    renderLayerGrid(
      result.executed_layers,
      result.total_layers,
      result.quantization,
      result.active_kv_heads,
      result.early_exit
    );

    // Display Response Container
    const container = document.getElementById("response-container");
    container.classList.remove("hidden");

    document.getElementById("res-mode-badge").innerText = result.execution_mode;
    document.getElementById("response-text-box").innerText = textOutput;

    // Side-by-Side Comparison Stats
    document.getElementById("base-latency-val").innerText = `${result.baseline_latency_sec.toFixed(3)} s`;
    document.getElementById("base-power-val").innerText = `${(result.power_draw_w * 1.55).toFixed(1)} W`;
    document.getElementById("base-energy-val").innerText = `${(result.baseline_latency_sec * result.power_draw_w * 1.55).toFixed(2)} J`;

    document.getElementById("volt-layers-val").innerText = `${result.executed_layers} / ${result.total_layers} Layers`;
    document.getElementById("volt-quant-val").innerText = `${result.quantization} Precision`;
    document.getElementById("volt-kv-val").innerText = `${result.active_kv_heads} / ${result.total_kv_heads} Heads`;
    document.getElementById("volt-latency-val").innerText = `${result.latency_sec.toFixed(3)} s`;
    document.getElementById("volt-power-val").innerText = `${result.power_draw_w.toFixed(1)} W`;
    document.getElementById("volt-energy-val").innerText = `${result.energy_joules.toFixed(2)} J`;

    // Micro Deltas Calculations
    const latencyDiff = (result.latency_sec - result.baseline_latency_sec).toFixed(3);
    const latencyPct = (((result.baseline_latency_sec - result.latency_sec) / result.baseline_latency_sec) * 100).toFixed(1);
    document.getElementById("delta-latency").innerText = `${latencyDiff}s (-${latencyPct}%)`;

    const powerDiff = (result.power_draw_w - (result.power_draw_w * 1.55)).toFixed(1);
    const powerPct = (((result.power_draw_w * 1.55 - result.power_draw_w) / (result.power_draw_w * 1.55)) * 100).toFixed(1);
    document.getElementById("delta-power").innerText = `${powerDiff} W (-${powerPct}%)`;

    document.getElementById("delta-joules").innerText = `-${result.joules_saved.toFixed(2)} J (${result.energy_saved_pct}% Saved)`;
    
    const heatAvoided = (result.hardware_pressure * 8.5).toFixed(1);
    document.getElementById("delta-thermal").innerText = `-${heatAvoided} °C Heat Avoided`;

  } catch (err) {
    console.error("Inference request failed:", err);
    alert("Inference request failed: " + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i class="fa-solid fa-play"></i> Execute VoltInfer Request`;
  }
}

/**
 * Trigger automated side-by-side benchmark test from UI
 */
async function runAutomatedBenchmark() {
  alert("Running side-by-side thermal benchmark suite across 4 scenarios...");
  
  // Cycle through simulation profiles sequentially to demonstrate live visual state transitions
  const profiles = ["PEAK_PERFORMANCE", "NORMAL", "THERMAL_THROTTLING", "BATTERY_SAVER"];
  for (const prof of profiles) {
    await setProfile(prof);
    await submitPrompt();
    await new Promise(r => setTimeout(r, 1200));
  }
}
