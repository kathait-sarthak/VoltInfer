/**
 * VoltInfer Web Dashboard Application Engine
 * Handles live telemetry streaming, layer visualizer rendering, hardware stress simulations,
 * and OpenAI-compatible proxy requests.
 */

const API_BASE = window.location.origin
const DEMO_MODE = !["localhost", "127.0.0.1"].includes(window.location.hostname)

// State management
let currentTelemetry = null
let lastInferenceResult = null
let pollTimer = null
let upstreamConfigured = false
let demoTelemetry = {
  temperature_c: 48.0,
  thermal_headroom: 0.78,
  power_draw_w: 65.4,
  cpu_utilization: 25,
  gpu_utilization: 35,
  ram_used_gb: 7.1,
  ram_total_gb: 16.0,
  ram_utilization: 44,
  vram_used_gb: 4.2,
  vram_total_gb: 16.0,
  vram_utilization: 26,
  battery_percent: 85,
  power_plugged: true,
  simulation_mode: "DEMO",
}

// Initialize on document ready
document.addEventListener("DOMContentLoaded", () => {
  renderLayerGrid(32, 32, "FP16", 32, false)
  startTelemetryPolling()
})

/**
 * Start high-frequency background polling of /v1/telemetry
 */
function startTelemetryPolling() {
  fetchTelemetry()
  pollTimer = setInterval(fetchTelemetry, 500)
}

async function fetchTelemetry() {
  if (DEMO_MODE) {
    updateTelemetryUI(demoTelemetry)
    document.getElementById("model-path-display").innerText =
      "Browser Demo (Simulated)"
    document.getElementById("model-backend-display").innerText =
      "VoltInfer Adaptive Demo"
    document.getElementById("model-layers-display").innerText = "32 Layers"
    return
  }

  try {
    const res = await fetch(`${API_BASE}/v1/telemetry`)
    if (!res.ok) throw new Error("Telemetry offline")
    const data = await res.json()
    currentTelemetry = data.telemetry
    updateTelemetryUI(data.telemetry)

    // Fetch Model info
    const modelRes = await fetch(`${API_BASE}/v1/model/info`)
    if (modelRes.ok) {
      const modelData = await modelRes.json()
      const info = modelData.model
      upstreamConfigured = info.backend_type === "OPENAI_COMPATIBLE_UPSTREAM"
      const modelLabel = info.upstream_configured
        ? `${info.upstream_url} (OpenAI-compatible upstream)`
        : info.is_loaded
        ? `${info.model_name || info.upstream_url || "Upstream"} (${info.backend_type})`
        : `${currentTelemetry.simulation_mode} MON`
      document.getElementById("mode-text").innerText = modelLabel

      // Update Banner Bar
      document.getElementById("model-path-display").innerText =
        info.upstream_url || info.model_path || "Demo mode (no upstream)"
      document.getElementById("model-backend-display").innerText =
        upstreamConfigured
          ? "OpenAI-compatible pass-through"
          : info.is_loaded
            ? info.backend_type
            : "Synthetic demo engine (not a real LLM)"
      document.getElementById("model-layers-display").innerText =
        upstreamConfigured
          ? "Not exposed by upstream API"
          : `${info.total_layers || 32} (demo only)`
      document.getElementById("layer-visualizer-section").hidden =
        upstreamConfigured
      document.getElementById("total-saved-callout").innerText =
        upstreamConfigured ? "Not measured" : "Demo estimate"
    }
  } catch (err) {
    document.getElementById("server-status-text").innerText = "Proxy Offline"
    document.querySelector(".status-indicator").classList.remove("online")
  }
}

/**
 * Update telemetry cards, progress bars, and values
 */
function updateTelemetryUI(state) {
  // Server Status
  document.getElementById("server-status-text").innerText = DEMO_MODE
    ? "Demo Simulation"
    : `Proxy Active (Port ${window.location.port || "80"})`
  document.querySelector(".status-indicator").classList.add("online")

  // Mode badge
  document.getElementById("mode-text").innerText =
    `${state.simulation_mode} MON`

  // 1. Core Temperature & Headroom
  const tempVal = state.temperature_c.toFixed(1)
  document.getElementById("temp-value").innerText = tempVal

  const headroomPct = (state.thermal_headroom * 100).toFixed(1)
  document.getElementById("thermal-headroom-badge").innerText =
    `${headroomPct}% Headroom`

  const tempBar = document.getElementById("temp-bar")
  const tempRatio = Math.max(
    0,
    Math.min(100, ((state.temperature_c - 38) / (85 - 38)) * 100),
  )
  tempBar.style.width = `${tempRatio}%`

  const thermalStatusEl = document.getElementById("thermal-status")
  if (state.temperature_c >= 80) {
    thermalStatusEl.innerText = "Severe Throttling"
    thermalStatusEl.style.color = "var(--accent-red)"
  } else if (state.temperature_c >= 65) {
    thermalStatusEl.innerText = "Thermal Pressure"
    thermalStatusEl.style.color = "var(--accent-yellow)"
  } else {
    thermalStatusEl.innerText = "Cool Headroom"
    thermalStatusEl.style.color = "var(--accent-green)"
  }

  // 2. Power Draw
  document.getElementById("power-value").innerText =
    state.power_draw_w.toFixed(1)
  const powerRatio = Math.max(
    0,
    Math.min(100, (state.power_draw_w / 180.0) * 100),
  )
  document.getElementById("power-bar").style.width = `${powerRatio}%`

  const thermalLeakage = Math.max(
    0,
    (state.temperature_c - 50.0) * 0.75,
  ).toFixed(1)
  document.getElementById("leakage-value").innerText = `+${thermalLeakage} W`

  // 3. System Load (CPU & GPU)
  document.getElementById("cpu-value").innerText = `${state.cpu_utilization}%`
  document.getElementById("cpu-bar").style.width = `${state.cpu_utilization}%`

  document.getElementById("gpu-value").innerText = `${state.gpu_utilization}%`
  document.getElementById("gpu-bar").style.width = `${state.gpu_utilization}%`

  // 4. Memory Pools & Battery
  document.getElementById("ram-value").innerText =
    `${state.ram_used_gb} / ${state.ram_total_gb} GB`
  document.getElementById("ram-bar").style.width = `${state.ram_utilization}%`

  document.getElementById("vram-value").innerText =
    `${state.vram_used_gb} / ${state.vram_total_gb} GB`
  document.getElementById("vram-bar").style.width = `${state.vram_utilization}%`

  const batBadge = document.getElementById("battery-badge")
  if (state.battery_percent !== null && state.battery_percent !== undefined) {
    const src = state.power_plugged ? "AC Charging" : "Battery"
    batBadge.innerText = `${state.battery_percent}% (${src})`
  } else {
    batBadge.innerText = "AC Connected"
  }

  // Update Dynamic Layer Grid preview if no active prompt result yet
  if (!lastInferenceResult) {
    updateEstimatedLayerGrid(state)
  }
}

/**
 * Predict active execution profile based on current telemetry state
 */
function updateEstimatedLayerGrid(state) {
  const p_h =
    0.5 * (1.0 - state.thermal_headroom) +
    0.35 * (state.gpu_utilization / 100.0)
  const layers = Math.max(10, Math.min(32, Math.round(32 * (1.0 - 0.55 * p_h))))
  const quant = p_h > 0.65 ? "INT4" : p_h > 0.35 ? "INT8" : "FP16"
  const kv_heads = Math.max(8, Math.round(32 * (1.0 - 0.6 * p_h)))
  const early_exit = layers < 32

  renderLayerGrid(layers, 32, quant, kv_heads, early_exit)
}

/**
 * Render 32 Transformer Layer Execution Blocks
 */
function renderLayerGrid(
  activeLayers,
  totalLayers,
  quant,
  activeHeads,
  earlyExit,
) {
  const container = document.getElementById("layers-container")
  container.innerHTML = ""

  // Update Stats Pills
  document.getElementById("active-layers-pill").innerHTML =
    `<i class="fa-solid fa-bars-staggered"></i> Active Layers: <strong>${activeLayers} / ${totalLayers}</strong>`
  document.getElementById("quant-pill").innerHTML =
    `<i class="fa-solid fa-microchip"></i> Precision: <strong>${quant}</strong>`
  document.getElementById("kv-pill").innerHTML =
    `<i class="fa-solid fa-sliders"></i> Active KV Heads: <strong>${activeHeads} / 32</strong>`

  const pathPill = document.getElementById("path-mode-pill")
  if (earlyExit) {
    pathPill.className = "pill pill-green"
    pathPill.innerHTML = `<i class="fa-solid fa-bolt"></i> <strong>Fast Path (Early Exit Layer ${activeLayers})</strong>`
  } else {
    pathPill.className = "pill"
    pathPill.innerHTML = `<i class="fa-solid fa-circle-check"></i> <strong>Full Depth Path (FP16 32L)</strong>`
  }

  for (let i = 1; i <= totalLayers; i++) {
    const block = document.createElement("div")
    block.className = "layer-block"

    const isExecuted = i <= activeLayers
    const isExitGate = i === activeLayers && earlyExit

    if (isExitGate) {
      block.classList.add("early-exit-gate")
      block.innerHTML = `
        <span class="layer-num">L${i}</span>
        <span class="layer-type"><i class="fa-solid fa-right-from-bracket"></i> EXIT</span>
      `
    } else if (isExecuted) {
      const clsMap = {
        INT4: "active-int4",
        INT8: "active-int8",
        FP16: "active-fp16",
      }
      block.classList.add(clsMap[quant] || "active-fp16")
      block.innerHTML = `
        <span class="layer-num">L${i}</span>
        <span class="layer-type">${quant}</span>
      `
    } else {
      block.classList.add("skipped-layer")
      block.innerHTML = `
        <span class="layer-num">L${i}</span>
        <span class="layer-type">SKIPPED</span>
      `
    }

    container.appendChild(block)
  }
}

/**
 * Control API - Set Hardware Power Profile
 */
async function setProfile(profileName) {
  if (DEMO_MODE) {
    const profiles = {
      PEAK_PERFORMANCE: {
        temperature_c: 36,
        power_draw_w: 92,
        cpu_utilization: 42,
        gpu_utilization: 58,
        battery_percent: 100,
        power_plugged: true,
      },
      NORMAL: {
        temperature_c: 48,
        power_draw_w: 65.4,
        cpu_utilization: 25,
        gpu_utilization: 35,
        battery_percent: 85,
        power_plugged: true,
      },
      THERMAL_THROTTLING: {
        temperature_c: 82.5,
        power_draw_w: 118,
        cpu_utilization: 88,
        gpu_utilization: 94,
        battery_percent: 62,
        power_plugged: true,
      },
      BATTERY_SAVER: {
        temperature_c: 65,
        power_draw_w: 32,
        cpu_utilization: 38,
        gpu_utilization: 20,
        battery_percent: 18,
        power_plugged: false,
      },
    }
    Object.assign(demoTelemetry, profiles[profileName], {
      simulation_mode: profileName,
    })
    demoTelemetry.thermal_headroom = Math.max(
      0,
      (95 - demoTelemetry.temperature_c) / 70,
    )
    currentTelemetry = demoTelemetry
    updateTelemetryUI(demoTelemetry)
    return
  }

  try {
    const res = await fetch(`${API_BASE}/v1/telemetry/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ profile: profileName }),
    })
    const data = await res.json()
    currentTelemetry = data.telemetry
    updateTelemetryUI(data.telemetry)
  } catch (err) {
    console.error("Profile set failed:", err)
  }
}

/**
 * Control API - Manual Temperature Slider Override
 */
async function updateTempSlider(val) {
  document.getElementById("slider-temp-val").innerText =
    `${parseFloat(val).toFixed(1)} °C`
  if (DEMO_MODE) {
    demoTelemetry.temperature_c = parseFloat(val)
    demoTelemetry.thermal_headroom = Math.max(
      0,
      (95 - demoTelemetry.temperature_c) / 70,
    )
    demoTelemetry.simulation_mode = "MANUAL DEMO"
    currentTelemetry = demoTelemetry
    updateTelemetryUI(demoTelemetry)
    return
  }

  try {
    const res = await fetch(`${API_BASE}/v1/telemetry/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ temperature_c: parseFloat(val) }),
    })
    const data = await res.json()
    currentTelemetry = data.telemetry
    updateTelemetryUI(data.telemetry)
  } catch (err) {
    console.error("Temp slider update failed:", err)
  }
}

/**
 * Control API - Reset to Pure Hardware Telemetry
 */
async function resetTelemetry() {
  if (DEMO_MODE) {
    Object.assign(demoTelemetry, {
      temperature_c: 48,
      thermal_headroom: 0.78,
      power_draw_w: 65.4,
      cpu_utilization: 25,
      gpu_utilization: 35,
      battery_percent: 85,
      power_plugged: true,
      simulation_mode: "DEMO",
    })
    lastInferenceResult = null
    updateTelemetryUI(demoTelemetry)
    return
  }

  try {
    const res = await fetch(`${API_BASE}/v1/telemetry/simulate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    })
    const data = await res.json()
    lastInferenceResult = null
    currentTelemetry = data.telemetry
    updateTelemetryUI(data.telemetry)
  } catch (err) {
    console.error("Reset telemetry failed:", err)
  }
}

/**
 * Submit Prompt Request to OpenAI-compatible Endpoint
 */
async function submitPrompt() {
  const promptInput = document.getElementById("prompt-input").value
  if (!promptInput.trim()) return

  const btn = document.getElementById("btn-submit-prompt")
  btn.disabled = true
  btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Processing VoltInfer Proxy...`

  try {
    let data
    if (DEMO_MODE) {
      const pressure =
        0.5 * (1 - demoTelemetry.thermal_headroom) +
        0.35 * (demoTelemetry.gpu_utilization / 100)
      const totalLayers = 32
      const executedLayers = Math.max(
        10,
        Math.min(totalLayers, Math.round(totalLayers * (1 - 0.55 * pressure))),
      )
      const quantization =
        pressure > 0.65 ? "INT4" : pressure > 0.35 ? "INT8" : "FP16"
      const powerDraw = demoTelemetry.power_draw_w * (0.75 + pressure * 0.25)
      const baselineLatency = 0.84
      const latency = Math.max(
        0.18,
        baselineLatency * (executedLayers / totalLayers),
      )
      const baselineEnergy = baselineLatency * demoTelemetry.power_draw_w
      const energy = latency * powerDraw
      const saved = Math.max(0, baselineEnergy - energy)
      const responseText =
        "This interactive response is simulated for the hosted VoltInfer demo. Run the Windows app to connect the real local proxy and hardware telemetry."
      data = {
        choices: [{ message: { content: responseText } }],
        volt_infer_telemetry: {
          execution_mode: "Browser Demo",
          executed_layers: executedLayers,
          total_layers: totalLayers,
          quantization,
          active_kv_heads: Math.max(8, Math.round(32 * (1 - 0.6 * pressure))),
          total_kv_heads: 32,
          early_exit: executedLayers < totalLayers,
          baseline_latency_sec: baselineLatency,
          latency_sec: latency,
          power_draw_w: powerDraw,
          energy_joules: energy,
          joules_saved: saved,
          energy_saved_pct: baselineEnergy
            ? Number(((saved / baselineEnergy) * 100).toFixed(1))
            : 0,
          hardware_pressure: pressure,
        },
      }
    } else {
      const requestBody = {
        model: document.getElementById("model-name-input").value.trim(),
        messages: [{ role: "user", content: promptInput }],
        max_tokens: 120,
      }
      const selectedProfile = document.getElementById("adaptive-profile").value
      if (selectedProfile !== "off") {
        requestBody.voltinfer = { adaptive: true, profile: selectedProfile }
      }
      if (!requestBody.model) throw new Error("Enter a model name configured on your LLM server.")
      const response = await fetch(`${API_BASE}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      })
      data = await response.json()
      if (!response.ok) {
        throw new Error(data.error?.message || data.error || `Upstream request failed (${response.status})`)
      }
    }
    const result = data.volt_infer_telemetry
    const choice = data.choices?.[0]
    if (!choice) throw new Error("The LLM response did not contain a completion.")
    const message = choice.message || {}
    const textOutput =
      message.content ??
      choice.text ??
      JSON.stringify(message.tool_calls || message, null, 2)

    const container = document.getElementById("response-container")
    container.classList.remove("hidden")
    document.getElementById("response-text-box").innerText = textOutput

    if (!result) {
      lastInferenceResult = null
      document.getElementById("res-mode-badge").innerText =
        "Real upstream response; layer and energy data unavailable"
      document.getElementById("comparison-grid").hidden = true
      document.getElementById("delta-callout-banner").hidden = true
      return
    }

    lastInferenceResult = result
    document.getElementById("comparison-grid").hidden = false
    document.getElementById("delta-callout-banner").hidden = false

    // Render exact executed layers grid
    renderLayerGrid(
      result.executed_layers,
      result.total_layers,
      result.quantization,
      result.active_kv_heads,
      result.early_exit,
    )

    // Display Response Container
    document.getElementById("res-mode-badge").innerText = result.execution_mode

    // Side-by-Side Comparison Stats
    document.getElementById("base-latency-val").innerText =
      `${result.baseline_latency_sec.toFixed(3)} s`
    document.getElementById("base-power-val").innerText =
      `${(result.power_draw_w * 1.55).toFixed(1)} W`
    document.getElementById("base-energy-val").innerText =
      `${(result.baseline_latency_sec * result.power_draw_w * 1.55).toFixed(2)} J`

    document.getElementById("volt-layers-val").innerText =
      `${result.executed_layers} / ${result.total_layers} Layers`
    document.getElementById("volt-quant-val").innerText =
      `${result.quantization} Precision`
    document.getElementById("volt-kv-val").innerText =
      `${result.active_kv_heads} / ${result.total_kv_heads} Heads`
    document.getElementById("volt-latency-val").innerText =
      `${result.latency_sec.toFixed(3)} s`
    document.getElementById("volt-power-val").innerText =
      `${result.power_draw_w.toFixed(1)} W`
    document.getElementById("volt-energy-val").innerText =
      `${result.energy_joules.toFixed(2)} J`

    // Micro Deltas Calculations
    const latencyDiff = (
      result.latency_sec - result.baseline_latency_sec
    ).toFixed(3)
    const latencyPct = (
      ((result.baseline_latency_sec - result.latency_sec) /
        result.baseline_latency_sec) *
      100
    ).toFixed(1)
    document.getElementById("delta-latency").innerText =
      `${latencyDiff}s (-${latencyPct}%)`

    const powerDiff = (
      result.power_draw_w -
      result.power_draw_w * 1.55
    ).toFixed(1)
    const powerPct = (
      ((result.power_draw_w * 1.55 - result.power_draw_w) /
        (result.power_draw_w * 1.55)) *
      100
    ).toFixed(1)
    document.getElementById("delta-power").innerText =
      `${powerDiff} W (-${powerPct}%)`

    document.getElementById("delta-joules").innerText =
      `-${result.joules_saved.toFixed(2)} J (${result.energy_saved_pct}% Saved)`

    const heatAvoided = (result.hardware_pressure * 8.5).toFixed(1)
    document.getElementById("delta-thermal").innerText =
      `-${heatAvoided} °C Heat Avoided`
  } catch (err) {
    console.error("Inference request failed:", err)
    alert("Inference request failed: " + err.message)
  } finally {
    btn.disabled = false
    btn.innerHTML = `<i class="fa-solid fa-play"></i> Execute VoltInfer Request`
  }
}

/**
 * Trigger automated side-by-side benchmark test from UI
 */
async function runAutomatedBenchmark() {
  if (upstreamConfigured) {
    alert("This dashboard benchmark uses modeled demo values, not measurements from your LLM. Use a separate real-model benchmark for performance claims.")
    return
  }
  if (!DEMO_MODE)
    alert("Running the synthetic demo-policy benchmark. These are not real-model measurements.")

  // Cycle through simulation profiles sequentially to demonstrate live visual state transitions
  const profiles = [
    "PEAK_PERFORMANCE",
    "NORMAL",
    "THERMAL_THROTTLING",
    "BATTERY_SAVER",
  ]
  for (const prof of profiles) {
    await setProfile(prof)
    await submitPrompt()
    await new Promise((r) => setTimeout(r, 1200))
  }
}
