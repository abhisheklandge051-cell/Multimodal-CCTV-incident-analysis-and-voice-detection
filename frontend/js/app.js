/**
 * Multimodal CCTV Incident Analysis & Surveillance System - Frontend Controller
 * ============================================================================
 * Handles:
 *  - Real-time WebSocket feed & CCTV Canvas rendering
 *  - YOLOv8 Bounding Box drawing & Threat warning overlays
 *  - Chart.js Probability Meters & Modal Radar visualizations
 *  - REST API integrations (Video forensics, Audio YAMNet, Late Fusion, Logs)
 *  - Webcam access & snapshot downloads
 */

const API_BASE = window.location.origin;
const WS_BASE = `ws://${window.location.host}/ws/live`;

// 7 XD-Violence Benchmark Classes & Color Map
const CLASSES = ["Abuse", "CarAccident", "Explosion", "Fighting", "Normal", "Riot", "Shooting"];
const CLASS_COLORS = {
  "Normal": "#10b981",
  "Abuse": "#f43f5e",
  "CarAccident": "#f97316",
  "Explosion": "#ef4444",
  "Fighting": "#eab308",
  "Riot": "#a855f7",
  "Shooting": "#dc2626"
};

// Global State
let ws = null;
let isStreamPaused = false;
let isUsingWebcam = false;
let webcamStream = null;
let currentCamera = "CAM-01";
let fusionAlpha = 0.60;
let incidentLogs = [];

// Chart Instances
let radarChart = null;
let videoTimelineChart = null;
let audioProbsChart = null;
let fusionComparisonChart = null;

// Canvas Simulation Objects
let liveObjects = [];
let liveIncident = {
  top_class: "Normal",
  confidence: 0.96,
  severity: "SAFE",
  probabilities: { "Normal": 0.96, "Abuse": 0.01, "CarAccident": 0.01, "Explosion": 0.0, "Fighting": 0.01, "Riot": 0.0, "Shooting": 0.01 }
};

// ==========================================================================
// Initialization
// ==========================================================================
document.addEventListener("DOMContentLoaded", () => {
  initClock();
  initCanvasAnimation();
  initProbabilityMeters();
  initRadarChart();
  connectWebSocket();
  fetchIncidents();
  setupDropzones();

  // Camera selector event
  document.getElementById("camera-select").addEventListener("change", (e) => {
    currentCamera = e.target.value;
    document.getElementById("active-cam-title").textContent = `LIVE STREAM: ${currentCamera}`;
    if (currentCamera === "WEBCAM") {
      startWebcam();
    } else {
      stopWebcam();
    }
  });
});

// ==========================================================================
// Top Clock & Tab Switching
// ==========================================================================
function initClock() {
  const clockEl = document.getElementById("live-clock");
  setInterval(() => {
    const now = new Date();
    clockEl.textContent = now.toTimeString().split(" ")[0];
  }, 1000);
}

function switchTab(tabId) {
  document.querySelectorAll(".tab-content").forEach(tab => tab.classList.add("hidden"));
  document.getElementById(tabId).classList.remove("hidden");

  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.classList.remove("text-cyan-400", "bg-cyan-500/10", "border", "border-cyan-500/30");
    btn.classList.add("text-slate-400");
  });

  const activeBtn = document.getElementById(`nav-${tabId}`);
  if (activeBtn) {
    activeBtn.classList.remove("text-slate-400");
    activeBtn.classList.add("text-cyan-400", "bg-cyan-500/10", "border", "border-cyan-500/30");
  }

  // Refresh active charts on tab display
  if (tabId === "fusion-tab") {
    setTimeout(initFusionChart, 100);
  }
}

// ==========================================================================
// WebSocket Real-Time Connection
// ==========================================================================
function connectWebSocket() {
  try {
    ws = new WebSocket(WS_BASE);

    ws.onopen = () => {
      console.log("[WebSocket] Connected to surveillance stream.");
      document.getElementById("system-connection-badge").textContent = "SYSTEM ONLINE";
      document.getElementById("system-connection-badge").parentElement.classList.remove("border-red-500/30", "text-red-400");
      document.getElementById("system-connection-badge").parentElement.classList.add("border-emerald-500/30", "text-emerald-400");
    };

    ws.onmessage = (event) => {
      if (isStreamPaused) return;
      const data = JSON.parse(event.data);
      handleLivePacket(data);
    };

    ws.onclose = () => {
      console.warn("[WebSocket] Stream closed. Attempting reconnect in 3s...");
      document.getElementById("system-connection-badge").textContent = "RECONNECTING";
      document.getElementById("system-connection-badge").parentElement.classList.add("border-yellow-500/30", "text-yellow-400");
      setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (err) => {
      console.error("[WebSocket] Error:", err);
      ws.close();
    };
  } catch (e) {
    console.error("[WebSocket] Exception:", e);
  }
}

function handleLivePacket(data) {
  // Update FPS & Audio dB badges
  document.getElementById("live-fps-badge").textContent = `FPS: ${data.fps}`;
  document.getElementById("live-db-badge").textContent = `AUDIO: ${data.audio_level_db} dB`;

  // Update State
  liveIncident = data.incident;
  liveObjects = data.objects || [];

  // HUD text labels
  const hudInc = document.getElementById("hud-incident-label");
  hudInc.textContent = `${data.incident.top_class.toUpperCase()} (${(data.incident.confidence * 100).toFixed(1)}%)`;
  hudInc.style.color = data.incident.color;

  // Object count HUD
  const count = liveObjects.length;
  document.getElementById("hud-objects-label").textContent = `${count} Object${count !== 1 ? 's' : ''} Tracked`;

  // Weapon Threat alert
  const hasThreat = liveObjects.some(o => o.is_threat) || data.threat_alert;
  const weaponBox = document.getElementById("hud-weapon-alert");
  if (hasThreat) {
    weaponBox.classList.remove("hidden");
    const threatObj = liveObjects.find(o => o.is_threat);
    document.getElementById("hud-weapon-text").textContent = threatObj ? `${threatObj.label.toUpperCase()} ALERT!` : "CRITICAL INCIDENT ALERT!";
    showAlertBanner(data.incident);
  } else {
    weaponBox.classList.add("hidden");
  }

  // Update Probability Bars
  updateProbabilityMeters(data.incident.probabilities, data.incident.top_class);

  // Update Radar Chart
  updateRadarChart(data.incident.probabilities);
}

// ==========================================================================
// Real-Time CCTV Canvas Animation & Bounding Boxes
// ==========================================================================
function initCanvasAnimation() {
  const canvas = document.getElementById("cctv-canvas");
  const ctx = canvas.getContext("2d");
  const video = document.getElementById("live-video");

  let scanlineOffset = 0;

  function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (isUsingWebcam && video.readyState === video.HAVE_ENOUGH_DATA) {
      // Draw actual webcam feed
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    } else {
      // High-tech synthetic CCTV surveillance scene
      drawSyntheticCCTVScene(ctx, canvas.width, canvas.height);
    }

    // Draw YOLO Bounding Boxes
    liveObjects.forEach(obj => {
      const [x1, y1, x2, y2] = obj.box;
      const w = x2 - x1;
      const h = y2 - y1;

      ctx.save();
      if (obj.is_threat) {
        // Red alert for weapons
        ctx.strokeStyle = "#ef4444";
        ctx.lineWidth = 3;
        ctx.fillStyle = "rgba(239, 68, 68, 0.15)";
      } else {
        ctx.strokeStyle = "#00f0ff";
        ctx.lineWidth = 2;
        ctx.fillStyle = "rgba(0, 240, 255, 0.08)";
      }

      ctx.fillRect(x1, y1, w, h);
      ctx.strokeRect(x1, y1, w, h);

      // Label Badge
      ctx.fillStyle = obj.is_threat ? "#dc2626" : "#0284c7";
      const labelText = `${obj.label} ${(obj.conf * 100).toFixed(0)}%`;
      ctx.font = "bold 11px 'JetBrains Mono', monospace";
      const textWidth = ctx.measureText(labelText).width;
      ctx.fillRect(x1, Math.max(0, y1 - 18), textWidth + 8, 18);

      ctx.fillStyle = "#ffffff";
      ctx.fillText(labelText, x1 + 4, Math.max(12, y1 - 4));
      ctx.restore();
    });

    // Surveillance scanlines & noise effect
    ctx.save();
    scanlineOffset = (scanlineOffset + 1.2) % 6;
    ctx.strokeStyle = "rgba(255, 255, 255, 0.025)";
    ctx.lineWidth = 1;
    for (let y = scanlineOffset; y < canvas.height; y += 4) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }

    // Grid Crosshairs
    ctx.strokeStyle = "rgba(0, 240, 255, 0.15)";
    ctx.lineWidth = 1;
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    ctx.beginPath();
    ctx.moveTo(cx - 20, cy); ctx.lineTo(cx + 20, cy);
    ctx.moveTo(cx, cy - 20); ctx.lineTo(cx, cy + 20);
    ctx.stroke();

    // Timestamp & Camera Watermark
    ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
    ctx.font = "12px 'JetBrains Mono', monospace";
    const timeStr = new Date().toISOString().replace("T", " ").substr(0, 19);
    ctx.fillText(`${currentCamera} - REC [●]  ${timeStr}`, 15, canvas.height - 15);

    ctx.restore();

    requestAnimationFrame(render);
  }

  requestAnimationFrame(render);
}

function drawSyntheticCCTVScene(ctx, w, h) {
  // Dark surveillance environment backdrop
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, "#080c14");
  grad.addColorStop(0.5, "#0d131f");
  grad.addColorStop(1, "#070a10");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);

  // Perspective floor lines
  ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
  ctx.lineWidth = 1;
  for (let i = -w; i <= w * 2; i += 80) {
    ctx.beginPath();
    ctx.moveTo(i, h);
    ctx.lineTo(w / 2 + (i - w / 2) * 0.15, h * 0.45);
    ctx.stroke();
  }

  // Horizon wall
  ctx.fillStyle = "rgba(0, 0, 0, 0.6)";
  ctx.fillRect(0, 0, w, h * 0.45);

  // Subtle architectural pillars
  ctx.fillStyle = "rgba(255, 255, 255, 0.03)";
  ctx.fillRect(100, 40, 60, h * 0.45);
  ctx.fillRect(w - 180, 40, 70, h * 0.45);
}

// ==========================================================================
// Probability Meters & Charts
// ==========================================================================
function initProbabilityMeters() {
  const container = document.getElementById("probability-meters");
  container.innerHTML = "";

  CLASSES.forEach(cls => {
    const row = document.createElement("div");
    row.id = `meter-row-${cls}`;
    row.className = "space-y-1";
    row.innerHTML = `
      <div class="flex justify-between items-center text-[11px]">
        <span class="text-slate-300 font-medium">${cls}</span>
        <span id="meter-val-${cls}" class="text-slate-400 font-mono">0.0%</span>
      </div>
      <div class="w-full bg-slate-900 rounded-full h-2 overflow-hidden border border-white/5">
        <div id="meter-bar-${cls}" class="h-full rounded-full transition-all duration-300" style="width: 0%; background-color: ${CLASS_COLORS[cls]}"></div>
      </div>
    `;
    container.appendChild(row);
  });
}

function updateProbabilityMeters(probs, topClass) {
  if (!probs) return;
  CLASSES.forEach(cls => {
    const p = probs[cls] || 0.0;
    const pct = (p * 100).toFixed(1);
    const bar = document.getElementById(`meter-bar-${cls}`);
    const val = document.getElementById(`meter-val-${cls}`);

    if (bar && val) {
      bar.style.width = `${Math.min(100, pct)}%`;
      val.textContent = `${pct}%`;

      if (cls === topClass && cls !== "Normal") {
        val.classList.add("text-cyan-400", "font-bold");
      } else {
        val.classList.remove("text-cyan-400", "font-bold");
      }
    }
  });
}

function initRadarChart() {
  const ctx = document.getElementById("radar-chart").getContext("2d");
  radarChart = new Chart(ctx, {
    type: "radar",
    data: {
      labels: CLASSES,
      datasets: [
        {
          label: "Visual Confidence",
          data: [5, 5, 2, 8, 95, 4, 3],
          borderColor: "#00f0ff",
          backgroundColor: "rgba(0, 240, 255, 0.2)",
          borderWidth: 1.5,
          pointRadius: 2
        },
        {
          label: "Acoustic Confidence",
          data: [4, 6, 3, 10, 92, 5, 2],
          borderColor: "#eab308",
          backgroundColor: "rgba(234, 179, 8, 0.2)",
          borderWidth: 1.5,
          pointRadius: 2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          labels: { color: "#94a3b8", font: { size: 10, family: 'JetBrains Mono' } }
        }
      },
      scales: {
        r: {
          angleLines: { color: "rgba(255, 255, 255, 0.08)" },
          grid: { color: "rgba(255, 255, 255, 0.08)" },
          pointLabels: { color: "#cbd5e1", font: { size: 9, family: 'JetBrains Mono' } },
          ticks: { display: false, max: 100, min: 0 }
        }
      }
    }
  });
}

function updateRadarChart(probs) {
  if (!radarChart || !probs) return;
  const values = CLASSES.map(c => Math.round((probs[c] || 0) * 100));
  radarChart.data.datasets[0].data = values;
  radarChart.update();
}

// ==========================================================================
// Dropzones & File Upload Handlers
// ==========================================================================
function setupDropzones() {
  // Video Dropzone
  const vDrop = document.getElementById("video-dropzone");
  const vInput = document.getElementById("video-file-input");
  vDrop.addEventListener("click", () => vInput.click());
  vInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      document.getElementById("video-file-name").textContent = `Selected: ${e.target.files[0].name} (${(e.target.files[0].size / 1024 / 1024).toFixed(1)} MB)`;
    }
  });

  // Audio Dropzone
  const aDrop = document.getElementById("audio-dropzone");
  const aInput = document.getElementById("audio-file-input");
  aDrop.addEventListener("click", () => aInput.click());
  aInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      document.getElementById("audio-file-name").textContent = `Selected: ${e.target.files[0].name} (${(e.target.files[0].size / 1024 / 1024).toFixed(1)} MB)`;
    }
  });
}

async function submitVideoAnalysis() {
  const fileInput = document.getElementById("video-file-input");
  if (!fileInput.files.length) {
    alert("Please select a video file first!");
    return;
  }

  const btn = document.getElementById("btn-analyze-video");
  btn.disabled = true;
  btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Processing ResNet50 + Bi-GRU...`;

  const formData = new FormData();
  formData.append("video_file", fileInput.files[0]);

  try {
    const res = await fetch(`${API_BASE}/api/analyze/video`, {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    renderVideoResults(data);
    fetchIncidents(); // Refresh logs
  } catch (err) {
    console.error("Video analysis failed:", err);
    alert("Video analysis failed. Please verify API server status.");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i class="fa-solid fa-microchip"></i> Execute Deep Forensic Scan`;
  }
}

function renderVideoResults(data) {
  document.getElementById("video-results-container").classList.remove("hidden");

  // Peak Card
  const peak = data.peak_incident;
  document.getElementById("forensic-peak-class").textContent = peak.class;
  document.getElementById("forensic-peak-severity").textContent = peak.severity;
  document.getElementById("forensic-peak-score").textContent = `${peak.risk_score} / 100`;
  document.getElementById("forensic-peak-desc").textContent = `Highest anomaly score recorded at timestamp ${peak.timestamp} with ${(peak.confidence * 100).toFixed(1)}% confidence.`;

  // Render Objects List
  const objList = document.getElementById("forensic-objects-list");
  objList.innerHTML = "";
  Object.entries(data.detected_objects_summary).forEach(([label, count]) => {
    const badge = document.createElement("span");
    badge.className = "px-2.5 py-1 rounded bg-slate-900 border border-cyan-500/30 text-cyan-300";
    badge.textContent = `${label}: ${count}`;
    objList.appendChild(badge);
  });

  // Render Checkpoints
  const chkList = document.getElementById("forensic-checkpoints-list");
  chkList.innerHTML = "";
  data.timeline.forEach(seg => {
    const item = document.createElement("div");
    item.className = "p-2 rounded bg-slate-900/60 border border-white/5 flex items-center justify-between";
    item.innerHTML = `
      <span>[${seg.time_formatted}] <strong class="text-white">${seg.top_class}</strong> (${(seg.confidence * 100).toFixed(0)}%)</span>
      <span class="px-2 py-0.5 rounded text-[10px] font-bold ${seg.severity === 'CRITICAL' ? 'bg-red-500/20 text-red-400' : (seg.severity === 'HIGH' ? 'bg-yellow-500/20 text-yellow-400' : 'bg-emerald-500/20 text-emerald-400')}">${seg.severity}</span>
    `;
    chkList.appendChild(item);
  });

  // Timeline Chart
  const ctx = document.getElementById("video-timeline-chart").getContext("2d");
  if (videoTimelineChart) videoTimelineChart.destroy();

  videoTimelineChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: data.timeline.map(s => s.time_formatted),
      datasets: [{
        label: "Anomaly Risk Score",
        data: data.timeline.map(s => s.risk_score),
        borderColor: "#00f0ff",
        backgroundColor: "rgba(0, 240, 255, 0.15)",
        fill: true,
        tension: 0.35,
        pointBackgroundColor: data.timeline.map(s => s.severity === 'CRITICAL' ? '#ef4444' : (s.severity === 'HIGH' ? '#eab308' : '#10b981')),
        pointRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: "rgba(255, 255, 255, 0.05)" }, ticks: { color: "#94a3b8", font: { family: 'JetBrains Mono' } } },
        y: { max: 100, min: 0, grid: { color: "rgba(255, 255, 255, 0.05)" }, ticks: { color: "#94a3b8", font: { family: 'JetBrains Mono' } } }
      }
    }
  });
}

async function submitAudioAnalysis() {
  const fileInput = document.getElementById("audio-file-input");
  if (!fileInput.files.length) {
    alert("Please select an audio file first!");
    return;
  }

  const btn = document.getElementById("btn-analyze-audio");
  btn.disabled = true;
  btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Extracting YAMNet Embeddings...`;

  const formData = new FormData();
  formData.append("audio_file", fileInput.files[0]);

  try {
    const res = await fetch(`${API_BASE}/api/analyze/audio`, {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    renderAudioResults(data);
    fetchIncidents();
  } catch (err) {
    console.error("Audio analysis failed:", err);
    alert("Audio analysis failed. Please verify API server status.");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<i class="fa-solid fa-chart-simple"></i> Run YAMNet Acoustic Inference`;
  }
}

function renderAudioResults(data) {
  document.getElementById("audio-results-container").classList.remove("hidden");

  document.getElementById("audio-result-class").textContent = data.top_class;
  document.getElementById("audio-result-severity").textContent = data.severity;
  document.getElementById("audio-result-conf").textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

  // Signatures list
  const sigList = document.getElementById("audio-signatures-list");
  sigList.innerHTML = "";
  data.acoustic_events.forEach(ev => {
    const li = document.createElement("li");
    li.textContent = ev;
    sigList.appendChild(li);
  });

  // Audio Bar Chart
  const ctx = document.getElementById("audio-probs-chart").getContext("2d");
  if (audioProbsChart) audioProbsChart.destroy();

  audioProbsChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: CLASSES,
      datasets: [{
        label: "YAMNet Probability",
        data: CLASSES.map(c => Math.round((data.probabilities[c] || 0) * 100)),
        backgroundColor: CLASSES.map(c => CLASS_COLORS[c]),
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { color: "#94a3b8", font: { family: 'JetBrains Mono', size: 10 } } },
        y: { max: 100, min: 0, grid: { color: "rgba(255, 255, 255, 0.05)" }, ticks: { color: "#94a3b8", font: { family: 'JetBrains Mono' } } }
      }
    }
  });
}

// ==========================================================================
// Multimodal Late-Fusion
// ==========================================================================
function updateFusionAlpha(val) {
  fusionAlpha = parseFloat(val);
  document.getElementById("alpha-val").textContent = fusionAlpha.toFixed(2);
  const videoPct = Math.round(fusionAlpha * 100);
  const audioPct = 100 - videoPct;
  document.getElementById("fusion-balance-label").textContent = `${videoPct}% Video / ${audioPct}% Audio`;
  runMultimodalDemo();
}

async function runMultimodalDemo() {
  // Create dummy blob for multimodal synthesis demo
  const blob = new Blob(["surveillance media simulation"], { type: "text/plain" });
  const formData = new FormData();
  formData.append("media_file", blob, "surveillance_clip.mp4");
  formData.append("alpha", fusionAlpha);

  try {
    const res = await fetch(`${API_BASE}/api/analyze/multimodal`, {
      method: "POST",
      body: formData
    });
    const data = await res.json();
    renderFusionResults(data);
  } catch (err) {
    console.error("Fusion error:", err);
  }
}

function renderFusionResults(data) {
  const brk = data.modality_breakdown;
  document.getElementById("fusion-video-class").textContent = brk.video.top_class;
  document.getElementById("fusion-video-conf").textContent = `${(brk.video.confidence * 100).toFixed(1)}%`;

  document.getElementById("fusion-audio-class").textContent = brk.audio.top_class;
  document.getElementById("fusion-audio-conf").textContent = `${(brk.audio.confidence * 100).toFixed(1)}%`;

  document.getElementById("fusion-fused-class").textContent = brk.fusion.top_class;
  document.getElementById("fusion-fused-conf").textContent = `${(brk.fusion.confidence * 100).toFixed(1)}%`;

  initFusionChart(brk);
}

function initFusionChart(brkData = null) {
  const ctx = document.getElementById("fusion-comparison-chart")?.getContext("2d");
  if (!ctx) return;

  const defaultVideo = [10, 5, 2, 82, 1, 0, 0];
  const defaultAudio = [8, 4, 1, 87, 0, 0, 0];
  const defaultFused = [9, 5, 2, 84, 0, 0, 0];

  const vData = brkData ? CLASSES.map(c => Math.round(brkData.video.probabilities[c] * 100)) : defaultVideo;
  const aData = brkData ? CLASSES.map(c => Math.round(brkData.audio.probabilities[c] * 100)) : defaultAudio;
  const fData = brkData ? CLASSES.map(c => Math.round(brkData.fusion.probabilities[c] * 100)) : defaultFused;

  if (fusionComparisonChart) fusionComparisonChart.destroy();

  fusionComparisonChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: CLASSES,
      datasets: [
        {
          label: "Visual ResNet-BiGRU (%)",
          data: vData,
          backgroundColor: "#00f0ff",
          borderRadius: 3
        },
        {
          label: "Acoustic YAMNet (%)",
          data: aData,
          backgroundColor: "#eab308",
          borderRadius: 3
        },
        {
          label: "Fused Decision Probability (%)",
          data: fData,
          backgroundColor: "#10b981",
          borderRadius: 3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: "#94a3b8", font: { family: 'JetBrains Mono', size: 11 } }
        }
      },
      scales: {
        x: { grid: { display: false }, ticks: { color: "#94a3b8", font: { family: 'JetBrains Mono' } } },
        y: { max: 100, min: 0, grid: { color: "rgba(255, 255, 255, 0.05)" }, ticks: { color: "#94a3b8", font: { family: 'JetBrains Mono' } } }
      }
    }
  });
}

// ==========================================================================
// Incident History & Export
// ==========================================================================
async function fetchIncidents() {
  try {
    const res = await fetch(`${API_BASE}/api/incidents`);
    const data = await res.json();
    incidentLogs = data.incidents || [];
    renderIncidentTable(incidentLogs);
  } catch (err) {
    console.error("Failed to fetch incidents:", err);
  }
}

function renderIncidentTable(logs) {
  const tbody = document.getElementById("incident-table-body");
  tbody.innerHTML = "";

  if (logs.length === 0) {
    tbody.innerHTML = `<tr><td colspan="8" class="text-center py-6 text-slate-500">No incident records found.</td></tr>`;
    return;
  }

  logs.forEach(item => {
    const tr = document.createElement("tr");
    tr.className = "hover:bg-slate-900/60 transition";

    let sevClass = "bg-slate-800 text-slate-400";
    if (item.severity === "CRITICAL") sevClass = "bg-red-500/20 text-red-400 border border-red-500/40";
    else if (item.severity === "HIGH") sevClass = "bg-yellow-500/20 text-yellow-400 border border-yellow-500/40";
    else if (item.severity === "SAFE") sevClass = "bg-emerald-500/20 text-emerald-400 border border-emerald-500/40";

    tr.innerHTML = `
      <td class="py-3 px-4 text-cyan-400 font-bold">${item.id}</td>
      <td class="py-3 px-4 text-slate-400">${item.timestamp}</td>
      <td class="py-3 px-4 text-white font-semibold">${item.camera_id}</td>
      <td class="py-3 px-4"><span class="font-bold" style="color:${CLASS_COLORS[item.incident_type] || '#ffffff'}">${item.incident_type}</span></td>
      <td class="py-3 px-4"><span class="px-2 py-0.5 rounded text-[10px] font-bold ${sevClass}">${item.severity}</span></td>
      <td class="py-3 px-4 text-slate-300 font-mono">${(item.confidence * 100).toFixed(1)}%</td>
      <td class="py-3 px-4 text-slate-400">${item.modality}</td>
      <td class="py-3 px-4 text-slate-400 truncate max-w-xs" title="${item.details}">${item.details}</td>
    `;
    tbody.appendChild(tr);
  });
}

function filterLogs() {
  const sev = document.getElementById("log-filter-severity").value;
  const search = document.getElementById("log-search-input").value.toLowerCase();

  const filtered = incidentLogs.filter(item => {
    const matchSev = sev === "ALL" || item.severity === sev;
    const matchSearch = item.camera_id.toLowerCase().includes(search) || item.incident_type.toLowerCase().includes(search) || item.details.toLowerCase().includes(search);
    return matchSev && matchSearch;
  });

  renderIncidentTable(filtered);
}

function exportLogsJSON() {
  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(incidentLogs, null, 2));
  const dlAnchor = document.createElement('a');
  dlAnchor.setAttribute("href", dataStr);
  dlAnchor.setAttribute("download", `surveillance_incident_audit_${Date.now()}.json`);
  dlAnchor.click();
}

function exportLogsCSV() {
  if (!incidentLogs.length) return;
  const headers = ["id", "timestamp", "camera_id", "incident_type", "severity", "confidence", "modality", "details"];
  const rows = incidentLogs.map(inc => headers.map(h => `"${(inc[h] || '').toString().replace(/"/g, '""')}"`).join(","));
  const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows].join("\n");
  const dlAnchor = document.createElement('a');
  dlAnchor.setAttribute("href", encodeURI(csvContent));
  dlAnchor.setAttribute("download", `surveillance_incident_audit_${Date.now()}.csv`);
  dlAnchor.click();
}

// ==========================================================================
// Controls & Helpers
// ==========================================================================
function toggleLiveStream() {
  isStreamPaused = !isStreamPaused;
  const btn = document.getElementById("btn-toggle-stream");
  if (isStreamPaused) {
    btn.innerHTML = `<i class="fa-solid fa-play"></i> Resume Stream`;
    btn.classList.replace("bg-cyan-500", "bg-emerald-500");
  } else {
    btn.innerHTML = `<i class="fa-solid fa-pause"></i> Pause Stream`;
    btn.classList.replace("bg-emerald-500", "bg-cyan-500");
  }
}

function triggerQuickSim(incidentType) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: "force_incident", incident_type: incidentType }));
  }
}

function showAlertBanner(incident) {
  const banner = document.getElementById("live-alert-banner");
  banner.classList.remove("hidden");

  const title = document.getElementById("alert-banner-title");
  const desc = document.getElementById("alert-banner-desc");
  const icon = document.getElementById("alert-banner-icon");

  if (incident.severity === "CRITICAL") {
    banner.className = "p-4 rounded-xl border border-red-500/50 bg-red-950/40 text-red-300 flex items-center justify-between pulse-critical";
    icon.className = "w-10 h-10 rounded-lg bg-red-600/20 text-red-400 flex items-center justify-center text-lg";
    title.textContent = `CRITICAL THREAT ALERT: ${incident.top_class.toUpperCase()}`;
    desc.textContent = `Immediate automated detection. Neural network confidence at ${(incident.confidence * 100).toFixed(1)}%.`;
  } else if (incident.severity === "HIGH") {
    banner.className = "p-4 rounded-xl border border-yellow-500/50 bg-yellow-950/40 text-yellow-300 flex items-center justify-between";
    icon.className = "w-10 h-10 rounded-lg bg-yellow-600/20 text-yellow-400 flex items-center justify-center text-lg";
    title.textContent = `HIGH PRIORITY EVENT: ${incident.top_class.toUpperCase()}`;
    desc.textContent = `Elevated activity detected on ${currentCamera}. Active surveillance tracking engaged.`;
  }
}

function dismissAlertBanner() {
  document.getElementById("live-alert-banner").classList.add("hidden");
}

function captureSnapshot() {
  const canvas = document.getElementById("cctv-canvas");
  const link = document.createElement("a");
  link.download = `cctv_snapshot_${currentCamera}_${Date.now()}.png`;
  link.href = canvas.toDataURL("image/png");
  link.click();
}

async function startWebcam() {
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
    const video = document.getElementById("live-video");
    video.srcObject = webcamStream;
    video.play();
    isUsingWebcam = true;
  } catch (err) {
    console.error("Webcam access error:", err);
    alert("Could not access local webcam. Please allow camera permissions in browser.");
  }
}

function stopWebcam() {
  if (webcamStream) {
    webcamStream.getTracks().forEach(track => track.stop());
    webcamStream = null;
  }
  isUsingWebcam = false;
}

function toggleWebcam() {
  const select = document.getElementById("camera-select");
  if (isUsingWebcam) {
    select.value = "CAM-01";
    currentCamera = "CAM-01";
    document.getElementById("active-cam-title").textContent = `LIVE STREAM: CAM-01 (Main Gate)`;
    stopWebcam();
  } else {
    select.value = "WEBCAM";
    currentCamera = "WEBCAM";
    document.getElementById("active-cam-title").textContent = `LIVE STREAM: Local Webcam Feed`;
    startWebcam();
  }
}
