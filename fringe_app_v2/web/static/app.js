// ── element refs ──────────────────────────────────────────────────────────────

const scanButtons = {
  captureRoi: document.getElementById("captureRoi"),
  scan:       document.getElementById("scan"),
  full:       document.getElementById("full"),
};

const ttBtn = {
  connect:   document.getElementById("ttConnect"),
  discover:  document.getElementById("ttDiscover"),
  home:      document.getElementById("ttHome"),
  multiScan: document.getElementById("multiScan"),
};

const ttIpInput  = document.getElementById("ttIpInput");
const ttControls = document.getElementById("ttControls");

// ── generic helpers ───────────────────────────────────────────────────────────

async function post(path, body = undefined) {
  const opts = { method: "POST" };
  if (body !== undefined) {
    opts.body    = JSON.stringify(body);
    opts.headers = { "Content-Type": "application/json" };
  }
  const response = await fetch(path, opts);
  const payload  = await response.json();
  if (!response.ok && payload?.error) {
    document.getElementById("error").textContent = payload.error;
  }
  return payload;
}

function setAllScanBusy(busy) {
  Object.values(scanButtons).forEach((b) => (b.disabled = busy));
}

function setBtnBusy(btn, busy, label) {
  btn.disabled    = busy;
  btn.textContent = busy ? label : btn.dataset.label || btn.textContent;
  if (!btn.dataset.label && !busy) btn.dataset.label = btn.textContent;
}

// ── turntable UI ──────────────────────────────────────────────────────────────

function updateTurntableUI(tt) {
  const statusEl = document.getElementById("ttStatus");
  const ipEl     = document.getElementById("ttIp");
  const posEl    = document.getElementById("ttPos");

  const status   = tt?.status || "disconnected";
  const connected = status === "connected";

  // Status text + colour
  statusEl.textContent = {
    connected:    "Connected",
    discovering:  "Searching…",
    not_found:    "Not found",
    disconnected: "Disconnected",
    error:        "Error",
  }[status] || status;
  statusEl.className = `status-${status}`;

  ipEl.textContent  = tt?.ip  || "—";
  posEl.textContent = connected && tt?.pos_deg != null
    ? `${Number(tt.pos_deg).toFixed(1)}°`
    : "—";

  // If we know the IP (even if not connected) prefill the input
  if (tt?.ip && !ttIpInput.value) {
    ttIpInput.value = tt.ip;
  }

  // Show turntable controls only when connected
  ttControls.style.display = connected ? "" : "none";

  // Disable turntable action buttons when a scan is running
  const pipelineBusy = document.getElementById("pipelineState").textContent === "running";
  ttBtn.home.disabled      = !connected || pipelineBusy;
  ttBtn.multiScan.disabled = !connected || pipelineBusy;
}

// ── main status poll ──────────────────────────────────────────────────────────

async function refresh() {
  try {
    const [status, calibration] = await Promise.all([
      fetch("/api/status").then((r) => r.json()),
      fetch("/api/calibration/status").then((r) => r.json()),
    ]);

    const pipeline = status.pipeline || {};
    document.getElementById("pipelineState").textContent = pipeline.state || "idle";
    document.getElementById("stage").textContent         = pipeline.stage || "idle";
    document.getElementById("runId").textContent         = pipeline.run_id || "none";
    document.getElementById("error").textContent         = pipeline.error || "none";
    document.getElementById("calibration").textContent   =
      `${calibration.camera_image_size?.join("x") || "?"} / ${calibration.projector_size?.join("x") || "?"}`;

    const busy = pipeline.state === "running";
    setAllScanBusy(busy);
    updateTurntableUI(status.turntable);
  } catch (_) {
    // network blip — silently ignore
  }
}

// ── scan button handlers ──────────────────────────────────────────────────────

scanButtons.captureRoi.addEventListener("click", async () => {
  await post("/api/capture_roi");
  await refresh();
});

scanButtons.scan.addEventListener("click", async () => {
  await post("/api/scan");
  await refresh();
});

scanButtons.full.addEventListener("click", async () => {
  await post("/api/run_full");
  await refresh();
});

// ── turntable button handlers ─────────────────────────────────────────────────

ttBtn.connect.addEventListener("click", async () => {
  const ip = ttIpInput.value.trim();
  if (!ip) { return; }
  setBtnBusy(ttBtn.connect, true, "Connecting…");
  try {
    const result = await post("/api/turntable/connect", { ip });
    updateTurntableUI(result);
  } finally {
    setBtnBusy(ttBtn.connect, false, "Connect");
  }
});

ttBtn.discover.addEventListener("click", async () => {
  setBtnBusy(ttBtn.discover, true, "Searching…");
  // Kick off background discovery; the status poll will pick up the result.
  await post("/api/turntable/discover");
  // Poll more frequently while discovering
  let polls = 0;
  const id = setInterval(async () => {
    await refresh();
    const status = document.getElementById("ttStatus").textContent;
    if (status !== "Searching…" || ++polls > 30) {
      clearInterval(id);
      setBtnBusy(ttBtn.discover, false, "Auto-detect");
    }
  }, 1000);
});

ttBtn.home.addEventListener("click", async () => {
  setBtnBusy(ttBtn.home, true, "Homing…");
  try {
    await post("/api/turntable/home");
    await refresh();
  } finally {
    setBtnBusy(ttBtn.home, false, "Home (0°)");
  }
});

ttBtn.multiScan.addEventListener("click", async () => {
  await post("/api/multi_scan");
  await refresh();
});

// ── allow pressing Enter in the IP input to connect ──────────────────────────

ttIpInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") ttBtn.connect.click();
});

// ── poll every second ─────────────────────────────────────────────────────────

setInterval(refresh, 1000);
refresh();
