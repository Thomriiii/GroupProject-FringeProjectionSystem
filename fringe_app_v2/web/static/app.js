const buttons = {
  captureRoi: document.getElementById("captureRoi"),
  scan: document.getElementById("scan"),
  full: document.getElementById("full"),
};

async function post(path) {
  const response = await fetch(path, { method: "POST" });
  const payload = await response.json();
  if (!response.ok || payload.ok === false) {
    const message = payload?.status?.error || payload?.error || `request failed: ${response.status}`;
    document.getElementById("error").textContent = message;
  }
  return payload;
}

async function refresh() {
  const [status, calibration] = await Promise.all([
    fetch("/api/status").then((r) => r.json()),
    fetch("/api/calibration/status").then((r) => r.json()),
  ]);
  const pipeline = status.pipeline || {};
  document.getElementById("pipelineState").textContent = pipeline.state || "idle";
  document.getElementById("stage").textContent = pipeline.stage || "idle";
  document.getElementById("runId").textContent = pipeline.run_id || "none";
  document.getElementById("error").textContent = pipeline.error || "none";
  document.getElementById("calibration").textContent =
    `${calibration.camera_image_size?.join("x") || "camera ?"} / ${calibration.projector_size?.join("x") || "projector ?"}`;
  const busy = pipeline.state === "running";
  Object.values(buttons).forEach((button) => {
    button.disabled = busy;
  });
}

buttons.captureRoi.addEventListener("click", async () => {
  await post("/api/capture_roi");
  await refresh();
});

buttons.scan.addEventListener("click", async () => {
  await post("/api/scan");
  await refresh();
});

buttons.full.addEventListener("click", async () => {
  await post("/api/run_full");
  await refresh();
});

setInterval(refresh, 1000);
refresh();
