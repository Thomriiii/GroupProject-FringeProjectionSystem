const previewImg = document.getElementById('preview_img');
const runSelect = document.getElementById('uv_run_select');
const refreshRunsBtn = document.getElementById('refresh_runs_btn');
const reconstructBtn = document.getElementById('reconstruct_btn');
const statusEl = document.getElementById('status');
const errorEl = document.getElementById('error');
const summaryEl = document.getElementById('summary_json');
const outputsList = document.getElementById('outputs_list');
const depthFixedImg = document.getElementById('depth_fixed_img');
const depthAutoImg = document.getElementById('depth_auto_img');

function setError(msg) {
  errorEl.textContent = msg || '';
}

function selectedRunId() {
  return runSelect.value || '';
}

async function loadRuns() {
  const res = await fetch('/api/reconstruction/runs');
  const data = await res.json();
  if (!data.ok) {
    setError(data.error || 'Failed to load runs');
    return;
  }
  const runs = data.runs || [];
  const prev = selectedRunId();
  runSelect.innerHTML = '';
  if (!runs.length) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'No UV runs found';
    runSelect.appendChild(opt);
    reconstructBtn.disabled = true;
    summaryEl.textContent = '';
    outputsList.innerHTML = '';
    return;
  }
  for (const r of runs) {
    const opt = document.createElement('option');
    opt.value = r.run_id;
    opt.textContent = `${r.run_id}${r.has_reconstruction ? ' (reconstructed)' : ''}`;
    runSelect.appendChild(opt);
  }
  if (prev && runs.some((r) => r.run_id === prev)) {
    runSelect.value = prev;
  }
  reconstructBtn.disabled = false;
}

async function loadOutputs(runId) {
  if (!runId) return;
  const ts = Date.now();
  const [statusRes, outRes] = await Promise.all([
    fetch(`/api/reconstruction/status?run_id=${encodeURIComponent(runId)}`),
    fetch(`/api/reconstruction/outputs?run_id=${encodeURIComponent(runId)}`),
  ]);
  const statusData = await statusRes.json();
  const outData = await outRes.json();
  if (!statusData.ok || !outData.ok || !statusData.exists || !outData.exists) {
    statusEl.textContent = `No reconstruction outputs for ${runId}`;
    summaryEl.textContent = '';
    outputsList.innerHTML = '';
    depthFixedImg.removeAttribute('src');
    depthAutoImg.removeAttribute('src');
    return;
  }

  statusEl.textContent = `Reconstruction available for ${runId}`;
  summaryEl.textContent = JSON.stringify(outData.meta || statusData.meta || {}, null, 2);

  outputsList.innerHTML = '';
  for (const f of outData.files || []) {
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.href = f.url;
    a.textContent = f.name;
    a.target = '_blank';
    li.appendChild(a);
    const size = document.createElement('span');
    size.textContent = ` (${f.size_bytes} bytes)`;
    li.appendChild(size);
    outputsList.appendChild(li);
  }

  depthFixedImg.src = `/api/reconstruction/file/${encodeURIComponent(runId)}/depth_debug_fixed.png?ts=${ts}`;
  depthAutoImg.src = `/api/reconstruction/file/${encodeURIComponent(runId)}/depth_debug_autoscale.png?ts=${ts}`;
}

async function reconstructSelected() {
  const runId = selectedRunId();
  if (!runId) return;
  setError('');
  statusEl.textContent = `Running reconstruction for ${runId}...`;
  reconstructBtn.disabled = true;
  try {
    const res = await fetch('/api/reconstruction/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ run_id: runId }),
    });
    const data = await res.json();
    if (!data.ok) {
      setError(data.error || 'Reconstruction failed');
      statusEl.textContent = `Reconstruction failed for ${runId}`;
      return;
    }
    statusEl.textContent = `Reconstruction complete for ${runId}`;
    summaryEl.textContent = JSON.stringify(data.summary || {}, null, 2);
    await loadOutputs(runId);
  } catch (err) {
    setError('Reconstruction failed');
  } finally {
    reconstructBtn.disabled = false;
  }
}

function setupPreview() {
  const ws = new WebSocket(`ws://${location.host}/ws/preview`);
  ws.binaryType = 'arraybuffer';
  let lastUrl = null;
  ws.onmessage = (event) => {
    const blob = new Blob([event.data], { type: 'image/jpeg' });
    const url = URL.createObjectURL(blob);
    if (lastUrl) URL.revokeObjectURL(lastUrl);
    lastUrl = url;
    previewImg.src = url;
  };
}

refreshRunsBtn.addEventListener('click', async () => {
  await loadRuns();
  await loadOutputs(selectedRunId());
});
reconstructBtn.addEventListener('click', reconstructSelected);
runSelect.addEventListener('change', async () => {
  await loadOutputs(selectedRunId());
});

async function init() {
  setupPreview();
  await loadRuns();
  await loadOutputs(selectedRunId());
}

init();
