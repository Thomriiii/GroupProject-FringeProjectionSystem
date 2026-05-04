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

const hqEnableCb = document.getElementById('hq_enable_cb');
const hqReprojSlider = document.getElementById('hq_reproj_slider');
const hqReprojValue = document.getElementById('hq_reproj_value');
const hqModSlider = document.getElementById('hq_mod_slider');
const hqModValue = document.getElementById('hq_mod_value');
const hqSmoothingCb = document.getElementById('hq_smoothing_cb');
const runQualityBtn = document.getElementById('run_quality_btn');
const runSweepBtn = document.getElementById('run_sweep_btn');

const standardPlyLink = document.getElementById('standard_ply_link');
const qualityPlyLink = document.getElementById('quality_ply_link');
const qualityOutputsList = document.getElementById('quality_outputs_list');
const qualityMaskImg = document.getElementById('quality_mask_img');
const qualityReprojImg = document.getElementById('quality_reproj_img');
const qualityModImg = document.getElementById('quality_mod_img');
const qualityDepthBeforeImg = document.getElementById('quality_depth_before_img');
const qualityDepthAfterImg = document.getElementById('quality_depth_after_img');

function setError(msg) {
  errorEl.textContent = msg || '';
}

function selectedRunId() {
  return runSelect.value || '';
}

function updateSliderLabels() {
  hqReprojValue.textContent = Number(hqReprojSlider.value).toFixed(1);
  hqModValue.textContent = String(Math.round(Number(hqModSlider.value)));
}

function setComparisonLinks(runId, hasQuality) {
  if (!runId) {
    standardPlyLink.href = '#';
    qualityPlyLink.href = '#';
    return;
  }
  const ts = Date.now();
  standardPlyLink.href = `/api/reconstruction/file/${encodeURIComponent(runId)}/standard_reconstruction.ply?ts=${ts}`;
  qualityPlyLink.href = hasQuality
    ? `/api/reconstruction/quality/file/${encodeURIComponent(runId)}/high_quality.ply?ts=${ts}`
    : '#';
}

async function loadRuns() {
  const res = await fetch('/api/reconstruction/runs');
  const data = await res.json();
  if (!data.ok) {
    setError(data.error || 'Failed to load runs');
    return;
  }
  const runs = Array.isArray(data.runs) ? data.runs : [];
  const prev = selectedRunId();
  runSelect.innerHTML = '';
  if (!runs.length) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'No UV runs found';
    runSelect.appendChild(opt);
    reconstructBtn.disabled = true;
    runQualityBtn.disabled = true;
    runSweepBtn.disabled = true;
    summaryEl.textContent = '';
    outputsList.innerHTML = '';
    qualityOutputsList.innerHTML = '';
    setComparisonLinks('', false);
    return;
  }
  for (const r of runs) {
    const opt = document.createElement('option');
    opt.value = r.run_id;
    let label = r.run_id;
    if (r.has_reconstruction) {
      label += ' (standard)';
    }
    if (r.has_reconstruction_quality) {
      label += ' (quality)';
    }
    opt.textContent = label;
    runSelect.appendChild(opt);
  }
  if (prev && runs.some((r) => r.run_id === prev)) {
    runSelect.value = prev;
  }
  reconstructBtn.disabled = false;
  runQualityBtn.disabled = false;
  runSweepBtn.disabled = false;
}

async function loadStandardOutputs(runId) {
  if (!runId) return false;
  const ts = Date.now();
  const [statusRes, outRes] = await Promise.all([
    fetch(`/api/reconstruction/status?run_id=${encodeURIComponent(runId)}`),
    fetch(`/api/reconstruction/outputs?run_id=${encodeURIComponent(runId)}`),
  ]);
  const statusData = await statusRes.json();
  const outData = await outRes.json();
  if (!statusData.ok || !outData.ok || !statusData.exists || !outData.exists) {
    statusEl.textContent = `No reconstruction outputs for ${runId}`;
    outputsList.innerHTML = '';
    depthFixedImg.removeAttribute('src');
    depthAutoImg.removeAttribute('src');
    return false;
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
  return true;
}

async function loadQualityOutputs(runId) {
  if (!runId) return false;
  const ts = Date.now();
  const res = await fetch(`/api/reconstruction/quality/outputs?run_id=${encodeURIComponent(runId)}`);
  const data = await res.json();
  if (!data.ok || !data.exists) {
    qualityOutputsList.innerHTML = '';
    qualityMaskImg.removeAttribute('src');
    qualityReprojImg.removeAttribute('src');
    qualityModImg.removeAttribute('src');
    qualityDepthBeforeImg.removeAttribute('src');
    qualityDepthAfterImg.removeAttribute('src');
    setComparisonLinks(runId, false);
    return false;
  }

  qualityOutputsList.innerHTML = '';
  for (const f of data.files || []) {
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.href = f.url;
    a.textContent = f.name;
    a.target = '_blank';
    li.appendChild(a);
    const size = document.createElement('span');
    size.textContent = ` (${f.size_bytes} bytes)`;
    li.appendChild(size);
    qualityOutputsList.appendChild(li);
  }

  qualityMaskImg.src = `/api/reconstruction/quality/file/${encodeURIComponent(runId)}/mask_quality.png?ts=${ts}`;
  qualityReprojImg.src = `/api/reconstruction/quality/file/${encodeURIComponent(runId)}/reprojection_error_map.png?ts=${ts}`;
  qualityModImg.src = `/api/reconstruction/quality/file/${encodeURIComponent(runId)}/modulation_map.png?ts=${ts}`;
  qualityDepthBeforeImg.src = `/api/reconstruction/quality/file/${encodeURIComponent(runId)}/depth_before.png?ts=${ts}`;
  qualityDepthAfterImg.src = `/api/reconstruction/quality/file/${encodeURIComponent(runId)}/depth_after.png?ts=${ts}`;

  setComparisonLinks(runId, true);
  return true;
}

async function loadOutputs(runId) {
  if (!runId) return;
  const hasStandard = await loadStandardOutputs(runId);
  const hasQuality = await loadQualityOutputs(runId);
  if (hasStandard && !hasQuality) {
    setComparisonLinks(runId, false);
  }
}

async function reconstructSelected() {
  const runId = selectedRunId();
  if (!runId) return;
  setError('');
  statusEl.textContent = `Running standard reconstruction for ${runId}...`;
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
      statusEl.textContent = `Standard reconstruction failed for ${runId}`;
      return;
    }
    statusEl.textContent = `Standard reconstruction complete for ${runId}`;
    summaryEl.textContent = JSON.stringify(data.summary || {}, null, 2);
    await loadOutputs(runId);
  } catch (err) {
    setError('Reconstruction failed');
  } finally {
    reconstructBtn.disabled = false;
  }
}

async function runQualityMode(enableSweep) {
  const runId = selectedRunId();
  if (!runId) return;
  setError('');
  statusEl.textContent = enableSweep
    ? `Running quality sweep for ${runId}...`
    : `Running high quality reconstruction for ${runId}...`;
  runQualityBtn.disabled = true;
  runSweepBtn.disabled = true;
  try {
    const payload = {
      run_id: runId,
      enable_confidence_filter: Boolean(hqEnableCb.checked),
      max_reproj_error_px: Number(hqReprojSlider.value),
      min_modulation_B: Math.round(Number(hqModSlider.value)),
      enable_smoothing: Boolean(hqSmoothingCb.checked),
      enable_sweep: Boolean(enableSweep),
    };
    const res = await fetch('/api/reconstruction/quality', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!data.ok) {
      setError(data.error || 'High quality reconstruction failed');
      statusEl.textContent = `High quality reconstruction failed for ${runId}`;
      return;
    }
    statusEl.textContent = enableSweep
      ? `Quality sweep complete for ${runId}`
      : `High quality reconstruction complete for ${runId}`;
    summaryEl.textContent = JSON.stringify(data.summary || {}, null, 2);
    await loadOutputs(runId);
  } catch (err) {
    setError('High quality reconstruction failed');
  } finally {
    runQualityBtn.disabled = false;
    runSweepBtn.disabled = false;
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
runQualityBtn.addEventListener('click', async () => {
  await runQualityMode(false);
});
runSweepBtn.addEventListener('click', async () => {
  await runQualityMode(true);
});
runSelect.addEventListener('change', async () => {
  await loadOutputs(selectedRunId());
});
hqReprojSlider.addEventListener('input', updateSliderLabels);
hqModSlider.addEventListener('input', updateSliderLabels);

async function init() {
  setupPreview();
  updateSliderLabels();
  await loadRuns();
  await loadOutputs(selectedRunId());
}

init();
