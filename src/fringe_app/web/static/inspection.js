const runSelect = document.getElementById('inspection_run_select');
const refreshRunsBtn = document.getElementById('inspection_refresh_runs_btn');
const referenceSelect = document.getElementById('reference_select');
const referencePathInput = document.getElementById('reference_path_input');
const toleranceSlider = document.getElementById('tolerance_slider');
const toleranceValue = document.getElementById('tolerance_value');
const recomputeCb = document.getElementById('recompute_cb');
const runInspectionBtn = document.getElementById('run_inspection_btn');

const statusEl = document.getElementById('inspection_status');
const errorEl = document.getElementById('inspection_error');
const resultEl = document.getElementById('inspection_result');
const reportEl = document.getElementById('inspection_report_json');
const outputsList = document.getElementById('inspection_outputs_list');
const defectsList = document.getElementById('defects_list');
const deviationPlyLink = document.getElementById('deviation_ply_link');
const overlayPlyLink = document.getElementById('overlay_ply_link');
const reportLink = document.getElementById('report_link');

function setError(msg) {
  errorEl.textContent = msg || '';
}

function selectedRunId() {
  return runSelect.value || '';
}

function updateToleranceLabel() {
  toleranceValue.textContent = Number(toleranceSlider.value).toFixed(2);
}

function clearOutputs() {
  outputsList.innerHTML = '';
  defectsList.innerHTML = '';
  deviationPlyLink.href = '#';
  overlayPlyLink.href = '#';
  reportLink.href = '#';
}

function renderDefects(report) {
  defectsList.innerHTML = '';
  const defects = Array.isArray(report?.defects) ? report.defects : [];
  if (!defects.length) {
    const li = document.createElement('li');
    li.textContent = 'No defects above tolerance.';
    defectsList.appendChild(li);
    return;
  }
  for (const d of defects) {
    const li = document.createElement('li');
    const id = d.id ?? '?';
    const sev = d.severity || 'unknown';
    const maxDev = Number(d.max_deviation ?? 0).toFixed(3);
    const area = Number(d.area ?? 0).toFixed(2);
    li.textContent = `#${id} | severity=${sev} | max=${maxDev} mm | area=${area} mm^2`;
    defectsList.appendChild(li);
  }
}

async function loadRuns() {
  const res = await fetch('/api/inspection/runs');
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
    opt.textContent = 'No reconstructed runs found';
    runSelect.appendChild(opt);
    runInspectionBtn.disabled = true;
    clearOutputs();
    return;
  }

  for (const r of runs) {
    const opt = document.createElement('option');
    opt.value = r.run_id;
    opt.textContent = r.has_inspection ? `${r.run_id} (inspection)` : r.run_id;
    runSelect.appendChild(opt);
  }

  if (prev && runs.some((r) => r.run_id === prev)) {
    runSelect.value = prev;
  }
  runInspectionBtn.disabled = false;
}

async function loadReferences() {
  const res = await fetch('/api/inspection/references');
  const data = await res.json();
  if (!data.ok) {
    setError(data.error || 'Failed to load references');
    return;
  }

  const refs = Array.isArray(data.references) ? data.references : [];
  referenceSelect.innerHTML = '';

  const blank = document.createElement('option');
  blank.value = '';
  blank.textContent = 'Select known reference (optional)';
  referenceSelect.appendChild(blank);

  for (const r of refs) {
    const opt = document.createElement('option');
    opt.value = r.path;
    opt.textContent = `${r.kind}: ${r.label}`;
    referenceSelect.appendChild(opt);
  }
}

async function loadInspectionOutputs(runId) {
  if (!runId) {
    clearOutputs();
    return;
  }
  const ts = Date.now();
  const res = await fetch(`/api/inspection/outputs?run_id=${encodeURIComponent(runId)}`);
  const data = await res.json();
  if (!data.ok || !data.exists) {
    clearOutputs();
    reportEl.textContent = '';
    resultEl.textContent = 'Result: n/a';
    return;
  }

  outputsList.innerHTML = '';
  for (const f of data.files || []) {
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.href = f.url;
    a.target = '_blank';
    a.textContent = f.name;
    li.appendChild(a);
    const s = document.createElement('span');
    s.textContent = ` (${f.size_bytes} bytes)`;
    li.appendChild(s);
    outputsList.appendChild(li);
  }

  deviationPlyLink.href = `/api/inspection/file/${encodeURIComponent(runId)}/deviation_map_colored.ply?ts=${ts}`;
  overlayPlyLink.href = `/api/inspection/file/${encodeURIComponent(runId)}/defect_overlay.ply?ts=${ts}`;
  reportLink.href = `/api/inspection/file/${encodeURIComponent(runId)}/defect_report.json?ts=${ts}`;

  const report = data.report || {};
  reportEl.textContent = JSON.stringify(report, null, 2);
  const defects = Number(report.defects_detected || 0);
  const pass = Boolean(report.pass);
  resultEl.textContent = pass ? `Result: PASS (${defects} defects)` : `Result: FAIL (${defects} defects)`;
  renderDefects(report);
}

async function runInspection() {
  const runId = selectedRunId();
  if (!runId) return;

  const referenceModel = (referencePathInput.value || '').trim();
  if (!referenceModel) {
    setError('Reference model path is required');
    return;
  }

  setError('');
  statusEl.textContent = `Running inspection for ${runId}...`;
  runInspectionBtn.disabled = true;

  try {
    const payload = {
      run_id: runId,
      reference_model: referenceModel,
      tolerance_mm: Number(toleranceSlider.value),
      recompute: Boolean(recomputeCb.checked),
    };

    const res = await fetch('/api/inspection/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!data.ok) {
      setError(data.error || 'Inspection failed');
      statusEl.textContent = `Inspection failed for ${runId}`;
      return;
    }

    statusEl.textContent = `Inspection complete for ${runId}`;
    const summary = data.summary || {};
    const defects = Number(summary.defects_detected || data.defects_detected || 0);
    const pass = Boolean(summary.pass ?? data.pass);
    resultEl.textContent = pass ? `Result: PASS (${defects} defects)` : `Result: FAIL (${defects} defects)`;

    await loadInspectionOutputs(runId);
  } catch (err) {
    setError('Inspection failed');
    statusEl.textContent = `Inspection failed for ${runId}`;
  } finally {
    runInspectionBtn.disabled = false;
  }
}

referenceSelect.addEventListener('change', () => {
  const v = referenceSelect.value || '';
  if (v) {
    referencePathInput.value = v;
  }
});

runSelect.addEventListener('change', async () => {
  await loadInspectionOutputs(selectedRunId());
});

refreshRunsBtn.addEventListener('click', async () => {
  await loadRuns();
  await loadInspectionOutputs(selectedRunId());
});

runInspectionBtn.addEventListener('click', runInspection);
toleranceSlider.addEventListener('input', updateToleranceLabel);

async function init() {
  updateToleranceLabel();
  await loadRuns();
  await loadReferences();
  await loadInspectionOutputs(selectedRunId());
}

init();
