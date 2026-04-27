const previewImg = document.getElementById('preview_img');
const statusEl = document.getElementById('status');
const errorEl = document.getElementById('error');
const captureFeedbackEl = document.getElementById('capture_feedback');
const solveFeedbackEl = document.getElementById('solve_feedback');
const summaryEl = document.getElementById('summary');
const viewsGrid = document.getElementById('views_grid');
const resultsEl = document.getElementById('results_json');
const coverageSummaryEl = document.getElementById('coverage_summary');
const coverageHeatmapEl = document.getElementById('coverage_heatmap');
const coverageImageEl = document.getElementById('coverage_image');

const newSessionBtn = document.getElementById('new_session_btn');
const sessionSelect = document.getElementById('session_select');
const continueSessionBtn = document.getElementById('continue_session_btn');
const captureBtn = document.getElementById('capture_view_btn');
const solveBtn = document.getElementById('solve_btn');

let currentSessionId = null;
let pollTimer = null;
let latestSolveState = 'idle';
let lastViewsSignature = '';
let lastCoverageSignature = '';
let lastResultsSignature = '';
let imageRevision = 0;

function setError(msg) {
  errorEl.textContent = msg || '';
}

function setCaptureFeedback(msg) {
  captureFeedbackEl.textContent = msg || '';
}

function setSolveFeedback(msg) {
  solveFeedbackEl.textContent = msg || '';
}

async function fetchJson(url, opts = undefined) {
  const res = await fetch(url, opts);
  const data = await res.json();
  if (!res.ok || data.ok === false) {
    throw new Error(data.error || `HTTP ${res.status}`);
  }
  return data;
}

function renderCoverage(coverage) {
  const gridSize = Array.isArray(coverage?.grid_size) ? coverage.grid_size : [];
  const binsX = Number(coverage?.bins_x || gridSize[0] || 8);
  const binsY = Number(coverage?.bins_y || gridSize[1] || 8);
  const grid = Array.isArray(coverage?.grid) ? coverage.grid : [];
  const covered = Number(coverage?.covered_bins || coverage?.bins_covered_count || 0);
  const total = Number(coverage?.total_bins || coverage?.bins_total || binsX * binsY);
  const ratio = Number(coverage?.coverage_ratio || 0);
  const edgeRatio = Number(coverage?.edge_coverage_ratio || 0);
  const uniformity = Number(coverage?.uniformity_metric || 0);
  const sufficient = Boolean(coverage?.sufficient);
  const guidance = Array.isArray(coverage?.guidance) ? coverage.guidance : [];
  const guidanceText = guidance.length ? ` | Guidance: ${guidance.join(', ')}` : '';
  const suffText = sufficient ? ' | Coverage target reached' : '';
  coverageSummaryEl.textContent = `Coverage: ${covered}/${total} bins (${(100 * ratio).toFixed(1)}%) | edge ${(100 * edgeRatio).toFixed(1)}% | uniformity ${uniformity.toFixed(3)}${suffText}${guidanceText}`;

  coverageHeatmapEl.style.gridTemplateColumns = `repeat(${binsX}, 20px)`;
  const cells = [];
  for (let y = 0; y < binsY; y += 1) {
    for (let x = 0; x < binsX; x += 1) {
      const filled = Number((grid[y] || [])[x] || 0) > 0;
      cells.push(`<div class="coverage-cell ${filled ? 'filled' : ''}" title="${x},${y}"></div>`);
    }
  }
  coverageHeatmapEl.innerHTML = cells.join('');
}

function renderCoverageImage(revisionTag = '0') {
  if (!currentSessionId) {
    coverageImageEl.innerHTML = '';
    return;
  }
  coverageImageEl.innerHTML = `
    <a href="/api/calibration/projector/session/${currentSessionId}/coverage.png?rev=${encodeURIComponent(revisionTag)}" target="_blank">Open coverage heatmap</a>
    <img src="/api/calibration/projector/session/${currentSessionId}/coverage.png?rev=${encodeURIComponent(revisionTag)}" alt="coverage heatmap"
         style="margin-top:8px;max-width:420px;border:1px solid #ccc;" onerror="this.style.display='none';" />
  `;
}

function viewCardHtml(view) {
  const diag = view.diag || {};
  const viewQuality = diag.view_quality || {};
  const measured = diag.measured || {};
  const bd = diag.corner_validity_breakdown || {};
  const ok = view.status === 'valid';
  const ratio = Number(view.valid_corner_ratio || measured.valid_corner_ratio || 0);
  const reasons = Array.isArray(diag.reject_reasons) ? diag.reject_reasons : (view.reject_reasons || []);
  const hints = Array.isArray(diag.hints) ? diag.hints : (view.hints || []);
  return `
    <div class="capture-card">
      <div class="capture-title">${view.view_id} - ${ok ? 'PASS' : 'FAIL'} (ratio=${ratio.toFixed(3)})</div>
      <img src="/api/calibration/projector/session/${currentSessionId}/view/${view.view_id}/overlay?rev=${imageRevision}" alt="${view.view_id}"
           onerror="this.onerror=null;this.src='/api/calibration/projector/session/${currentSessionId}/view/${view.view_id}/image?rev=${imageRevision}';" />
      <div style="margin-top:6px;display:flex;gap:8px;align-items:center;">
        <a href="/api/calibration/projector/session/${currentSessionId}/view/${view.view_id}/uv_preview?rev=${imageRevision}" target="_blank">UV preview</a>
        <button data-view="${view.view_id}" class="delete-view-btn">Delete</button>
      </div>
      <div style="font-size:12px;color:#444;margin-top:6px;">ok:${bd.ok ?? 0} nan:${bd.nan_uv ?? 0} mask:${bd.mask_uv_false ?? 0} edge:${(bd.near_edge ?? 0) + (bd.oob ?? 0)}</div>
      <div style="font-size:12px;color:#333;">residual_p95_board=${Number(measured.residual_p95_board ?? NaN).toFixed(3)}</div>
      <div style="font-size:12px;color:#333;">B_median_board=${Number(measured.B_median_board ?? NaN).toFixed(3)}</div>
      <div style="font-size:12px;color:#333;">conditioning=${Number(viewQuality.conditioning_score ?? NaN).toFixed(3)} tilt=${Number(viewQuality.tilt_angle_deg ?? NaN).toFixed(1)}°</div>
      ${reasons.length ? `<div class="error">${reasons.join('; ')}</div>` : ''}
      ${hints.length ? `<div style="font-size:12px;color:#333;">${hints[0]}</div>` : ''}
    </div>
  `;
}

function updateButtons(viewCountValid = 0, coverageSufficient = false) {
  const hasSession = Boolean(currentSessionId);
  const busy = (latestSolveState === 'solving') || (latestSolveState === 'capturing');
  captureBtn.disabled = !hasSession || busy || coverageSufficient;
  solveBtn.disabled = !hasSession || busy || viewCountValid < 10;
  continueSessionBtn.disabled = !sessionSelect.value || busy;
  newSessionBtn.disabled = busy;
}

async function loadSessionsList() {
  const data = await fetchJson('/api/calibration/projector/sessions');
  const sessions = data.sessions || [];
  sessionSelect.innerHTML = '';
  if (!sessions.length) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'No existing sessions';
    sessionSelect.appendChild(opt);
  } else {
    for (const s of sessions) {
      const opt = document.createElement('option');
      opt.value = s.session_id;
      opt.textContent = `${s.session_id} (${s.views_valid}/${s.views_total} valid)`;
      sessionSelect.appendChild(opt);
    }
  }
}

async function refreshSession() {
  if (!currentSessionId) return;
  try {
    const [statusData, viewsData, resultsData, sessionData] = await Promise.all([
      fetchJson(`/api/calibration/projector/session/${currentSessionId}/status`),
      fetchJson(`/api/calibration/projector/session/${currentSessionId}/views`),
      fetchJson(`/api/calibration/projector/session/${currentSessionId}/results`),
      fetchJson(`/api/calibration/projector/session/${currentSessionId}`),
    ]);
    latestSolveState = (statusData.status || {}).state || 'idle';
    statusEl.textContent = `Session ${currentSessionId} | state=${latestSolveState}`;

    const views = viewsData.views || [];
    const validCount = Number(viewsData.views_valid || 0);
    summaryEl.textContent = `Views: ${views.length} | Valid: ${validCount}`;
    const viewsSignature = JSON.stringify(views.map((v) => ({
      id: v.view_id,
      status: v.status,
      accept: !!v.accept,
      ratio: Number(v.valid_corner_ratio || 0),
      reasons: (v.diag?.reject_reasons || v.reject_reasons || []),
      hints: (v.diag?.hints || v.hints || []),
      measured: v.diag?.measured || {},
      breakdown: v.diag?.corner_validity_breakdown || {},
    })));
    if (viewsSignature !== lastViewsSignature) {
      lastViewsSignature = viewsSignature;
      viewsGrid.innerHTML = views.map(viewCardHtml).join('');
      document.querySelectorAll('.delete-view-btn').forEach((btn) => {
        btn.addEventListener('click', async () => {
          const viewId = btn.getAttribute('data-view');
          await fetchJson(`/api/calibration/projector/session/${currentSessionId}/view/${viewId}`, { method: 'DELETE' });
          imageRevision += 1;
          await refreshSession();
        });
      });
    }

    const coverage = (resultsData.coverage && Object.keys(resultsData.coverage).length)
      ? resultsData.coverage
      : (sessionData.coverage || {});
    const coverageSufficient = Boolean(coverage?.sufficient);
    const coverageSignature = JSON.stringify({
      coverage,
      has_results: !!resultsData.has_results,
      solve_state: latestSolveState,
      solve_ended: statusData.status?.ended_at || null,
    });
    if (coverageSignature !== lastCoverageSignature) {
      lastCoverageSignature = coverageSignature;
      renderCoverage(coverage);
      renderCoverageImage(coverageSignature.slice(0, 48));
    }

    if (resultsData.has_results && resultsData.stereo) {
      const st = resultsData.stereo;
      setSolveFeedback(`Solve done. RMS projector=${Number(st.rms_projector_intrinsics || NaN).toFixed(3)} | RMS stereo=${Number(st.rms_stereo || NaN).toFixed(3)} | views=${st.views_used ?? '-'}`);
      const resultsSignature = JSON.stringify(resultsData.stereo);
      if (resultsSignature !== lastResultsSignature) {
        lastResultsSignature = resultsSignature;
        resultsEl.textContent = JSON.stringify(resultsData.stereo, null, 2);
      }
    } else {
      if (lastResultsSignature !== '') {
        lastResultsSignature = '';
        resultsEl.textContent = '';
      }
      if (latestSolveState === 'idle') {
        setSolveFeedback('No solve results yet.');
      }
    }

    const last = sessionData.last_capture_result;
    if (last) {
      if (last.accept) {
        const done = last.coverage_sufficient ? ' Coverage sufficient; run Solve when ready.' : '';
        setCaptureFeedback(`View accepted. Valid corners: ${(100 * Number(last.valid_corner_ratio || 0)).toFixed(1)}%.${done}`);
      } else {
        const reason = (last.reject_reasons && last.reject_reasons.length) ? last.reject_reasons[0] : 'View rejected';
        const hint = (last.hints && last.hints.length) ? last.hints[0] : '';
        setCaptureFeedback(`View rejected. Reason: ${reason}${hint ? ` | Hint: ${hint}` : ''}`);
      }
    }

    if (latestSolveState === 'solving') {
      setSolveFeedback('Solving session (batch mode)...');
    } else if (latestSolveState === 'capturing') {
      setCaptureFeedback('Capturing and processing UV for current pose...');
    } else if (latestSolveState === 'error') {
      setError((statusData.status || {}).error || 'Solve/capture error');
    }

    updateButtons(validCount, coverageSufficient);
  } catch (err) {
    setError(err.message || 'Failed to refresh projector session');
  }
}

async function setProjectorIllumination(enabled) {
  try {
    await fetchJson('/api/calibration/projector/illumination', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: Boolean(enabled) }),
    });
  } catch (err) {
    setError(err.message || 'Failed to set projector illumination');
  }
}

function ensurePolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(() => {
    if (currentSessionId) {
      refreshSession();
    }
  }, 2500);
}

newSessionBtn.addEventListener('click', async () => {
  setError('');
  try {
    const data = await fetchJson('/api/calibration/projector/session/start', { method: 'POST' });
    currentSessionId = data.session_id;
    imageRevision += 1;
    lastViewsSignature = '';
    lastCoverageSignature = '';
    lastResultsSignature = '';
    await setProjectorIllumination(true);
    await loadSessionsList();
    sessionSelect.value = currentSessionId;
    await refreshSession();
  } catch (err) {
    setError(err.message || 'Failed to create session');
  }
});

continueSessionBtn.addEventListener('click', async () => {
  setError('');
  if (!sessionSelect.value) return;
  currentSessionId = sessionSelect.value;
  imageRevision += 1;
  lastViewsSignature = '';
  lastCoverageSignature = '';
  lastResultsSignature = '';
  await setProjectorIllumination(true);
  await refreshSession();
});

sessionSelect.addEventListener('change', () => {
  updateButtons(0, false);
});

captureBtn.addEventListener('click', async () => {
  if (!currentSessionId) return;
  setError('');
  setCaptureFeedback('Capturing pose...');
  latestSolveState = 'capturing';
  updateButtons(0);
  try {
    const data = await fetchJson(`/api/calibration/projector/session/${currentSessionId}/capture`, { method: 'POST' });
    setCaptureFeedback(data.message || 'Capture complete');
    imageRevision += 1;
    lastViewsSignature = '';
  } catch (err) {
    setError(err.message || 'Capture failed');
  } finally {
    await refreshSession();
  }
});

solveBtn.addEventListener('click', async () => {
  if (!currentSessionId) return;
  setError('');
  setSolveFeedback('Starting solve...');
  try {
    await fetchJson(`/api/calibration/projector/session/${currentSessionId}/solve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ background: true }),
    });
    latestSolveState = 'solving';
    lastCoverageSignature = '';
    lastResultsSignature = '';
  } catch (err) {
    setError(err.message || 'Solve failed to start');
  } finally {
    await refreshSession();
  }
});

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

async function init() {
  setupPreview();
  ensurePolling();
  setCaptureFeedback('No capture yet');
  setSolveFeedback('No solve results yet.');
  await loadSessionsList();
  if (sessionSelect.value) {
    currentSessionId = sessionSelect.value;
    await setProjectorIllumination(true);
    await refreshSession();
  } else {
    updateButtons(0);
  }
}

window.addEventListener('beforeunload', () => {
  fetch('/api/calibration/projector/illumination', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled: false }),
    keepalive: true,
  });
});

init();
