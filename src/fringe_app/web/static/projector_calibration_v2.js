const previewImg = document.getElementById('preview_img');
const statusEl = document.getElementById('status');
const errorEl = document.getElementById('error');
const coverageIndicatorEl = document.getElementById('coverage_indicator');
const coverageSummaryEl = document.getElementById('coverage_summary');
const viewsGridEl = document.getElementById('views_grid');
const solveSummaryEl = document.getElementById('solve_summary');
const downloadLinkEl = document.getElementById('download_link');
const plotCamEl = document.getElementById('plot_cam');
const plotProjEl = document.getElementById('plot_proj');
const plotCovEl = document.getElementById('plot_cov');
const plotHistEl = document.getElementById('plot_hist');

const captureBtn = document.getElementById('capture_btn');
const solveBtn = document.getElementById('solve_btn');
const resetBtn = document.getElementById('reset_btn');
const newSessionBtn = document.getElementById('new_session_btn');
const sessionSelect = document.getElementById('session_select');
const continueSessionBtn = document.getElementById('continue_session_btn');

let currentSessionId = null;
let busy = false;
let pollTimer = null;
let imageRev = 0;

function updateControlState() {
  const hasSession = Boolean(currentSessionId);
  const hasSelection = Boolean(sessionSelect.value);
  captureBtn.disabled = busy || !hasSession;
  solveBtn.disabled = busy || !hasSession;
  resetBtn.disabled = busy || !hasSession;
  newSessionBtn.disabled = busy;
  continueSessionBtn.disabled = busy || !hasSelection;
}

function setBusy(v) {
  busy = Boolean(v);
  updateControlState();
}

function setError(msg) {
  errorEl.textContent = msg || '';
}

async function fetchJson(url, opts = undefined) {
  const res = await fetch(url, opts);
  const data = await res.json();
  if (!res.ok || data.ok === false) {
    throw new Error(data.error || `HTTP ${res.status}`);
  }
  return data;
}

function viewCardHtml(view) {
  const accepted = view.status === 'accepted';
  const reason = view.reason || (accepted ? 'Accepted' : 'Rejected');
  const hints = Array.isArray(view.hints) ? view.hints : [];
  const metrics = view.metrics || {};
  const uvCount = Number(metrics.uv_valid_count || 0);
  const uvRatio = Number(metrics.uv_valid_ratio || 0);
  const areaRatio = Number(metrics.area_ratio || 0);
  const tilt = metrics.tilt_deg;

  return `
    <div class="capture-card">
      <div class="capture-title">${view.view_id} - ${accepted ? 'ACCEPTED' : 'REJECTED'}</div>
      <img src="${view.overlay_url}?rev=${encodeURIComponent(String(imageRev))}" alt="${view.view_id}"
           onerror="this.onerror=null;this.src='${view.image_url}?rev=${encodeURIComponent(String(imageRev))}';" />
      <div style="margin-top:6px;font-size:12px;color:#333;">Reason: ${reason}</div>
      <div style="margin-top:4px;font-size:12px;color:#333;">UV valid: ${uvCount} (${(100 * uvRatio).toFixed(1)}%)</div>
      <div style="margin-top:4px;font-size:12px;color:#333;">Area ratio: ${areaRatio.toFixed(3)} | Tilt: ${Number.isFinite(Number(tilt)) ? Number(tilt).toFixed(1) + '°' : '-'}</div>
      ${hints.length ? `<div style="margin-top:4px;font-size:12px;color:#333;">Hint: ${hints[0]}</div>` : ''}
      <div style="margin-top:6px;display:flex;gap:8px;align-items:center;">
        <a href="${view.uv_overlay_url}?rev=${encodeURIComponent(String(imageRev))}" target="_blank">UV overlay</a>
        <button data-view-id="${view.view_id}" class="delete-view-btn">Delete</button>
      </div>
    </div>
  `;
}

function updateLinks(sessionId, hasSolvePlots = false) {
  const rev = encodeURIComponent(String(imageRev));
  downloadLinkEl.href = `/api/calibration/projector_v2/session/${sessionId}/download_zip`;
  if (hasSolvePlots) {
    plotCamEl.href = `/api/calibration/projector_v2/session/${sessionId}/plot/reproj_cam.png?rev=${rev}`;
    plotProjEl.href = `/api/calibration/projector_v2/session/${sessionId}/plot/reproj_proj.png?rev=${rev}`;
    plotCovEl.href = `/api/calibration/projector_v2/session/${sessionId}/plot/coverage.png?rev=${rev}`;
    plotHistEl.href = `/api/calibration/projector_v2/session/${sessionId}/plot/residual_hist.png?rev=${rev}`;
  } else {
    plotCamEl.href = '#';
    plotProjEl.href = '#';
    plotCovEl.href = '#';
    plotHistEl.href = '#';
  }
}

async function loadSessionsList(preferredId = null) {
  const data = await fetchJson('/api/calibration/projector_v2/sessions');
  const sessions = Array.isArray(data.sessions) ? data.sessions : [];
  sessionSelect.innerHTML = '';

  if (sessions.length === 0) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = 'No previous sessions';
    sessionSelect.appendChild(opt);
    updateControlState();
    return sessions;
  }

  for (const s of sessions) {
    const opt = document.createElement('option');
    opt.value = String(s.session_id || '');
    const sid = String(s.session_id || '');
    const a = Number(s.views_accepted || 0);
    const t = Number(s.views_total || 0);
    const solved = s.solved ? ' solved' : '';
    opt.textContent = `${sid} (${a}/${t}${solved})`;
    sessionSelect.appendChild(opt);
  }

  if (preferredId && sessions.some((s) => String(s.session_id) === String(preferredId))) {
    sessionSelect.value = String(preferredId);
  } else if (!sessionSelect.value) {
    sessionSelect.value = String(sessions[0].session_id || '');
  }

  updateControlState();
  return sessions;
}

async function refreshSession() {
  if (!currentSessionId) return;
  try {
    const data = await fetchJson(`/api/calibration/projector_v2/session/${currentSessionId}`);
    statusEl.textContent = `Session ${currentSessionId}`;

    const indicator = data.coverage_indicator || 'Need more variety';
    coverageIndicatorEl.textContent = indicator;
    coverageIndicatorEl.style.background = indicator === 'Sufficient coverage' ? '#d8f0dd' : '#f7e8cf';

    const coverage = data.coverage || {};
    const binsFilled = Number(coverage.bins_filled || 0);
    const minBins = Number(coverage.min_bins_filled || 0);
    const bucketCounts = Array.isArray(coverage.bucket_counts) ? coverage.bucket_counts : [];
    const minBucketCounts = Array.isArray(coverage.min_bucket_counts) ? coverage.min_bucket_counts : [];
    coverageSummaryEl.textContent = `Bins: ${binsFilled}/${minBins} | Tilt buckets: ${JSON.stringify(bucketCounts)} / min ${JSON.stringify(minBucketCounts)}`;

    const views = Array.isArray(data.views) ? data.views : [];
    viewsGridEl.innerHTML = views.map(viewCardHtml).join('');

    document.querySelectorAll('.delete-view-btn').forEach((btn) => {
      btn.addEventListener('click', async () => {
        const viewId = btn.getAttribute('data-view-id');
        if (!viewId) return;
        setBusy(true);
        setError('');
        try {
          await fetchJson(`/api/calibration/projector_v2/session/${currentSessionId}/delete_view/${viewId}`, { method: 'POST' });
          imageRev += 1;
          await loadSessionsList(currentSessionId);
          await refreshSession();
        } catch (err) {
          setError(err.message || 'Failed to delete view');
        } finally {
          setBusy(false);
        }
      });
    });

    const solve = data.solve || {};
    const hasSolvePlots = Boolean(solve && typeof solve === 'object' && solve.selected_model);
    if (hasSolvePlots) {
      const raw = Number(solve.raw_rms_stereo || NaN);
      const pruned = Number(solve.pruned_rms_stereo || NaN);
      solveSummaryEl.textContent = `Solved: model=${solve.selected_model} | raw RMS=${raw.toFixed(4)} | pruned RMS=${pruned.toFixed(4)} | views=${solve.views_used ?? '-'}`;
    } else {
      solveSummaryEl.textContent = 'No solve result yet.';
    }

    updateLinks(currentSessionId, hasSolvePlots);
  } catch (err) {
    setError(err.message || 'Failed to refresh session');
  }
}

async function createSession() {
  setError('');
  const data = await fetchJson('/api/calibration/projector_v2/session/new', { method: 'POST' });
  currentSessionId = String(data.session_id || '');
  imageRev += 1;
  await loadSessionsList(currentSessionId);
  updateLinks(currentSessionId, false);
  await refreshSession();
}

async function continueSelectedSession() {
  const sid = String(sessionSelect.value || '').trim();
  if (!sid) {
    setError('Select a session to continue.');
    return;
  }
  currentSessionId = sid;
  imageRev += 1;
  updateLinks(currentSessionId, false);
  await refreshSession();
}

captureBtn.addEventListener('click', async () => {
  if (!currentSessionId) return;
  setBusy(true);
  setError('');
  try {
    await fetchJson(`/api/calibration/projector_v2/session/${currentSessionId}/capture`, { method: 'POST' });
    imageRev += 1;
    await loadSessionsList(currentSessionId);
    await refreshSession();
  } catch (err) {
    setError(err.message || 'Capture failed');
  } finally {
    setBusy(false);
  }
});

solveBtn.addEventListener('click', async () => {
  if (!currentSessionId) return;
  setBusy(true);
  setError('');
  try {
    await fetchJson(`/api/calibration/projector_v2/session/${currentSessionId}/solve`, { method: 'POST' });
    imageRev += 1;
    await loadSessionsList(currentSessionId);
    await refreshSession();
  } catch (err) {
    setError(err.message || 'Solve failed');
  } finally {
    setBusy(false);
  }
});

resetBtn.addEventListener('click', async () => {
  if (!currentSessionId) return;
  setBusy(true);
  setError('');
  try {
    await fetchJson(`/api/calibration/projector_v2/session/${currentSessionId}/reset`, { method: 'POST' });
    imageRev += 1;
    await loadSessionsList(currentSessionId);
    await refreshSession();
  } catch (err) {
    setError(err.message || 'Reset failed');
  } finally {
    setBusy(false);
  }
});

newSessionBtn.addEventListener('click', async () => {
  setBusy(true);
  setError('');
  try {
    await createSession();
  } catch (err) {
    setError(err.message || 'Failed to create session');
  } finally {
    setBusy(false);
  }
});

continueSessionBtn.addEventListener('click', async () => {
  setBusy(true);
  setError('');
  try {
    await continueSelectedSession();
  } catch (err) {
    setError(err.message || 'Failed to continue session');
  } finally {
    setBusy(false);
  }
});

sessionSelect.addEventListener('change', () => {
  updateControlState();
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

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = setInterval(() => {
    if (!busy && currentSessionId) {
      refreshSession();
    }
  }, 2500);
}

async function init() {
  setBusy(true);
  setupPreview();
  try {
    const sessions = await loadSessionsList();
    if (sessions.length > 0) {
      await continueSelectedSession();
    } else {
      await createSession();
    }
  } catch (err) {
    setError(err.message || 'Failed to initialize session');
  } finally {
    setBusy(false);
  }
  startPolling();
}

init();
