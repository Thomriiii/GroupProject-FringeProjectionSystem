const previewImg = document.getElementById('preview_img');
const statusEl = document.getElementById('status');
const errorEl = document.getElementById('error');
const summaryEl = document.getElementById('summary');
const viewsGrid = document.getElementById('views_grid');
const resultsEl = document.getElementById('results_json');
const newSessionBtn = document.getElementById('new_session_btn');
const actionsEl = document.querySelector('section.actions');
let sessionSelect = document.getElementById('session_select');
let continueSessionBtn = document.getElementById('continue_session_btn');
const captureBtn = document.getElementById('capture_view_btn');
const calibrateBtn = document.getElementById('calibrate_btn');

if (!sessionSelect && actionsEl) {
  sessionSelect = document.createElement('select');
  sessionSelect.id = 'session_select';
  if (newSessionBtn && newSessionBtn.nextSibling) {
    actionsEl.insertBefore(sessionSelect, newSessionBtn.nextSibling);
  } else {
    actionsEl.appendChild(sessionSelect);
  }
}
if (!continueSessionBtn && actionsEl) {
  continueSessionBtn = document.createElement('button');
  continueSessionBtn.id = 'continue_session_btn';
  continueSessionBtn.textContent = 'Continue Session';
  if (sessionSelect && sessionSelect.nextSibling) {
    actionsEl.insertBefore(continueSessionBtn, sessionSelect.nextSibling);
  } else {
    actionsEl.appendChild(continueSessionBtn);
  }
}

let currentSessionId = null;

function setError(msg) {
  errorEl.textContent = msg || '';
}

function updateButtons(viewCount = 0) {
  const hasSession = Boolean(currentSessionId);
  captureBtn.disabled = !hasSession;
  calibrateBtn.disabled = !hasSession || viewCount < 10;
  if (continueSessionBtn && sessionSelect) {
    continueSessionBtn.disabled = !sessionSelect.value;
  }
}

function viewCardHtml(v) {
  const ok = v.status === 'valid';
  const ratio = Number(v.valid_corner_ratio || 0).toFixed(3);
  const diag = v.diag || {};
  const bd = diag.corner_validity_breakdown || {};
  const hint = (diag.hints && diag.hints.length) ? diag.hints[0] : '';
  const breakdown = `ok:${bd.ok ?? 0} nan:${bd.nan_uv ?? 0} mask:${bd.mask_uv_false ?? 0} edge:${(bd.near_edge ?? 0) + (bd.oob ?? 0)}`;
  return `
    <div class="capture-card">
      <div class="capture-title">
        ${v.view_id} - ${ok ? 'valid' : 'invalid'} (ratio=${ratio})
      </div>
      <img src="/api/calibration/projector/session/${currentSessionId}/view/${v.view_id}/overlay?ts=${Date.now()}" alt="${v.view_id}" />
      <div style="margin-top:6px;display:flex;gap:8px;align-items:center;">
        <a href="/api/calibration/projector/session/${currentSessionId}/view/${v.view_id}/uv_preview?ts=${Date.now()}" target="_blank">UV preview</a>
        <button data-view="${v.view_id}" class="delete-view-btn">Delete</button>
      </div>
      <div style="font-size:12px;color:#444;margin-top:6px;">${breakdown}</div>
      ${v.reason ? `<div class="error">${v.reason}</div>` : ''}
      ${hint ? `<div style="font-size:12px;color:#333;">${hint}</div>` : ''}
    </div>
  `;
}

async function refreshSession() {
  if (!currentSessionId) return;
  const res = await fetch(`/api/calibration/projector/session/${currentSessionId}`);
  const data = await res.json();
  if (!data.ok) {
    setError(data.error || 'Failed to load session');
    return;
  }
  const views = data.views || [];
  statusEl.textContent = `Session ${currentSessionId}`;
  summaryEl.textContent = `Views: ${views.length} | Valid: ${views.filter(v => v.status === 'valid').length}`;
  updateButtons(views.filter(v => v.status === 'valid').length);

  viewsGrid.innerHTML = views.map(viewCardHtml).join('');
  document.querySelectorAll('.delete-view-btn').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const vid = btn.getAttribute('data-view');
      await fetch(`/api/calibration/projector/session/${currentSessionId}/view/${vid}`, { method: 'DELETE' });
      await refreshSession();
    });
  });

  const result = data.session?.results;
  resultsEl.textContent = result ? JSON.stringify(result, null, 2) : '';
}

async function setProjectorIllumination(enabled) {
  try {
    await fetch('/api/calibration/projector/illumination', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enabled: Boolean(enabled) }),
    });
  } catch (_) {
    // Keep UI responsive; endpoint failures are surfaced elsewhere.
  }
}

async function loadSessionsList() {
  if (!sessionSelect) return;
  const res = await fetch('/api/calibration/projector/sessions');
  const data = await res.json();
  if (!data.ok) {
    return;
  }
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
  updateButtons();
}

async function continueSession(sessionId) {
  if (!sessionId) return;
  currentSessionId = sessionId;
  await setProjectorIllumination(true);
  await refreshSession();
}

newSessionBtn.addEventListener('click', async () => {
  setError('');
  const res = await fetch('/api/calibration/projector/session/new', { method: 'POST' });
  const data = await res.json();
  if (!data.ok) {
    setError(data.error || 'Failed to create session');
    return;
  }
  currentSessionId = data.session_id;
  await setProjectorIllumination(true);
  resultsEl.textContent = '';
  await loadSessionsList();
  if (sessionSelect) {
    sessionSelect.value = currentSessionId;
  }
  await refreshSession();
});

if (continueSessionBtn) {
  continueSessionBtn.addEventListener('click', async () => {
    setError('');
    await continueSession(sessionSelect ? sessionSelect.value : '');
  });
}

if (sessionSelect) {
  sessionSelect.addEventListener('change', () => updateButtons());
}

captureBtn.addEventListener('click', async () => {
  if (!currentSessionId) return;
  setError('');
  captureBtn.disabled = true;
  try {
    statusEl.textContent = `Session ${currentSessionId} | Capturing pose...`;
    const res = await fetch(`/api/calibration/projector/session/${currentSessionId}/capture`, { method: 'POST' });
    const data = await res.json();
    if (!data.ok) {
      setError(data.error || 'Capture failed');
    }
  } catch (err) {
    setError('Capture failed');
  } finally {
    await refreshSession();
  }
});

calibrateBtn.addEventListener('click', async () => {
  if (!currentSessionId) return;
  setError('');
  statusEl.textContent = `Session ${currentSessionId} | Calibrating...`;
  const res = await fetch(`/api/calibration/projector/session/${currentSessionId}/calibrate`, { method: 'POST' });
  const data = await res.json();
  if (!data.ok) {
    setError(data.error || 'Calibration failed');
    await refreshSession();
    return;
  }
  resultsEl.textContent = JSON.stringify(data.result, null, 2);
  await refreshSession();
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
  await loadSessionsList();
  if (sessionSelect && sessionSelect.value) {
    await continueSession(sessionSelect.value);
  } else {
    const r = await fetch('/api/calibration/projector/session/new', { method: 'POST' });
    const d = await r.json();
    if (d.ok) {
      currentSessionId = d.session_id;
      await setProjectorIllumination(true);
      await loadSessionsList();
      if (sessionSelect) {
        sessionSelect.value = currentSessionId;
      }
      await refreshSession();
    } else {
      setError(d.error || 'Failed to init session');
    }
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
