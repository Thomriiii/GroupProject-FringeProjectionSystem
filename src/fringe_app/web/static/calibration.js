const previewImg = document.getElementById('preview_img');
const statusEl = document.getElementById('calib_status');
const errorEl = document.getElementById('calib_error');
const summaryEl = document.getElementById('calib_summary');
const capturesGrid = document.getElementById('captures_grid');
const newSessionBtn = document.getElementById('new_session_btn');
const captureBtn = document.getElementById('capture_btn');
const calibrateBtn = document.getElementById('calibrate_btn');

let currentSessionId = null;

function setError(msg) {
  errorEl.textContent = msg || '';
}

function updateButtons() {
  const hasSession = Boolean(currentSessionId);
  captureBtn.disabled = !hasSession;
  calibrateBtn.disabled = !hasSession;
}

async function refreshSession() {
  if (!currentSessionId) return;
  const res = await fetch(`/api/calibration/sessions/${currentSessionId}`);
  const data = await res.json();
  if (!data.ok) {
    setError(data.error || 'Failed to load calibration session');
    return;
  }
  const session = data.session;
  const captures = session.captures || [];
  const found = captures.filter((c) => c.found).length;
  statusEl.textContent = `Session ${session.session_id}`;
  summaryEl.textContent = `Captures: ${captures.length} | Found: ${found}`;
  renderCaptures(captures);
}

function renderCaptures(captures) {
  capturesGrid.innerHTML = '';
  captures.forEach((capture) => {
    const card = document.createElement('div');
    card.className = 'capture-card';

    const title = document.createElement('div');
    title.className = 'capture-title';
    title.textContent = `${capture.capture_id} - ${capture.found ? 'found' : 'not found'}`;

    const image = document.createElement('img');
    image.alt = capture.capture_id;
    image.src = `/api/calibration/sessions/${currentSessionId}/captures/${capture.capture_id}/overlay?ts=${Date.now()}`;

    card.appendChild(title);
    card.appendChild(image);
    capturesGrid.appendChild(card);
  });
}

newSessionBtn.addEventListener('click', async () => {
  setError('');
  const res = await fetch('/api/calibration/sessions', { method: 'POST' });
  const data = await res.json();
  if (!data.ok) {
    setError(data.error || 'Failed to create calibration session');
    return;
  }
  currentSessionId = data.session.session_id;
  updateButtons();
  await refreshSession();
});

captureBtn.addEventListener('click', async () => {
  if (!currentSessionId) return;
  setError('');
  const res = await fetch(`/api/calibration/sessions/${currentSessionId}/capture`, { method: 'POST' });
  const data = await res.json();
  if (!data.ok) {
    setError(data.error || 'Capture failed');
    return;
  }
  await refreshSession();
});

calibrateBtn.addEventListener('click', async () => {
  if (!currentSessionId) return;
  setError('');
  const res = await fetch(`/api/calibration/sessions/${currentSessionId}/calibrate`, { method: 'POST' });
  const data = await res.json();
  if (!data.ok) {
    setError(data.error || 'Calibration failed');
    return;
  }
  const intrinsics = data.intrinsics || {};
  const rms = Number(intrinsics.rms || 0).toFixed(4);
  summaryEl.textContent = `${summaryEl.textContent} | RMS: ${rms}`;
});

function setupPreview() {
  if (!previewImg) return;
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
  updateButtons();
  setupPreview();
  const res = await fetch('/api/calibration/sessions');
  const data = await res.json();
  const sessions = (data.sessions || []);
  if (sessions.length > 0) {
    currentSessionId = sessions[0].session_id;
    updateButtons();
    await refreshSession();
  }
}

init();
