const statusEl = document.getElementById('status');
const runsList = document.getElementById('runs_list');
const previewImg = document.getElementById('preview_img');

function getValue(id) {
  const el = document.getElementById(id);
  if (el.type === 'checkbox') return el.checked;
  if (el.value === '') return null;
  return el.value;
}

async function refreshStatus() {
  const res = await fetch('/api/status');
  const data = await res.json();
  statusEl.textContent = `${data.state} (${data.progress}/${data.total})`;
}

async function refreshRuns() {
  const res = await fetch('/api/runs');
  const runs = await res.json();
  runsList.innerHTML = '';
  runs.forEach(run => {
    const li = document.createElement('li');
    const link = document.createElement('a');
    link.href = `/api/runs/${run.run_id}/download`;
    link.textContent = `Download ${run.run_id}`;
    const title = document.createElement('div');
    title.textContent = `${run.run_id} - ${run.status} `;
    title.appendChild(link);

    const computeBtn = document.createElement('button');
    computeBtn.textContent = 'Compute Phase';
    computeBtn.addEventListener('click', async () => {
      computeBtn.disabled = true;
      try {
        const res = await fetch(`/api/runs/${run.run_id}/phase/compute`, { method: 'POST' });
        if (!res.ok) {
          const err = await res.json();
          alert(`Phase compute failed: ${err.error || res.status}`);
        }
      } catch (e) {
        alert(`Phase compute failed: ${e}`);
      } finally {
        computeBtn.disabled = false;
        await refreshRuns();
      }
    });

    const phaseStatus = document.createElement('div');
    const phaseImg = document.createElement('img');
    phaseImg.className = 'phase-preview';
    const captureImg = document.createElement('img');
    captureImg.className = 'phase-preview';
    const roiImg = document.createElement('img');
    roiImg.className = 'phase-preview';
    const validRoiImg = document.createElement('img');
    validRoiImg.className = 'phase-preview';

    captureImg.src = `/api/runs/${run.run_id}/capture/preview?ts=${Date.now()}`;

    fetch(`/api/runs/${run.run_id}/phase/status`).then(r => r.json()).then(stat => {
      if (stat.exists) {
        const valid = stat.meta.valid_pct?.toFixed(2);
        const bmed = stat.meta.B_median?.toFixed(2);
        const roiValid = stat.meta.roi_valid_ratio?.toFixed(2);
        const roiB = stat.meta.roi_b_median?.toFixed(2);
        phaseStatus.textContent = `Phase: valid ${valid}% | B median ${bmed} | ROI valid ${roiValid}% | ROI B ${roiB}`;
        phaseImg.src = `/api/runs/${run.run_id}/phase/preview?ts=${Date.now()}`;
        roiImg.src = `/api/runs/${run.run_id}/roi/preview?ts=${Date.now()}`;
        validRoiImg.src = `/api/runs/${run.run_id}/phase/overlay?ts=${Date.now()}`;
      } else {
        phaseStatus.textContent = 'Phase: not computed';
      }
    });

    li.appendChild(title);
    li.appendChild(computeBtn);
    li.appendChild(captureImg);
    li.appendChild(phaseStatus);
    li.appendChild(phaseImg);
    li.appendChild(roiImg);
    li.appendChild(validRoiImg);
    runsList.appendChild(li);
  });
}

document.getElementById('start_btn').addEventListener('click', async () => {
  const payload = {
    n_steps: parseInt(getValue('n_steps'), 10),
    frequency: parseFloat(getValue('frequency')),
    orientation: getValue('orientation'),
    brightness: parseFloat(getValue('brightness')),
    settle_ms: parseInt(getValue('settle_ms'), 10),
    preview_fps: parseFloat(getValue('preview_fps')),
    exposure_us: getValue('exposure_us') ? parseInt(getValue('exposure_us'), 10) : null,
    save_patterns: getValue('save_patterns'),
    preview_enabled: getValue('preview_enabled'),
  };

  await fetch('/api/scan/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });
  refreshStatus();
});

document.getElementById('stop_btn').addEventListener('click', async () => {
  await fetch('/api/scan/stop', { method: 'POST' });
  refreshStatus();
});

function setupPreview() {
  const ws = new WebSocket(`ws://${location.host}/ws/preview`);
  ws.binaryType = 'arraybuffer';
  let lastUrl = null;
  ws.onmessage = (event) => {
    const blob = new Blob([event.data], { type: 'image/jpeg' });
    const url = URL.createObjectURL(blob);
    if (lastUrl) {
      URL.revokeObjectURL(lastUrl);
    }
    lastUrl = url;
    previewImg.src = url;
  };
}

setInterval(refreshStatus, 1000);
setInterval(refreshRuns, 4000);
refreshStatus();
refreshRuns();
setupPreview();
