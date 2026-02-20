const statusEl = document.getElementById('status');
const runsList = document.getElementById('runs_list');
const previewImg = document.getElementById('preview_img');
const startPipelineBtn = document.getElementById('start_pipeline_btn');

async function refreshPipelineStatus() {
  const [scanRes, pipeRes] = await Promise.all([
    fetch('/api/status'),
    fetch('/api/pipeline/status'),
  ]);
  const scan = await scanRes.json();
  const pipe = await pipeRes.json();
  const runId = pipe.run_id ? ` | run ${pipe.run_id}` : '';
  const step = scan.total ? ` (${scan.progress}/${scan.total})` : '';
  statusEl.textContent = `${pipe.state}${step}${runId}`;
  startPipelineBtn.disabled = pipe.state === 'running';
}

async function refreshRuns() {
  const res = await fetch('/api/runs');
  const runs = await res.json();
  runsList.innerHTML = '';

  for (const run of runs) {
    const li = document.createElement('li');
    const label = document.createElement('span');
    label.textContent = `${run.run_id} - ${run.status} `;

    const link = document.createElement('a');
    link.href = `/api/runs/${run.run_id}/download`;
    link.textContent = 'Download ZIP';

    li.appendChild(label);
    li.appendChild(link);
    runsList.appendChild(li);
  }
}

startPipelineBtn.addEventListener('click', async () => {
  await fetch('/api/pipeline/start', {
    method: 'POST',
  });
  refreshPipelineStatus();
  refreshRuns();
});

function setupPreview() {
  if (!previewImg) return;
  previewImg.style.display = '';
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

setInterval(refreshPipelineStatus, 1000);
setInterval(refreshRuns, 4000);
refreshPipelineStatus();
refreshRuns();
setupPreview();
