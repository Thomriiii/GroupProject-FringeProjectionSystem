#!/usr/bin/env python3
import os
import time
import threading
from datetime import datetime
import numpy as np
import cv2
import pygame

from flask import Flask, Response, render_template_string
from picamera2 import Picamera2

# ============================================================
# CONFIG
# ============================================================
FREQS = [4, 8, 16, 32]     # fringe periods across width
N_PHASE = 4                # 4-step PSP: 0, 90, 180, 270 deg

# Tuned thresholds (from analysis)
TH1 = 0.18                 # gamma threshold (visibility)
TH2 = 20.0                 # modulation threshold B > TH2

SCAN_ROOT = "fringe_web_test1/scans"   # where scans are saved
os.makedirs(SCAN_ROOT, exist_ok=True)

# ============================================================
# KMSDRM + PYGAME DISPLAY (MAIN THREAD ONLY)
# ============================================================
os.environ.setdefault("SDL_VIDEODRIVER", "kmsdrm")

pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
W, H = pygame.display.get_window_size()
print(f"[HDMI] Resolution {W}x{H}")

surface_lock = threading.Lock()
current_surface = None  # will be set after pattern generation

# ============================================================
# PATTERN GENERATION
# ============================================================
def generate_psp_patterns(width, height, freqs, n_phase):
    """
    Generate PSP patterns:
        patterns[f][n] is a pygame.Surface for frequency f, phase index n.
        raw_patterns[f][n] is a float array [0..1] (not strictly needed).
    """
    patterns = {}
    raw_patterns = {}
    deltas = 2.0 * np.pi * np.arange(n_phase) / n_phase  # [0, pi/2, pi, 3pi/2]

    xs = np.linspace(0, 1, width, endpoint=False)  # normalized 0..1 along width

    for f in freqs:
        freq_patterns = []
        freq_raw = []
        for n, delta in enumerate(deltas):
            # sin(2π f x + delta)
            phase = 2.0 * np.pi * f * xs + delta
            stripe = 0.5 + 0.5 * np.sin(phase)  # 0..1
            img = np.tile(stripe, (height, 1))  # HxW
            img_u8 = (img * 255).astype(np.uint8)
            rgb = np.dstack([img_u8] * 3)

            surf = pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))
            freq_patterns.append(surf)
            freq_raw.append(img)

        patterns[f] = freq_patterns
        raw_patterns[f] = freq_raw

    return patterns, raw_patterns

print("[PATTERN] Generating PSP patterns...")
patterns, raw_patterns = generate_psp_patterns(W, H, FREQS, N_PHASE)
# choose first frequency, first phase as idle pattern
with surface_lock:
    current_surface = patterns[FREQS[0]][0]
print("[PATTERN] Done.")

# ============================================================
# CAMERA SETUP (HEADLESS, NO DISPLAY)
# ============================================================
picam2 = Picamera2()
cam_config = picam2.create_video_configuration(
    main={"size": (1280, 720), "format": "RGB888"},
    lores={"size": (640, 480), "format": "YUV420"},
    display=None,           # CRITICAL: no DRM preview
    buffer_count=3,
    controls={"FrameRate": 30}
)
picam2.configure(cam_config)
picam2.start()
time.sleep(0.3)
print("[CAMERA] Picamera2 active.")

camera_lock = threading.Lock()

# ============================================================
# MJPEG STREAM
# ============================================================
def mjpeg_stream():
    while True:
        with camera_lock:
            frame = picam2.capture_array("main")  # RGB
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, jpg = cv2.imencode(".jpg", frame_bgr,
                               [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue
        buf = jpg.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(buf)).encode() + b"\r\n\r\n" +
               buf + b"\r\n")

# ============================================================
# PHASE-SHIFT PROCESSING FUNCTIONS
# ============================================================
def run_psp_and_unwrap(I_dict, freqs, n_phase, Th1, Th2):
    """
    I_dict[(f, n)] = grayscale image (H x W) for frequency f and phase index n.
    Returns:
        Phi_final (H x W): unwrapped phase at highest frequency (f_max)
        mask_final (H x W): boolean mask of valid pixels (combined up to f=16)
    """
    freqs_sorted = sorted(freqs)
    deltas = 2.0 * np.pi * np.arange(n_phase) / n_phase

    phi_wrapped = {}
    A = {}
    B = {}
    gamma = {}
    valid = {}

    # ----- Per-frequency PSP -----
    for f in freqs_sorted:
        stack = np.stack([I_dict[(f, n)] for n in range(n_phase)], axis=0)  # (N, H, W)

        sin_d = np.sin(deltas)[:, None, None]
        cos_d = np.cos(deltas)[:, None, None]

        S = np.sum(stack * sin_d, axis=0)
        C = np.sum(stack * cos_d, axis=0)

        phi_f = np.arctan2(S, C)
        phi_wrapped[f] = phi_f

        A_f = stack.mean(axis=0)
        B_f = (2.0 / n_phase) * np.sqrt(S**2 + C**2)

        A[f] = A_f
        B[f] = B_f

        gamma_f = np.zeros_like(A_f)
        mask_nonzero = A_f > 1e-6
        gamma_f[mask_nonzero] = B_f[mask_nonzero] / A_f[mask_nonzero]
        gamma[f] = gamma_f

        I_min = stack.min(axis=0)
        I_max = stack.max(axis=0)
        sat_mask = (I_max > 250) | (I_min < 5)

        valid_f = (gamma_f > Th1) & (B_f > Th2) & (~sat_mask)
        valid[f] = valid_f

    # ----- Temporal unwrapping -----
    Phi = {}
    f0 = freqs_sorted[0]
    Phi[f0] = phi_wrapped[f0]

    # Mask combining: only AND up to f = 16 (i.e. indices 0,1,2)
    valid_total = valid[f0].copy()

    for i in range(1, len(freqs_sorted)):
        f_low = freqs_sorted[i - 1]
        f_high = freqs_sorted[i]
        r = f_high / f_low

        phi_low = Phi[f_low]
        phi_high_wrapped = phi_wrapped[f_high]

        k = np.round((r * phi_low - phi_high_wrapped) / (2.0 * np.pi))
        Phi_high = phi_high_wrapped + 2.0 * np.pi * k
        Phi[f_high] = Phi_high

        # Only AND masks up to f=16 (i.e. for f = 4, 8, 16)
        if i <= 2:  # freqs_sorted[1] and freqs_sorted[2]
            valid_total &= valid[f_high]

    # Final phase from highest frequency (f = 32)
    Phi_final = Phi[freqs_sorted[-1]]
    mask_final = valid_total

    return Phi_final, mask_final

def save_phase_outputs(Phi_final, mask_final, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Optional: hook for phase smoothing (if you later add scipy / your own filter)
    # from scipy.ndimage import median_filter
    # Phi_final = median_filter(Phi_final, size=3)

    np.save(os.path.join(out_dir, "phase_final.npy"), Phi_final)
    np.save(os.path.join(out_dir, "mask_final.npy"), mask_final)

    phase_display = Phi_final.copy()
    phase_display[~mask_final] = np.nan

    finite_vals = phase_display[np.isfinite(phase_display)]
    if finite_vals.size > 0:
        p_min, p_max = np.percentile(finite_vals, [1, 99])
        phase_display = np.clip(phase_display, p_min, p_max)
    else:
        p_min, p_max = -np.pi, np.pi

    norm = (phase_display - p_min) / (p_max - p_min + 1e-9)
    norm[~np.isfinite(norm)] = 0.0
    img_u8 = (norm * 255).astype(np.uint8)
    img_color = cv2.applyColorMap(img_u8, cv2.COLORMAP_TURBO)
    cv2.imwrite(os.path.join(out_dir, "phase_debug.png"), img_color)

    print(f"[PHASE] Saved phase_final.npy, mask_final.npy, phase_debug.png in {out_dir}")

# ============================================================
# SCAN PIPELINE
# ============================================================
scan_status_lock = threading.Lock()
scan_status = "Idle"
last_scan_dir = None

def run_scan():
    global scan_status, last_scan_dir, current_surface

    with scan_status_lock:
        scan_status = "Running"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scan_dir = os.path.join(SCAN_ROOT, timestamp)
    os.makedirs(scan_dir, exist_ok=True)
    print(f"[SCAN] Starting scan: {scan_dir}")

    I_dict = {(f, n): None for f in FREQS for n in range(N_PHASE)}

    for f in FREQS:
        for n in range(N_PHASE):
            with surface_lock:
                current_surface = patterns[f][n]  # instruct projector
            time.sleep(0.15)  # allow pattern to stabilize

            with camera_lock:
                frame = picam2.capture_array("main")  # RGB
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            I_dict[(f, n)] = frame_gray.astype(np.float32)

            fname = f"f{f:03d}_n{n:02d}.png"
            cv2.imwrite(os.path.join(scan_dir, fname), frame_gray)
            print(f"[SCAN] Captured {fname}, mean={frame_gray.mean():.2f}")

    print("[SCAN] All frames captured, running PSP + unwrap...")
    Phi_final, mask_final = run_psp_and_unwrap(I_dict, FREQS, N_PHASE, TH1, TH2)
    save_phase_outputs(Phi_final, mask_final, scan_dir)

    with surface_lock:
        current_surface = patterns[FREQS[0]][0]

    with scan_status_lock:
        scan_status = "Complete"
        last_scan_dir = scan_dir

    print(f"[SCAN] Complete. Results in {scan_dir}")

def start_scan_thread():
    with scan_status_lock:
        if scan_status == "Running":
            print("[SCAN] Already running.")
            return False
        else:
            t = threading.Thread(target=run_scan, daemon=True)
            t.start()
            return True

# ============================================================
# FLASK WEB UI
# ============================================================
app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <title>Fringe Unit</title>
</head>
<body>
  <h1>Fringe Projection – Minimal Stack</h1>
  <h2>Live Camera</h2>
  <img src="/video" style="max-width: 640px;"><br>

  <h2>Scan Control</h2>
  <p>Status: <strong>{{ status }}</strong></p>
  {% if last_dir %}
    <p>Last scan: {{ last_dir }}</p>
  {% endif %}
  <form method="post" action="/scan">
    <button type="submit">Run Scan</button>
  </form>

  <p>HDMI output is continuously showing the current fringe pattern.</p>
</body>
</html>
"""

@app.route("/")
def index():
    with scan_status_lock:
        st = scan_status
        ld = last_scan_dir
    return render_template_string(INDEX_HTML, status=st, last_dir=ld)

@app.route("/video")
def video_route():
    return Response(mjpeg_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.post("/scan")
def scan_route():
    started = start_scan_thread()
    if not started:
        return "Scan already running. <a href='/'>Back</a>"
    return "Scan started. <a href='/'>Back</a>"

def run_flask():
    print("[WEB] Serving on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)

# ============================================================
# MAIN LOOP – PROJECTOR IN MAIN THREAD
# ============================================================
if __name__ == "__main__":
    web_thread = threading.Thread(target=run_flask, daemon=True)
    web_thread.start()

    clock = pygame.time.Clock()
    print("[PROJECTOR] Main loop running.")

    while True:
        with surface_lock:
            surf = current_surface
        if surf is not None:
            screen.blit(surf, (0, 0))
            pygame.display.flip()
        clock.tick(60)
