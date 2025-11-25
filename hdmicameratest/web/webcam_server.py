#!/usr/bin/env python3
import io
import os
import time
import threading
import cv2
import numpy as np
from flask import Flask, Response, request, redirect, url_for, render_template_string
from picamera2 import Picamera2

# ---------------- Camera setup ----------------
picam2 = Picamera2()

# Default stream output size (what the browser gets) and capture size (what the sensor captures).
# By default we capture at the same size we stream. You can choose a larger `capture_size` to
# capture at the sensor/native resolution and downscale server-side for the stream.
DEFAULT_STREAM_SIZE = (1280, 720)
# A commonly-available larger sensor size on Raspberry Pi camera modules; change if your sensor differs.
DEFAULT_NATIVE_CAPTURE = (3280, 2464)

# Start with capture == stream
stream_w, stream_h = DEFAULT_STREAM_SIZE
capture_w, capture_h = DEFAULT_STREAM_SIZE

# pick a fast live-preview size; you can change later via a setting
video_config = picam2.create_video_configuration(
  main={"size": (capture_w, capture_h)},  # capture size (may be larger than streamed size)
  controls={"FrameRate": 30}
)
picam2.configure(video_config)
picam2.start()
time.sleep(0.2)  # let AE/AF settle (IMX296 has fixed focus, but AE/Gain settle)

# Protects concurrent access to zoom and stream/capture size state
config_lock = threading.Lock()

# ---------------- Flask app ----------------
app = Flask(__name__)

# Global zoom state (digital zoom). 1.0 = no zoom. Values >1 crop-center and scale up.
zoom_lock = threading.Lock()
zoom_factor = 1.0

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>RPi Fringe Unit</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    :root { color-scheme: dark light; }
    body { font-family: system-ui, sans-serif; margin: 1rem; }
    header { display:flex; align-items:center; gap:.75rem; margin-bottom:1rem; }
    h1 { font-size: 1.2rem; margin:0; }
    .row { display:flex; gap:1rem; flex-wrap:wrap; }
    .card { border:1px solid #ccc; border-radius:.5rem; padding:1rem; flex:1 1 360px; }
    img { max-width:100%; height:auto; display:block; background:#000; }
    label { display:block; margin:.5rem 0 .25rem; font-weight:600; }
    button, input[type="submit"] { padding:.5rem .75rem; border-radius:.4rem; border:1px solid #888; cursor:pointer; }
    input[type="text"], select { padding:.4rem; width:100%; }
    .muted { font-size:.9rem; opacity:.75; }
  </style>
</head>
<body>
  <header>
    <h1>Defect Detection – Web Control</h1>
    <span class="muted">Live camera + future projector controls</span>
  </header>

  <div class="row">
    <section class="card">
      <h2>Live Camera</h2>
      <img id="stream" src="{{ url_for('video_feed') }}" alt="Camera stream">
      <p class="muted">If the stream stalls, reload the page.</p>
      <form method="post" action="{{ url_for('apply_camera_settings') }}">
        <label for="size">Stream Resolution (applies on restart)</label>
        <select id="size" name="size">
          <option value="1280x720" selected>1280×720 @30</option>
          <option value="1920x1080">1920×1080 @30</option>
          <option value="640x480">640×480 @30</option>
        </select>
        <label for="capture">Capture size (server-side) — capture larger and downscale for streaming</label>
        <select id="capture" name="capture">
          <option value="same" selected>Same as stream (default)</option>
          <option value="native">Native sensor (capture at sensor max, downscale for stream)</option>
          <option value="3280x2464">3280×2464 (explicit)</option>
        </select>
        <label for="fps">FPS</label>
        <select id="fps" name="fps">
          <option value="30" selected>30</option>
          <option value="24">24</option>
          <option value="15">15</option>
        </select>
        <div style="margin-top:.5rem">
          <input type="submit" value="Apply & Restart Stream">
        </div>
      </form>
      <form id="zoomForm" method="post" action="{{ url_for('set_zoom') }}" style="margin-top:.5rem">
        <label for="zoom">Digital zoom (1 = no zoom)</label>
        <input id="zoom" name="zoom" type="range" min="1" max="8" step="0.1" value="1" oninput="zoomVal.innerText=this.value">
        <div style="display:flex; gap:.5rem; align-items:center; margin-top:.25rem">
          <span class="muted">Zoom:</span> <strong id="zoomVal">1</strong>
          <button type="submit" style="margin-left:1rem">Apply zoom</button>
          <button type="button" onclick="fetch('{{ url_for('set_zoom') }}', {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'}, body:'zoom=1'}).then(()=>location.reload())">Max out (zoom out)</button>
        </div>
      </form>
    </section>

    <section class="card">
      <h2>Projector (future)</h2>
      <p>Hooks ready for selecting which image/pattern to show on HDMI.</p>
      <form method="post" action="{{ url_for('project_placeholder') }}">
        <label for="image_path">Image to display (path on Pi)</label>
        <input id="image_path" name="image_path" type="text" placeholder="/home/pi/fringe/assets/pattern1.png">
        <div style="margin-top:.5rem">
          <input type="submit" value="Show on Projector">
        </div>
      </form>
    </section>
  </div>
</body>
</html>
"""

def mjpeg_generator():
    """
    MJPEG multipart stream from the Picamera2 main output.
    Uses OpenCV to JPEG-encode frames.
    """
    while True:
        # RGB array from Picamera2
        frame = picam2.capture_array("main")

        # Read current config (stream size) and zoom under locks
        with config_lock:
            out_w, out_h = stream_w, stream_h
        with zoom_lock:
            z = float(zoom_factor)

        # Work on the full-resolution captured frame (capture_w, capture_h). Apply zoom cropping
        # on the captured image, then always resize to the output (stream) size so the browser
        # receives frames at a consistent resolution.
        cap_h, cap_w = frame.shape[:2]
        if z > 1.0:
            # compute crop size (must be integers)
            cw = max(1, int(cap_w / z))
            ch = max(1, int(cap_h / z))
            cx = cap_w // 2
            cy = cap_h // 2
            x0 = max(0, cx - cw // 2)
            y0 = max(0, cy - ch // 2)
            x1 = min(cap_w, x0 + cw)
            y1 = min(cap_h, y0 + ch)
            crop = frame[y0:y1, x0:x1]
            frame = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        else:
            # No zoom: if captured frame differs from output size, downscale/upscale to out size
            if (cap_w, cap_h) != (out_w, out_h):
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        # Convert to BGR for OpenCV, then to JPEG
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue
        buf = jpg.tobytes()
        # Yield as multipart/x-mixed-replace
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n"
               b"Content-Length: " + str(len(buf)).encode() + b"\r\n\r\n" +
               buf + b"\r\n")

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.post("/apply_camera_settings")
def apply_camera_settings():
    size = request.form.get("size", "1280x720")
    fps = int(request.form.get("fps", "30"))
    try:
        w, h = map(int, size.lower().split("x"))
    except Exception:
        w, h = 1280, 720
    # Read capture-size option (if present). If 'same', capture at the stream size.
    capture_opt = request.form.get("capture", "same")
    if capture_opt == "same":
        new_capture_w, new_capture_h = w, h
    elif capture_opt == "native":
        new_capture_w, new_capture_h = DEFAULT_NATIVE_CAPTURE
    else:
        try:
            new_capture_w, new_capture_h = map(int, capture_opt.lower().split("x"))
        except Exception:
            new_capture_w, new_capture_h = w, h

    # Reconfigure camera on the fly. We configure the camera to capture at
    # `new_capture_w,new_capture_h` but will stream (downscale) to w,h.
    picam2.stop()
    new_cfg = picam2.create_video_configuration(
        main={"size": (new_capture_w, new_capture_h)},
        controls={"FrameRate": fps}
    )
    picam2.configure(new_cfg)
    picam2.start()
    # update global stream/capture state so mjpeg_generator knows to downscale
    with config_lock:
        stream_w, stream_h = w, h
        capture_w, capture_h = new_capture_w, new_capture_h

    # Small delay to avoid a blank stream right after restart
    time.sleep(0.1)
    return redirect(url_for("index"))


@app.post("/set_zoom")
def set_zoom():
    """Set digital zoom factor. Accepts numeric zoom >= 1.0. '1' means no zoom (max out).
    The UI posts a form field named 'zoom'.
    """
    global zoom_factor
    val = request.form.get("zoom", "1").strip().lower()
    try:
        z = float(val)
    except Exception:
        # if someone posts 'max' or invalid, treat as no-zoom
        z = 1.0
    # clamp sensible bounds
    if z < 1.0:
        z = 1.0
    if z > 8.0:
        z = 8.0
    with zoom_lock:
        zoom_factor = z
    print(f"[camera] Set digital zoom to {zoom_factor}")
    return redirect(url_for("index"))

# Placeholder endpoint to hook your projector logic later
@app.post("/project")
def project_placeholder():
    image_path = request.form.get("image_path", "").strip()
    # For now, just acknowledge; we’ll wire this to fullscreen projector code.
    print(f"[project] Requested to show image: {image_path}")
    return redirect(url_for("index"))

if __name__ == "__main__":
    # Listen on all interfaces so you can open it from your laptop/phone
    # e.g. http://pi-fringe.local:5000/ or http://<Pi_IP>:5000/
    app.run(host="0.0.0.0", port=5000, threaded=True)
