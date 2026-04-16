import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
import socket
import subprocess
from flask import (Flask, render_template, send_from_directory, redirect,
                   url_for, g, abort, jsonify, request, Response)
from datetime import datetime
from PIL import Image, ImageDraw
import json
import threading
import time
import types
import database as db
from config import Config
from SentrySpike_Camera.motion_detection import BgSubMotionDetector

CFG = Config()

# Resolve captures directory to an absolute path relative to the project root,
# so send_from_directory works regardless of where flask run is invoked from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAPTURES_DIR = os.path.join(PROJECT_ROOT, CFG.save_directory)

app = Flask(__name__)
db.init_db()


'''
Database Connection Management
'''

def get_db():
    '''
    Return a per-request DB connection, opening it on first access.

    Stored in flask.g so it's shared within a request and automatically
    closed on teardown by close_db().
    '''
    if 'db' not in g:
        g.db = db.get_connection()
    return g.db

@app.teardown_appcontext
def close_db(exc):
    '''Close the per-request DB connection if one was opened.'''
    conn = g.pop('db', None)
    if conn is not None:
        conn.close()


'''
Jinja2 Filters
'''

@app.template_filter('format_ts')
def format_ts(iso_str):
    '''
    Format an ISO timestamp string for human-readable display.

    "2026-03-19T14:49:47.557589"  ->  "Mar 19, 2026 · 2:49 PM"
    Returns the raw string unchanged if parsing fails.
    '''
    if not iso_str:
        return ''
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime('%-d %b %Y · %-I:%M %p')
    except ValueError:
        return iso_str


'''
Motion Tuning — live feed with adjustable parameters
'''

class _TuningState:
    '''
    Shared mutable state between the MJPEG stream generator thread and the
    parameter-update request threads.  All access is protected by a lock.
    '''
    def __init__(self):
        self._lock = threading.Lock()
        self._params = {
            'bgsub_var_threshold':   getattr(CFG, 'bgsub_var_threshold', 16),
            'motion_min_area':       getattr(CFG, 'motion_min_area', 80),
            'motion_confirm_frames': CFG.motion_confirm_frames,
            'bgsub_open_iters':      getattr(CFG, 'bgsub_open_iters', 2),
            'bgsub_dilate_iters':    getattr(CFG, 'bgsub_dilate_iters', 2),
            'bgsub_warmup_frames':   getattr(CFG, 'bgsub_warmup_frames', 30),
        }
        self._view = 'bbox_vis'
        self._reset = False
        self.status = {'motion': False, 'triggers': 0, 'fps': 0.0,
                       'warmup': True, 'warmup_progress': '0/30'}

    def get_params(self):
        with self._lock:
            return dict(self._params)

    def get_view(self):
        with self._lock:
            return self._view

    def update(self, params, view=None):
        with self._lock:
            self._params.update(params)
            if view is not None:
                self._view = view
            self._reset = True

    def check_reset(self):
        '''Return True (and clear flag) if the detector should be recreated.'''
        with self._lock:
            if self._reset:
                self._reset = False
                return True
            return False

    def set_status(self, **kwargs):
        with self._lock:
            self.status.update(kwargs)

    def get_status(self):
        with self._lock:
            return dict(self.status)


_tuning = _TuningState()


def _make_tuning_detector(params):
    cfg_ns = types.SimpleNamespace(
        bgsub_method              = getattr(CFG, 'bgsub_method', 'MOG2'),
        bgsub_history             = getattr(CFG, 'bgsub_history', 500),
        bgsub_detect_shadows      = getattr(CFG, 'bgsub_detect_shadows', False),
        bgsub_bin_thresh          = getattr(CFG, 'bgsub_bin_thresh', 200),
        bgsub_freeze_during_event = getattr(CFG, 'bgsub_freeze_during_event', True),
        bgsub_dist2_threshold     = getattr(CFG, 'bgsub_dist2_threshold', 400.0),
        motion_blur_ksize         = getattr(CFG, 'motion_blur_ksize', 5),
        **params
    )
    return BgSubMotionDetector(cfg_ns)


TUNING_FLAG = os.path.join(PROJECT_ROOT, '.tuning_active')


def _tuning_generate():
    '''
    MJPEG frame generator for /tuning/stream.
    Opens the camera, runs the motion detector on each frame, overlays status
    text, and yields JPEG-encoded frames.  The camera is released automatically
    when the client disconnects (GeneratorExit -> finally block).
    '''
    import cv2
    import numpy as np

    open(TUNING_FLAG, 'w').close()
    time.sleep(0.5)  # give camera_service time to release the camera

    cap = cv2.VideoCapture(CFG.camera_index)
    if not cap.isOpened():
        err = np.zeros((240, 480, 3), dtype=np.uint8)
        cv2.putText(err, 'Camera unavailable', (70, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 200), 2)
        cv2.putText(err, 'Is camera_service.py running?', (50, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1)
        _, buf = cv2.imencode('.jpg', err)
        frame_bytes = buf.tobytes()
        while True:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + frame_bytes + b'\r\n')
            time.sleep(1.0)
        return

    params = _tuning.get_params()
    detector = _make_tuning_detector(params)
    confirm_frames = params['motion_confirm_frames']
    consecutive = 0
    triggers = 0
    fps_smooth = 0.0
    t_last = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            if _tuning.check_reset():
                params = _tuning.get_params()
                detector = _make_tuning_detector(params)
                confirm_frames = params['motion_confirm_frames']
                consecutive = 0
                triggers = 0

            bbox, dbg = detector.detect(frame, event_active=consecutive > 0)
            in_warmup = detector.frame_count <= detector.warmup_frames

            if bbox is not None:
                consecutive += 1
            else:
                consecutive = 0

            if consecutive >= confirm_frames:
                triggers += 1
                consecutive = 0

            now = time.time()
            fps_smooth = 0.85 * fps_smooth + 0.15 * (1.0 / max(now - t_last, 1e-3))
            t_last = now

            _tuning.set_status(
                motion=bbox is not None,
                triggers=triggers,
                fps=fps_smooth,
                warmup=in_warmup,
                warmup_progress=(
                    f"{min(detector.frame_count, detector.warmup_frames)}"
                    f"/{detector.warmup_frames}"
                ),
            )

            view = _tuning.get_view()
            display = dbg.get(view, frame)
            if display is None:
                display = frame
            if len(display.shape) == 2:
                display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
            else:
                display = display.copy()

            h, w = display.shape[:2]
            if in_warmup:
                label = (f"Warmup  "
                         f"{min(detector.frame_count, detector.warmup_frames)}"
                         f"/{detector.warmup_frames}")
                fg = (0, 210, 210)
            elif bbox is not None:
                label = f"MOTION  (triggers: {triggers})"
                fg = (0, 220, 60)
            else:
                label = f"Idle  (triggers: {triggers})"
                fg = (110, 110, 110)

            for col, thick in [((0, 0, 0), 3), (fg, 1)]:
                cv2.putText(display, label, (10, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, thick)
                cv2.putText(display, f"{fps_smooth:.1f} fps", (w - 95, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, thick)

            _, buf = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 75])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + buf.tobytes() + b'\r\n')

    finally:
        cap.release()
        try:
            os.remove(TUNING_FLAG)
        except OSError:
            pass


'''
Routes
'''

@app.route('/')
def event_list():
    '''
    Main page — all events newest first, rendered as a card grid.
    Accepts an optional ?category= query param to filter by top_prediction.
    '''
    category = request.args.get('category', '').strip() or None

    rows = db.get_all_events(category=category)
    categories = db.get_distinct_predictions()

    # Convert sqlite3.Row objects to plain dicts and attach a URL-ready
    # thumbnail path relative to CAPTURES_DIR (e.g. "20260319_145428/frame_07.jpg")
    # so the template can pass it straight to url_for('captures', filename=...).
    events = []
    for row in rows:
        ev = dict(row)
        if ev['thumbnail_path']:
            ev['thumbnail_rel'] = os.path.relpath(ev['thumbnail_path'], CAPTURES_DIR).replace('\\', '/')
        else:
            ev['thumbnail_rel'] = None
        events.append(ev)

    return render_template('event_list.html', events=events,
                           categories=categories, active_category=category)


@app.route('/event/<string:event_id>')
def event_detail(event_id):
    '''
    Event detail view — shows the burst GIF and inference results.

    Note: event_id is a timestamp string (e.g. "20260319_145428_670900"),
    not an integer, so this route uses <string:event_id>.
    '''
    row = db.get_event(event_id)
    if row is None:
        abort(404)

    event = dict(row)

    db.mark_reviewed(event_id)

    # Parse stability / agreement details stored by the inference service
    if event.get('predictions_json'):
        event['predictions'] = json.loads(event['predictions_json'])
    else:
        event['predictions'] = None

    return render_template('event_detail.html', event=event)


@app.route('/event/<string:event_id>/gif')
def event_gif(event_id):
    '''
    Return an animated GIF of the full burst, generating and caching it on first request.

    The GIF is cached alongside the source frames at captures/{event_id}/burst.gif
    so it is only built once per event.  Frame duration is taken from
    CFG.burst_sleep_seconds so playback speed matches the original capture rate.
    '''
    gif_path = os.path.join(CAPTURES_DIR, event_id, 'burst.gif')

    if not os.path.exists(gif_path):
        frame_rows = db.get_event_frames(event_id)
        if not frame_rows:
            abort(404)

        # ROI is identical across all frames in a burst; grab it from the first
        # frame that has one so we can overlay it on every GIF frame.
        roi_bbox = None
        for row in frame_rows:
            if row['roi_json']:
                roi_data = json.loads(row['roi_json'])
                if roi_data.get('roi_bbox'):
                    roi_bbox = tuple(roi_data['roi_bbox'])   # (x, y, w, h)
                    break

        pil_frames = []
        for row in frame_rows:
            try:
                img = Image.open(row['image_path']).convert('RGB')
                # Resize to max 640 px wide, preserving aspect ratio.
                # Track the scale factor so the ROI coordinates can be mapped.
                w_orig, h_orig = img.size
                if w_orig > 640:
                    scale = 640 / w_orig
                    img = img.resize((640, int(h_orig * scale)), Image.LANCZOS)
                else:
                    scale = 1.0

                if roi_bbox is not None:
                    x, y, bw, bh = roi_bbox
                    draw = ImageDraw.Draw(img)
                    draw.rectangle(
                        [int(x * scale), int(y * scale),
                         int((x + bw) * scale), int((y + bh) * scale)],
                        outline='#00ff88',
                        width=2,
                    )

                pil_frames.append(img)
            except Exception:
                pass        # skip any unreadable frames

        if not pil_frames:
            abort(404)

        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            loop=0,
            duration=int(CFG.burst_sleep_seconds * 1000),
        )

    return send_from_directory(
        os.path.join(CAPTURES_DIR, event_id),
        'burst.gif',
        mimetype='image/gif'
    )


@app.route('/event/<string:event_id>/delete', methods=['POST'])
def event_delete(event_id):
    '''
    Delete an event: removes DB records and the captures directory from disk,
    then redirects back to the event list.
    '''
    row = db.get_event(event_id)
    if row is None:
        abort(404)

    db.delete_event_with_files(event_id)

    return redirect(url_for('event_list'))


@app.route('/tuning')
def tuning():
    '''Tuning page — live camera feed with adjustable motion detection parameters.'''
    return render_template('tuning.html', params=_tuning.get_params())


@app.route('/tuning/stream')
def tuning_stream():
    '''MJPEG stream endpoint consumed by the tuning page img tag.'''
    return Response(_tuning_generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/tuning/params', methods=['POST'])
def tuning_params():
    '''Accept updated parameter values from the tuning page sliders.'''
    data = request.get_json(force=True)
    int_keys = ['bgsub_var_threshold', 'motion_min_area', 'motion_confirm_frames',
                'bgsub_open_iters', 'bgsub_dilate_iters', 'bgsub_warmup_frames']
    params = {k: int(data[k]) for k in int_keys if k in data}
    _tuning.update(params, view=data.get('view'))
    return jsonify({'ok': True})


@app.route('/tuning/status')
def tuning_status():
    '''Return current detector status as JSON for the polling status panel.'''
    return jsonify(_tuning.get_status())


@app.route('/captures/<path:filename>')
def captures(filename):
    '''
    Serve captured images from the configured captures directory.

    Images live outside the Flask static folder so this dedicated route
    hands them off via send_from_directory.
    '''
    return send_from_directory(CAPTURES_DIR, filename)


'''
System Info Page
'''

def _fmt_bytes(n):
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024 or unit == 'TB':
            return f'{n:.1f} {unit}'
        n /= 1024


def _get_system_info():
    info = {}

    info['hostname'] = socket.gethostname()

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        info['ip'] = s.getsockname()[0]
        s.close()
    except Exception:
        info['ip'] = 'unavailable'

    try:
        with open('/proc/uptime') as f:
            total_seconds = int(float(f.read().split()[0]))
        days, rem = divmod(total_seconds, 86400)
        hours, rem = divmod(rem, 3600)
        minutes = rem // 60
        parts = []
        if days:
            parts.append(f'{days}d')
        if hours:
            parts.append(f'{hours}h')
        parts.append(f'{minutes}m')
        info['uptime'] = ' '.join(parts)
    except Exception:
        info['uptime'] = None

    try:
        mem = {}
        with open('/proc/meminfo') as f:
            for line in f:
                k, v = line.split(':')
                mem[k.strip()] = int(v.split()[0])
        total = mem['MemTotal'] * 1024
        used = (mem['MemTotal'] - mem['MemAvailable']) * 1024
        info['mem_used'] = _fmt_bytes(used)
        info['mem_total'] = _fmt_bytes(total)
        info['mem_pct'] = round(used / total * 100)
    except Exception:
        info['mem_used'] = info['mem_total'] = info['mem_pct'] = None

    try:
        usage = shutil.disk_usage(PROJECT_ROOT)
        info['disk_used'] = _fmt_bytes(usage.used)
        info['disk_total'] = _fmt_bytes(usage.total)
        info['disk_pct'] = round(usage.used / usage.total * 100)
    except Exception:
        info['disk_used'] = info['disk_total'] = info['disk_pct'] = None

    try:
        with open('/sys/class/thermal/thermal_zone0/temp') as f:
            info['cpu_temp'] = f'{int(f.read().strip()) / 1000:.1f} °C'
    except Exception:
        info['cpu_temp'] = None

    service_names = {
        'sentryspike-camera':    'Camera',
        'sentryspike-inference': 'Inference',
        'sentryspike-flask':     'Web UI',
    }
    info['services'] = {}
    for svc, label in service_names.items():
        try:
            result = subprocess.run(
                ['systemctl', 'is-active', f'{svc}.service'],
                capture_output=True, text=True, timeout=2
            )
            status = result.stdout.strip()
        except Exception:
            status = 'unknown'
        info['services'][label] = status

    try:
        result = subprocess.run(
            ['git', '-C', PROJECT_ROOT, 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=2
        )
        info['version'] = result.stdout.strip() or 'unknown'
    except Exception:
        info['version'] = 'unknown'

    captures_size = 0
    for dirpath, _, filenames in os.walk(CAPTURES_DIR):
        for fname in filenames:
            try:
                captures_size += os.path.getsize(os.path.join(dirpath, fname))
            except OSError:
                pass
    info['captures_size'] = _fmt_bytes(captures_size)

    try:
        db_path = os.path.join(PROJECT_ROOT, CFG.database_path)
        info['db_size'] = _fmt_bytes(os.path.getsize(db_path))
    except OSError:
        info['db_size'] = 'unavailable'

    return info


@app.route('/info')
def info_page():
    '''System info page — device status, service health, and database stats.'''
    system = _get_system_info()

    conn = get_db()
    total_events  = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    complete      = conn.execute("SELECT COUNT(*) FROM events WHERE status='complete'").fetchone()[0]
    pending       = conn.execute("SELECT COUNT(*) FROM events WHERE status='awaiting_inference'").fetchone()[0]
    total_frames  = conn.execute("SELECT COUNT(*) FROM event_frames").fetchone()[0]

    db_stats = {
        'total_events':  total_events,
        'complete':      complete,
        'pending':       pending,
        'total_frames':  total_frames,
    }

    return render_template('info.html', system=system, db_stats=db_stats)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
