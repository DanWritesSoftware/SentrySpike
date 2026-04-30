import subprocess
import sys
import os
import threading
import signal
import time
from collections import deque

ROOT = os.path.dirname(os.path.abspath(__file__))

if sys.platform == "win32":
    VENV_PYTHON = os.path.join(ROOT, "venv", "Scripts", "python.exe")
else:
    VENV_PYTHON = os.path.join(ROOT, "venv", "bin", "python")

REQUIREMENTS = [
    "requirements_Camera.txt",
    "requirements_Inference.txt",
    "requirements_Flask.txt",
]

SERVICES = [
    ("Camera",    ["SentrySpike_Camera/camera_service.py"]),
    ("Inference", ["SentrySpike_Inference/inference_service.py"]),
    ("Flask",     ["-m", "SentrySpike_Flask.web_service"]),
]

SERVICE_COLORS = {
    "Camera":    "cyan",
    "Inference": "yellow",
    "Flask":     "green",
}

LOGO = (
    "\n"
    " ▄█████ ▄▄▄▄▄ ▄▄  ▄▄ ▄▄▄▄▄▄ ▄▄▄▄  ▄▄ ▄▄ ▄█████ ▄▄▄▄  ▄▄ ▄▄ ▄▄ ▄▄▄▄▄ \n"
    " ▀▀▀▄▄▄ ██▄▄  ███▄██   ██   ██▄█▄ ▀███▀ ▀▀▀▄▄▄ ██▄█▀ ██ ██▄█▀ ██▄▄  \n"
    " █████▀ ██▄▄▄ ██ ▀██   ██   ██ ██   █   █████▀ ██    ██ ██ ██ ██▄▄▄ \n"
)

HEADER_SIZE = 12
MAX_LOG_LINES = 200
REFRESH_HZ = 4

log_buffers = {label: deque(maxlen=MAX_LOG_LINES) for label, _ in SERVICES}
procs = {}
start_times = {}
_shutdown = threading.Event()


def _fmt_uptime(seconds):
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def _build_header():
    from rich.table import Table
    from rich.text import Text
    from rich.panel import Panel
    from rich.console import Group

    logo = Text(LOGO, style="bold cyan", no_wrap=True)

    grid = Table.grid(padding=(0, 3))
    grid.add_column(min_width=10)
    grid.add_column(min_width=20)
    grid.add_column(min_width=12)
    grid.add_column()

    for label, _ in SERVICES:
        color = SERVICE_COLORS[label]
        proc = procs.get(label)
        if proc is None:
            status, pid_str, uptime_str = "[grey50]not started[/]", "-", "-"
        elif proc.poll() is None:
            status = "[green]● running[/]"
            pid_str = str(proc.pid)
            uptime_str = _fmt_uptime(time.time() - start_times.get(label, time.time()))
        else:
            status = f"[red]✗ exited ({proc.returncode})[/]"
            pid_str = str(proc.pid)
            uptime_str = "-"

        grid.add_row(
            f"[{color} bold]{label}[/]",
            status,
            f"[dim]PID[/] {pid_str}",
            f"[dim]Up:[/] {uptime_str}",
        )

    url = Text("  http://localhost:5000", style="dim")
    return Panel(Group(logo, grid, url), border_style="dim", padding=(0, 1))


def _build_log_panel(label):
    from rich.panel import Panel
    from rich.text import Text

    color = SERVICE_COLORS[label]
    content = Text("\n".join(log_buffers[label]), no_wrap=True)
    return Panel(content, title=f"[{color} bold]{label}[/]", border_style=color, padding=0)


def _stream_output(proc, label):
    for line in proc.stdout:
        log_buffers[label].append(line.rstrip())
    if not _shutdown.is_set():
        log_buffers[label].append(f"[dim][process exited with code {proc.wait()}][/dim]")


def install():
    print("Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", os.path.join(ROOT, "venv")], check=True)

    pip = [VENV_PYTHON, "-m", "pip", "install", "--upgrade", "pip"]
    subprocess.run(pip, check=True)

    for req in REQUIREMENTS:
        path = os.path.join(ROOT, req)
        print(f"\nInstalling {req}...")
        subprocess.run([VENV_PYTHON, "-m", "pip", "install", "-r", path], check=True)

    os.makedirs(os.path.join(ROOT, "captures"), exist_ok=True)
    print("\nInstall complete. Run with: python SentrySpike.py run")


def run():
    if not os.path.exists(VENV_PYTHON):
        print("Virtual environment not found. Run: python SentrySpike.py install")
        sys.exit(1)

    # Re-exec with the venv Python if we're not already running from it,
    # so all installed dependencies (rich, waitress, etc.) are available.
    if os.path.abspath(sys.executable) != os.path.abspath(VENV_PYTHON):
        os.execv(VENV_PYTHON, [VENV_PYTHON, os.path.abspath(__file__)] + sys.argv[1:])

    from rich.live import Live
    from rich.layout import Layout

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=HEADER_SIZE),
        Layout(name="logs"),
    )
    layout["logs"].split_row(
        Layout(name="camera"),
        Layout(name="inference"),
        Layout(name="flask"),
    )

    for label, args in SERVICES:
        cmd = [VENV_PYTHON, "-u"] + args
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        procs[label] = proc
        start_times[label] = time.time()
        threading.Thread(target=_stream_output, args=(proc, label), daemon=True).start()

    def shutdown(sig=None, frame=None):
        _shutdown.set()
        for p in procs.values():
            p.terminate()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    with Live(layout, refresh_per_second=REFRESH_HZ, screen=True):
        while not _shutdown.is_set():
            layout["header"].update(_build_header())
            layout["camera"].update(_build_log_panel("Camera"))
            layout["inference"].update(_build_log_panel("Inference"))
            layout["flask"].update(_build_log_panel("Flask"))
            _shutdown.wait(timeout=1 / REFRESH_HZ)

    for p in procs.values():
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()


def update():
    # fetch first so the comparison is against the remote
    subprocess.run(["git", "fetch", "origin", "main"], cwd=ROOT, check=True)

    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "origin/main"],
        cwd=ROOT, capture_output=True, text=True, check=True
    )
    changed = result.stdout.splitlines()

    if not changed:
        print("Already up to date.")
        return

    print("Pulling latest changes...")
    subprocess.run(["git", "pull", "origin", "main"], cwd=ROOT, check=True)

    if any("requirements" in f for f in changed):
        print("\nDependencies changed — reinstalling...")
        for req in REQUIREMENTS:
            path = os.path.join(ROOT, req)
            print(f"Installing {req}...")
            subprocess.run([VENV_PYTHON, "-m", "pip", "install", "-r", path], check=True)

    print("\nUpdate complete. Run with: python SentrySpike.py run")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("install", "run", "update"):
        print("Usage: python SentrySpike.py [install|run|update]")
        sys.exit(1)

    if sys.argv[1] == "install":
        install()
    elif sys.argv[1] == "update":
        update()
    else:
        run()


if __name__ == "__main__":
    main()
