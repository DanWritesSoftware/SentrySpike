import subprocess
import sys
import os
import threading
import signal

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
    ("Flask",     ["-m", "flask", "--app", "SentrySpike_Flask.web_service", "run"]),
]

COLORS = {
    "Camera":    "\033[36m",   # cyan
    "Inference": "\033[33m",   # yellow
    "Flask":     "\033[32m",   # green
    "reset":     "\033[0m",
}


def install():
    print("Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", os.path.join(ROOT, "venv")], check=True)

    pip = [VENV_PYTHON, "-m", "pip", "install", "--upgrade", "pip"]
    subprocess.run(pip, check=True)

    for req in REQUIREMENTS:
        path = os.path.join(ROOT, req)
        print(f"\nInstalling {req}...")
        subprocess.run([VENV_PYTHON, "-m", "pip", "install", "-r", path], check=True)

    print("\nInstall complete. Run with: python SentrySpike.py run")


def stream_output(proc, label):
    color = COLORS.get(label, "")
    reset = COLORS["reset"]
    for line in proc.stdout:
        print(f"{color}[{label}]{reset} {line}", end="")


def run():
    if not os.path.exists(VENV_PYTHON):
        print("Virtual environment not found. Run: python SentrySpike.py install")
        sys.exit(1)

    procs = []
    threads = []

    for label, args in SERVICES:
        cmd = [VENV_PYTHON] + args
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        procs.append(proc)
        t = threading.Thread(target=stream_output, args=(proc, label), daemon=True)
        t.start()
        threads.append(t)
        print(f"Started {label} (pid {proc.pid})")

    def shutdown(sig=None, frame=None):
        print("\nShutting down...")
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    for proc in procs:
        proc.wait()


def update():
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "origin/main"],
        cwd=ROOT, capture_output=True, text=True
    )
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
