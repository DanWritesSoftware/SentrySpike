# SentrySpike Wildlife Camera

AI-powered motion camera system — detects, captures, and classifies wildlife using a neuromorphic Akida inference model.

## Overview

SentrySpike is a three-process application. All three processes share a single virtual environment and communicate via a shared SQLite database and a shared captures directory on disk.

| Process | Responsibility | Entry Point |
|---------|---------------|-------------|
| SentrySpike_Camera | Motion detection, frame capture, event creation | `SentrySpike_Camera/camera_service.py` |
| SentrySpike_Inference | Akida-based gate & species inference on captured events | `SentrySpike_Inference/inference_service.py` |
| SentrySpike_Flask | Web UI — event timeline, live tuning, system info | `SentrySpike_Flask/web_service.py` |

**Camera hardware:** The application uses the first available USB webcam, or the integrated laptop camera if none is connected. See the Configuration section to change this.

## Prerequisites

Clone the repository before starting any of the steps below:

```bash
git clone https://github.com/DanWritesSoftware/SentrySpike.git
cd SentrySpike
```

| Process | Python Version | Notes |
|---------|---------------|-------|
| SentrySpike_Camera | 3.9+ (latest OK) | No special constraints |
| SentrySpike_Inference | 3.12 recommended | Newer versions may break inference dependencies |
| SentrySpike_Flask | 3.9+ (latest OK) | No special constraints |

## Setup

> ⚠️ Python 3.12 is recommended. The inference dependencies are not compatible with newer Python releases.

Install all dependencies and start all three services with two commands:

```bash
python SentrySpike.py install
python SentrySpike.py run
```

`install` creates a `venv/` directory and installs all requirements into it. `run` starts all three services in parallel and prefixes their output with color-coded `[Camera]`, `[Inference]`, and `[Flask]` labels. Press **Ctrl+C** to stop everything.

Once running, open a browser and navigate to **http://localhost:5000** to access the web UI.

### Manual Setup (advanced)

If you need to run processes individually, each service can still be started directly using the shared venv:

**Linux**
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements_Camera.txt -r requirements_Inference.txt -r requirements_Flask.txt

# In separate terminals:
python SentrySpike_Camera/camera_service.py
python SentrySpike_Inference/inference_service.py
python -m flask --app SentrySpike_Flask.web_service run
```

**Windows**
```bash
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements_Camera.txt -r requirements_Inference.txt -r requirements_Flask.txt

# In separate terminals:
python SentrySpike_Camera\camera_service.py
python SentrySpike_Inference\inference_service.py
python -m flask --app SentrySpike_Flask.web_service run
```

## Configuration

All settings are controlled by a single shared `config.py` file in the root `SentrySpike/` directory. All three processes read from this file.

| Setting | Description |
|---------|-------------|
| `camera_index` | Index of the camera device to use (0 = first USB/integrated camera) |
| `cooldown_seconds` | Cooldown period (seconds) after a successful capture event before motion detection resumes |

### Live Tuning

Motion detection parameters can be visualised and adjusted interactively at **http://localhost:5000/tuning**. While the tuning page is open the camera service pauses automatically; it resumes as soon as you navigate away. Changes are reflected immediately — no restart required.

## Troubleshooting

| Symptom | Likely cause & fix |
|---------|--------------------|
| `akida` fails to install | Python version is too new. Run `python3.12 SentrySpike.py install` (Linux) or `py -3.12 SentrySpike.py install` (Windows). |
| Inference fails to start | Wrong Python version used for `install`. Delete `venv/` and rerun with Python 3.12 as above. |
| Camera error / device busy | Another process (e.g. the `/tuning` page) is already using the camera. Stop all other users first. |
| Web UI shows no events | Ensure the camera and inference processes are both running and writing to the shared SQLite database. |
| Wrong camera selected | Update `camera_index` in `config.py`. Index 0 is the first device; increment for additional cameras. |
| Port 5000 already in use | Another application is using port 5000. Edit `web_service.py` to change the port, or stop the conflicting process. |

## Updating

Pull the latest code and reinstall any changed dependencies:

```bash
python SentrySpike.py update
```

## Quick-Start Checklist

- Clone the repository and `cd SentrySpike`
- Run `python SentrySpike.py install` (requires Python 3.12)
- Run `python SentrySpike.py run`
- Open http://localhost:5000 in a browser to verify the web UI is running
- (Optional) Stop the camera process and open `/tuning` to adjust detection parameters

## Raspberry Pi

Image-based deployment for Raspberry Pi is currently in development. Deployment scaffolding — systemd services, OTA updates, and first-boot WiFi configuration — lives in the `rpi/` directory.
