# SentrySpike Wildlife Camera

AI-powered motion camera system — detects, captures, and classifies wildlife using a neuromorphic Akida inference model.

## Overview

SentrySpike is a three-process application. Each process runs independently in its own Python virtual environment and communicates with the others via a shared SQLite database and a shared captures directory on disk.

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

Each process requires its own virtual environment. Open three separate terminal windows, each in the `SentrySpike/` directory.

### 1 — SentrySpike_Camera

**Linux**
```bash
python3 -m venv venv_Camera
source venv_Camera/bin/activate
pip install -r requirements_Camera.txt
python3 SentrySpike_Camera/camera_service.py
```

**Windows**
```bash
python -m venv venv_Camera
venv_Camera\Scripts\activate
pip install -r requirements_Camera.txt
python SentrySpike_Camera\camera_service.py
```

### 2 — SentrySpike_Inference

> ⚠️ Python 3.12 is recommended for this virtual environment. The inference dependencies are not compatible with newer Python releases.

**Linux**
```bash
python3.12 -m venv venv_Inference
source venv_Inference/bin/activate
pip install -r requirements_Inference.txt
python SentrySpike_Inference/inference_service.py
```

**Windows**
```bash
py -3.12 -m venv venv_Inference
venv_Inference\Scripts\activate
pip install -r requirements_Inference.txt
python SentrySpike_Inference\inference_service.py
```

### 3 — SentrySpike_Flask

**Linux**
```bash
python3 -m venv venv_Flask
source venv_Flask/bin/activate
pip install -r requirements_Flask.txt
python SentrySpike_Flask/web_service.py
```

**Windows**
```bash
python -m venv venv_Flask
venv_Flask\Scripts\activate
pip install -r requirements_Flask.txt
python SentrySpike_Flask\web_service.py
```

Once running, open a browser and navigate to **http://localhost:5000** to access the web UI.

## Configuration

All settings are controlled by a single shared `config.py` file in the root `SentrySpike/` directory. All three processes read from this file.

| Setting | Description |
|---------|-------------|
| `camera_index` | Index of the camera device to use (0 = first USB/integrated camera) |
| `cooldown_seconds` | Cooldown period (seconds) after a successful capture event before motion detection resumes |

### Live Tuning

> ⚠️ Stop the camera process before opening the `/tuning` page. Only one process can access the camera at a time — running both simultaneously will cause an error.

Motion detection parameters can be visualised and adjusted interactively at **http://localhost:5000/tuning**. Changes are reflected immediately — no restart required.

## Troubleshooting

| Symptom | Likely cause & fix |
|---------|--------------------|
| Inference fails to start | Wrong Python version in `venv_Inference`. Recreate the venv using Python 3.12 explicitly. |
| Camera error / device busy | Another process (e.g. the `/tuning` page) is already using the camera. Stop all other users first. |
| Web UI shows no events | Ensure the camera and inference processes are both running and writing to the shared SQLite database. |
| Wrong camera selected | Update `camera_index` in `config.py`. Index 0 is the first device; increment for additional cameras. |
| Port 5000 already in use | Another application is using port 5000. Edit `web_service.py` to change the port, or stop the conflicting process. |

## Quick-Start Checklist

- Clone the repository
- Open three separate terminal windows, each in the `SentrySpike/` directory
- Terminal 1 — create `venv_Camera`, install requirements, start `camera_service.py`
- Terminal 2 — create `venv_Inference` with Python 3.12, install requirements, start `inference_service.py`
- Terminal 3 — create `venv_Flask`, install requirements, start `web_service.py`
- Open http://localhost:5000 in a browser to verify the web UI is running
- (Optional) Stop the camera process and open `/tuning` to adjust detection parameters

## Raspberry Pi

Image-based deployment for Raspberry Pi is currently in development. Deployment scaffolding — systemd services, OTA updates, and first-boot WiFi configuration — lives in the `rpi/` directory.
