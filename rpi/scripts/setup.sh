#!/bin/bash
# SentrySpike First-Boot Setup Script
#
# Runs once on first boot to:
#   1. Build and install the Akida PCIe kernel module
#   2. Install Akida Python package system-wide
#   3. Create the SentrySpike virtual environment and install dependencies
#   4. Enable and start all SentrySpike systemd services
#
# Triggered by sentryspike-setup.service on first boot.
# Creates /etc/sentryspike/.setup_complete on success so it never runs again.

INSTALL_DIR=/opt/sentryspike
DRIVER_DIR=$INSTALL_DIR/rpi/akida_dw_edma
FLAG=/etc/sentryspike/.setup_complete
LOG=/var/log/sentryspike-setup.log
SERVICE_USER=ubuntu  # matches the default user on the BrainChip base image

exec >> "$LOG" 2>&1

echo "[$(date)] ===== SentrySpike setup started ====="

if [ -f "$FLAG" ]; then
    echo "[$(date)] Already set up. Exiting."
    exit 0
fi

mkdir -p /etc/sentryspike
mkdir -p "$INSTALL_DIR/captures"

# ── 1. Akida PCIe kernel module ────────────────────────────────
echo "[$(date)] Installing kernel headers..."
if apt-get install -y build-essential linux-headers-$(uname -r); then
    echo "[$(date)] Kernel headers installed."
else
    echo "[$(date)] ERROR: Failed to install kernel headers. Is network available?"
fi

echo "[$(date)] Building Akida PCIe driver..."
if cd "$DRIVER_DIR" && make clean 2>/dev/null; then true; fi
if cd "$DRIVER_DIR" && ./install.sh; then
    echo "[$(date)] Akida driver installed."
else
    echo "[$(date)] ERROR: Akida driver build failed. Check kernel headers and PCIe card."
fi

echo "[$(date)] Verifying Akida device..."
if lspci | grep -q "Co-processor"; then
    echo "[$(date)] Akida PCIe device detected."
else
    echo "[$(date)] WARNING: Akida PCIe device not detected. Is the card seated correctly?"
fi

# ── 2. Akida Python package ────────────────────────────────────
echo "[$(date)] Installing Akida Python package..."
if pip3 install akida==2.18.2 numpy; then
    echo "[$(date)] Akida Python package installed."
else
    echo "[$(date)] ERROR: Failed to install akida Python package."
fi

# ── 3. SentrySpike virtual environment ────────────────────────
echo "[$(date)] Creating virtual environment..."
if python3 -m venv --system-site-packages "$INSTALL_DIR/venv"; then
    echo "[$(date)] Virtual environment created."
else
    echo "[$(date)] ERROR: Failed to create virtual environment."
fi

echo "[$(date)] Installing SentrySpike dependencies..."
if "$INSTALL_DIR/venv/bin/pip" install --upgrade pip -q \
    && "$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements_Camera.txt" -q \
    && "$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements_Flask.txt" -q \
    && "$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements_Inference.txt" -q; then
    echo "[$(date)] SentrySpike dependencies installed."
else
    echo "[$(date)] ERROR: Failed to install SentrySpike dependencies."
fi

chown -R "$SERVICE_USER":"$SERVICE_USER" "$INSTALL_DIR/venv"
chown -R "$SERVICE_USER":"$SERVICE_USER" "$INSTALL_DIR/captures"

# ── 4. Enable and start services ──────────────────────────────
echo "[$(date)] Enabling SentrySpike services..."
systemctl daemon-reload
systemctl enable sentryspike-camera.service
systemctl enable sentryspike-inference.service
systemctl enable sentryspike-flask.service
systemctl enable sentryspike-update.timer

echo "[$(date)] Starting SentrySpike services..."
systemctl start sentryspike-camera.service || echo "[$(date)] WARNING: camera service failed to start."
systemctl start sentryspike-inference.service || echo "[$(date)] WARNING: inference service failed to start."
systemctl start sentryspike-flask.service || echo "[$(date)] WARNING: flask service failed to start."

# ── Done ──────────────────────────────────────────────────────
touch "$FLAG"
echo "[$(date)] ===== SentrySpike setup complete ====="
