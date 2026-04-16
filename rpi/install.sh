#!/bin/bash
# SentrySpike RPi Install Script
#
# Run this once on a fresh Raspberry Pi OS (64-bit) image.
# Installs dependencies, sets up the Python venv, registers systemd services,
# and installs wifi-connect for first-boot WiFi configuration.
#
# Usage:
#   sudo bash install.sh

set -e

INSTALL_DIR=/opt/sentryspike
REPO_URL=https://github.com/DanWritesSoftware/SentrySpike.git
SERVICE_USER=pi
WIFI_CONNECT_VERSION=0.11.0  # Check https://github.com/balena-os/wifi-connect/releases

echo "==> Installing system dependencies..."
apt-get update -q
apt-get install -y -q \
    python3 python3-pip python3-venv \
    git \
    libcap2-bin \
    network-manager \
    dnsmasq-base

echo "==> Cloning SentrySpike..."
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "    Already cloned. Pulling latest..."
    git -C "$INSTALL_DIR" pull origin main
else
    git clone "$REPO_URL" "$INSTALL_DIR"
fi

chown -R "$SERVICE_USER":"$SERVICE_USER" "$INSTALL_DIR"

echo "==> Creating Python virtual environment..."
sudo -u "$SERVICE_USER" python3 -m venv "$INSTALL_DIR/venv"

echo "==> Installing Python dependencies..."
sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install --upgrade pip -q
sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements_Camera.txt" -q
sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements_Flask.txt" -q
sudo -u "$SERVICE_USER" "$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements_Inference.txt" -q

echo "==> Installing wifi-connect..."
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    WC_ARCH="aarch64"
elif [ "$ARCH" = "armv7l" ]; then
    WC_ARCH="armv7hf"
else
    echo "WARNING: Unknown architecture $ARCH, skipping wifi-connect install."
    WC_ARCH=""
fi

if [ -n "$WC_ARCH" ]; then
    WC_URL="https://github.com/balena-os/wifi-connect/releases/download/v${WIFI_CONNECT_VERSION}/wifi-connect-${WC_ARCH}.tar.gz"
    curl -sSL "$WC_URL" | tar -xz -C /usr/local/bin wifi-connect
    chmod +x /usr/local/bin/wifi-connect
fi

echo "==> Making scripts executable..."
chmod +x "$INSTALL_DIR/rpi/scripts/update.sh"
chmod +x "$INSTALL_DIR/rpi/scripts/firstboot.sh"
chmod +x "$INSTALL_DIR/rpi/scripts/setup.sh"

echo "==> Installing systemd services..."
cp "$INSTALL_DIR/rpi/services/"*.service /etc/systemd/system/
cp "$INSTALL_DIR/rpi/services/"*.timer /etc/systemd/system/

systemctl daemon-reload

systemctl enable sentryspike-firstboot.service
systemctl enable sentryspike-setup.service
systemctl enable sentryspike-camera.service
systemctl enable sentryspike-inference.service
systemctl enable sentryspike-flask.service
systemctl enable sentryspike-update.timer

echo "==> Creating config directories..."
mkdir -p /etc/sentryspike
mkdir -p "$INSTALL_DIR/captures"
chown "$SERVICE_USER":"$SERVICE_USER" "$INSTALL_DIR/captures"

echo ""
echo "Install complete. Reboot to start SentrySpike."
echo "On first boot, connect to the 'SentrySpike-Setup' WiFi network to configure."
