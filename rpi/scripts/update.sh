#!/bin/bash
# SentrySpike OTA update script
# Checks the public GitHub repo for new commits on main.
# If updates are found, pulls them, refreshes dependencies, and reboots.

set -e

INSTALL_DIR=/opt/sentryspike
REPO_URL=https://github.com/DanWritesSoftware/SentrySpike.git
LOG=/var/log/sentryspike-update.log

echo "[$(date)] Checking for updates..." >> "$LOG"

cd "$INSTALL_DIR"

# Abort if no network
if ! ping -c 1 github.com &>/dev/null; then
    echo "[$(date)] No network, skipping update check." >> "$LOG"
    exit 0
fi

git fetch origin main 2>> "$LOG"

LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [ "$LOCAL" = "$REMOTE" ]; then
    echo "[$(date)] Already up to date ($LOCAL)." >> "$LOG"
    exit 0
fi

echo "[$(date)] Update found: $LOCAL -> $REMOTE. Pulling..." >> "$LOG"

git pull origin main >> "$LOG" 2>&1

# Reinstall dependencies only if requirements files changed
if git diff "$LOCAL" "$REMOTE" --name-only | grep -q "requirements"; then
    echo "[$(date)] Requirements changed, reinstalling..." >> "$LOG"
    venv/bin/pip install \
        -r requirements_Camera.txt \
        -r requirements_Inference.txt \
        -r requirements_Flask.txt \
        -q >> "$LOG" 2>&1
fi

echo "[$(date)] Update complete. Rebooting." >> "$LOG"
systemctl reboot
