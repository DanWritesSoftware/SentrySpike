#!/bin/bash
# SentrySpike first-boot / WiFi setup script
#
# On first boot (or when WiFi is not configured), this script:
#   1. Starts a WiFi access point named "SentrySpike-Setup"
#   2. Launches wifi-connect, which hosts a captive portal for network selection
#   3. After the user connects and enters credentials, wifi-connect reboots the device
#      into normal client mode, and this script is skipped on subsequent boots.
#
# wifi-connect (by Balena) is used for the captive portal:
#   https://github.com/balena-os/wifi-connect
# Install it via: install.sh

CONFIGURED_FLAG=/etc/sentryspike/.configured
WIFI_CONNECT=/usr/local/bin/wifi-connect
LOG=/var/log/sentryspike-firstboot.log

echo "[$(date)] Running first-boot check..." >> "$LOG"

# Skip if already configured
if [ -f "$CONFIGURED_FLAG" ]; then
    echo "[$(date)] Already configured. Skipping." >> "$LOG"
    exit 0
fi

# Check if already connected to a network
if nmcli -t -f STATE general | grep -q "connected"; then
    echo "[$(date)] Network already connected. Marking as configured." >> "$LOG"
    touch "$CONFIGURED_FLAG"
    exit 0
fi

echo "[$(date)] No network found. Starting captive portal..." >> "$LOG"

# Launch wifi-connect — this blocks until the user configures WiFi,
# then reboots automatically.
"$WIFI_CONNECT" \
    --portal-ssid "SentrySpike-Setup" \
    --portal-interface uap0 \
    --activity-timeout 300 \
    >> "$LOG" 2>&1

# If wifi-connect exits without rebooting (e.g. timeout), mark configured
# so we don't loop. The user can re-trigger setup manually if needed.
touch "$CONFIGURED_FLAG"
