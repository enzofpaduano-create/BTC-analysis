#!/usr/bin/env bash
# Uninstall the btc-quant alerts launchd agent.
set -euo pipefail

PLIST_NAME="com.btc-quant.alerts.plist"
TARGET="${HOME}/Library/LaunchAgents/${PLIST_NAME}"
LABEL="com.btc-quant.alerts"

if launchctl list | grep -q "${LABEL}"; then
  launchctl unload "${TARGET}" || true
  echo "✓ Unloaded ${LABEL}"
fi

if [ -f "${TARGET}" ]; then
  rm -f "${TARGET}"
  echo "✓ Removed ${TARGET}"
fi

echo "Done."
