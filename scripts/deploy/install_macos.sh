#!/usr/bin/env bash
# Install the btc-quant alerts runner as a launchd agent.
#
# Once installed, the runner :
#   - starts at login (and right now)
#   - restarts on crash
#   - keeps the Mac awake via `caffeinate -i`
#
# Logs go to logs/alerts.{out,err}.log inside the project.
#
# Usage::
#
#     bash scripts/deploy/install_macos.sh
#
# Uninstall::
#
#     bash scripts/deploy/uninstall_macos.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
PLIST_NAME="com.btc-quant.alerts.plist"
TEMPLATE="${PROJECT_DIR}/scripts/deploy/${PLIST_NAME}.template"
TARGET="${HOME}/Library/LaunchAgents/${PLIST_NAME}"
LABEL="com.btc-quant.alerts"

if [ ! -f "${TEMPLATE}" ]; then
  echo "error: template not found at ${TEMPLATE}" >&2
  exit 1
fi

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${HOME}/Library/LaunchAgents"

# Substitute placeholders. We use sed because envsubst is not on macOS by default.
sed \
  -e "s|\${HOME}|${HOME}|g" \
  -e "s|\${PROJECT_DIR}|${PROJECT_DIR}|g" \
  "${TEMPLATE}" >"${TARGET}"

# Reload if already loaded.
if launchctl list | grep -q "${LABEL}"; then
  echo "Reloading existing agent…"
  launchctl unload "${TARGET}" || true
fi

launchctl load -w "${TARGET}"

echo
echo "✓ Installed ${TARGET}"
echo "✓ Logs:"
echo "    tail -f ${PROJECT_DIR}/logs/alerts.out.log"
echo "    tail -f ${PROJECT_DIR}/logs/alerts.err.log"
echo
echo "Useful commands:"
echo "  launchctl list | grep ${LABEL}        # is it running?"
echo "  launchctl stop ${LABEL}               # stop now (will auto-restart)"
echo "  launchctl unload ${TARGET}            # disable until next install"
echo
echo "Make sure the Mac is set to NEVER sleep:"
echo "  System Settings → Battery / Energy → Prevent automatic sleeping when display is off"
