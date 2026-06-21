#!/usr/bin/env bash
# Install (or refresh) the rock_paper_lose systemd unit on the Pi.
# Run *on the Pi*, after the binary + model + labels are deployed.
#
# Defaults match the layout that rps-project/sync.sh + the binary rsync produce
# on the Pi: the binary lives in INSTALL_DIR and the trained model from
# rps-project/model/ is rsynced into INSTALL_DIR/model/. Override via env vars
# if your layout differs:
#   INSTALL_DIR=/home/kit-23/rps-project
#   BIN_NAME=rock_paper_lose
#   MODEL_NAME=model/model.tflite
#   LABELS_NAME=model/labels.txt
#   USER_NAME=kit-23
set -euo pipefail

USER_NAME=${USER_NAME:-$(id -un)}
INSTALL_DIR=${INSTALL_DIR:-/home/${USER_NAME}/rps-project}
BIN_NAME=${BIN_NAME:-rock_paper_lose}
MODEL_NAME=${MODEL_NAME:-model/model.tflite}
LABELS_NAME=${LABELS_NAME:-model/labels.txt}
UNIT_NAME=rock_paper_lose.service

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/rock_paper_lose.service.in"

if [[ ! -f "$TEMPLATE" ]]; then
  echo "Missing service template: $TEMPLATE" >&2
  exit 1
fi

for f in "${INSTALL_DIR}/${BIN_NAME}" "${INSTALL_DIR}/${MODEL_NAME}" "${INSTALL_DIR}/${LABELS_NAME}"; do
  if [[ ! -e "$f" ]]; then
    echo "Expected file is missing: $f" >&2
    exit 1
  fi
done

TMP_UNIT="$(mktemp)"
trap 'rm -f "$TMP_UNIT"' EXIT

sed \
  -e "s|__USER__|${USER_NAME}|g" \
  -e "s|__INSTALL_DIR__|${INSTALL_DIR}|g" \
  -e "s|__BIN_NAME__|${BIN_NAME}|g" \
  -e "s|__MODEL_NAME__|${MODEL_NAME}|g" \
  -e "s|__LABELS_NAME__|${LABELS_NAME}|g" \
  "$TEMPLATE" > "$TMP_UNIT"

sudo install -m 644 "$TMP_UNIT" "/etc/systemd/system/${UNIT_NAME}"
sudo systemctl daemon-reload
sudo systemctl enable "$UNIT_NAME"
sudo systemctl restart "$UNIT_NAME"

echo "Installed and started ${UNIT_NAME}."
echo "Status: systemctl status ${UNIT_NAME}"
echo "Logs:   journalctl -u ${UNIT_NAME} -f"
