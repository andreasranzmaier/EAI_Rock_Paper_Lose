#!/bin/bash
set -euo pipefail

# Load PI_USER, PI_HOST, PI_REMOTE_DIR from the project root .env
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  set -a; source "$ROOT/.env"; set +a
fi

: "${PI_USER:?Set PI_USER in .env}"
: "${PI_HOST:?Set PI_HOST in .env}"
: "${PI_REMOTE_DIR:?Set PI_REMOTE_DIR in .env}"

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sync_to_pi() {
    echo "Syncing files to $PI_USER@$PI_HOST:$PI_REMOTE_DIR ..."
    rsync -avz "$LOCAL_DIR/" "$PI_USER@$PI_HOST:$PI_REMOTE_DIR/"
    echo "Sync complete."
}

sync_from_pi() {
    echo "Syncing files from $PI_USER@$PI_HOST:$PI_REMOTE_DIR ..."
    rsync -avz "$PI_USER@$PI_HOST:$PI_REMOTE_DIR/" "$LOCAL_DIR/"
    echo "Sync complete."
}

if [ "$1" == "to" ]; then
    sync_to_pi
elif [ "$1" == "from" ]; then
    sync_from_pi
elif [ "$1" == "both" ]; then
    sync_to_pi
    sync_from_pi
else
    echo "Usage: $0 [to|from|both]"
    echo "  to   - Sync local files to remote Pi"
    echo "  from - Sync files from remote Pi to local"
    echo "  both - Sync files in both directions"
    exit 1
fi
