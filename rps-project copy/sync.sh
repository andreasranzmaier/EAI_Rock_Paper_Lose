#!/bin/bash

# This script syncs the local directory with a remote Pi using rsync.
# Pi IP_ADDRESS should be set to the IP address of the remote Pi.
IP_ADDRESS="rpspi.local"
USERNAME="kepeterz"
SYNC_DIR="/home/kepeterz/rps-project"

# Function to sync files from local to remote Pi without removing files on the remote machine
sync_to_pi() {
    echo "Syncing files to remote Pi at $IP_ADDRESS..."
    rsync -avz ./ $USERNAME@$IP_ADDRESS:$SYNC_DIR
    echo "Sync complete."
}

# Function to sync files from remote Pi to local, without removing files on the local machine
sync_from_pi() {
    echo "Syncing files from remote Pi at $IP_ADDRESS..."
    rsync -avz $USERNAME@$IP_ADDRESS:$SYNC_DIR/ .
    echo "Sync complete."
}

# Main script logic
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