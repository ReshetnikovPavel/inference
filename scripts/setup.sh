#!/usr/bin/env bash
set -euo pipefail

if [[ $EUID -eq 0 ]]; then
    echo "Run as your normal user, not root (it uses sudo internally)." >&2
    exit 1
fi

echo "Installing Level Zero runtime dependencies..."
sudo pacman -S --needed --noconfirm \
    level-zero-loader \
    intel-compute-runtime

echo "Runtime setup complete!"
