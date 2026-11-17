#!/usr/bin/env bash
set -euo pipefail

if [[ $EUID -eq 0 ]]; then
    echo "Run as your normal user, not root (it uses sudo internally)." >&2
    exit 1
fi

echo "Installing Intel Level Zero runtime, headers, compiler, and Rust..."
sudo pacman -S --needed --noconfirm \
    level-zero-loader \
    level-zero-headers \
    intel-compute-runtime \
    intel-graphics-compiler \
    rustup

rustup default stable

echo "Setup complete!"
