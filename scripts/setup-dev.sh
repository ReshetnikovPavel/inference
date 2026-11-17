#!/usr/bin/env bash
set -euo pipefail

if [[ $EUID -eq 0 ]]; then
    echo "Run as your normal user, not root (it uses sudo internally)." >&2
    exit 1
fi

echo "Installing build and development dependencies..."
sudo pacman -S --needed --noconfirm \
    rustup \
    level-zero-headers \
    intel-graphics-compiler \
    clang \
    rust-bindgen

rustup default stable

echo "Dev setup complete!"
