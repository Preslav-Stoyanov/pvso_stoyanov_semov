#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run.sh – Start the Gaussian Splatting Docker container
#
# Usage:
#   ./run.sh              # interactive shell
#   ./run.sh gs-train     # run a named command directly
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Sanity checks ─────────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    echo "ERROR: docker not found. Install Docker first."
    exit 1
fi

if ! docker info --format '{{.Runtimes}}' 2>/dev/null | grep -q nvidia; then
    echo "WARNING: NVIDIA container runtime not detected."
    echo "  Install nvidia-container-toolkit and restart Docker."
    echo "  Continuing anyway – GPU features may not work."
fi

# ── Allow Docker to use your X display ───────────────────────────────────────
if command -v xhost &>/dev/null; then
    xhost +local:docker >/dev/null 2>&1 || true
else
    echo "WARNING: xhost not found. Install x11-xserver-utils for GUI support."
fi

# ── Build image if it does not exist yet ──────────────────────────────────────
if ! docker image inspect gaussian-splatting:latest &>/dev/null; then
    echo "Image not found. Building (this takes ~30-60 min on first run)..."
    docker compose build
fi

# ── Run ───────────────────────────────────────────────────────────────────────
if [ $# -eq 0 ]; then
    echo "Starting interactive shell..."
    docker compose run --rm gaussian-splatting /bin/bash
else
    echo "Running: $*"
    docker compose run --rm gaussian-splatting "$@"
fi

# ── Revoke X11 permission on exit ─────────────────────────────────────────────
if command -v xhost &>/dev/null; then
    xhost -local:docker >/dev/null 2>&1 || true
fi
