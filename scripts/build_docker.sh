#!/bin/bash
# Usage: scripts/build_docker.sh [cu126|cpu]   (PyTorch build, cu126 by default)
docker build --build-arg TORCH="${1:-cu126}" -t deepanedet "$(dirname "$0")/.."
