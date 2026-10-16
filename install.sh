#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
git submodule update --init --recursive
"${PYTHON:-python3}" -m pip install --no-build-isolation --no-deps -e . "$@"
