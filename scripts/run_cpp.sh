#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
)"

ORT_LIB_DIR="$PROJECT_ROOT/third_party/onnxruntime-linux-x64-gpu-1.19.2/lib"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "Error: CONDA_PREFIX is not set." >&2
    exit 1
fi

CUDNN_LIBRARY="$(
    find "$CONDA_PREFIX/lib" \
        -path "*/nvidia/cudnn/lib/libcudnn.so.9" \
        -print \
        -quit
)"

if [[ -z "$CUDNN_LIBRARY" ]]; then
    echo "Error: libcudnn.so.9 was not found in CONDA_PREFIX." >&2
    exit 1
fi

CUDNN_LIB_DIR="$(dirname "$CUDNN_LIBRARY")"

export LD_LIBRARY_PATH="$ORT_LIB_DIR:$CUDNN_LIB_DIR:${LD_LIBRARY_PATH:-}"

exec "$PROJECT_ROOT/build/road_perception" "$@"