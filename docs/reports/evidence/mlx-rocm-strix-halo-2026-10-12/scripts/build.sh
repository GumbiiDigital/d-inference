#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK=${WORK:-$HOME/mlxrocm}
MODELS=${MODELS:-$HOME/models}
LLAMA_BUILD=${LLAMA_BUILD:-$HOME/build}
cd "$WORK" && . venv/bin/activate && cd mlx
export CMAKE_ARGS="-DMLX_BUILD_ROCM=ON -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CPU=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151 -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF"
export CMAKE_BUILD_PARALLEL_LEVEL=24
export PATH=/opt/rocm/bin:$PATH
start=$(date +%s)
pip install -v . > ../build.log 2>&1; rc=$?
echo "rc=$rc secs=$(( $(date +%s) - start ))" >> ../build.log
pip freeze > ../pip-freeze.txt
echo DONE >> ../build.log
