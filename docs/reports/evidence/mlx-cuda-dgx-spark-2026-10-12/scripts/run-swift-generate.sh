#!/bin/sh
# Build and run the Swift greedy-generate CLI (mlx-swift-lm 3fd4944 + mlx-swift 6923a80, CUDA) on the Spark,
# then the Python MLX-CUDA trace on the same model and prompts, then compare. Log: logs/swift-generate.log
set -eu
D="$HOME/projects/darkbloom-mlx-cuda"; W="$D/swift-work"
OB="$D/openblas"; CUDNN="$D/.venv-mlx-base/lib/python3.12/site-packages/nvidia/cudnn"; SHIM="$D/libshim"
CCCL=$D/third_party/cccl-v3.1.3/include; CUTLASS=$D/third_party/cutlass-v4.4.2/include
exec > "$D/logs/swift-generate.log" 2>&1
trap 'rc=$?; [ $rc -eq 0 ] || echo "GEN_FAIL rc=$rc"' EXIT
export NVCC_APPEND_FLAGS="-allow-unsupported-compiler -I$CCCL -I$CUTLASS -I$W/mlx-swift/Source/Cmlx/mlx-generated/cuda"
export PATH="$D/swift-6.3.3/usr/bin:/usr/local/cuda-13.0/bin:$PATH" CUDA_ARCH=sm_121
export LD_LIBRARY_PATH="$SHIM:$OB/lib:$CUDNN/lib:/usr/local/cuda/lib64"
export MLX_NVRTC_EXTRA_INCLUDE="$D/third_party/cutlass-v4.4.2/include"
FLAGS="-Xcxx -I$CCCL -Xcxx -I$OB/include -Xcc -I$OB/include -Xcxx -I$CUDNN/include -Xcc -I$CUDNN/include \
-Xcxx -I$D/third_party/cudnn-frontend-v1.16.0/include -Xcxx -I$CUTLASS -Xlinker -L$SHIM -Xlinker -L$OB/lib -Xlinker -L$CUDNN/lib \
-Xlinker -L/usr/local/cuda/lib64 -Xlinker -lcufft -Xlinker -rpath -Xlinker $SHIM -Xlinker -rpath -Xlinker $OB/lib \
-Xlinker -rpath -Xlinker $CUDNN/lib -Xlinker -rpath -Xlinker /usr/local/cuda/lib64"
cd "$W/swift-generate"
echo "== build $(date -Is)"; swift build -c release -j 16 $FLAGS
JIT_INC="$W/swift-generate/.build/aarch64-unknown-linux-gnu/include"
mkdir -p "$JIT_INC"
# NVRTC JIT expects CUTLASS/CuTe headers beside the release binary (see mlx jit_module.cpp).
rsync -a --delete "$CUTLASS/cute/" "$JIT_INC/cute/"
rsync -a --delete "$CUTLASS/cutlass/" "$JIT_INC/cutlass/"
rsync -a --delete "$CCCL/" "$JIT_INC/cccl/"
mkdir -p "$D/results/swift"
echo "== swift run $(date -Is)"
PROMPTS="$W/data/trace-prompts.jsonl"
TRACE="$W/tools/mlx_trace.py"
rm -rf /tmp/mlx 2>/dev/null || true
SWIFT_OUT="$D/results/swift/trace-spark-b-swift-q3-8b-q4g64.jsonl"
set +e
./.build/release/MLXGenerate --model "$D/models/mlx-q3-8b-q4g64" --prompts "$PROMPTS" --max-tokens 64 \
  --out "$SWIFT_OUT" --label "mlx-swift-lm 3fd4944 + mlx-swift 6923a80 CUDA spark-b"
SWIFT_RC=$?
set -e
test -s "$SWIFT_OUT" || exit "$SWIFT_RC"
echo "== python run $(date -Is)"
cd "$D"
LD_LIBRARY_PATH="$OB/lib" .venv-mlx-base/bin/python "$TRACE" run --model models/mlx-q3-8b-q4g64 --prompts "$PROMPTS" \
  --max-tokens 64 --out results/swift/trace-spark-b-python-q3-8b-q4g64.jsonl --label "python mlx 0.32.3 source build (base) spark-b"
.venv-mlx-base/bin/pip download -q --no-deps --only-binary=:all: -d wheels "mlx-cuda-13==0.32.3" "mlx==0.32.3" >/dev/null 2>&1 || true
echo "== compare $(date -Is)"
python3 "$TRACE" compare results/swift/trace-spark-b-python-q3-8b-q4g64.jsonl results/swift/trace-spark-b-swift-q3-8b-q4g64.jsonl || true
echo "GEN_DONE $(date -Is)"
