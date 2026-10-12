#!/bin/sh
# Build the Layr-Labs mlx-swift fork (6923a80) with CUDA for GB10 and run a one-op GPU smoke test.
# Run on the Spark in ~/projects/darkbloom-mlx-cuda. Needs: swift-6.3.3/ (GPG-verified), openblas/,
# .venv-mlx-base (pip nvidia-cudnn-cu13), swift-work/mlx-swift (clone), swift-work/swift-smoke.
# User space only. Log: logs/build-mlx-swift.log (last line SWIFT_OK or SWIFT_FAIL).
set -eu
D="$HOME/projects/darkbloom-mlx-cuda"; W="$D/swift-work"; OB="$D/openblas"
CUDNN="$D/.venv-mlx-base/lib/python3.12/site-packages/nvidia/cudnn"; SHIM="$D/libshim"
exec > "$D/logs/build-mlx-swift.log" 2>&1
trap 'rc=$?; [ $rc -eq 0 ] || echo "SWIFT_FAIL rc=$rc"' EXIT
mkdir -p "$SHIM"
ln -sf "$OB/lib/libopenblas.so" "$SHIM/libblas.so"
ln -sf "$OB/lib/libopenblas.so" "$SHIM/liblapack.so"
ln -sf "$CUDNN/lib/libcudnn.so.9" "$SHIM/libcudnn.so"
export PATH="$D/swift-6.3.3/usr/bin:/usr/local/cuda-13.0/bin:$PATH" CUDA_ARCH=sm_121
export LD_LIBRARY_PATH="$SHIM:$OB/lib:$CUDNN/lib:/usr/local/cuda/lib64"
FLAGS="-Xcxx -I$CUDNN/include -Xcc -I$CUDNN/include -Xcxx -I$D/third_party/cudnn-frontend-v1.16.0/include -Xcxx -I$D/third_party/cutlass-v4.4.2/include -Xlinker -L$SHIM -Xlinker -L$OB/lib -Xlinker -L$CUDNN/lib \
-Xlinker -rpath -Xlinker $SHIM -Xlinker -rpath -Xlinker $OB/lib -Xlinker -rpath -Xlinker $CUDNN/lib -Xlinker -rpath -Xlinker /usr/local/cuda/lib64"
echo "== $(date -Is) $(swift --version 2>&1 | head -1)"
cd "$W/swift-smoke"
swift build -c release -j 16 $FLAGS
echo "== run $(date -Is)"
./.build/release/MLXSmoke
echo "SWIFT_OK $(date -Is)"
