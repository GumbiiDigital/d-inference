#!/bin/sh
# Build MLX v0.32.3 (64ea011) from source with CUDA for GB10 (sm_121) into its own venv.
# Usage: sh build-mlx-src.sh <variant>   variant = base | qmm
#   base: unpatched source build (control for "source build vs pip wheel")
#   qmm:  + mlx-serve patches/mlx-cuda-qmm-read-weights-once.patch (MIT, Copyright (c) 2026 mlx-serve patch author;
#         licence text in patches/LICENSE-mlx-serve)
# Needs in ~/projects/darkbloom-mlx-cuda: wheels/ (pinned, hashed), mlx-src/ (git clone v0.32.3), patches/.
# CPU BLAS/LAPACK: OpenBLAS v0.3.33 (62bcfb0) built from source into ./openblas (user space).
# No sudo, no system changes. Log: logs/build-mlx-<variant>.log (last line BUILD_OK_<v> or BUILD_FAIL_<v>).
set -eu
V="$1"; D="$HOME/projects/darkbloom-mlx-cuda"; cd "$D"
SRC="$D/mlx-src-$V"; VENV="$D/.venv-mlx-$V"; LOG="$D/logs/build-mlx-$V.log"; OB="$D/openblas"
mkdir -p logs
exec > "$LOG" 2>&1
trap 'rc=$?; [ $rc -eq 0 ] || echo "BUILD_FAIL_$V rc=$rc"' EXIT
echo "== $(date -Is) variant=$V host=$(hostname)"

if [ ! -f "$OB/lib/libopenblas.so" ]; then
  echo "== OpenBLAS v0.3.33"
  rm -rf "$D/openblas-src"
  git clone -q --depth 1 --branch v0.3.33 https://github.com/OpenMathLib/OpenBLAS "$D/openblas-src"
  test "$(git -C "$D/openblas-src" rev-parse HEAD)" = 62bcfb0dc9f1cfa685fc04135c50e2780c303137
  make -C "$D/openblas-src" -j 16 TARGET=ARMV8 USE_OPENMP=0 NUM_THREADS=64 NO_FORTRAN=0 > "$D/logs/openblas-make.log" 2>&1 \
    || make -C "$D/openblas-src" -j 16 TARGET=ARMV8 USE_OPENMP=0 NUM_THREADS=64 NOFORTRAN=1 C_LAPACK=1 > "$D/logs/openblas-make.log" 2>&1
  make -C "$D/openblas-src" PREFIX="$OB" install > "$D/logs/openblas-install.log" 2>&1
fi
ls "$OB/include/cblas.h" "$OB/include/lapacke.h" "$OB/lib/libopenblas.so"

rm -rf "$SRC"; git clone -q "$D/mlx-src" "$SRC"
test "$(git -C "$SRC" rev-parse HEAD)" = 64ea011cb65f14d9ce2737e60db9a4ae91ed7441
if [ "$V" = qmm ]; then
  sha256sum patches/mlx-cuda-qmm-read-weights-once.patch
  # Source part only: the tests/cuda_tests.cpp hunk does not apply at 64ea011, and tests are off.
  git -C "$SRC" apply -p1 --include='mlx/*' "$D/patches/mlx-cuda-qmm-read-weights-once.patch"
  git -C "$SRC" diff --stat
fi
rm -rf "$VENV"; python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --no-index --find-links wheels "setuptools==80.9.0" "nvidia-cudnn-cu13==9.27.0.42" \
  "nvidia-cublas==13.8.1.7" "nvidia-cuda-nvrtc==13.4.92" "nvidia-cuda-runtime==13.4.92" "nvidia-cufft==12.4.0.43" \
  "nvidia-cusolver==12.3.4.7" "nvidia-nccl-cu13==2.32.3" typing_extensions numpy
CUDNN="$VENV/lib/python3.12/site-packages/nvidia/cudnn"
export PATH=/usr/local/cuda-13.0/bin:$PATH
# Same CUDA arch list and python-lib lookup as the official wheel (setup.py, MLX_BUILD_BACKEND_PACKAGE path).
export CMAKE_ARGS="-DMLX_BUILD_CUDA=ON -DMLX_CUDA_ARCHITECTURES=121a-real -DMLX_LOAD_CUDA_LIBS_FROM_PYTHON=ON \
-DCUDNN_INCLUDE_PATH=$CUDNN/include -DCUDNN_LIBRARY_PATH=$CUDNN/lib \
-DBLAS_LIBRARIES=$OB/lib/libopenblas.so -DLAPACK_LIBRARIES=$OB/lib/libopenblas.so \
-DLAPACK_INCLUDE_DIRS=$OB/include -DBLAS_INCLUDE_DIRS=$OB/include -DCMAKE_BUILD_RPATH=$OB/lib -DCMAKE_INSTALL_RPATH=$OB/lib"
# Python 3.12 headers: Ubuntu libpython3.12-dev/python3.12-dev .debs unpacked into ./pydev/root (apt-get download + dpkg -x, no install).
PYINC="$D/pydev/root/usr/include"
CMAKE_ARGS="$CMAKE_ARGS -DPython_INCLUDE_DIR=$PYINC/python3.12"
export CFLAGS="-I$PYINC" CXXFLAGS="-I$PYINC"
export CMAKE_BUILD_PARALLEL_LEVEL=16
cd "$SRC"
"$VENV/bin/pip" wheel -v --no-build-isolation --no-deps -w "$D/wheels-mlx-$V" . > "$D/logs/build-mlx-$V-pipwheel.log" 2>&1
cd "$D"
ls wheels-mlx-$V/*.whl
sha256sum wheels-mlx-$V/*.whl
"$VENV/bin/pip" install -q --no-index --no-deps wheels-mlx-$V/*.whl
"$VENV/bin/pip" install -q --no-index --find-links wheels --no-deps "mlx-lm==0.32.0"
"$VENV/bin/pip" install -q --no-index --find-links wheels transformers sentencepiece protobuf pyyaml jinja2 huggingface_hub safetensors tokenizers regex tqdm packaging filelock fsspec
"$VENV/bin/pip" freeze > logs/pip-freeze-mlx-$V.txt
LD_LIBRARY_PATH="$OB/lib" "$VENV/bin/python" -c "import mlx.core as mx, mlx_lm; print(mx.__version__, mlx_lm.__version__, mx.default_device(), mx.device_info()['architecture']); a=mx.random.normal((1024,1024)); mx.eval(a@a); print('smoke ok')"
echo "BUILD_OK_$V $(date -Is)"
