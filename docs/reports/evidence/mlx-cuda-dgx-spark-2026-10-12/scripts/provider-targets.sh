#!/bin/sh
# Build each Darkbloom provider-swift target alone on Linux (CUDA) and record pass/fail + first errors.
# d-inference 7796fe2 in swift-work/d-inference (libs/mlx-swift has the regenerated cuda_jit_sources.h).
# Same toolchain/flags as build-mlx-swift.sh. Output: logs/provider-targets/<target>.log and logs/provider-targets/SUMMARY.tsv
set -u
D="$HOME/projects/darkbloom-mlx-cuda"; P="$D/swift-work/d-inference/provider-swift"; O="$D/logs/provider-targets"
mkdir -p "$O"; exec > "$O/run.log" 2>&1
OB="$D/openblas"; CUDNN="$D/.venv-mlx-base/lib/python3.12/site-packages/nvidia/cudnn"; SHIM="$D/libshim"
CCCL=$D/third_party/cccl-v3.1.3/include; CUTLASS=$D/third_party/cutlass-v4.4.2/include
export NVCC_APPEND_FLAGS="-allow-unsupported-compiler -I$CCCL -I$CUTLASS -I$D/swift-work/d-inference/libs/mlx-swift/Source/Cmlx/mlx-generated/cuda"
export PATH="$D/swift-6.3.3/usr/bin:/usr/local/cuda-13.0/bin:$PATH" CUDA_ARCH=sm_121
export LD_LIBRARY_PATH="$SHIM:$OB/lib:$CUDNN/lib:/usr/local/cuda/lib64"
FLAGS="-Xcxx -I$CCCL -Xcxx -I$OB/include -Xcc -I$OB/include -Xcxx -I$CUDNN/include -Xcc -I$CUDNN/include \
-Xcxx -I$D/third_party/cudnn-frontend-v1.16.0/include -Xcxx -I$CUTLASS -Xlinker -L$SHIM -Xlinker -L$OB/lib -Xlinker -L$CUDNN/lib \
-Xlinker -L/usr/local/cuda/lib64 -Xlinker -lcufft -Xlinker -rpath -Xlinker $SHIM -Xlinker -rpath -Xlinker $OB/lib \
-Xlinker -rpath -Xlinker $CUDNN/lib -Xlinker -rpath -Xlinker /usr/local/cuda/lib64"
cd "$P"
echo "== $(date -Is) resolve"; swift package resolve || { echo RESOLVE_FAIL; exit 1; }
swift package dump-package > "$O/package.json"
TARGETS=$(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(" ".join(t["name"] for t in d["targets"] if t["type"] in ("regular","executable")))' "$O/package.json")
echo "targets: $TARGETS"
printf "target\tresult\tfirst_errors\n" > "$O/SUMMARY.tsv"
for T in $TARGETS; do
  echo "== $(date -Is) $T"
  if timeout 3600 swift build -c release -j 16 --target "$T" $FLAGS > "$O/$T.log" 2>&1; then R=PASS; E=""; else
    R=FAIL; E=$(grep -E "error:" "$O/$T.log" | sed "s|$D/swift-work/d-inference/||" | sort | uniq | head -3 | tr '\n' ' ' | cut -c1-600); fi
  printf "%s\t%s\t%s\n" "$T" "$R" "$E" >> "$O/SUMMARY.tsv"; echo "$T $R"
done
echo "PROVIDER_TARGETS_DONE $(date -Is)"
