#!/bin/bash
# step3.sh: MLX-ROCm vs llama.cpp Vulkan/HIP on Qwen3 8B and 0.6B. Two rounds, order swapped. Run under the GPU lock.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
set -u
WORK=${WORK:-$HOME/mlxrocm}
MODELS=${MODELS:-$HOME/models}
LLAMA_BUILD=${LLAMA_BUILD:-$HOME/build}
O=$WORK/step3; mkdir -p $O; cd "$MODELS"
export HF_HUB_OFFLINE=1
source "$WORK/venv/bin/activate"
lb() { # backend model tag round
  "$LLAMA_BUILD/up-$1/bin/llama-bench" -m $2 -ngl 999 -fa 1 -p 512 -n 128 -r 5 -o json > $O/$1-$3-r$4.json 2> $O/$1-$3-r$4.err; echo "$1 $3 r$4 rc=$?"; }
mb() { # repo tag round
  python -m mlx_lm.benchmark --model $1 -p 512 -g 128 -n 5 > $O/mlx-$2-r$3.txt 2>&1; echo "mlx $2 r$3 rc=$?"; }
for r in 1 2; do
  steps=(mlx vk hip); [ $r = 2 ] && steps=(hip vk mlx)
  for s in "${steps[@]}"; do
    if [ $s = mlx ]; then mb mlx-community/Qwen3-8B-4bit 8b $r; mb mlx-community/Qwen3-0.6B-4bit 0.6b $r
    else for q in Q4_0 Q4_K_M; do lb $s Qwen_Qwen3-8B-GGUF/Qwen3-8B-$q.gguf 8b-$q $r; done
         lb $s Qwen_Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_0.gguf 0.6b-Q4_0 $r; fi
  done
done
amd-smi version > $O/amd-smi-version.txt 2>&1; pip freeze > $O/pip-freeze.txt
echo DONE > $O/DONE
