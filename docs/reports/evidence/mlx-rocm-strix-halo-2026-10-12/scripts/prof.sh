#!/bin/bash
# prof.sh: rocprofv3 kernel trace of MLX-ROCm Qwen3-8B 4-bit, -m 1 vs -m 129. Decode cost per kernel = (B - A) / 128.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK=${WORK:-$HOME/mlxrocm}
MODELS=${MODELS:-$HOME/models}
LLAMA_BUILD=${LLAMA_BUILD:-$HOME/build}
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
O=$WORK/prof; rm -rf $O; mkdir -p $O; export HF_HUB_OFFLINE=1
source "$WORK/venv/bin/activate"
M=$HF_CACHE/models--mlx-community--Qwen3-8B-4bit/snapshots/545dc4251c05440727734bcd94334791f6ab0192
for n in 1 129; do
  rocprofv3 --kernel-trace --stats -d $O/m$n -o run --output-format csv -- \
    python -m mlx_lm generate --model $M --prompt "The history of the printing press begins in" --ignore-chat-template --temp 0 -m $n > $O/m$n.log 2>&1
  echo "m$n rc=$?"
done
echo DONE > $O/DONE
