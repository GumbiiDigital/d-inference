#!/bin/bash
# sweep2.sh: second MLX-ROCm env sweep on Qwen3-8B 4-bit decode. 2 runs each.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK=${WORK:-$HOME/mlxrocm}
MODELS=${MODELS:-$HOME/models}
LLAMA_BUILD=${LLAMA_BUILD:-$HOME/build}
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
O=$WORK/sweep; mkdir -p $O; source "$WORK/venv/bin/activate"; export HF_HUB_OFFLINE=1
M=$HF_CACHE/models--mlx-community--Qwen3-8B-4bit/snapshots/545dc4251c05440727734bcd94334791f6ab0192
run() { # name env...
  for i in 1 2; do env "${@:2}" python -m mlx_lm generate --model $M --prompt "The history of the printing press begins in" --ignore-chat-template --temp 0 -m 128 > $O/$1-$i.txt 2>&1; done
  sed -n "/^==========/,/^==========/p" $O/$1-1.txt > $O/$1.text
  same=$(cmp -s $O/base.text $O/$1.text && echo same || echo DIFF)
  echo "$1: $(grep -h Generation $O/$1-[12].txt | grep -oE "[0-9.]+ tokens-per-sec" | tr "\n" " ") tokens=$same"
}
run serial3 AMD_SERIALIZE_KERNEL=3
run nointr HSA_ENABLE_INTERRUPT=0
run kernarg HIP_FORCE_DEV_KERNARG=1
run profcheck X=1
