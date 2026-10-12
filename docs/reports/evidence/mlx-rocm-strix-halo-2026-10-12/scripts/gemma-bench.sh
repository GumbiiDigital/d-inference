#!/bin/bash
# gemma-bench.sh: Gemma 4 26B A4B, MLX-ROCm (pinned 75915908) vs llama.cpp Vulkan/HIP on the same box. p512/g128, 2 rounds.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK=${WORK:-$HOME/mlxrocm}
MODELS=${MODELS:-$HOME/models}
LLAMA_BUILD=${LLAMA_BUILD:-$HOME/build}
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
O=$WORK/gemma; mkdir -p $O; export HF_HUB_OFFLINE=1; source "$WORK/venv/bin/activate"
M=$HF_CACHE/models--mlx-community--gemma-4-26b-a4b-it-4bit/snapshots/0d77464eeb233a2da68ebf9d7dc4edaac7db956d
G=$MODELS/gemma-4-26B_q4_0-it.gguf
cd "$WORK"; python3 "$SCRIPT_DIR/tokref.py" mlx-community/gemma-4-26b-a4b-it-4bit 0d77464eeb233a2da68ebf9d7dc4edaac7db956d gpu $O/tokref-gpu-pinned.json > /dev/null 2>&1; echo "tokref rc=$?"
for r in 1 2; do
  python -m mlx_lm.benchmark --model $M -p 512 -g 128 -n 5 > $O/mlx-default-r$r.txt 2>&1; echo "mlx default r$r rc=$?"
  HSA_ENABLE_INTERRUPT=0 python -m mlx_lm.benchmark --model $M -p 512 -g 128 -n 5 > $O/mlx-nointr-r$r.txt 2>&1; echo "mlx nointr r$r rc=$?"
  for b in vk hip; do "$LLAMA_BUILD/up-$b/bin/llama-bench" -m $G -ngl 999 -fa 1 -p 512 -n 128 -d 0,512 -r 5 -o json > $O/$b-r$r.json 2> $O/$b-r$r.err; echo "$b r$r rc=$?"; done
done
echo DONE > $O/DONE
