#!/bin/bash
# step3-mlx.sh: MLX-ROCm half of step 3, by snapshot path (no refs/main in the HF cache). Two rounds.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK=${WORK:-$HOME/mlxrocm}
MODELS=${MODELS:-$HOME/models}
LLAMA_BUILD=${LLAMA_BUILD:-$HOME/build}
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
O=$WORK/step3; export HF_HUB_OFFLINE=1
source "$WORK/venv/bin/activate"
for r in 1 2; do
  python -m mlx_lm.benchmark --model $HF_CACHE/models--mlx-community--Qwen3-8B-4bit/snapshots/545dc4251c05440727734bcd94334791f6ab0192 -p 512 -g 128 -n 5 > $O/mlx-8b-r$r.txt 2>&1; echo "mlx 8b r$r rc=$?"
  python -m mlx_lm.benchmark --model $HF_CACHE/models--mlx-community--Qwen3-0.6B-4bit/snapshots/73e3e38d981303bc594367cd910ea6eb48349da8 -p 512 -g 128 -n 5 > $O/mlx-0.6b-r$r.txt 2>&1; echo "mlx 0.6b r$r rc=$?"
done
echo DONE > $O/DONE-mlx
