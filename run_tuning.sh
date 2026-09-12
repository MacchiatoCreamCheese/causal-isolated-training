#!/usr/bin/env bash
# Tune one model over every dataset x arm x seed, cheapest-first.
#
# Written to be launched from Windows without quoting gymnastics:
#
#     wsl.exe -d Ubuntu -- bash /mnt/c/Users/nguye/uniyear/causal-isolated-training/run_tuning.sh neumf pretrain
#
# Keep that window open: WSL shuts the VM down when its last session ends, and
# takes every background job with it -- `nohup` does not survive that.
#
# Usage:  bash run_tuning.sh <bpr|neumf|lightgcn> [pretrain]
set -u

MODEL="${1:?usage: run_tuning.sh <bpr|neumf|lightgcn> [pretrain]}"
VARIANT="${2:-}"

# `bash -lc` from Windows is not an interactive shell, so conda is not on PATH
# and `python` resolves to nothing. Activate the env explicitly.
CONDA_ENV="${CONDA_ENV:-leakage24}"
# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$(dirname "$(readlink -f "$0")")"

# The CSVs live under data/, one level below the repo root that paths.py
# assumes by default.
export RESEARCH_DATA_DIR="$PWD/data"

LABEL="$MODEL"
if [ "$VARIANT" = "pretrain" ]; then
    export RESEARCH_NEUMF_PRETRAIN=1
    LABEL="$MODEL-pretrain"
fi

echo "[run_tuning] $LABEL | env=$CONDA_ENV | data=$RESEARCH_DATA_DIR | $(date)"
python -c "import cornac, torch; print('[run_tuning] cornac', cornac.__version__, '| torch', torch.__version__)"

# Seed 42 first: the ablation reads seed-42 winners, so an interrupted sweep
# still leaves the cells that matter most finished.
#
# SEEDS / DATASETS / RECIPES split one model's sweep across machines. Each cell
# writes to its own results/tuning/.../seed<n>/ directory, so two machines on
# different seeds never touch the same file:
#
#     SEEDS='123 2026' bash run_tuning.sh lightgcn
SEEDS="${SEEDS:-42 123 2026}"
DATASETS="${DATASETS:-musical baby cellphone}"
RECIPES="${RECIPES:-uniform causal}"

for seed in $SEEDS; do
    for ds in $DATASETS; do
        for recipe in $RECIPES; do
            log="tune_${LABEL}_${ds}_${recipe}_s${seed}.log"
            echo "[run_tuning] $ds/$recipe/seed$seed -> $log  ($(date +%H:%M))"
            python -u -m research.runners.tuning \
                --model "$MODEL" --dataset "$ds" --recipe "$recipe" --seed "$seed" \
                >> "$log" 2>&1 \
                || echo "[run_tuning] FAILED: $ds/$recipe/seed$seed -- see $log"
        done
    done
done
echo "[run_tuning] done $(date)"
