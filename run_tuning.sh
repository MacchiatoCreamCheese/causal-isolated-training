#!/usr/bin/env bash
set -u

MODEL="${1:?usage: run_tuning.sh <bpr|neumf|lightgcn> [pretrain]}"
VARIANT="${2:-}"

CONDA_ENV="${CONDA_ENV:-leakage24}"
# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$(dirname "$(readlink -f "$0")")"

export RESEARCH_DATA_DIR="$PWD/data"

LABEL="$MODEL"
if [ "$VARIANT" = "pretrain" ]; then
    export RESEARCH_NEUMF_PRETRAIN=1
    LABEL="$MODEL-pretrain"
fi

echo "[run_tuning] $LABEL | env=$CONDA_ENV | data=$RESEARCH_DATA_DIR | $(date)"
python -c "import cornac, torch; print('[run_tuning] cornac', cornac.__version__, '| torch', torch.__version__)"

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
