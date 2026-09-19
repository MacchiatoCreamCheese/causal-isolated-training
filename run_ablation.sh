#!/usr/bin/env bash
set -u

MODEL="${1:?usage: run_ablation.sh <bpr|neumf|lightgcn> [pretrain]}"
VARIANT="${2:-}"

CONDA_ENV="${CONDA_ENV:-leakage24}"
# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "$(dirname "$(readlink -f "$0")")"
export RESEARCH_DATA_DIR="$PWD/data"
export RESEARCH_TUNED_ARM=per_arm

LABEL="$MODEL"
if [ "$VARIANT" = "pretrain" ]; then
    export RESEARCH_NEUMF_PRETRAIN=1
    LABEL="$MODEL-pretrain"
fi

SEEDS="${SEEDS:-42 123 2026}"
DATASETS="${DATASETS:-musical baby cellphone}"

echo "[run_ablation] $LABEL | env=$CONDA_ENV | data=$RESEARCH_DATA_DIR | $(date)"
python -c "import cornac, torch; print('[run_ablation] cornac', cornac.__version__, '| torch', torch.__version__)"

for seed in $SEEDS; do
    for ds in $DATASETS; do
        log="ablation_${LABEL}_${ds}_s${seed}.log"
        echo "[run_ablation] $ds/seed$seed -> $log  ($(date +%H:%M))"
        python -u -m research.mecha2.runner \
            --model "$MODEL" --datasets "$ds" --seeds "$seed" \
            >> "$log" 2>&1 \
            || echo "[run_ablation] FAILED: $ds/seed$seed -- see $log"
    done
done
echo "[run_ablation] done $(date)"
