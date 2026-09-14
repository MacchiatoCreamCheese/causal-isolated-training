#!/usr/bin/env bash
# Tune all three models on ONE dataset -- one machine per dataset.
#
# Two tracks run side by side, because they do not compete for the same device:
#   CPU track: bpr                    (our NumPy BPR never touches the GPU)
#   GPU track: neumf, then lightgcn   (one after the other on the GPU)
# Each track walks seed 42 first (the ablation's first need), then the rest,
# over both arms. Cells already tuned are skipped by the tuner itself.
#
# Usage (repo root, conda env with cornac):
#   bash run_dataset_tuning.sh <dataset> [all|cpu|gpu]
#   bash run_dataset_tuning.sh philadelphia gpu    # GPU machine: neumf + lightgcn only
#   bash run_dataset_tuning.sh philadelphia cpu    # CPU-only machine: bpr only
#   bash run_dataset_tuning.sh movielens           # both tracks on one machine
#
# Options (environment variables):
#   SEEDS='42 123 2026'     seeds, in order
#   RECIPES='uniform causal'
#   CPU_MODELS='bpr'        models for the CPU track ('' to skip it)
#   GPU_MODELS='neumf lightgcn'
#   NEUMF_PRETRAIN=1        1 = NeuMF with pre-training (default), 0 = from scratch
#   CONDA_ENV=leakage24
#   DRY_RUN=1               print what would run, run nothing
#
# From Windows/WSL, keep the window open (WSL stops with its last session):
#   wsl.exe -d Ubuntu -- bash -lc "bash ~/causal-isolated-training/run_dataset_tuning.sh philadelphia"
# On Linux over SSH:
#   nohup bash run_dataset_tuning.sh philadelphia > run_dataset_tuning_philadelphia.out 2>&1 &
set -u

DATASET="${1:?usage: run_dataset_tuning.sh <dataset key, e.g. philadelphia or movielens> [all|cpu|gpu]}"
TRACK="${2:-all}"
SEEDS="${SEEDS:-42 123 2026}"
RECIPES="${RECIPES:-uniform causal}"
CPU_MODELS="${CPU_MODELS-bpr}"
GPU_MODELS="${GPU_MODELS-neumf lightgcn}"
case "$TRACK" in
    all) ;;
    cpu) GPU_MODELS="" ;;   # CPU-only machine: never touch the GPU models
    gpu) CPU_MODELS="" ;;   # BPR runs on a separate CPU machine
    *) echo "[run_dataset_tuning] track must be all, cpu or gpu (got '$TRACK')"; exit 1 ;;
esac
NEUMF_PRETRAIN="${NEUMF_PRETRAIN:-1}"
CONDA_ENV="${CONDA_ENV:-leakage24}"
DRY_RUN="${DRY_RUN:-0}"

# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$(dirname "$(readlink -f "$0")")"
export RESEARCH_DATA_DIR="$PWD/data"

# Fail now, not hours in, if the dataset key or its CSV is wrong.
python - "$DATASET" <<'EOF' || exit 1
import os, sys
from research.lib.data import DATASETS, dataset_path
key = sys.argv[1]
if key not in DATASETS:
    sys.exit(f"[run_dataset_tuning] unknown dataset {key!r}; known: {sorted(DATASETS)}")
if not os.path.exists(dataset_path(key)):
    sys.exit(f"[run_dataset_tuning] missing {dataset_path(key)} -- copy data/ to this machine")
print(f"[run_dataset_tuning] {key}: {dataset_path(key)}")
EOF
if [ -n "$GPU_MODELS" ]; then
    python -c "import cornac, torch; print('[run_dataset_tuning] cornac', cornac.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())" || exit 1
else
    # BPR is NumPy only: no torch or GPU needed on a CPU machine.
    python -c "import cornac; print('[run_dataset_tuning] cornac', cornac.__version__, '| cpu-only track')" || exit 1
fi

# One tuning cell. Log names keep the backfill's pattern: tune_<model dir>_<dataset>_<recipe>_s<seed>.log
run_cell() {
    local model="$1" seed="$2" recipe="$3" label="$1"
    if [ "$model" = "neumf" ] && [ "$NEUMF_PRETRAIN" = "1" ]; then
        label="neumf-pretrain"
    fi
    local log="tune_${label}_${DATASET}_${recipe}_s${seed}.log"
    echo "[$(date +%m-%d\ %H:%M)] $label $DATASET/$recipe/seed$seed -> $log"
    if [ "$DRY_RUN" = "1" ]; then
        return 0
    fi
    if [ "$label" = "neumf-pretrain" ]; then
        RESEARCH_NEUMF_PRETRAIN=1 python -u -m research.runners.tuning \
            --model "$model" --dataset "$DATASET" --recipe "$recipe" --seed "$seed" >> "$log" 2>&1
    else
        python -u -m research.runners.tuning \
            --model "$model" --dataset "$DATASET" --recipe "$recipe" --seed "$seed" >> "$log" 2>&1
    fi || echo "[$(date +%m-%d\ %H:%M)] FAILED: $label $DATASET/$recipe/seed$seed -- see $log"
}

# A track: every model in it, seed-first, both arms.
run_track() {
    local name="$1"; shift
    for seed in $SEEDS; do
        for model in "$@"; do
            for recipe in $RECIPES; do
                run_cell "$model" "$seed" "$recipe"
            done
        done
    done
    echo "[$(date +%m-%d\ %H:%M)] $name track done"
}

echo "[run_dataset_tuning] $DATASET | seeds: $SEEDS | cpu: ${CPU_MODELS:-none} | gpu: ${GPU_MODELS:-none} | neumf pretrain: $NEUMF_PRETRAIN | $(date)"

pids=()
if [ -n "$CPU_MODELS" ]; then
    # shellcheck disable=SC2086
    run_track cpu $CPU_MODELS > "run_dataset_tuning_${DATASET}_cpu.out" 2>&1 &
    pids+=($!)
fi
if [ -n "$GPU_MODELS" ]; then
    # shellcheck disable=SC2086
    run_track gpu $GPU_MODELS > "run_dataset_tuning_${DATASET}_gpu.out" 2>&1 &
    pids+=($!)
fi
echo "[run_dataset_tuning] progress: run_dataset_tuning_${DATASET}_cpu.out, run_dataset_tuning_${DATASET}_gpu.out"
wait "${pids[@]}"
echo "[run_dataset_tuning] all done $(date)"
