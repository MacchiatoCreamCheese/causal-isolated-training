# Running MovieLens-10M

Thank you very much for helping us run MovieLens. This page collects everything we know about setting it up, so that it takes as little of your time as possible. If anything here is unclear or doesn't match what you see on your machine, please don't hesitate to message me; I'm happy to debug it together.

## What we are hoping to run

Only the hyperparameter tuning: 3 models × 2 samplers × 3 seeds = **18 cells**.

| | |
|---|---|
| models | `bpr` (CPU only), `neumf` (GPU), `lightgcn` (GPU) |
| samplers (`--recipe`) | `uniform`, `causal` |
| seeds | 42, 123, 2026 |

Each cell writes to its own directory, so cells never collide and can be split across jobs in whatever way suits your cluster. Finished trials are cached: if a cell is stopped by a walltime limit, resubmitting the same command resumes where it stopped (the log shows `[skip] ... [cached]`).

## 1. Environment

There is no environment file in the repository yet, sorry about that. This is the setup we use: python 3.12, cornac 2.6.0, torch 2.4.1+cu121.

```bash
conda create -n leakage24 python=3.12 -y
conda activate leakage24
pip install -r requirements.txt
pip install dgl          # needed for LightGCN only; see "Known issues" below
```

## 2. Data

The CSV can be downloaded here: [MovieLens-10M.csv](https://drive.google.com/file/d/1PJ3A_HOQpC9RmVbO2iI0gUl5yOX2EKKK/view?usp=sharing). Please place it at this path:

```
data/movielens_dataset/MovieLens-10M.csv
```

To confirm the file arrived intact: it should be **290,971,576 bytes**, with the header `user_id,item_id,rating,timestamp` and timestamps in **milliseconds** (first row: `1,122,5.0,838985046000`).

We would be grateful if you could use this file rather than a fresh conversion of GroupLens' `ratings.dat`. It has already been iteratively 5-cored and converted to milliseconds, and nothing filters it at load time, so a new export would give slightly different numbers from ours. The validation and test cutoffs (2006-04-18 and 2007-08-27) are fixed in the code, so nothing needs to be configured.

## 3. Preflight check

From the repository root, this checks the dataset key, the CSV, cornac, torch and CUDA, and lists all 18 cells without running anything:

```bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
DRY_RUN=1 bash run_dataset_tuning.sh movielens
```

## 4. Running

**One cell per job**, which should fit an sbatch array:

```bash
cd <repo root>                        # python -m needs the repository root
export RESEARCH_DATA_DIR=$PWD/data
export RESEARCH_NEUMF_PRETRAIN=1      # NeuMF only; see "Known issues"

python -u -m research.runners.tuning \
    --model neumf --dataset movielens --recipe uniform --seed 42
```

The options are `--model bpr|neumf|lightgcn`, `--recipe uniform|causal` and `--seed 42|123|2026`. To resume a partly finished cell from a particular knob, add `--restart-from <knob>`.

### sbatch array

In case it saves some time, here is a template that runs one model's 6 cells (2 samplers × 3 seeds) as a 6-task array. Please save it as `movielens_tune.sbatch` in the repository root. The partition, account, memory and time lines are placeholders; please adjust them to your cluster.

```bash
#!/bin/bash
#SBATCH --job-name=ml-tune
#SBATCH --partition=<partition>
#SBATCH --account=<account>
#SBATCH --array=0-5
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/%x_%A_%a.out

RECIPES=(uniform causal)
SEEDS=(42 123 2026)
RECIPE=${RECIPES[$((SLURM_ARRAY_TASK_ID % 2))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / 2))]}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate leakage24

cd "$SLURM_SUBMIT_DIR"
export RESEARCH_DATA_DIR="$PWD/data"

LABEL=$MODEL
if [ "$MODEL" = "neumf" ]; then
    export RESEARCH_NEUMF_PRETRAIN=1
    LABEL=neumf-pretrain
fi

python -u -m research.runners.tuning \
    --model "$MODEL" --dataset movielens --recipe "$RECIPE" --seed "$SEED" \
    >> "tune_${LABEL}_movielens_${RECIPE}_s${SEED}.log" 2>&1
```

Submitting, from the repository root:

```bash
mkdir -p slurm_logs
sbatch --export=ALL,MODEL=neumf    --gres=gpu:1 movielens_tune.sbatch
sbatch --export=ALL,MODEL=lightgcn --gres=gpu:1 movielens_tune.sbatch
sbatch --export=ALL,MODEL=bpr      --cpus-per-task=1 movielens_tune.sbatch
```

If a task stops at the time limit, submitting the same line again is safe: finished trials are skipped, and it continues from the next one. To resubmit only particular tasks, `--array=3,5` works as usual (task *i* is sampler `uniform`/`causal` for even/odd *i*, and seed 42, 123, 2026 for *i* = 0–1, 2–3, 4–5).

Resources per job:

| model | device | notes |
|---|---|---|
| `bpr` | CPU | NumPy, single-threaded, so extra cores do not help. |
| `neumf` | 1 GPU | torch |
| `lightgcn` | 1 GPU | torch + dgl |

Memory: loading MovieLens takes roughly 3.6× our largest Amazon dataset, so a generous memory request would be safest.

**Runtime, especially for BPR.** BPR now trains for the full epoch budget (early stopping is off), and one BPR trial on our Baby Products dataset (about 1M training rows) takes up to about 2.5 hours on our machine. MovieLens has about 8M training rows, so we expect a BPR trial there could take on the order of a day, and each cell runs 11 trials. This is only an estimate from our own runs. Because trials are cached, it is safe to run BPR under repeated walltime-limited jobs; each resubmission continues from the last finished trial. If BPR turns out to be too heavy for the cluster, please just let me know, and we can prioritise NeuMF and LightGCN or discuss another plan.

**Interactive smoke test** (for one long-lived session rather than the scheduler; it runs a CPU track and a GPU track in the background and waits for both):

```bash
bash run_dataset_tuning.sh movielens gpu     # or: cpu, all
```

## 5. Sending results back

When convenient, these are the files we need:

```
research/results/tuning/*/movielens/     # JSON + winner.users.npz
tune_*_movielens_*.log                   # written to the repository root
```

Partial results are also very welcome; everything is cached per trial, so we can merge whatever is finished.

## Known issues

| symptom | fix |
|---|---|
| `LD_LIBRARY_PATH: unbound variable` on `conda activate` | The scripts run under `set -u`. Running `export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}` before launching fixes it. |
| conda not found by the scripts | They hard-code `$HOME/miniconda3/etc/profile.d/conda.sh`. With a module-loaded conda, the `python -m` form above works without the scripts. |
| `ModuleNotFoundError: dgl` at the first LightGCN fit | cornac imports dgl lazily and it is not in `requirements.txt`. `pip install dgl`, with the CUDA build matching torch. |
| NeuMF results land in `neumf/` instead of `neumf-pretrain/` | `RESEARCH_NEUMF_PRETRAIN=1` was not set. That variant is not the one we report; the `neumf/` directory can be deleted and the cell re-run with the variable set. |
| `No module named research` | The command needs to be launched from the repository root. |

Thank you again for your time and help. It makes a real difference to the paper.
