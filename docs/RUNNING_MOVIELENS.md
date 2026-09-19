# Running MovieLens-10M

Tuning only: 3 models x 2 samplers x 3 seeds = **18 cells**.

| | |
|---|---|
| models | `bpr` (CPU only), `neumf` (GPU), `lightgcn` (GPU) |
| samplers (`--recipe`) | `uniform`, `causal` |
| seeds | 42, 123, 2026 |

Each cell writes its own directory, so cells never collide and can be split
across jobs however you like. Finished trials are cached: a cell killed on
walltime resumes where it stopped (`[skip] ... [cached]`) when resubmitted.

## 1. Environment

There is no environment file in the repo; build it by hand. What we run:
python 3.12, cornac 2.6.0, torch 2.4.1+cu121.

```bash
conda create -n leakage24 python=3.12 -y
conda activate leakage24
pip install -r requirements.txt
pip install dgl          # LightGCN only -- see gotchas
```

## 2. Data

Put the CSV we sent you at exactly this path:

```
data/movielens_dataset/MovieLens-10M.csv
```

Check it arrived whole: **290,971,576 bytes**, header
`user_id,item_id,rating,timestamp`, timestamps in **milliseconds**
(`1,122,5.0,838985046000`).

Use our file, not a fresh conversion of GroupLens' `ratings.dat`. It is already
iteratively 5-cored and converted to milliseconds, and nothing filters at load
time, so a re-export would quietly produce different numbers. The validation and
test cutoffs are fixed in code (2006-04-18 and 2007-08-27); nothing to configure.

## 3. Preflight

From the repo root. Checks the dataset key, the CSV, cornac, torch and CUDA,
lists all 18 cells, runs nothing:

```bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
DRY_RUN=1 bash run_dataset_tuning.sh movielens
```

## 4. Run

**One cell per job — use this for an sbatch array.**

```bash
cd <repo root>                        # python -m needs the repo root
export RESEARCH_DATA_DIR=$PWD/data
export RESEARCH_NEUMF_PRETRAIN=1      # NeuMF only, see gotchas

python -u -m research.runners.tuning \
    --model neumf --dataset movielens --recipe uniform --seed 42
```

`--model bpr|neumf|lightgcn`, `--recipe uniform|causal`, `--seed 42|123|2026`.
Add `--restart-from <knob>` to resume a part-finished cell at a given knob.

Resource request per job:

| model | device | notes |
|---|---|---|
| `bpr` | CPU | NumPy, single-threaded. Extra cores do not help. |
| `neumf` | 1 GPU | torch |
| `lightgcn` | 1 GPU | torch + dgl |

Budget memory generously: MovieLens costs roughly 3.6x our largest Amazon
dataset to load.

**Interactive smoke test** (one long-lived session, not for the scheduler — it
backgrounds a CPU and a GPU track and waits on both):

```bash
bash run_dataset_tuning.sh movielens gpu     # or: cpu, all
```

## 5. Sending results back

```
research/results/tuning/*/movielens/     # JSON + winner.users.npz
tune_*_movielens_*.log                   # written to the repo root
```

## Gotchas

| symptom | fix |
|---|---|
| `LD_LIBRARY_PATH: unbound variable` on `conda activate` | The scripts run under `set -u`. `export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}` before launching. |
| conda not found by the scripts | They hard-code `$HOME/miniconda3/etc/profile.d/conda.sh`. With a module-loaded conda, skip the scripts and use the `python -m` form. |
| `ModuleNotFoundError: dgl` at the first LightGCN fit | cornac imports dgl lazily and it is not in `requirements.txt`. `pip install dgl`, CUDA build matching torch. |
| NeuMF results land in `neumf/` | `RESEARCH_NEUMF_PRETRAIN=1` was not set. That variant is not the one we report; delete the directory and re-run with it set. |
| `No module named research` | Launch from the repo root. |

Two things that are expected, not bugs: MovieLens' rating burst leaves only
about 1,750 evaluable test users, and LightGCN runs at batch size 1024 here (the
per-dataset table has no MovieLens entry, so it falls back).
