"""Filesystem anchors for the research package.

Every dataset path and output directory is resolved through the constants here
rather than being hard-coded relative to the current working directory, so the
package runs the same whether invoked from the repo root, from a relocated
standalone project, or from anywhere else.

Layout:

    PROJECT_ROOT/            parent of this package (repo root, or relocated root)
      research/              this package
        results/            per-run JSON, summaries (git-tracked)
        logs/               cornac Experiment save_dir
        figures/            generated plots
      baby_dataset/          dataset CSVs (or point RESEARCH_DATA_DIR elsewhere)
      cellphone_dataset/
      healthcare_dataset/

Outputs default to *inside* this package (matching the historical on-disk
layout, so existing `research/results` and `research/logs` keep working).

Overrides via environment variables:
  - RESEARCH_DATA_DIR    where the *_dataset/ folders live (default: PROJECT_ROOT)
  - RESEARCH_OUTPUT_DIR  where results/logs/figures are written
                         (default: the package directory)
"""

import os
from pathlib import Path

# PACKAGE_DIR = this `research/` package; PROJECT_ROOT = its parent.
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

DATA_DIR = Path(os.environ.get("RESEARCH_DATA_DIR", PROJECT_ROOT))
OUTPUT_DIR = Path(os.environ.get("RESEARCH_OUTPUT_DIR", PACKAGE_DIR))

RESULTS_DIR = OUTPUT_DIR / "results"
LOGS_DIR = OUTPUT_DIR / "logs"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Reports (markdown tables/docs) are tracked alongside the package source.
REPORTS_DIR = PACKAGE_DIR / "reports"


def data_path(*parts) -> str:
    """Absolute path to a dataset file under DATA_DIR, as a string
    (cornac's Reader wants a str)."""
    return str(DATA_DIR.joinpath(*parts))


def results_path(*parts) -> Path:
    return RESULTS_DIR.joinpath(*parts)


def logs_dir() -> str:
    """cornac.Experiment(save_dir=...) wants a string path; ensure it exists."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return str(LOGS_DIR)
