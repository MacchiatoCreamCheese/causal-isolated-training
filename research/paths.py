import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

DATA_DIR = Path(os.environ.get("RESEARCH_DATA_DIR", PROJECT_ROOT))
OUTPUT_DIR = Path(os.environ.get("RESEARCH_OUTPUT_DIR", PACKAGE_DIR))

RESULTS_DIR = OUTPUT_DIR / "results"
LOGS_DIR = OUTPUT_DIR / "logs"
FIGURES_DIR = OUTPUT_DIR / "figures"

REPORTS_DIR = PACKAGE_DIR / "reports"


def data_path(*parts) -> str:
    return str(DATA_DIR.joinpath(*parts))


def results_path(*parts) -> Path:
    return RESULTS_DIR.joinpath(*parts)


def logs_dir() -> str:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return str(LOGS_DIR)
