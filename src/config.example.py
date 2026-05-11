"""Project paths and constants.

Copy this file to ``config.py`` and edit the values for your environment.
``config.py`` is gitignored so your local paths stay out of version control.

Convention: anywhere a script needs a path, it should ``from config import ...``
rather than hardcoding strings.
"""
from pathlib import Path

# config.py lives at src/config.py — repo root is one level up
REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Raw NRD data — request access from https://hcup-us.ahrq.gov/
# ---------------------------------------------------------------------------
# Pooled 2016-2020 NRD CSV used by preprocessing.py
NRD_RAW_CSV = "/path/to/NRD_2016_2020.csv"

# ---------------------------------------------------------------------------
# Preprocessed train/test splits, organized by outcome
# ---------------------------------------------------------------------------
# Expected layout:
#   <DATA_DIR>/<outcome>/X_train_downsampled.csv
#   <DATA_DIR>/<outcome>/y_train_downsampled.csv
#   <DATA_DIR>/<outcome>/X_test.csv
#   <DATA_DIR>/<outcome>/y_test.csv
# where <outcome> is one of {mort, mort_nodie, readmit}.
DATA_DIR = REPO_ROOT / "data"

# ---------------------------------------------------------------------------
# External validation data (held out)
# ---------------------------------------------------------------------------
NRD_2021_TEST = DATA_DIR / "NRD_2021_test.csv"
NRD_2022_TEST = DATA_DIR / "NRD_2022_test.csv"

# ---------------------------------------------------------------------------
# Trained model + encoder artifacts
# ---------------------------------------------------------------------------
MODEL_DIR = REPO_ROOT / "Model"
BASELINES_DIR = REPO_ROOT / "Baselines"
LABEL_ENCODER_PATH = MODEL_DIR / "full_label_encoder.pkl"
AGE_SCALER_PATH = MODEL_DIR / "full_age_scaler.pkl"

# ---------------------------------------------------------------------------
# ICD-10 pretrained embeddings (cui2vec-style)
# ---------------------------------------------------------------------------
EMBEDDINGS_DIR = REPO_ROOT / "embeddings"

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
RESULTS_DIR = REPO_ROOT / "results"
LOGS_DIR = REPO_ROOT / "logs"

FIGURES_DIR = RESULTS_DIR / "figures"
FEATURE_IMPORTANCE_DIR = RESULTS_DIR / "feature_importance"
FEATURE_IMPORTANCE_FIG_DIR = FIGURES_DIR / "feature_importance"
DELONG_RESULTS_DIR = RESULTS_DIR / "delong"
SMALL_DATASET_DIR = RESULTS_DIR / "small_dataset"
PREDICTIONS_CSV = REPO_ROOT / "predictions.csv"

# ---------------------------------------------------------------------------
# Default outcome variable: 'DIED', 'MOR30', or 'REA30'
# ---------------------------------------------------------------------------
OUTCOME = "MOR30"
