# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a deep learning research project for predicting healthcare outcomes (mortality and hospital readmissions) using ICD-10 diagnosis codes from the National Readmission Database (NRD). The project uses TensorFlow/Keras with custom neural network architectures including DeepSet aggregation and optional Transformer blocks to model sets of diagnosis codes.

## Environment Setup

This project runs on the Oscar HPC cluster at Brown University.

### Conda Environment
```bash
# Create environment from file
conda env create -f environment.yml
# or minimal version
conda env create -f env_min.yml

# Activate environment
conda activate icd_gpu
```

Dependencies: Python 3.10, TensorFlow 2.15.0, Keras 2.15.0, pandas, numpy, scikit-learn, tqdm

### Using Apptainer (Singularity)
The project also uses a pre-built Apptainer image:
```bash
apptainer exec --nv tensorflow-24.03-tf2-py3.simg python <script.py>
```

## Core Pipeline

The project follows a standard ML pipeline with three main stages:

### 1. Data Preprocessing (`preprocessing.py`)
- Loads NRD data (2016-2020) from `/users/xwang259/scratch/NRD_2016_2020.csv`
- Encodes ICD-10 diagnosis codes using LabelEncoder
- Normalizes AGE using MinMaxScaler
- One-hot encodes PAY1 (payer) and ZIPINC_QRTL (income quartile)
- Handles missing values: replaces -8/-9 codes with NaN for PAY1 and ZIPINC_QRTL
- Filters patients where DIED==1 when predicting readmission (REA30)
- Performs stratified train/test split and downsampling to balance classes
- Saves preprocessed datasets and encoders to `/users/xwang259/scratch/NRD/` or `/users/xwang259/icd/data/`

**Run via SLURM:**
```bash
sbatch preprocessing.sh
```

### 2. Model Training (`transformer.py`)
- Builds a deep learning model with:
  - Embedding layer for 40 ICD-10 diagnosis codes (I10_DX1-I10_DX40)
  - DeepSet aggregation (phi/rho networks) for set-based ICD code modeling
  - Optional TransformerBlock layers (currently disabled in production)
  - Demographic feature processing (AGE, FEMALE, PAY1, ZIPINC_QRTL)
  - Multiple dense layers with batch normalization and dropout
  - Binary classification output (sigmoid activation)
- Metrics: AUC, Precision, Recall, F2 Score
- Uses early stopping on validation AUC (patience=2)
- Saves trained models to `Model/` directory as `.keras` files

**Run via SLURM:**
```bash
sbatch run.sh
```

### 3. Model Evaluation (`evaluate.py`)
- Loads trained models from `Model/` directory
- Tests on held-out 2021-2022 NRD data
- Finds optimal classification threshold using Youden index on validation set
- Compares against traditional clinical risk scores:
  - ECI (Elixhauser Comorbidity Index) - INDEX_MORTALITY or INDEX_READMISSION
  - CCI (Charlson Comorbidity Index) - CHARLINDEX
  - CCI Age Adjusted - CHARLINDEX_AGE_ADJUST
- Generates ROC curves comparing model to baselines
- Computes AUC with 95% CI using Hanley-McNeil method
- Outputs comprehensive metrics and comparison graphs

**Run via SLURM:**
```bash
sbatch evaluate.sh
```

## Additional Components

### Hyperparameter Tuning (`hyper_tune.py`)
- Uses Keras Tuner (RandomSearch) to optimize model architecture
- Tunable parameters:
  - Embedding dimension (32-64)
  - DeepSet hidden dimensions (128-512)
  - Number of encoding/decoding layers (1-3)
  - MLP layer sizes and dropout rates
  - Learning rate
- Saves best model to `Model/` with naming convention `*_hypertrial_auc.keras`

### Model Interpretation (`IG.py`)
- Implements Integrated Gradients (IG) for feature importance analysis
- Identifies top ICD codes by:
  - Global impact (absolute total)
  - Per-occurrence effect (mean)
  - Positive vs negative contributions (signed values)
- Uses stratified sampling for computational efficiency
- Supports different baselines: "pad", "zero", "mean"
- Exports CSV files with top contributing diagnosis codes

**Run via SLURM:**
```bash
sbatch interpretation.sh
```

## Custom Keras Components

The codebase defines three custom serializable Keras components that must be registered:

1. **DeepSet** (custom Model): Permutation-invariant set aggregation with phi (encoder) and rho (decoder) networks. Takes ICD embeddings, applies element-wise transformation, aggregates via sum, then applies post-aggregation transformation.

2. **TransformerBlock** (custom Layer): Multi-head self-attention with feed-forward network, layer normalization, and residual connections. Currently optional/disabled in production models.

3. **F2Score** (custom Metric): Computes F2 score (weights recall higher than precision) with configurable threshold (default 0.5).

When loading models, all three components must be properly registered with `@tf.keras.utils.register_keras_serializable(package="Custom")`.

## Key File Paths

**Data locations:**
- Training data: `/users/xwang259/scratch/NRD/[outcome]/X_train_downsampled.csv`, `y_train_downsampled.csv`
- Test data: `/users/xwang259/scratch/NRD/[outcome]/X_test.csv`, `y_test.csv`
- External validation: `/users/xwang259/scratch/NRD/NRD_2021_test.csv`, `NRD_2022_test.csv`

**Model artifacts:**
- Saved models: `Model/*.keras`
- Encoders: `Model/full_label_encoder.pkl`, `Model/full_age_scaler.pkl`

**Outcomes:**
- `DIED`: In-hospital mortality
- `MOR30`: 30-day mortality
- `REA30`: 30-day readmission

## SLURM Job Configuration

All scripts use SLURM batch files with standard configurations:
- **Preprocessing**: `--partition bigmem --mem=256G` (CPU only, 24 hours)
- **Training**: `--partition gpu --gres=gpu:1 --mem=96G` (24 hours)
- **Evaluation**: `--partition gpu --gres=gpu:1 --mem=96G` (12 hours)
- **Interpretation**: `--partition gpu --gres=gpu:1 --mem=96G` (varies)

All jobs bind paths using: `/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data`

## Model Architecture Notes

- ICD codes are processed as sets (permutation-invariant) using DeepSet aggregation
- The DeepSet architecture explicitly handles variable-length diagnosis lists (up to 40 codes)
- Models can be configured with or without demographic features by modifying input layers
- Current production models use ICD codes + demographics with DeepSet aggregation (no Transformer blocks)
- F2 score is used because it weights recall higher than precision, appropriate for healthcare outcomes
- Class imbalance is handled via downsampling the majority class to match minority class size

## Common Tasks

**To train a model for a different outcome:**
1. Modify outcome variable in `preprocessing.py` (line 83): `outcome_var = 'DIED'` or `'MOR30'` or `'REA30'`
2. Update data paths in `transformer.py` (lines 19-22) to match preprocessed outcome directory
3. Update model save path in `transformer.py` (line 346)

**To run the complete pipeline:**
```bash
sbatch preprocessing.sh      # Preprocess data
sbatch run.sh                 # Train model
sbatch evaluate.sh            # Evaluate and compare to baselines
```

**To interpret a trained model:**
1. Update model path in `IG.py` (line 181)
2. Update outcome variable in `IG.py` (line 209)
3. Run: `sbatch interpretation.sh`
