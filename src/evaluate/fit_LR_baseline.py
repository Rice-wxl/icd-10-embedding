"""
Fit logistic regression models for each baseline comorbidity index.

Uses the same downsampled training data as the deep learning model.
Each LR learns: P(outcome=1) = sigmoid(a * score + b), which maps
raw integer scores to calibrated probabilities.

Since training data is downsampled (balanced 50/50), the LR predictions
need beta recalibration at inference time — same as the ICD model.

Saves fitted LR models to Model/ as pickle files.
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pickle

from config import OUTCOME, OUTCOME_LOWER, OUTCOME_DATA_DIR, BASELINES_DIR

# ============================================
# CONFIGURATION
# ============================================

# Outcome + data paths come from config.py — change OUTCOME there.
OUTCOME_VAR = OUTCOME

# Paths to training data (downsampled) — same as hyper_tune.py / transformer.py
TRAIN_X_PATH = OUTCOME_DATA_DIR / 'X_train_downsampled.csv'
TRAIN_Y_PATH = OUTCOME_DATA_DIR / 'y_train_downsampled.csv'
VAL_X_PATH = OUTCOME_DATA_DIR / 'X_test.csv'
VAL_Y_PATH = OUTCOME_DATA_DIR / 'y_test.csv'

# Baseline indices to fit
BASELINE_INDICES = ['INDEX_MORTALITY', 'INDEX_READMISSION',
                    'CHARLINDEX', 'CHARLINDEX_AGE_ADJUST']

# ============================================
# MAIN
# ============================================

def main():
    print("=" * 70)
    print(f"FITTING LOGISTIC REGRESSION BASELINES — {OUTCOME_VAR}")
    print("=" * 70)

    # Load preprocessed training data (same data used for ICD model training)
    print("\nLoading training data...")
    X_train = pd.read_csv(TRAIN_X_PATH)
    y_train = pd.read_csv(TRAIN_Y_PATH).values.ravel()
    print(f"  Training set: {X_train.shape[0]:,} samples")
    print(f"  Positive rate: {np.mean(y_train):.4f}")

    # Load validation data
    print("\nLoading validation data...")
    X_val = pd.read_csv(VAL_X_PATH)
    y_val = pd.read_csv(VAL_Y_PATH).values.ravel()
    print(f"  Validation set: {X_val.shape[0]:,} samples")
    print(f"  Positive rate: {np.mean(y_val):.4f}")

    for index_name in BASELINE_INDICES:
        print(f"\n{'='*50}")
        print(f"  Fitting LR for: {index_name}")
        print(f"{'='*50}")

        if index_name not in X_train.columns:
            print(f"  WARNING: {index_name} not in training data. Skipping.")
            continue

        # Extract scores, drop NaN
        train_scores = X_train[index_name].to_numpy(dtype=np.float64)
        train_valid = ~np.isnan(train_scores)
        X_fit = train_scores[train_valid].reshape(-1, 1)
        y_fit = y_train[train_valid]

        print(f"  Training samples (non-NaN): {len(X_fit):,}")
        print(f"  Score range: [{X_fit.min():.1f}, {X_fit.max():.1f}]")
        print(f"  Unique score values: {len(np.unique(X_fit))}")

        # Fit logistic regression: P(outcome) = sigmoid(a * score + b)
        lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        lr.fit(X_fit, y_fit)

        print(f"  Coefficients: a={lr.coef_[0][0]:.6f}, b={lr.intercept_[0]:.6f}")

        # Training AUC
        train_probs = lr.predict_proba(X_fit)[:, 1]
        train_auc = roc_auc_score(y_fit, train_probs)
        print(f"  Training AUC: {train_auc:.4f}")

        # Validation AUC
        if index_name in X_val.columns:
            val_scores = X_val[index_name].to_numpy(dtype=np.float64)
            val_valid = ~np.isnan(val_scores)
            X_val_fit = val_scores[val_valid].reshape(-1, 1)
            y_val_fit = y_val[val_valid]

            val_probs = lr.predict_proba(X_val_fit)[:, 1]
            val_auc = roc_auc_score(y_val_fit, val_probs)
            print(f"  Validation AUC: {val_auc:.4f}")

            # Show probability range on validation set
            print(f"  Validation prob range: [{val_probs.min():.6f}, {val_probs.max():.6f}]")

        # Save the fitted LR model
        BASELINES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = BASELINES_DIR / f'lr_{index_name.lower()}_{OUTCOME_LOWER}.pkl'
        with open(out_path, 'wb') as f:
            pickle.dump(lr, f)
        print(f"  Saved to: {out_path}")

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
