"""
Verify that fitting logistic regression on baseline comorbidity indices
does NOT change the ranking-based and threshold-optimized metrics.

Since LR is a monotonic transformation (prob = sigmoid(a*score + b)):
  - AUC should be IDENTICAL (rank-based)
  - F1, F2, precision, recall, accuracy at Youden-optimal threshold
    should be IDENTICAL (same confusion matrix, just different threshold value)

Uses a small stratified sample of the 2021-2022 test set for speed.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score, accuracy_score
import pickle
import tensorflow as tf

from config import DATA_DIR, NRD_2021_TEST, NRD_2022_TEST

# ============================================
# CONFIGURATION
# ============================================

# Outcome variable: 'MOR30' or 'REA30'
OUTCOME_VAR = 'REA30'

# Test data paths
TEST_2021_PATH = NRD_2021_TEST
TEST_2022_PATH = NRD_2022_TEST

# Validation data (for Youden threshold on raw scores & LR scores)
if OUTCOME_VAR == 'MOR30':
    VAL_X_PATH = DATA_DIR / 'mort_nodie' / 'X_test.csv'
    VAL_Y_PATH = DATA_DIR / 'mort_nodie' / 'y_test.csv'
elif OUTCOME_VAR == 'REA30':
    VAL_X_PATH = DATA_DIR / 'readmit' / 'X_test.csv'
    VAL_Y_PATH = DATA_DIR / 'readmit' / 'y_test.csv'

# Small sample fraction for test set
TEST_SAMPLE_FRACTION = 0.02  # 2% for quick verification

# Baseline indices
BASELINE_INDICES = ['INDEX_MORTALITY', 'INDEX_READMISSION',
                    'CHARLINDEX', 'CHARLINDEX_AGE_ADJUST']

# Tolerance for floating-point comparison
TOLERANCE = 1e-10


# ============================================
# HELPERS
# ============================================

def find_youden_threshold(y_true, scores):
    """Find optimal threshold using Youden's J (TPR - FPR)."""
    fpr, tpr, thr = roc_curve(y_true, scores)
    mask = np.isfinite(thr)
    youden = tpr[mask] - fpr[mask]
    i = int(np.argmax(youden))
    return float(thr[mask][i])


def compute_metrics(y_true, y_pred_binary):
    """Compute accuracy, precision, recall, F1, F2 from binary predictions."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    beta2 = 4.0
    f2 = ((1 + beta2) * precision * recall) / (beta2 * precision + recall) \
         if (beta2 * precision + recall) > 0 else 0.0
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f2': f2,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
    }


def stratified_pick(y, frac, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    picked = []
    for cls in np.unique(y):
        cls_idx = idx[y == cls]
        n = max(1, int(np.floor(frac * len(cls_idx))))
        picked.append(rng.choice(cls_idx, size=n, replace=False))
    return np.concatenate(picked)


def load_test_sample():
    """Load a small stratified sample of 2021-2022 test data with index columns intact."""
    print("Loading 2021-2022 test data...")
    df1 = pd.read_csv(TEST_2021_PATH)
    df2 = pd.read_csv(TEST_2022_PATH)
    test_data = pd.concat([df1, df2], ignore_index=True)
    test_data.columns = test_data.columns.str.upper()

    # Filter DIED==1 (same as other scripts for consistency)
    if 'DIED' in test_data.columns:
        test_data = test_data[test_data['DIED'] != 1]
    test_data = test_data.dropna(subset=[OUTCOME_VAR])

    print(f"  Full test set: {len(test_data):,} samples")

    # Stratified sample
    y = test_data[OUTCOME_VAR].to_numpy(dtype=np.int32)
    idx = stratified_pick(y, frac=TEST_SAMPLE_FRACTION, seed=42)
    sample = test_data.iloc[idx].copy()
    print(f"  Sampled: {len(sample):,} samples ({TEST_SAMPLE_FRACTION*100:.0f}%)")
    print(f"  Positive rate: {np.mean(sample[OUTCOME_VAR]):.4f}")
    return sample


# ============================================
# MAIN
# ============================================

def main():
    print("=" * 70)
    print(f"VERIFYING LR TRANSFORMATION PRESERVES METRICS — {OUTCOME_VAR}")
    print("=" * 70)

    # Load validation data for Youden threshold fitting
    print("\nLoading validation data for threshold fitting...")
    X_val = pd.read_csv(VAL_X_PATH)
    y_val = pd.read_csv(VAL_Y_PATH).values.ravel().astype(np.int32)
    print(f"  Validation set: {len(y_val):,} samples")

    # Load small test sample
    test_sample = load_test_sample()
    y_test = test_sample[OUTCOME_VAR].to_numpy(dtype=np.int32)

    all_checks_pass = True

    for index_name in BASELINE_INDICES:
        print(f"\n{'='*60}")
        print(f"  {index_name}")
        print(f"{'='*60}")

        # Validate presence
        if index_name not in test_sample.columns:
            print(f"  SKIP: not in test data")
            continue
        if index_name not in X_val.columns:
            print(f"  SKIP: not in validation data")
            continue

        # Load fitted LR model
        lr_path = f'Baselines/lr_{index_name.lower()}_{OUTCOME_VAR.lower()}.pkl'
        try:
            with open(lr_path, 'rb') as f:
                lr_model = pickle.load(f)
            print(f"  Loaded LR: {lr_path}")
            print(f"    Coefficients: a={lr_model.coef_[0][0]:.6f}, b={lr_model.intercept_[0]:.6f}")
        except FileNotFoundError:
            print(f"  SKIP: LR model not found at {lr_path}")
            continue

        # ---- Raw scores ----
        val_raw = X_val[index_name].to_numpy(dtype=np.float64)
        val_mask = ~np.isnan(val_raw)
        val_raw = val_raw[val_mask]
        y_val_clean = y_val[val_mask]

        test_raw = test_sample[index_name].to_numpy(dtype=np.float64)
        test_mask = ~np.isnan(test_raw)
        test_raw = test_raw[test_mask]
        y_test_clean = y_test[test_mask]

        # ---- LR-transformed scores ----
        val_lr = lr_model.predict_proba(val_raw.reshape(-1, 1))[:, 1]
        test_lr = lr_model.predict_proba(test_raw.reshape(-1, 1))[:, 1]

        print(f"\n  Test samples (non-NaN): {len(test_raw):,}")
        print(f"  Raw score range [test]: [{test_raw.min():.2f}, {test_raw.max():.2f}]")
        print(f"  LR prob range  [test]: [{test_lr.min():.6f}, {test_lr.max():.6f}]")

        # ---- AUC comparison ----
        auc_raw = roc_auc_score(y_test_clean, test_raw)
        auc_lr = roc_auc_score(y_test_clean, test_lr)

        print(f"\n  AUC (raw scores):       {auc_raw:.10f}")
        print(f"  AUC (LR probabilities): {auc_lr:.10f}")
        print(f"  Difference:             {abs(auc_raw - auc_lr):.2e}")

        auc_match = abs(auc_raw - auc_lr) < TOLERANCE
        print(f"  {'PASS' if auc_match else 'FAIL'}: AUC preserved")
        if not auc_match:
            all_checks_pass = False

        # ---- Threshold-optimized metrics comparison ----
        # Fit Youden threshold on validation set, apply on test set
        thr_raw = find_youden_threshold(y_val_clean, val_raw)
        thr_lr = find_youden_threshold(y_val_clean, val_lr)

        print(f"\n  Youden threshold (raw): {thr_raw:.6f}")
        print(f"  Youden threshold (LR):  {thr_lr:.6f}")

        pred_raw = (test_raw > thr_raw).astype(np.int8)
        pred_lr = (test_lr > thr_lr).astype(np.int8)

        m_raw = compute_metrics(y_test_clean, pred_raw)
        m_lr = compute_metrics(y_test_clean, pred_lr)

        print(f"\n  {'Metric':<12} {'Raw':>12} {'LR':>12} {'Diff':>12}")
        print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        for key in ['accuracy', 'precision', 'recall', 'f1', 'f2']:
            diff = abs(m_raw[key] - m_lr[key])
            flag = '' if diff < TOLERANCE else '  <-- DIFFERS'
            print(f"  {key:<12} {m_raw[key]:>12.6f} {m_lr[key]:>12.6f} {diff:>12.2e}{flag}")

        # Confusion matrix sanity check
        tp_match = (m_raw['tp'] == m_lr['tp']) and (m_raw['fp'] == m_lr['fp']) \
                   and (m_raw['fn'] == m_lr['fn']) and (m_raw['tn'] == m_lr['tn'])
        print(f"\n  Confusion matrix (raw): TP={m_raw['tp']} FP={m_raw['fp']} FN={m_raw['fn']} TN={m_raw['tn']}")
        print(f"  Confusion matrix (LR):  TP={m_lr['tp']}  FP={m_lr['fp']}  FN={m_lr['fn']}  TN={m_lr['tn']}")
        print(f"  {'PASS' if tp_match else 'FAIL'}: Confusion matrix identical")
        if not tp_match:
            all_checks_pass = False

    print("\n" + "=" * 70)
    print(f"OVERALL: {'ALL CHECKS PASSED' if all_checks_pass else 'SOME CHECKS FAILED'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
