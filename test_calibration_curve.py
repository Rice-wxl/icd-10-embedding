"""
Test script for calibration_curve.py using small test dataset.
This validates the binning and plotting functions before running on large datasets.

Usage:
    python test_calibration_curve.py
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from keras.saving import load_model
from keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention
import pickle
import sys

from config import REPO_ROOT, MODEL_DIR, SMALL_DATASET_DIR

# Add the repo root to path to import from calibration_curve
sys.path.insert(0, str(REPO_ROOT))

# Import the calibration functions
from calibration_curve import (
    compute_calibration_curve,
    compute_calibration_curve_quantile,
    plot_calibration_curves,
    plot_calibration_curves_separate,
    calibrate_probability,
    DeepSet,
    TransformerBlock,
    F2Score,
    f2_score
)

# ============================================
# CONFIGURATION
# ============================================

# Model and data paths
MODEL_PATH = MODEL_DIR / 'readmit_hypertrial_auc.keras'
TEST_DATA_PATH = SMALL_DATASET_DIR / 'small_test_dataset_mor30.csv'
OUTCOME_VAR = 'MOR30'

# Test settings
NUM_BINS = 5  # Use fewer bins for small dataset
MIN_SAMPLES_PER_BIN = 2  # Lower threshold for small dataset

print("=" * 70)
print("CALIBRATION CURVE TEST SCRIPT")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Model: {MODEL_PATH}")
print(f"  Test data: {TEST_DATA_PATH}")
print(f"  Outcome: {OUTCOME_VAR}")
print(f"  Number of bins: {NUM_BINS}")
print(f"  Min samples per bin: {MIN_SAMPLES_PER_BIN}")
print()

# ============================================
# Load model and encoders
# ============================================

print("Loading model and encoders...")
try:
    model = load_model(MODEL_PATH)
    print(f"  ✓ Model loaded: {model.name}")
except Exception as e:
    print(f"  ✗ Error loading model: {e}")
    sys.exit(1)

try:
    with open('Model/full_label_encoder.pkl', 'rb') as file:
        encoder = pickle.load(file)
    with open('Model/full_age_scaler.pkl', 'rb') as file:
        age_scaler = pickle.load(file)
    print(f"  ✓ Encoders loaded ({len(encoder.classes_)} ICD codes)")
except Exception as e:
    print(f"  ✗ Error loading encoders: {e}")
    sys.exit(1)

# ============================================
# Load and preprocess test data
# ============================================

print("\nLoading test data...")
try:
    test_data_original = pd.read_csv(TEST_DATA_PATH)
    print(f"  ✓ Loaded {len(test_data_original)} samples")
except Exception as e:
    print(f"  ✗ Error loading data: {e}")
    sys.exit(1)

# Ensure uppercase column names
test_data_original.columns = test_data_original.columns.str.upper()
test_data = test_data_original.copy()

# Define ICD columns
icd_columns = [f'I10_DX{i}' for i in range(1, 41)]

print("\nPreprocessing data...")

# 1. Encode ICD codes
label_to_int = {label: idx for idx, label in enumerate(encoder.classes_)}
unknown_label_int = encoder.transform(["NAN"])[0]

for col in icd_columns:
    test_data[col] = test_data[col].astype(str).str.upper()
    test_data[col] = test_data[col].map(label_to_int).fillna(unknown_label_int).astype(int)

# 2. Normalize AGE
test_data['AGE'] = age_scaler.transform(test_data[['AGE']])

# 3. Handle missing value codes
test_data['PAY1'] = test_data['PAY1'].replace([-8, -9], np.nan)
test_data['ZIPINC_QRTL'] = test_data['ZIPINC_QRTL'].replace([-8, -9], np.nan)

# 4. One-hot encode
test_data = pd.get_dummies(test_data, columns=['PAY1', 'ZIPINC_QRTL'],
                            prefix=['PAY1', 'ZIPINC_QRTL'])

# 5. Ensure all expected columns are present
expected_pay1_columns = ['PAY1_1.0', 'PAY1_2.0', 'PAY1_3.0', 'PAY1_4.0', 'PAY1_5.0', 'PAY1_6.0']
expected_zipinc_columns = ['ZIPINC_QRTL_1.0', 'ZIPINC_QRTL_2.0', 'ZIPINC_QRTL_3.0', 'ZIPINC_QRTL_4.0']

for col in expected_pay1_columns:
    if col not in test_data.columns:
        test_data[col] = 0

for col in expected_zipinc_columns:
    if col not in test_data.columns:
        test_data[col] = 0

pay1_columns = expected_pay1_columns
zipinc_qrtl_columns = expected_zipinc_columns

# 6. Extract features
X_test = test_data[['AGE', 'FEMALE'] + pay1_columns + zipinc_qrtl_columns + icd_columns]
X_test = X_test.dropna()
test_data = test_data.loc[X_test.index]

print(f"  ✓ Preprocessed: {len(X_test)} samples")

# Get true labels
if OUTCOME_VAR not in test_data.columns:
    print(f"  ✗ Error: {OUTCOME_VAR} not found in dataset")
    sys.exit(1)

y_test = test_data[OUTCOME_VAR].to_numpy(np.int32)
print(f"  ✓ Outcome variable: {OUTCOME_VAR}")
print(f"     Positive samples: {y_test.sum()}")
print(f"     Negative samples: {len(y_test) - y_test.sum()}")

# ============================================
# Make predictions
# ============================================

print("\nMaking predictions...")
try:
    inputs = [
        X_test[icd_columns],
        X_test['AGE'],
        X_test['FEMALE'],
    ] + [X_test[c] for c in pay1_columns] \
      + [X_test[c] for c in zipinc_qrtl_columns]

    y_pred = model.predict(inputs, batch_size=32, verbose=0).squeeze()
    print(f"  ✓ Predictions generated")
    print(f"     Min probability: {y_pred.min():.4f}")
    print(f"     Max probability: {y_pred.max():.4f}")
    print(f"     Mean probability: {y_pred.mean():.4f}")
except Exception as e:
    print(f"  ✗ Error making predictions: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================
# Test calibration probability adjustment
# ============================================

print("\nTesting calibration probability adjustment...")
try:
    # For MOR30
    train_num_posi = 272069
    train_num_nega = 70180061
    beta = train_num_posi / train_num_nega

    y_pred_calibrated = calibrate_probability(y_pred, beta)
    print(f"  ✓ Calibration applied (beta={beta:.6f})")
except Exception as e:
    print(f"  ✗ Error in calibration: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Split into "train" and "test" for calibration curve testing
# Use first half as "train", second half as "test"
n_samples = len(y_test)
split_idx = n_samples // 2

y_train_true = y_test[:split_idx]
y_train_pred = y_pred[:split_idx]

y_test_true = y_test[split_idx:]
y_test_pred_calibrated = y_pred_calibrated[split_idx:]

print(f"\nSplit data for testing:")
print(f"  'Training' set: {len(y_train_true)} samples")
print(f"  'Test' set: {len(y_test_true)} samples")

# ============================================
# Test equal-width binning
# ============================================

print("\n" + "=" * 70)
print("TEST 1: Equal-width binning")
print("=" * 70)

try:
    train_pred, train_actual, train_counts = compute_calibration_curve(
        y_train_true, y_train_pred, num_bins=NUM_BINS, min_samples=MIN_SAMPLES_PER_BIN
    )

    print(f"✓ Equal-width binning successful")
    print(f"  Number of bins created: {len(train_pred)}")
    for i, (pred, actual, count) in enumerate(zip(train_pred, train_actual, train_counts)):
        print(f"    Bin {i+1}: Predicted={pred:.4f}, Actual={actual:.4f}, N={count}")
except Exception as e:
    print(f"✗ Error in equal-width binning: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================
# Test quantile binning
# ============================================

print("\n" + "=" * 70)
print("TEST 2: Quantile binning")
print("=" * 70)

try:
    train_pred_q, train_actual_q, train_counts_q = compute_calibration_curve_quantile(
        y_train_true, y_train_pred, num_bins=NUM_BINS, min_samples=MIN_SAMPLES_PER_BIN
    )

    print(f"✓ Quantile binning successful")
    print(f"  Number of bins created: {len(train_pred_q)}")
    for i, (pred, actual, count) in enumerate(zip(train_pred_q, train_actual_q, train_counts_q)):
        print(f"    Bin {i+1}: Mean Predicted={pred:.4f}, Actual={actual:.4f}, N={count}")
except Exception as e:
    print(f"✗ Error in quantile binning: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Also compute for "test" set
try:
    test_pred_q, test_actual_q, test_counts_q = compute_calibration_curve_quantile(
        y_test_true, y_test_pred_calibrated, num_bins=NUM_BINS, min_samples=MIN_SAMPLES_PER_BIN
    )
    print(f"\n✓ Quantile binning for 'test' set successful")
    print(f"  Number of bins created: {len(test_pred_q)}")
except Exception as e:
    print(f"✗ Error in quantile binning for test set: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================
# Test combined plotting
# ============================================

print("\n" + "=" * 70)
print("TEST 3: Combined plotting")
print("=" * 70)

try:
    plot_calibration_curves(
        train_pred_q, train_actual_q, train_counts_q,
        test_pred_q, test_actual_q, test_counts_q,
        OUTCOME_VAR, 'test_calibration_combined.png'
    )
    print(f"✓ Combined plot created: test_calibration_combined.png")
except Exception as e:
    print(f"✗ Error in combined plotting: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================
# Test separate plotting
# ============================================

print("\n" + "=" * 70)
print("TEST 4: Separate plotting")
print("=" * 70)

try:
    plot_calibration_curves_separate(
        train_pred_q, train_actual_q, train_counts_q,
        test_pred_q, test_actual_q, test_counts_q,
        OUTCOME_VAR, 'test_calibration_separate'
    )
    print(f"✓ Separate plots created:")
    print(f"    test_calibration_separate_train.png")
    print(f"    test_calibration_separate_test.png")
except Exception as e:
    print(f"✗ Error in separate plotting: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================
# Summary
# ============================================

print("\n" + "=" * 70)
print("ALL TESTS PASSED!")
print("=" * 70)
print("\nSummary:")
print(f"  ✓ Model loading and prediction")
print(f"  ✓ Probability calibration adjustment")
print(f"  ✓ Equal-width binning")
print(f"  ✓ Quantile binning")
print(f"  ✓ Combined plotting")
print(f"  ✓ Separate plotting")
print("\nThe calibration_curve.py functions are working correctly!")
print("You can now run the full calibration_curve.py on your complete dataset.")
print()
