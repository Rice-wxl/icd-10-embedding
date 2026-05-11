"""
Model Evaluation Script

This script evaluates trained models on validation and test datasets:
1. Loads validation set to find optimal Youden threshold
2. Loads 2021-2022 test data and preprocesses it
3. Generates predictions and computes metrics
4. Compares against traditional clinical risk scores (ECI, CCI)
5. Generates ROC curves
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from keras.saving import load_model
from keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import pickle
from scipy import stats

from config import (
    OUTCOME, OUTCOME_LOWER, OUTCOME_DATA_DIR, MODEL_PATH,
    LABEL_ENCODER_PATH, AGE_SCALER_PATH,
    NRD_2021_TEST, NRD_2022_TEST, FIGURES_DIR,
)

# ============================================
# CONFIGURATION
# ============================================

# Input features flag
# Set to True to use only ICD codes as inputs (no demographics)
# Set to False to use all inputs (ICD codes + AGE + FEMALE + PAY1 + ZIPINC_QRTL)
USE_ICD_ONLY = False

# Outcome + model + data paths come from config.py — change OUTCOME there.
OUTCOME_VAR = OUTCOME

# Validation data paths (for finding Youden threshold)
VALIDATION_X_PATH = OUTCOME_DATA_DIR / 'X_test.csv'
VALIDATION_Y_PATH = OUTCOME_DATA_DIR / 'y_test.csv'

# Test data paths (2021-2022 actual test data)
TEST_2021_PATH = NRD_2021_TEST
TEST_2022_PATH = NRD_2022_TEST

# Test data sampling (to reduce computational cost)
TEST_SAMPLE_FRACTION = 0.10  # Use 10% of test data (stratified)

# Output file for ROC curve
OUTPUT_PLOT = FIGURES_DIR / 'roc' / f'graph_newci_{OUTCOME_LOWER}.png'

# ============================================
# CUSTOM KERAS COMPONENTS
# ============================================

@tf.keras.utils.register_keras_serializable(package="Custom")
def f2_score(y_true, y_pred):
    """Custom F2 score function"""
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    epsilon = tf.keras.backend.epsilon()
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f2 = (5 * precision * recall) / (4 * precision + recall + epsilon)
    return f2.numpy()


@tf.keras.utils.register_keras_serializable(package="Custom")
class DeepSet(tf.keras.Model):
    """DeepSet aggregation for permutation-invariant set modeling"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_encode, num_decode, **kwargs):
        super(DeepSet, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_encode = num_encode
        self.num_decode = num_decode

        self.phi = tf.keras.Sequential([
            Dense(self.hidden_dim, activation='relu') for _ in range(self.num_encode)
        ])

        self.rho = tf.keras.Sequential([
            Dense(self.hidden_dim, activation='relu') for _ in range(self.num_decode - 1)
        ] + [Dense(self.output_dim, activation='relu')])

    def call(self, x):
        transformed = self.phi(x)
        aggregated = tf.reduce_sum(transformed, axis=1)
        output = self.rho(aggregated)
        return output

    def get_config(self):
        config = super(DeepSet, self).get_config()
        config.update({
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_encode": self.num_encode,
            "num_decode": self.num_decode
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="Custom")
class TransformerBlock(tf.keras.layers.Layer):
    """Transformer encoder block with multi-head attention"""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="Custom")
class F2Score(tf.keras.metrics.Metric):
    """F2 score metric (weights recall higher than precision)"""
    def __init__(self, name='f2_score', threshold=0.5, **kwargs):
        super(F2Score, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')
        self.epsilon = tf.keras.backend.epsilon()
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        self.tp.assign_add(tf.reduce_sum(y_true * y_pred))
        self.fp.assign_add(tf.reduce_sum((1 - y_true) * y_pred))
        self.fn.assign_add(tf.reduce_sum(y_true * (1 - y_pred)))

    def result(self):
        precision = self.tp / (self.tp + self.fp + self.epsilon)
        recall = self.tp / (self.tp + self.fn + self.epsilon)
        f2 = (5 * precision * recall) / (4 * precision + recall + self.epsilon)
        return f2

    def reset_state(self, sample_weight=None):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)

    def get_config(self):
        config = super(F2Score, self).get_config()
        config.update({'name': self.name, 'threshold': self.threshold})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ============================================
# DATA PREPROCESSING FUNCTIONS
# ============================================

def preprocess_test_data(test_2021_path, test_2022_path, encoder, age_scaler,
                         outcome_var, sample_fraction=0.10):
    """
    Preprocess 2021-2022 test data with ICD encoding, normalization, and one-hot encoding.

    Args:
        test_2021_path: Path to 2021 test CSV
        test_2022_path: Path to 2022 test CSV
        encoder: LabelEncoder for ICD codes
        age_scaler: MinMaxScaler for age
        outcome_var: Outcome variable name (e.g., 'MOR30', 'REA30')
        sample_fraction: Fraction of data to keep (stratified sampling)

    Returns:
        X_test: Preprocessed features DataFrame
        y_test: True labels array
        test_data: Full test DataFrame including comorbidity indices
    """
    print(f"\nPreprocessing test data...")

    # Load and combine 2021-2022 data
    print("  Loading 2021 and 2022 files...")
    df1 = pd.read_csv(test_2021_path)
    df2 = pd.read_csv(test_2022_path)
    test_data = pd.concat([df1, df2], ignore_index=True)
    print(f"  Combined shape: {test_data.shape}")

    # Standardize column names
    test_data.columns = test_data.columns.str.upper()

    # Filter out DIED==1 patients (for consistency across outcomes)
    if 'DIED' in test_data.columns:
        test_data = test_data[test_data['DIED'] != 1]
        print(f"  After filtering DIED==1: {test_data.shape}")
    
    # Remove missing outcome values
    test_data = test_data.dropna(subset=[outcome_var])

    # Define ICD columns
    icd_columns = [f'I10_DX{i}' for i in range(1, 41)]

    # Encode ICD codes
    print("  Encoding ICD codes...")
    label_to_int = {label: idx for idx, label in enumerate(encoder.classes_)}
    unknown_label_int = encoder.transform(["NAN"])[0]

    for col in icd_columns:
        test_data[col] = test_data[col].astype(str).str.upper()
        test_data[col] = test_data[col].map(label_to_int).fillna(unknown_label_int).astype(int)

    # Normalize AGE
    print("  Normalizing AGE...")
    test_data['AGE'] = age_scaler.transform(test_data[['AGE']])

    # Handle missing value codes in PAY1 and ZIPINC_QRTL
    print("  Handling missing value codes...")
    test_data['PAY1'] = test_data['PAY1'].replace([-8, -9], np.nan)
    test_data['ZIPINC_QRTL'] = test_data['ZIPINC_QRTL'].replace([-8, -9], np.nan)

    # One-hot encode categorical variables
    print("  One-hot encoding PAY1 and ZIPINC_QRTL...")
    test_data = pd.get_dummies(test_data, columns=['PAY1', 'ZIPINC_QRTL'],
                                prefix=['PAY1', 'ZIPINC_QRTL'])

    # Ensure all expected columns exist (add missing ones with zeros)
    expected_pay1_cols = ['PAY1_1.0', 'PAY1_2.0', 'PAY1_3.0', 'PAY1_4.0', 'PAY1_5.0', 'PAY1_6.0']
    expected_zipinc_cols = ['ZIPINC_QRTL_1.0', 'ZIPINC_QRTL_2.0', 'ZIPINC_QRTL_3.0', 'ZIPINC_QRTL_4.0']

    for col in expected_pay1_cols + expected_zipinc_cols:
        if col not in test_data.columns:
            test_data[col] = 0

    pay1_columns = expected_pay1_cols
    zipinc_qrtl_columns = expected_zipinc_cols

    # Extract features and drop rows with missing values
    X_test = test_data[['AGE', 'FEMALE'] + pay1_columns + zipinc_qrtl_columns + icd_columns]
    X_test = X_test.dropna()
    test_data = test_data.loc[X_test.index]

    print(f"  After preprocessing: {X_test.shape}")

    # Stratified sampling to reduce computational cost
    if sample_fraction < 1.0:
        print(f"  Applying stratified sampling ({sample_fraction*100:.0f}%)...")
        y_all = test_data[outcome_var].to_numpy()
        idx_all = np.arange(len(test_data))

        # Stratified sampling helper
        def stratified_pick(idx, y_sub, frac, seed=42):
            rng = np.random.default_rng(seed)
            picked = []
            for cls in np.unique(y_sub):
                cls_idx = idx[y_sub == cls]
                n = max(1, int(np.floor(frac * len(cls_idx))))
                pick = rng.choice(cls_idx, size=n, replace=False)
                picked.append(pick)
            return np.concatenate(picked)

        sampled_idx = stratified_pick(idx_all, y_all, frac=sample_fraction)
        test_data = test_data.iloc[sampled_idx].copy()
        X_test = test_data[['AGE', 'FEMALE'] + pay1_columns + zipinc_qrtl_columns + icd_columns]
        print(f"  After sampling: {X_test.shape}")

    y_test = test_data[outcome_var].to_numpy(np.int32)

    return X_test, y_test, test_data


def prepare_model_inputs(X, icd_columns, pay1_columns, zipinc_qrtl_columns, icd_only=False):
    """
    Prepare inputs in the format expected by the model.

    Args:
        X: DataFrame with all features
        icd_columns: List of ICD column names
        pay1_columns: List of PAY1 one-hot column names
        zipinc_qrtl_columns: List of ZIPINC_QRTL one-hot column names
        icd_only: If True, return only ICD columns; if False, return all inputs

    Returns:
        List of input tensors for the model (or single tensor if icd_only=True)
    """
    if icd_only:
        # Return only ICD columns as a single input
        return [X[icd_columns]]
    else:
        # Return all inputs: ICD codes + demographics
        return [
            X[icd_columns],
            X['AGE'],
            X['FEMALE'],
        ] + [X[c] for c in pay1_columns] \
          + [X[c] for c in zipinc_qrtl_columns]


# ============================================
# METRICS CALCULATION FUNCTIONS
# ============================================

def auc_ci_delong(y_true, y_pred_prob, ci=0.95):
    """
    Calculate AUC with confidence interval using DeLong method.

    Args:
        y_true: True binary labels
        y_pred_prob: Predicted probabilities
        ci: Confidence level (default 0.95)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    y_true = np.asarray(y_true, dtype=np.int8).reshape(-1)
    y_pred_prob = np.asarray(y_pred_prob, dtype=np.float32).reshape(-1)

    # Separate scores by class
    X = y_pred_prob[y_true == 1]  # cases (positive class)
    Y = y_pred_prob[y_true == 0]  # controls (negative class)

    m = len(X)
    n = len(Y)

    if m <= 1 or n <= 1:
        return (np.nan, np.nan)

    # Compute placement values
    V_X, V_Y, theta = _delong_placements_fast(X, Y)

    # Compute variance components
    S_X = np.sum((V_X - theta) ** 2) / (m - 1)
    S_Y = np.sum((V_Y - theta) ** 2) / (n - 1)
    S = S_X / m + S_Y / n

    # Compute confidence interval using normal approximation
    alpha = 1 - ci
    z = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(S)

    lower = theta - z * se
    upper = theta + z * se

    # Clip to [0, 1]
    lower = max(0.0, lower)
    upper = min(1.0, upper)

    return (lower, upper)


def _compute_midrank(values):
    """
    Compute midranks (average ranks for ties) for a sorted array.
    
    Example: [1, 2, 2, 3] -> [1.0, 2.5, 2.5, 4.0]
    The two 2s share ranks 2 and 3, so they both get (2+3)/2 = 2.5
    """
    n = len(values)
    midranks = np.empty(n, dtype=np.float64)
    
    i = 0
    while i < n:
        # Find the end of the current group of ties
        j = i
        while j < n and values[j] == values[i]:
            j += 1
        
        # All elements from i to j-1 are tied
        # They share ranks (i+1), (i+2), ..., j
        # Average = (i+1 + j) / 2
        avg_rank = (i + 1 + j) / 2.0
        midranks[i:j] = avg_rank
        
        i = j
    
    return midranks


def _delong_placements_fast(X, Y):
    """
    Compute DeLong placement values using ranking.
    
    V_X[i] = (overall_rank[i] - rank_among_cases[i]) / n
    V_Y[j] = (rank_among_controls[j] - overall_rank[j] + len(combined)) / m
           = (number of cases that beat control j) / m
    """
    m = len(X)
    n = len(Y)
    
    # === STEP 1: Combine all values and track their original positions ===
    # Stack cases first, then controls
    combined = np.concatenate([X, Y])
    
    # === STEP 2: Sort and compute overall midranks ===
    # Get indices that would sort the combined array
    order = np.argsort(combined)
    
    # Get the sorted values
    sorted_values = combined[order]
    
    # Compute midranks for sorted array (handles ties)
    sorted_midranks = _compute_midrank(sorted_values)
    
    # Map midranks back to original positions
    # overall_ranks[i] = midrank of combined[i]
    overall_ranks = np.empty(m + n, dtype=np.float64)
    overall_ranks[order] = sorted_midranks
    
    # Split back into cases and controls
    X_overall_ranks = overall_ranks[:m]  # overall ranks for cases
    Y_overall_ranks = overall_ranks[m:]  # overall ranks for controls
    
    # === STEP 3: Compute internal ranks for cases ===
    X_order = np.argsort(X)
    X_sorted = X[X_order]
    X_sorted_midranks = _compute_midrank(X_sorted)
    
    # Map back to original case positions
    X_internal_ranks = np.empty(m, dtype=np.float64)
    X_internal_ranks[X_order] = X_sorted_midranks
    
    # === STEP 4: Compute internal ranks for controls ===
    Y_order = np.argsort(Y)
    Y_sorted = Y[Y_order]
    Y_sorted_midranks = _compute_midrank(Y_sorted)
    
    # Map back to original control positions
    Y_internal_ranks = np.empty(n, dtype=np.float64)
    Y_internal_ranks[Y_order] = Y_sorted_midranks
    
    # === STEP 5: Compute placement values ===
    # V_X[i] = (# of controls that case i beats) / n
    #        = (overall_rank - internal_rank_among_cases) / n
    V_X = (X_overall_ranks - X_internal_ranks) / n
    
    # V_Y[j] = (# of cases that beat control j) / m
    # If overall_rank is R and internal_rank is r, then:
    # - There are (R - 1) observations below this control
    # - Of those, (r - 1) are other controls
    # - So (R - 1) - (r - 1) = R - r cases are below this control
    # - But we want cases ABOVE, so it's m - (R - r) = m - R + r... 
    V_Y = 1 - (Y_overall_ranks - Y_internal_ranks) / m
    
    # === STEP 6: Compute AUC (theta) ===
    theta = V_X.mean()
    
    return V_X, V_Y, theta


def _delong_placements(X, Y, batch_size=1000):
    """
    Compute DeLong placement values using batch processing to avoid memory issues.

    For each case i: V_X[i] = proportion of controls that case i beats
    For each control j: V_Y[j] = proportion of cases that beat control j

    Args:
        X: Scores for positive class (cases)
        Y: Scores for negative class (controls)
        batch_size: Size of batches for processing (default 1000)

    Returns:
        Tuple of (V_X, V_Y, theta) where theta is the AUC
    """
    m = len(X)
    n = len(Y)

    # Initialize accumulators
    V_X = np.zeros(m, dtype=np.float64)
    V_Y_sum = np.zeros(n, dtype=np.float64)  # Will accumulate sums for V_Y
    total_sum = 0.0  # For theta

    # Process X in batches to avoid creating full (m, n) matrix
    num_batches = int(np.ceil(m / batch_size))

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, m)
        X_batch = X[start_idx:end_idx]

        # Compute comparisons for this batch: shape (batch_size, n)
        # Each X_batch[i] compared against all Y[j]
        comparisons = (X_batch[:, np.newaxis] > Y[np.newaxis, :]).astype(np.float64)
        comparisons += 0.5 * (X_batch[:, np.newaxis] == Y[np.newaxis, :])

        # V_X for this batch: mean across all controls (axis=1)
        V_X[start_idx:end_idx] = comparisons.mean(axis=1)

        # Accumulate for V_Y: sum across cases in this batch (axis=0)
        V_Y_sum += comparisons.sum(axis=0)

        # Accumulate for theta
        total_sum += comparisons.sum()

    # V_Y[j] = mean across all cases
    V_Y = V_Y_sum / m

    # theta = overall mean
    theta = total_sum / (m * n)

    return V_X, V_Y, theta


def auc_ci_hanley_mcneil(y_true, y_pred_prob, ci=0.95):
    """
    Calculate AUC with 95% confidence interval using Hanley-McNeil method.

    Args:
        y_true: True binary labels
        y_pred_prob: Predicted probabilities
        ci: Confidence level (default 0.95)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    y_true = np.asarray(y_true, dtype=np.int8).reshape(-1)
    y_pred_prob = np.asarray(y_pred_prob, dtype=np.float32).reshape(-1)

    auc = roc_auc_score(y_true, y_pred_prob)
    n1 = int((y_true == 1).sum())
    n0 = int((y_true == 0).sum())

    Q1 = auc / (2 - auc)
    Q2 = 2 * auc * auc / (1 + auc)
    var = (auc * (1 - auc) + (n1 - 1) * (Q1 - auc * auc) + (n0 - 1) * (Q2 - auc * auc)) / (n1 * n0)
    se = np.sqrt(max(var, 0.0))
    z = 1.96  # 95% CI

    return max(0.0, auc - z * se), min(1.0, auc + z * se)


def compute_metrics_from_predictions(y_true, y_pred_binary):
    """
    Compute classification metrics from binary predictions.

    Args:
        y_true: True binary labels
        y_pred_binary: Binary predictions

    Returns:
        Dictionary of metrics
    """
    y_true = y_true.astype(np.int8).reshape(-1)
    y_pred_binary = y_pred_binary.astype(np.int8).reshape(-1)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)

    # F2 score (weights recall 2x more than precision)
    beta = 2.0
    f2 = ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'f2': f2,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


def find_optimal_youden_threshold(y_true, y_pred_prob):
    """
    Find optimal classification threshold using Youden's J statistic.
    Youden's J = Sensitivity + Specificity - 1 = TPR - FPR

    Args:
        y_true: True binary labels
        y_pred_prob: Predicted probabilities

    Returns:
        Dictionary with threshold, Youden index, sensitivity, and specificity
    """
    y_true = np.asarray(y_true, dtype=np.int8).reshape(-1)
    y_pred_prob = np.asarray(y_pred_prob, dtype=np.float32).reshape(-1)

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)

    # Remove infinite thresholds
    mask = np.isfinite(thresholds)
    youden = tpr[mask] - fpr[mask]

    # Find maximum Youden index
    idx = int(np.argmax(youden))

    return {
        'threshold': float(thresholds[mask][idx]),
        'youden_index': float(youden[idx]),
        'sensitivity': float(tpr[mask][idx]),
        'specificity': float(1.0 - fpr[mask][idx])
    }


# ============================================
# PLOTTING FUNCTIONS
# ============================================

def plot_roc_comparison(main_y, main_p, baseline_curves, outcome_name, output_file,
                        auc_value=None, auc_ci=None):
    """
    Plot ROC curves comparing model against traditional risk scores.

    Args:
        main_y: True labels for main model
        main_p: Predicted probabilities for main model
        baseline_curves: List of dicts with baseline model data
        outcome_name: Name of outcome variable
        output_file: Path to save plot
        auc_value: AUC value for main model (optional)
        auc_ci: Tuple of (lower, upper) CI bounds (optional)
    """
    plt.figure(figsize=(7, 6), dpi=140)

    # Main model ROC
    main_fpr, main_tpr, _ = roc_curve(main_y, main_p)
    main_auc = roc_auc_score(main_y, main_p) if auc_value is None else auc_value

    main_label = f"ICD model (AUC={main_auc:.4f})"
    if auc_ci is not None:
        lo, hi = auc_ci
        main_label = f"ICD model (AUC={main_auc:.4f}, 95% CI [{lo:.4f}, {hi:.4f}])"

    plt.plot(main_fpr, main_tpr, lw=2.5, label=main_label)

    # Baseline models
    for baseline in baseline_curves:
        yb = baseline["y_true"]
        sb = baseline["scores"]
        name = baseline["name"]
        auc_lower = baseline['AUC Lower CI']
        auc_upper = baseline['AUC Upper CI']

        if len(np.unique(yb)) < 2:
            continue

        fpr, tpr, _ = roc_curve(yb, sb)
        auc_b = roc_auc_score(yb, sb)
        plt.plot(fpr, tpr, lw=1.8,
                label=f"{name} (AUC={auc_b:.4f}, 95% CI [{auc_lower:.4f}, {auc_upper:.4f}])")

    # Chance line
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color='gray')

    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Comparison - {outcome_name}", fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

    print(f"\nROC curve saved to: {output_file}")


# ============================================
# MAIN EVALUATION PIPELINE
# ============================================

def main():
    print("=" * 70)
    print("MODEL EVALUATION PIPELINE")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Outcome: {OUTCOME_VAR}")
    print(f"  Validation data: {VALIDATION_X_PATH}")
    print(f"  Test data: 2021-2022 combined (sampled {TEST_SAMPLE_FRACTION*100:.0f}%)")
    print()

    # ========================================
    # Step 1: Load model and encoders
    # ========================================
    print("Step 1: Loading model and encoders...")
    model = load_model(MODEL_PATH)
    print(f"  Model loaded: {model.name}")

    with open(LABEL_ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)
    with open(AGE_SCALER_PATH, 'rb') as f:
        age_scaler = pickle.load(f)
    print(f"  ICD encoder: {len(encoder.classes_)} unique codes")

    # ========================================
    # Step 2: Load validation data
    # ========================================
    print("\nStep 2: Loading validation data...")
    X_validate = pd.read_csv(VALIDATION_X_PATH)
    y_validate = pd.read_csv(VALIDATION_Y_PATH).values.ravel()
    print(f"  Validation set: {X_validate.shape[0]:,} samples")
    print(f"  Positive rate: {np.mean(y_validate):.4f}")

    # ========================================
    # Step 3: Load and preprocess test data
    # ========================================
    print("\nStep 3: Loading and preprocessing test data...")
    X_test, y_test, test_data_full = preprocess_test_data(
        TEST_2021_PATH, TEST_2022_PATH, encoder, age_scaler,
        OUTCOME_VAR, sample_fraction=TEST_SAMPLE_FRACTION
    )
    print(f"  Test set: {X_test.shape[0]:,} samples")
    print(f"  Positive rate: {np.mean(y_test):.4f}")

    # ========================================
    # Step 4: Prepare model inputs
    # ========================================
    print("\nStep 4: Preparing model inputs...")
    print(f"  Input mode: {'ICD codes only' if USE_ICD_ONLY else 'All features (ICD + demographics)'}")
    icd_columns = [f'I10_DX{i}' for i in range(1, 41)]
    pay1_columns = [col for col in X_validate.columns if col.startswith('PAY1_')]
    zipinc_qrtl_columns = [col for col in X_validate.columns if col.startswith('ZIPINC_QRTL_')]

    val_inputs = prepare_model_inputs(X_validate, icd_columns, pay1_columns, zipinc_qrtl_columns, icd_only=USE_ICD_ONLY)
    test_inputs = prepare_model_inputs(X_test, icd_columns, pay1_columns, zipinc_qrtl_columns, icd_only=USE_ICD_ONLY)

    # ========================================
    # Step 5: Generate predictions
    # ========================================
    print("\nStep 5: Generating predictions...")
    print("  Predicting on validation set...")
    y_val_pred_prob = model.predict(val_inputs, batch_size=1024, verbose=0).squeeze()
    print("  Predicting on test set...")
    y_test_pred_prob = model.predict(test_inputs, batch_size=1024, verbose=0).squeeze()

    # ========================================
    # Step 6: Find optimal threshold on validation set
    # ========================================
    print("\nStep 6: Finding optimal Youden threshold on validation set...")
    if len(np.unique(y_validate)) > 1:
        youden_result = find_optimal_youden_threshold(y_validate, y_val_pred_prob)
        best_threshold = youden_result['threshold']

        print(f"  Best threshold: {best_threshold:.6f}")
        print(f"  Youden index: {youden_result['youden_index']:.4f}")
        print(f"  Sensitivity: {youden_result['sensitivity']:.4f}")
        print(f"  Specificity: {youden_result['specificity']:.4f}")

        # Compute validation metrics
        y_val_pred_binary = (y_val_pred_prob > best_threshold).astype(np.int8)
        val_metrics = compute_metrics_from_predictions(y_validate, y_val_pred_binary)
        val_auc = roc_auc_score(y_validate, y_val_pred_prob)
        # val_auc_ci = auc_ci_hanley_mcneil(y_validate, y_val_pred_prob)
        val_auc_ci = auc_ci_delong(y_validate, y_val_pred_prob)

        print(f"\n  Validation metrics:")
        print(f"    AUC: {val_auc:.4f} (95% CI: [{val_auc_ci[0]:.4f}, {val_auc_ci[1]:.4f}])")
        print(f"    Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"    Precision: {val_metrics['precision']:.4f}")
        print(f"    Recall: {val_metrics['recall']:.4f}")
        print(f"    F1 Score: {val_metrics['f1']:.4f}")
        print(f"    F2 Score: {val_metrics['f2']:.4f}")
    else:
        print("  Warning: Only one class in validation set, cannot find threshold")
        best_threshold = 0.5

    # ========================================
    # Step 7: Evaluate on test set
    # ========================================
    print("\nStep 7: Evaluating on test set with threshold={:.6f}...".format(best_threshold))
    if len(np.unique(y_test)) > 1:
        y_test_pred_binary = (y_test_pred_prob > best_threshold).astype(np.int8)
        test_metrics = compute_metrics_from_predictions(y_test, y_test_pred_binary)
        test_auc = roc_auc_score(y_test, y_test_pred_prob)
        # test_auc_ci = auc_ci_hanley_mcneil(y_test, y_test_pred_prob)
        test_auc_ci = auc_ci_delong(y_test, y_test_pred_prob)

        print(f"\n  Test metrics:")
        print(f"    AUC: {test_auc:.4f} (95% CI: [{test_auc_ci[0]:.4f}, {test_auc_ci[1]:.4f}])")
        print(f"    Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"    Precision: {test_metrics['precision']:.4f}")
        print(f"    Recall: {test_metrics['recall']:.4f}")
        print(f"    F1 Score: {test_metrics['f1']:.4f}")
        print(f"    F2 Score: {test_metrics['f2']:.4f}")
        print(f"\n  Confusion matrix:")
        print(f"    TN: {test_metrics['tn']}, FP: {test_metrics['fp']}")
        print(f"    FN: {test_metrics['fn']}, TP: {test_metrics['tp']}")
    else:
        print("  Warning: Only one class in test set, cannot compute metrics")

    # ========================================
    # Step 8: Compare against traditional scores
    # ========================================
    print("\nStep 8: Comparing against traditional clinical risk scores...")
    traditional_indexes = ['INDEX_MORTALITY', 'INDEX_READMISSION', 'CHARLINDEX', 'CHARLINDEX_AGE_ADJUST']
    baseline_curves = []
    results = []

    for index in traditional_indexes:
        if index not in test_data_full.columns:
            print(f"  Skipping {index}: not found in test data")
            continue

        print(f"  Evaluating {index}...")

        # Find threshold on validation set
        if index in X_validate.columns and len(np.unique(y_validate)) > 1:
            val_scores = X_validate[index].values.astype(np.float32)
            val_youden = find_optimal_youden_threshold(y_validate, val_scores)
            index_threshold = val_youden['threshold']
        else:
            index_threshold = 0

        # Evaluate on test set
        test_scores = test_data_full[index].values.astype(np.float32)

        if len(np.unique(y_test)) < 2:
            print(f"    Skipped: only one class in test set")
            continue

        test_pred_binary = (test_scores > index_threshold).astype(np.int8)
        metrics = compute_metrics_from_predictions(y_test, test_pred_binary)
        auc = roc_auc_score(y_test, test_scores)
        # auc_ci = auc_ci_hanley_mcneil(y_test, test_scores)
        auc_ci = auc_ci_delong(y_test, test_scores)

        results.append({
            'Index': index,
            'AUC': auc,
            'AUC Lower CI': auc_ci[0],
            'AUC Upper CI': auc_ci[1],
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1'],
            'F2 Score': metrics['f2']
        })

        # Add to baseline curves for plotting
        if OUTCOME_VAR == 'REA30':
            if index == "INDEX_READMISSION":
                baseline_curves.append({
                    "name": "ECI", "y_true": y_test, "scores": test_scores,
                    'AUC Lower CI': auc_ci[0], 'AUC Upper CI': auc_ci[1]
                })
        else:
            if index == "INDEX_MORTALITY":
                baseline_curves.append({
                    "name": "ECI", "y_true": y_test, "scores": test_scores,
                    'AUC Lower CI': auc_ci[0], 'AUC Upper CI': auc_ci[1]
                })

        if index == "CHARLINDEX":
            baseline_curves.append({
                "name": "CCI", "y_true": y_test, "scores": test_scores,
                'AUC Lower CI': auc_ci[0], 'AUC Upper CI': auc_ci[1]
            })
        elif index == "CHARLINDEX_AGE_ADJUST":
            baseline_curves.append({
                "name": "CCI Age Adjusted", "y_true": y_test, "scores": test_scores,
                'AUC Lower CI': auc_ci[0], 'AUC Upper CI': auc_ci[1]
            })

    if results:
        results_df = pd.DataFrame(results).round(4)
        print("\n  Traditional scores performance:")
        print(results_df.to_string(index=False))

    # ========================================
    # Step 9: Generate ROC comparison plot
    # ========================================
    # print("\nStep 9: Generating ROC comparison plot...")
    # plot_roc_comparison(
    #     y_test, y_test_pred_prob, baseline_curves,
    #     outcome_name=OUTCOME_VAR,
    #     output_file=OUTPUT_PLOT,
    #     auc_value=test_auc,
    #     auc_ci=test_auc_ci
    # )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
