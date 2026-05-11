"""
Script to generate calibration curves for model predictions and baseline indices.
Creates binned calibration plots comparing predicted vs actual outcome rates.

A well-calibrated model should have points close to the diagonal line (y=x).
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from keras.saving import load_model
from keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention
import pickle
import matplotlib.pyplot as plt

from config import DATA_DIR, MODEL_DIR, NRD_2021_TEST, NRD_2022_TEST, BASELINES_DIR, FIGURES_DIR

# ============================================
# CONFIGURATION - MODIFY THESE AS NEEDED
# ============================================

# Path to trained model
MODEL_PATH = MODEL_DIR / 'mort_nodie_hypertrial_auc.keras'

# Outcome variable (should match what the model was trained on)
# Options: 'DIED', 'MOR30', 'REA30'
OUTCOME_VAR = 'MOR30'

# Paths to training data (downsampled)
TRAIN_X_PATH = DATA_DIR / 'mort_nodie' / 'X_train_downsampled.csv'
TRAIN_Y_PATH = DATA_DIR / 'mort_nodie' / 'y_train_downsampled.csv'

# Paths to 2021-2022 test data (actual test set)
TEST_2021_PATH = NRD_2021_TEST
TEST_2022_PATH = NRD_2022_TEST

# Sample fraction for test set (to reduce computational cost)
TEST_SAMPLE_FRACTION = 0.10  # Use 10% of test data (stratified)

# Calibration settings
NUM_BINS = 10
MIN_SAMPLES_PER_BIN = 10

# Binning method: 'equal_width' or 'quantile'
# - 'equal_width': bins have equal probability ranges (0-0.1, 0.1-0.2, etc.)
# - 'quantile': bins have equal number of samples (useful for skewed distributions)
BINNING_METHOD = 'quantile'

# What to compute: 'model', 'baselines', or 'both'
CALIBRATION_MODE = 'baselines'

# ============================================
# CUSTOM KERAS COMPONENTS
# ============================================

@tf.keras.utils.register_keras_serializable(package="Custom")
def f2_score(y_true, y_pred):
    """Custom F2 score metric"""
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    epsilon = tf.keras.backend.epsilon()
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f2 = (5 * precision * recall) / (4 * precision + recall + epsilon)
    return f2

@tf.keras.utils.register_keras_serializable(package="Custom")
class DeepSet(tf.keras.Model):
    """DeepSet aggregation layer for set-based modeling"""
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
    """Transformer encoder block"""
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
    """F2 score metric for model evaluation"""
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
# CALIBRATION FUNCTIONS
# ============================================

def calibrate_probability(p_sampled, beta, eps=1e-8):
    """
    Correct predicted probabilities after undersampling.

    Args:
        p_sampled: predicted probability from model trained on undersampled data
        beta: undersampling ratio = (# majority after undersampling) / (# majority original)

    Returns:
        calibrated probability reflecting true population distribution
    """

    p_sampled = tf.clip_by_value(p_sampled, eps, 1-eps, name=None)
    return p_sampled / (p_sampled + (1 - p_sampled) / beta)


def compute_calibration_curve(y_true, y_pred_prob, num_bins=10, min_samples=10):
    """
    Compute calibration curve by binning predictions (equal-width bins).

    Returns:
        predicted_probs, actual_probs, bin_counts
    """
    bin_edges = np.linspace(0, 1, num_bins + 1)

    predicted_probs = []
    actual_probs = []
    bin_counts = []

    for i in range(num_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == num_bins - 1:
            mask = (y_pred_prob >= lower) & (y_pred_prob <= upper)
        else:
            mask = (y_pred_prob >= lower) & (y_pred_prob < upper)

        n_samples = np.sum(mask)
        if n_samples >= min_samples:
            midpoint = (lower + upper) / 2.0
            actual_rate = np.mean(y_true[mask])
            predicted_probs.append(midpoint)
            actual_probs.append(actual_rate)
            bin_counts.append(n_samples)

    return predicted_probs, actual_probs, bin_counts


def compute_calibration_curve_quantile(y_true, y_pred_prob, num_bins=10, min_samples=10):
    """
    Compute calibration curve using quantile-based binning (equal sample count per bin).

    Returns:
        predicted_probs, actual_probs, bin_counts
    """
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(y_pred_prob, percentiles)
    bin_edges = np.unique(bin_edges)
    bin_edges[-1] = bin_edges[-1] + 1e-8

    predicted_probs = []
    actual_probs = []
    bin_counts = []

    for i in range(len(bin_edges) - 1):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == len(bin_edges) - 2:
            mask = (y_pred_prob >= lower) & (y_pred_prob <= upper)
        else:
            mask = (y_pred_prob >= lower) & (y_pred_prob < upper)

        n_samples = np.sum(mask)
        if n_samples >= min_samples:
            mean_pred = np.mean(y_pred_prob[mask])
            actual_rate = np.mean(y_true[mask])
            predicted_probs.append(mean_pred)
            actual_probs.append(actual_rate)
            bin_counts.append(n_samples)

    return predicted_probs, actual_probs, bin_counts


def compute_ece(pred_probs, actual_probs, counts):
    """Compute Expected Calibration Error."""
    if len(pred_probs) == 0:
        return np.nan
    total_samples = sum(counts)
    weighted_diff = sum(
        count * abs(pred - actual)
        for pred, actual, count in zip(pred_probs, actual_probs, counts)
    )
    return weighted_diff / total_samples


def run_calibration(y_true, probs, name, outcome_name, beta=None,
                    num_bins=10, min_samples=10, binning_method='quantile',
                    output_file=None):
    """
    Unified calibration pipeline: optionally recalibrate, bin, plot, compute ECE.

    Args:
        y_true: True binary labels
        probs: Predicted probabilities in [0, 1] (from model output or sigmoid)
        name: Label for this source (e.g., 'ICD Model', 'ECI')
        outcome_name: Outcome variable name
        beta: Downsampling ratio for recalibration (None = skip recalibration)
        num_bins: Number of bins
        min_samples: Min samples per bin
        binning_method: 'quantile' or 'equal_width'
        output_file: Output plot path (auto-generated if None)

    Returns:
        (predicted_probs, actual_probs, bin_counts) or None if no valid bins
    """
    probs = np.asarray(probs, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.int32)

    # Step 1: beta recalibration (optional)
    if beta is not None:
        probs = calibrate_probability(probs, beta)
        if hasattr(probs, 'numpy'):
            probs = probs.numpy()

    print(f"  Probability range after recalibration: [{probs.min():.6f}, {probs.max():.6f}]")
    print(f"  Unique probability values: {len(np.unique(probs))}")

    # Step 2: bin
    if binning_method == 'quantile':
        pred_probs, actual_probs, counts = compute_calibration_curve_quantile(
            y_true, probs, num_bins=num_bins, min_samples=min_samples)
    else:
        pred_probs, actual_probs, counts = compute_calibration_curve(
            y_true, probs, num_bins=num_bins, min_samples=min_samples)
    print(f"  Actual bins produced: {len(pred_probs)} (requested {num_bins})")

    if len(pred_probs) == 0:
        print(f"  No bins with enough samples for {name}. Skipping.")
        return None

    # Step 3: plot
    max_val = max(max(pred_probs), max(actual_probs))
    max_val = min(1.0, max_val * 1.1)

    plt.figure(figsize=(10, 8), dpi=150)

    plt.plot([0, max_val], [0, max_val], 'k--', linewidth=2,
             label='Perfect Calibration', alpha=0.7)

    plt.scatter(pred_probs, actual_probs, s=150, alpha=0.8,
               marker='s', edgecolors='black', linewidths=1.5, color='#A23B72',
               label=f'{name} (n={sum(counts):,})')
    plt.plot(pred_probs, actual_probs, alpha=0.5, linewidth=2, color='#A23B72')

    plt.xlabel('Predicted Probability', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Probability', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=12, frameon=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim([0, max_val])
    plt.ylim([0, max_val])
    plt.tight_layout()

    if output_file is None:
        safe_name = name.lower().replace(' ', '_')
        output_file = FIGURES_DIR / 'calibration' / f'calibration_{safe_name}_{outcome_name.lower()}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Calibration curve saved to: {output_file}")

    # Step 4: print bin details and ECE
    for i, (pred, actual, count) in enumerate(zip(pred_probs, actual_probs, counts)):
        print(f"    Bin {i+1}: Predicted={pred:.6f}, Actual={actual:.4f}, N={count:,}")

    ece = compute_ece(pred_probs, actual_probs, counts)
    print(f"  ECE: {ece:.4f}")

    return pred_probs, actual_probs, counts


# ============================================
# DATA PREPROCESSING FUNCTION
# ============================================

def preprocess_test_data(test_2021_path, test_2022_path, encoder, age_scaler,
                         outcome_var, sample_fraction=0.10):
    """
    Preprocess 2021-2022 test data following the same steps as evaluate.py

    Returns:
        X_test: Preprocessed features
        y_test: True labels
        test_data: Full DataFrame (includes index columns for baselines)
    """
    print("  Loading 2021 and 2022 test files...")
    df1 = pd.read_csv(test_2021_path)
    df2 = pd.read_csv(test_2022_path)

    # Combine
    test_data = pd.concat([df1, df2], ignore_index=True)
    print(f"  Combined shape: {test_data.shape}")

    # Convert to uppercase
    test_data.columns = test_data.columns.str.upper()

    # Filter out DIED==1 patients (always do this to be consistent)
    if 'DIED' in test_data.columns:
        print("  Filtering out patients who died...")
        test_data = test_data[test_data['DIED'] != 1]

    # Handle missing outcome
    test_data = test_data.dropna(subset=[outcome_var])
    print(f"  After filtering: {test_data.shape}")

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

    # Replace -8 and -9 with NaN in PAY1 and ZIPINC_QRTL
    print("  Handling missing value codes...")
    test_data['PAY1'] = test_data['PAY1'].replace([-8, -9], np.nan)
    test_data['ZIPINC_QRTL'] = test_data['ZIPINC_QRTL'].replace([-8, -9], np.nan)

    # One-hot encode
    print("  One-hot encoding PAY1 and ZIPINC_QRTL...")
    test_data = pd.get_dummies(test_data, columns=['PAY1', 'ZIPINC_QRTL'],
                                prefix=['PAY1', 'ZIPINC_QRTL'])

    # Ensure all expected columns are present
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

    # Extract features and drop missing
    X_test = test_data[['AGE', 'FEMALE'] + pay1_columns + zipinc_qrtl_columns + icd_columns]
    X_test = X_test.dropna()
    test_data = test_data.loc[X_test.index]

    print(f"  After preprocessing: {X_test.shape}")

    # Stratified sampling to reduce computational cost
    if sample_fraction < 1.0:
        print(f"  Applying stratified sampling ({sample_fraction*100:.0f}%)...")
        y_all = test_data[outcome_var].to_numpy()
        idx_all = np.arange(len(test_data))

        def stratified_pick(idx, y_sub, frac, seed):
            rng = np.random.default_rng(seed)
            picked = []
            for cls in np.unique(y_sub):
                cls_idx = idx[y_sub == cls]
                n = max(1, int(np.floor(frac * len(cls_idx))))
                pick = rng.choice(cls_idx, size=n, replace=False)
                picked.append(pick)
            return np.concatenate(picked)

        sampled_idx = stratified_pick(idx_all, y_all, frac=sample_fraction, seed=42)
        test_data = test_data.iloc[sampled_idx].copy()
        X_test = test_data[['AGE', 'FEMALE'] + pay1_columns + zipinc_qrtl_columns + icd_columns]
        print(f"  After sampling: {X_test.shape}")

    y_test = test_data[outcome_var].to_numpy(np.int32)

    return X_test, y_test, test_data


def get_beta(outcome_var):
    """Return downsampling ratio (positive / negative) for the given outcome."""
    if outcome_var == "MOR30":
        return 272069 / 70180061
    elif outcome_var == "REA30":
        return 8600497 / 61851633
    else:
        print("WARNING: Outcome variable not recognized for beta calculation!")
        return 1.0


# ============================================
# MAIN FUNCTION
# ============================================

def main():
    print("=" * 70)
    print("CALIBRATION CURVE GENERATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Outcome: {OUTCOME_VAR}")
    print(f"  Training data: {TRAIN_X_PATH}")
    print(f"  Test data: 2021-2022 combined (sampled {TEST_SAMPLE_FRACTION*100:.0f}%)")
    print(f"  Number of bins: {NUM_BINS}")
    print(f"  Min samples per bin: {MIN_SAMPLES_PER_BIN}")
    print(f"  Binning method: {BINNING_METHOD}")
    print(f"  Calibration mode: {CALIBRATION_MODE}")
    print()

    run_model = CALIBRATION_MODE in ('model', 'both')
    run_baselines = CALIBRATION_MODE in ('baselines', 'both')

    beta = get_beta(OUTCOME_VAR)
    print(f"Beta value: {beta:.6f}")

    # Load encoders (needed for both modes to preprocess test data)
    print("\nLoading encoders...")
    with open('Model/full_label_encoder.pkl', 'rb') as file:
        encoder = pickle.load(file)
    with open('Model/full_age_scaler.pkl', 'rb') as file:
        age_scaler = pickle.load(file)
    print(f"  ICD encoder: {len(encoder.classes_)} unique codes")

    # Load and preprocess test data (2021-2022) -- needed for both modes
    print("\nLoading and preprocessing test data (2021-2022)...")
    X_test, y_test, test_data = preprocess_test_data(
        TEST_2021_PATH, TEST_2022_PATH, encoder, age_scaler,
        OUTCOME_VAR, sample_fraction=TEST_SAMPLE_FRACTION
    )
    print(f"  Test set: {X_test.shape[0]:,} samples")
    print(f"  Positive rate: {np.mean(y_test):.4f}")

    # ==================== MODEL CALIBRATION ====================
    if run_model:
        print("\n" + "=" * 70)
        print("MODEL CALIBRATION")
        print("=" * 70)

        # Load model
        print("\nLoading model...")
        model = load_model(MODEL_PATH)
        print(f"  Model loaded: {model.name}")

        # Load training data
        print("\nLoading training data...")
        X_train = pd.read_csv(TRAIN_X_PATH)
        y_train = pd.read_csv(TRAIN_Y_PATH).values.ravel()
        print(f"  Training set: {X_train.shape[0]:,} samples")
        print(f"  Positive rate: {np.mean(y_train):.4f}")

        # Prepare model inputs
        icd_columns = [f'I10_DX{i}' for i in range(1, 41)]
        pay1_columns = [col for col in X_train.columns if col.startswith('PAY1_')]
        zipinc_qrtl_columns = [col for col in X_train.columns if col.startswith('ZIPINC_QRTL_')]

        train_inputs = [
            X_train[icd_columns],
            X_train['AGE'],
            X_train['FEMALE'],
        ] + [X_train[c] for c in pay1_columns] \
          + [X_train[c] for c in zipinc_qrtl_columns]

        test_inputs = [
            X_test[icd_columns],
            X_test['AGE'],
            X_test['FEMALE'],
        ] + [X_test[c] for c in pay1_columns] \
          + [X_test[c] for c in zipinc_qrtl_columns]

        # Get probabilities (already in [0, 1] from sigmoid output)
        print("\nGenerating predictions...")
        y_train_pred = model.predict(train_inputs, batch_size=1024, verbose=0).squeeze()
        y_test_pred = model.predict(test_inputs, batch_size=1024, verbose=0).squeeze()

        # Training set: no beta recalibration (already balanced 50/50)
        print("\n--- ICD Model (Training Set) ---")
        run_calibration(
            y_train, y_train_pred,
            name='ICD Model (Train)', outcome_name=OUTCOME_VAR,
            beta=None,
            num_bins=NUM_BINS, min_samples=MIN_SAMPLES_PER_BIN,
            binning_method=BINNING_METHOD,
            output_file=FIGURES_DIR / 'calibration' / f'calibration_model_train_{OUTCOME_VAR.lower()}.png',
        )

        # Test set: apply beta recalibration
        print("\n--- ICD Model (Test Set) ---")
        run_calibration(
            y_test, y_test_pred,
            name='ICD Model (Test)', outcome_name=OUTCOME_VAR,
            beta=beta,
            num_bins=NUM_BINS, min_samples=MIN_SAMPLES_PER_BIN,
            binning_method=BINNING_METHOD,
            output_file=FIGURES_DIR / 'calibration' / f'calibration_model_test_{OUTCOME_VAR.lower()}.png',
        )

    # ==================== BASELINE CALIBRATION ====================
    if run_baselines:
        print("\n" + "=" * 70)
        print("BASELINE COMORBIDITY INDEX CALIBRATION")
        print("=" * 70)

        baseline_indices = ['INDEX_MORTALITY', 'INDEX_READMISSION',
                            'CHARLINDEX', 'CHARLINDEX_AGE_ADJUST']

        display_names = {
            'INDEX_READMISSION': 'ECI',
            'INDEX_MORTALITY': 'ECI',
            'CHARLINDEX': 'CCI',
            'CHARLINDEX_AGE_ADJUST': 'CCI age_adjusted',
        }

        for index_name in baseline_indices:
            if index_name not in test_data.columns:
                print(f"\n  WARNING: {index_name} not found in test data. Skipping.")
                continue

            raw_scores = test_data[index_name].to_numpy(dtype=np.float64)

            # Filter NaN
            valid_mask = ~np.isnan(raw_scores)
            if valid_mask.sum() < MIN_SAMPLES_PER_BIN:
                print(f"\n  WARNING: {index_name} has too few valid values. Skipping.")
                continue

            scores_valid = raw_scores[valid_mask]
            y_valid = y_test[valid_mask]

            display_name = display_names.get(index_name, index_name)
            print(f"\n--- {index_name} ---")
            print(f"  Raw score range: [{scores_valid.min():.1f}, {scores_valid.max():.1f}]")

            # Load fitted logistic regression to convert scores to [0, 1]
            lr_path = BASELINES_DIR / f'lr_{index_name.lower()}_{OUTCOME_VAR.lower()}.pkl'
            try:
                with open(lr_path, 'rb') as f:
                    lr_model = pickle.load(f)
                print(f"  Loaded LR model from: {lr_path}")
                print(f"  LR coefficients: a={lr_model.coef_[0][0]:.6f}, b={lr_model.intercept_[0]:.6f}")
                probs = lr_model.predict_proba(scores_valid.reshape(-1, 1))[:, 1]
            except FileNotFoundError:
                print(f"  WARNING: LR model not found at {lr_path}. Run fit_LR_baseline.py first. Skipping.")
                continue

            print(f"  After LR: [{probs.min():.4f}, {probs.max():.4f}]")

            # Same pipeline: beta recalibration -> bin -> plot -> ECE
            run_calibration(
                y_valid, probs,
                name=display_name, outcome_name=OUTCOME_VAR,
                beta=beta,
                num_bins=NUM_BINS, min_samples=MIN_SAMPLES_PER_BIN,
                binning_method=BINNING_METHOD,
                output_file=FIGURES_DIR / 'calibration' / f'calibration_{index_name.lower()}_{OUTCOME_VAR.lower()}.png',
            )

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
