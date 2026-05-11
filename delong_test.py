"""
DeLong Test for Model Comparison using rocauc_comparison package

This script performs DeLong's test to statistically compare the model's AUC
against traditional comorbidity indices (ECI, CCI, CCI Age Adjusted).

DeLong's test is used to compare two correlated ROC curves from the same sample.
It provides a p-value indicating whether the difference in AUCs is statistically significant.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from keras.saving import load_model
from keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention
from sklearn.metrics import roc_auc_score
import pickle
from rocauc_comparison import delong_roc_test

from config import MODEL_DIR, NRD_2021_TEST, NRD_2022_TEST, SMALL_DATASET_DIR, DELONG_RESULTS_DIR

# ============================================
# CONFIGURATION
# ============================================

# Model to evaluate
MODEL_PATH = MODEL_DIR / 'mort_nodie_hypertrial_auc.keras'

# Outcome variable (must match what model was trained on)
# Options: 'DIED', 'MOR30', 'REA30'
OUTCOME_VAR = 'MOR30'

# Test data paths (2021-2022 actual test data)
TEST_2021_PATH = NRD_2021_TEST
TEST_2022_PATH = NRD_2022_TEST
TEST_DATA_PATH = SMALL_DATASET_DIR / f'small_test_dataset_{OUTCOME_VAR.lower()}.csv'

# Test data sampling (to reduce computational cost)
TEST_SAMPLE_FRACTION = 0.10  # Use 10% of test data (stratified)

# Significance level
ALPHA = 0.05

# ============================================
# CUSTOM KERAS COMPONENTS
# ============================================

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

    # Filter out DIED==1 patients
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

    # Handle missing value codes
    print("  Handling missing value codes...")
    test_data['PAY1'] = test_data['PAY1'].replace([-8, -9], np.nan)
    test_data['ZIPINC_QRTL'] = test_data['ZIPINC_QRTL'].replace([-8, -9], np.nan)

    # One-hot encode categorical variables
    print("  One-hot encoding PAY1 and ZIPINC_QRTL...")
    test_data = pd.get_dummies(test_data, columns=['PAY1', 'ZIPINC_QRTL'],
                                prefix=['PAY1', 'ZIPINC_QRTL'])

    # Ensure all expected columns exist
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

    # Stratified sampling
    if sample_fraction < 1.0:
        print(f"  Applying stratified sampling ({sample_fraction*100:.0f}%)...")
        y_all = test_data[outcome_var].to_numpy()
        idx_all = np.arange(len(test_data))

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


def prepare_model_inputs(X, icd_columns, pay1_columns, zipinc_qrtl_columns):
    """
    Prepare inputs in the format expected by the model
    """
    return [
        X[icd_columns],
        X['AGE'],
        X['FEMALE'],
    ] + [X[c] for c in pay1_columns] \
      + [X[c] for c in zipinc_qrtl_columns]


# ============================================
# MAIN COMPARISON PIPELINE
# ============================================

def main():
    print("=" * 80)
    print("DELONG TEST: MODEL vs COMORBIDITY INDICES")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Outcome: {OUTCOME_VAR}")
    print(f"  Test data: 2021-2022 combined (sampled {TEST_SAMPLE_FRACTION*100:.0f}%)")
    print(f"  Significance level (alpha): {ALPHA}")
    print()

    # ========================================
    # Step 1: Load model and encoders
    # ========================================
    print("Step 1: Loading model and encoders...")
    model = load_model(MODEL_PATH)
    print(f"  Model loaded: {model.name}")

    with open('Model/full_label_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('Model/full_age_scaler.pkl', 'rb') as f:
        age_scaler = pickle.load(f)
    print(f"  ICD encoder: {len(encoder.classes_)} unique codes")

    # ========================================
    # Step 2: Load and preprocess test data
    # ========================================
    print("\nStep 2: Loading and preprocessing test data...")
    X_test, y_test, test_data_full = preprocess_test_data(
        TEST_2021_PATH, TEST_2022_PATH, encoder, age_scaler,
        OUTCOME_VAR, sample_fraction=TEST_SAMPLE_FRACTION
    )
    print(f"  Test set: {X_test.shape[0]:,} samples")
    print(f"  Positive rate: {np.mean(y_test):.4f}")

    # ========================================
    # Step 3: Prepare model inputs and predict
    # ========================================
    print("\nStep 3: Generating model predictions...")
    icd_columns = [f'I10_DX{i}' for i in range(1, 41)]
    pay1_columns = [col for col in X_test.columns if col.startswith('PAY1_')]
    zipinc_qrtl_columns = [col for col in X_test.columns if col.startswith('ZIPINC_QRTL_')]

    test_inputs = prepare_model_inputs(X_test, icd_columns, pay1_columns, zipinc_qrtl_columns)
    y_pred_model = model.predict(test_inputs, batch_size=1024, verbose=0).squeeze()

    model_auc = roc_auc_score(y_test, y_pred_model)
    print(f"  Model AUC: {model_auc:.4f}")

    # ========================================
    # Step 4: Perform DeLong test for each index
    # ========================================
    print("\nStep 4: Performing DeLong tests...")

    # Determine which indices to test based on outcome
    if OUTCOME_VAR == 'REA30':
        indices_to_test = {
            'INDEX_READMISSION': 'ECI (Readmission)',
            'CHARLINDEX': 'CCI',
            'CHARLINDEX_AGE_ADJUST': 'CCI Age Adjusted'
        }
    else:
        indices_to_test = {
            'INDEX_MORTALITY': 'ECI (Mortality)',
            'CHARLINDEX': 'CCI',
            'CHARLINDEX_AGE_ADJUST': 'CCI Age Adjusted'
        }

    results = []

    print(f"\n{'Index':<30} {'Index AUC':<12} {'Model AUC':<12} {'Diff':<10} {'P-value':<12} {'Significant':<12}")
    print("-" * 100)

    for index_col, index_name in indices_to_test.items():
        if index_col not in test_data_full.columns:
            print(f"{index_name:<30} NOT FOUND IN DATA")
            continue

        # Get index scores (test_data_full is already aligned with X_test and y_test)
        index_scores = test_data_full[index_col].values.astype(np.float32)

        # Calculate index AUC
        index_auc = roc_auc_score(y_test, index_scores)

        # Perform DeLong test using rocauc_comparison package
        # delong_roc_test returns: log10(p_value)
        p_value_log10 = delong_roc_test(y_test, y_pred_model, index_scores)

        print(f"p value log 10: {p_value_log10}")
        print(f"p value log 10 type: {type(p_value_log10)}")

        # Convert numpy array to scalar
        if isinstance(p_value_log10, np.ndarray):
            p_value_log10 = float(p_value_log10.item())

        # Convert log10 p-value to regular p-value
        p_value = 10 ** p_value_log10

        # Calculate AUCs directly
        auc1 = model_auc
        auc2 = index_auc

        # Store results
        results.append({
            'Index': index_name,
            'Index_Column': index_col,
            'Model_AUC': float(auc1),
            'Index_AUC': float(auc2),
            'AUC_Difference': float(auc1 - auc2),
            'P_Value': float(p_value),
            'Significant': 'Yes' if p_value < ALPHA else 'No'
        })

        # Print row
        sig_marker = '*' if p_value < ALPHA else ''
        print(f"{index_name:<30} {auc2:<12.4f} {auc1:<12.4f} "
              f"{(auc1-auc2):<10.4f} {p_value:<12.6f} "
              f"{('Yes' if p_value < ALPHA else 'No'):<12}{sig_marker}")

    # ========================================
    # Step 5: Summary
    # ========================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    results_df = pd.DataFrame(results)

    print(f"\nModel AUC: {model_auc:.4f}")
    print(f"\nComparisons (alpha = {ALPHA}):")
    for _, row in results_df.iterrows():
        diff = row['AUC_Difference']
        direction = "higher" if diff > 0 else "lower"
        sig_text = "statistically significant" if row['Significant'] == 'Yes' else "not statistically significant"
        print(f"  vs {row['Index']}: {abs(diff):.4f} {direction} (p={row['P_Value']:.6f}, {sig_text})")

    # Save results to CSV
    output_file = DELONG_RESULTS_DIR / f'delong_results_{OUTCOME_VAR.lower()}.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    print("\n" + "=" * 80)
    print("DELONG TEST COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
