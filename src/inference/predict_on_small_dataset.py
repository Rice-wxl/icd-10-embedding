"""
Script to run predictions on a small test dataset.
Assumes the model and Youden threshold are already given.
Outputs both raw probabilities and binary predictions.

Usage:
    python predict_on_small_dataset.py

Configuration:
    - Modify the MODEL_PATH to point to your trained model
    - Modify the YOUDEN_THRESHOLD based on your validation results
    - Modify the TEST_DATA_PATH to point to your test dataset
    - Modify the OUTCOME_VAR to match your prediction task
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from keras.saving import load_model
from keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention
from keras.models import Model
import pickle

# ============================================
# CONFIGURATION - MODIFY THESE AS NEEDED
# ============================================

# Path to trained model
MODEL_PATH = 'Model/readmit_hypertrial_auc.keras'

# Youden threshold (obtained from validation set)
YOUDEN_THRESHOLD = 0.5022004246711731  # Replace with actual threshold from evaluate.py

# Outcome variable (should match what the model was trained on)
# Options: 'DIED', 'MOR30', 'REA30'
OUTCOME_VAR = 'rea30'


# Path to test dataset
# TEST_DATA_PATH = f'small_test_dataset_{OUTCOME_VAR}.csv'
TEST_DATA_PATH = f'small_dataset_{OUTCOME_VAR}_custom.csv'
# Output file for predictions
# OUTPUT_FILE = f'small_dataset_predictions_{OUTCOME_VAR}.csv'
OUTPUT_FILE = f'small_dataset_custom_preds_{OUTCOME_VAR}.csv'


OUTCOME_VAR = OUTCOME_VAR.upper()

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

        # Element-wise transformation: phi network
        self.phi = tf.keras.Sequential([
            Dense(self.hidden_dim, activation='relu') for _ in range(self.num_encode)
        ])

        # Post-aggregation transformation: rho network
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
# MAIN PREDICTION PIPELINE
# ============================================

def main():
    print("=" * 60)
    print("Small Dataset Prediction Script")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Youden threshold: {YOUDEN_THRESHOLD}")
    print(f"  Test data: {TEST_DATA_PATH}")
    print(f"  Outcome: {OUTCOME_VAR}")
    print()

    # Load the trained model
    print("Loading model...")
    model = load_model(MODEL_PATH)
    print(f"Model loaded successfully: {model.name}")

    # Load encoders
    print("\nLoading encoders...")
    with open('Model/full_label_encoder.pkl', 'rb') as file:
        encoder = pickle.load(file)
    with open('Model/full_age_scaler.pkl', 'rb') as file:
        age_scaler = pickle.load(file)
    print(f"  ICD encoder: {len(encoder.classes_)} unique codes")
    print(f"  Age scaler: loaded")

    # Load test data
    print(f"\nLoading test data from {TEST_DATA_PATH}...")
    test_data_original = pd.read_csv(TEST_DATA_PATH)
    print(f"  Dataset shape: {test_data_original.shape}")

    # Ensure uppercase column names
    test_data_original.columns = test_data_original.columns.str.upper()

    # Create a copy for preprocessing (keep original untouched)
    test_data = test_data_original.copy()

    # Define ICD columns
    icd_columns = [f'I10_DX{i}' for i in range(1, 41)]

    # Preprocess the data
    print("\nPreprocessing data...")

    # 1. Encode ICD codes
    print("  - Encoding ICD codes...")
    label_to_int = {label: idx for idx, label in enumerate(encoder.classes_)}
    unknown_label_int = encoder.transform(["NAN"])[0]

    for col in icd_columns:
        test_data[col] = test_data[col].astype(str).str.upper()
        test_data[col] = test_data[col].map(label_to_int).fillna(unknown_label_int).astype(int)

    # 2. Normalize AGE
    print("  - Normalizing AGE...")
    test_data['AGE'] = age_scaler.transform(test_data[['AGE']])

    # 3. Handle missing value codes in PAY1 and ZIPINC_QRTL
    print("  - Handling missing value codes...")
    test_data['PAY1'] = test_data['PAY1'].replace([-8, -9], np.nan)
    test_data['ZIPINC_QRTL'] = test_data['ZIPINC_QRTL'].replace([-8, -9], np.nan)

    # 4. One-hot encode PAY1 and ZIPINC_QRTL
    print("  - One-hot encoding PAY1 and ZIPINC_QRTL...")
    test_data = pd.get_dummies(test_data, columns=['PAY1', 'ZIPINC_QRTL'],
                                prefix=['PAY1', 'ZIPINC_QRTL'])

    # 5. Hard-code expected columns (from training data)
    # PAY1 has values 1-6, ZIPINC_QRTL has values 1-4
    expected_pay1_columns = ['PAY1_1.0', 'PAY1_2.0', 'PAY1_3.0', 'PAY1_4.0', 'PAY1_5.0', 'PAY1_6.0']
    expected_zipinc_columns = ['ZIPINC_QRTL_1.0', 'ZIPINC_QRTL_2.0', 'ZIPINC_QRTL_3.0', 'ZIPINC_QRTL_4.0']

    print(f"  - Expected PAY1 columns: {len(expected_pay1_columns)}")
    print(f"  - Expected ZIPINC_QRTL columns: {len(expected_zipinc_columns)}")

    # Add missing columns with zeros if they don't exist in test data
    for col in expected_pay1_columns:
        if col not in test_data.columns:
            test_data[col] = 0
            print(f"    Added missing column: {col}")

    for col in expected_zipinc_columns:
        if col not in test_data.columns:
            test_data[col] = 0
            print(f"    Added missing column: {col}")

    # Use the expected columns in the correct order
    pay1_columns = expected_pay1_columns
    zipinc_qrtl_columns = expected_zipinc_columns

    print(f"  - Final PAY1 columns: {len(pay1_columns)}")
    print(f"  - Final ZIPINC_QRTL columns: {len(zipinc_qrtl_columns)}")

    # 6. Extract features
    X_test = test_data[['AGE', 'FEMALE'] + pay1_columns + zipinc_qrtl_columns + icd_columns]

    # Handle any remaining missing values
    X_test = X_test.dropna()
    test_data = test_data.loc[X_test.index]

    # Keep track of valid indices for mapping back to original data
    valid_indices = X_test.index

    print(f"\nFinal test data shape: {X_test.shape}")
    print(f"Number of rows after preprocessing: {len(valid_indices)}")

    # Prepare inputs for the model
    print("\nPreparing model inputs...")
    inputs = [
        X_test[icd_columns],
        X_test['AGE'],
        X_test['FEMALE'],
    ] + [X_test[c] for c in pay1_columns] \
      + [X_test[c] for c in zipinc_qrtl_columns]

    # Run predictions
    print("\nRunning predictions...")
    y_pred_prob = model.predict(inputs, batch_size=1, verbose=0).squeeze()
    print(f"  Predictions shape: {y_pred_prob.shape}")
    # Apply Youden threshold for binary predictions
    print(f"\nApplying Youden threshold ({YOUDEN_THRESHOLD})...")
    y_pred_binary = (y_pred_prob > YOUDEN_THRESHOLD).astype(int)

    # Add predictions to the ORIGINAL dataframe (not preprocessed)
    # Only add predictions for rows that passed preprocessing
    test_data_with_pred = test_data_original.copy()
    test_data_with_pred['predicted_probability'] = np.nan  # Initialize with NaN
    test_data_with_pred['predicted_class'] = np.nan

    # Fill in predictions only for valid rows
    test_data_with_pred.loc[valid_indices, 'predicted_probability'] = y_pred_prob
    test_data_with_pred.loc[valid_indices, 'predicted_class'] = y_pred_binary

    # Calculate summary statistics
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    print(f"\nProbability statistics:")
    print(f"  Min: {y_pred_prob.min():.4f}")
    print(f"  Max: {y_pred_prob.max():.4f}")
    print(f"  Mean: {y_pred_prob.mean():.4f}")
    print(f"  Median: {np.median(y_pred_prob):.4f}")

    print(f"\nBinary predictions (using threshold={YOUDEN_THRESHOLD}):")
    print(f"  Predicted negative (0): {(y_pred_binary == 0).sum()}")
    print(f"  Predicted positive (1): {(y_pred_binary == 1).sum()}")

    # If true labels are available, compute metrics (only for rows with predictions)
    if OUTCOME_VAR in test_data_with_pred.columns:
        # Only compute metrics for rows that have predictions (valid_indices)
        y_true = test_data_with_pred.loc[valid_indices, OUTCOME_VAR].values
        y_pred_prob_for_metrics = test_data_with_pred.loc[valid_indices, 'predicted_probability'].values
        y_pred_binary_for_metrics = test_data_with_pred.loc[valid_indices, 'predicted_class'].values.astype(int)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

        accuracy = accuracy_score(y_true, y_pred_binary_for_metrics)
        precision = precision_score(y_true, y_pred_binary_for_metrics, zero_division=0)
        recall = recall_score(y_true, y_pred_binary_for_metrics, zero_division=0)
        f1 = f1_score(y_true, y_pred_binary_for_metrics, zero_division=0)

        # F2 score (weights recall higher than precision)
        beta = 2.0
        f2 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (precision + recall) > 0 else 0

        # AUC
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_pred_prob_for_metrics)
        else:
            auc = np.nan

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary_for_metrics).ravel()

        print(f"\nPerformance Metrics (on test set):")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  F2 Score:  {f2:.4f}")
        if not np.isnan(auc):
            print(f"  AUC:       {auc:.4f}")

        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  True Positives:  {tp}")


    # Save results
    print(f"\nSaving predictions to {OUTPUT_FILE}...")
    test_data_with_pred.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nOutput file contains:")
    print(f"  - All original columns from input (AGE, ICD codes, etc. in original format)")
    print(f"  - predicted_probability: Raw model output (0-1)")
    print(f"  - predicted_class: Binary prediction using threshold={YOUDEN_THRESHOLD}")
    print(f"\nNote: Rows that were dropped during preprocessing have NaN for predictions.")

if __name__ == "__main__":
    main()
