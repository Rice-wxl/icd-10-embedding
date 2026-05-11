"""
Script to run predictions for a single patient using command-line arguments.

Usage:
    python predict_single_patient.py \\
        --model-path Model/readmit_hypertrial_auc.keras \\
        --age 65 \\
        --female 1 \\
        --pay1 1 \\
        --zipinc-qrtl 2 \\
        --icd-codes I10 E119 I2510 \\
        --threshold 0.5022

Arguments:
    --model-path: Path to the trained .keras model file (required)
    --age: Patient age in years (required)
    --female: Binary indicator (0=male, 1=female) (required)
    --pay1: Primary payer code (1-6, or omit for missing)
    --zipinc-qrtl: Income quartile (1-4, or omit for missing)
    --icd-codes: Space-separated list of ICD-10 diagnosis codes (up to 40)
    --threshold: Classification threshold (default: 0.5)
    --encoder-path: Path to label encoder (default: Model/full_label_encoder.pkl)
    --scaler-path: Path to age scaler (default: Model/full_age_scaler.pkl)
"""

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.saving import load_model
from keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention
from keras.models import Model
import pickle
import sys

from config import LABEL_ENCODER_PATH, AGE_SCALER_PATH

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
# PREDICTION FUNCTIONS
# ============================================

def preprocess_patient_data(age, female, pay1, zipinc_qrtl, icd_codes, encoder, age_scaler, icd_only):
    """
    Preprocess a single patient's data for model prediction.

    Args:
        age: Patient age (numeric)
        female: Binary indicator (0 or 1)
        pay1: Primary payer (1-6, or None)
        zipinc_qrtl: Income quartile (1-4, or None)
        icd_codes: List of ICD-10 diagnosis codes (up to 40)
        encoder: Fitted LabelEncoder for ICD codes
        age_scaler: Fitted MinMaxScaler for age

    Returns:
        List of preprocessed inputs ready for model.predict()
    """
    # Define ICD column names
    icd_columns = [f'I10_DX{i}' for i in range(1, 41)]

    # Create a dataframe with a single row
    data = {}

    # Add age and female
    data['AGE'] = [age]
    data['FEMALE'] = [female]

    # Add PAY1 and ZIPINC_QRTL (could be None/NaN)
    data['PAY1'] = [pay1]
    data['ZIPINC_QRTL'] = [zipinc_qrtl]

    # Add ICD codes (pad with 'NAN' if fewer than 40 codes provided)
    for i, col in enumerate(icd_columns):
        if i < len(icd_codes):
            data[col] = [icd_codes[i]]
        else:
            data[col] = ['NAN']

    df = pd.DataFrame(data)

    # Preprocess
    # 1. Encode ICD codes
    label_to_int = {label: idx for idx, label in enumerate(encoder.classes_)}
    unknown_label_int = encoder.transform(["NAN"])[0]

    for col in icd_columns:
        df[col] = df[col].astype(str).str.upper()
        df[col] = df[col].map(label_to_int).fillna(unknown_label_int).astype(int)

    # 2. Normalize age
    df['AGE'] = age_scaler.transform(df[['AGE']])

    # 3. Handle missing value codes in PAY1 and ZIPINC_QRTL
    df['PAY1'] = df['PAY1'].replace([-8, -9], np.nan)
    df['ZIPINC_QRTL'] = df['ZIPINC_QRTL'].replace([-8, -9], np.nan)
    print(f"pay1 before dummies: {df['PAY1']}")
    print(f"zipinc before dummies: {df['ZIPINC_QRTL']}")
    # 4. One-hot encode PAY1 and ZIPINC_QRTL
    df = pd.get_dummies(df, columns=['PAY1', 'ZIPINC_QRTL'],
                        prefix=['PAY1', 'ZIPINC_QRTL'])
    print(f"after dummies: {df.columns}")
    # 5. Ensure all expected columns exist
    expected_pay1_columns = ['PAY1_1.0', 'PAY1_2.0', 'PAY1_3.0', 'PAY1_4.0', 'PAY1_5.0', 'PAY1_6.0']
    expected_zipinc_columns = ['ZIPINC_QRTL_1.0', 'ZIPINC_QRTL_2.0', 'ZIPINC_QRTL_3.0', 'ZIPINC_QRTL_4.0']

    for col in expected_pay1_columns:
        if col not in df.columns:
            df[col] = 0

    for col in expected_zipinc_columns:
        if col not in df.columns:
            df[col] = 0

    # 6. Extract features in the correct order
    X = df[['AGE', 'FEMALE'] + expected_pay1_columns + expected_zipinc_columns + icd_columns]

    # 7. Prepare inputs for the model
    if icd_only:
        inputs = [
        X[icd_columns],
        ]
    else: 
        inputs = [
            X[icd_columns],
            X['AGE'],
            X['FEMALE'],
        ] + [X[c] for c in expected_pay1_columns] \
        + [X[c] for c in expected_zipinc_columns]

    print(f"all inputs: ")
    print(f"age: {X['AGE']}")
    print(f"female: {X['FEMALE']}")
    print(f"pay1: {[X[c] for c in expected_pay1_columns]}")
    print(f"zipinc: {[X[c] for c in expected_zipinc_columns]}")

    return inputs

def predict_patient(model, inputs, threshold=0.5):
    """
    Run prediction for a single patient.

    Args:
        model: Loaded Keras model
        inputs: Preprocessed inputs
        threshold: Classification threshold (default 0.5)

    Returns:
        dict with 'probability' and 'prediction' keys
    """
    # Run prediction
    y_pred_prob = model.predict(inputs, batch_size=1, verbose=0).squeeze()

    # Apply threshold
    y_pred_binary = int(y_pred_prob > threshold)

    return {
        'probability': float(y_pred_prob),
        'prediction': y_pred_binary
    }

# ============================================
# MAIN CLI INTERFACE
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description='Predict healthcare outcomes for a single patient using ICD codes and demographics.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic prediction with 3 ICD codes
  python predict_single_patient.py \\
      --model-path Model/readmit_hypertrial_auc.keras \\
      --age 65 --female 1 --pay1 1 --zipinc-qrtl 2 \\
      --icd-codes I10 E119 I2510

  # Prediction with missing payer/income data
  python predict_single_patient.py \\
      --model-path Model/mortality_model.keras \\
      --age 72 --female 0 \\
      --icd-codes I10 I509 E119 N179 \\
      --threshold 0.45

  # Many ICD codes
  python predict_single_patient.py \\
      --model-path Model/model.keras \\
      --age 55 --female 1 --pay1 2 --zipinc-qrtl 3 \\
      --icd-codes I10 E119 I2510 I509 J449 N179 E785 Z794 I739 M109 \\
      --threshold 0.5022
        """
    )

    # Required arguments
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the trained .keras model file')
    parser.add_argument('--age', type=float, required=True,
                        help='Patient age in years')
    parser.add_argument('--female', type=int, required=True, choices=[0, 1],
                        help='Sex indicator: 0=male, 1=female')
    parser.add_argument('--icd-only', action='store_true',
                        help='icd_only models')

    # Optional demographic arguments
    parser.add_argument('--pay1', type=int, choices=[1, 2, 3, 4, 5, 6],
                        help='Primary payer code (1-6). Omit if missing.')
    parser.add_argument('--zipinc-qrtl', type=int, choices=[1, 2, 3, 4],
                        help='ZIP code income quartile (1-4). Omit if missing.')

    # ICD codes
    parser.add_argument('--icd-codes', nargs='+', default=[],
                        help='Space-separated list of ICD-10 diagnosis codes (up to 40 codes)')

    # Model configuration
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold (default: 0.5)')
    parser.add_argument('--encoder-path', type=str, default=str(LABEL_ENCODER_PATH),
                        help='Path to ICD label encoder pickle file')
    parser.add_argument('--scaler-path', type=str, default=str(AGE_SCALER_PATH),
                        help='Path to age scaler pickle file')

    # Output options
    parser.add_argument('--quiet', action='store_true',
                        help='Only output the prediction result (no verbose logging)')

    args = parser.parse_args()

    # Validate ICD codes count
    if len(args.icd_codes) > 40:
        print(f"ERROR: Too many ICD codes provided ({len(args.icd_codes)}). Maximum is 40.", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print("=" * 60)
        print("Single Patient Prediction")
        print("=" * 60)
        print(f"\nInput:")
        print(f"  Age: {args.age}")
        print(f"  Female: {args.female}")
        print(f"  PAY1: {args.pay1 if args.pay1 is not None else 'Missing'}")
        print(f"  ZIPINC_QRTL: {args.zipinc_qrtl if args.zipinc_qrtl is not None else 'Missing'}")
        print(f"  ICD codes ({len(args.icd_codes)}): {', '.join(args.icd_codes) if args.icd_codes else 'None'}")
        print(f"\nConfiguration:")
        print(f"  Model: {args.model_path}")
        print(f"  Threshold: {args.threshold}")
        print()

    # Load model
    if not args.quiet:
        print("Loading model...")
    try:
        model = load_model(args.model_path)
        if not args.quiet:
            print(f"  Model loaded: {model.name}")
    except Exception as e:
        print(f"ERROR: Failed to load model from {args.model_path}", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        sys.exit(1)

    # Load encoders
    if not args.quiet:
        print("\nLoading encoders...")
    try:
        with open(args.encoder_path, 'rb') as f:
            encoder = pickle.load(f)
        with open(args.scaler_path, 'rb') as f:
            age_scaler = pickle.load(f)
        if not args.quiet:
            print(f"  ICD encoder: {len(encoder.classes_)} unique codes")
            print(f"  Age scaler: loaded")
    except Exception as e:
        print(f"ERROR: Failed to load encoders", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        sys.exit(1)

    # Preprocess data
    if not args.quiet:
        print("\nPreprocessing patient data...")
    try:
        inputs = preprocess_patient_data(
            age=args.age,
            female=args.female,
            pay1=args.pay1,
            zipinc_qrtl=args.zipinc_qrtl,
            icd_codes=args.icd_codes,
            encoder=encoder,
            age_scaler=age_scaler,
            icd_only=args.icd_only,
        )
    except Exception as e:
        print(f"ERROR: Failed to preprocess data", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        sys.exit(1)

    # Run prediction
    if not args.quiet:
        print("\nRunning prediction...")
    try:
        result = predict_patient(model, inputs, threshold=args.threshold)
    except Exception as e:
        print(f"ERROR: Failed to run prediction", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        sys.exit(1)

    # Output results
    if not args.quiet:
        print("\n" + "=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)
        print(f"\Raw probability: {result['probability']:.6f}")

        ## Readmit beta
        # beta = 0.139050
        ## Mort beta
        beta = 0.003877

        prob_actual = tf.clip_by_value(result['probability'], 1e-8, 1-1e-8, name=None)
        y_pred_calibrated = prob_actual / (prob_actual + (1 - prob_actual) / beta)

        print(f"\Adjusted probability: {y_pred_calibrated:.6f}")

        print(f"  ✓ Calibration applied (beta={beta:.6f})")
        print(f"Predicted class (threshold={args.threshold}): {result['prediction']}")
        print()

        if result['prediction'] == 1:
            print(f"⚠️  HIGH RISK - Predicted positive outcome")
        else:
            print(f"✓  LOW RISK - Predicted negative outcome")
        print()
    else:
        # Quiet mode: just output the key results
        print(f"{result['probability']:.6f}\t{result['prediction']}")

    return result

if __name__ == "__main__":
    main()
