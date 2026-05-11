"""
Actually run both preprocessing methods and make predictions to see the difference
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from keras.saving import load_model
from keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention
from keras.models import Model
import pickle

from config import MODEL_DIR, LABEL_ENCODER_PATH, AGE_SCALER_PATH, SMALL_DATASET_DIR

# Register custom components
@tf.keras.utils.register_keras_serializable(package="Custom")
def f2_score(y_true, y_pred):
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

# Load model
print("Loading model...")
model = load_model(MODEL_DIR / 'readmit_hypertrial_auc.keras')

# Load encoders
print("Loading encoders...")
with open(LABEL_ENCODER_PATH, 'rb') as f:
    encoder = pickle.load(f)
with open(AGE_SCALER_PATH, 'rb') as f:
    age_scaler = pickle.load(f)

# Load CSV
csv_file = SMALL_DATASET_DIR / 'small_dataset_rea30_custom.csv'
df_original = pd.read_csv(csv_file)
df_original.columns = df_original.columns.str.upper()

# Define columns
icd_columns = [f'I10_DX{i}' for i in range(1, 41)]
expected_pay1_columns = ['PAY1_1.0', 'PAY1_2.0', 'PAY1_3.0', 'PAY1_4.0', 'PAY1_5.0', 'PAY1_6.0']
expected_zipinc_columns = ['ZIPINC_QRTL_1.0', 'ZIPINC_QRTL_2.0', 'ZIPINC_QRTL_3.0', 'ZIPINC_QRTL_4.0']

# METHOD 1: CSV method (predict_on_small_dataset.py)
print("\n" + "="*60)
print("METHOD 1: CSV method")
print("="*60)

df_csv = df_original.copy()

label_to_int = {label: idx for idx, label in enumerate(encoder.classes_)}
unknown_label_int = encoder.transform(["NAN"])[0]

for col in icd_columns:
    df_csv[col] = df_csv[col].astype(str).str.upper()
    df_csv[col] = df_csv[col].map(label_to_int).fillna(unknown_label_int).astype(int)

df_csv['AGE'] = age_scaler.transform(df_csv[['AGE']])
df_csv['PAY1'] = df_csv['PAY1'].replace([-8, -9], np.nan)
df_csv['ZIPINC_QRTL'] = df_csv['ZIPINC_QRTL'].replace([-8, -9], np.nan)
df_csv = pd.get_dummies(df_csv, columns=['PAY1', 'ZIPINC_QRTL'], prefix=['PAY1', 'ZIPINC_QRTL'])
print(f"after preprocessing: {df_csv.columns}")

for col in expected_pay1_columns + expected_zipinc_columns:
    if col not in df_csv.columns:
        df_csv[col] = 0

X_test = df_csv[['AGE', 'FEMALE'] + expected_pay1_columns + expected_zipinc_columns + icd_columns]

inputs_csv = [
    X_test[icd_columns],
    X_test['AGE'],
    X_test['FEMALE'],
] + [X_test[c] for c in expected_pay1_columns] \
  + [X_test[c] for c in expected_zipinc_columns]

print("Input types and shapes:")
for i, inp in enumerate(inputs_csv):
    print(f"  {i}: type={type(inp).__name__}, shape={inp.shape}, dtype={inp.dtype if hasattr(inp, 'dtype') else 'N/A'}")

y_pred_csv = model.predict(inputs_csv, batch_size=1, verbose=0).squeeze()
print(f"\nPrediction (CSV method): {y_pred_csv}")

# METHOD 2: Manual method (predict_single_patient.py)
print("\n" + "="*60)
print("METHOD 2: Manual method")
print("="*60)

icd_codes_list = ['I61']
data = {
    'AGE': [60],
    'FEMALE': [0],
    'PAY1': [1.0],
    'ZIPINC_QRTL': [2.0]
}

for i, col in enumerate(icd_columns):
    if i < len(icd_codes_list):
        data[col] = [icd_codes_list[i]]
    else:
        data[col] = ['NAN']

df_manual = pd.DataFrame(data)

for col in icd_columns:
    df_manual[col] = df_manual[col].astype(str).str.upper()
    df_manual[col] = df_manual[col].map(label_to_int).fillna(unknown_label_int).astype(int)

df_manual['AGE'] = age_scaler.transform(df_manual[['AGE']])
df_manual['PAY1'] = df_manual['PAY1'].replace([-8, -9], np.nan)
df_manual['ZIPINC_QRTL'] = df_manual['ZIPINC_QRTL'].replace([-8, -9], np.nan)
df_manual = pd.get_dummies(df_manual, columns=['PAY1', 'ZIPINC_QRTL'], prefix=['PAY1', 'ZIPINC_QRTL'])

for col in expected_pay1_columns + expected_zipinc_columns:
    if col not in df_manual.columns:
        df_manual[col] = 0

X_manual = df_manual[['AGE', 'FEMALE'] + expected_pay1_columns + expected_zipinc_columns + icd_columns]

inputs_manual = [
    X_manual[icd_columns],
    X_manual['AGE'],
    X_manual['FEMALE'],
] + [X_manual[c] for c in expected_pay1_columns] \
  + [X_manual[c] for c in expected_zipinc_columns]

print(f"all inputs: ")
print(f"age: {X_manual['AGE']}")
print(f"female: {X_manual['FEMALE']}")
print(f"pay1: {[X_manual[c] for c in expected_pay1_columns]}")
print(f"zipinc: {[X_manual[c] for c in expected_zipinc_columns]}")

y_pred_manual = model.predict(inputs_manual, batch_size=1, verbose=0).squeeze()
print(f"\nPrediction (Manual method): {y_pred_manual}")

# COMPARE
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"CSV method prediction:    {y_pred_csv:.6f}")
print(f"Manual method prediction: {y_pred_manual:.6f}")
print(f"Difference:               {abs(y_pred_csv - y_pred_manual):.6f}")

if abs(y_pred_csv - y_pred_manual) < 0.0001:
    print("\n✓ Predictions match!")
else:
    print("\n✗ Predictions differ!")

    # Check for data type differences
    print("\nDetailed input comparison:")
    for i in range(len(inputs_csv)):
        csv_arr = inputs_csv[i].values if hasattr(inputs_csv[i], 'values') else np.array(inputs_csv[i])
        manual_arr = inputs_manual[i].values if hasattr(inputs_manual[i], 'values') else np.array(inputs_manual[i])

        if not np.array_equal(csv_arr, manual_arr):
            print(f"  Input {i} differs:")
            print(f"    CSV:    {csv_arr}, dtype={csv_arr.dtype}")
            print(f"    Manual: {manual_arr}, dtype={manual_arr.dtype}")
