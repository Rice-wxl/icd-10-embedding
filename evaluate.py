import pandas as pd
import numpy as np
import tensorflow as tf
from keras.saving import load_model
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_curve
)
# from tqdm import tqdm
import pickle
import keras
from keras.layers import Input, Dense, Dropout, BatchNormalization, Embedding, Flatten, concatenate, MultiHeadAttention, LayerNormalization, Add
from keras.models import Model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from config import DATA_DIR, NRD_2021_TEST, NRD_2022_TEST

@tf.keras.utils.register_keras_serializable(package="Custom")
def f2_score(y_true, y_pred):
    # Ensure inputs are tensors
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    # Calculate true positives, false positives, and false negatives
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    # Calculate precision and recall
    epsilon = tf.keras.backend.epsilon()  # Small constant to avoid division by zero
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    # Calculate F2 score
    f2 = (5 * precision * recall) / (4 * precision + recall + epsilon)
    return f2.numpy()


# @tf.keras.utils.register_keras_serializable(package="Custom")
# class DeepSet(tf.keras.Model):
#     def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
#         super(DeepSet, self).__init__(**kwargs)
        
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
#         # Element-wise transformation: phi network
#         self.phi = tf.keras.Sequential([
#             Dense(self.hidden_dim, activation='relu'),
#             Dense(self.hidden_dim, activation='relu')
#         ])
#         # Post-aggregation transformation: rho network
#         self.rho = tf.keras.Sequential([
#             Dense(self.hidden_dim, activation='relu'),
#             Dense(self.output_dim, activation='relu')
#         ])

    
#     def call(self, x):
#         # Apply phi to each ICD code embedding
#         transformed = self.phi(x)  # Shape: (batch_size, num_codes, output_dim)
#         # Aggregate using sum (or other aggregation functions)
#         aggregated = tf.reduce_sum(transformed, axis=1)  # Shape: (batch_size, output_dim)
#         # Apply rho to the aggregated representation
#         output = self.rho(aggregated)  # Shape: (batch_size, output_dim)
#         return output

#     def get_config(self):
#         # Return the parameters required for serialization
#         config = super(DeepSet, self).get_config()
#         config.update({
#             "input_dim": self.input_dim,
#             "hidden_dim": self.hidden_dim,
#             "output_dim": self.output_dim
#         })
#         return config

#     @classmethod
#     def from_config(cls, config):
#         # Recreate the layer from its config
#         return cls(**config)

@tf.keras.utils.register_keras_serializable(package="Custom")
class DeepSet(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, output_dim, num_encode, num_decode, **kwargs):
        super(DeepSet, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_encode = num_encode
        self.num_decode = num_decode
        # # Element-wise transformation: phi network
        # self.phi = tf.keras.Sequential([
        #     Dense(self.hidden_dim, activation='relu')
        # ])
        # # Post-aggregation transformation: rho network
        # self.rho = tf.keras.Sequential([
        #     Dense(self.output_dim, activation='relu')
        # ])

        # Element-wise transformation: phi network
        self.phi = tf.keras.Sequential([
            Dense(self.hidden_dim, activation='relu') for _ in range(self.num_encode)
        ])

        # Post-aggregation transformation: rho network
        self.rho = tf.keras.Sequential([
            Dense(self.hidden_dim, activation='relu') for _ in range(self.num_decode - 1)
        ] + [Dense(self.output_dim, activation='relu')])  # Last layer should output correct dimension

    def call(self, x):
        transformed = self.phi(x)  # (batch_size, num_codes, hidden_dim)
        aggregated = tf.reduce_sum(transformed, axis=1)  # (batch_size, hidden_dim)
        output = self.rho(aggregated)  # (batch_size, output_dim)
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

import tensorflow as tf
from keras.layers import Input, Dense, Dropout, BatchNormalization, Embedding, Flatten, concatenate, MultiHeadAttention, LayerNormalization, Add
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.metrics import AUC, Precision, Recall
import keras.backend as K

@tf.keras.utils.register_keras_serializable(package="Custom")
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
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
        # Recreate the layer from its config
        return cls(**config)





# Load the trained model
MODEL_PATH = 'Model/readmit_auc_icd_only.keras'
model = load_model(MODEL_PATH)
print(f"Model: {MODEL_PATH}")

# Load the LabelEncoder for ICD codes
with open('Model/full_label_encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

# Load the MinMaxScaler for 'AGE'
with open('Model/full_age_scaler.pkl', 'rb') as file:
    age_scaler = pickle.load(file)


X_validate = pd.read_csv(DATA_DIR / 'readmit' / 'X_test.csv')
y_validate = pd.read_csv(DATA_DIR / 'readmit' / 'y_test.csv')


file_2021 = NRD_2021_TEST
file_2022 = NRD_2022_TEST

# Read and combine files
df1 = pd.read_csv(file_2021)
df2 = pd.read_csv(file_2022)

# Combine rows from both files
new_data = pd.concat([df1, df2], ignore_index=True)

# Convert all column names to uppercase
new_data.columns = new_data.columns.str.upper()

# Define the outcome variable and file path
outcome_var = 'REA30'  # Set outcome variable here, e.g., 'DIED', 'MOR30', 'REA30'

# Filter out observations where DIED == 1 if outcome_var is REA30
# if outcome_var == 'REA30' and 'DIED' in new_data.columns:
if 'DIED' in new_data.columns:
    new_data = new_data[new_data['DIED'] != 1]

# Handle missing values in the target variable
new_data = new_data.dropna(subset=[outcome_var])

# Define ICD columns
icd_columns = [f'I10_DX{i}' for i in range(1, 41)]

# Encode ICD codes
label_to_int = {label: idx for idx, label in enumerate(encoder.classes_)}
unknown_label_int = encoder.transform(["NAN"])[0]  # Assign unknown codes to index 0

for col in icd_columns:
    new_data[col] = new_data[col].astype(str).str.upper()  # Convert to uppercase
    new_data[col] = new_data[col].map(label_to_int).fillna(unknown_label_int).astype(int)

# Normalize 'AGE'
new_data['AGE'] = age_scaler.transform(new_data[['AGE']])


# ============================================
# NEW: Replace negative values with NaN
# ============================================
# print("\n=== HANDLING MISSING VALUE CODES ===")
# print(f"Before replacement:")
# print(f"  PAY1 unique values: {sorted(new_data['PAY1'].dropna().unique())}")
# print(f"  PAY1 NaN count: {new_data['PAY1'].isna().sum()}")
# print(f"  ZIPINC_QRTL unique values: {sorted(new_data['ZIPINC_QRTL'].dropna().unique())}")
# print(f"  ZIPINC_QRTL NaN count: {new_data['ZIPINC_QRTL'].isna().sum()}")

# Replace -8 and -9 with NaN in PAY1 and ZIPINC_QRTL
new_data['PAY1'] = new_data['PAY1'].replace([-8, -9], np.nan)
new_data['ZIPINC_QRTL'] = new_data['ZIPINC_QRTL'].replace([-8, -9], np.nan)

# print(f"\nAfter replacement:")
# print(f"  PAY1 unique values: {sorted(new_data['PAY1'].dropna().unique())}")
# print(f"  PAY1 NaN count: {new_data['PAY1'].isna().sum()}")
# print(f"  ZIPINC_QRTL unique values: {sorted(new_data['ZIPINC_QRTL'].dropna().unique())}")
# print(f"  ZIPINC_QRTL NaN count: {new_data['ZIPINC_QRTL'].isna().sum()}")


# One-hot encode 'PAY1' and 'ZIPINC_QRTL' only (excluding 'RACE')
new_data = pd.get_dummies(new_data, columns=['PAY1', 'ZIPINC_QRTL'], prefix=['PAY1', 'ZIPINC_QRTL'])

# Ensure that all expected one-hot encoded columns are present
def ensure_columns(data, expected_columns):
    for col in expected_columns:
        if col not in data.columns:
            data[col] = 0
    return data

# Define one-hot encoded columns for 'PAY1' and 'ZIPINC_QRTL' based on training data
pay1_columns = [col for col in new_data.columns if col.startswith('PAY1_')]
# print(f"all payment column names: {pay1_columns}")
zipinc_qrtl_columns = [col for col in new_data.columns if col.startswith('ZIPINC_QRTL_')]
# print(f"all zipinc quartile column names: {zipinc_qrtl_columns}")


# Ensure all expected columns are present
new_data = ensure_columns(new_data, pay1_columns + zipinc_qrtl_columns)

# Drop rows with missing values
X_new = new_data[['AGE', 'FEMALE'] + pay1_columns + zipinc_qrtl_columns + icd_columns]
X_new = X_new.dropna()
new_data = new_data.loc[X_new.index]


y = new_data[outcome_var].to_numpy()
idx_all = np.arange(len(new_data))
def stratified_pick(idx, y_sub, frac, seed):
    rng = np.random.default_rng(seed)
    picked = []
    for cls in np.unique(y_sub):
        cls_idx = idx[y_sub == cls]
        n = max(1, int(np.floor(frac * len(cls_idx))))
        pick = rng.choice(cls_idx, size=n, replace=False)
        picked.append(pick)
    return np.concatenate(picked)

idx10 = stratified_pick(idx_all, y, frac=0.10, seed=42)       # 10%
df10 = new_data.iloc[idx10].copy()


# # Separate positive and negative rows
# positives = new_data[new_data[outcome_var] == 1]
# negatives = new_data[new_data[outcome_var] == 0]

# # Sample 10% from each
# sampled_positives = positives.sample(frac=0.1, random_state=42)
# sampled_negatives = negatives.sample(frac=0.1, random_state=42)

# # Combine the samples
# combined_sampled_data = pd.concat([sampled_positives, sampled_negatives])

# # Shuffle the combined data
# new_data = shuffle(combined_sampled_data, random_state=42)

# Prepare input features. 
X_new = df10[['AGE', 'FEMALE'] + pay1_columns + zipinc_qrtl_columns + icd_columns]
y_true = df10[outcome_var].to_numpy(np.int32)


# inputs = [
#     X_new[icd_columns],
#     X_new['AGE'],
#     X_new['FEMALE'],
# ] + [X_new[c] for c in pay1_columns] \
#   + [X_new[c] for c in zipinc_qrtl_columns]


# val_inputs = [
#     X_validate[icd_columns],       # embedding indices
#     X_validate['AGE'],
#     X_validate['FEMALE'],
# ] + [X_validate[c] for c in pay1_columns] \
#   + [X_validate[c] for c in zipinc_qrtl_columns]

inputs = [
    X_new[icd_columns],       # embedding indices
]

val_inputs = [
    X_validate[icd_columns],       # embedding indices
]

# --- 1) one-shot predict with built-in batching (biggest win) ---
y_pred_prob_val = model.predict(val_inputs, batch_size=1024, verbose=2).squeeze()
y_pred_prob = model.predict(inputs, batch_size=1024, verbose=2).squeeze()

# ---------- helpers ----------
def auc_ci_hm(y, p, ci=0.95):
    y = np.asarray(y, dtype=np.int8).reshape(-1)
    p = np.asarray(p, dtype=np.float32).reshape(-1) 
    A = roc_auc_score(y, p)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    Q1 = A / (2 - A)
    Q2 = 2 * A * A / (1 + A)
    var = (A*(1-A) + (n1-1)*(Q1 - A*A) + (n0-1)*(Q2 - A*A)) / (n1*n0)
    se = np.sqrt(max(var, 0.0))
    z = 1.96  # 95%
    return max(0.0, A - z*se), min(1.0, A + z*se)

def counts_from_preds(y_true, y_pred):
    y_true = y_true.astype(np.int8).reshape(-1)
    y_pred = y_pred.astype(np.int8).reshape(-1)
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP, FP, FN, TN

def metrics_from_counts(TP, FP, FN, TN):
    denom = TP + FP + FN + TN
    accuracy  = (TP + TN) / denom if denom else 0.0
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = (2*precision*recall) / (precision + recall) if (precision + recall) else 0.0
    beta2 = 4.0  # beta=2
    f2 = ((1+beta2)*precision*recall) / (beta2*precision + recall) if (beta2*precision + recall) else 0.0
    return accuracy, precision, recall, f1, f2


def plot_multi_roc(main_y, main_p,
                   baseline_curves,
                   title="ROC Comparison",
                   outfile="roc_comparison.png",
                   auc_value=None, auc_ci=None):
    """
    baseline_curves: list of dicts, each with:
      {
        'name': str,
        'y_true': array-like,
        'scores': array-like,
        'marker_threshold': float  # e.g., 0 for your indices
      }
    """
    plt.figure(figsize=(7, 6), dpi=140)

    # --- Main model ---
    main_fpr, main_tpr, main_thr = roc_curve(main_y, main_p)
    main_auc = roc_auc_score(main_y, main_p) if auc_value is None else auc_value
    main_label = f"ICD model (AUC={main_auc:.3f})"
    if auc_ci is not None:
        lo, hi = auc_ci
        main_label = f"ICD model (AUC={main_auc:.3f}, 95% CI [{lo:.3f}, {hi:.3f}])"
    plt.plot(main_fpr, main_tpr, lw=2.5, label=main_label)

    # --- Baselines ---
    for b in baseline_curves:
        yb = b["y_true"]
        sb = b["scores"]
        name = b["name"]
        auc_lower = b['AUC Lower CI']
        auc_upper = b['AUC Upper CI']

        # Skip if only one class (safety)
        if len(np.unique(yb)) < 2:
            continue

        fpr, tpr, thr = roc_curve(yb, sb)
        auc_b = roc_auc_score(yb, sb)
        plt.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC={auc_b:.3f}, 95% CI [{auc_lower:.3f}, {auc_upper:.3f}])")

    # Chance line
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()
    return outfile


# def plot_roc_with_threshold(y_true, y_prob, best_threshold, auc_value=None, auc_ci=None, outfile="roc_curve.png"):
#     # Compute ROC points
#     fpr, tpr, thresholds = roc_curve(y_true, y_prob)

#     # Find the ROC point nearest your chosen threshold
#     # Note: thresholds[0] can be inf; shift if needed
#     diffs = np.abs(thresholds - best_threshold)
#     idx = np.argmin(diffs)
#     if np.isinf(thresholds[idx]) and len(thresholds) > 1:
#         idx = 1  # avoid the 'inf' threshold point

#     # Plot
#     plt.figure(figsize=(6, 5), dpi=140)
#     plt.plot(fpr, tpr, lw=2)
#     plt.plot([0, 1], [0, 1], linestyle="--")  # chance line

#     # Mark chosen threshold
#     plt.scatter([fpr[idx]], [tpr[idx]], s=50, zorder=5)
#     plt.text(fpr[idx], tpr[idx], f"  τ={best_threshold:.3f}", va="center")

#     # Title with AUC (and CI if provided)
#     if auc_value is not None and auc_ci is not None:
#         lo, hi = auc_ci
#         plt.title(f"ROC Curve (AUC={auc_value:.3f}, 95% CI [{lo:.3f}, {hi:.3f}])")
#     elif auc_value is not None:
#         plt.title(f"ROC Curve (AUC={auc_value:.3f})")
#     else:
#         plt.title("ROC Curve")

#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.tight_layout()
#     plt.savefig(outfile, bbox_inches="tight")
#     plt.close()
#     return outfile


print("Calculating the best Youden threshold using the validation data")

# =========================
# 1) FAST Youden threshold
# =========================
if len(np.unique(y_validate)) > 1:
    y = np.asarray(y_validate, dtype=np.int8).reshape(-1)
    p = np.asarray(y_pred_prob_val, dtype=np.float32).reshape(-1)

    # Exact Youden optimum from ROC (Youden = TPR - FPR)
    fpr, tpr, thr = roc_curve(y, p)
    mask = np.isfinite(thr)         # drop the initial 'inf' threshold
    youden = tpr[mask] - fpr[mask]  # vectorized
    i = int(np.argmax(youden))

    best_threshold = float(thr[mask][i])
    best_value     = float(youden[i])
    sensitivity    = float(tpr[mask][i])
    specificity    = float(1.0 - fpr[mask][i])

    print(f"Best Youden index threshold: {best_threshold}")
    print(f"Best Youden index value: {best_value:.6f}")
    print(f"Sensitivity at best: {sensitivity:.4f}")
    print(f"Specificity at best: {specificity:.4f}")

    # Metrics at optimal threshold (single pass)
    y_pred_optimal = (p > best_threshold).astype(np.int8)
    TP, FP, FN, TN = counts_from_preds(y, y_pred_optimal)
    acc, prec, rec, f1, f2 = metrics_from_counts(TP, FP, FN, TN)

    auc = roc_auc_score(y, p)
    auc_lower, auc_upper = auc_ci_hm(y, p)

    print(f"AUC: {auc:.4f}")
    print(f"95% CI for AUC: [{auc_lower:.4f}, {auc_upper:.4f}]")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"F2 Score: {f2:.4f}")


print("Using the threhsold on the test data...")
if len(np.unique(y_true)) > 1:
    y_main = np.asarray(y_true, dtype=np.int8).reshape(-1)
    p_main = np.asarray(y_pred_prob, dtype=np.float32).reshape(-1)

    # Metrics at optimal threshold (single pass)
    y_pred_optimal = (p_main > best_threshold).astype(np.int8)
    TP, FP, FN, TN = counts_from_preds(y_main, y_pred_optimal)
    acc, prec, rec, f1, f2 = metrics_from_counts(TP, FP, FN, TN)

    auc_main = roc_auc_score(y_main, p_main)
    auc_lower_main, auc_upper_main = auc_ci_hm(y_main, p_main)

    print(f"AUC: {auc_main:.4f}")
    print(f"95% CI for AUC: [{auc_lower_main:.4f}, {auc_upper_main:.4f}]")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"F2 Score: {f2:.4f}")

#     # --- Use right after your prints ---
#     roc_path = plot_roc_with_threshold(
#         y, p, best_threshold,
#         auc_value=auc,
#         auc_ci=(auc_lower, auc_upper),
#         outfile='mort_nodie_auc_graph.png'
# )


# =============================================
# 2) FAST validation of traditional score indexes
# =============================================
traditional_indexes = ['INDEX_MORTALITY', 'INDEX_READMISSION', 'CHARLINDEX', 'CHARLINDEX_AGE_ADJUST']
results = []
baseline_curves = []  # collect baselines for plotting

for index in traditional_indexes:
    print(f"Validating {index} against {outcome_var}...")

    if len(np.unique(y_validate)) > 1:
        y = np.asarray(y_validate, dtype=np.int8).reshape(-1)
        p = np.asarray(X_validate[index], dtype=np.float32).reshape(-1)
        # Exact Youden optimum from ROC (Youden = TPR - FPR)
        fpr, tpr, thr = roc_curve(y, p)
        mask = np.isfinite(thr)         # drop the initial 'inf' threshold
        youden = tpr[mask] - fpr[mask]  # vectorized
        i = int(np.argmax(youden))

        best_threshold = float(thr[mask][i])
        best_value     = float(youden[i])
        sensitivity    = float(tpr[mask][i])
        specificity    = float(1.0 - fpr[mask][i])

        print(f"Best Youden index threshold for {index}: {best_threshold}")
        print(f"Best Youden index value for {index}: {best_value:.6f}")
        print(f"Sensitivity at best for {index}: {sensitivity:.4f}")
        print(f"Specificity at best for {index}: {specificity:.4f}")

        # Metrics at optimal threshold (single pass)
        y_pred_optimal = (p > best_threshold).astype(np.int8)
        TP, FP, FN, TN = counts_from_preds(y, y_pred_optimal)
        acc, prec, rec, f1, f2 = metrics_from_counts(TP, FP, FN, TN)

        auc = roc_auc_score(y, p)
        auc_lower, auc_upper = auc_ci_hm(y, p)
    
    print("Using the threhsold on the test data...")
    y_true = np.asarray(y_true, dtype=np.int8).reshape(-1)
    scores = np.asarray(df10[index], dtype=np.float32).reshape(-1)

    if len(np.unique(y_true)) < 2:
        print(f"Skipped {index}: Only one class present in {outcome_var}")
        continue

    # Metrics at optimal threshold (single pass)
    y_pred_optimal = (scores > best_threshold).astype(np.int8)
    TP, FP, FN, TN = counts_from_preds(y_true, y_pred_optimal)
    acc, prec, rec, f1, f2 = metrics_from_counts(TP, FP, FN, TN)

    auc = roc_auc_score(y_true, scores)
    auc_lower, auc_upper = auc_ci_hm(y_true, scores)

    results.append({
        'Index': index,
        'AUC': auc,
        'AUC Lower CI': auc_lower,
        'AUC Upper CI': auc_upper,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'F2 Score': f2
    })
    
    if outcome_var == 'REA30':
        if index == "INDEX_READMISSION":
            baseline_curves.append({
                "name": "ECI",
                "y_true": y_true,
                "scores": scores,
                'AUC Lower CI': auc_lower,
                'AUC Upper CI': auc_upper,
            })
    else: 
        if index == "INDEX_MORTALITY":
            baseline_curves.append({
                "name": "ECI",
                "y_true": y_true,
                "scores": scores,
                'AUC Lower CI': auc_lower,
                'AUC Upper CI': auc_upper,
            })
    if index == "CHARLINDEX":
        baseline_curves.append({
            "name": "CCI",
            "y_true": y_true,
            "scores": scores,
            'AUC Lower CI': auc_lower,
            'AUC Upper CI': auc_upper,
        })
    if index == "CHARLINDEX_AGE_ADJUST":
        baseline_curves.append({
            "name": "CCI Age Adjusted",
            "y_true": y_true,
            "scores": scores,
            'AUC Lower CI': auc_lower,
            'AUC Upper CI': auc_upper,
        })

if results:
    results_df = pd.DataFrame(results).round(4)
    print("\nValidation Metrics for Traditional Scores:")
    print(results_df.to_string(index=False))
else:
    print("No metrics were calculated due to insufficient class diversity in the data.")

# # Finally, make the combined ROC figure
# out_path = plot_multi_roc(
#     main_y=y_main,
#     main_p=p_main,
#     baseline_curves=baseline_curves,
#     title=f"ROC Comparison vs. Traditional Scores (30-days Mortality)",
#     outfile="graph_mortality_nodie.png",
#     auc_value=auc_main,
#     auc_ci=(auc_lower_main, auc_upper_main)
# )


# # Save predictions to a CSV file
# new_data_filtered = new_data.loc[X_new.index].reset_index(drop=True)
# new_data_filtered['Predicted_Probability'] = y_pred_prob
# if y_new is not None:
#     new_data_filtered['Predicted_Label'] = y_pred
# new_data_filtered.to_csv('predictions.csv', index=False)