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
import tensorflow as tf
from keras.layers import Input, Dense, Dropout, BatchNormalization, Embedding, Flatten, concatenate, MultiHeadAttention, LayerNormalization, Add
from keras.models import Model
from sklearn.utils import shuffle
from tqdm import tqdm

from config import (
    OUTCOME_SUBDIR, MODEL_PATH,
    LABEL_ENCODER_PATH, AGE_SCALER_PATH,
    NRD_2021_TEST, NRD_2022_TEST, FEATURE_IMPORTANCE_DIR,
)

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
model = load_model(MODEL_PATH)

# Load the LabelEncoder for ICD codes
with open(LABEL_ENCODER_PATH, 'rb') as file:
    encoder = pickle.load(file)

# Load the MinMaxScaler for 'AGE'
with open(AGE_SCALER_PATH, 'rb') as file:
    age_scaler = pickle.load(file)

# new_data_file_path = '/users/xwang259/icd/data/NRD_2019_Small.dta'
# # Load new data
# new_data = pd.read_stata(new_data_file_path)

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
if outcome_var == 'REA30' and 'DIED' in new_data.columns:
# if 'DIED' in new_data.columns:
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
zipinc_qrtl_columns = [col for col in new_data.columns if col.startswith('ZIPINC_QRTL_')]

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
idx01 = stratified_pick(idx10,  y[idx10], frac=0.10, seed=123)  # 1% ⊂ 10%

# df10 = new_data.iloc[idx10].copy()
df01 = new_data.iloc[idx01].copy()


# Prepare input features. 
X_new = df01[['AGE', 'FEMALE'] + pay1_columns + zipinc_qrtl_columns + icd_columns]
y_true = df01[outcome_var].to_numpy(np.int32)

print(df01.shape)



# ---------- config ----------
STEPS = 32              # IG Riemann steps (try 32–64)
BATCH_SIZE = 1024
SAMPLE_SIZE = None       # e.g., 50_000 for speed; None = all rows
PAD_LABEL = "NAN"        # your PAD/unknown label
TARGET = "logit"         # "logit" or "loss"
ICD_EMBED_VAR_NAME = None  # set to exact var name if auto-detect fails

# ---------- utilities reused from before ----------
def assemble_inputs(df, icd_array, pay1_cols, zip_cols):
    return [
        icd_array.astype(np.float32),
        df['AGE'].to_numpy(np.float32),
        df['FEMALE'].to_numpy(np.float32),
    ] + [df[c].to_numpy(np.float32) for c in pay1_cols] \
      + [df[c].to_numpy(np.float32) for c in zip_cols]

def stratified_sample_indices(y, sample_size, rng):
    if sample_size is None or sample_size >= len(y):
        return np.arange(len(y))
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    pos_target = max(1, int(round(sample_size * len(pos) / len(y))))
    neg_target = sample_size - pos_target
    pos_idx = rng.choice(pos, size=min(pos_target, len(pos)), replace=False)
    neg_idx = rng.choice(neg, size=min(neg_target, len(neg)), replace=False)
    idx = np.concatenate([pos_idx, neg_idx])
    rng.shuffle(idx)
    return np.sort(idx)

def find_icd_embedding_var(model, vocab_size_hint=None, name_hint=None):
    candidates = []
    for v in model.trainable_variables:
        if v.shape.rank == 2 and "emb" in v.name.lower():
            candidates.append((v.name, tuple(v.shape.as_list())))
    if candidates:
        print("Embedding-like variables:")
        for n, s in candidates:
            print(f"  - {n} shape={s}")
    if name_hint is not None:
        for v in model.trainable_variables:
            if v.name == name_hint:
                return v
    if vocab_size_hint is not None:
        best = None
        for v in model.trainable_variables:
            if v.shape.rank == 2 and v.shape[0] >= vocab_size_hint:
                if ("emb" in v.name.lower()) or best is None:
                    best = v
        if best is not None:
            return best
    two_d_embs = [v for v in model.trainable_variables if v.shape.rank == 2 and "emb" in v.name.lower()]
    if two_d_embs:
        return max(two_d_embs, key=lambda t: int(t.shape[0]) * int(t.shape[1]))
    raise ValueError("Could not identify ICD embedding variable. Set ICD_EMBED_VAR_NAME explicitly.")

# ---------- main IG function ----------
def integrated_gradients_icd_global(
    model, df, y_true, icd_cols, pay1_cols, zip_cols, encoder,
    steps=STEPS, batch_size=BATCH_SIZE, sample_size=SAMPLE_SIZE,
    pad_label=PAD_LABEL, target=TARGET, baseline="pad",  # "pad" | "zero" | "mean"
    icd_embed_var_name=ICD_EMBED_VAR_NAME, random_state=0,
):
    rng = np.random.default_rng(random_state)
    # subset (optional)
    X_icd_full = df[icd_cols].to_numpy(np.int32)
    idx_eval = stratified_sample_indices(y_true, sample_size, rng)
    df_eval = df.iloc[idx_eval]
    y_eval = y_true[idx_eval].astype(np.float32)
    X_icd_eval = X_icd_full[idx_eval]

    # counts per code for per-occurrence normalization
    vocab_size_hint = len(encoder.classes_)
    counts = np.bincount(X_icd_eval.reshape(-1), minlength=vocab_size_hint)

    # embedding var & PAD
    emb_var = find_icd_embedding_var(model, vocab_size_hint=vocab_size_hint, name_hint=icd_embed_var_name)
    E_orig = emb_var.read_value()                # (V, D)
    PAD_ID = int(encoder.transform([pad_label])[0])

    # build baseline matrix E0
    if baseline == "pad":
        e_pad = tf.gather(E_orig, [PAD_ID])[0]   # (D,)
        E0 = tf.repeat(e_pad[None, :], repeats=int(E_orig.shape[0]), axis=0)
    elif baseline == "zero":
        E0 = tf.zeros_like(E_orig)
    elif baseline == "mean":
        E0 = tf.repeat(tf.reduce_mean(E_orig, axis=0, keepdims=True), repeats=int(E_orig.shape[0]), axis=0)
    else:
        raise ValueError("baseline must be one of {'pad','zero','mean'}")

    deltaE = E_orig - E0                         # (V, D)

    # data pipeline
    inputs_eval = assemble_inputs(df_eval, X_icd_eval, pay1_cols, zip_cols)
    ds = tf.data.Dataset.from_tensor_slices((tuple(tf.convert_to_tensor(a) for a in inputs_eval),
                                             tf.convert_to_tensor(y_eval))).batch(batch_size)

    # accumulators
    grads_accum = tf.zeros_like(E_orig)          # sum over steps & batches
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

    # IG loop over steps
    try:
        for s in tqdm(range(1, steps + 1)):
            alpha = tf.cast(s / steps, tf.float32)
            E_interp = E0 + alpha * deltaE       # (V, D)
            emb_var.assign(E_interp)             # temporarily set embeddings

            for (batch_inputs, yb) in ds:
                with tf.GradientTape() as tape:
                    tape.watch(emb_var)
                    out = model(batch_inputs, training=False)
                    out = tf.reshape(out, (-1,))   # (B,)
                    if target == "logit":
                        eps = tf.constant(1e-6, out.dtype)
                        logit = tf.math.log(tf.clip_by_value(out, eps, 1.0 - eps)) \
                                - tf.math.log1p(-tf.clip_by_value(out, eps, 1.0 - eps))
                        scalar = tf.reduce_mean(logit)
                    elif target == "loss":
                        loss = bce(yb, out)
                        scalar = tf.reduce_mean(loss)
                    else:
                        raise ValueError("target must be 'logit' or 'loss'")
                g = tape.gradient(scalar, emb_var)   # (V, D); non-zero only for codes present in batch
                grads_accum += g

        grads_avg = grads_accum / steps              # approximate integral average
        # per-code IG: sum over D of (deltaE * grads_avg)
        per_code_signed = tf.reduce_sum(deltaE * grads_avg, axis=1)  # (V,)
        per_code_mag = tf.math.abs(per_code_signed)
        # per_code_mag = per_code_signed

        # build results
        codes = np.arange(per_code_mag.shape[0])
        mask_valid = (codes != PAD_ID) & (counts > 0)
        codes = codes[mask_valid]
        occ = counts[mask_valid]
        ig_abs_total = per_code_mag.numpy()[mask_valid]
        ig_abs_mean = ig_abs_total / np.maximum(occ, 1)

        ig_signed_total = per_code_signed.numpy()[mask_valid]
        ig_signed_mean = ig_signed_total / np.maximum(occ, 1)

        icd_str = encoder.inverse_transform(codes)
        res = pd.DataFrame({
            "icd_code": icd_str,
            "code_id": codes,
            "support_eval": occ,
            "ig_absolute_total": ig_abs_total,
            "ig_absolute_mean": ig_abs_mean,
            "ig_signed_total": ig_signed_total,
            "ig_signed_mean": ig_signed_mean,
            "target": target,
            "baseline": baseline,
            "steps": steps,
        })
        return res
    finally:
        # restore original embeddings no matter what
        emb_var.assign(E_orig)


res_ig = integrated_gradients_icd_global(
    model=model,
    df=new_data,
    y_true=y_true.astype(np.float32),
    icd_cols=icd_columns,
    pay1_cols=pay1_columns,
    zip_cols=zipinc_qrtl_columns,
    encoder=encoder,
    steps=STEPS,                # try 64 if you want tighter completeness
    batch_size=BATCH_SIZE,
    sample_size=None,        # e.g., 100_000 for speed
    pad_label="NAN",
    target="logit",          # or "loss"
    baseline="pad",          # "pad" | "zero" | "mean"
)

min_support = 50
top_num = 20

# Two views: "global impact" vs "per-occurrence effect"
top20_global = res_ig.sort_values("ig_absolute_total", ascending=False).head(top_num)
top20_per_occ = res_ig[res_ig["support_eval"] >= min_support].sort_values("ig_absolute_mean", ascending=False).head(top_num)

print(f"\nTop {top_num} ICDs by GLOBAL impact (absolute value):")
print(top20_global[["icd_code","support_eval","ig_absolute_total","ig_absolute_mean"]].to_string(index=False))

print(f"\nTop {top_num} ICDs by PER-OCCURRENCE effect (absolute value, min support={min_support}):")
print(top20_per_occ[["icd_code","support_eval","ig_absolute_mean","ig_absolute_total"]].to_string(index=False))

top20_global.to_csv(FEATURE_IMPORTANCE_DIR / f"{OUTCOME_SUBDIR}_top20_global.csv", index=False)
top20_per_occ.to_csv(FEATURE_IMPORTANCE_DIR / f"{OUTCOME_SUBDIR}_top20_per_occurrence.csv", index=False)



# Two views: "global impact" vs "per-occurrence effect"
top10_positive = res_ig[res_ig["support_eval"] >= min_support].sort_values("ig_signed_mean", ascending=False).head(10)
top10_negative = res_ig[res_ig["support_eval"] >= min_support].sort_values("ig_signed_mean", ascending=True).head(10)

print(f"\nTop 10 ICDs by PER-OCCURRENCE with the most positive effect:")
print(top10_positive[["icd_code","support_eval","ig_signed_total","ig_signed_mean"]].to_string(index=False))

print(f"\nTop 10 ICDs by PER-OCCURRENCE with the most negative effect:")
print(top10_negative[["icd_code","support_eval","ig_signed_total","ig_signed_mean"]].to_string(index=False))

top10_positive.to_csv(FEATURE_IMPORTANCE_DIR / f"{OUTCOME_SUBDIR}_top10_positive.csv", index=False)
top10_negative.to_csv(FEATURE_IMPORTANCE_DIR / f"{OUTCOME_SUBDIR}_top10_negative.csv", index=False)