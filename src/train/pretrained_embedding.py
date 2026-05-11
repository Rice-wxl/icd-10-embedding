from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
from keras import backend as K
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout, BatchNormalization, Embedding, Flatten, concatenate, MultiHeadAttention, LayerNormalization, Add
from keras.metrics import AUC, Precision, Recall
from sklearn.metrics import f1_score
# from imblearn.over_sampling import SMOTE
import pickle
from keras.saving import register_keras_serializable
import keras
import numpy as np

from config import OUTCOME_DATA_DIR, OUTCOME_SUBDIR, MODEL_DIR, LABEL_ENCODER_PATH


# Load train and test datasets — outcome is set in config.py (OUTCOME)
X_train_downsampled = pd.read_csv(OUTCOME_DATA_DIR / 'X_train_downsampled.csv')
y_train_downsampled = pd.read_csv(OUTCOME_DATA_DIR / 'y_train_downsampled.csv')
X_test = pd.read_csv(OUTCOME_DATA_DIR / 'X_test.csv')
y_test = pd.read_csv(OUTCOME_DATA_DIR / 'y_test.csv')


# Load the LabelEncoder for ICD codes
with open(LABEL_ENCODER_PATH, 'rb') as file:
    encoder = pickle.load(file)

# Get the actual number of unique ICD codes
num_unique_icd_codes = len(encoder.classes_)
print("Number of unique ICD codes:", num_unique_icd_codes)

# Custom Transformer Encoder Block
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
        # Return the parameters required for serialization
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
        # Recreate the layer from its config
        return cls(**config)

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
    
@tf.keras.utils.register_keras_serializable(package="Custom")
def f2_score(y_true, y_pred):
    # y_pred = tf.cast(y_pred > 0.5, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    epsilon = tf.keras.backend.epsilon()
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f2 = (5 * precision * recall) / (4 * precision + recall + epsilon)
    return f2


embeddings_50 = pd.read_csv('embeddings/icd-10-cm-2019-0050.csv')
embeddings_50 = embeddings_50.set_index('code')
embeddings_50 = embeddings_50.drop(columns=['desc'], errors='ignore')
codes = list(encoder.classes_) 
emb_df = pd.DataFrame(index=codes, columns=embeddings_50.columns, dtype=float)

common = emb_df.index.intersection(embeddings_50.index)
emb_df.loc[common] = embeddings_50.loc[common]
    
# Set the null row to be zero vector
emb_df.loc["NAN"] = 0.

# 4) For each “missing” code, find all more detailed codes and average
missing = emb_df.index[emb_df.isna().any(axis=1)]

for coarse in missing:
    # look for any base codes that start with the coarse code string:
    matches = embeddings_50.index[embeddings_50.index.str.startswith(coarse)]
    if len(matches):
        emb_df.loc[coarse] = embeddings_50.loc[matches].mean(axis=0)

# 4) Build a list of all base codes for fast lookup
missing = emb_df.index[emb_df.isna().any(axis=1)]
base_codes = list(embeddings_50.index)

for fine in missing:
    # find all base codes that are a prefix of the fine code
    prefixes = [b for b in base_codes if fine.startswith(b)]
    if prefixes:
        # pick the longest matching prefix
        best = max(prefixes, key=len)
        emb_df.loc[fine] = embeddings_50.loc[best]
        
missing = emb_df.index[emb_df.isna().any(axis=1)]
for each in missing:
    emb_df.loc[each] = 0.

embedding_matrix = emb_df.values.astype(np.float32)
print(embedding_matrix.shape)
assert embedding_matrix.shape == (num_unique_icd_codes, 50)



icd_columns = [f'I10_DX{i}' for i in range(1, 41)]

icd_inputs = Input(shape=(len(icd_columns),), name='icd_codes')
icd_embedding = Embedding(
    input_dim=num_unique_icd_codes,
    output_dim=50,
    weights=[embedding_matrix],
    trainable=True,           # or True if you still want to fine-tune
    name='icd_pretrained_embedding'
)
x = icd_embedding(icd_inputs)

# # Add Transformer layer to ICD embeddings
# num_transformer_blocks = 3  # Set the number of transformer blocks to stack
# for i in range(num_transformer_blocks):
#     transformer_block = TransformerBlock(embed_dim=32, num_heads=3, ff_dim=128, rate=0.3)
#     x = transformer_block(x, training=True)
    
deep_set_output_dim = int(416 * 0.5)
agg_block = DeepSet(input_dim = 50, hidden_dim = 416, output_dim = deep_set_output_dim, num_encode = 1, num_decode = 3)
x = agg_block(x) 
    
# # Flatten the transformer output for input into the MLP
# x = Flatten()(x)
# x = Dense(416, activation='relu', kernel_regularizer=l2(0.001), name='mlp_dense_1')(x)
# # x = Dropout(0.2, name='mlp_dropout_1')(x)
# x = Dense(deep_set_output_dim, activation='relu', kernel_regularizer=l2(0.001), name='mlp_dense_2')(x)
# # x = Dropout(0.2, name='mlp_dropout_2')(x)

# Define demographic and one-hot encoded inputs
age_input = Input(shape=(1,), name='age')
female_input = Input(shape=(1,), name='female')
pay1_inputs = [Input(shape=(1,), name=f'PAY1_{col}') for col in X_train_downsampled.filter(regex='PAY1_').columns]
zipinc_qrtl_inputs = [Input(shape=(1,), name=f'ZIPINC_QRTL_{col}') for col in X_train_downsampled.filter(regex='ZIPINC_QRTL_').columns]

# Concatenate demographic inputs
demographic_inputs = [age_input, female_input] + pay1_inputs + zipinc_qrtl_inputs
demographic_concat = concatenate(demographic_inputs, name='demographic_concat')

# Process demographics through a small MLP
demographic_hidden = Dense(64, activation='relu', name='demographic_dense1')(demographic_concat)
demographic_hidden = Dense(32, activation='relu', name='demographic_dense2')(demographic_hidden)

# Combine with DeepSet output
concatenated = concatenate([x, demographic_hidden], name='concatenate')
# concatenated = concatenate([x, age_input, female_input] + pay1_inputs + zipinc_qrtl_inputs, name='concatenate')
# concatenated = x

# Add BatchNormalization and dense layers
hidden = BatchNormalization(name='batch_norm')(concatenated)

for i in range(4):        
    units = max(32, int(480 * (0.5 ** i)))
    hidden = Dense(units, activation='relu', kernel_regularizer=l2(0.001), name=f'dense_{i}')(hidden)
    hidden = Dropout(0.1, name=f'dropout_{i}')(hidden)


# Output layer for mortality prediction
output = Dense(1, activation='sigmoid', name='output')(hidden)

# Build the model
model = Model(inputs=[icd_inputs, age_input, female_input] + pay1_inputs + zipinc_qrtl_inputs, outputs=output)
# model = Model(inputs=[icd_inputs], outputs=output)

# Compile the model with the weighted loss function
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=2e-5),
    loss=tf.keras.losses.binary_crossentropy,  # Use the custom weighted loss
    metrics=[AUC(name='auc'), Precision(name='precision'), Recall(name='recall'), F2Score()]
)

# Implement early stopping
early_stopping = EarlyStopping(
    monitor='val_auc', patience=2, mode='max', restore_best_weights=True
)

# Build input lists (order must match model.inputs)
train_inputs = [
    X_train_downsampled[icd_columns],       # embedding indices
    X_train_downsampled['AGE'],
    X_train_downsampled['FEMALE'],
    *[X_train_downsampled[c] for c in X_train_downsampled.filter(regex='PAY1_').columns],
    *[X_train_downsampled[c] for c in X_train_downsampled.filter(regex='ZIPINC_QRTL_').columns],
]

val_inputs = [
    X_test[icd_columns],       # embedding indices
    X_test['AGE'],
    X_test['FEMALE'],
    *[X_test[c] for c in X_test.filter(regex='PAY1_').columns],
    *[X_test[c] for c in X_test.filter(regex='ZIPINC_QRTL_').columns],
]

# train_inputs = [
#     X_train_downsampled[icd_columns],       # embedding indices
# ]

# val_inputs = [
#     X_test[icd_columns],       # embedding indices
# ]

# Train the model
history = model.fit(
    train_inputs,
    y_train_downsampled,
    epochs=10,
    batch_size=128,
    validation_data=(val_inputs, y_test),
    callbacks=[early_stopping],
    verbose=2
)


# Step 4: Evaluate the model
test_loss, test_auc, test_precision, test_recall, test_f2 = model.evaluate(
    [X_test[icd_columns], X_test['AGE'], X_test['FEMALE']] + [X_test[col] for col in X_test.filter(regex='PAY1_').columns] + [X_test[col] for col in X_test.filter(regex='ZIPINC_QRTL_').columns], 
    y_test, verbose=2)

# test_loss, test_auc, test_precision, test_recall, test_f2 = model.evaluate(
#     [X_test[icd_columns]], 
#     y_test, verbose=2) 

print(f'Test AUC: {test_auc:.4f}')
print(f'Test Precision: {test_precision:.4f}')
print(f'Test Recall: {test_recall:.4f}')
print(f'Test F2 Score: {test_f2:.4f}')

# # Save the trained model
model.save(MODEL_DIR / f'{OUTCOME_SUBDIR}_pretrained_50.keras')