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
import keras
from keras.saving import register_keras_serializable
import keras.backend as K
import keras_tuner as kt

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

class ICDHyperModel(kt.HyperModel):
    def build(self, hp):
        # Hyperparameters for ICD embedding and transformer/deep set
        embed_dim = hp.Int("embedding_size", 32, 64, step=32, default=32) 
        # embed_dim = 32
        # Number of transformer blocks to stack (if you wish to use transformer blocks)
        # num_transformer_blocks = hp.Int("num_transformer_blocks", 0, 6, default=3)
        num_transformer_blocks = 0
        # transformer_ff_dim = hp.Int("transformer_ff_dim", 64, 128, step=32, default=128)
        transformer_ff_dim = 128
        # transformer_dropout = hp.Float("transformer_dropout", 0.1, 0.5, step=0.1, default=0.3)
        transformer_dropout = 0.3
        # num_heads = hp.Int("transformer_heads", 1, 10, step=1, default=5)
        num_heads = 5

        # Hyperparameters for DeepSet aggregation (used here regardless of transformer use)
        deep_set_hidden_dim = hp.Int("deep_set_hidden_dim", 128, 512, step=32, default=256)
        # deep_set_hidden_dim = 224
        hidden_output_ratio = hp.Float("deep_set_ratio", 0.5, 1, step=0.1, default=0.5)
        deep_set_output_dim = int(deep_set_hidden_dim * hidden_output_ratio)
        # deep_set_output_dim = hp.Int("deep_set_output_dim", 128, 512, step=32, default=256)
        num_encode = hp.Int("num_deepset_encoding", 1, 3, step=1, default=2)
        num_decode = hp.Int("num_deepset_decoding", 1, 3, step=1, default=2)
        # num_encode = 2
        # num_decode = 1

        # Hyperparameters for post-aggregation dense layers
        # dense_units_1 = hp.Int("dense_units_1", 128, 256, step=32, default=128)
        # dense_units_2 = hp.Int("dense_units_2", 32, 128, step=32, default=64)
        # dense_units_3 = hp.Int("dense_units_3", 16, 64, step=16, default=32)
        # dense_units_4 = hp.Int("dense_units_3", 16, 64, step=16, default=32)
        # dropout_rate = hp.Float("dense_dropout_rate", 0.2, 0.5, step=0.1, default=0.3)


        demographic_units = hp.Int('demographic_initial_units', 32, 64, step=32, default=64)

        # num_MLP = hp.Int("num_MLP_layers", 2, 8, step=1, default=3)
        num_MLP = 4
        # Hyperparameters for initial number of units and decay factor
        initial_units = hp.Int('initial_units', 128, 512, step=32, default=256)
        # initial_units = 
        # decay_factor = hp.Float('decay_factor', 0.5, 1.0, step=0.1, default=0.8)
        decay_factor = 0.5

        # dense_units_1 = 128
        # dense_units_2 = 64
        # dense_units_3 = 32
        # dropout_rate = 0.4
        dropout_rate = hp.Float("dense_dropout_rate", 0.1, 0.5, step=0.1, default=0.3)
        
        # learning_rate = hp.Float("learning_rate", 1e-5, 1e-4, sampling="log", default=5e-5)
        learning_rate = 2e-5    

        # Define ICD-related inputs
        icd_columns = [f'I10_DX{i}' for i in range(1, 41)]
        icd_inputs = Input(shape=(len(icd_columns),), name='icd_codes')
        icd_embedding = Embedding(
            input_dim=num_unique_icd_codes,
            output_dim=embed_dim,
            name='icd_embedding',
            trainable=True
        )(icd_inputs)
        
        x = icd_embedding

        # # Optionally apply transformer blocks
        # for i in range(num_transformer_blocks):
        #     transformer_block = TransformerBlock(embed_dim=embed_dim,
        #                                          num_heads=num_heads,
        #                                          ff_dim=transformer_ff_dim,
        #                                          rate=transformer_dropout)
        #     x = transformer_block(x, training=True)
        
        # Aggregate ICD embeddings via DeepSet (or you could choose sum/mean pooling)
        agg_block = DeepSet(input_dim=embed_dim, hidden_dim=deep_set_hidden_dim, output_dim=deep_set_output_dim, num_encode = num_encode, num_decode = num_decode)
        x = agg_block(x)
        
        # Define demographic and one-hot encoded inputs
        age_input = Input(shape=(1,), name='age')
        female_input = Input(shape=(1,), name='female')
        pay1_inputs = [Input(shape=(1,), name=f'PAY1_{col}') for col in X_train_downsampled.filter(regex='PAY1_').columns]
        zipinc_qrtl_inputs = [Input(shape=(1,), name=f'ZIPINC_QRTL_{col}') for col in X_train_downsampled.filter(regex='ZIPINC_QRTL_').columns]
        
        # Concatenate demographic inputs
        demographic_inputs = [age_input, female_input] + pay1_inputs + zipinc_qrtl_inputs
        demographic_concat = concatenate(demographic_inputs, name='demographic_concat')

        # Process demographics through a small MLP
        demographic_hidden = Dense(demographic_units, activation='relu', name='demographic_dense1')(demographic_concat)
        demographic_hidden = Dense(demographic_units * 0.5, activation='relu', name='demographic_dense2')(demographic_hidden)

        # Combine with DeepSet output
        concatenated = concatenate([x, demographic_hidden], name='concatenate')

        # Concatenate all inputs
        # concatenated = concatenate([x, age_input, female_input] + pay1_inputs + zipinc_qrtl_inputs, name='concatenate')
        
        hidden = BatchNormalization(name='batch_norm')(concatenated)
        for i in range(num_MLP):        
            units = max(32, int(initial_units * (decay_factor ** i)))
            hidden = Dense(units, activation='relu', kernel_regularizer=l2(0.001), name=f'dense_{i}')(hidden)
            hidden = Dropout(dropout_rate, name=f'dropout_{i}')(hidden)

        # Output layer for mortality prediction
        output = Dense(1, activation='sigmoid', name='output')(hidden)
        
        # Build and compile the model
        model = Model(inputs=[icd_inputs, age_input, female_input] + pay1_inputs + zipinc_qrtl_inputs, outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.binary_crossentropy,
            metrics=[AUC(name='auc'), Precision(name='precision'), Recall(name='recall'), F2Score()]
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        # Tune batch size as well
        # batch_size = hp.Choice("batch_size", [32, 64, 128], default=64)
        batch_size = 128
        kwargs["batch_size"] = batch_size
        return model.fit(*args, **kwargs)
    
early_stopping = EarlyStopping(
    monitor='val_auc', patience=2, mode='max', restore_best_weights=True
)

# Create an instance of the HyperModel
hypermodel = ICDHyperModel()

# Initialize the tuner (here using RandomSearch; you can also try Hyperband or BayesianOptimization)
tuner = kt.RandomSearch(
    hypermodel,
    objective=kt.Objective("val_auc", direction="max"),
    max_trials=32,
    executions_per_trial=1,
    directory="logs",
    project_name=f"ICD_hyperparameter_tuning_{OUTCOME_SUBDIR}_auc"
)

icd_columns = [f'I10_DX{i}' for i in range(1, 41)]

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

# Run the hyperparameter search
tuner.search(
    train_inputs,
    y_train_downsampled,
    epochs=10,
    validation_data=(val_inputs, y_test),
    callbacks=[early_stopping],
    verbose=2
)

# Retrieve the best model.
best_model = tuner.get_best_models(num_models=1)[0]
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("The optimal hyperparameters are:")
for hp_name, hp_value in best_hps.values.items():
    print(f"{hp_name}: {hp_value}")


# Step 4: Evaluate the model
test_loss, test_auc, test_precision, test_recall, test_f2 = best_model.evaluate(
    [X_test[icd_columns], X_test['AGE'], X_test['FEMALE']] + [X_test[col] for col in X_test.filter(regex='PAY1_').columns] + [X_test[col] for col in X_test.filter(regex='ZIPINC_QRTL_').columns], 
    y_test, 
    verbose=2)

print(f'Test AUC: {test_auc:.4f}')
print(f'Test Precision: {test_precision:.4f}')
print(f'Test Recall: {test_recall:.4f}')
print(f'Test F2 Score: {test_f2:.4f}')

# Save the trained model
best_model.save(MODEL_DIR / f'{OUTCOME_SUBDIR}_hypertrial_auc.keras')