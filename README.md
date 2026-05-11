# ICD-10 Outcome Prediction

Deep learning models for predicting in-hospital mortality (`DIED`, `MOR30`)
and 30-day readmission (`REA30`) from ICD-10 diagnosis codes in the
HCUP National Readmission Database (NRD), 2016–2022.

The model treats the up-to-40 ICD codes per admission as an unordered set
and aggregates them with a permutation-invariant **DeepSet** encoder
(optionally followed by a **Transformer block**), then concatenates patient
demographics (age, sex, payer, ZIP-code income quartile) before a small MLP
head predicts the binary outcome. Performance is benchmarked against
clinical comorbidity indices (Elixhauser, Charlson, age-adjusted Charlson)
on held-out NRD 2021–2022 data.

## Repository layout

```
icd/
├── README.md                       this file
├── LICENSE                         MIT
├── .gitignore
├── environment.yml                 conda environment (icd_gpu)
├── requirements.txt                pip-installable subset
├── CLAUDE.md                       guidance for Claude Code
│
├── src/                            all Python source
│   ├── config.example.py           copy to config.py for your env
│   ├── config.py                   (gitignored) your real local paths
│   ├── preprocessing.py            NRD preprocessing pipeline
│   ├── train/                      model training
│   │   ├── transformer.py
│   │   ├── hyper_tune.py
│   │   └── pretrained_embedding.py
│   ├── evaluate/                   evaluation + baselines + statistics
│   │   ├── evaluate.py
│   │   ├── fit_LR_baseline.py
│   │   ├── delong_test.py
│   │   └── compute_p_value.py
│   ├── calibration/                calibration curves vs. baselines
│   │   └── calibration_curve.py
│   ├── feature_importance/         Integrated Gradients interpretation
│   │   ├── IG.py
│   │   └── visualize_icd_importance.py
│   └── inference/                  ad-hoc inference utilities
│       ├── predict_single_patient.py
│       ├── predict_on_small_dataset.py
│       └── create_small_test_dataset.py
│
├── scripts/                        SLURM batch scripts (one per pipeline stage)
├── notebooks/                      Jupyter notebooks (outputs stripped)
└── results/                        small artifacts safe to commit
    ├── figures/                    PNG plots
    ├── feature_importance/         top-N ICD CSV summaries
    ├── delong/                     DeLong AUC-comparison results
    └── small_dataset/              tiny stratified test samples
```

The following directories are referenced by the scripts but **not**
checked into git (listed in `.gitignore`):

| Path               | Contents                                          |
|--------------------|---------------------------------------------------|
| `data/`            | Raw + preprocessed NRD CSVs (~20 GB)              |
| `Model/`           | Trained `.keras` models, label encoder, scaler    |
| `Baselines/`       | Logistic-regression baselines for comorbidity indices |
| `embeddings/`      | Pretrained ICD-10 embedding tables                |
| `predictions.csv`  | Generated patient-level predictions               |
| `logs/`            | SLURM `.out` job logs                             |

## Data access

This project uses the HCUP National Readmission Database (NRD). The data
are restricted and **not redistributable** — request access from the
Healthcare Cost and Utilization Project at
[https://hcup-us.ahrq.gov/](https://hcup-us.ahrq.gov/).

Once obtained, place the pooled CSV at the path you set in
`src/config.py` (default: the value of `NRD_RAW_CSV`).

## Environment setup

Developed on Brown University's Oscar HPC cluster.

```bash
conda env create -f environment.yml
conda activate icd_gpu
```

`requirements.txt` lists the same packages for `pip`-based installs.

## Configuration

All on-disk paths live in `src/config.py`. To set up a fresh checkout:

```bash
cp src/config.example.py src/config.py
# edit src/config.py — point NRD_RAW_CSV, DATA_DIR, NRD_2021_TEST,
# NRD_2022_TEST at your local copies of the NRD data.
```

Every Python script does `from config import ...`, so changing a path
once in `src/config.py` is enough. `src/config.py` is gitignored so
personal paths never enter version control.

## Pipeline

All long-running jobs are launched via SLURM batch scripts in `scripts/`.
Each script `cd`s to the repo root, prepends `src/` to `PYTHONPATH`, then
invokes the corresponding Python module. Submit them from the repo root:

```bash
# 1. Preprocess the pooled NRD CSV → train/test splits per outcome
sbatch scripts/preprocessing.sh

# 2. Train the model (edit src/train/transformer.py to switch outcome)
sbatch scripts/run.sh                # train transformer
sbatch scripts/hyper_tune.sh         # hyperparameter search

# 3. Evaluate on held-out 2021-2022 data (vs. ECI / CCI baselines)
sbatch scripts/evaluate.sh

# 4. Fit logistic-regression baselines for comorbidity indices
sbatch scripts/fit_LR.sh

# 5. Calibration curves
sbatch scripts/calibration.sh

# 6. Statistical AUC comparison
sbatch scripts/delong_test.sh

# 7. Feature importance via Integrated Gradients
sbatch scripts/interpretation.sh
```

Each script writes its `.out` log into `logs/<category>/`, which is
gitignored.

## Notebooks

| Notebook                              | Purpose                                       |
|---------------------------------------|-----------------------------------------------|
| `notebooks/data_preprocessing.ipynb`  | Exploratory NRD preprocessing                 |
| `notebooks/Transformers.ipynb`        | Prototype model architectures                 |
| `notebooks/Hyperparameter.ipynb`      | Manual hyperparameter sweep notes             |
| `notebooks/Validate.ipynb`            | Held-out evaluation walkthrough               |
| `notebooks/Embeddings_trial.ipynb`    | Pretrained ICD-10 embedding experiments       |
| `notebooks/Dataset_size_examine.ipynb`| Sample-size scaling analysis                  |

Outputs are stripped from committed notebooks. Re-run cells locally to
regenerate plots.

## Custom Keras components

Three custom serializable Keras objects are used across the pipeline:

- **`DeepSet`** — permutation-invariant set aggregation (phi/rho networks)
- **`TransformerBlock`** — multi-head self-attention encoder block
- **`F2Score`** — F2 metric (weights recall higher than precision)

All are registered with `@register_keras_serializable(package="Custom")`.
They are currently duplicated inline at the top of each script that loads
a trained model. See **TODO** below.

## Outcomes

| Outcome    | Definition                                              |
|------------|---------------------------------------------------------|
| `DIED`     | In-hospital mortality                                   |
| `MOR30`    | 30-day mortality (in or out of hospital)                |
| `REA30`    | 30-day readmission (excludes patients who died)         |

To switch outcome, edit `OUTCOME` in `src/config.py` (used by
`preprocessing.py`) and the `OUTCOME_VAR` constant inside the relevant
training/evaluation script.

## Results

Committed result artifacts live under `results/`:

- `results/figures/calibration/` — calibration curves vs. baselines
- `results/figures/roc/` — ROC curves on held-out 2021-2022 data
- `results/figures/feature_importance/` — top ICD code lollipop charts
- `results/feature_importance/` — top-N ICD CSV tables
- `results/delong/` — DeLong test p-values
- `results/small_dataset/` — tiny stratified test samples

## Citation

> _Add citation block when paper is published._

## License

This code is released under the MIT License — see [LICENSE](LICENSE).
The NRD data itself is licensed separately by HCUP/AHRQ and is not
redistributed here.

## Known TODOs

- **Deduplicate custom Keras layers.** `DeepSet`, `TransformerBlock`, and
  `F2Score` are copy-pasted into ~9 scripts. They should move to a
  shared `src/custom_layers.py`. Deferred because the
  `register_keras_serializable(package="Custom")` registration string is
  embedded in saved `.keras` files; consolidating without first re-saving
  the existing models could break `load_model()`.
