### Offline hyperparameter model bank (RF, XGB, SVM)

This folder contains scripts to pre-generate and persist all requested hyperparameter combinations for Random Forest, XGBoost, and SVM, along with CSV metadata and recommended-model shortcuts.

Directories created on first run:

```
models/
├── random_forest/
├── xgboost/
├── svm/
├── recommended_models/
│   ├── rf_recommended.pkl
│   ├── xgb_recommended.pkl
│   └── svm_recommended.pkl
└── metadata/
    ├── random_forest_models.csv
    ├── xgboost_models.csv
    └── svm_models.csv
```

Run generation (single preprocessed dataset with internal split; keeps CSVs and .pkl files by default):

```bash
python ml/generate_models.py \
  --data-csv kepler_preprocessed.csv \
  --target koi_disposition_encoded \
  --algorithms rf xgb svm \
  --max-models-per-algo 10000 \
  --jobs 4
```

Notes:
- No retraining at prediction time. All models are built ahead of time.
- The CSVs include: model_id, hyperparameters (JSON), accuracy, precision, recall, f1_macro, f1_weighted, confusion_matrix JSON, model_path, is_recommended, training_time_seconds, model_size_mb, feature_importances (if applicable).
- The best model per algorithm (highest f1_weighted) is copied to `models/recommended_models` during generation. To remove all `.pkl` files and keep only metadata CSVs, run with `--cleanup true`.

Loading a model by hyperparameters with fallback:

```python
from ml.model_loader import ModelRegistry

reg = ModelRegistry(root=".")
pkg = reg.find_and_load(
    algo="rf",
    params={
        "n_estimators": 200,
        "max_depth": 11,
        "min_samples_split": 3,
        "min_samples_leaf": 1,
        "criterion": "gini",
        "max_features": "sqrt",
        "bootstrap": True,
        "random_state": 42,
        "n_jobs": -1,
    },
)
model = pkg["model"]
```


