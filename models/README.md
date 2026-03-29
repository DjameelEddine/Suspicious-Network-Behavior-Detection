# Model Artifact Contract

The dashboard expects the following files in this directory:

- `binary_model.pkl`
- `multiclass_model.pkl`
- `scaler.pkl`
- `feature_list.pkl`
- `metadata.json` (recommended)
- `label_encoder.pkl` (optional)

## Recommended source (notebook parity)

After training in `SNBD.ipynb`, export runtime artifacts with:

```python
from scripts.export_notebook_artifacts import export_notebook_artifacts

export_notebook_artifacts(
    binary_model=best_xgb_binary_model,
    multiclass_model=best_knn_multi_model,
    scaler=scaler,
    selected_features=selected_features,
    label_encoder=le,
)
```

## Quick local fallback

If notebook objects are not available, generate runnable artifacts with:

```bash
python scripts/build_artifacts.py --dataset clean_traffic.csv
```
