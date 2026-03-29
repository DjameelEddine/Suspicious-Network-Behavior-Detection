# Real-Time Explainable Intrusion Detection System

Single-page Streamlit dashboard for simulated live network-flow intrusion detection with:

- Binary prediction: Benign vs Attack
- Conditional multi-class prediction: Attack type
- Per-flow explainability summary (SHAP-backed when available)
- Live metrics, risk scoring, and exportable logs

## Project Layout

```text
.
|-- app.py
|-- clean_traffic.csv
|-- traffic.csv
|-- models/
|-- scripts/
|-- src/
`-- tests/
```

## 1) Install Dependencies

```bash
pip install -r requirements.txt
```

## 2) Build Runtime Artifacts

The dashboard requires these files in `models/`:

- `binary_model.pkl`
- `multiclass_model.pkl`
- `scaler.pkl`
- `feature_list.pkl`

### Option A (recommended): Export notebook-trained artifacts

After training in `SNBD.ipynb`, run:

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

### Option B: Fast local fallback artifact build

```bash
python scripts/build_artifacts.py --dataset clean_traffic.csv
```

## 3) Run Dashboard

```bash
streamlit run app.py
```

## 4) Features Implemented

- Real-time row/batch simulation from CSV with start/pause/resume/stop
- Configurable rows-per-second and batch size
- Strict feature alignment and invalid-flow rejection
- Two-stage inference (binary then multi-class)
- Per-flow confidence, risk level, and risk score
- Explanation summary with top positive/negative contributors
- Live prediction table with high-risk highlighting
- Traffic summary and confidence distribution
- Detailed inspection panel for selected flow
- Metrics panel (latency, throughput, errors, health)
- Export prediction history to CSV/JSON in `exports/`

## 5) Tests

```bash
pytest -q
```

## Notes

- If SHAP is unavailable or fails to initialize for a model, the dashboard falls back to model-based contribution proxies.
- Large CSV files are supported by simulation, but high rows-per-second and large batch sizes can reduce UI responsiveness.
