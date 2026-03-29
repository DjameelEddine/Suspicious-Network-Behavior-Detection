from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import joblib


def export_notebook_artifacts(
    binary_model: Any,
    multiclass_model: Any,
    scaler: Any,
    selected_features: Sequence[str],
    output_dir: str | Path = "models",
    label_encoder: Optional[Any] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Export notebook-trained artifacts to the dashboard runtime contract.

    Expected use from SNBD.ipynb:
        from scripts.export_notebook_artifacts import export_notebook_artifacts
        export_notebook_artifacts(
            binary_model=best_xgb_binary_model,
            multiclass_model=best_knn_multi_model,
            scaler=scaler,
            selected_features=selected_features,
            label_encoder=le,
        )
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(binary_model, out_dir / "binary_model.pkl")
    joblib.dump(multiclass_model, out_dir / "multiclass_model.pkl")
    joblib.dump(scaler, out_dir / "scaler.pkl")
    joblib.dump(list(selected_features), out_dir / "feature_list.pkl")

    if label_encoder is not None:
        joblib.dump(label_encoder, out_dir / "label_encoder.pkl")

    payload = {
        "binary_model": "binary_model.pkl",
        "multiclass_model": "multiclass_model.pkl",
        "scaler": "scaler.pkl",
        "feature_list": "feature_list.pkl",
        "label_encoder": "label_encoder.pkl" if label_encoder is not None else None,
    }

    if metadata:
        payload.update(metadata)

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)

    return out_dir


if __name__ == "__main__":
    raise SystemExit(
        "This script is intended to be imported from SNBD.ipynb after model training."
    )
