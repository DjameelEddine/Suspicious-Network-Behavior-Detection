from __future__ import annotations

from time import perf_counter
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from .config import APP_CONFIG, AppConfig, resolve_first_existing
from .utils import compute_risk_level, compute_risk_score, now_utc_iso


class PredictionService:
    """Runs binary then conditional multi-class inference for each flow."""

    def __init__(
        self,
        binary_model: object,
        multiclass_model: object,
        config: AppConfig = APP_CONFIG,
        label_encoder: Optional[object] = None,
    ) -> None:
        self.binary_model = binary_model
        self.multiclass_model = multiclass_model
        self.config = config
        self.label_encoder = label_encoder

    @classmethod
    def from_artifacts(cls, config: AppConfig = APP_CONFIG) -> "PredictionService":
        binary_model_path = resolve_first_existing(config.artifact_paths.binary_candidates())
        multiclass_model_path = resolve_first_existing(config.artifact_paths.multiclass_candidates())

        if binary_model_path is None:
            raise FileNotFoundError("Binary model artifact could not be resolved.")
        if multiclass_model_path is None:
            raise FileNotFoundError("Multi-class model artifact could not be resolved.")

        binary_model = joblib.load(binary_model_path)
        multiclass_model = joblib.load(multiclass_model_path)

        label_encoder = None
        if config.artifact_paths.label_encoder.exists():
            label_encoder = joblib.load(config.artifact_paths.label_encoder)

        return cls(
            binary_model=binary_model,
            multiclass_model=multiclass_model,
            config=config,
            label_encoder=label_encoder,
        )

    def _prediction_confidence(self, model: object, vector: np.ndarray, predicted_value: object) -> float:
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(vector)
            if probabilities is not None and len(probabilities) > 0:
                classes = getattr(model, "classes_", None)
                if classes is not None:
                    try:
                        class_index = list(classes).index(predicted_value)
                        return float(probabilities[0][class_index])
                    except ValueError:
                        pass
                return float(np.max(probabilities[0]))

        if hasattr(model, "decision_function"):
            decision = model.decision_function(vector)
            decision_value = float(np.ravel(decision)[0])
            return float(1.0 / (1.0 + np.exp(-decision_value)))

        return 0.5

    def _is_attack(self, binary_prediction: object) -> bool:
        if isinstance(binary_prediction, (int, np.integer, float, np.floating)):
            return int(binary_prediction) == 1

        normalized = str(binary_prediction).strip().upper()
        if normalized in {"0", "BENIGN", "NORMAL"}:
            return False
        if normalized in {"1", "ATTACK", "MALICIOUS", "ANOMALY"}:
            return True

        return normalized != "BENIGN"

    def _decode_multiclass_label(self, raw_prediction: object) -> str:
        if self.label_encoder is not None and isinstance(raw_prediction, (int, np.integer)):
            try:
                return str(self.label_encoder.inverse_transform([int(raw_prediction)])[0])
            except Exception:
                pass
        return str(raw_prediction)

    def predict_batch(self, transformed: pd.DataFrame, raw_valid_rows: pd.DataFrame) -> List[Dict[str, object]]:
        if transformed.empty:
            return []

        events: List[Dict[str, object]] = []

        for flow_id, row in transformed.iterrows():
            row_start = perf_counter()
            vector = np.array(row.values, dtype=float).reshape(1, -1)

            binary_raw = self.binary_model.predict(vector)[0]
            is_attack = self._is_attack(binary_raw)
            binary_confidence = self._prediction_confidence(self.binary_model, vector, binary_raw)
            binary_label = "Attack" if is_attack else "Benign"

            if is_attack:
                multi_raw = self.multiclass_model.predict(vector)[0]
                attack_label = self._decode_multiclass_label(multi_raw)
                attack_confidence = self._prediction_confidence(self.multiclass_model, vector, multi_raw)
                confidence = attack_confidence
            else:
                attack_label = "N/A"
                attack_confidence = 0.0
                confidence = binary_confidence

            risk_level = compute_risk_level(
                is_attack=is_attack,
                confidence=float(confidence),
                attack_type=attack_label,
                severity_map=self.config.risk.attack_severity,
                low_threshold=self.config.risk.low_confidence_threshold,
                high_threshold=self.config.risk.high_confidence_threshold,
            )
            risk_score = compute_risk_score(
                is_attack=is_attack,
                confidence=float(confidence),
                attack_type=attack_label,
                severity_map=self.config.risk.attack_severity,
            )

            latency_ms = (perf_counter() - row_start) * 1000.0

            raw_row = raw_valid_rows.loc[flow_id].to_dict() if flow_id in raw_valid_rows.index else {}

            event = {
                "flow_id": str(flow_id),
                "timestamp_utc": now_utc_iso(),
                "status": "processed",
                "binary_prediction": binary_label,
                "attack_type": attack_label,
                "confidence": round(float(confidence), 4),
                "binary_confidence": round(float(binary_confidence), 4),
                "attack_confidence": round(float(attack_confidence), 4),
                "risk_level": risk_level,
                "risk_score": risk_score,
                "latency_ms": round(float(latency_ms), 2),
                "actual_label": raw_row.get("Label", None),
                "feature_vector": {column: float(value) for column, value in row.items()},
                "shap_summary": "",
                "top_positive_features": [],
                "top_negative_features": [],
            }
            events.append(event)

        return events
