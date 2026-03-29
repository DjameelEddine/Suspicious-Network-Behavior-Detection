from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd

from .utils import now_utc_iso


@dataclass
class PreprocessingOutput:
    transformed: pd.DataFrame
    raw_valid_rows: pd.DataFrame
    invalid_events: List[Dict[str, object]]


class Preprocessor:
    """Applies inference-time validation and feature alignment."""

    def __init__(self, feature_list: Sequence[str], scaler: object | None = None) -> None:
        self.feature_list = [str(feature).strip() for feature in feature_list]
        self.scaler = scaler

    @classmethod
    def from_artifacts(cls, scaler_path: str, feature_list_path: str) -> "Preprocessor":
        scaler = joblib.load(scaler_path)
        feature_list = joblib.load(feature_list_path)

        if not isinstance(feature_list, (list, tuple)):
            raise ValueError("Feature list artifact is invalid. Expected list or tuple.")

        return cls(feature_list=feature_list, scaler=scaler)

    def transform_batch(self, raw_batch: pd.DataFrame) -> PreprocessingOutput:
        if raw_batch.empty:
            return PreprocessingOutput(
                transformed=pd.DataFrame(columns=self.feature_list),
                raw_valid_rows=pd.DataFrame(),
                invalid_events=[],
            )

        batch = raw_batch.copy()
        batch.columns = [str(column).strip() for column in batch.columns]

        missing_features = [feature for feature in self.feature_list if feature not in batch.columns]

        valid_vectors: List[Dict[str, float]] = []
        valid_rows: List[Dict[str, object]] = []
        valid_flow_ids: List[str] = []
        invalid_events: List[Dict[str, object]] = []

        for row_index, row in batch.iterrows():
            flow_id = str(row.get("flow_id", f"flow-{row_index}"))

            if missing_features:
                invalid_events.append(
                    {
                        "flow_id": flow_id,
                        "timestamp_utc": now_utc_iso(),
                        "status": "invalid",
                        "reason": "missing_required_columns",
                        "details": f"Missing columns: {', '.join(missing_features[:8])}",
                    }
                )
                continue

            vector: Dict[str, float] = {}
            row_errors: List[str] = []

            for feature in self.feature_list:
                value = row.get(feature)
                numeric_value = pd.to_numeric(value, errors="coerce")
                if pd.isna(numeric_value):
                    row_errors.append(f"non_numeric_or_missing:{feature}")
                    continue
                if np.isinf(float(numeric_value)):
                    row_errors.append(f"infinite_value:{feature}")
                    continue
                vector[feature] = float(numeric_value)

            if row_errors:
                invalid_events.append(
                    {
                        "flow_id": flow_id,
                        "timestamp_utc": now_utc_iso(),
                        "status": "invalid",
                        "reason": "invalid_feature_values",
                        "details": "; ".join(row_errors[:8]),
                    }
                )
                continue

            valid_vectors.append(vector)
            valid_rows.append(row.to_dict())
            valid_flow_ids.append(flow_id)

        if not valid_vectors:
            return PreprocessingOutput(
                transformed=pd.DataFrame(columns=self.feature_list),
                raw_valid_rows=pd.DataFrame(),
                invalid_events=invalid_events,
            )

        feature_df = pd.DataFrame(valid_vectors, index=valid_flow_ids, columns=self.feature_list)

        if self.scaler is not None:
            scaled_array = self.scaler.transform(feature_df.values)
            transformed = pd.DataFrame(scaled_array, index=feature_df.index, columns=self.feature_list)
        else:
            transformed = feature_df

        raw_valid_rows = pd.DataFrame(valid_rows, index=valid_flow_ids)

        return PreprocessingOutput(
            transformed=transformed,
            raw_valid_rows=raw_valid_rows,
            invalid_events=invalid_events,
        )
