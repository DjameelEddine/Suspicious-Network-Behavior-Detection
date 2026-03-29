from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .utils import summarize_explanation


class ExplainabilityService:
    """Provides local and global explanations with SHAP when available."""

    def __init__(self, model: object, feature_names: Sequence[str]) -> None:
        self.model = model
        self.feature_names = list(feature_names)
        self._explainer = None
        self._shap = None
        self.available = False
        self.status_message = "SHAP not initialized"

        try:
            import shap  # type: ignore

            self._shap = shap
            self.available = True
            self.status_message = "SHAP ready"
        except Exception:
            self.available = False
            self.status_message = "SHAP package is not available. Using fallback attributions."

    def _build_explainer(self, background: pd.DataFrame | None = None) -> None:
        if not self.available or self._explainer is not None:
            return

        try:
            if hasattr(self.model, "feature_importances_"):
                self._explainer = self._shap.TreeExplainer(self.model)
            else:
                if background is None:
                    background = pd.DataFrame(
                        np.zeros((1, len(self.feature_names))),
                        columns=self.feature_names,
                    )
                self._explainer = self._shap.Explainer(self.model, background)
            self.status_message = "SHAP explainer loaded"
        except Exception:
            self.available = False
            self._explainer = None
            self.status_message = "SHAP explainer could not be built. Using fallback attributions."

    def global_importance(self, top_n: int = 20) -> pd.DataFrame:
        if hasattr(self.model, "feature_importances_"):
            values = np.abs(np.array(self.model.feature_importances_, dtype=float))
        elif hasattr(self.model, "coef_"):
            coef = np.array(self.model.coef_, dtype=float)
            values = np.abs(coef).mean(axis=0)
        else:
            values = np.ones(len(self.feature_names), dtype=float)

        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance": values,
            }
        ).sort_values("importance", ascending=False)
        return importance_df.head(top_n)

    def _fallback_contributions(self, row: pd.Series) -> np.ndarray:
        vector = row.reindex(self.feature_names).astype(float).values

        if hasattr(self.model, "feature_importances_"):
            importance = np.array(self.model.feature_importances_, dtype=float)
        elif hasattr(self.model, "coef_"):
            coef = np.array(self.model.coef_, dtype=float)
            importance = np.abs(coef).mean(axis=0)
        else:
            importance = np.ones(len(self.feature_names), dtype=float)

        if np.allclose(importance.sum(), 0.0):
            importance = np.ones(len(self.feature_names), dtype=float)

        importance = importance / np.sum(np.abs(importance))
        centered = vector - np.mean(vector)
        return centered * importance

    def explain_row(
        self,
        row: pd.Series,
        predicted_label: str,
        top_k: int = 5,
    ) -> Dict[str, object]:
        row_series = row.reindex(self.feature_names).astype(float)
        contribution_values: np.ndarray | None = None

        if self.available:
            self._build_explainer()
            if self._explainer is not None:
                try:
                    row_df = pd.DataFrame([row_series.values], columns=self.feature_names)
                    shap_output = self._explainer(row_df)
                    raw_values = shap_output.values

                    if isinstance(raw_values, list):
                        raw_values = np.array(raw_values)

                    raw_values = np.array(raw_values)
                    if raw_values.ndim == 3:
                        contribution_values = raw_values[0, :, 0]
                    elif raw_values.ndim == 2:
                        contribution_values = raw_values[0]
                    elif raw_values.ndim == 1:
                        contribution_values = raw_values
                except Exception:
                    contribution_values = None

        if contribution_values is None:
            contribution_values = self._fallback_contributions(row_series)

        contribution_series = pd.Series(contribution_values, index=self.feature_names).sort_values(ascending=False)

        top_positive = [
            (feature, float(value))
            for feature, value in contribution_series.items()
            if value > 0
        ][:top_k]

        top_negative = [
            (feature, float(value))
            for feature, value in contribution_series.sort_values().items()
            if value < 0
        ][:top_k]

        summary = summarize_explanation(
            predicted_label=predicted_label,
            top_positive=top_positive,
            top_negative=top_negative,
        )

        return {
            "summary": summary,
            "top_positive": top_positive,
            "top_negative": top_negative,
        }

    def enrich_events(self, events: List[Dict[str, object]], top_k: int = 3) -> List[Dict[str, object]]:
        enriched: List[Dict[str, object]] = []

        for event in events:
            feature_vector = event.get("feature_vector")
            if not isinstance(feature_vector, dict):
                enriched.append(event)
                continue

            row = pd.Series(feature_vector)
            explanation = self.explain_row(
                row=row,
                predicted_label=str(event.get("binary_prediction", "Unknown")),
                top_k=top_k,
            )

            event["shap_summary"] = explanation["summary"]
            event["top_positive_features"] = explanation["top_positive"]
            event["top_negative_features"] = explanation["top_negative"]
            enriched.append(event)

        return enriched
