from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def now_utc_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def normalize_columns(columns: Iterable[str]) -> List[str]:
    return [str(column).strip() for column in columns]


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_risk_score(
    is_attack: bool,
    confidence: float,
    attack_type: str,
    severity_map: Dict[str, str],
) -> float:
    if not is_attack:
        return max(5.0, round((1.0 - confidence) * 20.0, 2))

    severity = severity_map.get(attack_type, "medium").lower()
    severity_boost = {"low": 0.1, "medium": 0.25, "high": 0.4}.get(severity, 0.25)
    score = min(1.0, (confidence * 0.7) + severity_boost)
    return round(score * 100.0, 2)


def compute_risk_level(
    is_attack: bool,
    confidence: float,
    attack_type: str,
    severity_map: Dict[str, str],
    low_threshold: float,
    high_threshold: float,
) -> str:
    if not is_attack:
        return "Low"

    severity = severity_map.get(attack_type, "medium").lower()

    if confidence >= high_threshold and severity in {"medium", "high"}:
        return "High"
    if severity == "high" and confidence >= low_threshold:
        return "High"
    if confidence >= low_threshold:
        return "Medium"
    return "Low"


def health_status(avg_latency_ms: float, error_rate: float) -> str:
    if error_rate >= 0.1 or avg_latency_ms >= 2500:
        return "Critical"
    if error_rate >= 0.03 or avg_latency_ms >= 1200:
        return "Warning"
    return "Healthy"


def redact_internal_path(message: str, project_root: Path) -> str:
    root_text = str(project_root).replace("\\", "/")
    return message.replace(root_text, "[project-root]")


def summarize_explanation(
    predicted_label: str,
    top_positive: Sequence[Tuple[str, float]],
    top_negative: Sequence[Tuple[str, float]],
) -> str:
    if not top_positive and not top_negative:
        return f"Prediction '{predicted_label}' was generated without available feature attributions."

    positive_part = ""
    negative_part = ""

    if top_positive:
        positive_feature, positive_value = top_positive[0]
        positive_part = (
            f"The strongest push toward the current prediction came from '{positive_feature}' "
            f"({positive_value:.4f})."
        )

    if top_negative:
        negative_feature, negative_value = top_negative[0]
        negative_part = (
            f" The strongest counter-signal came from '{negative_feature}' ({negative_value:.4f})."
        )

    return f"{positive_part}{negative_part}".strip()
