from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
EXPORTS_DIR = ROOT_DIR / "exports"


DEFAULT_DATASET_CANDIDATES = (
    ROOT_DIR / "clean_traffic.csv",
    ROOT_DIR / "traffic.csv",
    ROOT_DIR / "data" / "clean_traffic.csv",
    ROOT_DIR / "data" / "traffic.csv",
)


LEGACY_BINARY_MODEL_CANDIDATES = (
    MODELS_DIR / "binary" / "xgboost_binary_tuned.joblib",
    MODELS_DIR / "binary" / "random_forest_binary_baseline.joblib",
    MODELS_DIR / "binary" / "logistic_regression_binary.joblib",
    MODELS_DIR / "binary" / "decision_tree_binary.joblib",
    MODELS_DIR / "binary" / "knn_binary.joblib",
)


LEGACY_MULTI_MODEL_CANDIDATES = (
    MODELS_DIR / "multi" / "knn_multi_tuned.joblib",
    MODELS_DIR / "multi" / "random_forest_multi_baseline.joblib",
    MODELS_DIR / "multi" / "xgboost_multi.joblib",
    MODELS_DIR / "multi" / "logistic_regression_multi.joblib",
    MODELS_DIR / "multi" / "decision_tree_multi.joblib",
)


DEFAULT_ATTACK_SEVERITY = {
    "BENIGN": "low",
    "Bot": "high",
    "DDoS": "high",
    "DoS GoldenEye": "high",
    "DoS Hulk": "high",
    "DoS Slowhttptest": "medium",
    "DoS slowloris": "medium",
    "FTP-Patator": "medium",
    "Heartbleed": "high",
    "Infiltration": "high",
    "PortScan": "medium",
    "SSH-Patator": "medium",
    "Web Attack - Brute Force": "medium",
    "Web Attack - Sql Injection": "high",
    "Web Attack - XSS": "medium",
}


@dataclass(frozen=True)
class ArtifactPaths:
    binary_model: Path = MODELS_DIR / "binary_model.pkl"
    multiclass_model: Path = MODELS_DIR / "multiclass_model.pkl"
    scaler: Path = MODELS_DIR / "scaler.pkl"
    feature_list: Path = MODELS_DIR / "feature_list.pkl"
    metadata: Path = MODELS_DIR / "metadata.json"
    label_encoder: Path = MODELS_DIR / "label_encoder.pkl"

    def binary_candidates(self) -> List[Path]:
        return [self.binary_model, *LEGACY_BINARY_MODEL_CANDIDATES]

    def multiclass_candidates(self) -> List[Path]:
        return [self.multiclass_model, *LEGACY_MULTI_MODEL_CANDIDATES]


@dataclass(frozen=True)
class RiskConfig:
    low_confidence_threshold: float = 0.55
    high_confidence_threshold: float = 0.85
    attack_severity: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ATTACK_SEVERITY))


@dataclass(frozen=True)
class RuntimeConfig:
    default_rows_per_second: float = 5.0
    default_batch_size: int = 10
    max_events_kept: int = 5000
    max_display_rows: int = 300
    auto_refresh_default: bool = True


@dataclass(frozen=True)
class AppConfig:
    root_dir: Path = ROOT_DIR
    models_dir: Path = MODELS_DIR
    exports_dir: Path = EXPORTS_DIR
    artifact_paths: ArtifactPaths = field(default_factory=ArtifactPaths)
    risk: RiskConfig = field(default_factory=RiskConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


APP_CONFIG = AppConfig()


def resolve_first_existing(candidates: Iterable[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_dataset_path(preferred: Optional[str] = None) -> Optional[Path]:
    if preferred:
        preferred_path = Path(preferred)
        if preferred_path.exists():
            return preferred_path

    for candidate in DEFAULT_DATASET_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def validate_artifacts(config: AppConfig = APP_CONFIG) -> List[str]:
    errors: List[str] = []

    binary_model = resolve_first_existing(config.artifact_paths.binary_candidates())
    multiclass_model = resolve_first_existing(config.artifact_paths.multiclass_candidates())

    if binary_model is None:
        errors.append(
            "Binary model artifact not found. Expected models/binary_model.pkl or a supported legacy binary model file."
        )
    if multiclass_model is None:
        errors.append(
            "Multi-class model artifact not found. Expected models/multiclass_model.pkl or a supported legacy multi-class model file."
        )
    if not config.artifact_paths.scaler.exists():
        errors.append("Scaler artifact missing: models/scaler.pkl")
    if not config.artifact_paths.feature_list.exists():
        errors.append("Feature list artifact missing: models/feature_list.pkl")

    return errors
