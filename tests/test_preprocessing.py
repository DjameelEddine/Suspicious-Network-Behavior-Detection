import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.preprocessing import Preprocessor


def test_preprocessor_rejects_missing_required_columns():
    scaler = StandardScaler().fit([[1.0, 2.0], [2.0, 3.0]])
    preprocessor = Preprocessor(feature_list=["a", "b"], scaler=scaler)

    raw = pd.DataFrame([{"flow_id": "flow-1", "a": 1.0}])
    output = preprocessor.transform_batch(raw)

    assert output.transformed.empty
    assert len(output.invalid_events) == 1
    assert output.invalid_events[0]["reason"] == "missing_required_columns"


def test_preprocessor_transforms_valid_rows():
    scaler = StandardScaler().fit([[1.0, 2.0], [2.0, 3.0]])
    preprocessor = Preprocessor(feature_list=["a", "b"], scaler=scaler)

    raw = pd.DataFrame(
        [
            {"flow_id": "flow-1", "a": 1.0, "b": 2.0},
            {"flow_id": "flow-2", "a": 2.0, "b": 3.0},
        ]
    )
    output = preprocessor.transform_batch(raw)

    assert len(output.invalid_events) == 0
    assert list(output.transformed.index) == ["flow-1", "flow-2"]
    assert output.transformed.shape == (2, 2)
