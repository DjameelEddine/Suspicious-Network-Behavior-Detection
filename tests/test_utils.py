from src.utils import compute_risk_level, compute_risk_score, health_status


def test_benign_flow_has_low_risk():
    severity = {"BENIGN": "low"}
    risk = compute_risk_level(False, 0.95, "BENIGN", severity, 0.55, 0.85)
    assert risk == "Low"


def test_high_confidence_severe_attack_is_high_risk():
    severity = {"Heartbleed": "high"}
    risk = compute_risk_level(True, 0.91, "Heartbleed", severity, 0.55, 0.85)
    assert risk == "High"


def test_risk_score_bounds():
    severity = {"DoS GoldenEye": "high"}
    score = compute_risk_score(True, 0.99, "DoS GoldenEye", severity)
    assert 0.0 <= score <= 100.0


def test_health_status_warning_on_latency():
    assert health_status(avg_latency_ms=1300.0, error_rate=0.0) == "Warning"
