from aegisml.modeling import train_model
from aegisml.monitoring import evaluate_batch
from aegisml.storage import initialize


def test_scenarios_have_expected_severity() -> None:
    initialize()
    train_model()
    assert evaluate_batch("healthy")["severity"] == "healthy"
    assert evaluate_batch("warning")["severity"] == "warning"
    assert evaluate_batch("critical")["severity"] == "critical"
