from aegisml.monitoring import psi


def test_psi_is_zero_for_equal_distributions() -> None:
    assert psi([0.3, 0.7], [0.3, 0.7]) == 0.0


def test_psi_increases_for_shifted_distribution() -> None:
    assert psi([0.5, 0.5], [0.95, 0.05]) > 0.5

