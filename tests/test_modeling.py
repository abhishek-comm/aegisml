from aegisml.modeling import generate_customers


def test_critical_scenario_is_observably_different() -> None:
    healthy, _ = generate_customers(1000, "healthy", seed=1)
    critical, _ = generate_customers(1000, "critical", seed=1)
    assert critical[:, 1].mean() > healthy[:, 1].mean()
    assert critical[:, 2].mean() > healthy[:, 2].mean()

