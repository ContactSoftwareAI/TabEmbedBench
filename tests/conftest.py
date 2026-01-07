def pytest_addoption(parser):
    """Add custom command-line options for pytest."""
    parser.addoption(
        "--evaluator-class",
        action="store",
        default=None,
        help="Specific evaluator class name to test (e.g., KNNRegressor)",
    )
