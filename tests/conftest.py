def pytest_addoption(parser):
    """Add custom command-line options for pytest."""
    parser.addoption(
        "--evaluator-class",
        action="store",
        default=None,
        help="Specific evaluator class name to test (e.g., LogisticRegressionEvaluator)",
    )

    parser.addoption(
        "--embedding-class",
        action="store",
        default=None,
        help="Specific embedding model name to test (e.g., TabICLEmbedding)",
    )
