import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests that require model inference.",
    )

@pytest.fixture
def run_slow(request):
    return request.config.getoption("--run-slow")

