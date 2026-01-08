import pytest

from ml4investment.utils.logging import configure_logging


def test_configure_logging_rejects_invalid_env():
    with pytest.raises(ValueError):
        configure_logging(env="unknown-env")
