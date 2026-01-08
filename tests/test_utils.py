import numpy as np

from ml4investment.utils.utils import _coerce_stock_id, id_to_stock_code, stock_code_to_id


def test_stock_code_round_trip():
    for code in ["AAPL", "QQQ", "BRK-B"]:
        code_id = stock_code_to_id(code)
        assert id_to_stock_code(code_id) == code


def test_coerce_stock_id_handles_numpy_scalars():
    value = _coerce_stock_id(np.int64(123))
    assert isinstance(value, int)
    assert value == 123
