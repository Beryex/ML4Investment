import os
import random
import numpy as np
import logging

from ml4investment.config import settings

logger = logging.getLogger(__name__)


def set_random_seed(seed: int) -> None:
    """ Set random seed for reproducible usage """
    logger.info(f"Set random seed: {seed}")
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def stock_code_to_id(stock_code: str) -> int:
    """ Change the stock string to the sum of ASCII value of each char within the stock code """
    return sum(ord(c) * 256 ** i for i, c in enumerate(reversed(stock_code)))


def id_to_stock_code(code_id: int) -> str:
    """  Change the stock id to the string of stock code """
    chars = []
    while code_id > 0:
        ascii_val = code_id % 256
        chars.append(chr(ascii_val))
        code_id //= 256
    return ''.join(reversed(chars))
    