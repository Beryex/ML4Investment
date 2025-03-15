import os
import random
import numpy as np
import logging

logger = logging.getLogger(__name__)


def set_random_seed(seed: int) -> None:
    """ Set random seed for reproducible usage """
    logger.info(f"Set random seed: {seed}")
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
