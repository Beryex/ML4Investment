import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from ml4investment.utils.model_training import model_training


def test_model_training_runs_on_small_dataset():
    rng = np.random.default_rng(0)
    X_train = pd.DataFrame(rng.normal(size=(20, 3)), columns=["f1", "f2", "f3"])
    y_train = pd.Series(rng.normal(size=20))
    X_validate = pd.DataFrame(rng.normal(size=(8, 3)), columns=["f1", "f2", "f3"])
    y_validate = pd.Series(rng.normal(size=8))

    model_hyperparams = {
        "objective": "regression_l1",
        "metric": "l1",
        "verbosity": -1,
        "num_rounds": 5,
        "seed": 1,
        "force_row_wise": True,
        "deterministic": True,
        "num_threads": 1,
    }

    model, score = model_training(
        X_train,
        y_train,
        X_validate,
        y_validate,
        categorical_features=[],
        model_hyperparams=model_hyperparams,
        show_training_log=False,
    )

    assert model.best_iteration > 0
    assert isinstance(score, float)
