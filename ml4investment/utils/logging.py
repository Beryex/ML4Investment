import logging.config
import os
from datetime import datetime
from pathlib import Path

import wandb
from wandb.sdk.wandb_run import Run

from ml4investment.config.global_settings import settings

BASE_DIR = Path(__file__).resolve().parent.parent
ENVS = ("fetch_data", "train", "backtest", "predict", "test")
DEFAULT_NAMES = {
    "fetch_data": "fetch_data.log",
    "train": "train.log",
    "backtest": "backtest.log",
    "predict": "predict.log",
    "test": "test.log",
}


def configure_logging(env: str = "prod", file_name: str = "") -> None:
    """Setup logging based on chosen environment with custom file name"""
    if env not in ENVS:
        raise ValueError(f"Invalide input env: {env}, available env: {ENVS}")

    if file_name == "":
        file_name = DEFAULT_NAMES[env]

    """ Make file name unique by adding timestamp """
    original_path = Path(file_name)
    stem = original_path.stem
    suffix = original_path.suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{stem}_{timestamp}{suffix}"

    log_dir = BASE_DIR / "logs" / env
    log_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "verbose": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "verbose",
                "stream": "ext://sys.stdout",
            }
        },
        "root": {
            "handlers": [],  # set dynamically later
            "level": "DEBUG" if env in ("test", "dev") else "INFO",
        },
        "loggers": {
            "yfinance": {"level": "WARNING", "propagate": True},
            "peewee": {"level": "WARNING", "propagate": True},
            "urllib3": {"level": "WARNING", "propagate": True},
            "optuna": {
                "level": "INFO",
                "propagate": True,
            },
            "pandas": {"level": "WARNING", "propagate": True},
            "lightgbm": {"level": "WARNING", "propagate": True},
            "arch": {"level": "WARNING", "propagate": True},
        },
    }

    file_handler_key = f"{env}_file"
    config["handlers"][file_handler_key] = {
        "class": "logging.FileHandler",
        "filename": str(log_dir / file_name),
        "mode": "w",
        "encoding": "utf8",
        "formatter": "verbose",
    }

    config["root"]["handlers"] = ["console", file_handler_key]

    logging.config.dictConfig(config)
    logging.captureWarnings(True)


def setup_wandb(config: dict) -> Run:
    """Setup WandB for experiment tracking"""
    wandb_mode = os.environ.get("WANDB_MODE", "disabled")

    run: Run
    if wandb_mode == "online":
        wandb_group = os.environ.get("WANDB_RUN_GROUP")
        wandb_name = os.environ.get("WANDB_RUN_NAME")
        wandb_job_type = os.environ.get("WANDB_JOB_TYPE")

        run = wandb.init(
            mode=wandb_mode,
            project=settings.PROJECT_NAME,
            name=wandb_name,
            group=wandb_group,
            job_type=wandb_job_type,
            config=config,
            reinit="finish_previous",
        )
        logging.info(f"WandB tracking is enabled. Run name: {run.name}")

    else:
        run = wandb.init(mode="disabled")
        logging.info("WandB tracking is disabled.")

    return run
