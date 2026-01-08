import logging.config
import os
from datetime import datetime
from pathlib import Path
from typing import Any

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


def _validate_env(env: str) -> None:
    """Validate the logging environment.

    Args:
        env: Environment name.
    """
    if env not in ENVS:
        raise ValueError(f"Invalide input env: {env}, available env: {ENVS}")


def _resolve_log_filename(env: str, file_name: str) -> str:
    """Resolve log filename with timestamp.

    Args:
        env: Environment name.
        file_name: Base log filename.

    Returns:
        Timestamped log filename.
    """
    if not file_name:
        file_name = DEFAULT_NAMES[env]

    original_path = Path(file_name)
    stem = original_path.stem
    suffix = original_path.suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stem}_{timestamp}{suffix}"


def _build_log_dir(env: str) -> Path:
    """Build log directory for the environment.

    Args:
        env: Environment name.

    Returns:
        Path to the log directory.
    """
    log_dir = BASE_DIR / settings.LOG_DIR / env
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _build_logging_config(env: str, log_file_path: Path) -> dict[str, Any]:
    """Build logging configuration dictionary.

    Args:
        env: Environment name.
        log_file_path: Full log file path.

    Returns:
        Logging configuration dictionary.
    """
    config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "verbose": {
                "format": (
                    "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
                )
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
            "handlers": [],
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
        "filename": str(log_file_path),
        "mode": "w",
        "encoding": "utf8",
        "formatter": "verbose",
    }

    config["root"]["handlers"] = ["console", file_handler_key]

    return config


def configure_logging(env: str, file_name: str = "") -> None:
    """Setup logging based on chosen environment with custom file name."""
    _validate_env(env)

    resolved_name = _resolve_log_filename(env, file_name)
    log_dir = _build_log_dir(env)
    log_file_path = log_dir / resolved_name

    config = _build_logging_config(env, log_file_path)

    logging.config.dictConfig(config)
    logging.captureWarnings(True)


def _init_wandb_run(config: dict, wandb_mode: str) -> Run:
    """Initialize a WandB run based on mode.

    Args:
        config: Run configuration.
        wandb_mode: WandB mode string.

    Returns:
        W&B Run object.
    """
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
        logging.info("WandB tracking is enabled. Run name: %s", run.name)
        return run

    run = wandb.init(mode="disabled")
    logging.info("WandB tracking is disabled.")
    return run


def setup_wandb(config: dict) -> Run:
    """Setup WandB for experiment tracking."""
    wandb_mode = os.environ.get("WANDB_MODE", "disabled")
    return _init_wandb_run(config, wandb_mode)
