import argparse
import json
import logging
import os
import pickle
from multiprocessing import cpu_count

import pandas as pd
import wandb
from wandb.sdk.wandb_run import Run

from ml4investment.config.global_settings import settings
from ml4investment.utils.data_loader import sample_training_data
from ml4investment.utils.feature_calculating import calculate_features
from ml4investment.utils.feature_processing import process_features_for_train_and_validate
from ml4investment.utils.logging import configure_logging, setup_wandb
from ml4investment.utils.model_training import (
    model_training,
    optimize_data_sampling_proportion,
    optimize_features,
    optimize_model_hyperparameters,
    validate_model,
)
from ml4investment.utils.utils import set_random_seed


def train(
    run: Run,
    train_stock_list: list,
    target_stock_list: list,
    fetched_data_df: pd.DataFrame,
    seed: int,
):
    """Train model based on the given stocks"""
    logger.info(f"Start training model given stocks: {train_stock_list}")
    logger.info(f"Current trading time: {pd.Timestamp.now(tz='America/New_York')}")
    set_random_seed(seed)

    """ 1. Load necessary data """
    train_data_start_date = settings.TRAINING_DATA_START_DATE
    validation_data_end_date = settings.VALIDATION_DATA_END_DATE
    logger.info(
        f"Load input fetched data, starting from {train_data_start_date} "
        f"to {validation_data_end_date}"
    )
    train_data_df = fetched_data_df[fetched_data_df["stock_code"].isin(train_stock_list)]
    train_data_df = train_data_df.loc[train_data_start_date:validation_data_end_date]

    """ 2. Calculate and process features for all data used """
    daily_features_df = calculate_features(train_data_df)

    (
        X_train,
        y_train,
        X_validate,
        y_validate,
        process_feature_config,
    ) = process_features_for_train_and_validate(
        daily_features_df,
        apply_clip=settings.APPLY_CLIP,
        apply_scale=settings.APPLY_SCALE,
        seed=seed,
    )

    """ 3. Load data sampling proportion, features and hyperparameters """
    if args.optimize_features:
        logger.info("Optimize model features from the scratch, using all features initially")
        optimal_features = X_train.columns.tolist()
    else:
        try:
            logger.info(f"Attempting to load input model features from {args.features_pth}")
            optimal_features = json.load(open(args.features_pth, "r"))["features"]

            logger.info("Successfully loaded features.")
            X_train = X_train[optimal_features]
            X_validate = X_validate[optimal_features]

        except Exception as e:
            logger.warning(
                f"Failed to load features from {args.features_pth}. Error: {e}. "
                f"Falling back to using all features."
            )
            optimal_features = X_train.columns.tolist()

    if args.optimize_model_hyperparameters:
        logger.info("Optimize model hyperparameters from the scratch")
        optimal_model_hyperparams = settings.FIXED_TRAINING_CONFIG.copy()
        optimal_model_hyperparams.update({"seed": seed})
        optimal_model_hyperparams.update(
            {"num_threads": min(max(1, cpu_count()), settings.MAX_NUM_PROCESSES)}
        )
    else:
        try:
            logger.info(
                f"Attempting to load input model hyperparameters from {args.model_hyperparams_pth}"
            )
            optimal_model_hyperparams = json.load(open(args.model_hyperparams_pth, "r"))
            logger.info("Successfully loaded model hyperparameters.")

        except Exception as e:
            logger.warning(
                f"Failed to load model hyperparameters from {args.model_hyperparams_pth}. "
                f"Error: {e}. "
                f"Falling back to default hyperparameters."
            )
            optimal_model_hyperparams = settings.FIXED_TRAINING_CONFIG.copy()
            optimal_model_hyperparams.update({"seed": seed})
            optimal_model_hyperparams.update(
                {"num_threads": min(max(1, cpu_count()), settings.MAX_NUM_PROCESSES)}
            )

    logger.info(f"Training model with objective: {optimal_model_hyperparams['objective']}")
    logger.info(f"Training model with optimize metric: {optimal_model_hyperparams['metric']}")

    """ 4. Optimize data sampling proportion if required """
    if args.optimize_data_sampling_proportion:
        logger.info("Optimize data sampling proportion from the scratch")
        optimal_data_sampling_proportion = optimize_data_sampling_proportion(
            X_train,
            y_train,
            X_validate,
            y_validate,
            train_stock_list=train_stock_list,
            categorical_features=settings.CATEGORICAL_FEATURES,
            model_hyperparams=optimal_model_hyperparams,
            given_data_sampling_proportion_pth=args.data_sampling_proportion_pth,
            seed=seed,
            verbose=args.verbose,
        )
    else:
        try:
            logger.info(
                f"Attempting to load input data sampling proportion "
                f"from {args.data_sampling_proportion_pth}"
            )
            optimal_data_sampling_proportion = json.load(
                open(args.data_sampling_proportion_pth, "r")
            )
            logger.info("Successfully loaded data sampling proportion.")

        except Exception as e:
            logger.warning(
                f"Failed to load data sampling proportion "
                f"from {args.data_sampling_proportion_pth}. "
                f"Error: {e}. "
                f"Falling back to use all data."
            )
            optimal_data_sampling_proportion = {
                stock: 1.0 for stock in train_stock_list if stock not in settings.SELECTIVE_ETF
            }

    X_train, y_train = sample_training_data(
        X_train,
        y_train,
        sampling_proportion=optimal_data_sampling_proportion,
        seed=seed,
    )

    """ 5. Optimize features if required """
    if args.optimize_features:
        optimal_features = optimize_features(
            X_train,
            y_train,
            X_validate,
            y_validate,
            all_features=X_train.columns.tolist(),
            categorical_features=settings.CATEGORICAL_FEATURES,
            given_features_pth=args.features_pth,
            model_hyperparams=optimal_model_hyperparams,
            seed=seed,
            verbose=args.verbose,
        )
        X_train = X_train[optimal_features]
        X_validate = X_validate[optimal_features]

    """ 6. Optimize the model hyperparameters if required """
    if args.optimize_model_hyperparameters:
        optimal_model_hyperparams = optimize_model_hyperparameters(
            X_train,
            y_train,
            X_validate,
            y_validate,
            categorical_features=settings.CATEGORICAL_FEATURES,
            given_model_hyperparams_pth=args.model_hyperparams_pth,
            seed=seed,
            verbose=args.verbose,
        )

    """ 7. Train, validate the final model and apply prediction stock optimization if required """
    logger.info(f"Oldest date in training data: {X_train.index.min()}")
    logger.info(f"Newest date in training data: {X_train.index.max()}")
    logger.info(f"Total processed samples in training data: {X_train.shape[0]}")
    logger.info(f"Number of features in training data: {X_train.shape[1]}")

    logger.info(f"Oldest date in validation data: {X_validate.index.min()}")
    logger.info(f"Newest date in validation data: {X_validate.index.max()}")
    logger.info(f"Total processed samples in validation data: {X_validate.shape[0]}")
    logger.info(f"Number of features in validation data: {X_validate.shape[1]}")

    final_model, _ = model_training(
        X_train,
        y_train,
        X_validate,
        y_validate,
        categorical_features=settings.CATEGORICAL_FEATURES,
        model_hyperparams=optimal_model_hyperparams,
        show_training_log=True,
    )

    (
        valid_day_number,
        valid_mae,
        valid_mse,
        valid_sign_acc,
        valid_precision,
        valid_recall,
        valid_f1,
        vaild_average_daily_gain,
        vaild_overall_gain,
        valid_annualized_sharpe_ratio,
        validate_max_drawdown,
        sorted_stocks,
    ) = validate_model(
        final_model,
        X_validate,
        y_validate,
        target_stock_list=target_stock_list,
        verbose=args.verbose,
    )

    if args.optimize_predict_stocks:
        logger.info("Begin predict stocks optimization")
        logger.info(
            f"Using {settings.PREDICT_STOCK_OPTIMIZE_METRIC} "
            f"as the predict stocks optimization metric "
            f"with target number {settings.PREDICT_STOCK_NUMBER}"
        )

        predict_stock_list = sorted_stocks[: settings.PREDICT_STOCK_NUMBER]
        logger.info(
            f"Selected {len(predict_stock_list)} stocks for prediction: "
            f"{', '.join(predict_stock_list)}"
        )
    else:
        logger.info("No predict stocks optimization. Using all target stocks as predict stocks")
        predict_stock_list = target_stock_list

    """ 8. Save all results"""
    os.makedirs(os.path.dirname(args.save_process_feature_config_pth), exist_ok=True)
    with open(args.save_process_feature_config_pth, "wb") as f:
        pickle.dump(process_feature_config, f)
    logger.info(
        f"Processing features configuration saved to {args.save_process_feature_config_pth}"
    )

    os.makedirs(os.path.dirname(args.save_model_pth), exist_ok=True)
    final_model.save_model(args.save_model_pth)
    logger.info(f"Model saved to {args.save_model_pth}")

    os.makedirs(os.path.dirname(args.save_data_sampling_proportion_pth), exist_ok=True)
    with open(args.save_data_sampling_proportion_pth, "w") as f:
        json.dump(optimal_data_sampling_proportion, f, indent=4)
    logger.info(
        f"Optimized data sampling proportion saved to {args.save_data_sampling_proportion_pth}"
    )

    os.makedirs(os.path.dirname(args.save_features_pth), exist_ok=True)
    optimal_features = {"features": optimal_features}
    with open(args.save_features_pth, "w") as f:
        json.dump(optimal_features, f, indent=4)
    logger.info(f"Optimized features saved to {args.save_features_pth}")

    os.makedirs(os.path.dirname(args.save_model_hyperparams_pth), exist_ok=True)
    with open(args.save_model_hyperparams_pth, "w") as f:
        json.dump(optimal_model_hyperparams, f, indent=4)
    logger.info(f"Optimized model hyperparameters saved to {args.save_model_hyperparams_pth}")

    predict_stocks = {"predict_stocks": predict_stock_list}
    os.makedirs(os.path.dirname(args.save_predict_stocks_pth), exist_ok=True)
    with open(args.save_predict_stocks_pth, "w") as f:
        json.dump(predict_stocks, f, indent=4)
    logger.info(f"Predict stocks saved to {args.save_predict_stocks_pth}")

    wandb.log(
        {
            "valid_day_number": valid_day_number,
            "valid_mae": valid_mae,
            "valid_mse": valid_mse,
            "valid_sign_acc": valid_sign_acc,
            "valid_precision": valid_precision,
            "valid_recall": valid_recall,
            "valid_f1": valid_f1,
            "valid_average_daily_gain": vaild_average_daily_gain,
            "valid_overall_gain": vaild_overall_gain,
            "valid_annualized_sharpe_ratio": valid_annualized_sharpe_ratio,
            "valid_max_drawdown": validate_max_drawdown,
        }
    )

    run.finish()

    logger.info("Training process completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_stocks", type=str, default="config/train_stocks.json")
    parser.add_argument("--target_stocks", type=str, default="config/target_stocks.json")
    parser.add_argument(
        "--fetched_data_pth", "-fdp", type=str, default="data/fetched_data.parquet"
    )
    parser.add_argument(
        "--optimize_data_sampling_proportion",
        "-odsp",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--data_sampling_proportion_pth",
        "-dspp",
        type=str,
        default="data/prod_data_sampling_proportion.json",
    )
    parser.add_argument("--optimize_features", "-of", action="store_true", default=False)
    parser.add_argument("--features_pth", "-fp", type=str, default="data/prod_features.json")
    parser.add_argument(
        "--optimize_model_hyperparameters", "-omhp", action="store_true", default=False
    )
    parser.add_argument(
        "--model_hyperparams_pth",
        "-mhpp",
        type=str,
        default="data/prod_model_hyperparams.json",
    )
    parser.add_argument("--optimize_predict_stocks", "-ops", action="store_true", default=False)

    parser.add_argument(
        "--save_process_feature_config_pth",
        "-spfcp",
        type=str,
        default="data/prod_process_feature_config.pkl",
    )
    parser.add_argument(
        "--save_data_sampling_proportion_pth",
        "-sdspp",
        type=str,
        default="data/prod_data_sampling_proportion.json",
    )
    parser.add_argument("--save_model_pth", "-smp", type=str, default="data/prod_model.model")
    parser.add_argument("--save_features_pth", "-sfp", type=str, default="data/prod_features.json")
    parser.add_argument(
        "--save_model_hyperparams_pth",
        "-smhpp",
        type=str,
        default="data/prod_model_hyperparams.json",
    )
    parser.add_argument(
        "--save_predict_stocks_pth", type=str, default="data/predict_stocks.json"
    )

    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    parser.add_argument("--seed", "-s", type=int, default=settings.SEED)

    args = parser.parse_args()

    configure_logging(env="train", file_name="train.log")
    logger = logging.getLogger("ml4investment.train")

    train_stock_list = json.load(open(args.train_stocks, "r"))["train_stocks"]
    target_stock_list = json.load(open(args.target_stocks, "r"))["target_stocks"]
    fetched_data_df = pd.read_parquet(args.fetched_data_pth)
    seed = args.seed

    activated_optimizations = [
        args.optimize_data_sampling_proportion,
        args.optimize_features,
        args.optimize_model_hyperparameters,
    ]

    num_activated = sum(1 for x in activated_optimizations if x)

    if num_activated > 1:
        error_message = (
            f"Error: Only one optimization option can be active at a time. "
            f"Detected {num_activated} active optimizations: "
        )
        active_options_names = []
        if args.optimize_data_sampling_proportion:
            active_options_names.append("optimize_data_sampling_proportion")
        if args.optimize_features:
            active_options_names.append("optimize_features")
        if args.optimize_model_hyperparameters:
            active_options_names.append("optimize_model_hyperparameters")

        error_message += ", ".join(active_options_names) + ". Please enable only one."

        logging.error(error_message)
        raise ValueError(error_message)

    run = setup_wandb(config=vars(args))

    train(run, train_stock_list, target_stock_list, fetched_data_df, seed)
