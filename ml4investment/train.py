import argparse
import logging
import pandas as pd
import json
import os
import pickle
from argparse import Namespace

from ml4investment.config import settings
from ml4investment.utils.utils import set_random_seed
from ml4investment.utils.logging import configure_logging
from ml4investment.utils.data_loader import sample_training_data
from ml4investment.utils.feature_engineering import calculate_features, process_features_for_train_and_validate
from ml4investment.utils.model_training import model_training, optimize_data_sampling_proportion, optimize_model_features

configure_logging(env="train", file_name="train.log")
logger = logging.getLogger("ml4investment.train")


def train(train_stock_list: list, 
          target_stock_list: list,
          fetched_data: dict, 
          seed: int,
          args: Namespace):
    """ Train model based on the given stocks """
    logger.info(f"Start training model given stocks: {train_stock_list}")
    logger.info(f"Current trading time: {pd.Timestamp.now(tz='America/New_York')}")

    """ 1. Load necessary data """
    train_data = {}
    train_data_start_date = settings.TRAINING_DATA_START_DATE
    validation_data_end_date = settings.VALIDATION_DATA_END_DATE
    logger.info(f"Load input fetched data, starting from {train_data_start_date} to {validation_data_end_date}")
    for stock in train_stock_list:
        train_data[stock] = fetched_data[stock].loc[train_data_start_date:validation_data_end_date]

    """ 2. Calculate and process features for all data used """
    daily_features_data = calculate_features(train_data)

    X_train, y_train, X_validate, y_validate, process_feature_config = process_features_for_train_and_validate(
        daily_features_data,  
        seed=seed
    )

    """ 3. Load model features, hyperparameters and data sampling proportion """
    if args.optimize_model_features:
        logger.info("Optimize model features from the scratch, using all features initially")
    else:
        logger.info(f"Load input selected features")
        selected_features = json.load(open(args.features_pth, 'r'))["features"]
        X_train = X_train[selected_features]
        X_validate = X_validate[selected_features]

    if args.optimize_model_hyperparameters:
        logger.info("Optimize model hyperparameters from the scratch")
        model_hyperparams = None
    else:
        logger.info("Load input model hyperparameter")
        model_hyperparams = json.load(open(args.model_hyperparams_pth, 'r'))

    if args.optimize_data_sampling_proportion:
        logger.info("Optimize data sampling proportion from the scratch")
        optimal_data_sampling_proportion = optimize_data_sampling_proportion(
            X_train, y_train, 
            X_validate, y_validate,
            categorical_features=settings.CATEGORICAL_FEATURES,
            model_hyperparams=model_hyperparams,
            target_stock_list=target_stock_list,
            seed=seed,
            verbose=args.verbose
        )
    else:
        logger.info(f"Load input data sampling proportion")
        optimal_data_sampling_proportion = json.load(open(args.data_sampling_proportion_pth, 'r'))

    """ 4. Apply sampling to training data """
    # X_train, y_train = sample_training_data(X_train, y_train, optimal_data_sampling_proportion, seed=seed)

    """ 5. Log training data information """
    logger.info(f"Oldest date in training data: {X_train.index.min()}")
    logger.info(f"Newest date in training data: {X_train.index.max()}")
    logger.info(f"Total processed samples: {X_train.shape[0]}")
    logger.info(f"Number of features: {X_train.shape[1]}")
    
    """ 6. Train the model"""
    optimal_model, optimal_model_hyperparams, optimal_mae, optimal_sign_accuracy, sorted_feature_imp, optimal_predict_stock_list = model_training(
        X_train, y_train, 
        X_validate, y_validate,
        categorical_features=settings.CATEGORICAL_FEATURES,
        model_hyperparams=model_hyperparams,
        target_stock_list=target_stock_list,
        optimize_predict_stocks=args.optimize_predict_stocks,
        seed=seed,
        verbose=args.verbose
    )
    
    """ 7. Optimize model features if required """
    if args.optimize_model_features:
        optimal_model, optimal_model_hyperparams, optimal_features, optimal_predict_stock_list = optimize_model_features(
            X_train, y_train, 
            categorical_features=settings.CATEGORICAL_FEATURES,
            model_hyperparams=model_hyperparams,
            target_stock_list=target_stock_list,
            optimize_predict_stocks=args.optimize_predict_stocks,
            original_model=optimal_model,
            original_sorted_feature_imp=sorted_feature_imp,
            original_features=list(X_train.columns),
            original_mae=optimal_mae,
            original_sign_accuracy=optimal_sign_accuracy,
            seed=seed,
            verbose=args.verbose
        )

    """ 8. Save all results"""
    os.makedirs(os.path.dirname(args.save_process_feature_config_pth), exist_ok=True)
    with open(args.save_process_feature_config_pth, 'wb') as f:
        pickle.dump(process_feature_config, f)
    logger.info(f"Processing features configuration saved to {args.save_process_feature_config_pth}")

    os.makedirs(os.path.dirname(args.save_model_pth), exist_ok=True)
    optimal_model.save_model(args.save_model_pth)
    logger.info(f"Model saved to {args.save_model_pth}")

    if args.optimize_data_sampling_proportion:
        os.makedirs(os.path.dirname(args.save_data_sampling_proportion_pth), exist_ok=True)
        with open(args.save_data_sampling_proportion_pth, 'w') as f:
            json.dump(optimal_data_sampling_proportion, f, indent=4)
        logger.info(f"Optimized data sampling proportion saved to {args.save_data_sampling_proportion_pth}")

    if args.optimize_model_features:
        os.makedirs(os.path.dirname(args.save_features_pth), exist_ok=True)
        optimal_features = {"features": optimal_features}
        with open(args.save_features_pth, 'w') as f:
            json.dump(optimal_features, f, indent=4)
        logger.info(f"Optimized features saved to {args.save_features_pth}")

    if args.optimize_model_hyperparameters:
        os.makedirs(os.path.dirname(args.save_model_hyperparams_pth), exist_ok=True)
        with open(args.save_model_hyperparams_pth, 'w') as f:
            json.dump(optimal_model_hyperparams, f, indent=4)
        logger.info(f"Optimized model hyperparameters saved to {args.save_model_hyperparams_pth}")

    predict_stocks = {"predict_stocks": optimal_predict_stock_list}
    os.makedirs(os.path.dirname(args.save_predict_stocks_pth), exist_ok=True)
    with open(args.save_predict_stocks_pth, 'w') as f:
        json.dump(predict_stocks, f, indent=4)
    logger.info(f"Target stocks saved to {args.save_predict_stocks_pth}")
    
    logger.info("Training process completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_stocks", type=str, default='config/train_stocks.json')
    parser.add_argument("--target_stocks", type=str, default='config/target_stocks.json')
    parser.add_argument("--fetched_data_pth", "-fdp", type=str, default='data/fetched_data.pkl')
    parser.add_argument("--optimize_data_sampling_proportion", "-odsp", action='store_true', default=False)
    parser.add_argument("--data_sampling_proportion_pth", "-dspp", type=str, default='data/prod_data_sampling_proportion.json')
    parser.add_argument("--optimize_model_features", "-omf", action='store_true', default=False)
    parser.add_argument("--features_pth", "-fp", type=str, default='data/prod_model_features.json')
    parser.add_argument("--optimize_model_hyperparameters", "-omhp", action='store_true', default=False)
    parser.add_argument("--model_hyperparams_pth", "-mhpp", type=str, default='data/prod_model_hyperparams.json')
    parser.add_argument("--optimize_predict_stocks", "-ops", action='store_true', default=False)

    parser.add_argument("--save_process_feature_config_pth", "-spfcp", type=str, default='data/prod_process_feature_config.pkl')
    parser.add_argument("--save_data_sampling_proportion_pth", "-sdspp", type=str, default='data/prod_data_sampling_proportion.json')
    parser.add_argument("--save_model_pth", "-smp", type=str, default='data/prod_model.model')
    parser.add_argument("--save_features_pth", "-sfp", type=str, default='data/prod_model_features.json')
    parser.add_argument("--save_model_hyperparams_pth", "-smhpp", type=str, default='data/prod_model_hyperparams.json')
    parser.add_argument("--save_predict_stocks_pth", type=str, default='config/predict_stocks.json')

    parser.add_argument("--verbose", "-v", action='store_true', default=False)
    parser.add_argument("--seed", "-s", type=int, default=42)

    args = parser.parse_args()
    
    seed = args.seed
    set_random_seed(seed)

    train_stock_list = json.load(open(args.train_stocks, 'r'))["train_stocks"]
    target_stock_list = json.load(open(args.target_stocks, 'r'))["target_stocks"]
    fetched_data = pickle.load(open(args.fetched_data_pth, 'rb'))

    train(train_stock_list, target_stock_list, fetched_data, seed, args)
