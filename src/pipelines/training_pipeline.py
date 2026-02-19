import mlflow
import datetime
from mlflow.sklearn import log_model as mlflow_log_model
from mlflow.sklearn import autolog as mlflow_autolog
from typing import Optional

from src.monitoring.drift_detection import save_reference_data
from src.training.data_preparer import prepare_training_data
from src.training.model_trainer import train_model
from src.training.tuner import optimize_hyperparameters
from src.training.model_evaluator import evaluate_model
from src.training.model_saver import save_model
from src.sagemaker_deployment.deploy_all import deploy_to_sagemaker

from src.logging_utils.loggers import training_logger as logger
from src.config.config import config

# Enable autologging
mlflow_autolog(silent=True)


def run_training_pipeline(
    data_path: Optional[str] = None,
    verbose: Optional[bool] = None,
    deploy: Optional[bool] = None,
    tune_hyperparameters: Optional[bool] = None,
    n_trials: Optional[int] = None,
) -> tuple[str, dict]:
    """
    Orchestrates the complete training pipeline.

    Args:
        data_path (Optional[str]): Path to the training data CSV file. If None, uses default path from config.
        verbose (Optional[bool]): Whether to print detailed evaluation metrics
        deploy (Optional[bool]): Whether to automatically deploy the model to SageMaker after training. If None, uses default from config.
        tune_hyperparameters (Optional[bool]): Whether to perform hyperparameter tuning using Optuna. If None, uses default from config.
        n_trials (Optional[int]): Number of Optuna trials for hyperparameter tuning. If None, uses default from config.
    Returns:
        tuple: (model_path, metrics) where model_path is the path to the saved model and metrics is a dictionary containing evaluation metrics for train and test sets.
    """
    logger.info("Training Pipeline Started")

    verbose = (
        verbose
        if verbose is not None
        else config.get_pipeline_config.get("verbose", True)
    )
    data_path = data_path or config.get_historical_data_config.get(
        "file_path", "data/Bitcoin_history_data.csv"
    )
    deploy = (
        deploy
        if deploy is not None
        else config.get_pipeline_config.get("deploy_after_training", False)
    )
    tune_hyperparameters = (
        tune_hyperparameters
        if tune_hyperparameters is not None
        else config.get_pipeline_config.get("optimize_hyperparameters", False)
    )
    n_trials = n_trials or config.get_pipeline_config.get("n_trials", 50)

    # Set MLflow experiment
    experiment_name = config.get_mlflow_config.get(
        "experiment_name", "volatility_prediction"
    )
    mlflow.set_experiment(experiment_name)

    experiment_start_time = datetime.datetime.now()

    # Start the run
    experiment_name = experiment_name + experiment_start_time.strftime(
        "-%Y-%m-%d_%H-%M-%S"
    )

    with mlflow.start_run(run_name=experiment_name):

        mlflow.set_tag("model_type", "production")
        mlflow.set_tag("has_model_artifact", "true")

        # 1. Load and prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_training_data(
            data_path=data_path
        )

        # 2. Hyperparameter tuning (optional)
        if tune_hyperparameters:
            best_params = optimize_hyperparameters(
                X_train, y_train, X_val, y_val, n_trials
            )
            for param_name, param_value in best_params.items():
                mlflow.log_param(param_name, param_value)
            logger.info(f"Best hyperparameters from tuning: {best_params}")
            mlflow.set_tag("hyperparameter_tuning", "performed")
        else:
            best_params = {
                "C": config.get_hyperparameters_config.get("C", 0.1),
                "max_iter": config.get_hyperparameters_config.get("max_iter", 1000),
                "solver": config.get_hyperparameters_config.get("solver", "lbfgs"),
            }
            logger.info("Using default hyperparameters from config")
            mlflow.set_tag("hyperparameter_tuning", "skipped")

        # 3. Train model
        pipeline = train_model(X_train, y_train, best_params)

        # 4. Evaluate model
        metrics = evaluate_model(
            pipeline, X_train, y_train, X_val, y_val, X_test, y_test, verbose=verbose
        )

        # 5. Save model
        model_path = save_model(pipeline)

        # 6. Save reference data for Evidently monitoring
        y_train_pred = pipeline.predict(X_train)
        save_reference_data(X_train, y_train, y_train_pred)

        # 7. Log model artifacts
        model_name = config.get_mlflow_config.get("default_model_name", "model")
        mlflow_log_model(pipeline, model_name)
        thresholds = config.get_thresholds_config
        for class_name, threshold in thresholds.items():
            mlflow.log_param(f"threshold_{class_name}", threshold)

    logger.info("Training Pipeline Completed")
    logger.info(f"Model saved to: {model_path}")

    if deploy:
        logger.info("Starting deployment to SageMaker...")
        deploy_to_sagemaker()
        logger.info("Deployment to SageMaker completed.")

    return model_path, metrics


if __name__ == "__main__":
    model_path, metrics = run_training_pipeline(
        deploy=False, tune_hyperparameters=False
    )
