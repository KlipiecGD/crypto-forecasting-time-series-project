from typing import Optional

from src.pipelines.training_pipeline import run_training_pipeline
from src.sagemaker_deployment.deploy_all import deploy_to_sagemaker
from src.logging_utils.loggers import deployment_logger as logger
from src.config.config import config


def run_full_pipeline(verbose: Optional[bool] = None) -> dict:
    """
    Orchestrates the full end-to-end pipeline: training + cloud deployment.

    Args:
        verbose (bool): Whether to print detailed logs. If None, defaults to the value in config.

    Returns:
        dict: Summary containing model_path, training metrics, and deployment info.
    """

    # ===== STEP 1: Train Model =====
    verbose = (
        verbose
        if verbose is not None
        else config.get_pipeline_config.get("verbose", True)
    )

    logger.info("=" * 60)
    logger.info("PHASE 1: MODEL TRAINING")
    logger.info("=" * 60)
    model_path, metrics = run_training_pipeline(verbose=verbose)

    # ===== STEP 2: Deploy to SageMaker =====
    logger.info("=" * 60)
    logger.info("PHASE 2: CLOUD DEPLOYMENT")
    logger.info("=" * 60)
    deployment_info = deploy_to_sagemaker()

    # ===== DONE =====
    logger.info("=" * 60)
    logger.info("ORCHESTRATION COMPLETE")
    logger.info("=" * 60)

    return {
        "model_path": model_path,
        "metrics": metrics,
        "deployment_info": deployment_info,
    }


if __name__ == "__main__":
    run_full_pipeline()
