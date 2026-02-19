from src.pipelines.training_pipeline import run_training_pipeline
from src.sagemaker_deployment.deploy_all import deploy_to_sagemaker
from src.elastic_beanstalk.deploy_elasticbeanstalk import deploy_streamlit_app

from src.logging_utils.loggers import deployment_logger as logger


def run_complete_deployment() -> str:
    """
    Main orchestrator for the crypto forecasting project.
    Steps:
        1. Training the model using the training pipeline.
        2. Deploy the model to AWS SageMaker.
        3. Deploy Streamlit app to AWS Elastic Beanstalk.
    Returns:
        string: The URL of the deployed Streamlit app.
    """
    # Step 1: Train the model
    run_training_pipeline()

    # Step 2: Deploy the model to AWS SageMaker
    deploy_to_sagemaker()

    # Step 3: Deploy Streamlit app to AWS Elastic Beanstalk
    result = deploy_streamlit_app()

    # Step 4: Return the URL of the deployed Streamlit app
    url = result["url"]
    logger.info(f"Deployment completed successfully. Streamlit app URL: {url}")

    return url


if __name__ == "__main__":
    run_complete_deployment()
