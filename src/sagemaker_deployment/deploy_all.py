import os
import json
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from src.logging_utils.loggers import deployment_logger as logger
from src.config.config import config

from src.sagemaker_deployment.package_model import package_model_for_docker
from src.sagemaker_deployment.build_and_push import DockerBuilder
from src.sagemaker_deployment.deploy_container import ContainerDeployer


def deploy_to_sagemaker(rebuild_image: Optional[bool] = None) -> dict:
    """
    Complete end-to-end SageMaker deployment pipeline.
    Executes all three steps: package -> build (optional) -> deploy
    If rebuild_image is True, it will force rebuild the Docker image. If False, it will check if an image with the same name already exists and reuse it. If None, it will be determined by config.
    It will check whether the endpoint already exists and if it does, it will update the existing endpoint with the new model instead of creating a new one. This ensures zero downtime during deployment.
    Args:
        rebuild_image (bool): Whether to force rebuild the Docker image. If None, it will be determined by config.
    Returns:
        dict: Deployment information including endpoint name, model name, region, and image URI.
    """
    logger.info("Starting SageMaker deployment pipeline...")

    rebuild_image = (
        rebuild_image
        if rebuild_image is not None
        else config.get_sagemaker_deployment_config.get("rebuild_image", True)
    )

    try:
        # ===== STEP 1: Package Model =====
        logger.info("\n[STEP 1/3] Packaging model...")
        model_tar_path = package_model_for_docker(
            model_path=config.get_model_config.get(
                "model_save_path", "models/volatility_model.joblib"
            ),
            output_path=config.get_sagemaker_deployment_config.get(
                "model_output_path", "sagemaker_model.tar.gz"
            ),
        )
        logger.info(f"Model packaged: {model_tar_path}")

        # ===== STEP 2: Build and Push Docker Image (optional) =====
        logger.info("\n[STEP 2/3] Building and pushing Docker image...")

        image_uri_path = config.get_sagemaker_deployment_config.get(
            "docker_image_uri_path", "docker_image_uri.json"
        )

        if rebuild_image:
            logger.info("\n[STEP 2/3] Building and pushing Docker image...")
            builder = DockerBuilder(
                region=config.get_sagemaker_deployment_config.get(
                    "region", "us-east-1"
                ),
                image_name=config.get_sagemaker_deployment_config.get(
                    "image_name", "volatility-predictor"
                ),
            )
            image_uri = builder.build_and_push()
            with open(image_uri_path, "w") as f:
                json.dump({"image_uri": image_uri}, f, indent=2)
            logger.info(f"Image built and pushed: {image_uri}")
        else:
            logger.info("\n[STEP 2/3] Skipping Docker build, reusing existing image...")
            with open(image_uri_path, "r") as f:
                image_uri = json.load(f)["image_uri"]
            logger.info(f"Reusing image: {image_uri}")

        # ===== STEP 3: Deploy to SageMaker =====
        logger.info("\n[STEP 3/3] Deploying to SageMaker endpoint...")

        # Get IAM role from environment
        role_arn = os.getenv("SAGEMAKER_ROLE_ARN", "")
        if not role_arn:
            raise ValueError(
                "SAGEMAKER_ROLE_ARN not found in environment variables. "
                "Please set it in your .env file."
            )

        deployer = ContainerDeployer(
            role_arn=role_arn,
            image_uri=image_uri,
            region=config.get_sagemaker_deployment_config.get("region", "us-east-1"),
        )

        deployment_info = deployer.full_deployment(model_tar_path)

        # Get deployment info path from config
        deployment_info_path = config.get_sagemaker_deployment_config.get(
            "deployment_info_path", "deployment_info.json"
        )

        # Save deployment info
        with open(deployment_info_path, "w") as f:
            json.dump(deployment_info, f, indent=2)

        logger.info(f"Deployment info saved to: {deployment_info_path}")

        # ===== SUCCESS =====
        logger.info("Deployment pipeline completed successfully!")
        logger.info(f"Endpoint Name: {deployment_info['endpoint_name']}")
        logger.info(f"Model Name: {deployment_info['model_name']}")
        logger.info(f"Region: {deployment_info['region']}")
        logger.info(f"Image URI: {deployment_info['image_uri']}")

        return deployment_info

    except Exception as e:
        logger.error("Deployment failed!")
        logger.error(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    deploy_to_sagemaker(rebuild_image=False)
