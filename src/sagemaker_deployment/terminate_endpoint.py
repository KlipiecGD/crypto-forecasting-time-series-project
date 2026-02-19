import json
import os
from dotenv import load_dotenv

load_dotenv()

from src.sagemaker_deployment.deploy_container import ContainerDeployer
from src.config.config import config
from src.logging_utils.loggers import deployment_logger as logger


def terminate_sagemaker_endpoint():
    deployment_info_path = config.get_sagemaker_deployment_config.get(
        "deployment_info_path", "deployment_info.json"
    )
    with open(deployment_info_path, "r") as f:
        info = json.load(f)

    endpoint_name = info["endpoint_name"]
    region = info["region"]
    role_arn = os.getenv("SAGEMAKER_ROLE_ARN", "")

    logger.info(f"Deleting endpoint: {endpoint_name} in {region}")

    deployer = ContainerDeployer(
        role_arn=role_arn,
        image_uri=info["image_uri"],
        region=region,
    )
    deployer.delete_endpoint(endpoint_name)

    logger.info("Done!")


if __name__ == "__main__":
    terminate_sagemaker_endpoint()
