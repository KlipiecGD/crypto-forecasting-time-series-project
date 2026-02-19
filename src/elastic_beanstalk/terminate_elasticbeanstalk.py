import boto3
import time

from src.config.config import config
from src.logging_utils.loggers import deployment_logger as logger

# Configuration
EB_REGION = config.get_elastic_beanstalk_deployment_config.get("region", "eu-north-1")
EB_APP_NAME = config.get_elastic_beanstalk_deployment_config.get(
    "app_name", "volatility-predictor-app"
)
EB_ENV_NAME = config.get_elastic_beanstalk_deployment_config.get(
    "env_name", "volatility-predictor-app-env"
)


def terminate_eb_environment() -> None:
    """Terminate the Elastic Beanstalk environment if it exists and is active. Waits until the environment is fully terminated before returning."""
    eb_client = boto3.client("elasticbeanstalk", region_name=EB_REGION)

    # Check if the environment exists and is active
    envs = eb_client.describe_environments(
        ApplicationName=EB_APP_NAME,
        EnvironmentNames=[EB_ENV_NAME],
        IncludeDeleted=False,
    )

    # Filter out environments that are not active (i.e., those that are terminated or terminating)
    active_envs = [
        e
        for e in envs["Environments"]
        if e["Status"] not in ("Terminated", "Terminating")
    ]

    # If there are no active environments, log and return
    if not active_envs:
        logger.info("No active environment found, nothing to delete.")
        return

    # If there is an active environment, terminate it
    logger.info(f"Terminating environment: {EB_ENV_NAME}")
    eb_client.terminate_environment(EnvironmentName=EB_ENV_NAME)

    logger.info("Waiting for environment to terminate...")

    # Poll the environment status every 30 seconds until it is terminated
    while True:
        time.sleep(30)
        envs = eb_client.describe_environments(
            ApplicationName=EB_APP_NAME,
            EnvironmentNames=[EB_ENV_NAME],
            IncludeDeleted=True,
        )
        status = envs["Environments"][0]["Status"]
        logger.info(f"Status: {status}")
        if status == "Terminated":
            logger.info("Environment terminated.")
            break


if __name__ == "__main__":
    terminate_eb_environment()
