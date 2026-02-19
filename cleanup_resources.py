from src.elastic_beanstalk.terminate_elasticbeanstalk import terminate_eb_environment
from src.sagemaker_deployment.terminate_endpoint import terminate_sagemaker_endpoint


def cleanup_resources() -> None:
    """
    Cleanup resources after deployment. This includes:
    1. Terminating the Elastic Beanstalk environment.
    2. Terminating the SageMaker endpoint.
    It is recommended to run this function after you are done with the deployed resources to avoid unnecessary costs.
    """
    terminate_eb_environment()
    terminate_sagemaker_endpoint()


if __name__ == "__main__":
    cleanup_resources()
