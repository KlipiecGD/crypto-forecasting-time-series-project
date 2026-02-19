import boto3
import subprocess
import json
from typing import Optional

from src.logging_utils.loggers import deployment_logger as logger
from src.config.config import config


class DockerBuilder:
    """Build and push Docker image to AWS ECR."""

    def __init__(
        self, region: Optional[str] = None, image_name: Optional[str] = None
    ) -> None:
        """
        Initialize Docker builder.

        Args:
            region (Optional[str]): AWS region
            image_name (Optional[str]): Name for Docker image
        """
        self.region = region or config.get_sagemaker_deployment_config.get(
            "region", "us-east-1"
        )
        self.image_name = image_name or config.get_sagemaker_deployment_config.get(
            "image_name", "volatility-predictor"
        )

        # Get AWS account ID
        sts = boto3.client("sts", region_name=region)
        self.account_id = sts.get_caller_identity()["Account"]

        # ECR repository name
        self.repository_name = image_name
        self.ecr_uri = (
            f"{self.account_id}.dkr.ecr.{region}.amazonaws.com/{self.repository_name}"
        )

        logger.info(f"AWS Account ID: {self.account_id}")
        logger.info(f"ECR Repository: {self.ecr_uri}")

    def create_ecr_repository(self) -> str:
        """
        Create ECR repository if it doesn't exist.
        Returns the repository URI or raises an error message if creation/access fails.
        """
        ecr = boto3.client("ecr", region_name=self.region)

        try:
            response = ecr.create_repository(repositoryName=self.repository_name)
            logger.info(f"Created ECR repository: {self.repository_name}")
            return response["repository"]["repositoryUri"]
        except ecr.exceptions.RepositoryAlreadyExistsException:
            logger.info(f"ECR repository already exists: {self.repository_name}")
            response = ecr.describe_repositories(repositoryNames=[self.repository_name])
            return response["repositories"][0]["repositoryUri"]
        except Exception as e:
            raise Exception(f"Failed to create or access ECR repository: {str(e)}")

    def docker_login(self):
        """Login to ECR."""
        logger.info("\nLogging into ECR...")
        # Get ECR login password and login
        cmd = f"aws ecr get-login-password --region {self.region} | docker login --username AWS --password-stdin {self.account_id}.dkr.ecr.{self.region}.amazonaws.com"

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("Docker login successful")
        else:
            raise Exception(f"Docker login failed: {result.stderr}")

    def build_image(self, dockerfile_dir: Optional[str] = None) -> str:
        """
        Build Docker image.

        Args:
            dockerfile_dir (Optional[str]): Directory containing Dockerfile

        Returns:
            str: The Docker image tag
        """
        logger.info(f"\nBuilding Docker image...")

        dockerfile_dir = dockerfile_dir or config.get_sagemaker_deployment_config.get(
            "dockerfile_dir", "src/sagemaker_deployment"
        )

        logger.info(f"Dockerfile directory: {dockerfile_dir}")

        tag = f"{self.ecr_uri}:latest"

        cmd = f"docker build --platform=linux/amd64 --provenance=false -t {self.image_name} {dockerfile_dir}"
        logger.info(f"Command: {cmd}")

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("Docker image built successfully")

            # Tag for ECR
            tag_cmd = f"docker tag {self.image_name}:latest {tag}"
            subprocess.run(tag_cmd, shell=True, check=True)
            logger.info(f"Tagged image: {tag}")

            return tag
        else:
            raise Exception(f"Docker build failed: {result.stderr}")

    def push_image(self, tag: str) -> str:
        """
        Push Docker image to ECR.

        Args:
            tag (str): Docker image tag

        Returns:
            str: The pushed Docker image tag
        """
        logger.info(f"\nPushing image to ECR...")

        cmd = f"docker push {tag}"

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info(f"Image pushed successfully: {tag}")
            return tag
        else:
            raise Exception(f"Docker push failed: {result.stderr}")

    def build_and_push(self) -> str:
        """
        Complete build and push pipeline.
        Returns:
            str: The final Docker image tag that was pushed
        """
        logger.info("Starting Docker build and push process...")

        # 1. Create ECR repository
        self.create_ecr_repository()

        # 2. Docker login
        self.docker_login()

        # 3. Build image
        tag = self.build_image()

        # 4. Push image
        self.push_image(tag)

        logger.info("Docker build and push process completed successfully")
        logger.info(f"Image URI: {tag}")

        return tag


def main():
    """Build and push Docker image."""
    builder = DockerBuilder(
        region=config.get_sagemaker_deployment_config.get("region", "us-east-1"),
        image_name=config.get_sagemaker_deployment_config.get(
            "image_name", "volatility-predictor"
        ),
    )

    image_uri = builder.build_and_push()

    # Save image URI for deployment
    save_path = config.get_sagemaker_deployment_config.get(
        "docker_image_uri_path", "docker_image_uri.json"
    )
    with open(save_path, "w") as f:
        json.dump({"image_uri": image_uri}, f, indent=2)

    logger.info(f"\nImage URI saved to: {save_path}")

    return image_uri


if __name__ == "__main__":
    main()
