import os
import boto3
import json
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from src.logging_utils.loggers import deployment_logger as logger
from src.config.config import config


class ContainerDeployer:
    """Deploy model using Docker container."""

    def __init__(self, role_arn: str, image_uri: str, region: str) -> None:
        """
        Initialize deployer.

        Args:
            role_arn (str): IAM role ARN
            image_uri (str): ECR image URI
            region (str): AWS region
        """
        self.role_arn = role_arn
        self.image_uri = image_uri
        self.region = region

        self.boto_session = boto3.Session(region_name=region)
        self.s3_client = self.boto_session.client("s3")
        self.sagemaker_client = self.boto_session.client("sagemaker")

        # Get default S3 bucket for SageMaker
        sts_client = boto3.client("sts")
        account_id = sts_client.get_caller_identity()["Account"]
        self.s3_bucket = f"sagemaker-{region}-{account_id}"

        # Create bucket if it doesn't exist
        self._ensure_bucket_exists()

        logger.info("Deployer initialized")
        logger.info(f"Image: {image_uri}")
        logger.info(f"Region: {region}")
        logger.info(f"S3 Bucket: {self.s3_bucket}")

    def _ensure_bucket_exists(self) -> None:
        """Create S3 bucket if it doesn't exist."""
        try:
            self.s3_client.head_bucket(Bucket=self.s3_bucket)
            logger.info(f"S3 bucket exists: {self.s3_bucket}")
        except:
            logger.info(f"Creating S3 bucket: {self.s3_bucket}")
            try:
                if self.region == "us-east-1":
                    # us-east-1 doesn't need LocationConstraint
                    self.s3_client.create_bucket(Bucket=self.s3_bucket)
                else:
                    self.s3_client.create_bucket(
                        Bucket=self.s3_bucket,
                        CreateBucketConfiguration={"LocationConstraint": self.region},
                    )
                logger.info(f"S3 bucket created: {self.s3_bucket}")
            except Exception as e:
                logger.info(f"Warning: Could not create bucket: {e}")
                logger.info(
                    f"You may need to create it manually: aws s3 mb s3://{self.s3_bucket} --region {self.region}"
                )
                raise

    def upload_model_to_s3(
        self, model_tar_path: str, s3_prefix: Optional[str] = None
    ) -> str:
        """
        Upload model tarball to S3.
        Args:
            model_tar_path (str): Local path to model tarball
            s3_prefix (Optional[str]): S3 key prefix for organizing model uploads
        Returns:
            str: S3 URI of the uploaded model
        """
        s3_prefix = s3_prefix or config.get_sagemaker_deployment_config.get(
            "s3_prefix", "volatility-model"
        )

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        s3_key = f"{s3_prefix}/{timestamp}/model.tar.gz"

        logger.info(f"\nUploading model to S3...")
        self.s3_client.upload_file(model_tar_path, self.s3_bucket, s3_key)

        s3_uri = f"s3://{self.s3_bucket}/{s3_key}"
        logger.info(f"Model uploaded: {s3_uri}")

        return s3_uri

    def create_model(self, model_s3_uri: str, model_name: Optional[str] = None) -> str:
        """
        Create SageMaker model.

        Args:
            model_s3_uri (str): S3 URI of the model
            model_name (Optional[str]): Name of the model

        Returns:
            str: Name of created model
        """
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_name = f"volatility-custom-{timestamp}"

        logger.info(f"\nCreating SageMaker model: {model_name}")

        # Use boto3 client directly to create model
        self.sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer={"Image": self.image_uri, "ModelDataUrl": model_s3_uri},
            ExecutionRoleArn=self.role_arn,
        )

        logger.info(f"Model created: {model_name}")

        return model_name

    def delete_endpoint(self, endpoint_name: str) -> None:
        """
        Delete a SageMaker endpoint and its configuration.

        Args:
            endpoint_name (str): Name of the endpoint to delete
        """
        # Delete endpoint
        try:
            self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            logger.info(f"Deleted endpoint: {endpoint_name}")

            # Wait for deletion to complete
            waiter = self.sagemaker_client.get_waiter("endpoint_deleted")
            waiter.wait(EndpointName=endpoint_name)
            logger.info(f"Endpoint fully deleted: {endpoint_name}")
        except self.sagemaker_client.exceptions.ClientError as e:
            if "Could not find endpoint" in str(e):
                logger.info(f"Endpoint not found (already deleted?): {endpoint_name}")
            else:
                raise

        # Delete its endpoint config too (avoids orphaned configs piling up)
        endpoint_config_name = f"{endpoint_name}-config"
        try:
            self.sagemaker_client.delete_endpoint_config(
                EndpointConfigName=endpoint_config_name
            )
            logger.info(f"Deleted endpoint config: {endpoint_config_name}")
        except self.sagemaker_client.exceptions.ClientError as e:
            if "Could not find endpoint configuration" in str(e):
                logger.info(f"Endpoint config not found: {endpoint_config_name}")
            else:
                raise

    def deploy_endpoint(
        self,
        model_name: str,
        endpoint_name: Optional[str] = None,
        instance_type: Optional[str] = None,
        instance_count: Optional[int] = None,
    ):
        """
        Deploy model to endpoint.

        Args:
            model_name (str): Name of the SageMaker model to deploy
            endpoint_name (Optional[str]): Optional name for the endpoint (will be auto-generated if not provided)
            instance_type (str): EC2 instance type for deployment (default: ml.t2.medium)
            instance_count (int): Number of instances to deploy (default: 1)

        Returns:
            endpoint_name: Name of deployed endpoint
        """
        instance_type = instance_type or config.get_sagemaker_deployment_config.get(
            "instance_type", "ml.t2.medium"
        )
        instance_count = instance_count or config.get_sagemaker_deployment_config.get(
            "instance_count", 1
        )
        endpoint_name = endpoint_name or config.get_sagemaker_deployment_config.get(
            "endpoint_name", "volatility-predictor"
        )

        endpoint_config_name = (
            f"{endpoint_name}-config-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )

        # Always create a new endpoint config (required even for updates)
        logger.info(f"Creating endpoint configuration: {endpoint_config_name}")
        logger.info(f"Instance: {instance_type} x {instance_count}")
        self.sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "InitialInstanceCount": instance_count,
                    "InstanceType": instance_type,
                }
            ],
        )
        logger.info(f"Endpoint configuration created: {endpoint_config_name}")

        # Check if endpoint already exists
        try:
            self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            endpoint_exists = True
        except self.sagemaker_client.exceptions.ClientError:
            endpoint_exists = False

        if endpoint_exists:
            # Update existing endpoint - zero downtime
            logger.info(f"Updating existing endpoint: {endpoint_name}")
            self.sagemaker_client.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name,
            )
        else:
            # Create fresh endpoint
            logger.info(f"Creating new endpoint: {endpoint_name}")
            self.sagemaker_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name,
            )

        logger.info(
            "Waiting for endpoint to be in service (this may take 5-10 minutes)..."
        )

        # Wait for endpoint to be in service
        waiter = self.sagemaker_client.get_waiter("endpoint_in_service")
        waiter.wait(EndpointName=endpoint_name)

        logger.info(f"Endpoint deployed and ready: {endpoint_name}")

        return endpoint_name

    def _cleanup_old_endpoint_configs(self, endpoint_name: str) -> None:
        """
        Delete old endpoint configs for this endpoint, keeping only the latest.
        Args:
            endpoint_name (str): Name of the endpoint to clean up configs for
        """
        try:
            configs = self.sagemaker_client.list_endpoint_configs(
                NameContains=endpoint_name,
                SortBy="CreationTime",
                SortOrder="Descending",
            )
            # Skip the first (most recent), delete the rest
            for config in configs["EndpointConfigs"][1:]:
                self.sagemaker_client.delete_endpoint_config(
                    EndpointConfigName=config["EndpointConfigName"]
                )
                logger.info(
                    f"Deleted old endpoint config: {config['EndpointConfigName']}"
                )
        except Exception as e:
            logger.warning(f"Could not clean up old endpoint configs: {e}")

    def full_deployment(self, model_tar_path: Optional[str] = None) -> dict:
        """
        Complete deployment pipeline.
        Args:
            model_tar_path (Optional[str]): Local path to the model tarball to deploy
        Returns:
            dict: Deployment details including endpoint name, model name, S3 URI, image URI, and region
        """
        logger.info("Deploying model using Docker container...")

        model_tar_path = model_tar_path or config.get_sagemaker_deployment_config.get(
            "model_output_path", "sagemaker_model.tar.gz"
        )

        # Upload model
        model_s3_uri = self.upload_model_to_s3(model_tar_path)

        # Create model
        model_name = self.create_model(model_s3_uri)

        # Deploy
        endpoint_name = self.deploy_endpoint(model_name)
        # Cleanup old endpoint configs to avoid clutter if endpoint already existed
        self._cleanup_old_endpoint_configs(endpoint_name)

        logger.info("Deployment complete")
        logger.info(f"Endpoint: {endpoint_name}")
        logger.info(f"Model: {model_name}")

        return {
            "endpoint_name": endpoint_name,
            "model_name": model_name,
            "model_s3_uri": model_s3_uri,
            "image_uri": self.image_uri,
            "region": self.region,
        }

    def invoke_endpoint(self, endpoint_name: str, payload: dict | str) -> dict:
        """
        Invoke endpoint.

        Args:
            endpoint_name (str): Name of the endpoint
            payload (dict | str): Input data (dict or JSON string)

        Returns:
            dict: Prediction result
        """
        runtime_client = self.boto_session.client("sagemaker-runtime")

        # Convert payload to JSON if needed
        if isinstance(payload, dict):
            payload = json.dumps(payload)

        # Invoke endpoint
        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name, ContentType="application/json", Body=payload
        )

        # Parse response
        result = json.loads(response["Body"].read().decode())

        return result


def deploy():
    """Main deployment function."""

    # Load image URI
    image_uri_path = config.get_sagemaker_deployment_config.get(
        "docker_image_uri_path", "docker_image_uri.json"
    )
    with open(image_uri_path, "r") as f:
        data = json.load(f)
        image_uri = data["image_uri"]

    # Configuration
    ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN", "")
    region = config.get_sagemaker_deployment_config.get("region", "us-east-1")
    model_tar_path = config.get_sagemaker_deployment_config.get(
        "model_output_path", "sagemaker_model.tar.gz"
    )

    # Deploy
    deployer = ContainerDeployer(role_arn=ROLE_ARN, image_uri=image_uri, region=region)

    deployment_info_path = config.get_sagemaker_deployment_config.get(
        "deployment_info_path", "deployment_info.json"
    )

    deployment_info = deployer.full_deployment(model_tar_path)

    # Save deployment info
    with open(deployment_info_path, "w") as f:
        json.dump(deployment_info, f, indent=2)

    logger.info(f"\nDeployment info saved to: {deployment_info_path}")

    return deployment_info


if __name__ == "__main__":
    deploy()
