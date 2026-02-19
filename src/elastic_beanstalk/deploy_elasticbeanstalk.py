import time
import json
import os
import subprocess
import zipfile
import boto3
from datetime import datetime
from typing import Any
from dotenv import load_dotenv

load_dotenv()

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
EB_SOLUTION_STACK = config.get_elastic_beanstalk_deployment_config.get(
    "solution_stack", "64bit Amazon Linux 2023 v4.9.3 running Docker"
)
EB_INSTANCE_PROFILE = config.get_elastic_beanstalk_deployment_config.get(
    "instance_profile", "testEC2Role"
)
EB_INSTANCE_TYPE = config.get_elastic_beanstalk_deployment_config.get(
    "instance_type", "t3.small"
)
EB_S3_BUCKET = os.getenv("EB_S3_BUCKET", "")
EB_S3_PREFIX = config.get_elastic_beanstalk_deployment_config.get(
    "s3_prefix", "volatility-predictor"
)
AWS_DEFAULT_REGION = config.get_elastic_beanstalk_deployment_config.get(
    "aws_default_region", "us-east-1"
)
CLOUDWATCH_REGION = config.get_elastic_beanstalk_deployment_config.get(
    "cloudwatch_region", "eu-north-1"
)
ENABLE_CLOUD_LOGGING = config.get_elastic_beanstalk_deployment_config.get(
    "enable_cloud_logging", "true"
)


def get_endpoint_name() -> str:
    """
    Load SageMaker endpoint name from deployment_info.json.
    Returns:
        str: The name of the SageMaker endpoint to connect to.
    """
    # Get the path to deployment_info.json from config, default to "deployment_info.json"
    deployment_info_path = config.get_sagemaker_deployment_config.get(
        "deployment_info_path", "deployment_info.json"
    )
    with open(deployment_info_path, "r") as f:
        info = json.load(f)
    endpoint_name = info["endpoint_name"]
    logger.info(f"SageMaker endpoint: {endpoint_name}")
    return endpoint_name


def build_docker_image(image_name: str) -> None:
    """
    Build Docker image for the Streamlit app.
    Args:
        image_name (str): The name to tag the Docker image with.
    """
    logger.info("Building Docker image...")

    cmd = (
        f"docker build --platform=linux/amd64 --provenance=false "
        f"-f Dockerfile.streamlit -t {image_name} ."
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Docker build failed:\n{result.stderr}")
    logger.info(f"Docker image built: {image_name}")


def push_to_ecr(image_name: str, region: str) -> str:
    """
    Build and push image to ECR, return full image URI.
    Args:
        image_name (str): The name of the local Docker image to push.
        region (str): AWS region where ECR repository is located.
    Returns:
        str: The full URI of the pushed image in ECR.
    """
    sts = boto3.client("sts", region_name=region)
    account_id = sts.get_caller_identity()["Account"]
    ecr = boto3.client("ecr", region_name=region)
    repo_name = image_name

    # Create ECR repo if needed
    try:
        response = ecr.create_repository(repositoryName=repo_name)
        repo_uri = response["repository"]["repositoryUri"]
        logger.info(f"Created ECR repository: {repo_uri}")
    except ecr.exceptions.RepositoryAlreadyExistsException:
        response = ecr.describe_repositories(repositoryNames=[repo_name])
        repo_uri = response["repositories"][0]["repositoryUri"]
        logger.info(f"ECR repository exists: {repo_uri}")
    except Exception as e:
        raise Exception(f"ECR repository error: {e}")

    tag = f"{repo_uri}:latest"

    # Docker login
    login_cmd = (
        f"aws ecr get-login-password --region {region} | "
        f"docker login --username AWS --password-stdin "
        f"{account_id}.dkr.ecr.{region}.amazonaws.com"
    )
    result = subprocess.run(login_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"ECR login failed:\n{result.stderr}")
    logger.info("ECR login successful")

    # Tag and push
    subprocess.run(f"docker tag {image_name}:latest {tag}", shell=True, check=True)
    result = subprocess.run(
        f"docker push {tag}", shell=True, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise Exception(f"Docker push failed:\n{result.stderr}")
    logger.info(f"Image pushed: {tag}")

    return tag


def create_app_bundle(image_uri: str, endpoint_name: str) -> str:
    """
    Create a zip bundle containing Dockerrun.aws.json.
    This is what Elastic Beanstalk deploys.

    Args:
        image_uri (str): The URI of the Docker image in ECR.
        endpoint_name (str): The name of the SageMaker endpoint to connect to.
    Returns:
        str: The file path to the created zip bundle.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    bundle_name = f"volatility-predictor-{timestamp}.zip"

    # Define the Dockerrun.aws.json content with environment variables for the SageMaker endpoint and logging
    dockerrun = {
        "AWSEBDockerrunVersion": "1",
        "Image": {"Name": image_uri, "Update": "true"},
        "Ports": [{"ContainerPort": "8080", "HostPort": "80"}],
        "Environment": [
            {"Name": "SAGEMAKER_ENDPOINT_NAME", "Value": endpoint_name},
            {"Name": "AWS_DEFAULT_REGION", "Value": AWS_DEFAULT_REGION},
            {"Name": "CLOUDWATCH_REGION", "Value": CLOUDWATCH_REGION},
            {
                "Name": "ENABLE_CLOUD_LOGGING",
                "Value": str(ENABLE_CLOUD_LOGGING).lower(),
            },
        ],
        "Logging": "/var/log/streamlit",
    }

    # Create the zip file
    with zipfile.ZipFile(bundle_name, "w") as zf:
        # Add Dockerrun.aws.json
        zf.writestr("Dockerrun.aws.json", json.dumps(dockerrun, indent=2))

    logger.info(f"App bundle created: {bundle_name}")
    return bundle_name


def upload_bundle_to_s3(bundle_path: str) -> str:
    """
    Upload the app bundle to S3 and return the S3 key.

    Args:
        bundle_path (str): The local file path to the app bundle zip file.
    Returns:
        str: The S3 key where the bundle was uploaded.
    """
    s3 = boto3.client("s3", region_name=EB_REGION)
    s3_key = f"{EB_S3_PREFIX}/{os.path.basename(bundle_path)}"

    s3.upload_file(bundle_path, EB_S3_BUCKET, s3_key)
    logger.info(f"Bundle uploaded to s3://{EB_S3_BUCKET}/{s3_key}")

    return s3_key


def ensure_eb_application(eb_client: Any) -> None:
    """
    Create EB application if it doesn't exist.
    Args:
        eb_client: The boto3 Elastic Beanstalk client.
    """
    # Check if application exists
    apps = eb_client.describe_applications(ApplicationNames=[EB_APP_NAME])
    # If no applications are returned, it means the application doesn't exist
    if not apps["Applications"]:
        eb_client.create_application(
            ApplicationName=EB_APP_NAME,
            Description="Bitcoin Volatility Predictor Streamlit App",
        )
        logger.info(f"Created EB application: {EB_APP_NAME}")
    else:
        logger.info(f"EB application already exists: {EB_APP_NAME}")


def create_app_version(eb_client: Any, s3_key: str) -> str:
    """
    Create a new EB application version and wait for processing.
    Args:
        eb_client: The boto3 Elastic Beanstalk client.
        s3_key (str): The S3 key where the app bundle is located.
    Returns:
        str: The version label of the created application version.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    version_label = f"v-{timestamp}"

    eb_client.create_application_version(
        ApplicationName=EB_APP_NAME,
        VersionLabel=version_label,
        SourceBundle={"S3Bucket": EB_S3_BUCKET, "S3Key": s3_key},
        AutoCreateApplication=False,
        Process=True,
    )
    logger.info(f"Created app version: {version_label}")

    # Wait for version to finish processing, poll every 10 seconds
    logger.info("Waiting for app version to finish processing...")
    while True:
        time.sleep(10)
        response = eb_client.describe_application_versions(
            ApplicationName=EB_APP_NAME, VersionLabels=[version_label]
        )
        status = response["ApplicationVersions"][0]["Status"]
        logger.info(f"Version status: {status}")

        if status == "PROCESSED":
            logger.info("App version ready")
            break
        elif status == "FAILED":
            raise Exception(f"App version processing failed")

    return version_label


def deploy_or_update_environment(
    eb_client: Any, version_label: str, endpoint_name: str
) -> str:

    option_settings = [
        {
            "Namespace": "aws:autoscaling:launchconfiguration",
            "OptionName": "IamInstanceProfile",
            "Value": EB_INSTANCE_PROFILE,
        },
        {
            "Namespace": "aws:autoscaling:launchconfiguration",
            "OptionName": "InstanceType",
            "Value": EB_INSTANCE_TYPE,
        },
        {
            "Namespace": "aws:elasticbeanstalk:environment",
            "OptionName": "EnvironmentType",
            "Value": "SingleInstance",
        },
        {
            "Namespace": "aws:elasticbeanstalk:application:environment",
            "OptionName": "SAGEMAKER_ENDPOINT_NAME",
            "Value": endpoint_name,
        },
        {
            "Namespace": "aws:elasticbeanstalk:application:environment",
            "OptionName": "AWS_DEFAULT_REGION",
            "Value": AWS_DEFAULT_REGION,
        },
        {
            "Namespace": "aws:elasticbeanstalk:application:environment",
            "OptionName": "CLOUDWATCH_REGION",
            "Value": CLOUDWATCH_REGION,
        },
        {
            "Namespace": "aws:elasticbeanstalk:application:environment",
            "OptionName": "ENABLE_CLOUD_LOGGING",
            "Value": str(ENABLE_CLOUD_LOGGING).lower(),
        },
    ]

    envs = eb_client.describe_environments(
        ApplicationName=EB_APP_NAME,
        EnvironmentNames=[EB_ENV_NAME],
        IncludeDeleted=False,
    )
    active_envs = [
        e
        for e in envs["Environments"]
        if e["Status"] not in ("Terminated", "Terminating")
    ]

    if active_envs:
        logger.info(f"Updating existing environment: {EB_ENV_NAME}")
        eb_client.update_environment(
            ApplicationName=EB_APP_NAME,
            EnvironmentName=EB_ENV_NAME,
            VersionLabel=version_label,
            OptionSettings=option_settings,
        )
    else:
        logger.info(f"Creating new environment: {EB_ENV_NAME}")
        eb_client.create_environment(
            ApplicationName=EB_APP_NAME,
            EnvironmentName=EB_ENV_NAME,
            VersionLabel=version_label,
            SolutionStackName=EB_SOLUTION_STACK,
            OptionSettings=option_settings,
        )

    return EB_ENV_NAME


def wait_for_environment(eb_client: Any) -> str:
    """
    Poll until environment is Ready.
    Args:
        eb_client: The boto3 Elastic Beanstalk client.
    Returns:
        str: The URL of the deployed environment once it's ready.
    """
    logger.info("Waiting for environment to be ready, it may take a few minutes...")

    # Poll every 30 seconds until environment is Ready or fails
    while True:
        time.sleep(30)
        envs = eb_client.describe_environments(
            ApplicationName=EB_APP_NAME,
            EnvironmentNames=[EB_ENV_NAME],
            IncludeDeleted=False,
        )
        if not envs["Environments"]:
            continue

        env = envs["Environments"][0]
        status = env["Status"]
        health = env.get("Health", "Unknown")
        logger.info(f"Status: {status} | Health: {health}")

        if status == "Ready":
            url = f"http://{env['CNAME']}"
            logger.info(f"Environment is Ready!")
            logger.info(f"URL: {url}")
            return url
        elif status in ("Terminated", "Terminating"):
            raise Exception(f"Environment failed with status: {status}")


def deploy_streamlit_app():
    """
    Main deployment function. Executes all steps to deploy the Streamlit app to Elastic Beanstalk.
    Steps:
        1. Build Docker image
        2. Push to ECR
        3. Create app bundle
        4. Upload to S3
        5. Create EB app version
        6. Deploy environment
        7. Wait for environment to be ready
    """
    logger.info("Starting Elastic Beanstalk deployment...")

    image_name = config.get_elastic_beanstalk_deployment_config.get(
        "image_name", "volatility-predictor-streamlit"
    )
    endpoint_name = get_endpoint_name()
    eb_client = boto3.client("elasticbeanstalk", region_name=EB_REGION)

    # Step 1: Build Docker image
    logger.info("\n[STEP 1/7] Building Docker image...")
    build_docker_image(image_name)

    # Step 2: Push to ECR
    logger.info("\n[STEP 2/7] Pushing to ECR...")
    image_uri = push_to_ecr(image_name, EB_REGION)

    # Step 3: Create app bundle
    logger.info("\n[STEP 3/7] Creating app bundle...")
    bundle_path = create_app_bundle(image_uri, endpoint_name)

    # Step 4: Upload to S3
    logger.info("\n[STEP 4/7] Uploading bundle to S3...")
    s3_key = upload_bundle_to_s3(bundle_path)
    os.remove(bundle_path)  # Clean up local zip

    # Step 5: Create EB app + version
    logger.info("\n[STEP 5/7] Creating EB application version...")
    ensure_eb_application(eb_client)
    version_label = create_app_version(eb_client, s3_key)

    # Step 6: Deploy environment
    logger.info("\n[STEP 6/7] Deploying environment...")
    deploy_or_update_environment(eb_client, version_label, endpoint_name)

    # Step 7: Wait for environment to be ready and get URL
    logger.info("\n[STEP 7/7] Waiting for environment to be ready...")
    url = wait_for_environment(eb_client)

    # Save result to JSON file
    result = {
        "app_name": EB_APP_NAME,
        "env_name": EB_ENV_NAME,
        "url": url,
        "image_uri": image_uri,
        "sagemaker_endpoint": endpoint_name,
        "region": EB_REGION,
    }
    deployment_info_path = config.get_elastic_beanstalk_deployment_config.get(
        "deployment_info_path", "elasticbeanstalk_info.json"
    )

    with open(deployment_info_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("\nDeployment complete!")
    logger.info(f"App URL: {url}")
    logger.info(f"Info saved to: {deployment_info_path}")

    return result


if __name__ == "__main__":
    deploy_streamlit_app()
