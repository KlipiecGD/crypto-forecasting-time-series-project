import os
import shutil
import tarfile
from pathlib import Path
from typing import Optional

from src.config.config import config
from src.logging_utils.loggers import deployment_logger as logger


def package_model_for_docker(
    model_path: Optional[str] = None, output_path: Optional[str] = None
) -> str:
    """
    Package the model file for Docker deployment.

    Args:
        model_path (Optional[str]): Path to trained model (.joblib file)
        output_path (Optional[str]): Output tarball path
    Returns:
        str: Path to the created tarball containing the model file
    """
    logger.info("Packaging model for Docker deployment...")

    model_path = model_path or config.get_model_config.get(
        "model_save_path", "models/volatility_model.joblib"
    )
    output_path = output_path or config.get_sagemaker_deployment_config.get(
        "model_output_path", "sagemaker_model.tar.gz"
    )

    # Create temporary directory
    package_dir = Path("model_package")
    package_dir.mkdir(exist_ok=True)

    try:
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Copy the model file
        shutil.copy(model_path, package_dir / "volatility_model.joblib")
        logger.info(f"Copied model: {model_path}")

        # Create tarball
        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(
                package_dir / "volatility_model.joblib",
                arcname="volatility_model.joblib",
            )

        file_size_kb = os.path.getsize(output_path) / 1024

        logger.info(f"Model package created successfully")
        logger.info(f"Output: {output_path}")
        logger.info(f"Size: {file_size_kb:.2f} KB")

    finally:
        # Cleanup
        if package_dir.exists():
            shutil.rmtree(package_dir)

    return output_path


if __name__ == "__main__":
    logger.info("Starting model packaging process...")
    package_model_for_docker()
