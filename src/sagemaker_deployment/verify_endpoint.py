import boto3
import json
import pandas as pd
from typing import Optional
from src.inference.data_loader import load_data_for_inference
from src.config.config import config
from src.logging_utils.loggers import inference_logger as logger


class SageMakerPredictor:
    """Make predictions using deployed SageMaker endpoint."""

    def __init__(self, endpoint_name: str, region: str) -> None:
        """
        Initialize predictor.

        Args:
            endpoint_name (str): Name of deployed endpoint
            region (str): AWS region
        """
        self.region = region
        self.endpoint_name = endpoint_name
        self.runtime = boto3.client("sagemaker-runtime", region_name=self.region)
        logger.info(f"Predictor initialized for endpoint: {endpoint_name}")

    def predict(self, input_data: dict | pd.DataFrame) -> dict:
        """
        Make prediction on input data.

        Args:
            input_data (dict | pd.DataFrame): dict or DataFrame with OHLC data

        Returns:
            dict: Prediction results
        """
        # Convert DataFrame to dict if needed
        if isinstance(input_data, pd.DataFrame):
            payload = input_data.to_dict(orient="records")
        else:
            payload = input_data

        # Invoke endpoint
        response = self.runtime.invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload),
        )

        # Parse response
        result = json.loads(response["Body"].read().decode())

        return result

    def predict_from_live_data(self, forecast_horizon: int) -> dict:
        """
        Fetch live data and make prediction.

        Args:
            forecast_horizon (int): Number of days ahead to predict volatility for (1-30

        Returns:
            dict: Prediction results
        """
        # Fetching and preparing data for inference
        logger.info("Fetching and preparing data for inference...")
        features = load_data_for_inference(forecast_horizon)
        if features is None:
            raise ValueError("Failed to load data for inference. Aborting prediction.")

        # Predict
        logger.info("Making prediction...")
        result = self.predict(features)

        return result[0]  # Return first (and only) prediction


def test_endpoint(
    forecast_horizon: int,
    endpoint_name: str,
    region: Optional[str] = None,
) -> dict:
    """
    Test deployed SageMaker endpoint.

    Args:
        forecast_horizon (int): Number of days ahead to predict volatility for (1-30)
        endpoint_name (str): Name of endpoint to test
        region (Optional[str]): AWS region
    Returns:
        dict: Prediction results
    """
    region = region or config.get_sagemaker_deployment_config.get("region", "us-east-1")

    logger.info(
        f"Testing SageMaker Endpoint named '{endpoint_name}' in region '{region}'"
    )

    predictor = SageMakerPredictor(endpoint_name, region)

    # Test with live data
    try:
        result = predictor.predict_from_live_data(forecast_horizon)

        logger.info("\nPREDICTION SUCCESSFUL")
        logger.info(f"Predicted Class: {result['predicted_class']}")
        logger.info(f"Confidence: {result['confidence']:.2%}")
        logger.info(f"\nProbabilities:")
        for class_name, prob in result["probabilities"].items():
            logger.info(f"  {class_name}: {prob:.2%}")

        return result

    except Exception as e:
        logger.info(f"\nPREDICTION FAILED")
        logger.info(f"Error: {e}")
        raise


if __name__ == "__main__":
    # Load deployment info
    deployment_info_path = config.get_sagemaker_deployment_config.get(
        "deployment_info_path", "deployment_info.json"
    )
    with open(deployment_info_path, "r") as f:
        info = json.load(f)

    test_endpoint(
        forecast_horizon=config.get_forecast_config.get("default_horizon", 30),
        endpoint_name=info["endpoint_name"],
        region=info["region"],
    )
