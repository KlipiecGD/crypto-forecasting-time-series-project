import json
import boto3
import os
import streamlit as st
from datetime import datetime, timedelta
from typing import Optional

from src.config.config import config
from src.logging_utils.loggers import ui_logger as logger
from src.inference.data_loader import load_data_for_inference
from src.ui.components import (
    render_page_config,
    get_volatility_ranges,
    render_sidebar,
    render_header,
    render_prediction_button,
    render_prediction_results,
    render_market_context,
    render_historical_volatility_chart,
    render_historical_patterns,
    render_model_information,
    render_volatility_explanation,
    render_forecast_selector,
    render_historical_performance_tab,
)
from src.ui.data_utils import (
    load_historical_data,
    fetch_current_market_data,
)


def load_aws_configuration() -> tuple[Optional[str], str]:
    """
    Load AWS SageMaker endpoint configuration.

    Returns:
        tuple containing:
            - endpoint_name (str): SageMaker endpoint name
            - aws_region (str): AWS region for the endpoint
    """
    try:
        deploy_info_path = config.get_sagemaker_deployment_config.get(
            "deployment_info_path", "deployment_info.json"
        )
        with open(deploy_info_path, "r") as f:
            deploy_info = json.load(f)
            endpoint_name = deploy_info["endpoint_name"]
            aws_region = deploy_info["region"]
        logger.info(f"Loaded SageMaker endpoint configuration from {deploy_info_path}")
    except FileNotFoundError:
        endpoint_name = os.getenv("SAGEMAKER_ENDPOINT_NAME")
        aws_region = config.get_sagemaker_deployment_config.get("region", "us-east-1")
        logger.info(
            "Loaded SageMaker endpoint configuration from environment variables."
        )

    if not endpoint_name:
        logger.error("SageMaker endpoint name is missing.")
        st.error(
            "⚠️ Endpoint configuration missing! Please ensure deployment_info.json "
            "exists or set SAGEMAKER_ENDPOINT_NAME environment variable."
        )

    return endpoint_name, aws_region


def make_prediction(
    forecast_horizon: int,
    endpoint_name: str,
    aws_region: str,
) -> tuple[Optional[str], Optional[dict], Optional[datetime]]:
    """
    Run inference using the AWS SageMaker Endpoint.

    Args:
        forecast_horizon (int): Number of days ahead for the forecast (1-30)
        endpoint_name (str): Name of the SageMaker endpoint
        aws_region (str): AWS region where the endpoint is deployed

    Returns:
        tuple containing:
            - predicted_class (str): Predicted volatility class
            - probabilities (dict): Confidence scores for each class
            - target_date (datetime): Date of the prediction
        Returns (None, None, None) if error occurs
    """
    # Fetch & Process Live Data
    try:
        features = load_data_for_inference(forecast_horizon=forecast_horizon)
        if features is None:
            logger.error("Data loading returned None — aborting prediction")
            st.error("⚠️ Failed to fetch or process live data for inference.")
            return None, None, None

    except Exception as e:
        logger.error(f"Error processing data: {e}")
        st.error(f"Error processing data: {e}")
        return None, None, None

    # Invoke SageMaker Endpoint
    try:
        client = boto3.client("sagemaker-runtime", region_name=aws_region)

        payload = features.to_dict(orient="records")

        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps(payload),
        )

        result = json.loads(response["Body"].read().decode())
        prediction = result[0]

        predicted_class = prediction["predicted_class"]
        probabilities = prediction["probabilities"]

        target_date = datetime.now() + timedelta(days=forecast_horizon)

        logger.info(
            f"Prediction successful — class: {predicted_class}, "
            f"confidence: {probabilities[predicted_class]:.2%}, "
            f"horizon: {forecast_horizon} days"
        )

        return predicted_class, probabilities, target_date

    except Exception as e:
        logger.error(f"SageMaker Inference Error: {e}")
        st.error(f"⚠️ SageMaker Inference Error: {e}")
        return None, None, None


def render_ui_components(volatility_ranges: dict) -> int:
    """
    Render all UI components and return selected forecast days.

    Args:
        volatility_ranges (dict): Volatility classification thresholds

    Returns:
        int: Selected forecast horizon in days
    """
    # Sidebar
    render_sidebar(volatility_ranges, deployment_type="aws")

    # Main content - Header
    render_header()

    # Forecast horizon selector
    forecast_days = render_forecast_selector()
    st.markdown("---")

    return forecast_days


def handle_prediction(forecast_days: int, endpoint_name: str, aws_region: str) -> None:
    """
    Handle prediction logic and display results.

    Args:
        forecast_days (int): Number of days ahead to predict
        endpoint_name (str): SageMaker endpoint name
        aws_region (str): AWS region
    """
    predict_button = render_prediction_button()

    if predict_button:
        logger.info(f"User clicked predict button — horizon: {forecast_days} days")
        with st.spinner("Fetching live data and analyzing market conditions..."):
            predicted_class, probabilities, target_date = make_prediction(
                forecast_days, endpoint_name, aws_region
            )

            if predicted_class is None or probabilities is None or target_date is None:
                logger.error(
                    "Prediction pipeline returned None — displaying error to user"
                )
                st.error(
                    "❌ Failed to make prediction. Please check your data connection and try again."
                )
            else:
                render_prediction_results(
                    predicted_class,
                    probabilities,
                    target_date,
                )


def render_context_and_history(volatility_ranges: dict) -> None:
    """
    Render market context, historical charts, and additional information.

    Args:
        volatility_ranges (dict): Volatility classification thresholds
    """
    # Market context
    market_context_data = fetch_current_market_data(limit=31)
    render_market_context(market_context_data)

    # Historical volatility chart
    historical_data = load_historical_data()
    render_historical_volatility_chart(historical_data, volatility_ranges)

    # Additional details
    st.markdown("---")
    st.subheader("📌 Additional Details")

    # Volatility explanation
    render_volatility_explanation()

    # Historical patterns
    render_historical_patterns(historical_data)

    # Model information
    render_model_information()


def main() -> None:
    """
    Main application entry point for AWS deployment.

    Orchestrates the entire Streamlit application flow:
    1. Configure page settings
    2. Load AWS configuration
    3. Render UI components
    4. Handle predictions via SageMaker
    5. Display context and historical data
    """
    logger.info("App started")
    render_page_config()

    endpoint_name, aws_region = load_aws_configuration()
    if endpoint_name is None:
        st.stop()

    volatility_ranges = get_volatility_ranges()

    tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Historical Performance"])

    with tab1:
        forecast_days = render_ui_components(volatility_ranges)
        handle_prediction(forecast_days, endpoint_name, aws_region)
        render_context_and_history(volatility_ranges)

    with tab2:
        render_historical_performance_tab(
            endpoint_name=endpoint_name,
            aws_region=aws_region,
        )


if __name__ == "__main__":
    main()
