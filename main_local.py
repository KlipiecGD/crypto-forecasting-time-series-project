import streamlit as st
from datetime import datetime, timedelta
from typing import Optional

from src.config.config import config
from src.pipelines.inference_pipeline import run_inference
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


def make_prediction(
    forecast_horizon: int,
) -> tuple[Optional[str], Optional[dict], Optional[datetime]]:
    """
    Run the local inference pipeline and format results.

    Args:
        forecast_horizon (int): Number of days ahead for the forecast (1-30)

    Returns:
        tuple containing:
            - predicted_class (str): Predicted volatility class
            - probabilities (dict): Probability dictionary for each class
            - target_date (datetime): Date of the prediction
        Returns (None, None, None) if error occurs
    """
    result = run_inference(forecast_horizon=forecast_horizon, verbose=False)
    if result is None:
        return None, None, None

    predicted_class, prob_array, target_date = result

    # Convert probabilities array to dictionary
    mapping = config.get_model_config.get("mapping", {"Low": 0, "Normal": 1, "High": 2})
    probabilities = {
        class_name.capitalize(): prob_array[idx] for class_name, idx in mapping.items()
    }

    target_date = datetime.now() + timedelta(days=forecast_horizon)

    return predicted_class.capitalize(), probabilities, target_date


def render_ui_components(volatility_ranges: dict) -> int:
    """
    Render all UI components and return selected forecast days.

    Args:
        volatility_ranges (dict): Volatility classification thresholds

    Returns:
        int: Selected forecast horizon in days
    """
    # Sidebar
    render_sidebar(volatility_ranges, deployment_type="local")

    # Main content - Header
    render_header()

    # Forecast horizon selector
    forecast_days = render_forecast_selector()
    st.markdown("---")

    return forecast_days


def handle_prediction(forecast_days: int) -> None:
    """
    Handle prediction logic and display results.

    Args:
        forecast_days (int): Number of days ahead to predict
    """
    predict_button = render_prediction_button()

    if predict_button:
        with st.spinner("Fetching live data and analyzing market conditions..."):
            predicted_class, probabilities, target_date = make_prediction(forecast_days)

            if predicted_class is None or probabilities is None or target_date is None:
                st.error(
                    "❌ Failed to make prediction. Please check your data connection and try again."
                )
            else:
                render_prediction_results(predicted_class, probabilities, target_date)


def render_context_and_history(volatility_ranges: dict) -> None:
    """
    Render market context, historical charts, and additional information.

    Args:
        volatility_ranges (dict): Volatility classification thresholds
    """
    # Market context
    market_context_data = fetch_current_market_data(limit=2)
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
    Main application entry point.

    Orchestrates the entire Streamlit application flow:
    1. Configure page settings
    2. Load configuration
    3. Render UI components
    4. Handle predictions
    5. Display context and historical data
    """
    # Page configuration
    render_page_config()

    # Get volatility thresholds
    volatility_ranges = get_volatility_ranges()

    tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Historical Performance"])
    with tab1:
        # Render UI and get forecast selection
        forecast_days = render_ui_components(volatility_ranges)

        # Handle prediction logic
        handle_prediction(forecast_days)

        # Render additional information
        render_context_and_history(volatility_ranges)
    with tab2:
        render_historical_performance_tab()


if __name__ == "__main__":
    main()
