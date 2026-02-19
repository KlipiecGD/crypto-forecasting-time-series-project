import boto3
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional
from datetime import datetime, timedelta

from sklearn.metrics import confusion_matrix

from src.training.model_evaluator import calculate_metrics
from src.fetch_data.live_data import fetch_live_data
from src.features.preprocess_data import preprocess_data
from src.features.data_transformer import DataTransformer
from src.config.config import config


def render_page_config() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Bitcoin Volatility Predictor", page_icon="🪙", layout="wide"
    )


def get_volatility_ranges() -> dict:
    """
    Get volatility classification ranges from config.

    Returns:
        dict: Volatility classification ranges with keys "Low", "Normal", "High"
    """
    return {
        "Low": config.get_thresholds_config.get("Low", 0.40),
        "Normal": config.get_thresholds_config.get("Normal", 0.65),
        "High": np.inf,
    }


def render_sidebar(volatility_ranges: dict, deployment_type: str = "local") -> None:
    """
    Render the sidebar with app information.

    Args:
        volatility_ranges (dict): Volatility classification ranges
        deployment_type (str): "local" or "aws" to customize the description
    """
    with st.sidebar:
        st.title("🪙 Bitcoin Volatility Predictor")
        st.markdown("---")

        st.subheader("📊 Volatility Classification")
        st.markdown(f"""
        - **Low**: <= {volatility_ranges["Low"] * 100}%
        - **Normal**: {volatility_ranges["Low"] * 100}% - {volatility_ranges["Normal"] * 100}%
        - **High**: > {volatility_ranges["Normal"] * 100}%
        """)

        st.markdown("---")

        st.subheader("ℹ️ About")
        model_type = config.get_model_config.get("model_type", "Logistic Regression")

        deployment_info = ""
        if deployment_type == "aws":
            deployment_info = ", hosted on AWS SageMaker"

        st.markdown(f"""
        This app predicts Bitcoin volatility 1-30 days ahead using machine learning.
        
        **Data Range**: 2014 - 2026
        
        **Model**: {model_type} with feature engineering {deployment_info}.
        """)

        st.markdown("---")

        st.subheader("🔍 Features")
        st.markdown("""
        **🔮 Prediction Tab**
        - Forecast volatility 1–30 days ahead
        - Confidence scores per class
        - Live market context & current price
        - Historical volatility chart (2014–2026)
                    
        **📊 Historical Performance Tab**
        - Evaluate model on any date range
        - Accuracy, F1, precision & recall metrics
        - Per-class breakdown
        - Confusion matrix
        """)


def render_header() -> datetime:
    """
    Render the main header with today's date.

    Returns:
        datetime: Today's date
    """
    st.title("🔮 Bitcoin Volatility Prediction")

    today = datetime.now()

    st.markdown(f"""
    **Today**: {today.strftime("%B %d, %Y")}  
    """)
    st.markdown("")

    return today


def render_forecast_selector() -> int:
    """
    Render forecast horizon selector.

    Returns:
        int: Selected forecast days (1-30)
    """
    st.markdown("### 🎯 Select Forecast Horizon")

    col1, col2 = st.columns([3, 1])

    with col1:
        forecast_days = st.slider(
            "How many days ahead do you want to predict?",
            min_value=config.get_forecast_config.get("min_horizon", 1),
            max_value=config.get_forecast_config.get("max_horizon", 30),
            value=config.get_forecast_config.get("default_horizon", 30),
            help="Select the number of days into the future for volatility prediction",
        )

    with col2:
        st.metric("Days Ahead", f"{forecast_days}", help="Selected forecast horizon")

    target_date = datetime.now() + timedelta(days=forecast_days)
    st.info(f"📅 Prediction target: **{target_date.strftime('%B %d, %Y')}**")

    return forecast_days


def render_prediction_button() -> bool:
    """
    Render the prediction button and return whether it was clicked.

    Returns:
        bool: True if the button was clicked, False otherwise
    """
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "🎯 PREDICT VOLATILITY", width="stretch", type="primary"
        )
    st.markdown("")
    return predict_button


def create_confidence_chart(probabilities: dict, predicted_class: str) -> go.Figure:
    """
    Create a bar chart showing confidence scores for each volatility class.

    Args:
        probabilities (dict): Probability scores for each class
        predicted_class (str): The predicted class (for highlighting)

    Returns:
        go.Figure: Plotly bar chart figure
    """
    classes = config.get_model_config.get("class_names", ["Low", "Normal", "High"])
    probs = [probabilities.get(cls, 0) * 100 for cls in classes]

    # Color scheme: highlight predicted class
    colors = []
    for cls in classes:
        if cls == predicted_class:
            if cls == "Low":
                colors.append("#28a745")  # Green
            elif cls == "Normal":
                colors.append("#ffc107")  # Yellow/Orange
            else:  # High
                colors.append("#dc3545")  # Red
        else:
            colors.append("#e0e0e0")  # Gray for non-predicted classes

    fig = go.Figure(
        data=[
            go.Bar(
                x=classes,
                y=probs,
                marker_color=colors,
                text=[f"{p:.1f}%" for p in probs],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title={
            "text": "Model Confidence Distribution",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18, "color": "#333"},
        },
        xaxis_title="Volatility Class",
        yaxis_title="Confidence (%)",
        yaxis_range=[0, 100],
        height=400,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=14),
        margin=dict(t=80, b=60, l=60, r=40),
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.2)")

    return fig


def render_volatility_explanation() -> None:
    """Render expandable explanation of how volatility is calculated."""
    with st.expander("📐 How is Volatility Calculated?", expanded=False):
        st.markdown("""
        ### Understanding Historical Volatility
        
        **Historical Volatility (HV)** measures how much an asset's price fluctuates over time. 
        Higher volatility means larger price swings, while lower volatility indicates more stable prices.
        
        ---
        
        ### The Formula for Historical Volatility
        """)

        st.latex(r"HV = 100 \cdot \sqrt{\frac{365}{N} \sum_{i=1}^{N} R_i^2}")

        st.markdown("""
        **Where:**
        - **$HV$** = Historical Volatility (annualized percentage)
        - **$N$** = Number of trading days in the period (e.g., 30 for a month)
        - **$R_i$** = Daily log returns (continuously compounded returns)
        - **365** = Annualization factor for crypto (trades 24/7, unlike traditional markets that use 252 trading days)
        - **100** = Converts the result to a percentage
        
        ---
        
        ### Example Interpretation
        
        - **HV = 20%**: Low volatility — price is relatively stable
        - **HV = 50%**: Moderate volatility — typical for crypto markets
        - **HV = 100%+**: High volatility — expect large price swings
        
        ---
 
        This model predicts **future volatility class** (Low/Normal/High) to help you make informed decisions.
        """)


def render_prediction_results(
    predicted_class: str,
    probabilities: dict,
    target_date: datetime,
) -> None:
    """
    Render prediction results with confidence scores.

    Args:
        predicted_class (str): Predicted volatility class
        probabilities (dict): Probability scores for each class
        target_date (datetime): Target prediction date
    """
    # Determine container style based on prediction
    if predicted_class == "Low":
        result_container = st.success
        emoji = "📉"
    elif predicted_class == "Normal":
        result_container = st.warning
        emoji = "📊"
    else:  # High
        result_container = st.error
        emoji = "📈"

    # Display main result
    result_container(f"### {emoji} Predicted Volatility: **{predicted_class.upper()}**")

    st.markdown(f"**Target Date:** {target_date.strftime('%B %d, %Y')}")
    st.markdown("---")

    # Confidence visualization with chart
    st.markdown("### 📊 Confidence Distribution")

    # Create two columns: chart on left, details on right
    col1, col2 = st.columns([2, 1])

    with col1:
        # Show interactive chart
        fig = create_confidence_chart(probabilities, predicted_class)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown("**Detailed Scores:**")
        st.markdown("")

        # Sort by probability (highest first)
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

        for class_name, prob in sorted_probs:
            percentage = prob * 100

            st.markdown(f"**{class_name}**")
            st.progress(prob)
            st.markdown(
                f"<p style='text-align: right; margin-top: -10px; color: #666;'>{percentage:.2f}%</p>",
                unsafe_allow_html=True,
            )
            st.markdown("")

    st.markdown("---")

    # Interpretation helper
    max_prob = max(probabilities.values())

    if max_prob > 0.7:
        confidence_level = "**High Confidence**"
        interpretation = "The model is very confident in this prediction."
    elif max_prob > 0.5:
        confidence_level = "**Moderate Confidence**"
        interpretation = "The model is fairly confident, but there's some uncertainty."
    else:
        confidence_level = "**Low Confidence**"
        interpretation = "The model is uncertain. Multiple outcomes are possible."

    st.info(
        f"**{confidence_level}**\n\n"
        f"{interpretation}\n\n"
        f"💡 The model predicts **{predicted_class.upper()}** volatility "
        f"with {max_prob * 100:.1f}% confidence on {target_date.strftime('%B %d, %Y')}."
    )

    # What are confidence scores? (Expandable explanation)
    with st.expander("ℹ️ What are Confidence Scores?", expanded=False):
        st.markdown("""
        **Confidence scores** represent how certain the model is about each possible outcome:
        
        - **Higher scores** (closer to 100%) = Model is more confident
        - **Lower scores** (closer to 0%) = Model is less confident
        - **All scores add up to 100%** across the three categories

        The predicted class is the one with the **highest confidence score**.
        """)

    st.markdown("")


def render_market_context(preprocessed_data: pd.DataFrame | None) -> None:
    """
    Render market context section with current price.

    Args:
        preprocessed_data (pd.DataFrame | None): Preprocessed market data
    """
    st.markdown("---")
    st.subheader("📈 Market Context")

    if preprocessed_data is None or len(preprocessed_data) < 30:
        st.warning("Insufficient data for market context.")
        return

    # Get latest data point
    latest = preprocessed_data.iloc[-1]
    prev = preprocessed_data.iloc[-2]

    # Calculate Price Change
    price_change = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100

    # Calculate Current Volatility (30-day annualized)
    # matching the logic used in the data transformer
    log_returns = np.log(
        preprocessed_data["Close"] / preprocessed_data["Close"].shift(1)
    )
    current_volatility = log_returns[-30:].std() * np.sqrt(365) * 100

    # Create 4 columns for metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="💰 Current Price",
            value=f"${latest['Close']:,.2f}",
            delta=f"{price_change:+.2f}%",
            help="Latest closing price",
        )

    with col2:
        st.metric(
            label="📊 24h Volume",
            value=f"{latest['Volume']:,.0f} BTC",
            help="Volume from the last 24h candle",
        )

    with col3:
        st.metric(
            label="⚡ Current Volatility (30d)",
            value=f"{current_volatility:.2f}%",
            help="Annualized historical volatility (last 30 days)",
        )

    with col4:
        st.metric(
            label="🕒 Fetch Time",
            value=datetime.now().strftime("%H:%M:%S"),
            help="Time when the latest data was fetched",
        )


def render_historical_volatility_chart(
    historical_data: pd.DataFrame | None, volatility_ranges: dict
) -> None:
    """
    Render historical volatility chart.

    Args:
        historical_data (pd.DataFrame | None): Historical volatility data
        volatility_ranges (dict): Volatility classification ranges
    """
    st.markdown("---")
    st.subheader("📊 Historical Volatility")

    if historical_data is None:
        st.warning("Could not load historical volatility data.")
        return

    # Create color mapping
    color_map = config.get_ui_config.get(
        "color_map", {"Low": "green", "Normal": "orange", "High": "red"}
    )

    # Create the chart
    fig = go.Figure()

    # Add scatter plot for each class
    for volatility_class in config.get_model_config.get(
        "class_names", ["Low", "Normal", "High"]
    ):
        class_data = historical_data[historical_data["class"] == volatility_class]
        fig.add_trace(
            go.Scatter(
                x=class_data["date"],
                y=class_data["volatility"] * 100,  # Convert to percentage
                mode="markers",
                name=volatility_class,
                marker=dict(color=color_map[volatility_class], size=4, opacity=0.6),
                hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>Volatility</b>: %{y:.2f}%<br><b>Class</b>: "
                + volatility_class
                + "<extra></extra>",
            )
        )

    # Add threshold lines
    fig.add_hline(
        y=volatility_ranges["Low"] * 100,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Low/Normal threshold ({volatility_ranges['Low'] * 100}%)",
        annotation_position="right",
    )

    fig.add_hline(
        y=volatility_ranges["Normal"] * 100,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Normal/High threshold ({volatility_ranges['Normal'] * 100}%)",
        annotation_position="right",
    )

    # Update layout
    fig.update_layout(
        title="Bitcoin Volatility Over Time (2014-2026)",
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        hovermode="closest",
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, width="stretch")


def render_historical_patterns(historical_data: pd.DataFrame | None) -> None:
    """
    Render historical patterns expandable section.

    Args:
        historical_data (pd.DataFrame | None): Historical volatility data
    """
    with st.expander("📊 Historical Patterns"):
        if historical_data is None:
            st.warning("Historical data not available.")
            return

        st.markdown("### Overall Distribution (2014-2026)")

        # Calculate distribution
        distribution = (
            historical_data["class"]
            .value_counts(normalize=True)
            .reindex(["Low", "Normal", "High"], fill_value=0)
        )

        # Create bar chart
        color_map = config.get_ui_config.get(
            "color_map", {"Low": "green", "Normal": "orange", "High": "red"}
        )
        fig_dist = px.bar(
            x=distribution.index,
            y=np.array(distribution.values) * 100,
            labels={"x": "Volatility Class", "y": "Percentage (%)"},
            title="Distribution of Volatility Classes",
            color=distribution.index,
            color_discrete_map=color_map,
        )

        fig_dist.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_dist, width="stretch")

        # Display percentages
        col1, col2, col3 = st.columns(3)
        for idx, (class_name, percentage) in enumerate(distribution.items()):
            with [col1, col2, col3][idx]:
                st.metric(label=f"{class_name} Days", value=f"{percentage * 100:.1f}%")

        # Additional statistics
        st.markdown("---")
        st.markdown("### Statistical Summary")
        stats_col1, stats_col2 = st.columns(2)

        with stats_col1:
            st.metric(
                "Historical Mean Volatility",
                f"{historical_data['volatility'].mean() * 100:.2f}%",
            )
            st.metric(
                "Historical Median Volatility",
                f"{historical_data['volatility'].median() * 100:.2f}%",
            )

        with stats_col2:
            st.metric(
                "Historical Std Dev",
                f"{historical_data['volatility'].std() * 100:.2f}%",
            )
            st.metric(
                "Historical Max Volatility",
                f"{historical_data['volatility'].max() * 100:.2f}%",
            )


def render_model_information() -> None:
    """
    Render model information expandable section.
    """
    with st.expander("🤖 Model Information"):
        st.markdown("### Model Details")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            **Model Type**: {config.get_model_config.get("model_type", "Logistic Regression")}\n
            **Prediction Horizon**: {config.get_forecast_config.get("min_horizon", 1)} - {config.get_forecast_config.get("max_horizon", 30)} days ahead
            
            **Pipeline**:
            - Data Fetching (Binance API)
            - Feature Engineering
            - Standard Scaling
            - Classification
            """)

        with col2:
            st.markdown("""
            **Key Features Used**:
            - Historical Volatility
            - Parkinson Volatility
            - Garman-Klass Volatility
            - Roger-Satchel Volatility
            - Lagged Features
            - Rolling Statistics
            - Volatility Changes & Momentum
            """)


def render_historical_performance_tab(
    endpoint_name: Optional[str] = None, aws_region: Optional[str] = None
) -> None:
    """
    Render the historical performance analysis tab.
    If endpoint_name is provided, invokes SageMaker endpoint (AWS mode).
    Otherwise loads model locally (local mode).
    """
    st.header("📊 Historical Performance Analysis")
    st.markdown(
        "Select a date range to evaluate how the model performed on data from the recent past."
    )

    # Config values
    forecast_horizon = config.get_forecast_config.get("default_horizon", 30)
    buffer = config.get_live_data_config.get("historical_data_buffer", 61)
    min_range_days = buffer + forecast_horizon

    # Calculate date limits based on dataset and forecast horizon
    max_end_date = (
        datetime.now().date() - timedelta(days=forecast_horizon)
    )  # We need to ensure we have labels for the end date, so we cap it to today minus forecast horizon
    data_start_str = config.get_historical_data_config.get("data_start", "2014-09-17")
    data_start = datetime.strptime(data_start_str, "%Y-%m-%d").date()
    data_end = datetime.now().date()  # Assume we have data up to today in our dataset
    total_days = (
        (data_end - data_start).days - forecast_horizon - buffer
    )  # We need to subtract the forecast horizon and buffer from the total available days because during transformation we will lose the first 'buffer' days and the last 'forecast_horizon' days (no labels for those)
    train_cutoff_days = int(
        total_days
        * (
            1
            - config.get_model_config.get("val_size", 0.15)
            - config.get_model_config.get("test_size", 0.15)
        )
    )  # Approximately 70% of the data for training, so we can only evaluate on the last 30%
    min_start_date = data_start + timedelta(days=train_cutoff_days)

    min_days_to_predict = config.get_ui_config.get(
        "min_days_to_predict", 30
    )  # Minimum number of days in the selected range to perform evaluation

    # Controls
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        start_date = st.date_input(
            "Start Date",
            value=max_end_date
            - timedelta(days=360),  # Default to evaluating the last year
            min_value=min_start_date,
            max_value=max_end_date - timedelta(days=min_days_to_predict),
            help=(
                f"Select a start date for evaluation. Must be between {min_start_date} and "
                f"{max_end_date - timedelta(days=min_days_to_predict)} to allow for a valid range."
            ),
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=max_end_date,
            min_value=min_start_date
            + timedelta(
                days=min_range_days
            ),  # Ensure end date is at least 'min_range_days' after the minimum start date to allow for a valid range
            max_value=max_end_date,
            help=(
                f"Select an end date for evaluation. Must be between {min_start_date + timedelta(days=min_range_days)} and "
                f"{max_end_date} to allow for a valid range."
            ),
        )
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_button = st.button("▶ Analyze", type="primary", width="stretch")

    # Validation
    selected_days = (end_date - start_date).days
    if selected_days < min_days_to_predict:
        st.warning(
            f"Please select a date range of at least {min_days_to_predict} days to perform evaluation."
        )
        return

    if not run_button:
        return

    # Pipeline
    with st.spinner("Fetching data and evaluating model..."):
        # Step 1: Fetch data
        fetch_start = datetime.combine(start_date, datetime.min.time()) - timedelta(
            days=buffer
        )
        fetch_end = datetime.combine(end_date, datetime.min.time()) + timedelta(
            days=forecast_horizon
        )

        raw_data = fetch_live_data(
            start_date=fetch_start, end_date=fetch_end, save=False
        )
        if raw_data is None or raw_data.empty:
            st.error(
                "❌ Failed to fetch data from Binance for the selected date range."
            )
            return

        # Step 2: Preprocess
        preprocessed = preprocess_data(raw_data)
        if preprocessed is None or preprocessed.empty:
            st.error("❌ Preprocessing failed. Try a different date range.")
            return

        # Step 3: Transform — training_mode=True to get labels
        transformer = DataTransformer(
            training_mode=True, forecast_horizon=forecast_horizon
        )
        transformed = transformer.transform(preprocessed)

        if transformed.empty or len(transformed) < min_days_to_predict:
            st.error(
                f"❌ Not enough samples ({len(transformed)} rows). "
                "Try a wider date range."
            )
            return

        X = transformed.drop(columns=["Target"])
        y_true = transformed["Target"].astype(int)

        # Step 4: Predict — SageMaker or local
        if endpoint_name and aws_region:
            try:
                client = boto3.client("sagemaker-runtime", region_name=aws_region)
                payload = X.to_dict(orient="records")
                response = client.invoke_endpoint(
                    EndpointName=endpoint_name,
                    ContentType="application/json",
                    Body=json.dumps(payload),
                )
                results = json.loads(response["Body"].read().decode())
                mapping = config.get_model_config.get(
                    "mapping", {"Low": 0, "Normal": 1, "High": 2}
                )
                y_pred = [mapping[r["predicted_class"]] for r in results]

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                return

        else:
            from src.inference.model_loader import load_inference_model

            model = load_inference_model()
            y_pred = model.predict(X)

        # Step 5: Metrics
        metrics = calculate_metrics(y_true, y_pred)

    # Results
    st.success(f"✅ Evaluated on **{len(X)} samples** | {start_date} -> {end_date}")
    st.markdown("---")

    # Metric cards
    st.subheader("📈 Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    with col2:
        st.metric("F1 Macro", f"{metrics['f1_macro']:.2%}")
    with col3:
        st.metric("F1 Weighted", f"{metrics['f1_weighted']:.2%}")
    with col4:
        thresholds = config.get_monitoring_config.get("performance_thresholds", {})
        failed = [k for k, v in thresholds.items() if metrics.get(k, 1.0) < v]
        if failed:
            st.metric(
                "Quality Threshold Check",
                "⚠️ Failed",
                delta=f"{len(failed)} metric(s)",
                delta_color="inverse",
            )
        else:
            st.metric("Quality Threshold Check", "✅ Passed")

    # Thresholds reference
    with st.expander("⚙️ Quality Thresholds"):
        thresholds = config.get_monitoring_config.get("performance_thresholds", {})
        threshold_df = pd.DataFrame(
            {
                "Metric": list(thresholds.keys()),
                "Threshold": [f"{v:.2%}" for v in thresholds.values()],
                "Actual": [f"{metrics.get(k, 0):.2%}" for k in thresholds.keys()],
                "Status": [
                    "✅ Pass" if metrics.get(k, 0) >= v else "⚠️ Fail"
                    for k, v in thresholds.items()
                ],
            }
        )
        st.dataframe(threshold_df, hide_index=True, width="stretch")

    # Per-class table
    st.markdown("---")
    st.subheader("📋 Per-Class Metrics")
    class_names = config.get_model_config.get("class_names", ["Low", "Normal", "High"])
    per_class_df = pd.DataFrame(
        {
            "Class": class_names,
            "F1": [f"{metrics[f'f1_{c}']:.2%}" for c in class_names],
            "Precision": [f"{metrics[f'precision_{c}']:.2%}" for c in class_names],
            "Recall": [f"{metrics[f'recall_{c}']:.2%}" for c in class_names],
        }
    )
    st.dataframe(per_class_df, hide_index=True, width="stretch")

    # Confusion matrix
    st.markdown("---")
    st.subheader("🔢 Confusion Matrix")
    cm_normalized = confusion_matrix(y_true, y_pred, normalize="true")
    cm_counts = confusion_matrix(y_true, y_pred)

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_normalized,
            x=class_names,
            y=class_names,
            colorscale="Blues",
            zmin=0,
            zmax=1,
            text=[[f"{val:.2%}" for val in row] for row in cm_normalized],
            texttemplate="%{text}",
            textfont={"size": 16},
            customdata=cm_counts,
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>Share: %{text}<br>Count: %{customdata}<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
        title=dict(
            text=f"Normalized Confusion Matrix — {start_date} → {end_date}",
            x=0.5,
            xanchor="center",
        ),
        height=600,
        margin=dict(t=60, b=60, l=80, r=40),
    )

    st.plotly_chart(fig, width="stretch")
