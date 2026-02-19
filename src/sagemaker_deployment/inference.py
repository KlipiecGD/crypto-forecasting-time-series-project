import os
import json
import joblib
import pandas as pd
import logging
from io import StringIO
from flask import Flask, request, Response
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Model path
MODEL_PATH = os.environ.get("MODEL_PATH", "/opt/ml/model")

# Global variable to hold the model
model = None


def load_model_global():
    global model
    if model is None:
        model = model_fn(MODEL_PATH)


def model_fn(model_dir: str) -> Pipeline:
    """
    Load the model from the model directory.
    Args:
        model_dir (str): Directory where the model is stored
    Returns:
        Pipeline: Loaded model pipeline
    """
    model_path = os.path.join(model_dir, "volatility_model.joblib")
    logger.info(f"Loading model from: {model_path}")
    pipeline = joblib.load(model_path)
    return pipeline


def input_fn(request_body: str, content_type: str = "application/json") -> pd.DataFrame:
    """
    Deserialize and prepare the input data.
    Args:
        request_body (str): Raw request body
        content_type (str): Content type of the input data
    Returns:
        pd.DataFrame: Prepared input data for prediction
    """
    if content_type == "application/json":
        input_data = json.loads(request_body)
        if isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, dict):
            df = pd.DataFrame(input_data.get("data", [input_data]))
        else:
            raise ValueError("Invalid input format")
        return df
    elif content_type == "text/csv":
        return pd.read_csv(StringIO(request_body))
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data: pd.DataFrame, model: Pipeline) -> list:
    """
    Make predictions.
    Args:
        input_data (pd.DataFrame): Input data for prediction
        model (Pipeline): Loaded model pipeline
    Returns:
        list: Prediction results with predicted class, confidence, and probabilities for each class
    """
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    # Define mapping explicitly
    class_mapping = {"Low": 0, "Normal": 1, "High": 2}
    inv_class_mapping = {v: k for k, v in class_mapping.items()}

    results = []
    for pred, probs in zip(predictions, probabilities):
        results.append(
            {
                "predicted_class": inv_class_mapping[pred],
                "confidence": float(probs[pred]),
                "probabilities": {
                    "Low": float(probs[0]),
                    "Normal": float(probs[1]),
                    "High": float(probs[2]),
                },
            }
        )
    return results


def output_fn(prediction: list, accept: str = "application/json") -> tuple[str, str]:
    """
    Serialize the prediction output.
    Args:
        prediction (list): Prediction results to serialize
        accept (str): Accept header from the request
    Returns:
        tuple: Serialized prediction and content type
    """
    if accept == "application/json" or accept == "*/*":
        return json.dumps(prediction), "application/json"
    else:
        raise ValueError(f"Unsupported accept type: {accept}")


# --- SERVER LOGIC ---


@app.route("/ping", methods=["GET"])
def ping():
    """Health check."""
    load_model_global()
    status = 200 if model else 404
    return Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def invoke():
    """Inference endpoint."""
    if model is None:
        load_model_global()

    try:
        body = request.data.decode("utf-8")
        data = input_fn(body, request.content_type)
        prediction = predict_fn(data, model)

        # Get accept header, defaulting to application/json
        accept = request.headers.get("Accept", "application/json")
        result, accept_type = output_fn(prediction, accept)

        return Response(response=result, status=200, mimetype=accept_type)
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return Response(response=str(e), status=500)


if __name__ == "__main__":
    load_model_global()
    app.run(host="0.0.0.0", port=8080)
