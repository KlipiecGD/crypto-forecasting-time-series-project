import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from typing import Optional
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from utils.common_utils import calculate_evaluation_metrics
from utils.model_development_classes import VolatilityDataset


# Parkinson's Volatility Estimator
def calculate_parkinson_volatility(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate Parkinson's volatility estimator.
    Args:
        df (pd.DataFrame): DataFrame containing 'High' and 'Low' price columns.
        window (int): Rolling window size for volatility calculation.
    Returns:
        pd.Series: Parkinson's volatility estimates.
    """
    term_1 = 1 / (4 * window * np.log(2))
    term_2 = (np.log(df["High"] / df["Low"])) ** 2
    parkinson_vol = np.sqrt(term_1 * term_2.rolling(window=window).sum()) * np.sqrt(365)
    return parkinson_vol


# Garman-Klass Volatility Estimator
def calculate_garman_klass_volatility(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate Garman-Klass volatility estimator.
    Args:
        df (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' price columns.
        window (int): Rolling window size for volatility calculation.
    Returns:
        pd.Series: Garman-Klass volatility estimates.
    """
    term_1 = 0.5 * (np.log(df["High"] / df["Low"])) ** 2
    term_2 = (2 * np.log(2) - 1) * (np.log(df["Close"] / df["Open"])) ** 2
    gk_vol = np.sqrt(
        (1 / window) * (term_1 - term_2).rolling(window=window).sum()
    ) * np.sqrt(365)
    return gk_vol


# Roger-Satchel Volatility Estimator
def calculate_roger_satchel_volatility(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Calculate Roger-Satchel volatility estimator.
    Args:
        df (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' price columns.
        window (int): Rolling window size for volatility calculation.
    Returns:
        pd.Series: Roger-Satchel volatility estimates.
    """
    term1 = np.log(df["High"] / df["Close"]) * np.log(df["High"] / df["Open"])
    term2 = np.log(df["Low"] / df["Close"]) * np.log(df["Low"] / df["Open"])
    rs_var = (term1 + term2).rolling(window=window).mean()
    return np.sqrt(rs_var) * np.sqrt(365)


def train_model_with_feature_sets(
    model,
    model_name: str,
    feature_sets: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    metric: str = "RMSE",
    scale: bool = False,
    scaler: Optional[StandardScaler] = None,
    verbose: bool = True,
) -> dict:
    """
    Train a model with multiple feature sets and select the best one.
    Args:
        model: The machine learning model to train (e.g., LinearRegression()).
        model_name (str): Name of the model (for reporting).
        feature_sets (dict): Dictionary of feature sets to evaluate.
            Format: { 'set_name': {'features': [...], 'lookback_days': int} }
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target variable.
        metric (str): The metric to optimize for (e.g., 'RMSE', 'MAE', 'R2').
        scale (bool): Whether to scale the features.
        scaler: The scaler object to use if scaling is True (e.g., StandardScaler()). If none and scale is True, StandardScaler will be used by default.
        verbose (bool): Whether to print detailed results for each feature set.
    Returns:
        dict: A dictionary containing the best model, best feature set name, best features, and best metrics. Also includes all results for comparison.
    """

    results = {}
    best_score = float("inf") if metric != "R2" else float("-inf")
    best_feature_set_name = None
    best_model = None
    best_scaler = None
    best_features = None
    best_metrics = None

    if scale and scaler is None:
        scaler = StandardScaler()

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"Training {model_name} with {len(feature_sets)} feature sets")
        print(f"Optimizing for: {metric}")
        print(f"Scaling: {'Yes' if scale else 'No'}")
        print(f"{'=' * 80}\n")

    for set_name, set_config in feature_sets.items():
        features = set_config["features"]
        lookback_days = set_config["lookback_days"]

        # Clone model to get fresh instance
        model_clone = clone(model)

        # Clone scaler if scaling is enabled
        scaler_clone = clone(scaler) if scale else None

        try:
            X_train_subset = X_train[features]
            X_val_subset = X_val[features]

            # Prepare data
            if scale:
                X_train_prepared = scaler_clone.fit_transform(X_train_subset)
                X_val_prepared = scaler_clone.transform(X_val_subset)
            else:
                X_train_prepared = X_train_subset
                X_val_prepared = X_val_subset

            # Train model
            model_clone.fit(X_train_prepared, y_train)

            # Predictions
            y_pred_train = model_clone.predict(X_train_prepared)
            y_pred_val = model_clone.predict(X_val_prepared)

            # Metrics
            metrics_train = calculate_evaluation_metrics(
                y_train, y_pred_train, include_r2_score=True
            )
            metrics_val = calculate_evaluation_metrics(
                y_val, y_pred_val, include_r2_score=True
            )

            # Store results
            results[set_name] = {
                "model": model_clone,
                "scaler": scaler_clone,
                "feature_set_name": set_name,
                "features": features,
                "n_features": len(features),
                "lookback_days": lookback_days,
                "train_metrics": metrics_train,
                "val_metrics": metrics_val,
            }

            # Check if this is the best
            val_score = metrics_val[metric]
            is_better = (
                (val_score < best_score) if metric != "R2" else (val_score > best_score)
            )

            if is_better:
                best_score = val_score
                best_feature_set_name = set_name
                best_model = model_clone
                best_scaler = scaler_clone
                best_features = features
                best_metrics = {"train": metrics_train, "val": metrics_val}

            # Print results
            if verbose:
                print(f"Feature Set: {set_name}")
                print(f"  Features: {len(features)}, Lookback: {lookback_days} days")
                print(f"  Training {metric}: {metrics_train[metric]:.6f}")
                print(f"  Validation {metric}: {metrics_val[metric]:.6f}")
                print()

        except Exception as e:
            if verbose:
                print(f"Feature Set: {set_name} - FAILED")
                print(f"  Error: {str(e)}\n")
            continue

    if verbose:
        print(f"{'=' * 80}")
        print(f"BEST FEATURE SET: {best_feature_set_name}")
        print(f"  Features: {len(best_features)}")
        print(
            f"  Lookback: {feature_sets[best_feature_set_name]['lookback_days']} days"
        )
        print(f"  Validation {metric}: {best_score:.6f}")
        print(f"{'=' * 80}\n")

    return {
        "best_model": best_model,
        "best_scaler": best_scaler,
        "best_feature_set_name": best_feature_set_name,
        "best_features": best_features,
        "best_metrics": best_metrics,
        "all_results": results,
    }


def train_neural_network_with_feature_sets(
    model_class,
    model_name: str,
    feature_sets: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    sequence_length: int = 30,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 0.001,
    patience: int = 10,
    metric: str = "RMSE",
    device: str = "mps" if torch.backends.mps.is_available() else "cpu",
    verbose: bool = True,
) -> dict:
    """
    Train a neural network model with multiple feature sets.

    Args:
        model_class: Neural network class (VolatilityLSTM, VolatilityGRU, AttentionLSTM)
        model_name (str): Name of the model
        feature_sets (dict): Dictionary of feature sets
        X_train, y_train: Training data
        X_val, y_val: Validation data
        sequence_length (int): Number of time steps in sequence
        hidden_size (int): Hidden layer size
        num_layers (int): Number of LSTM/GRU layers
        dropout (float): Dropout rate
        batch_size (int): Batch size for training
        epochs (int): Maximum number of epochs
        learning_rate (float): Learning rate
        patience (int): Early stopping patience
        metric (str): Metric to optimize ('RMSE', 'MAE', 'R2')
        device (str): 'cuda' or 'cpu'
        verbose (bool): Print training progress

    Returns:
        dict: Best model, scaler, features, and metrics
    """
    from sklearn.preprocessing import StandardScaler

    results = {}
    best_score = float("inf") if metric != "R2" else float("-inf")
    best_feature_set_name = None
    best_model = None
    best_scaler = None
    best_features = None
    best_metrics = None

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"Training {model_name} with {len(feature_sets)} feature sets")
        print(f"Sequence length: {sequence_length}, Device: {device}")
        print(f"Optimizing for: {metric}")
        print(f"{'=' * 80}\n")

    for set_name, set_config in feature_sets.items():
        features = set_config["features"]
        lookback_days = set_config["lookback_days"]

        try:
            # Select features
            X_train_subset = X_train[features]
            X_val_subset = X_val[features]

            # Scale features (neural networks need scaling)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_subset)
            X_val_scaled = scaler.transform(X_val_subset)

            # Convert back to DataFrame for dataset creation
            X_train_scaled_df = pd.DataFrame(
                X_train_scaled, columns=features, index=X_train_subset.index
            )
            X_val_scaled_df = pd.DataFrame(
                X_val_scaled, columns=features, index=X_val_subset.index
            )

            # Create datasets
            train_dataset = VolatilityDataset(
                X_train_scaled_df, y_train, sequence_length
            )
            val_dataset = VolatilityDataset(X_val_scaled_df, y_val, sequence_length)

            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # Initialize model
            input_size = len(features)
            model = model_class(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            ).to(device)

            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            # Training loop
            best_val_loss = float("inf")
            patience_counter = 0

            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0.0
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * X_batch.size(0)

                train_loss /= len(train_loader.dataset)

                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                        val_loss += loss.item() * X_batch.size(0)

                val_loss /= len(val_loader.dataset)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"  Early stopping at epoch {epoch + 1}")
                        break

            # Load best model state
            model.load_state_dict(best_model_state)

            # Get predictions for metrics
            model.eval()
            with torch.no_grad():
                # Training predictions
                train_preds = []
                train_targets = []
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(device)
                    outputs = model(X_batch)
                    train_preds.extend(outputs.cpu().numpy().flatten())
                    train_targets.extend(y_batch.numpy().flatten())

                # Validation predictions
                val_preds = []
                val_targets = []
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    outputs = model(X_batch)
                    val_preds.extend(outputs.cpu().numpy().flatten())
                    val_targets.extend(y_batch.numpy().flatten())

            # Calculate metrics
            metrics_train = calculate_evaluation_metrics(
                np.array(train_targets), np.array(train_preds)
            )
            metrics_val = calculate_evaluation_metrics(
                np.array(val_targets), np.array(val_preds)
            )

            # Store results
            results[set_name] = {
                "model": model,
                "scaler": scaler,
                "feature_set_name": set_name,
                "features": features,
                "n_features": len(features),
                "lookback_days": lookback_days,
                "sequence_length": sequence_length,
                "train_metrics": metrics_train,
                "val_metrics": metrics_val,
            }

            # Check if this is the best
            val_score = metrics_val[metric]
            is_better = (
                (val_score < best_score) if metric != "R2" else (val_score > best_score)
            )

            if is_better:
                best_score = val_score
                best_feature_set_name = set_name
                best_model = model
                best_scaler = scaler
                best_features = features
                best_metrics = {"train": metrics_train, "val": metrics_val}

            # Print results
            if verbose:
                print(f"Feature Set: {set_name}")
                print(f"  Features: {len(features)}, Lookback: {lookback_days} days")
                print(f"  Training {metric}: {metrics_train[metric]:.6f}")
                print(f"  Validation {metric}: {metrics_val[metric]:.6f}")
                print()

        except Exception as e:
            if verbose:
                print(f"Feature Set: {set_name} - FAILED")
                print(f"  Error: {str(e)}\n")
            continue

    if verbose:
        print(f"{'=' * 80}")
        print(f"BEST FEATURE SET: {best_feature_set_name}")
        print(f"  Features: {len(best_features)}")
        print(
            f"  Lookback: {feature_sets[best_feature_set_name]['lookback_days']} days"
        )
        print(f"  Validation {metric}: {best_score:.6f}")
        print(f"{'=' * 80}\n")

    return {
        "best_model": best_model,
        "best_scaler": best_scaler,
        "best_feature_set_name": best_feature_set_name,
        "best_features": best_features,
        "best_metrics": best_metrics,
        "sequence_length": sequence_length,
        "all_results": results,
    }


def compare_feature_sets_results(
    all_results: dict, metric: str = "RMSE", task: str = "regression"
) -> pd.DataFrame:
    """
    Create a comparison DataFrame of all feature set results.
    Args:
        all_results (dict): Dictionary containing results for all feature sets.
        metric (str): The metric to compare (e.g., 'RMSE', 'MAE', 'R2', 'F1_Macro').
        task (str): The task type ('regression' or 'classification').
    Returns:
        pd.DataFrame: Comparison DataFrame sorted by validation metric.
    """

    comparison_data = []
    for set_name, result in all_results.items():
        comparison_data.append(
            {
                "Feature_Set": set_name,
                "N_Features": result["n_features"],
                "Lookback_Days": result["lookback_days"],
                f"Train_{metric}": result["train_metrics"][metric],
                f"Val_{metric}": result["val_metrics"][metric],
                "Overfit_Gap": result["val_metrics"][metric]
                - result["train_metrics"][metric],
            }
        )

    df = pd.DataFrame(comparison_data)

    # Sort by validation metric (if linear regression task: sort ascending for RMSE/MAE/MAPE, descending for R2, if classification task: sort descending for all metrics)
    if task == "classification":
        df = df.sort_values(f"Val_{metric}", ascending=False).reset_index(drop=True)
    else:
        ascending = metric != "R2"
        df = df.sort_values(f"Val_{metric}", ascending=ascending).reset_index(drop=True)

    return df


def calculate_classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    mapping: dict,
    y_pred_proba: Optional[pd.Series] = None,
) -> dict:
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true (pd.Series): True labels
        y_pred (pd.Series): Predicted labels
        mapping (dict): Mapping of class labels to integers (e.g., {"Low": 0, "Normal": 1, "High": 2})
        y_pred_proba (pd.Series, optional): Predicted probabilities (for AUC)

    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_Macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Recall_Macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1_Macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Precision_Weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "Recall_Weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "F1_Weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    # Add per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    for i, class_name in enumerate(mapping.keys()):
        metrics[f"Precision_{class_name}"] = precision_per_class[i]
        metrics[f"Recall_{class_name}"] = recall_per_class[i]
        metrics[f"F1_{class_name}"] = f1_per_class[i]

    return metrics


def plot_confusion_matrix(
    y_true: pd.Series, y_pred: pd.Series, mapping: dict, title: str = "Confusion Matrix"
) -> None:
    """
    Plot confusion matrix.
    Args:
        y_true (pd.Series): True labels
        y_pred (pd.Series): Predicted labels
        mapping (dict): Mapping of class labels to integers (e.g., {"Low": 0, "Normal": 1, "High": 2})
        title (str): Title for the plot
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(mapping.keys()),
        yticklabels=list(mapping.keys()),
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=list(mapping.keys())))


def train_classifier_with_feature_sets(
    model,
    model_name: str,
    feature_sets: dict,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    mapping: dict,
    metric: str = "F1_Macro",
    scale: bool = False,
    scaler: Optional[StandardScaler] = None,
    verbose: bool = True,
) -> dict:
    """
    Train a classification model with multiple feature sets.

    Args:
        model: Classification model instance
        model_name (str): Name of the model
        feature_sets (dict): Dictionary of feature sets
        X_train, y_train (pd.DataFrame, np.ndarray): Training data
        X_val, y_val (pd.DataFrame, np.ndarray): Validation data
        mapping (dict): Mapping of class labels to integers (e.g., {"Low": 0, "Normal": 1, "High": 2})
        metric (str): Metric to optimize ('Accuracy', 'F1_Macro', 'F1_Weighted')
        scale (bool): Whether to scale features
        scaler: Scaler instance if scale=True
        verbose (bool): Print progress

    Returns:
        dict: Best model, features, and metrics
    """
    results = {}
    best_score = float("-inf")  # Higher is better for classification metrics
    best_feature_set_name = None
    best_model = None
    best_scaler = None
    best_features = None
    best_metrics = None

    if scale and scaler is None:
        scaler = StandardScaler()

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"Training {model_name} with {len(feature_sets)} feature sets")
        print(f"Optimizing for: {metric}")
        print(f"Scaling: {'Yes' if scale else 'No'}")
        print(f"{'=' * 80}\n")

    for set_name, set_config in feature_sets.items():
        features = set_config["features"]
        lookback_days = set_config["lookback_days"]

        model_clone = clone(model)
        scaler_clone = clone(scaler) if scale else None

        try:
            # Select features
            X_train_subset = X_train[features]
            X_val_subset = X_val[features]

            # Scale if needed
            if scale:
                X_train_prepared = scaler_clone.fit_transform(X_train_subset)
                X_val_prepared = scaler_clone.transform(X_val_subset)
            else:
                X_train_prepared = X_train_subset
                X_val_prepared = X_val_subset

            # Train model
            model_clone.fit(X_train_prepared, y_train)

            # Predictions
            y_pred_train = model_clone.predict(X_train_prepared)
            y_pred_val = model_clone.predict(X_val_prepared)

            # Get probabilities if available
            y_pred_train_proba = None
            y_pred_val_proba = None
            if hasattr(model_clone, "predict_proba"):
                y_pred_train_proba = model_clone.predict_proba(X_train_prepared)
                y_pred_val_proba = model_clone.predict_proba(X_val_prepared)

            # Calculate metrics
            metrics_train = calculate_classification_metrics(
                y_train, y_pred_train, mapping, y_pred_train_proba
            )
            metrics_val = calculate_classification_metrics(
                y_val, y_pred_val, mapping, y_pred_val_proba
            )

            # Store results
            results[set_name] = {
                "model": model_clone,
                "scaler": scaler_clone,
                "feature_set_name": set_name,
                "features": features,
                "n_features": len(features),
                "lookback_days": lookback_days,
                "train_metrics": metrics_train,
                "val_metrics": metrics_val,
            }

            # Check if best
            val_score = metrics_val[metric]
            if val_score > best_score:
                best_score = val_score
                best_feature_set_name = set_name
                best_model = model_clone
                best_scaler = scaler_clone
                best_features = features
                best_metrics = {"train": metrics_train, "val": metrics_val}

            # Print results
            if verbose:
                print(f"Feature Set: {set_name}")
                print(f"  Features: {len(features)}, Lookback: {lookback_days} days")
                print(f"  Training {metric}: {metrics_train[metric]:.4f}")
                print(f"  Validation {metric}: {metrics_val[metric]:.4f}")
                print(f"  Val Accuracy: {metrics_val['Accuracy']:.4f}")
                print()

        except Exception as e:
            if verbose:
                print(f"Feature Set: {set_name} - FAILED")
                print(f"  Error: {str(e)}\n")
            continue

    if verbose:
        print(f"{'=' * 80}")
        print(f"BEST FEATURE SET: {best_feature_set_name}")
        print(f"  Features: {len(best_features)}")
        print(
            f"  Lookback: {feature_sets[best_feature_set_name]['lookback_days']} days"
        )
        print(f"  Validation {metric}: {best_score:.4f}")
        print(f"  Validation Accuracy: {best_metrics['val']['Accuracy']:.4f}")
        print(f"{'=' * 80}\n")

    return {
        "best_model": best_model,
        "best_scaler": best_scaler,
        "best_feature_set_name": best_feature_set_name,
        "best_features": best_features,
        "best_metrics": best_metrics,
        "all_results": results,
    }
