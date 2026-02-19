import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import torch
import torch.nn as nn


from utils.model_development_classes import LSTMClassifier, GRUClassifier


def create_binary_target(df: pd.DataFrame, price_col: str = "Close") -> pd.Series:
    """
    Creates binary target: 1 if next day price goes up, 0 otherwise.

    Args:
        df (pd.DataFrame): DataFrame with price data.
        price_col (str): Column name for price. Defaults to "Close".

    Returns:
        pd.Series: Binary series (1 = up, 0 = down/flat).
    """
    return (df[price_col].shift(-1) > df[price_col]).astype(int)


def calculate_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_pred_proba: np.ndarray | pd.Series = None,
) -> dict:
    """
    Calculate classification metrics.

    Args:
        y_true (np.ndarray | pd.Series): Actual labels.
        y_pred (np.ndarray | pd.Series): Predicted labels.
        y_pred_proba (np.ndarray | pd.Series): Predicted probabilities (optional, for ROC-AUC).
            Defaults to None.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, F1 score, confusion matrix,
            classification report, and optionally ROC-AUC score.
    """
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1_Score": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred),
    }

    if y_pred_proba is not None:
        try:
            metrics["ROC_AUC"] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics["ROC_AUC"] = 0.0

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    title: str = "Confusion Matrix",
    figsize: tuple = (8, 6),
) -> None:
    """
    Plot confusion matrix heatmap.

    Args:
        y_true (np.ndarray | pd.Series): Actual labels.
        y_pred (np.ndarray | pd.Series): Predicted labels.
        title (str): Plot title. Defaults to "Confusion Matrix".
        figsize (tuple): Figure size. Defaults to (8, 6).
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Down (0)", "Up (1)"],
        yticklabels=["Down (0)", "Up (1)"],
        cbar=True,
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel("Actual", fontsize=12)
    plt.xlabel("Predicted", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_roc_curve(
    y_true: np.ndarray | pd.Series,
    y_pred_proba: np.ndarray | pd.Series,
    title: str = "ROC Curve",
    figsize: tuple = (8, 6),
) -> None:
    """
    Plot ROC curve.

    Args:
        y_true (np.ndarray | pd.Series): Actual labels.
        y_pred_proba (np.ndarray | pd.Series): Predicted probabilities.
        title (str): Plot title. Defaults to "ROC Curve".
        figsize (tuple): Figure size. Defaults to (8, 6).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=figsize)
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
    )
    plt.plot(
        [0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier"
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(
    model, feature_names: list, top_n: int = 20, figsize: tuple = (10, 8)
) -> None:
    """
    Plot feature importance from tree-based model.

    Args:
        model: Trained tree-based model with feature_importances_ attribute.
        feature_names (list): List of feature names.
        top_n (int): Number of top features to display. Defaults to 20.
        figsize (tuple): Figure size. Defaults to (10, 8).
    """
    importance = model.feature_importances_
    if top_n > len(feature_names):
        top_n = len(feature_names)
    indices = np.argsort(importance)[::-1][:top_n]

    plt.figure(figsize=figsize)
    plt.barh(range(top_n), importance[indices], color="steelblue")
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel("Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_top_classification_models(
    model_dict: dict, metric: str = "Accuracy", top_n: int = 5, figsize=(12, 8)
):
    """
    Plot top classification models by specified metric.

    Args:
        model_dict (dict): Dictionary of model results containing 'val_metrics' and optionally
            'train_metrics'.
        metric (str): Metric to rank by (Accuracy, F1_Score, ROC_AUC, etc.).
            Defaults to "Accuracy".
        top_n (int): Number of top models to display. Defaults to 5.
        figsize (tuple): Figure size. Defaults to (12, 8).
    """
    model_data = []

    for name, results in model_dict.items():
        if "val_metrics" in results:
            val_score = results["val_metrics"].get(metric, 0)
            train_score = results.get("train_metrics", {}).get(metric, val_score)
            model_data.append((name, train_score, val_score))

    if not model_data:
        print(f"No models found with {metric} metric.")
        return

    # Sort by validation score (higher is better for classification)
    sorted_models = sorted(model_data, key=lambda x: x[2], reverse=True)

    # Get top N models
    if len(sorted_models) < top_n:
        top_n = len(sorted_models)
    top_models = sorted_models[:top_n]

    # Unzip data
    names = [x[0] for x in top_models]
    train_scores = [x[1] for x in top_models]
    val_scores = [x[2] for x in top_models]

    # Create grouped bar chart
    y = np.arange(len(names))
    height = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    rects1 = ax.barh(
        y - height / 2,
        train_scores,
        height,
        label="Train",
        color="lightgray",
        edgecolor="grey",
    )

    val_colors = ["steelblue"] * len(val_scores)
    val_colors[0] = "darkgreen"  # Highlight best
    rects2 = ax.barh(
        y + height / 2, val_scores, height, label="Validation", color=val_colors
    )

    ax.set_xlabel(metric, fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    ax.set_title(
        f"Top {top_n} Models by {metric} (Train vs Val)", fontsize=14, fontweight="bold"
    )
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.legend()
    ax.invert_yaxis()

    # Add value labels
    def autolabel(rects):
        for rect in rects:
            width = rect.get_width()
            ax.annotate(
                f"{width:.3f}",
                xy=(width, rect.get_y() + rect.get_height() / 2),
                xytext=(3, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9,
            )

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.show()

    # Print rankings
    print(f"\n{'=' * 85}")
    print(f"Top {top_n} Models by {metric} (Sorted by Validation Performance)")
    print(f"{'=' * 85}")
    print(
        f"{'Rank':<5} {'Model Name':<35} {'Train ' + metric:<15} {'Val ' + metric:<15} {'Delta':<10}"
    )
    print(f"{'-' * 85}")

    for i, (name, t_score, v_score) in enumerate(top_models, 1):
        delta = t_score - v_score  # Shows overfitting
        print(f"{i:<5d} {name:<35s} {t_score:<15.4f} {v_score:<15.4f} {delta:<10.4f}")
    print(f"{'=' * 85}\n")


def evaluate_classifier_feature_set(
    model_name: str,
    features: list,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    random_seed: int = 42,
) -> tuple:
    """
    Evaluate a feature set using specified classifier.

    Args:
        model_name (str): Model type ("logistic_regression", "random_forest", "lightgbm").
        features (list): List of feature names to use.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.
        random_seed (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: Tuple containing trained model, training metrics dict, validation metrics dict,
            training predictions, validation predictions, and validation probabilities.
    """
    # Initialize model
    if model_name == "logistic_regression":
        model = LogisticRegression(random_state=random_seed)
    elif model_name == "random_forest":
        model = RandomForestClassifier(random_state=random_seed)
    elif model_name == "lightgbm":
        model = lgb.LGBMClassifier(random_state=random_seed)
    else:
        raise ValueError(
            "Unsupported model type. Choose from 'logistic_regression', 'random_forest', 'lightgbm'."
        )

    # Fit model
    model.fit(X_train[features], y_train)
    # Predictions
    train_pred = model.predict(X_train[features])
    val_pred = model.predict(X_val[features])
    val_pred_proba = model.predict_proba(X_val[features])[:, 1]

    # Calculate metrics
    train_metrics = calculate_classification_metrics(y_train, train_pred)
    val_metrics = calculate_classification_metrics(y_val, val_pred, val_pred_proba)

    return model, train_metrics, val_metrics, train_pred, val_pred, val_pred_proba


def train_nn_classifier(
    model_name: str,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    input_size: int,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    lr: float = 0.001,
    epochs: int = 100,
    device: torch.device = None,
    verbose: bool = True,
    print_every: int = 20,
):
    """
    Train LSTM/GRU classifier with specified hyperparameters.

    Args:
        model_name (str): Model architecture to use. Either 'LSTM' or 'GRU'.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        input_size (int): Number of input features.
        hidden_size (int): Hidden layer size for the neural network. Defaults to 64.
        num_layers (int): Number of recurrent layers. Defaults to 2.
        dropout (float): Dropout rate for regularization. Defaults to 0.2.
        lr (float): Learning rate for Adam optimizer. Defaults to 0.001.
        epochs (int): Number of training epochs. Defaults to 100.
        device (torch.device): Device to train on. Auto-detected if None. Defaults to None.
        verbose (bool): Whether to print training progress. Defaults to True.
        print_every (int): Print metrics every N epochs. Defaults to 20.

    Returns:
        dict: Dictionary containing trained model with best validation loss, training history
            with loss and metric values per epoch, best validation loss achieved during training,
            epoch number where best validation loss occurred, and device used for training.
    """
    # Device setup
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    if verbose:
        print(f"Using device: {device}")

    # Initialize model
    if model_name == "LSTM":
        model = LSTMClassifier(input_size, hidden_size, num_layers, dropout).to(device)
    elif model_name == "GRU":
        model = GRUClassifier(input_size, hidden_size, num_layers, dropout).to(device)
    else:
        raise ValueError("model_name must be 'LSTM' or 'GRU'")

    if verbose:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()  # Binary Cross Entropy for classification

    # Training loop
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Starting training for {epochs} epochs...")
        print(f"{'=' * 60}")

    history = defaultdict(list)
    best_val_loss = float("inf")
    best_epoch = 0
    best_model_state = None

    for epoch in range(epochs):
        # ============ Training Phase ============
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_actuals = []

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_predictions.extend(pred.detach().cpu().numpy())
            train_actuals.extend(batch_y.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)

        # ============ Validation Phase ============
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_actuals = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                pred = model(batch_X)
                loss = criterion(pred, batch_y)

                val_loss += loss.item()
                val_predictions.extend(pred.cpu().numpy())
                val_actuals.extend(batch_y.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)

        # ============ Calculate Metrics ============
        train_preds_binary = (np.array(train_predictions) > 0.5).astype(int)
        val_preds_binary = (np.array(val_predictions) > 0.5).astype(int)

        train_accuracy = accuracy_score(train_actuals, train_preds_binary)
        val_accuracy = accuracy_score(val_actuals, val_preds_binary)

        train_precision = precision_score(
            train_actuals, train_preds_binary, zero_division=0
        )
        val_precision = precision_score(val_actuals, val_preds_binary, zero_division=0)

        train_recall = recall_score(train_actuals, train_preds_binary, zero_division=0)
        val_recall = recall_score(val_actuals, val_preds_binary, zero_division=0)

        train_f1 = f1_score(train_actuals, train_preds_binary, zero_division=0)
        val_f1 = f1_score(val_actuals, val_preds_binary, zero_division=0)

        # Store history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        history["train_precision"].append(train_precision)
        history["val_precision"].append(val_precision)
        history["train_recall"].append(train_recall)
        history["val_recall"].append(val_recall)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        # Track best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()

        # Print progress
        if verbose and (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"  Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            print(f"  Train Acc: {train_accuracy:.4f} | Val Acc: {val_accuracy:.4f}")
            print(
                f"  Train Precision: {train_precision:.4f} | Val Precision: {val_precision:.4f}"
            )
            print(f"  Train Recall: {train_recall:.4f} | Val Recall: {val_recall:.4f}")
            print(f"  Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

    # Load best model
    model.load_state_dict(best_model_state)

    if verbose:
        print(f"{'=' * 60}")
        print("Training complete!")
        print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
        print(f"{'=' * 60}")

    return {
        "model": model,
        "history": dict(history),
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "device": device,
    }
