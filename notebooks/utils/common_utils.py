import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from collections import defaultdict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.model_development_classes import LSTMForecaster, GRUForecaster


def time_series_train_val_test_split(
    data: pd.DataFrame, train_size: float = 0.7, val_size: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits time series data into training, validation, and test sets.

    Args:
        data (pd.DataFrame): The time series data to split.
        train_size (float): Proportion of data to use for training.
        val_size (float): Proportion of data to use for validation.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing the training, validation, and test sets.
    """
    n = len(data)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))

    train = data.iloc[:train_end]
    val = data.iloc[train_end:val_end]
    test = data.iloc[val_end:]

    return train, val, test


def calculate_evaluation_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    include_r2_score: bool = False,
) -> dict:
    """
    Calculate evaluation metrics for time series forecasting.

    Args:
        y_true (np.ndarray | pd.Series): Actual values.
        y_pred (np.ndarray | pd.Series): Predicted values.
        include_r2_score (bool): Whether to include R^2 score in the metrics. Defaults to False.

    Returns:
        dict: Dictionary containing MAE, RMSE, MAPE, and Directional Accuracy.
    """

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    directional_accuracy = mean_directional_accuracy(y_true, y_pred)

    results = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Directional_Accuracy": directional_accuracy,
    }

    if include_r2_score:
        r2 = r2_score(y_true, y_pred)
        results["R2_Score"] = r2

    return results


def mean_directional_accuracy(
    actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series
) -> float:
    """
    Calculate the Mean Directional Accuracy (MDA) between actual and predicted values.

    Args:
        actual (np.ndarray | pd.Series): Actual values.
        predicted (np.ndarray | pd.Series): Predicted values.

    Returns:
        float: Mean Directional Accuracy score.
    """
    # Convert to numpy arrays to strip Pandas indices
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    # Calculate directions relative to previous actual price
    actual_diff = np.sign(actual[1:] - actual[:-1])
    pred_diff = np.sign(predicted[1:] - actual[:-1])

    # Return the mean of correct directions
    return np.mean((actual_diff == pred_diff).astype(int)) * 100


def plot_forecasts(
    dates: pd.Series | np.ndarray,
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    title: str = "Forecast vs Actual",
    x_label: str = "Date",
    y_label: str = "Value",
) -> None:
    """
    Plot the actual vs predicted values for time series forecasting.

    Args:
        dates (pd.Series | np.ndarray): Dates corresponding to the values.
        y_true (pd.Series | np.ndarray): Actual values.
        y_pred (pd.Series | np.ndarray): Predicted values.
        title (str): Title of the plot. Defaults to "Forecast vs Actual".
        x_label (str): Label for the x-axis. Defaults to "Date".
        y_label (str): Label for the y-axis. Defaults to "Value".
    """

    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_true, label="Actual", color="blue")
    plt.plot(dates, y_pred, label="Predicted", color="orange")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def plot_forecasts_with_biggest_mistakes(
    dates: pd.Series | np.ndarray,
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    title: str = "Forecast vs Actual",
    highlight_top_n: int = 10,
    show_error_values: bool = True,
    x_label: str = "Date",
    y_label: str = "Value",
) -> None:
    """
    Plot actual vs predicted values and highlight the top N biggest prediction mistakes.

    Args:
        dates (pd.Series | np.ndarray): Dates corresponding to the values.
        y_true (pd.Series | np.ndarray): Actual values.
        y_pred (pd.Series | np.ndarray): Predicted values.
        title (str): Title of the plot. Defaults to "Forecast vs Actual".
        highlight_top_n (int): Number of biggest mistakes to highlight. Defaults to 10.
        show_error_values (bool): Whether to show error values as text annotations.
            Defaults to True.
        x_label (str): Label for the x-axis. Defaults to "Date".
        y_label (str): Label for the y-axis. Defaults to "Value".
    """

    # Convert to numpy arrays for easier manipulation
    dates = np.array(dates)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate absolute errors
    errors = np.abs(y_true - y_pred)

    # Find indices of top N biggest mistakes
    top_n_indices = np.argsort(errors)[-highlight_top_n:][::-1]

    # Create the plot
    plt.figure(figsize=(14, 7))

    # Plot actual and predicted values
    plt.plot(dates, y_true, label="Actual", color="blue", linewidth=1.5, alpha=0.8)
    plt.plot(dates, y_pred, label="Predicted", color="orange", linewidth=1.5, alpha=0.8)

    # Highlight top N mistakes
    for idx in top_n_indices:
        # Draw vertical line from predicted to actual
        plt.plot(
            [dates[idx], dates[idx]],
            [y_true[idx], y_pred[idx]],
            color="red",
            linewidth=2,
            alpha=0.6,
            linestyle="--",
        )

        # Mark the points
        plt.scatter(
            dates[idx],
            y_true[idx],
            color="blue",
            s=100,
            zorder=5,
            edgecolors="red",
            linewidths=2,
        )
        plt.scatter(
            dates[idx],
            y_pred[idx],
            color="orange",
            s=100,
            zorder=5,
            edgecolors="red",
            linewidths=2,
        )

        # Optionally show error values
        if show_error_values:
            error_value = errors[idx]
            mid_y = (y_true[idx] + y_pred[idx]) / 2

            plt.annotate(
                f"Error: {error_value:.2f}",
                xy=(dates[idx], mid_y),
                xytext=(10, 0),
                textcoords="offset points",
                fontsize=9,
                color="red",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                arrowprops=dict(arrowstyle="->", color="red", lw=1),
            )

    plt.title(
        f"{title}\n(Top {highlight_top_n} biggest errors highlighted in red)",
        fontsize=14,
    )
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Print summary of top mistakes
    print(f"\n{'=' * 60}")
    print(f"Top {highlight_top_n} Biggest Prediction Errors:")
    print(f"{'=' * 60}")
    for i, idx in enumerate(top_n_indices, 1):
        print(
            f"{i}. Date: {dates[idx]}, "
            f"Actual: {y_true[idx]:.2f}, "
            f"Predicted: {y_pred[idx]:.2f}, "
            f"Error: {errors[idx]:.2f}"
        )


def plot_learning_curves(history: dict) -> None:
    """
    Plot comprehensive learning curves.

    Args:
        history (dict): Dictionary containing training and validation metrics per epoch.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs_range = range(1, len(history["train_loss"]) + 1)

    # Plot 1: Loss
    axes[0].plot(
        epochs_range,
        history["train_loss"],
        label="Train Loss",
        marker="o",
        markersize=3,
    )
    axes[0].plot(
        epochs_range, history["val_loss"], label="Val Loss", marker="s", markersize=3
    )
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss (MSE)", fontsize=12)
    axes[0].set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: RMSE
    axes[1].plot(
        epochs_range,
        history["train_rmse"],
        label="Train RMSE",
        marker="o",
        markersize=3,
    )
    axes[1].plot(
        epochs_range, history["val_rmse"], label="Val RMSE", marker="s", markersize=3
    )
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("RMSE", fontsize=12)
    axes[1].set_title("Training and Validation RMSE", fontsize=14, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: MAE
    axes[2].plot(
        epochs_range, history["train_mae"], label="Train MAE", marker="o", markersize=3
    )
    axes[2].plot(
        epochs_range, history["val_mae"], label="Val MAE", marker="s", markersize=3
    )
    axes[2].set_xlabel("Epoch", fontsize=12)
    axes[2].set_ylabel("MAE", fontsize=12)
    axes[2].set_title("Training and Validation MAE", fontsize=14, fontweight="bold")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_top_models(
    results_dict: dict, metric="RMSE", top_n=10, figsize=(12, 6)
) -> None:
    """
    Plot a bar chart of the top N models for a selected metric.

    Args:
        results_dict (dict): Dictionary with model results containing 'metrics' key.
        metric (str): The metric to compare ('MAE', 'RMSE', 'MAPE', 'Directional_Accuracy').
            Defaults to "RMSE".
        top_n (int): Number of top models to display. Defaults to 10.
        figsize (tuple): Figure size (width, height). Defaults to (12, 6).
    """
    # Extract metric values for all models
    model_scores = {}
    for model_name, results in results_dict.items():
        if "val_metrics" in results and metric in results["val_metrics"]:
            model_scores[model_name] = results["val_metrics"][metric]

    if not model_scores:
        print(f"No models found with metric: {metric}")
        return

    # Sort models by metric
    # For Directional Accuracy, higher is better; for others, lower is better
    reverse = metric == "Directional_Accuracy"
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=reverse)

    # Get top N models
    if len(sorted_models) < top_n:
        top_n = len(sorted_models)
    top_models = sorted_models[:top_n]
    model_names = [name for name, _ in top_models]
    scores = [score for _, score in top_models]

    # Create bar chart
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(model_names, scores, color="steelblue")

    # Highlight the best model
    bars[0].set_color("darkgreen")

    ax.set_xlabel(metric, fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    ax.set_title(f"Top {top_n} Models by {metric}", fontsize=14, fontweight="bold")
    ax.invert_yaxis()  # Best model at the top

    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score, i, f" {score:.4f}", va="center", fontsize=10)

    plt.tight_layout()
    plt.show()

    # Print the rankings
    print(f"\n{'=' * 60}")
    print(f"Top {top_n} Models by {metric}")
    print(f"{'=' * 60}")
    for i, (name, score) in enumerate(top_models, 1):
        print(f"{i:2d}. {name:30s} {metric}: {score:.4f}")
    print(f"{'=' * 60}\n")


def plot_top_models_train_val(
    results_dict: dict, metric="RMSE", top_n=10, figsize=(14, 8)
) -> None:
    """
    Plot a grouped bar chart of the top N models comparing Train vs Validation metrics.

    Args:
        results_dict (dict): Dictionary where keys are model names and values contain
            'train_metrics' and 'val_metrics'.
        metric (str): The metric to compare ('MAE', 'RMSE', 'MAPE', 'Directional_Accuracy').
            Defaults to "RMSE".
        top_n (int): Number of top models to display. Defaults to 10.
        figsize (tuple): Figure size (width, height). Defaults to (14, 8).
    """
    model_data = []

    # 1. Extract metric values for all models
    for model_name, results in results_dict.items():
        # Safely get validation score
        val_score = None
        if "val_metrics" in results and metric in results["val_metrics"]:
            val_score = results["val_metrics"][metric]

        # Safely get train score
        train_score = None
        if "train_metrics" in results and metric in results["train_metrics"]:
            train_score = results["train_metrics"][metric]

        # We only care if we have a validation score to rank by
        if val_score is not None:
            # Handle missing train score just in case
            t_score = train_score if train_score is not None else 0
            model_data.append((model_name, t_score, val_score))

    if not model_data:
        print(f"No models found with metric: {metric}")
        return

    # 2. Sort models by VALIDATION metric
    # For Directional Accuracy, higher is better; for others, lower is better
    reverse = metric == "Directional_Accuracy"
    # Sort key is the 3rd element (val_score)
    sorted_models = sorted(model_data, key=lambda x: x[2], reverse=reverse)

    # 3. Get top N models
    if len(sorted_models) < top_n:
        top_n = len(sorted_models)
    top_models = sorted_models[:top_n]

    # Unzip into lists for plotting
    names = [x[0] for x in top_models]
    train_scores = [x[1] for x in top_models]
    val_scores = [x[2] for x in top_models]

    # 4. Create Grouped Bar Chart
    y = np.arange(len(names))  # Label locations
    height = 0.35  # Width of the bars

    fig, ax = plt.subplots(figsize=figsize)

    # Plot Train bars (offset up)
    rects1 = ax.barh(
        y - height / 2,
        train_scores,
        height,
        label="Train",
        color="lightgray",
        edgecolor="grey",
    )
    # Plot Val bars (offset down) - Highlight the best one
    val_colors = ["steelblue"] * len(val_scores)
    val_colors[0] = "darkgreen"  # Highlight top 1
    rects2 = ax.barh(
        y + height / 2, val_scores, height, label="Validation", color=val_colors
    )

    # Formatting
    ax.set_xlabel(metric, fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    ax.set_title(
        f"Top {top_n} Models by {metric} (Train vs Val)", fontsize=14, fontweight="bold"
    )
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.legend()
    ax.invert_yaxis()  # Best model at the top

    # 5. Add value labels on bars
    def autolabel(rects):
        """Attach a text label to the right of each bar displaying its height."""
        for rect in rects:
            width = rect.get_width()
            ax.annotate(
                f"{width:.4f}",
                xy=(width, rect.get_y() + rect.get_height() / 2),
                xytext=(3, 0),  # 3 points horizontal offset
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9,
            )

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.show()

    # 6. Print the rankings
    print(f"\n{'=' * 85}")
    print(f"Top {top_n} Models by {metric} (Sorted by Validation Performance)")
    print(f"{'=' * 85}")
    print(
        f"{'Rank':<5} {'Model Name':<35} {'Train ' + metric:<15} {'Val ' + metric:<15} {'Delta':<10}"
    )
    print(f"{'-' * 85}")

    for i, (name, t_score, v_score) in enumerate(top_models, 1):
        # Delta shows how much worse val is than train (Overfitting indicator)
        delta = v_score - t_score
        print(f"{i:<5d} {name:<35s} {t_score:<15.4f} {v_score:<15.4f} {delta:<10.4f}")
    print(f"{'=' * 85}\n")


def train_nn_model(
    model_name: str,
    train_loader,
    val_loader,
    input_size,
    hidden_size=64,
    num_layers=2,
    dropout=0.2,
    lr=0.001,
    epochs=100,
    device=None,
    verbose=True,
    print_every=20,
) -> dict:
    """
    Train LSTM or GRU model with specified hyperparameters.

    Args:
        model_name (str): Model architecture to use. Either 'LSTM' or 'GRU'.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
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
        model = LSTMForecaster(input_size, hidden_size, num_layers, dropout).to(device)
    elif model_name == "GRU":
        model = GRUForecaster(input_size, hidden_size, num_layers, dropout).to(device)
    else:
        raise ValueError("model_name must be 'LSTM' or 'GRU'")

    if verbose:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

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
            train_predictions.append(pred.detach().cpu().numpy())
            train_actuals.append(batch_y.cpu().numpy())

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
                val_predictions.append(pred.cpu().numpy())
                val_actuals.append(batch_y.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)

        # ============ Calculate Metrics ============
        train_preds_flat = np.concatenate(train_predictions).flatten()
        train_actuals_flat = np.concatenate(train_actuals).flatten()
        val_preds_flat = np.concatenate(val_predictions).flatten()
        val_actuals_flat = np.concatenate(val_actuals).flatten()

        train_rmse = np.sqrt(np.mean((train_actuals_flat - train_preds_flat) ** 2))
        train_mae = np.mean(np.abs(train_actuals_flat - train_preds_flat))
        val_rmse = np.sqrt(np.mean((val_actuals_flat - val_preds_flat) ** 2))
        val_mae = np.mean(np.abs(val_actuals_flat - val_preds_flat))

        # Store history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_rmse"].append(train_rmse)
        history["val_rmse"].append(val_rmse)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)

        # Track best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()

        # Print progress
        if verbose and (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"  Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            print(f"  Train RMSE: {train_rmse:.6f} | Val RMSE: {val_rmse:.6f}")
            print(f"  Train MAE: {train_mae:.6f} | Val MAE: {val_mae:.6f}")

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
