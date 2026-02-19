import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# PRICE PREDICTION MODELS


# Fixed Naive Forecaster Class
class FixedNaiveForecaster:
    """
    A simple naive forecaster that always predicts the last observed value.
    """

    def __init__(self) -> None:
        self.last_observation = None

    def fit(self, y: pd.Series) -> None:
        """
        Fit the model to the training data.
        Args:
            y (pd.Series): The target time series data.
        """
        self.last_observation = y.iloc[-1]

    def predict(self, n_periods: int) -> np.ndarray:
        """
        Predict the next n_periods using the last observed value.
        Args:
            n_periods (int): Number of periods to forecast.
        Returns:
            np.ndarray: Array of predicted values.
        """
        if self.last_observation is None:
            raise ValueError("The model must be fitted before predicting.")
        return np.full(n_periods, self.last_observation)


# Rolling Naive Forecaster Class
class RollingNaiveForecaster:
    """
    A rolling naive forecaster that uses actual values from the previous time step as predictions.
    """

    def __init__(self) -> None:
        self.last_train_observation = None

    def fit(self, y) -> None:
        """We don't need to fit anything for rolling naive forecaster"""
        pass

    def predict(self, y_actual_val: pd.Series) -> np.ndarray:
        """
        We just return the actual values as predictions
        Args:
            y_actual_val (pd.Series): Actual values from the validation set.
        Returns:
            np.ndarray: Array of predicted values.
        """
        return y_actual_val.to_numpy()


# Sequence Dataset Class
class SequenceDataset(Dataset):
    """
    Dataset for creating sequences from time series data.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int) -> None:
        """
        Initializes the SequenceDataset.
        Args:
            X (np.ndarray): Feature data.
            y (np.ndarray): Target data.
            seq_len (int): Length of the input sequences.
        """
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]
        y_val = self.y[idx + self.seq_len]
        return torch.tensor(x_seq), torch.tensor([y_val])


# LSTM Model Class
class LSTMForecaster(nn.Module):
    """
    LSTM Model for Time Series Forecasting
    """

    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, dropout: float
    ) -> None:
        """
        Initializes the LSTMForecaster model.
        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of features in the hidden state.
            num_layers (int): Number of recurrent layers.
            dropout (float): Dropout probability between layers.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )  # LSTM layer
        self.fc = nn.Linear(hidden_size, 1)  # Output layer

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden)
        last_hidden = lstm_out[:, -1, :]  # last timestep only
        return self.fc(last_hidden)  # (batch, 1)


# GRU Model Class
class GRUForecaster(nn.Module):
    """
    GRU Model for Time Series Forecasting
    """

    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, dropout: float
    ) -> None:
        """
        Initializes the GRUForecaster model.
        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of features in the hidden state.
            num_layers (int): Number of recurrent layers.
            dropout (float): Dropout probability between layers.
        """
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )  # GRU layer
        self.fc = nn.Linear(hidden_size, 1)  # Output layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden)
        last_hidden = gru_out[:, -1, :]  # last timestep only
        return self.fc(last_hidden)  # (batch, 1)


# BINARY CLASSIFICATION MODELS


class SequenceDatasetClassifier(Dataset):
    """
    PyTorch Dataset for sequence classification.
    """

    def __init__(self, X, y, sequence_length=30):
        """
        Args:
            X: Feature array (n_samples, n_features) - numpy array or similar
            y: Target array (n_samples,) - binary labels - numpy array or similar
            sequence_length: Number of time steps in each sequence
        """
        # Convert to numpy first if needed
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        # Store as numpy arrays first
        self.X_np = X.astype(np.float32)
        self.y_np = y.astype(np.float32)
        self.sequence_length = sequence_length

        # Calculate valid length
        self.valid_length = len(self.X_np) - self.sequence_length

        if self.valid_length <= 0:
            raise ValueError(
                f"Dataset too small. Length {len(self.X_np)} with sequence length "
                f"{self.sequence_length} results in {self.valid_length} samples."
            )

    def __len__(self):
        return self.valid_length

    def __getitem__(self, idx):
        if idx >= self.valid_length:
            raise IndexError(
                f"Index {idx} out of range for dataset of length {self.valid_length}"
            )

        # Get sequence of features
        X_seq = self.X_np[idx : idx + self.sequence_length]
        # Get target (next day's direction after the sequence)
        y_target = self.y_np[idx + self.sequence_length]

        # Convert to tensors only when needed
        return torch.FloatTensor(X_seq), torch.FloatTensor([y_target]).squeeze()


class LSTMClassifier(nn.Module):
    """
    LSTM-based binary classifier for price direction prediction.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """
        Initializes the LSTMClassifier model.
        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of features in the hidden state.
            num_layers (int): Number of recurrent layers.
            dropout (float): Dropout probability between layers.
        """
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)

        # Take the last time step
        last_output = lstm_out[:, -1, :]

        # Apply dropout and fully connected layer
        out = self.dropout(last_output)
        out = self.fc(out)
        out = self.sigmoid(out)

        return out.squeeze()


class GRUClassifier(nn.Module):
    """
    GRU-based binary classifier for price direction prediction.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """
        Initializes the GRUClassifier model.
        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of features in the hidden state.
            num_layers (int): Number of recurrent layers.
            dropout (float): Dropout probability between layers.
        """
        super(GRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size,).
        """
        # x shape: (batch_size, sequence_length, input_size)
        gru_out, _ = self.gru(x)

        # Take the last time step
        last_output = gru_out[:, -1, :]

        # Apply dropout and fully connected layer
        out = self.dropout(last_output)
        out = self.fc(out)
        out = self.sigmoid(out)

        return out.squeeze()


# VOLATILITY PREDICTION MODELS


class VolatilityDataset(Dataset):
    """
    Dataset for volatility prediction with sequences.
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, sequence_length: int = 30):
        """
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target values
            sequence_length (int): Number of time steps to use for prediction
        """
        self.X = X.values.astype(np.float32)
        self.y = y.values.astype(np.float32)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.X) - self.sequence_length + 1

    def __getitem__(self, idx):
        X_seq = self.X[idx : idx + self.sequence_length]
        y_val = self.y[idx + self.sequence_length - 1]

        return torch.FloatTensor(X_seq), torch.FloatTensor([y_val])


# ============================================================================
# NEURAL NETWORK MODELS
# ============================================================================


class VolatilityLSTM(nn.Module):
    """LSTM model for volatility prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super(VolatilityLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Take last output
        out = lstm_out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class VolatilityGRU(nn.Module):
    """GRU model for volatility prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super(VolatilityGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = gru_out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class AttentionLSTM(nn.Module):
    """LSTM with attention mechanism for volatility prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)

        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)

        # Attention weights
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)

        # Weighted sum
        context = torch.sum(attention_weights * lstm_out, dim=1)

        out = self.fc1(context)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
