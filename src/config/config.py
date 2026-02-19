import yaml
from pathlib import Path
from typing import Any, Optional


class Config:
    """Configuration class that loads and provides access to config values."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        """
        Initialize configuration from YAML file.

        Args:
            config_path (str): Path to the config YAML file
        """
        # Determine the location of this file (src/config/config.py)
        current_file = Path(__file__)

        # Calculate Project Root: src/config/ -> src/ -> Project Root
        self.project_root = current_file.parent.parent.parent

        if config_path:
            config_file = Path(config_path)
        else:
            # Dynamically get the path relative to this python file
            config_file = current_file.parent / "config.yaml"

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file, "r") as f:
            self._config = yaml.safe_load(f)

    def get(self, key_path: str, default: Optional[Any] = None) -> Optional[Any]:
        """
        Get a configuration value using dot notation.

        Args:
            key_path (str): Dot-separated path to config value (e.g., "loader.document_path")
            default: Default value if key not found

        Returns:
                Configuration value or default

        """
        keys = key_path.split(".")
        value = self._config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    @property
    def get_historical_data_config(self) -> dict:
        """Get historical data configuration section."""
        return self._config.get("historical_data", {})

    @property
    def get_forecast_config(self) -> dict:
        """Get forecast configuration section."""
        return self._config.get("forecast", {})

    @property
    def get_live_data_config(self) -> dict:
        """Get live data configuration section."""
        return self._config.get("live_data", {})

    @property
    def get_pipeline_config(self) -> dict:
        """Get pipeline configuration section."""
        return self._config.get("pipeline", {})

    @property
    def get_transformations_config(self) -> dict:
        """Get transformations configuration section."""
        return self.get_pipeline_config.get("transformations", {})

    @property
    def get_volatility_transformations_config(self) -> dict:
        """Get volatility transformations configuration section."""
        return self.get_transformations_config.get("volatility", {})

    @property
    def get_model_config(self) -> dict:
        """Get model configuration section."""
        return self._config.get("model", {})

    @property
    def get_hyperparameters_config(self) -> dict:
        """Get hyperparameters configuration section."""
        return self.get_model_config.get("hyperparameters", {})

    @property
    def get_thresholds_config(self) -> dict:
        """Get thresholds configuration section."""
        return self.get_model_config.get("thresholds", {})

    @property
    def get_sagemaker_deployment_config(self) -> dict:
        """Get SageMaker deployment configuration section."""
        return self._config.get("sagemaker_deployment", {})

    @property
    def get_ui_config(self) -> dict:
        """Get UI configuration section."""
        return self._config.get("ui", {})

    @property
    def get_elastic_beanstalk_deployment_config(self) -> dict:
        """Get Elastic Beanstalk deployment configuration section."""
        return self._config.get("elastic_beanstalk_deployment", {})

    @property
    def get_mlflow_config(self) -> dict:
        """Get MLflow configuration section."""
        return self._config.get("mlflow", {})

    @property
    def get_monitoring_config(self) -> dict:
        """Get monitoring configuration section."""
        return self._config.get("monitoring", {})

    @property
    def get_dag_config(self) -> dict:
        """Get DAG configuration section."""
        return self._config.get("dag", {})

    @property
    def get_logging_config(self) -> dict:
        """Get logging configuration section."""
        return self._config.get("logging", {})


config = Config()
