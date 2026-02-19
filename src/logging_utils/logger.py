import logging
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from src.config.config import config

def setup_logger(
    name: str = "app",
    level: int = logging.DEBUG,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    module: Optional[str] = None,
) -> logging.Logger:
    """
    Configure and return a global logger instance.
    If ENABLE_CLOUD_LOGGING=true, also logs to AWS CloudWatch.

    Args:
        name (str): Logger name, default is "app"
        level (int): Logging level, default is logging.DEBUG
        log_file (str): Optional path to log file
        format_string (str): Custom format string for log messages
        module (str): Module name for CloudWatch log stream separation
                      e.g. "inference", "ui", "data", "training"

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    logger.propagate = False

    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s"

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # CloudWatch handler (only on EB when ENABLE_CLOUD_LOGGING=true)
    _add_cloudwatch_handler(logger, formatter, module)

    return logger


def _add_cloudwatch_handler(
    logger: logging.Logger,
    formatter: logging.Formatter,
    module: Optional[str] = None,
) -> None:
    """
    Add CloudWatch handler to logger if ENABLE_CLOUD_LOGGING is set.

    Args:
        logger (logging.Logger): Logger to add handler to
        formatter (logging.Formatter): Formatter to use
        module (Optional[str]): Module name for log stream
                                Maps to /volatility-predictor/{module}
    """
    import os

    if os.getenv("ENABLE_CLOUD_LOGGING", "false").lower() != "true":
        return

    try:
        import boto3
        import watchtower

        module_name = module or "app"
        log_group_prefix = config.get_logging_config.get(
            "log_group_prefix", "volatility-predictor"
        )

        log_group = f"{log_group_prefix}/{module_name}"

        boto3_client = boto3.client(
            "logs", region_name=os.getenv("CLOUDWATCH_REGION", "eu-north-1")
        )

        cw_handler = watchtower.CloudWatchLogHandler(
            log_group=log_group,
            boto3_client=boto3_client,
            stream_name="{strftime:%Y-%m-%d}",  # one stream per day
        )
        cw_handler.setFormatter(formatter)
        logger.addHandler(cw_handler)

        logger.info(f"CloudWatch logging enabled -> {log_group}")

    except Exception as e:
        logger.warning(f"CloudWatch logging setup failed: {e}")


# Initialize the global logger
logger = setup_logger(
    name="app",
    level=logging.DEBUG,
    module="app",
)
