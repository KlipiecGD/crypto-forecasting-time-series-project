import logging
import mlflow # We need to import mlflow here to ensure logging works correctly
from src.logging_utils.logger import setup_logger


# /inference
inference_logger = setup_logger(
    name="inference",
    level=logging.DEBUG,
    module="inference",
)

# /data
data_logger = setup_logger(
    name="data",
    level=logging.DEBUG,
    module="data",
)

# /training
training_logger = setup_logger(
    name="training",
    level=logging.DEBUG,
    module="training",
)

# /deployment
deployment_logger = setup_logger(
    name="deployment",
    level=logging.DEBUG,
    module="deployment",
)

# /monitoring
monitoring_logger = setup_logger(
    name="monitoring",
    level=logging.DEBUG,
    module="monitoring",
)

# /ui
ui_logger = setup_logger(
    name="ui",
    level=logging.DEBUG,
    module="ui",
)
