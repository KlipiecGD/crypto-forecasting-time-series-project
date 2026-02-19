import sys
import os

# Add the project root directory to the Python path to ensure imports work correctly in Airflow
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.python import BranchPythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from datetime import datetime, timedelta

from src.fetch_data.historical_data import download_historical_data
from src.monitoring.performance_monitoring import analyze_performance
from src.pipelines.training_pipeline import run_training_pipeline
from src.sagemaker_deployment.deploy_all import deploy_to_sagemaker

from src.config.config import config

default_args = {
    "owner": config.get_dag_config.get("owner", "airflow"),
    "retries": config.get_dag_config.get("retries", 1),
    "retry_delay": timedelta(minutes=config.get_dag_config.get("retry_delay", 5)),
}


def monitoring_task(**context) -> None:
    """Run performance monitoring and push degradation result to XCom."""
    degradation = analyze_performance()
    context["ti"].xcom_push(key="degradation_detected", value=degradation)


def branch_on_degradation(**context) -> str:
    """
    Branch task based on whether degradation was detected.
    Returns:
        str: Task ID to follow based on degradation result - either "download_data" or "no_degradation"
    """
    degradation = context["ti"].xcom_pull(
        task_ids="run_monitoring", key="degradation_detected"
    )
    return "download_data" if degradation else "no_degradation"


def download_task(**context) -> None:
    """Download historical data and push results to XCom."""
    has_new_data, path = download_historical_data()
    context["ti"].xcom_push(key="has_new_data", value=has_new_data)
    context["ti"].xcom_push(key="data_path", value=path)


def branch_on_new_data(**context) -> str:
    """
    Branch task based on whether new data was found.
    Returns:
        str: Task ID to follow based on new data availability - either "run_training" or "no_new_data"
    """
    has_new_data = context["ti"].xcom_pull(task_ids="download_data", key="has_new_data")
    return "run_training" if has_new_data else "no_new_data"


def training_task(**context) -> None:
    """Run the training pipeline using the downloaded data."""
    data_path = context["ti"].xcom_pull(task_ids="download_data", key="data_path")
    run_training_pipeline(
        data_path=data_path, deploy=False
    )  # We don't deploy here, we handle deployment separately in the DAG


# Get configuration from config
dag_id = config.get_dag_config.get("dag_id", "volatility_retraining_pipeline")
description = config.get_dag_config.get(
    "description", "Weekly retraining and deployment pipeline"
)
catchup = config.get_dag_config.get("catchup", False)
tags = config.get_dag_config.get("tags", ["retraining", "deployment"])


with DAG(
    dag_id=dag_id,
    default_args=default_args,
    description=description,
    schedule=timedelta(weeks=1),
    start_date=datetime(2026, 2, 1),
    catchup=catchup,
    tags=tags,
) as dag:
    run_monitoring_task = PythonOperator(
        task_id="run_monitoring",
        python_callable=monitoring_task,
    )

    check_degradation_task = BranchPythonOperator(
        task_id="check_degradation",
        python_callable=branch_on_degradation,
    )

    download_data_task = PythonOperator(
        task_id="download_data",
        python_callable=download_task,
    )

    check_new_data_task = BranchPythonOperator(
        task_id="check_new_data",
        python_callable=branch_on_new_data,
    )

    run_training_task = PythonOperator(
        task_id="run_training",
        python_callable=training_task,
    )

    deploy_task = PythonOperator(
        task_id="deploy_to_sagemaker",
        python_callable=deploy_to_sagemaker,
        op_kwargs={
            "rebuild_image": False
        },  # We assume the image is already built and pushed during the training pipeline
    )

    no_degradation_task = EmptyOperator(task_id="no_degradation")
    no_new_data_task = EmptyOperator(task_id="no_new_data")
    end_task = EmptyOperator(
        task_id="end",
        trigger_rule="none_failed_min_one_success",
    )

    (
        run_monitoring_task
        >> check_degradation_task
        >> [download_data_task, no_degradation_task]
    )
    download_data_task >> check_new_data_task >> [run_training_task, no_new_data_task]
    run_training_task >> deploy_task >> end_task
    no_degradation_task >> end_task
    no_new_data_task >> end_task
