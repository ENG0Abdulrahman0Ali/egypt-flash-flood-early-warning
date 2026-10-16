import os
from datetime import datetime, timedelta

from airflow.decorators import dag, task

PROJECT_DIR = os.environ.get("PROJECT_DIR", "/opt/airflow/project")
PYTHON = "/opt/uv-venv/env/bin/python"

default_args = {
    "owner": "mohamed_tarek",
    "retries": 2,
    "retry_delay": timedelta(minutes=3),
}


@dag(
    dag_id="hourly_ingestion",
    description="Hourly: Kafka -> Bronze -> Silver (Spark) -> Gold risk (Spark)",
    schedule="@hourly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1, 
    is_paused_upon_creation=False,
    default_args=default_args,
    tags=["flood-warning", "ingestion", "spark", "hourly"],
)
def hourly_ingestion():

    @task.bash
    def ensure_topic() -> str:
        return f"cd {PROJECT_DIR} && {PYTHON} ingestion/create_topic.py"

    @task.bash
    def producer_live() -> str:
        return f"cd {PROJECT_DIR} && {PYTHON} ingestion/producer_live.py"

    @task.bash
    def kafka_to_bronze_consumer() -> str:
        return f"cd {PROJECT_DIR} && {PYTHON} ingestion/kafka_to_bronze_consumer.py"

    @task.bash(execution_timeout=timedelta(minutes=40))
    def live_incremental_job() -> str:
        return f"cd {PROJECT_DIR} && {PYTHON} spark/live_incremental_job.py"

    @task.bash(execution_timeout=timedelta(minutes=40))
    def silver_to_gold_load() -> str:
        return f"cd {PROJECT_DIR} && {PYTHON} spark/silver_to_gold_load.py"

    (
        ensure_topic()
        >> producer_live()
        >> kafka_to_bronze_consumer()
        >> live_incremental_job()
        >> silver_to_gold_load()
    )


hourly_ingestion()
