"""JDBC helpers for Spark jobs that read/write the project's SQL Server."""

import os
import re
import sys

from dotenv import load_dotenv
from pyspark.sql import DataFrame, SparkSession


load_dotenv()

JDBC_DRIVER_CLASS = "com.microsoft.sqlserver.jdbc.SQLServerDriver"
JDBC_MAVEN_COORDINATE = "com.microsoft.sqlserver:mssql-jdbc:13.6.0.jre11"
FETCH_SIZE = "10000"
_TABLE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")


def create_spark_session(app_name: str) -> SparkSession:
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

    builder = (
        SparkSession.builder.master(os.getenv("SPARK_MASTER", "local[*]"))
        .appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.jars.packages", JDBC_MAVEN_COORDINATE)
        .config(
            "spark.sql.shuffle.partitions",
            os.getenv("SPARK_SHUFFLE_PARTITIONS", "16"),
        )
        .config("spark.ui.enabled", "false")
    )

    ivy_dir = os.getenv("SPARK_IVY_DIR")
    if ivy_dir:
        builder = builder.config("spark.jars.ivy", ivy_dir)

    return builder.getOrCreate()


def _jdbc_options() -> dict[str, str]:
    """Build connection options from environment variables, never source literals."""
    trusted = os.getenv("SQLSERVER_TRUSTED_CONNECTION", "no").lower() == "yes"
    if trusted:
        raise ValueError(
            "Spark JDBC in the Linux Airflow container expects SQL Server username/password. "
            "Windows trusted authentication needs additional Kerberos configuration."
        )

    required = [
        "SQLSERVER_HOST",
        "SQLSERVER_DATABASE",
        "SQLSERVER_USER",
        "SQLSERVER_PASSWORD",
    ]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise ValueError(f"Missing SQL Server environment variables: {', '.join(missing)}")

    host = os.environ["SQLSERVER_HOST"]
    port = os.getenv("SQLSERVER_PORT", "1433")
    database = os.environ["SQLSERVER_DATABASE"]
    url = (
        f"jdbc:sqlserver://{host}:{port};databaseName={database};"
        "encrypt=true;trustServerCertificate=true;loginTimeout=30"
    )

    return {
        "url": url,
        "driver": JDBC_DRIVER_CLASS,
        "user": os.environ["SQLSERVER_USER"],
        "password": os.environ["SQLSERVER_PASSWORD"],
    }


def read_sql_table(spark: SparkSession, table_name: str) -> DataFrame:
    """Read one fully-qualified project table, such as ``bronze.historical_weather``."""
    if not _TABLE_NAME_PATTERN.fullmatch(table_name):
        raise ValueError(f"Invalid SQL table name: {table_name!r}")

    return (
        spark.read.format("jdbc")
        .options(**_jdbc_options())
        .option("dbtable", table_name)
        .option("fetchsize", FETCH_SIZE)
        .load()
    )


def read_sql_query(spark: SparkSession, query: str) -> DataFrame:
    """Read a fixed SELECT query through JDBC, for filtered incremental reads."""
    statement = query.strip().rstrip(";")
    if not statement[:6].upper() == "SELECT" or ";" in statement:
        raise ValueError("JDBC subqueries must be one SELECT statement without a semicolon")

    return (
        spark.read.format("jdbc")
        .options(**_jdbc_options())
        .option("dbtable", f"({statement}) AS spark_source")
        .option("fetchsize", FETCH_SIZE)
        .load()
    )


def write_sql_table(df: DataFrame, table_name: str) -> None:
    """Append a DataFrame through JDBC without replacing the SQL table schema."""
    if not _TABLE_NAME_PATTERN.fullmatch(table_name):
        raise ValueError(f"Invalid SQL table name: {table_name!r}")

    (
        df.write.format("jdbc")
        .options(**_jdbc_options())
        .option("dbtable", table_name)
        .option("batchsize", "1000")
        .mode("append")
        .save()
    )


def overwrite_staging_table(df: DataFrame, table_name: str) -> None:
    """Replace only a table explicitly named as a staging table.

    Spark's JDBC overwrite can drop and recreate a SQL table. Restricting this
    helper to ``*_stage`` tables protects the warehouse tables and constraints.
    """
    if not _TABLE_NAME_PATTERN.fullmatch(table_name) or not table_name.endswith("_stage"):
        raise ValueError("JDBC overwrite is restricted to SQL tables ending in '_stage'")

    (
        df.write.format("jdbc")
        .options(**_jdbc_options())
        .option("dbtable", table_name)
        .option("batchsize", "1000")
        .mode("overwrite")
        .save()
    )
