FROM apache/airflow:2.10.2-python3.11

USER root
# ODBC (for pyodbc) + Java 17 (required by PySpark) in one layer
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl gnupg unixodbc-dev openjdk-17-jre-headless \
 && curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
 && curl -fsSL https://packages.microsoft.com/config/debian/12/prod.list > /etc/apt/sources.list.d/mssql-release.list \
 && apt-get update && ACCEPT_EULA=Y apt-get install -y msodbcsql18 \
 && ln -s "$(dirname "$(dirname "$(readlink -f "$(which java)")")")" /opt/java \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir /opt/uv-venv && chown airflow /opt/uv-venv

# Architecture-independent JAVA_HOME (amd64 / arm64)
ENV JAVA_HOME=/opt/java

USER airflow
RUN pip install --no-cache-dir -U uv
