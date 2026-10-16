from airflow.models import DagBag

d = DagBag(dag_folder="/opt/airflow/dags", include_examples=False)
dag = d.get_dag("historical_bootstrap")

print("DAG loaded:", dag is not None)
print("Import errors:", d.import_errors)

if dag:
    task = dag.get_task("fetch_historical")
    print("Callable:", task.python_callable)
    print("Python path in current DAG file:")
    with open("/opt/airflow/dags/dag_historical_bootstrap.py") as f:
        for line in f:
            if "PYTHON =" in line or "fetch_historical.py" in line:
                print(line.rstrip())
