import os
import mlflow
import mlflow.sklearn


def log_models_to_mlflow(models: dict[str, object]) -> dict[str, str]:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "default"))

    result: dict[str, str] = {}
    for name, model in models.items():
        with mlflow.start_run(run_name=f"{name}_Train") as run:
            mlflow.log_param("model_type", name)
            mlflow.sklearn.log_model(model, artifact_path="model_artifact")
            result[name] = run.info.run_id
            print(f"[MLflow] {name} 로깅 완료 → run_id={run.info.run_id}")

    return result
