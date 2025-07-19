# ml_scripts/training/train_with_mlflow.py
import os
from datetime import datetime
from dotenv import load_dotenv
import mlflow
import mlflow.sklearn
from train import train_model

load_dotenv(dotenv_path="/home/psycho/mlops-project/.env")

def train_and_log(train_path: str, output_dir: str, ts: str):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "default"))

    with mlflow.start_run(run_name="LinearRegression_Train"):
        model_path, model = train_model(train_path, output_dir, ts)

        mlflow.log_param("fit_intercept", model.fit_intercept)
        mlflow.log_param("n_jobs", model.n_jobs)
        mlflow.sklearn.log_model(model, artifact_path="model_artifact")

        print(f"[MLflow] 모델 로깅 완료: {model_path}")
        return model_path

if __name__ == "__main__":
    raw_path = os.environ.get("RAW_DATA_PATH")
    output_dir = os.environ.get("OUTPUT_DIR")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not raw_path or not output_dir:
        raise ValueError("환경변수 RAW_DATA_PATH 또는 OUTPUT_DIR가 설정되지 않았습니다!")

    result = train_and_log(raw_path, output_dir, ts)
    print(f"[테스트 실행] 반환된 경로: {result}")
