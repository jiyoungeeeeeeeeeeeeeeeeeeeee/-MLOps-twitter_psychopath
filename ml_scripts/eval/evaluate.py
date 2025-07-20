import mlflow
import pandas as pd
import numpy as np
# import joblib
from sklearn.metrics import r2_score , mean_squared_error , mean_absolute_error
from typing import Dict

def evaluate_model(run_ids: dict[str, str], test_path: str, target_col: str = "psychopathy") -> dict[str, dict[str, float]]:
    
    df = pd.read_csv(test_path)
    X = df.drop(target_col , axis = 1)
    y = df[target_col]

    metrics: dict[str, dict[str, float]] = {}

    for name,run_id in run_ids.items():
        model_uri = f"runs:/{run_id}/model_artifact"
        model = mlflow.sklearn.load_model(model_uri)
        y_pred = model.predict(X)


        r2 = r2_score(y , y_pred)
        rmse = np.sqrt(mean_squared_error(y ,  y_pred))
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)

        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)

        metrics[name] = {"r2": r2, "rmse": rmse, "mse": mse, "mae": mae}

        print(f"[평가완료] r2 = {r2} | rmse = {rmse} | mse = {mse} | mae = {mae}")


    return metrics
 

