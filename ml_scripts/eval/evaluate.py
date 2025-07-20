import mlflow
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score , mean_squared_error , mean_absolute_error
from typing import Dict

def evaluate_model(model_path: str, test_path: str, target_col: str = "psychopathy") -> Dict[str, float]:
    model = joblib.load(model_path)
    
    df = pd.read_csv(test_path)
    X = df.drop(target_col , axis = 1)
    y = df[target_col]

    y_pred = model.predict(X)
    r2 = r2_score(y , y_pred)
    rmse = np.sqrt(mean_squared_error(y ,  y_pred))
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    

    print(f"[평가완료] r2 = {r2} | rmse = {rmse} | mse = {mse} | mae = {mae}")

    with mlflow.start_run(run_name="LinearRression_Eval"):
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mse",mse)
        mlflow.log_metric("mae",mae)


    return {"r2": r2, "rmse": rmse , "mse": mse , "mae" : mae}
 

