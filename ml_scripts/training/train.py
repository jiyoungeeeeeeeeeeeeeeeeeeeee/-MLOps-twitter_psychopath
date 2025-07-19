import os
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

MODEL_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

def train_model(train_path: str) -> str:
    df = pd.read_csv(train_path)
    X = df.drop('psychopathy', axis=1)
    y = df['psychopathy']

    model = LinearRegression(
        fit_intercept=True,
        n_jobs=-1
    )
    model.fit(X, y)

    # 파일명 지정
    base_name = os.path.splitext(os.path.basename(train_path))[0]
    model_filename = f"{base_name}_lr.pkl"
    model_path = os.path.join(MODEL_OUTPUT_DIR, model_filename)

    # 모델 저장
    joblib.dump(model, model_path)

    return model_path
