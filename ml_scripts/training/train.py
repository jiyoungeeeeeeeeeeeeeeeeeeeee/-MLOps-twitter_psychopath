import os
import pandas as pd
import mlflow 
import mlflow.sklearn
from sklearn.linear_model import LinearRegression , LassoCV , RidgeCV
from sklearn.svm import SVR , LinearSVR
from sklearn.ensemble import AdaBoostRegressor , VotingRegressor , RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor


# MODEL_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
# os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

def train_model(train_path: str) -> dict[str , object]:
    df = pd.read_csv(train_path)
    X = df.drop('psychopathy', axis=1)
    y = df['psychopathy']


    base_models = {'LinearRegression' :LinearRegression(fit_intercept=True,n_jobs=-1),
                    'SVR' : SVR( kernel= 'rbf' , C= 1 , epsilon=0.1 ),
                    'XGBRegressor' : XGBRegressor(n_estimators = 200, learning_rate = 0.1, max_depth=5, random_state = 42)       
    }

    trained = {}

    for name, model in base_models.items():
        model.fit(X,y)
        trained[name] = model

    return trained
