import sys,os
sys.path.append("/home/psycho/mlops-project/ml_scripts")

from airflow  import DAG 
from airflow.operators.python import PythonOperator
from datetime import datetime
from pendulum.tz.timezone import Timezone
from dotenv import load_dotenv
import pandas as pd

load_dotenv(dotenv_path="/home/psycho/mlops-project/.env")
data_path = os.getenv("RAW_DATA_PATH")
output_dir = os.environ.get("OUTPUT_DIR")



from preprocessing.preprocess import run_preprocess
from training.train import train_model
from eval.evaluate import evaluate_model


kst = Timezone("Asia/Seoul")


def preprocess_task(input_path, output_dir, ts, **context):
    paths = run_preprocess(input_path, output_dir, ts)
    context['ti'].xcom_push(key='train_path', value=paths['train_path'])
    context['ti'].xcom_push(key='test_path', value=paths['test_path'])


def train_task(**context):
    train_path = context['ti'].xcom_pull(key = "train_path" , task_ids = 'preprocess_task')
    model_path = train_model(train_path)
    context["ti"].xcom_push(key = 'model_path' , value=model_path)


def evaluate_task(**context):
    test_path = context['ti'].xcom_pull(key = 'test_path' , task_ids = 'preprocess_task')
    model_path = context['ti'].xcom_pull(key = 'model_path' , task_ids = 'train_task')
    metrics = evaluate_model(model_path, test_path)
    context['ti'].xcom_push(key='r2' , value= metrics['r2' ])
    context['ti'].xcom_push(key = 'rmse' , value = metrics['rmse'])
    context['ti'].xcom_push(key = 'mse' , value = metrics['mse'])
    context['ti'].xcom_push(key = 'mae' , value = metrics['mae'])




with DAG(
    dag_id="ml_pipeline",
    start_date=datetime(2024,1,1 , tzinfo=kst),
    schedule_interval="*/45 * * * *",
    # schedule=None,
    catchup=False,
    tags = ['ml']
) as dag:
    
    
    
    t1 = PythonOperator(
        task_id="preprocess_task",
        python_callable= preprocess_task,
        op_kwargs={
            "input_path": os.environ.get("RAW_DATA_PATH"),
            "output_dir": os.environ.get("OUTPUT_DIR"),
            "ts": "{{ts_nodash}}"
        }
    )

    t2 = PythonOperator(
        task_id = 'train_task',
        python_callable=train_task,      
        # provide_context=True
    )

    t3 = PythonOperator(
        task_id = 'evaluate_task',
        python_callable=evaluate_task,
        # provide_context=True

    )

t1 >> t2 >> t3
