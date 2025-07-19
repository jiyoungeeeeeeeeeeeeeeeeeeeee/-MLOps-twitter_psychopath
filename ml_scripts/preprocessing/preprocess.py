# ml_scripts/preprocessing/run_preprocess.py
from dotenv import load_dotenv
import pandas as pd
import os
from datetime import datetime
from sklearn.model_selection import train_test_split

load_dotenv(dotenv_path="/home/psycho/mlops-project/.env")

def run_preprocess(input_path: str, output_dir: str, ts: str, **context):
    df = pd.read_csv(input_path)

    print(f"[전처리 시작] input={input_path}")
    df = df.dropna()
    df['Var20'] = pd.to_numeric(df['Var20'], errors='coerce')

    os.makedirs(output_dir, exist_ok=True)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_path = os.path.join(output_dir, f"train_{ts}.csv")
    test_path = os.path.join(output_dir, f"test_{ts}.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[전처리] train 저장 완료: {train_path}")
    print(f"[전처리] test 저장 완료: {test_path}")

    # XCom에 push
    if 'ti' in context:
        context['ti'].xcom_push(key="train_path", value=train_path)
        context['ti'].xcom_push(key="test_path", value=test_path)

    # 로컬 테스트나 Airflow 2.3+용 dict 리턴
    return {"train_path": train_path, "test_path": test_path}

if __name__ == "__main__":
    raw_path = os.environ.get("RAW_DATA_PATH")
    output_dir = os.environ.get("OUTPUT_DIR")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not raw_path or not output_dir:
        raise ValueError("환경변수 RAW_DATA_PATH 또는 OUTPUT_DIR가 설정되지 않았습니다!")

    result = run_preprocess(raw_path, output_dir, ts)
    print(f"[테스트 실행] 반환된 경로: {result}")
