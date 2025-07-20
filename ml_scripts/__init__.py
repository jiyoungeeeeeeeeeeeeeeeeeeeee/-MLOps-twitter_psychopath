if __name__ == "__main__":
    raw_path = os.environ.get("RAW_DATA_PATH")
    output_dir = os.environ.get("OUTPUT_DIR")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not raw_path or not output_dir:
        raise ValueError("환경변수 RAW_DATA_PATH 또는 OUTPUT_DIR가 설정되지 않았습니다!")

    result = run_preprocess(raw_path, output_dir, ts)
    print(f"[테스트 실행] 반환된 경로: {result}")


if __name__ == "__main__":
    raw_path = os.environ.get("RAW_DATA_PATH")
    output_dir = os.environ.get("OUTPUT_DIR")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not raw_path or not output_dir:
        raise ValueError("환경변수 RAW_DATA_PATH 또는 OUTPUT_DIR가 설정되지 않았습니다!")

    result = train_and_log(raw_path, output_dir, ts)
    print(f"[테스트 실행] 반환된 경로: {result}")
