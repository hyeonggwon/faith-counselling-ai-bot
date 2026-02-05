#!/usr/bin/env python3
"""
OpenAI API를 사용하여 GPT-4o-mini 파인튜닝을 실행하는 스크립트
"""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# 환경 변수 로드
load_dotenv(Path(__file__).parent.parent / ".env")

# 설정
SERMONS_DIR = Path(__file__).parent.parent / "sermons"
TRAINING_DATA_FILE = SERMONS_DIR / "training_data.jsonl"
FINETUNE_INFO_FILE = SERMONS_DIR / "finetune_info.json"


def validate_dataset(file_path: Path) -> bool:
    """데이터셋 유효성을 검사합니다."""
    print("데이터셋 검증 중...")

    if not file_path.exists():
        print(f"에러: 파일이 없습니다: {file_path}")
        return False

    errors = []
    examples = []

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line)

                # messages 필드 확인
                if "messages" not in data:
                    errors.append(f"Line {i}: 'messages' 필드 없음")
                    continue

                messages = data["messages"]

                # 최소 2개 메시지 (user, assistant)
                if len(messages) < 2:
                    errors.append(f"Line {i}: 메시지가 2개 미만")
                    continue

                # role 확인
                roles = [m.get("role") for m in messages]
                if "assistant" not in roles:
                    errors.append(f"Line {i}: 'assistant' 역할 없음")
                    continue

                examples.append(data)

            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: JSON 파싱 에러 - {e}")

    if errors:
        print(f"\n검증 에러 ({len(errors)}개):")
        for err in errors[:10]:
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... 외 {len(errors) - 10}개")

    print(f"\n유효한 예제: {len(examples)}개")

    # OpenAI 최소 요구사항: 10개 이상
    if len(examples) < 10:
        print("에러: 최소 10개 이상의 예제가 필요합니다.")
        return False

    return True


def upload_training_file(client: OpenAI, file_path: Path) -> str:
    """학습 파일을 OpenAI에 업로드합니다."""
    print(f"\n파일 업로드 중: {file_path}")

    with open(file_path, "rb") as f:
        response = client.files.create(
            file=f,
            purpose="fine-tune"
        )

    print(f"업로드 완료! 파일 ID: {response.id}")
    return response.id


def create_finetune_job(client: OpenAI, file_id: str) -> str:
    """파인튜닝 작업을 생성합니다."""
    print("\n파인튜닝 작업 생성 중...")

    response = client.fine_tuning.jobs.create(
        training_file=file_id,
        model="gpt-4o-mini-2024-07-18",
        suffix="sangrok-sermon",
        hyperparameters={
            "n_epochs": 3,
            "batch_size": "auto",
            "learning_rate_multiplier": "auto"
        }
    )

    print(f"작업 생성 완료! 작업 ID: {response.id}")
    return response.id


def monitor_finetune_job(client: OpenAI, job_id: str):
    """파인튜닝 작업 상태를 모니터링합니다."""
    print("\n파인튜닝 진행 상황 모니터링...")
    print("(Ctrl+C로 모니터링을 중단해도 작업은 계속 진행됩니다)")
    print("-" * 50)

    last_event_id = None

    while True:
        # 작업 상태 확인
        job = client.fine_tuning.jobs.retrieve(job_id)

        # 이벤트 확인
        events = client.fine_tuning.jobs.list_events(
            fine_tuning_job_id=job_id,
            limit=10
        )

        for event in reversed(events.data):
            if last_event_id is None or event.id > last_event_id:
                print(f"[{event.created_at}] {event.message}")
                last_event_id = event.id

        # 완료 또는 실패 확인
        if job.status == "succeeded":
            print("\n" + "=" * 50)
            print("파인튜닝 완료!")
            print(f"모델 ID: {job.fine_tuned_model}")
            print("=" * 50)
            return job.fine_tuned_model

        elif job.status == "failed":
            print("\n" + "=" * 50)
            print("파인튜닝 실패!")
            print(f"에러: {job.error}")
            print("=" * 50)
            return None

        elif job.status == "cancelled":
            print("\n파인튜닝이 취소되었습니다.")
            return None

        time.sleep(30)  # 30초마다 확인


def main():
    # OpenAI 클라이언트 초기화
    client = OpenAI()

    print("=" * 50)
    print("GPT-4o-mini 파인튜닝")
    print("=" * 50)

    # 1. 데이터셋 검증
    if not validate_dataset(TRAINING_DATA_FILE):
        return

    # 2. 파일 업로드
    file_id = upload_training_file(client, TRAINING_DATA_FILE)

    # 3. 파인튜닝 작업 생성
    job_id = create_finetune_job(client, file_id)

    # 정보 저장
    finetune_info = {
        "file_id": file_id,
        "job_id": job_id,
        "status": "running",
        "model": None
    }

    with open(FINETUNE_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(finetune_info, f, indent=2)

    # 4. 작업 모니터링
    try:
        model_id = monitor_finetune_job(client, job_id)

        if model_id:
            finetune_info["status"] = "succeeded"
            finetune_info["model"] = model_id

            with open(FINETUNE_INFO_FILE, "w", encoding="utf-8") as f:
                json.dump(finetune_info, f, indent=2)

            print(f"\n파인튜닝된 모델을 사용하려면:")
            print(f'  model="{model_id}"')

    except KeyboardInterrupt:
        print("\n\n모니터링 중단. 작업은 백그라운드에서 계속 진행됩니다.")
        print(f"나중에 상태를 확인하려면:")
        print(f"  python -c \"from openai import OpenAI; print(OpenAI().fine_tuning.jobs.retrieve('{job_id}'))\"")


if __name__ == "__main__":
    main()
