#!/usr/bin/env python3
"""
설교 텍스트를 OpenAI 파인튜닝용 JSONL 데이터셋으로 변환하는 스크립트
"""

import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# 환경 변수 로드
load_dotenv(Path(__file__).parent.parent / ".env")

# 설정
SERMONS_DIR = Path(__file__).parent.parent / "sermons"
TRANSCRIPTS_DIR = SERMONS_DIR / "transcripts"
METADATA_FILE = SERMONS_DIR / "video_metadata.json"
TRAINING_DATA_FILE = SERMONS_DIR / "training_data.jsonl"


def clean_transcript(text: str) -> str:
    """텍스트를 정제합니다."""
    # 메타데이터 헤더 제거
    lines = text.split("\n")
    content_lines = [l for l in lines if not l.startswith("#")]
    text = "\n".join(content_lines).strip()

    # 중복 공백 제거
    text = re.sub(r'\s+', ' ', text)

    # 불필요한 문자 제거
    text = re.sub(r'[^\w\s.,!?가-힣ㄱ-ㅎㅏ-ㅣ]', '', text)

    return text


def split_into_chunks(text: str, max_tokens: int = 2000) -> list:
    """텍스트를 적절한 크기의 청크로 분할합니다."""
    # 대략적인 토큰 수 추정 (한국어는 대략 2자당 1토큰)
    char_limit = max_tokens * 2

    # 문장 단위로 분할
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        if current_length + sentence_length > char_limit:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def generate_qa_pairs(client: OpenAI, sermon_text: str, title: str) -> list:
    """GPT-4o-mini를 사용하여 설교 텍스트에서 Q&A 쌍을 생성합니다."""

    prompt = f"""다음은 교회 설교의 일부입니다. 이 설교 내용을 바탕으로 신앙 상담에 활용할 수 있는 질문-답변 쌍을 3-5개 생성해주세요.

설교 제목: {title}

설교 내용:
{sermon_text[:3000]}

다음 형식으로 JSON 배열을 반환해주세요:
[
  {{
    "question": "신자가 물을 수 있는 질문",
    "answer": "설교 내용을 바탕으로 한 목회적 답변 (설교 스타일 유지)"
  }}
]

주의사항:
1. 질문은 일반 신자가 실제로 할 법한 신앙 질문이어야 합니다
2. 답변은 설교자의 어조와 스타일을 유지해야 합니다
3. 답변에 성경 구절이나 교리적 내용이 포함되면 좋습니다
4. JSON 형식만 반환하세요"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 기독교 신앙 상담 데이터셋을 만드는 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )

        content = response.choices[0].message.content

        # JSON 추출
        json_match = re.search(r'\[[\s\S]*\]', content)
        if json_match:
            qa_pairs = json.loads(json_match.group())
            return qa_pairs

    except Exception as e:
        print(f"    QA 생성 에러: {e}")

    return []


def create_training_example(question: str, answer: str) -> dict:
    """OpenAI 파인튜닝 형식의 학습 예제를 생성합니다."""
    return {
        "messages": [
            {
                "role": "system",
                "content": "당신은 상록교회의 신앙 상담 AI입니다. 장로교 교리에 기반하여 따뜻하고 목회적인 답변을 제공합니다. 성경 말씀과 교리를 인용하며, 상담자의 영적 성장을 돕습니다."
            },
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]
    }


def main():
    # OpenAI 클라이언트 초기화
    client = OpenAI()

    print("=" * 50)
    print("파인튜닝 데이터셋 생성")
    print("=" * 50)

    # 메타데이터 로드
    if not METADATA_FILE.exists():
        print("메타데이터 파일이 없습니다.")
        return

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    videos = [v for v in metadata.get("videos", []) if v.get("transcript_path")]
    print(f"\n{len(videos)}개의 설교 텍스트 처리 예정")

    all_examples = []
    sermon_only_count = 0

    for i, video in enumerate(videos, 1):
        title = video["title"]

        # 설교 전용 텍스트가 있으면 우선 사용
        if video.get("sermon_path") and Path(video["sermon_path"]).exists():
            text_path = Path(video["sermon_path"])
            is_sermon_only = True
        else:
            text_path = Path(video["transcript_path"])
            is_sermon_only = False

        print(f"\n[{i}/{len(videos)}] {title[:40]}...")
        if is_sermon_only:
            print("  (설교 전용 텍스트 사용)")
            sermon_only_count += 1

        if not text_path.exists():
            print("  텍스트 파일 없음, 스킵")
            continue

        # 텍스트 로드 및 정제
        with open(text_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        clean_text = clean_transcript(raw_text)

        if len(clean_text) < 500:
            print("  텍스트가 너무 짧음, 스킵")
            continue

        # 청크로 분할
        chunks = split_into_chunks(clean_text)
        print(f"  {len(chunks)}개 청크로 분할")

        # 각 청크에서 Q&A 생성
        for j, chunk in enumerate(chunks[:3]):  # 처음 3개 청크만 처리 (비용 절약)
            print(f"    청크 {j+1} 처리 중...")
            qa_pairs = generate_qa_pairs(client, chunk, title)

            for qa in qa_pairs:
                example = create_training_example(qa["question"], qa["answer"])
                all_examples.append(example)

            print(f"    {len(qa_pairs)}개 Q&A 생성")

    # JSONL 파일 저장
    with open(TRAINING_DATA_FILE, "w", encoding="utf-8") as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print("\n" + "=" * 50)
    print(f"데이터셋 생성 완료!")
    print(f"총 {len(all_examples)}개 학습 예제")
    print(f"설교 전용 텍스트 사용: {sermon_only_count}개")
    print(f"저장 위치: {TRAINING_DATA_FILE}")
    print("=" * 50)

    # 통계 출력
    if all_examples:
        total_tokens = sum(
            len(str(ex)) // 2  # 대략적인 토큰 수 추정
            for ex in all_examples
        )
        print(f"\n예상 토큰 수: ~{total_tokens:,}")
        print(f"예상 학습 비용: ~${total_tokens * 25 / 1_000_000:.2f}")


if __name__ == "__main__":
    main()
