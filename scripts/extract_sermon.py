#!/usr/bin/env python3
"""
기존 STT 텍스트에서 설교 부분만 추출하는 스크립트
"""

import re
import json
from pathlib import Path

SERMONS_DIR = Path(__file__).parent.parent / "sermons"
TRANSCRIPTS_DIR = SERMONS_DIR / "transcripts"
SERMON_ONLY_DIR = TRANSCRIPTS_DIR / "sermon_only"
METADATA_FILE = SERMONS_DIR / "video_metadata.json"


def extract_sermon_section(full_text: str) -> tuple[str, dict]:
    """
    전체 예배 텍스트에서 설교 부분만 추출합니다.
    """

    # 설교 시작 패턴들
    start_patterns = [
        r"오늘\s*(우리가)?\s*함께\s*볼\s*(하나님)?\s*말씀",
        r"오늘\s*우리가\s*읽은\s*본문",
        r"오늘\s*말씀은",
        r"본문\s*말씀",
        r"말씀을?\s*봉독",
        r"함께\s*읽겠습니다",
        r"절부터\s*읽겠습니다",
        r"장을?\s*봉독",
        r"교독해서\s*읽도록",
    ]

    # 설교 끝 패턴들 (설교 마무리 기도 또는 축건)
    # Whisper가 다양하게 변환: "기도드리옵나이다", "기도합니다", "기도하옵나이다", "기도를 드리옵나이다" 등
    # 금요기도회 등은 "추건합니다"로 끝나는 경우도 있음
    end_patterns = [
        r"예수\s*(그리스도?|그리스|님)?\s*의?\s*이름으로\s*기도를?\s*(드리옵나이다|드립니다|하옵나이다|합니다)",
        r"(축건|추건)합니다",  # 축복 기도 마무리
    ]

    # 시작점 찾기
    start_pos = None
    start_match_text = None

    for pattern in start_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            if start_pos is None or match.start() < start_pos:
                start_pos = match.start()
                start_match_text = match.group()

    if start_pos is None:
        return full_text, {
            "extraction_success": False,
            "reason": "시작 패턴을 찾지 못함",
            "full_length": len(full_text)
        }

    # 시작점 이후에서 끝점 찾기 (여러 패턴 검색)
    text_after_start = full_text[start_pos:]
    end_matches = []
    for end_pattern in end_patterns:
        matches = list(re.finditer(end_pattern, text_after_start, re.IGNORECASE))
        end_matches.extend(matches)

    # 위치순 정렬
    end_matches.sort(key=lambda m: m.start())

    if not end_matches:
        return full_text, {
            "extraction_success": False,
            "reason": "끝 패턴을 찾지 못함",
            "start_found_at": start_pos,
            "full_length": len(full_text)
        }

    # 시작점에서 최소 3000자 이후의 끝 패턴을 찾음 (설교 최소 길이)
    min_sermon_length = 3000

    end_pos = None
    for match in end_matches:
        if match.start() > min_sermon_length:
            end_pos = start_pos + match.end()
            break

    if end_pos is None:
        # 마지막 끝 패턴 사용
        end_pos = start_pos + end_matches[-1].end()

    # 설교 구간 추출
    sermon_text = full_text[start_pos:end_pos].strip()

    meta = {
        "extraction_success": True,
        "start_pattern": start_match_text,
        "start_position": start_pos,
        "end_position": end_pos,
        "full_length": len(full_text),
        "sermon_length": len(sermon_text),
        "extraction_ratio": len(sermon_text) / len(full_text) if full_text else 0
    }

    return sermon_text, meta


def main():
    SERMON_ONLY_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("설교 구간 추출")
    print("=" * 60)

    # 메타데이터 로드
    if not METADATA_FILE.exists():
        print("메타데이터 파일이 없습니다.")
        return

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    videos = metadata.get("videos", [])
    print(f"\n{len(videos)}개 영상 처리")

    for i, video in enumerate(videos, 1):
        video_id = video["id"]
        title = video["title"]
        transcript_path = TRANSCRIPTS_DIR / f"{video_id}.txt"
        sermon_path = SERMON_ONLY_DIR / f"{video_id}_sermon.txt"

        print(f"\n[{i}/{len(videos)}] {title[:45]}...")

        if not transcript_path.exists():
            print("  전체 텍스트 없음, 스킵")
            continue

        # 텍스트 로드
        with open(transcript_path, "r", encoding="utf-8") as f:
            full_text = f.read()

        # 설교 추출
        sermon_text, meta = extract_sermon_section(full_text)

        if meta["extraction_success"]:
            # 저장
            with open(sermon_path, "w", encoding="utf-8") as f:
                f.write(f"# {title}\n")
                f.write(f"# 추출 비율: {meta['extraction_ratio']:.1%}\n")
                f.write(f"# 시작 패턴: {meta['start_pattern']}\n\n")
                f.write(sermon_text)

            video["sermon_path"] = str(sermon_path)
            video["extraction_meta"] = meta

            print(f"  성공: {meta['sermon_length']}자 ({meta['extraction_ratio']:.1%})")
            print(f"  시작 패턴: '{meta['start_pattern']}'")
        else:
            print(f"  실패: {meta['reason']}")

    # 메타데이터 업데이트
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
