#!/usr/bin/env python3
"""
Whisper를 사용하여 설교 오디오를 텍스트로 변환하는 스크립트
설교 구간 자동 추출 기능 포함
"""

import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# 환경 변수 로드
load_dotenv(Path(__file__).parent.parent / ".env")

# 설정
SERMONS_DIR = Path(__file__).parent.parent / "sermons"
AUDIO_DIR = SERMONS_DIR / "audio"
TRANSCRIPTS_DIR = SERMONS_DIR / "transcripts"
METADATA_FILE = SERMONS_DIR / "video_metadata.json"


def transcribe_audio(client: OpenAI, audio_path: Path) -> str:
    """OpenAI Whisper API를 사용하여 오디오를 텍스트로 변환합니다."""

    # 파일 크기 확인 (25MB 제한)
    file_size = audio_path.stat().st_size
    max_size = 25 * 1024 * 1024  # 25MB

    if file_size > max_size:
        print(f"  경고: 파일이 25MB를 초과합니다. 분할이 필요합니다.")
        return transcribe_large_audio(client, audio_path)

    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="ko",
            response_format="text"
        )

    return transcript


def transcribe_large_audio(client: OpenAI, audio_path: Path) -> str:
    """25MB를 초과하는 오디오 파일을 분할하여 변환합니다."""
    import subprocess
    import tempfile

    # pydub 대신 ffmpeg 직접 사용
    duration_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path)
    ]

    result = subprocess.run(duration_cmd, capture_output=True, text=True)
    total_duration = float(result.stdout.strip())

    # 10분 단위로 분할
    segment_duration = 600  # 10분
    segments = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        start = 0
        segment_num = 0

        while start < total_duration:
            segment_file = temp_path / f"segment_{segment_num}.mp3"

            cmd = [
                "ffmpeg", "-y",
                "-i", str(audio_path),
                "-ss", str(start),
                "-t", str(segment_duration),
                "-acodec", "libmp3lame",
                str(segment_file)
            ]

            subprocess.run(cmd, capture_output=True)

            if segment_file.exists():
                with open(segment_file, "rb") as f:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        language="ko",
                        response_format="text"
                    )
                segments.append(transcript)
                print(f"    세그먼트 {segment_num + 1} 변환 완료")

            start += segment_duration
            segment_num += 1

    return "\n\n".join(segments)


def extract_sermon_section(full_text: str) -> tuple[str, dict]:
    """
    전체 예배 텍스트에서 설교 부분만 추출합니다.

    로직:
    1. "본문 말씀" / "봉독" / "함께 읽겠습니다" 등의 패턴 → 설교 시작점
    2. 시작점 이후 첫 번째 "예수님의 이름으로 기도드립니다" → 설교 끝점

    Returns:
        (설교 텍스트, 메타정보 dict)
    """

    # 설교 시작 패턴들 (본문 말씀 읽기/설교 시작 관련)
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
        # 시작 패턴을 찾지 못한 경우
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
        # 끝 패턴을 찾지 못한 경우
        return full_text, {
            "extraction_success": False,
            "reason": "끝 패턴을 찾지 못함",
            "start_found_at": start_pos,
            "full_length": len(full_text)
        }

    # 시작점 이후 첫 번째 "예수님의 이름으로 기도드립니다"가 설교 끝
    # 하지만 본문 봉독 직후 기도가 있을 수 있으므로,
    # 시작점에서 최소 1000자 이후의 끝 패턴을 찾음
    min_sermon_length = 1000  # 최소 설교 길이 (약 500단어)

    end_pos = None
    for match in end_matches:
        if match.start() > min_sermon_length:
            end_pos = start_pos + match.end()
            break

    if end_pos is None:
        # 충분히 긴 설교 구간을 찾지 못한 경우, 마지막 끝 패턴 사용
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
    # OpenAI 클라이언트 초기화
    client = OpenAI()

    # 디렉토리 생성
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("설교 음성→텍스트 변환 (Whisper)")
    print("=" * 50)

    # 메타데이터 로드
    if not METADATA_FILE.exists():
        print("메타데이터 파일이 없습니다. 먼저 download_sermons.py를 실행하세요.")
        return

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    videos = metadata.get("videos", [])
    print(f"\n총 {len(videos)}개 영상 처리 예정")

    transcribed = []

    # 설교 전용 디렉토리
    SERMON_ONLY_DIR = TRANSCRIPTS_DIR / "sermon_only"
    SERMON_ONLY_DIR.mkdir(parents=True, exist_ok=True)

    extraction_stats = {"success": 0, "failed": 0, "total_ratio": []}

    for i, video in enumerate(videos, 1):
        video_id = video["id"]
        title = video["title"]
        audio_path = Path(video.get("audio_path", AUDIO_DIR / f"{video_id}.mp3"))
        transcript_path = TRANSCRIPTS_DIR / f"{video_id}.txt"
        sermon_path = SERMON_ONLY_DIR / f"{video_id}_sermon.txt"

        print(f"\n[{i}/{len(videos)}] {title[:40]}...")

        # 이미 변환된 경우 스킵
        if transcript_path.exists():
            print("  이미 변환됨, 스킵")
            video["transcript_path"] = str(transcript_path)
            if sermon_path.exists():
                video["sermon_path"] = str(sermon_path)
            transcribed.append(video)
            continue

        # 오디오 파일 확인
        if not audio_path.exists():
            print(f"  오디오 파일 없음: {audio_path}")
            continue

        try:
            print("  변환 중...")
            transcript = transcribe_audio(client, audio_path)

            # 전체 텍스트 저장
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(f"# {title}\n")
                f.write(f"# 날짜: {video.get('upload_date', 'N/A')}\n")
                f.write(f"# URL: {video.get('url', 'N/A')}\n\n")
                f.write(transcript)

            video["transcript_path"] = str(transcript_path)
            print(f"  전체 텍스트: {len(transcript)} 글자")

            # 설교 구간 추출
            sermon_text, extract_meta = extract_sermon_section(transcript)

            if extract_meta["extraction_success"]:
                # 설교 텍스트 저장
                with open(sermon_path, "w", encoding="utf-8") as f:
                    f.write(f"# {title}\n")
                    f.write(f"# 날짜: {video.get('upload_date', 'N/A')}\n")
                    f.write(f"# URL: {video.get('url', 'N/A')}\n")
                    f.write(f"# 추출 비율: {extract_meta['extraction_ratio']:.1%}\n\n")
                    f.write(sermon_text)

                video["sermon_path"] = str(sermon_path)
                video["extraction_meta"] = extract_meta
                extraction_stats["success"] += 1
                extraction_stats["total_ratio"].append(extract_meta["extraction_ratio"])
                print(f"  설교 추출: {len(sermon_text)} 글자 ({extract_meta['extraction_ratio']:.1%})")
            else:
                extraction_stats["failed"] += 1
                print(f"  설교 추출 실패: {extract_meta['reason']}")

            transcribed.append(video)

        except Exception as e:
            print(f"  에러: {e}")

    # 메타데이터 업데이트
    metadata["videos"] = transcribed
    metadata["transcribed_count"] = len(transcribed)

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print(f"변환 완료: {len(transcribed)}/{len(videos)}개")
    print("=" * 50)

    # 설교 추출 통계
    if extraction_stats["total_ratio"]:
        avg_ratio = sum(extraction_stats["total_ratio"]) / len(extraction_stats["total_ratio"])
        print(f"\n설교 추출 통계:")
        print(f"  성공: {extraction_stats['success']}개")
        print(f"  실패: {extraction_stats['failed']}개")
        print(f"  평균 추출 비율: {avg_ratio:.1%}")


if __name__ == "__main__":
    main()
