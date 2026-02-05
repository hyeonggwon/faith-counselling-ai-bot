#!/usr/bin/env python3
"""
상록교회 유튜브 플레이리스트에서 설교 영상의 오디오를 다운로드하는 스크립트
"""

import os
import json
import subprocess
from pathlib import Path
from datetime import datetime

# 설정
OUTPUT_DIR = Path(__file__).parent.parent / "sermons" / "audio"
METADATA_FILE = Path(__file__).parent.parent / "sermons" / "video_metadata.json"

# 설교 플레이리스트 (주일예배, 수요예배, 금요기도회)
SERMON_PLAYLISTS = [
    {
        "id": "PLcc2F7_jwcdsi2d93qoBF8vB1iE99dMwb",
        "name": "주일 예배 (오전 11시)",
        "type": "sunday"
    },
    {
        "id": "PLcc2F7_jwcduSXMFnMyxBx3IMion-wj4r",
        "name": "수요예배",
        "type": "wednesday"
    },
    {
        "id": "PLcc2F7_jwcdvWKtkVR4IoEkpzGkEuZ_ND",
        "name": "금요기도회",
        "type": "friday"
    },
]


def get_playlist_videos(playlist_id: str, max_videos: int = None) -> list:
    """플레이리스트에서 영상 목록을 가져옵니다."""
    playlist_url = f"https://www.youtube.com/playlist?list={playlist_id}"

    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--print", "%(id)s|%(title)s|%(upload_date)s|%(duration)s",
        playlist_url
    ]

    if max_videos:
        cmd.extend(["--playlist-end", str(max_videos)])

    result = subprocess.run(cmd, capture_output=True, text=True)

    videos = []
    for line in result.stdout.strip().split("\n"):
        if line and "|" in line:
            parts = line.split("|")
            if len(parts) >= 4:
                videos.append({
                    "id": parts[0],
                    "title": parts[1],
                    "upload_date": parts[2],
                    "duration": parts[3],
                    "url": f"https://www.youtube.com/watch?v={parts[0]}"
                })

    return videos


def download_audio(video_id: str, output_dir: Path) -> str:
    """영상에서 오디오만 추출하여 다운로드합니다."""
    output_path = output_dir / f"{video_id}.mp3"

    if output_path.exists():
        print(f"  이미 다운로드됨: {video_id}")
        return str(output_path)

    cmd = [
        "yt-dlp",
        "-x",  # 오디오만 추출
        "--audio-format", "mp3",
        "--audio-quality", "0",  # 최고 품질
        "-o", str(output_path),
        f"https://www.youtube.com/watch?v={video_id}"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"  다운로드 완료: {video_id}")
        return str(output_path)
    else:
        print(f"  다운로드 실패: {video_id}")
        print(f"  에러: {result.stderr}")
        return None


def main():
    # 출력 디렉토리 생성
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("상록교회 유튜브 설교 다운로드")
    print("=" * 60)

    # 환경 변수로 제한 설정
    max_per_playlist = int(os.environ.get("MAX_VIDEOS", 10))
    selected_types = os.environ.get("PLAYLIST_TYPES", "sunday,wednesday,friday").split(",")

    print(f"\n설정:")
    print(f"  플레이리스트당 최대: {max_per_playlist}개")
    print(f"  선택된 예배: {', '.join(selected_types)}")

    # 기존 메타데이터 로드 (있으면)
    existing_ids = set()
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            existing_ids = {v["id"] for v in existing_data.get("videos", [])}
        print(f"  기존 영상: {len(existing_ids)}개")

    all_videos = []

    # 각 플레이리스트에서 영상 수집
    for playlist in SERMON_PLAYLISTS:
        if playlist["type"] not in selected_types:
            continue

        print(f"\n[{playlist['name']}] 영상 목록 가져오는 중...")
        videos = get_playlist_videos(playlist["id"], max_per_playlist)

        for video in videos:
            video["playlist_type"] = playlist["type"]
            video["playlist_name"] = playlist["name"]

        print(f"  {len(videos)}개 영상 발견")
        all_videos.extend(videos)

    # 중복 제거 (이미 다운로드된 것 제외)
    unique_videos = []
    seen_ids = set()

    for video in all_videos:
        if video["id"] not in seen_ids and video["id"] not in existing_ids:
            unique_videos.append(video)
            seen_ids.add(video["id"])

    print(f"\n총 {len(unique_videos)}개 새로운 영상 다운로드 예정")

    if not unique_videos:
        print("다운로드할 새로운 영상이 없습니다.")
        return

    # 메타데이터 초기화
    metadata = {
        "playlists": SERMON_PLAYLISTS,
        "download_date": datetime.now().isoformat(),
        "videos": []
    }

    # 오디오 다운로드
    print("\n오디오 다운로드 시작...")
    downloaded = []

    for i, video in enumerate(unique_videos, 1):
        print(f"\n[{i}/{len(unique_videos)}] [{video['playlist_type']}] {video['title'][:45]}...")
        audio_path = download_audio(video["id"], OUTPUT_DIR)
        if audio_path:
            video["audio_path"] = audio_path
            downloaded.append(video)

    # 기존 데이터와 병합
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            metadata["videos"] = existing_data.get("videos", [])

    metadata["videos"].extend(downloaded)
    metadata["total_count"] = len(metadata["videos"])

    # 메타데이터 저장
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"다운로드 완료: {len(downloaded)}개 (신규)")
    print(f"전체 영상: {metadata['total_count']}개")
    print("=" * 60)

    # 통계
    by_type = {}
    for v in metadata["videos"]:
        t = v.get("playlist_type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1

    print("\n예배별 영상 수:")
    for t, count in sorted(by_type.items()):
        print(f"  {t}: {count}개")


if __name__ == "__main__":
    main()
