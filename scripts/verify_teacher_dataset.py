#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Verify teacher dataset artifacts.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Dataset directory to verify.")
    return parser.parse_args()


def count_jsonl(path: Path):
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main():
    args = parse_args()
    videos_dir = args.dataset_dir / "videos"
    records_dir = args.dataset_dir / "records"
    logs_dir = args.dataset_dir / "logs"
    metadata_path = args.dataset_dir / "metadata.csv"

    video_count = len(list(videos_dir.glob("*.mp4"))) + len(list(videos_dir.glob("*.gif")))
    record_count = len(list(records_dir.glob("*.json")))
    manifest_count = sum(count_jsonl(path) for path in logs_dir.glob("manifest_rank*.jsonl"))
    failure_count = sum(count_jsonl(path) for path in logs_dir.glob("failures_rank*.jsonl"))
    missing_videos = []
    for record_path in sorted(records_dir.glob("*.json")):
        with record_path.open("r", encoding="utf-8") as f:
            record = json.load(f)
        video_path = args.dataset_dir / record["video_path"]
        if not video_path.exists():
            missing_videos.append(record["video_path"])

    summary = {
        "dataset_dir": str(args.dataset_dir),
        "video_count": video_count,
        "record_count": record_count,
        "manifest_entries": manifest_count,
        "failure_entries": failure_count,
        "metadata_exists": metadata_path.exists(),
        "missing_videos": missing_videos[:20],
        "missing_videos_count": len(missing_videos),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
