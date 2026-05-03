#!/usr/bin/env python3
import argparse
import csv
import json
import os
import uuid
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Build metadata.csv from teacher dataset records.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Dataset directory with records/ and videos/.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional explicit output path. Defaults to <dataset-dir>/metadata.csv",
    )
    parser.add_argument("--require-video", action="store_true", help="Drop records whose video file is missing.")
    return parser.parse_args()


def load_records(records_dir: Path):
    records = []
    for path in sorted(records_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            records.append(json.load(f))
    return records


def main():
    args = parse_args()
    records_dir = args.dataset_dir / "records"
    records = load_records(records_dir)
    output_csv = args.output_csv or (args.dataset_dir / "metadata.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for record in records:
        video_path = args.dataset_dir / record["video_path"]
        if args.require_video and not video_path.exists():
            continue
        rows.append(
            {
                "video": record["video_path"],
                "prompt": record["prompt"],
                "negative_prompt": record.get("negative_prompt", ""),
                "prompt_id": record["prompt_id"],
                "seed": record.get("seed", ""),
                "height": record.get("height", ""),
                "width": record.get("width", ""),
                "num_frames": record.get("num_frames", ""),
                "fps": record.get("fps", ""),
                "cfg_scale": record.get("cfg_scale", ""),
                "num_inference_steps": record.get("num_inference_steps", ""),
                "sigma_shift": record.get("sigma_shift", ""),
                "subject_group": record.get("tags", {}).get("subject_group", ""),
                "subject_type": record.get("tags", {}).get("subject_type", "")
            }
        )

    tmp_csv = output_csv.with_name(f"{output_csv.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with tmp_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["video", "prompt"])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_csv, output_csv)

    print(f"Wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
