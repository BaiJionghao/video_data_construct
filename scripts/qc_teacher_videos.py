#!/usr/bin/env python3
"""Automatic quality checks for teacher video datasets.

The checks are intentionally conservative: structural/file problems are hard
rejects, while visual symptoms such as low motion or flicker are marked for
review so human sampling can make the final call.
"""

import argparse
import csv
import json
import math
import os
import statistics
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DEFAULT_FIELDS = [
    "video",
    "prompt",
    "negative_prompt",
    "prompt_id",
    "seed",
    "height",
    "width",
    "num_frames",
    "fps",
    "cfg_scale",
    "num_inference_steps",
    "sigma_shift",
    "subject_group",
    "subject_type",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run automatic QC over generated teacher videos.")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Dataset directory with videos/ and records/.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to <dataset-dir>/qc.")
    parser.add_argument("--metadata-csv", type=Path, default=None, help="Defaults to <dataset-dir>/metadata.csv.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sample-frames", type=int, default=9, help="Frames sampled evenly across each video.")
    parser.add_argument("--expected-count", type=int, default=2000)
    parser.add_argument("--expected-width", type=int, default=832)
    parser.add_argument("--expected-height", type=int, default=480)
    parser.add_argument("--expected-num-frames", type=int, default=81)
    parser.add_argument("--expected-fps", type=float, default=16.0)
    parser.add_argument("--frame-tolerance", type=int, default=1)
    parser.add_argument("--fps-tolerance", type=float, default=0.05)
    parser.add_argument("--duration-tolerance", type=float, default=0.35)
    parser.add_argument("--min-video-bytes", type=int, default=1024)
    parser.add_argument("--warn-min-video-bytes", type=int, default=150_000)
    parser.add_argument("--hard-min-luma", type=float, default=3.0)
    parser.add_argument("--hard-max-luma", type=float, default=252.0)
    parser.add_argument("--warn-min-luma", type=float, default=15.0)
    parser.add_argument("--warn-max-luma", type=float, default=240.0)
    parser.add_argument("--hard-min-frame-std", type=float, default=2.0)
    parser.add_argument("--warn-min-frame-std", type=float, default=10.0)
    parser.add_argument("--low-motion-avg-diff", type=float, default=1.5)
    parser.add_argument("--low-motion-max-diff", type=float, default=4.0)
    parser.add_argument("--high-flicker-luma-std", type=float, default=35.0)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing QC reports.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def safe_stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return float(statistics.pstdev(values))


def round_or_none(value: float | None, ndigits: int = 4) -> float | None:
    if value is None or math.isnan(value):
        return None
    return round(float(value), ndigits)


def sample_positions(frame_count: int, sample_count: int) -> list[int]:
    if frame_count <= 0:
        return []
    if sample_count <= 1:
        return [0]
    return sorted({int(round(i * (frame_count - 1) / (sample_count - 1))) for i in range(sample_count)})


def qc_one(task: dict[str, Any]) -> dict[str, Any]:
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {
            "prompt_id": task["prompt_id"],
            "status": "reject",
            "reject_reasons": ["missing_video_decode_dependency"],
            "warning_reasons": [],
            "error": repr(exc),
        }

    dataset_dir = Path(task["dataset_dir"])
    record_path = Path(task["record_path"])
    args = task["args"]

    reject_reasons: list[str] = []
    warning_reasons: list[str] = []
    metrics: dict[str, Any] = {}

    try:
        record = load_json(record_path)
    except Exception as exc:  # noqa: BLE001
        return {
            "prompt_id": task["prompt_id"],
            "record_path": str(record_path),
            "status": "reject",
            "reject_reasons": ["sidecar_json_error"],
            "warning_reasons": [],
            "error": repr(exc),
        }

    prompt_id = str(record.get("prompt_id") or task["prompt_id"])
    video_rel = record.get("video_path") or f"videos/{prompt_id}.mp4"
    video_path = dataset_dir / video_rel
    metrics["record_path"] = str(record_path.relative_to(dataset_dir))
    metrics["video_path"] = str(Path(video_rel))

    if not video_path.exists():
        reject_reasons.append("missing_video")
        return {
            "prompt_id": prompt_id,
            "status": "reject",
            "reject_reasons": reject_reasons,
            "warning_reasons": warning_reasons,
            **metrics,
        }

    video_bytes = video_path.stat().st_size
    metrics["video_bytes"] = video_bytes
    if video_bytes < args["min_video_bytes"]:
        reject_reasons.append("video_too_small")
    elif video_bytes < args["warn_min_video_bytes"]:
        warning_reasons.append("small_video_file")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        reject_reasons.append("open_failed")
        return {
            "prompt_id": prompt_id,
            "status": "reject",
            "reject_reasons": reject_reasons,
            "warning_reasons": warning_reasons,
            **metrics,
        }

    width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0))
    height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    duration_s = frame_count / fps if fps > 0 else None
    expected_duration = args["expected_num_frames"] / args["expected_fps"]

    metrics.update(
        {
            "width": width,
            "height": height,
            "fps": round_or_none(fps),
            "frame_count": frame_count,
            "duration_s": round_or_none(duration_s),
        }
    )

    if width != args["expected_width"] or height != args["expected_height"]:
        reject_reasons.append("dimension_mismatch")
    if fps <= 0 or abs(fps - args["expected_fps"]) > args["fps_tolerance"]:
        reject_reasons.append("fps_mismatch")
    if frame_count <= 0 or abs(frame_count - args["expected_num_frames"]) > args["frame_tolerance"]:
        reject_reasons.append("frame_count_mismatch")
    if duration_s is None or abs(duration_s - expected_duration) > args["duration_tolerance"]:
        reject_reasons.append("duration_mismatch")

    positions = sample_positions(frame_count, args["sample_frames"])
    luma_means: list[float] = []
    luma_stds: list[float] = []
    lap_vars: list[float] = []
    diffs: list[float] = []
    prev_small = None
    decoded_positions: list[int] = []

    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        decoded_positions.append(pos)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        luma_means.append(float(gray.mean()))
        luma_stds.append(float(gray.std()))
        lap_vars.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        small = cv2.resize(gray, (96, 54), interpolation=cv2.INTER_AREA).astype(np.float32)
        if prev_small is not None:
            diffs.append(float(np.mean(np.abs(small - prev_small))))
        prev_small = small

    cap.release()

    metrics["sample_positions"] = decoded_positions
    metrics["decoded_sample_frames"] = len(decoded_positions)
    metrics["luma_mean"] = round_or_none(safe_mean(luma_means))
    metrics["luma_mean_std_over_time"] = round_or_none(safe_stdev(luma_means))
    metrics["frame_luma_std_mean"] = round_or_none(safe_mean(luma_stds))
    metrics["laplacian_var_mean"] = round_or_none(safe_mean(lap_vars))
    metrics["sample_diff_mean"] = round_or_none(safe_mean(diffs))
    metrics["sample_diff_max"] = round_or_none(max(diffs) if diffs else None)

    if len(decoded_positions) < max(1, min(3, len(positions))):
        reject_reasons.append("too_few_decodable_frames")

    luma_mean = metrics["luma_mean"]
    frame_std = metrics["frame_luma_std_mean"]
    diff_mean = metrics["sample_diff_mean"]
    diff_max = metrics["sample_diff_max"]
    luma_time_std = metrics["luma_mean_std_over_time"]

    if luma_mean is not None:
        if luma_mean < args["hard_min_luma"] or luma_mean > args["hard_max_luma"]:
            reject_reasons.append("black_or_white_video")
        elif luma_mean < args["warn_min_luma"]:
            warning_reasons.append("very_dark_video")
        elif luma_mean > args["warn_max_luma"]:
            warning_reasons.append("very_bright_video")

    if frame_std is not None:
        if frame_std < args["hard_min_frame_std"]:
            reject_reasons.append("blank_low_detail_video")
        elif frame_std < args["warn_min_frame_std"]:
            warning_reasons.append("low_detail_video")

    if diff_mean is not None and diff_max is not None:
        if diff_mean < args["low_motion_avg_diff"] and diff_max < args["low_motion_max_diff"]:
            warning_reasons.append("low_motion_or_freeze")

    if luma_time_std is not None and luma_time_std > args["high_flicker_luma_std"]:
        warning_reasons.append("high_global_flicker")

    status = "reject" if reject_reasons else ("review" if warning_reasons else "pass")
    return {
        "prompt_id": prompt_id,
        "status": status,
        "reject_reasons": sorted(set(reject_reasons)),
        "warning_reasons": sorted(set(warning_reasons)),
        "prompt": record.get("prompt", ""),
        "source_bucket": record.get("tags", {}).get("source_bucket", ""),
        "motion_type": record.get("tags", {}).get("motion_type", ""),
        "motion_intensity": record.get("tags", {}).get("motion_intensity", ""),
        **metrics,
    }


def load_metadata_rows(metadata_csv: Path) -> dict[str, dict[str, str]]:
    if not metadata_csv.exists():
        return {}
    with metadata_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {row.get("prompt_id", ""): row for row in reader if row.get("prompt_id")}


def sidecar_to_metadata_row(dataset_dir: Path, prompt_id: str) -> dict[str, Any]:
    record = load_json(dataset_dir / "records" / f"{prompt_id}.json")
    tags = record.get("tags", {})
    return {
        "video": record.get("video_path", ""),
        "prompt": record.get("prompt", ""),
        "negative_prompt": record.get("negative_prompt", ""),
        "prompt_id": record.get("prompt_id", prompt_id),
        "seed": record.get("seed", ""),
        "height": record.get("height", ""),
        "width": record.get("width", ""),
        "num_frames": record.get("num_frames", ""),
        "fps": record.get("fps", ""),
        "cfg_scale": record.get("cfg_scale", ""),
        "num_inference_steps": record.get("num_inference_steps", ""),
        "sigma_shift": record.get("sigma_shift", ""),
        "subject_group": tags.get("subject_group", ""),
        "subject_type": tags.get("subject_type", ""),
    }


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    fieldnames = list(rows[0].keys()) if rows else DEFAULT_FIELDS
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def summarize(results: list[dict[str, Any]], args: argparse.Namespace, elapsed_s: float) -> dict[str, Any]:
    statuses = Counter(row["status"] for row in results)
    reject_reasons = Counter(reason for row in results for reason in row.get("reject_reasons", []))
    warning_reasons = Counter(reason for row in results for reason in row.get("warning_reasons", []))
    numeric_fields = [
        "video_bytes",
        "fps",
        "frame_count",
        "duration_s",
        "luma_mean",
        "luma_mean_std_over_time",
        "frame_luma_std_mean",
        "sample_diff_mean",
        "sample_diff_max",
        "laplacian_var_mean",
    ]
    metric_summary: dict[str, dict[str, float]] = {}
    for field in numeric_fields:
        values = [float(row[field]) for row in results if row.get(field) is not None]
        if not values:
            continue
        values_sorted = sorted(values)
        metric_summary[field] = {
            "min": round(values_sorted[0], 4),
            "mean": round(sum(values_sorted) / len(values_sorted), 4),
            "median": round(values_sorted[len(values_sorted) // 2], 4),
            "max": round(values_sorted[-1], 4),
        }

    return {
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "checked_count": len(results),
        "expected_count": args.expected_count,
        "status_counts": dict(statuses),
        "reject_reason_counts": dict(reject_reasons),
        "warning_reason_counts": dict(warning_reasons),
        "metric_summary": metric_summary,
        "elapsed_s": round(elapsed_s, 3),
        "workers": args.workers,
        "sample_frames": args.sample_frames,
        "thresholds": {
            "expected_width": args.expected_width,
            "expected_height": args.expected_height,
            "expected_num_frames": args.expected_num_frames,
            "expected_fps": args.expected_fps,
            "frame_tolerance": args.frame_tolerance,
            "fps_tolerance": args.fps_tolerance,
            "duration_tolerance": args.duration_tolerance,
            "hard_min_luma": args.hard_min_luma,
            "hard_max_luma": args.hard_max_luma,
            "hard_min_frame_std": args.hard_min_frame_std,
            "low_motion_avg_diff": args.low_motion_avg_diff,
            "low_motion_max_diff": args.low_motion_max_diff,
        },
    }


def write_markdown(path: Path, summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    status_counts = summary["status_counts"]
    reject_reason_counts = summary["reject_reason_counts"]
    warning_reason_counts = summary["warning_reason_counts"]
    lines = [
        "# Teacher Video QC Report",
        "",
        f"Dataset: `{summary['dataset_dir']}`",
        "",
        "## Summary",
        "",
        f"- Checked: `{summary['checked_count']}`",
        f"- Pass: `{status_counts.get('pass', 0)}`",
        f"- Review: `{status_counts.get('review', 0)}`",
        f"- Reject: `{status_counts.get('reject', 0)}`",
        f"- Workers: `{summary['workers']}`",
        f"- Sample frames per video: `{summary['sample_frames']}`",
        f"- Elapsed seconds: `{summary['elapsed_s']}`",
        "",
        "## Reject Reasons",
        "",
    ]
    if reject_reason_counts:
        lines += [f"- `{key}`: {value}" for key, value in sorted(reject_reason_counts.items())]
    else:
        lines.append("- none")
    lines += ["", "## Review Warnings", ""]
    if warning_reason_counts:
        lines += [f"- `{key}`: {value}" for key, value in sorted(warning_reason_counts.items())]
    else:
        lines.append("- none")
    lines += ["", "## Metric Summary", ""]
    for field, stats in summary["metric_summary"].items():
        lines.append(
            f"- `{field}`: min={stats['min']}, mean={stats['mean']}, "
            f"median={stats['median']}, max={stats['max']}"
        )
    reject_examples = [row for row in results if row["status"] == "reject"][:20]
    review_examples = [row for row in results if row["status"] == "review"][:20]
    lines += ["", "## Reject Examples", ""]
    if reject_examples:
        lines += [
            f"- `{row['prompt_id']}`: {', '.join(row.get('reject_reasons', []))}"
            for row in reject_examples
        ]
    else:
        lines.append("- none")
    lines += ["", "## Review Examples", ""]
    if review_examples:
        lines += [
            f"- `{row['prompt_id']}`: {', '.join(row.get('warning_reasons', []))}"
            for row in review_examples
        ]
    else:
        lines.append("- none")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.dataset_dir = args.dataset_dir.resolve()
    args.output_dir = (args.output_dir or (args.dataset_dir / "qc")).resolve()
    args.metadata_csv = (args.metadata_csv or (args.dataset_dir / "metadata.csv")).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    result_jsonl = args.output_dir / "teacher_video_qc.jsonl"
    if result_jsonl.exists() and not args.overwrite:
        raise FileExistsError(f"{result_jsonl} already exists; pass --overwrite to replace QC outputs.")

    record_paths = sorted((args.dataset_dir / "records").glob("*.json"))
    if args.limit is not None:
        record_paths = record_paths[: args.limit]
    if not record_paths:
        raise ValueError(f"No sidecar records found under {args.dataset_dir / 'records'}")

    worker_args = {
        "expected_width": args.expected_width,
        "expected_height": args.expected_height,
        "expected_num_frames": args.expected_num_frames,
        "expected_fps": args.expected_fps,
        "frame_tolerance": args.frame_tolerance,
        "fps_tolerance": args.fps_tolerance,
        "duration_tolerance": args.duration_tolerance,
        "min_video_bytes": args.min_video_bytes,
        "warn_min_video_bytes": args.warn_min_video_bytes,
        "hard_min_luma": args.hard_min_luma,
        "hard_max_luma": args.hard_max_luma,
        "warn_min_luma": args.warn_min_luma,
        "warn_max_luma": args.warn_max_luma,
        "hard_min_frame_std": args.hard_min_frame_std,
        "warn_min_frame_std": args.warn_min_frame_std,
        "low_motion_avg_diff": args.low_motion_avg_diff,
        "low_motion_max_diff": args.low_motion_max_diff,
        "high_flicker_luma_std": args.high_flicker_luma_std,
        "sample_frames": args.sample_frames,
    }
    tasks = [
        {
            "dataset_dir": str(args.dataset_dir),
            "record_path": str(path),
            "prompt_id": path.stem,
            "args": worker_args,
        }
        for path in record_paths
    ]

    start = time.time()
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(qc_one, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    elapsed_s = time.time() - start
    results.sort(key=lambda row: row.get("prompt_id", ""))

    summary = summarize(results, args, elapsed_s)
    metadata_rows = load_metadata_rows(args.metadata_csv)
    pass_ids = {row["prompt_id"] for row in results if row["status"] == "pass"}
    review_ids = {row["prompt_id"] for row in results if row["status"] == "review"}
    reject_ids = {row["prompt_id"] for row in results if row["status"] == "reject"}

    def metadata_for(prompt_ids: set[str]) -> list[dict[str, Any]]:
        rows = []
        for prompt_id in sorted(prompt_ids):
            row = metadata_rows.get(prompt_id)
            if row is None:
                row = sidecar_to_metadata_row(args.dataset_dir, prompt_id)
            rows.append(row)
        return rows

    append_jsonl(result_jsonl, results)
    write_json(args.output_dir / "teacher_video_qc_summary.json", summary)
    write_markdown(args.output_dir / "teacher_video_qc_report.md", summary, results)
    write_manifest(args.output_dir / "metadata_qc_pass.csv", metadata_for(pass_ids))
    write_manifest(args.output_dir / "metadata_qc_review.csv", metadata_for(review_ids))
    write_manifest(args.output_dir / "metadata_qc_reject.csv", metadata_for(reject_ids))
    (args.output_dir / "pass_prompt_ids.txt").write_text("\n".join(sorted(pass_ids)) + ("\n" if pass_ids else ""), encoding="utf-8")
    (args.output_dir / "review_prompt_ids.txt").write_text(
        "\n".join(sorted(review_ids)) + ("\n" if review_ids else ""), encoding="utf-8"
    )
    (args.output_dir / "reject_prompt_ids.txt").write_text(
        "\n".join(sorted(reject_ids)) + ("\n" if reject_ids else ""), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
