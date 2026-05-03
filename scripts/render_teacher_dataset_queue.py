#!/usr/bin/env python3
"""Queue-based teacher video rendering for multi-machine custom jobs.

Each prompt is claimed through an atomic lock directory on the shared
filesystem. A heartbeat updates the lease while the video is being rendered,
so killed workers do not leave permanent deadlocks.
"""

import argparse
import hashlib
import json
import os
import random
import shutil
import socket
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry details, text overlay, watermark, logo, overexposed, "
    "underexposed, frozen motion, duplicate subject, deformed hands, malformed face, "
    "jittery camera, cluttered background"
)

DEFAULT_MODEL_ROOT = Path("/vepfs-mlp2/c20250518/241506050/ckpts/wan21/Wan2.1-T2V-1.3B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render teacher videos from a prompt bank using shared filesystem leases."
    )
    parser.add_argument("--prompt-bank", type=Path, required=True, help="JSONL prompt bank path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dataset directory.")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--output-ext", type=str, default="mp4", choices=["mp4", "gif"])
    parser.add_argument("--batch-size", type=int, default=1, help="Prompts rendered per forward pass.")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=6.0)
    parser.add_argument("--sigma-shift", type=float, default=8.0)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--tiled", dest="tiled", action="store_true", default=True)
    parser.add_argument("--no-tiled", dest="tiled", action="store_false")
    parser.add_argument("--max-items", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--max-jobs", type=int, default=0, help="Exit after this worker renders N jobs. 0 means no cap.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate complete items.")
    parser.add_argument("--worker-id", type=str, default=None, help="Stable worker id for logs and leases.")
    parser.add_argument("--lease-seconds", type=int, default=1800, help="Seconds before a silent lock is stale.")
    parser.add_argument("--heartbeat-seconds", type=int, default=30)
    parser.add_argument("--idle-sleep", type=float, default=10.0)
    parser.add_argument("--min-video-bytes", type=int, default=1024)
    parser.add_argument(
        "--default-negative-prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Fallback negative prompt if the record omits it.",
    )
    parser.add_argument("--status-only", action="store_true", help="Print queue status and exit.")
    parser.add_argument(
        "--cleanup-stale-locks",
        action="store_true",
        help="Move stale lock dirs aside before printing status or rendering.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not load the model. Write placeholder video files for lock/queue smoke tests only.",
    )
    parser.add_argument("--dry-run-sleep", type=float, default=0.05)
    return parser.parse_args()


def now() -> float:
    return time.time()


def iso_time(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts if ts is not None else now()))


def sanitize_name(value: str, max_len: int = 120) -> str:
    allowed = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            allowed.append(ch)
        else:
            allowed.append("_")
    text = "".join(allowed).strip("._")
    return text[:max_len] or "worker"


def stable_int(value: str) -> int:
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:12], 16)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_prompt_bank(path: Path, max_items: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if max_items is not None and len(records) >= max_items:
                break
    return records


@dataclass(frozen=True)
class DatasetPaths:
    output_dir: Path
    videos_dir: Path
    records_dir: Path
    logs_dir: Path
    locks_dir: Path
    stale_locks_dir: Path
    tmp_dir: Path

    @classmethod
    def create(cls, output_dir: Path) -> "DatasetPaths":
        paths = cls(
            output_dir=output_dir,
            videos_dir=output_dir / "videos",
            records_dir=output_dir / "records",
            logs_dir=output_dir / "logs",
            locks_dir=output_dir / "locks",
            stale_locks_dir=output_dir / "locks" / "_stale",
            tmp_dir=output_dir / "tmp",
        )
        for path in (
            paths.videos_dir,
            paths.records_dir,
            paths.logs_dir,
            paths.locks_dir,
            paths.stale_locks_dir,
            paths.tmp_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return paths


@dataclass
class Lease:
    prompt_id: str
    lock_dir: Path
    token: str
    worker_id: str
    lease_seconds: int
    heartbeat_seconds: int

    @property
    def lease_path(self) -> Path:
        return self.lock_dir / "lease.json"


def expected_video_path(paths: DatasetPaths, prompt_id: str, output_ext: str) -> Path:
    return paths.videos_dir / f"{prompt_id}.{output_ext}"


def expected_sidecar_path(paths: DatasetPaths, prompt_id: str) -> Path:
    return paths.records_dir / f"{prompt_id}.json"


def expected_lock_path(paths: DatasetPaths, prompt_id: str) -> Path:
    return paths.locks_dir / f"{prompt_id}.lock"


def is_complete(
    paths: DatasetPaths,
    prompt_id: str,
    output_ext: str,
    min_video_bytes: int,
    overwrite: bool = False,
) -> bool:
    if overwrite:
        return False
    video_path = expected_video_path(paths, prompt_id, output_ext)
    sidecar_path = expected_sidecar_path(paths, prompt_id)
    if not video_path.exists() or not sidecar_path.exists():
        return False
    try:
        if video_path.stat().st_size < min_video_bytes:
            return False
        with sidecar_path.open("r", encoding="utf-8") as f:
            json.load(f)
    except Exception:  # noqa: BLE001
        return False
    return True


def read_lease(lock_dir: Path) -> dict[str, Any] | None:
    path = lock_dir / "lease.json"
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def make_lease_payload(lease: Lease, state: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    ts = now()
    payload: dict[str, Any] = {
        "prompt_id": lease.prompt_id,
        "token": lease.token,
        "worker_id": lease.worker_id,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "state": state,
        "updated_at": ts,
        "updated_at_iso": iso_time(ts),
        "expires_at": ts + lease.lease_seconds,
        "expires_at_iso": iso_time(ts + lease.lease_seconds),
        "lease_seconds": lease.lease_seconds,
    }
    if extra:
        payload.update(extra)
    return payload


def write_lease(lease: Lease, state: str, extra: dict[str, Any] | None = None) -> None:
    atomic_write_json(lease.lease_path, make_lease_payload(lease, state=state, extra=extra))


def lease_is_stale(lock_dir: Path, lease_seconds: int) -> bool:
    lease = read_lease(lock_dir)
    if not lease:
        try:
            mtime = lock_dir.stat().st_mtime
        except OSError:
            return False
        return now() - mtime > lease_seconds
    expires_at = float(lease.get("expires_at") or 0)
    updated_at = float(lease.get("updated_at") or 0)
    return now() > max(expires_at, updated_at + lease_seconds)


def owns_lock(lease: Lease) -> bool:
    payload = read_lease(lease.lock_dir)
    return bool(payload and payload.get("token") == lease.token and payload.get("worker_id") == lease.worker_id)


def move_stale_lock(paths: DatasetPaths, lock_dir: Path) -> bool:
    stale_name = f"{lock_dir.name}.{int(now())}.{uuid.uuid4().hex[:8]}"
    stale_path = paths.stale_locks_dir / stale_name
    try:
        lock_dir.rename(stale_path)
        return True
    except OSError:
        return False


def cleanup_stale_locks(paths: DatasetPaths, lease_seconds: int) -> int:
    moved = 0
    for lock_dir in sorted(paths.locks_dir.glob("*.lock")):
        if lease_is_stale(lock_dir, lease_seconds) and move_stale_lock(paths, lock_dir):
            moved += 1
    return moved


def try_claim(
    record: dict[str, Any],
    paths: DatasetPaths,
    args: argparse.Namespace,
    worker_id: str,
) -> Lease | None:
    prompt_id = str(record["prompt_id"])
    if is_complete(paths, prompt_id, args.output_ext, args.min_video_bytes, args.overwrite):
        return None

    lock_dir = expected_lock_path(paths, prompt_id)
    token = uuid.uuid4().hex
    try:
        lock_dir.mkdir()
    except FileExistsError:
        if is_complete(paths, prompt_id, args.output_ext, args.min_video_bytes, args.overwrite):
            return None
        if not lease_is_stale(lock_dir, args.lease_seconds):
            return None
        if not move_stale_lock(paths, lock_dir):
            return None
        try:
            lock_dir.mkdir()
        except FileExistsError:
            return None
    except OSError:
        return None

    lease = Lease(
        prompt_id=prompt_id,
        lock_dir=lock_dir,
        token=token,
        worker_id=worker_id,
        lease_seconds=args.lease_seconds,
        heartbeat_seconds=args.heartbeat_seconds,
    )
    write_lease(lease, state="claimed", extra={"claimed_at": now(), "claimed_at_iso": iso_time()})
    return lease


def release_lock(lease: Lease) -> None:
    if not owns_lock(lease):
        return
    try:
        shutil.rmtree(lease.lock_dir)
    except FileNotFoundError:
        return
    except OSError:
        return


class HeartbeatGroup:
    def __init__(self, leases: list[Lease], state: str = "rendering"):
        self.leases = leases
        self.state = state
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def __enter__(self) -> "HeartbeatGroup":
        for lease in self.leases:
            write_lease(lease, self.state)
        interval = max(1.0, min(lease.heartbeat_seconds for lease in self.leases)) if self.leases else 1.0
        self.thread = threading.Thread(target=self._run, args=(interval,), daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)

    def _run(self, interval: float) -> None:
        while not self.stop_event.wait(interval):
            for lease in self.leases:
                if owns_lock(lease):
                    try:
                        write_lease(lease, self.state)
                    except Exception:  # noqa: BLE001
                        pass


def status_report(
    records: list[dict[str, Any]],
    paths: DatasetPaths,
    args: argparse.Namespace,
) -> dict[str, Any]:
    complete = 0
    incomplete = 0
    for record in records:
        prompt_id = str(record["prompt_id"])
        if is_complete(paths, prompt_id, args.output_ext, args.min_video_bytes, overwrite=False):
            complete += 1
        else:
            incomplete += 1

    active_locks = 0
    stale_locks = 0
    lock_examples = []
    for lock_dir in sorted(paths.locks_dir.glob("*.lock")):
        stale = lease_is_stale(lock_dir, args.lease_seconds)
        if stale:
            stale_locks += 1
        else:
            active_locks += 1
        if len(lock_examples) < 10:
            lease = read_lease(lock_dir) or {}
            lock_examples.append(
                {
                    "lock": lock_dir.name,
                    "stale": stale,
                    "worker_id": lease.get("worker_id", ""),
                    "updated_at_iso": lease.get("updated_at_iso", ""),
                    "expires_at_iso": lease.get("expires_at_iso", ""),
                }
            )

    return {
        "prompt_bank": str(args.prompt_bank),
        "output_dir": str(args.output_dir),
        "target_count": len(records),
        "complete": complete,
        "incomplete": incomplete,
        "active_locks": active_locks,
        "stale_locks": stale_locks,
        "done": complete >= len(records),
        "lock_examples": lock_examples,
    }


def shuffled_records(records: list[dict[str, Any]], worker_id: str, pass_index: int) -> list[dict[str, Any]]:
    indices = list(range(len(records)))
    rng = random.Random(stable_int(f"{worker_id}:{pass_index}"))
    rng.shuffle(indices)
    return [records[index] for index in indices]


def claim_batch(
    records: list[dict[str, Any]],
    paths: DatasetPaths,
    args: argparse.Namespace,
    worker_id: str,
    pass_index: int,
) -> list[tuple[dict[str, Any], Lease]]:
    claims: list[tuple[dict[str, Any], Lease]] = []
    for record in shuffled_records(records, worker_id, pass_index):
        lease = try_claim(record, paths, args, worker_id)
        if lease is None:
            continue
        claims.append((record, lease))
        if len(claims) >= max(1, args.batch_size):
            break
    return claims


def load_renderer():
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from render_teacher_dataset import build_pipeline, save_video, tensor_to_videos  # noqa: PLC0415

    return build_pipeline, save_video, tensor_to_videos


def write_dry_run_video(path: Path, sleep_seconds: float) -> None:
    time.sleep(max(0.0, sleep_seconds))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = ("dry-run teacher video placeholder\n" * 64).encode("utf-8")
    with path.open("wb") as f:
        f.write(payload)


def render_batch(
    pipe: Any,
    batch_records: list[dict[str, Any]],
    args: argparse.Namespace,
    device: str,
) -> list[tuple[dict[str, Any], int, Any]]:
    prompts = [record["prompt"] for record in batch_records]
    negative_prompts = [record.get("negative_prompt", args.default_negative_prompt) for record in batch_records]
    seeds = [int(record.get("seed_hint", 0)) + args.seed_offset for record in batch_records]
    decoded_videos = pipe(
        prompt=prompts,
        negative_prompt=negative_prompts,
        seed=seeds,
        rand_device=device,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        sigma_shift=args.sigma_shift,
        tiled=args.tiled,
        output_type="floatpoint",
    )
    _, _, tensor_to_videos = load_renderer()
    videos = tensor_to_videos(decoded_videos)
    if len(videos) != len(batch_records):
        raise RuntimeError(f"Expected {len(batch_records)} decoded videos, got {len(videos)}")
    return list(zip(batch_records, seeds, videos))


def commit_item(
    record: dict[str, Any],
    lease: Lease,
    seed: int,
    frames: Any,
    paths: DatasetPaths,
    args: argparse.Namespace,
    worker_id: str,
    worker_tmp_dir: Path,
    runtime_s: float,
    batch_runtime_s: float,
    batch_size: int,
    rank: int,
) -> str:
    prompt_id = str(record["prompt_id"])
    final_video_path = expected_video_path(paths, prompt_id, args.output_ext)
    final_sidecar_path = expected_sidecar_path(paths, prompt_id)
    tmp_video_path = worker_tmp_dir / f"{prompt_id}.{lease.token}.tmp.{args.output_ext}"
    tmp_sidecar_path = worker_tmp_dir / f"{prompt_id}.{lease.token}.tmp.json"
    video_rel_path = Path("videos") / f"{prompt_id}.{args.output_ext}"

    if args.dry_run:
        write_dry_run_video(tmp_video_path, args.dry_run_sleep)
    else:
        _, save_video, _ = load_renderer()
        save_video(frames, tmp_video_path, fps=args.fps, quality=args.quality)

    sidecar = {
        **record,
        "seed": seed,
        "video_path": str(video_rel_path),
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "cfg_scale": args.cfg_scale,
        "num_inference_steps": args.num_inference_steps,
        "sigma_shift": args.sigma_shift,
        "runtime_s": round(runtime_s, 3),
        "batch_runtime_s": round(batch_runtime_s, 3),
        "batch_size": batch_size,
        "queue_worker_id": worker_id,
        "queue_lock_token": lease.token,
        "rendered_at": iso_time(),
        "rank": rank,
    }
    atomic_write_json(tmp_sidecar_path, sidecar)

    if not owns_lock(lease):
        tmp_video_path.unlink(missing_ok=True)
        tmp_sidecar_path.unlink(missing_ok=True)
        return "lost_lock_after_render"

    if is_complete(paths, prompt_id, args.output_ext, args.min_video_bytes, args.overwrite):
        tmp_video_path.unlink(missing_ok=True)
        tmp_sidecar_path.unlink(missing_ok=True)
        return "completed_elsewhere"

    os.replace(tmp_video_path, final_video_path)
    os.replace(tmp_sidecar_path, final_sidecar_path)
    return "ok"


def detect_device() -> tuple[str, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", os.environ.get("QUEUE_WORKER_SLOT", "0")))
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            return f"cuda:{local_rank}", local_rank, rank
    except Exception:  # noqa: BLE001
        pass
    return "cpu", local_rank, rank


def main() -> None:
    args = parse_args()
    records = load_prompt_bank(args.prompt_bank, args.max_items)
    if not records:
        raise ValueError(f"No prompt records found in {args.prompt_bank}")
    paths = DatasetPaths.create(args.output_dir)
    if args.cleanup_stale_locks:
        cleanup_stale_locks(paths, args.lease_seconds)

    worker_id = args.worker_id or (
        f"{socket.gethostname()}_pid{os.getpid()}_slot{os.environ.get('QUEUE_WORKER_SLOT', '0')}_{uuid.uuid4().hex[:8]}"
    )
    worker_id = sanitize_name(worker_id)
    worker_tmp_dir = paths.tmp_dir / worker_id
    worker_tmp_dir.mkdir(parents=True, exist_ok=True)
    event_log_path = paths.logs_dir / f"queue_worker_{worker_id}.jsonl"

    if args.status_only:
        print(json.dumps(status_report(records, paths, args), ensure_ascii=False, indent=2))
        return

    device, local_rank, rank = detect_device()
    startup_event = {
        "event": "worker_start",
        "worker_id": worker_id,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "device": device,
        "local_rank": local_rank,
        "rank": rank,
        "target_count": len(records),
        "output_dir": str(args.output_dir),
        "dry_run": args.dry_run,
        "time": iso_time(),
    }
    append_jsonl(event_log_path, startup_event)
    print(json.dumps(startup_event, ensure_ascii=False), flush=True)

    pipe = None
    jobs_rendered = 0
    pass_index = 0
    last_status_print = 0.0

    while True:
        report = status_report(records, paths, args)
        if report["done"]:
            append_jsonl(event_log_path, {"event": "queue_complete", "worker_id": worker_id, **report, "time": iso_time()})
            print(json.dumps({"event": "queue_complete", **report}, ensure_ascii=False), flush=True)
            break
        if args.max_jobs and jobs_rendered >= args.max_jobs:
            append_jsonl(
                event_log_path,
                {"event": "max_jobs_reached", "worker_id": worker_id, "jobs_rendered": jobs_rendered, **report, "time": iso_time()},
            )
            break

        claims = claim_batch(records, paths, args, worker_id, pass_index)
        pass_index += 1
        if not claims:
            ts = now()
            if ts - last_status_print > 60:
                idle_event = {"event": "idle_wait", "worker_id": worker_id, **report, "time": iso_time(ts)}
                append_jsonl(event_log_path, idle_event)
                print(json.dumps(idle_event, ensure_ascii=False), flush=True)
                last_status_print = ts
            time.sleep(max(1.0, args.idle_sleep))
            continue

        batch_records = [record for record, _lease in claims]
        leases = [lease for _record, lease in claims]
        prompt_ids = [lease.prompt_id for lease in leases]
        append_jsonl(
            event_log_path,
            {"event": "claimed", "worker_id": worker_id, "prompt_ids": prompt_ids, "time": iso_time()},
        )

        start = now()
        try:
            with HeartbeatGroup(leases, state="rendering"):
                if args.dry_run:
                    rendered_items = [(record, int(record.get("seed_hint", 0)) + args.seed_offset, None) for record in batch_records]
                else:
                    if pipe is None:
                        build_pipeline, _save_video, _tensor_to_videos = load_renderer()
                        pipe = build_pipeline(args.model_root, device=device)
                    rendered_items = render_batch(pipe, batch_records, args, device=device)

                batch_runtime_s = now() - start
                runtime_per_item = batch_runtime_s / max(1, len(rendered_items))
                for (record, seed, frames), lease in zip(rendered_items, leases):
                    status = commit_item(
                        record=record,
                        lease=lease,
                        seed=seed,
                        frames=frames,
                        paths=paths,
                        args=args,
                        worker_id=worker_id,
                        worker_tmp_dir=worker_tmp_dir,
                        runtime_s=runtime_per_item,
                        batch_runtime_s=batch_runtime_s,
                        batch_size=len(rendered_items),
                        rank=rank,
                    )
                    if status == "ok":
                        jobs_rendered += 1
                    event = {
                        "event": "render_result",
                        "worker_id": worker_id,
                        "prompt_id": lease.prompt_id,
                        "status": status,
                        "runtime_s": round(runtime_per_item, 3),
                        "batch_runtime_s": round(batch_runtime_s, 3),
                        "time": iso_time(),
                    }
                    append_jsonl(event_log_path, event)
                    print(json.dumps(event, ensure_ascii=False), flush=True)
        except Exception as exc:  # noqa: BLE001
            elapsed = now() - start
            error = {
                "event": "render_failed",
                "worker_id": worker_id,
                "prompt_ids": prompt_ids,
                "status": "failed",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "runtime_s": round(elapsed, 3),
                "time": iso_time(),
            }
            append_jsonl(event_log_path, error)
            print(json.dumps(error, ensure_ascii=False), flush=True)
        finally:
            for lease in leases:
                release_lock(lease)

    final_report = status_report(records, paths, args)
    final_event = {
        "event": "worker_exit",
        "worker_id": worker_id,
        "jobs_rendered": jobs_rendered,
        "time": iso_time(),
        **final_report,
    }
    append_jsonl(event_log_path, final_event)
    print(json.dumps(final_event, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
