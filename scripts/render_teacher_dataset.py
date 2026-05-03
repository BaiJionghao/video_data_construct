#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    import imageio  # type: ignore
except Exception:  # noqa: BLE001
    imageio = None


SCRIPT_DIR = Path(__file__).resolve().parent
VIDEO_CODE_DIR = SCRIPT_DIR.parents[1]
DIFFSYNTH_DIR = VIDEO_CODE_DIR / "DiffSynth-Studio"
if str(DIFFSYNTH_DIR) not in sys.path:
    sys.path.insert(0, str(DIFFSYNTH_DIR))

from diffsynth.core import ModelConfig  # noqa: E402
from diffsynth.pipelines.wan_video import WanVideoPipeline  # noqa: E402


DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry details, text overlay, watermark, logo, overexposed, "
    "underexposed, frozen motion, duplicate subject, deformed hands, malformed face, "
    "jittery camera, cluttered background"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Render teacher videos from a prompt bank.")
    parser.add_argument("--prompt-bank", type=Path, required=True, help="JSONL prompt bank path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dataset directory.")
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("/vepfs-mlp2/c20250518/241506050/ckpts/wan21/Wan2.1-T2V-1.3B"),
        help="Local Wan2.1-T2V-1.3B checkpoint directory.",
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--output-ext", type=str, default="mp4", choices=["mp4", "gif"], help="Output container extension.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of prompts to render per forward pass.")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=6.0)
    parser.add_argument("--sigma-shift", type=float, default=8.0)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--tiled", action="store_true", default=True)
    parser.add_argument("--max-items", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing mp4/json files.")
    parser.add_argument(
        "--default-negative-prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Fallback negative prompt if the record omits it.",
    )
    return parser.parse_args()


def get_rank_info():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return local_rank, rank, world_size


def load_prompt_bank(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_pipeline(model_root: Path, device: str):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sharded_dit = sorted(model_root.glob("diffusion_pytorch_model-*-of-*.safetensors"))
    dit_path = [str(path) for path in sharded_dit] if sharded_dit else str(model_root / "diffusion_pytorch_model.safetensors")
    model_configs = [
        ModelConfig(path=dit_path),
        ModelConfig(path=str(model_root / "models_t5_umt5-xxl-enc-bf16.pth")),
        ModelConfig(path=str(model_root / "Wan2.1_VAE.pth")),
    ]
    tokenizer_config = ModelConfig(path=str(model_root / "google" / "umt5-xxl"))
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        redirect_common_files=False,
    )
    return pipe


def save_video(frames, save_path: Path, fps: int, quality: int):
    if not frames:
        raise ValueError("No frames were generated.")
    if save_path.suffix.lower() == ".mp4":
        if imageio is None:
            raise ModuleNotFoundError("imageio is required for mp4 output but is not available.")
        writer = imageio.get_writer(str(save_path), fps=fps, quality=quality)
        try:
            for frame in frames:
                if isinstance(frame, Image.Image):
                    writer.append_data(np.array(frame.convert("RGB")))
                else:
                    writer.append_data(np.array(frame))
        finally:
            writer.close()
        return

    duration_ms = max(1, int(round(1000 / fps)))
    pil_frames = []
    for frame in frames:
        if isinstance(frame, Image.Image):
            pil_frames.append(frame.convert("RGB"))
        else:
            pil_frames.append(Image.fromarray(frame).convert("RGB"))
    pil_frames[0].save(
        save_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def append_jsonl(path: Path, record: dict):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def tensor_to_videos(video_tensor: torch.Tensor):
    if video_tensor.ndim == 4:
        video_tensor = video_tensor.unsqueeze(0)
    if video_tensor.ndim != 5:
        raise ValueError(f"Expected decoded video tensor with 5 dims, got shape {tuple(video_tensor.shape)}")
    video_tensor = ((video_tensor.clamp(-1, 1) + 1.0) * 127.5).round().clamp(0, 255)
    video_tensor = video_tensor.to(device="cpu", dtype=torch.uint8).permute(0, 2, 3, 4, 1).contiguous().numpy()
    return [[Image.fromarray(frame) for frame in sample] for sample in video_tensor]


def chunk_records(records, chunk_size: int):
    for start in range(0, len(records), chunk_size):
        yield records[start : start + chunk_size]


def main():
    args = parse_args()
    local_rank, rank, world_size = get_rank_info()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    records = load_prompt_bank(args.prompt_bank)
    if args.max_items is not None:
        records = records[: args.max_items]
    assigned_records = records[rank::world_size]

    videos_dir = args.output_dir / "videos"
    records_dir = args.output_dir / "records"
    logs_dir = args.output_dir / "logs"
    videos_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = logs_dir / f"manifest_rank{rank:02d}.jsonl"
    failure_path = logs_dir / f"failures_rank{rank:02d}.jsonl"

    print(
        json.dumps(
            {
                "rank": rank,
                "world_size": world_size,
                "local_rank": local_rank,
                "device": device,
                "assigned_items": len(assigned_records),
                "batch_size": args.batch_size,
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    pipe = build_pipeline(args.model_root, device=device)

    pending_records = []
    for record in assigned_records:
        prompt_id = record["prompt_id"]
        video_rel_path = Path("videos") / f"{prompt_id}.{args.output_ext}"
        video_path = args.output_dir / video_rel_path
        sidecar_path = records_dir / f"{prompt_id}.json"

        if not args.overwrite and video_path.exists() and sidecar_path.exists():
            append_jsonl(
                manifest_path,
                {
                    "prompt_id": prompt_id,
                    "status": "skipped_existing",
                    "video_path": str(video_rel_path),
                },
            )
            continue
        pending_records.append(record)

    for batch_index, batch_records in enumerate(chunk_records(pending_records, max(1, args.batch_size))):
        prompts = [record["prompt"] for record in batch_records]
        negative_prompts = [record.get("negative_prompt", args.default_negative_prompt) for record in batch_records]
        seeds = [int(record.get("seed_hint", 0)) + args.seed_offset for record in batch_records]
        start = time.time()
        try:
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
            videos = tensor_to_videos(decoded_videos)
            elapsed = time.time() - start
            if len(videos) != len(batch_records):
                raise RuntimeError(f"Expected {len(batch_records)} decoded videos, got {len(videos)}")
            runtime_per_item = elapsed / len(batch_records)
            for record, seed, frames in zip(batch_records, seeds, videos):
                prompt_id = record["prompt_id"]
                video_rel_path = Path("videos") / f"{prompt_id}.{args.output_ext}"
                video_path = args.output_dir / video_rel_path
                sidecar_path = records_dir / f"{prompt_id}.json"
                save_video(frames, video_path, fps=args.fps, quality=args.quality)
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
                    "runtime_s": round(runtime_per_item, 3),
                    "batch_runtime_s": round(elapsed, 3),
                    "batch_size": len(batch_records),
                    "rank": rank,
                }
                with sidecar_path.open("w", encoding="utf-8") as f:
                    json.dump(sidecar, f, ensure_ascii=False, indent=2)
                append_jsonl(manifest_path, {"prompt_id": prompt_id, "status": "ok", **sidecar})
                print(
                    json.dumps(
                        {
                            "rank": rank,
                            "batch_index": batch_index,
                            "prompt_id": prompt_id,
                            "status": "ok",
                            "runtime_s": round(runtime_per_item, 3),
                            "batch_runtime_s": round(elapsed, 3),
                            "batch_size": len(batch_records),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - start
            for record in batch_records:
                failure_record = {
                    "prompt_id": record["prompt_id"],
                    "status": "failed",
                    "error": repr(exc),
                    "runtime_s": round(elapsed, 3),
                    "batch_size": len(batch_records),
                    "rank": rank,
                }
                append_jsonl(failure_path, failure_record)
                print(json.dumps(failure_record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
