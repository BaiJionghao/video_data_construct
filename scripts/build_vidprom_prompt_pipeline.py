#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_VIDPROM_CSV = Path("/vepfs-mlp2/c20250518/241506050/data/VidProm/VidProM_unique.csv")
DEFAULT_LLM_CONFIG = ROOT_DIR / "configs" / "llm_config.json"
DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry details, text overlay, watermark, logo, overexposed, "
    "underexposed, frozen motion, duplicate subject, deformed hands, malformed face, "
    "jittery camera, cluttered background"
)
PIPELINE_VERSION = "rf_vidprom_prompt_pipeline_v2"
SPLITS = ("train", "val")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a final-size VidProM prompt bank by orchestrating clean filtering, "
            "LLM extension, post-filtering, and deterministic final sampling."
        )
    )
    parser.add_argument("--vidprom-csv", type=Path, default=DEFAULT_VIDPROM_CSV, help="Path to VidProM_unique.csv.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "artifacts" / "prompt_banks" / "rf_vidprom_prompt_bank_v2",
        help="Final prompt bank directory with exact train/val counts.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT_DIR / "artifacts" / "prompt_banks" / "rf_vidprom_prompt_bank_v2_work",
        help="Intermediate stage directory.",
    )
    parser.add_argument("--train-count", type=int, default=2000, help="Final train prompt count.")
    parser.add_argument("--val-count", type=int, default=200, help="Final val prompt count.")
    parser.add_argument("--seed", type=int, default=20260503, help="Sampling seed.")
    parser.add_argument("--id-prefix", type=str, default="rfvp2", help="Final prompt id prefix.")
    parser.add_argument("--motion-rich-ratio", type=float, default=0.20, help="Target motion-rich ratio per split.")
    parser.add_argument(
        "--oversample-factor",
        type=float,
        default=1.25,
        help="Clean/LLM pool multiplier before post-filtering. Increase if final safe pool is short.",
    )
    parser.add_argument(
        "--min-oversample-extra",
        type=int,
        default=50,
        help="Minimum extra clean prompts per split before LLM/post-filtering.",
    )
    parser.add_argument("--min-words", type=int, default=8, help="Clean-stage minimum prompt word count.")
    parser.add_argument("--max-words", type=int, default=80, help="Clean-stage maximum prompt word count.")
    parser.add_argument("--max-safety", type=float, default=0.05, help="Clean-stage maximum VidProM safety score.")
    parser.add_argument("--keep-aspect-prompts", action="store_true", help="Forwarded to clean-stage builder.")
    parser.add_argument("--negative-prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt for records.")

    parser.add_argument(
        "--llm-config",
        type=Path,
        default=DEFAULT_LLM_CONFIG if DEFAULT_LLM_CONFIG.exists() else None,
        help="Optional JSON config with LLM_API_URL/LLM_BASE_MODEL/LLM_API_KEY or base_url/model/api_key.",
    )
    parser.add_argument("--base-url", type=str, default=os.environ.get("LLM_BASE_URL", ""), help="LLM base URL.")
    parser.add_argument("--api-key", type=str, default=os.environ.get("LLM_API_KEY", ""), help="LLM API key.")
    parser.add_argument("--model", type=str, default=os.environ.get("LLM_MODEL", ""), help="LLM model name.")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature.")
    parser.add_argument("--max-tokens", type=int, default=1000, help="LLM max completion tokens.")
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=os.environ.get("LLM_REASONING_EFFORT", "minimal"),
        help="Optional reasoning effort for compatible models.",
    )
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between LLM requests.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per LLM request.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent LLM requests.")
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic template extension instead of an LLM.")
    parser.add_argument(
        "--check-lineage-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Post-filter raw/clean/pre-LLM text as well as the final prompt.",
    )

    parser.add_argument(
        "--safe-pool-dir",
        type=Path,
        default=None,
        help="Existing safe pool directory for --finalize-only, or override the stage-3 safe pool location.",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="Skip clean/LLM/post-filter stages and sample exact counts from --safe-pool-dir.",
    )
    parser.add_argument(
        "--allow-short-final",
        action="store_true",
        help="Write fewer records if the safe pool cannot satisfy target counts.",
    )
    parser.add_argument(
        "--resume-stages",
        action="store_true",
        help="Reuse an existing clean pool and resume LLM extension in the work directory.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Remove known generated stage/final files before running.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT_DIR / path).resolve()


def first_config_value(config: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = config.get(key)
        if value:
            return str(value)
    return ""


def apply_llm_config(args) -> None:
    if not args.llm_config:
        return
    args.llm_config = resolve_path(args.llm_config)
    if not args.llm_config.exists():
        raise SystemExit(f"LLM config not found: {args.llm_config}")
    with args.llm_config.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise SystemExit(f"LLM config must be a JSON object: {args.llm_config}")
    args.base_url = args.base_url or first_config_value(config, ("base_url", "LLM_BASE_URL", "api_url", "LLM_API_URL"))
    args.api_key = args.api_key or first_config_value(config, ("api_key", "LLM_API_KEY"))
    args.model = args.model or first_config_value(config, ("model", "LLM_MODEL", "base_model", "LLM_BASE_MODEL"))
    if not args.reasoning_effort:
        args.reasoning_effort = first_config_value(config, ("reasoning_effort", "LLM_REASONING_EFFORT"))


def target_counts(args) -> dict[str, int]:
    return {"train": args.train_count, "val": args.val_count}


def pool_count(target: int, args) -> int:
    if target <= 0:
        return 0
    return max(target, int(math.ceil(target * args.oversample_factor)), target + args.min_oversample_extra)


def known_files(directory: Path, stats_name: str, include_rejected: bool = False) -> list[Path]:
    names = [f"{split}.jsonl" for split in SPLITS]
    if include_rejected:
        names.extend(f"{split}_rejected.jsonl" for split in SPLITS)
    names.append(stats_name)
    return [directory / name for name in names]


def unlink_known(files: list[Path]) -> None:
    for path in files:
        if path.exists():
            path.unlink()


def preflight(paths: list[Path], args) -> None:
    existing = [path for path in paths if path.exists()]
    if args.overwrite_existing:
        unlink_known(existing)
        return
    if existing and not args.resume_stages:
        sample = "\n".join(str(path) for path in existing[:8])
        raise SystemExit(
            "Refusing to append over existing generated files. Choose a new --work-dir/--output-dir, "
            "or pass --overwrite-existing.\nExisting files:\n" + sample
        )


def run_step(name: str, command: list[str], env: dict[str, str] | None = None) -> None:
    printable = " ".join(command)
    print(json.dumps({"step": name, "command": printable}, ensure_ascii=False), flush=True)
    subprocess.run(command, cwd=str(ROOT_DIR), env=env, check=True)


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_seed(seed: int, split: str) -> int:
    return seed + sum((idx + 1) * ord(ch) for idx, ch in enumerate(split)) * 1009


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    counters = {
        "source_bucket": Counter(),
        "subject_type": Counter(),
        "scene_type": Counter(),
        "motion_type": Counter(),
        "motion_intensity": Counter(),
        "challenge_tags": Counter(),
    }
    word_lengths = []
    for record in records:
        tags = record.get("tags", {})
        for key in ("source_bucket", "subject_type", "scene_type", "motion_type", "motion_intensity"):
            counters[key][str(tags.get(key, "unknown"))] += 1
        for tag in tags.get("challenge_tags", []) or []:
            counters["challenge_tags"][str(tag)] += 1
        word_lengths.append(len(str(record.get("prompt", "")).split()))
    return {
        "records": len(records),
        "counts": {key: dict(counter.most_common()) for key, counter in counters.items()},
        "word_length": {
            "min": min(word_lengths) if word_lengths else 0,
            "max": max(word_lengths) if word_lengths else 0,
            "avg": round(sum(word_lengths) / len(word_lengths), 2) if word_lengths else 0.0,
        },
    }


def select_final_records(records: list[dict[str, Any]], split: str, target: int, args) -> tuple[list[dict[str, Any]], list[str]]:
    warnings = []
    if len(records) < target and not args.allow_short_final:
        raise SystemExit(
            f"Safe pool for split={split} is short: need {target}, got {len(records)}. "
            "Increase --oversample-factor/--min-oversample-extra and rerun."
        )
    target = min(target, len(records)) if args.allow_short_final else target
    rng = random.Random(split_seed(args.seed, split))
    motion = [record for record in records if record.get("tags", {}).get("source_bucket") == "vidprom_motion_rich"]
    other = [record for record in records if record.get("tags", {}).get("source_bucket") != "vidprom_motion_rich"]
    rng.shuffle(motion)
    rng.shuffle(other)

    desired_motion = min(int(round(target * args.motion_rich_ratio)), len(motion))
    desired_other = min(target - desired_motion, len(other))
    selected = motion[:desired_motion] + other[:desired_other]

    leftovers = motion[desired_motion:] + other[desired_other:]
    rng.shuffle(leftovers)
    while len(selected) < target and leftovers:
        selected.append(leftovers.pop())
    if len(selected) < target and not args.allow_short_final:
        raise SystemExit(f"Unable to fill split={split}: need {target}, got {len(selected)}.")
    if desired_motion < int(round(target * args.motion_rich_ratio)):
        warnings.append(
            f"{split}: motion-rich pool short; selected {desired_motion} of target "
            f"{int(round(target * args.motion_rich_ratio))}."
        )

    rng.shuffle(selected)
    final_records = []
    for idx, record in enumerate(selected):
        out = dict(record)
        source_prompt_id = out.get("prompt_id", "")
        out["source_prompt_id"] = source_prompt_id
        out["pre_selection_source"] = out.get("source", "")
        out["prompt_id"] = f"{args.id_prefix}_{split}_{idx:08d}"
        out["split"] = split
        out["source"] = "vidprom_filtered_llm_extended_safe_v2"
        out["prompt_program_version"] = PIPELINE_VERSION
        out["prompt_selection"] = {
            "seed": args.seed,
            "safe_pool_split_size": len(records),
            "target_count": target,
            "motion_rich_ratio": args.motion_rich_ratio,
        }
        final_records.append(out)
    return final_records, warnings


def build_clean_pool(args, clean_dir: Path, counts: dict[str, int]) -> None:
    if args.resume_stages and all((clean_dir / f"{split}.jsonl").exists() for split in SPLITS):
        print(json.dumps({"step": "clean_pool", "status": "reuse_existing", "dir": str(clean_dir)}, ensure_ascii=False), flush=True)
        return

    command = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "build_vidprom_prompt_bank.py"),
        "--vidprom-csv",
        str(args.vidprom_csv),
        "--output-dir",
        str(clean_dir),
        "--train-count",
        str(counts["train"]),
        "--val-count",
        str(counts["val"]),
        "--seed",
        str(args.seed),
        "--id-prefix",
        "rfvppool",
        "--min-words",
        str(args.min_words),
        "--max-words",
        str(args.max_words),
        "--max-safety",
        str(args.max_safety),
        "--motion-rich-ratio",
        str(args.motion_rich_ratio),
        "--negative-prompt",
        args.negative_prompt,
    ]
    if args.keep_aspect_prompts:
        command.append("--keep-aspect-prompts")
    run_step("clean_pool", command)


def run_llm_extension(args, clean_dir: Path, extended_dir: Path) -> None:
    apply_llm_config(args)
    if not args.dry_run and (not args.base_url or not args.api_key or not args.model):
        raise SystemExit("Non-dry-run mode requires --base-url, --api-key, and --model; or --llm-config; or matching LLM_* env vars.")
    env = os.environ.copy()
    if args.api_key and not args.llm_config:
        env["LLM_API_KEY"] = args.api_key
    command = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "extend_vidprom_prompt_bank_with_llm.py"),
        "--input-dir",
        str(clean_dir),
        "--output-dir",
        str(extended_dir),
        "--splits",
        "train,val",
        "--temperature",
        str(args.temperature),
        "--max-tokens",
        str(args.max_tokens),
        "--reasoning-effort",
        args.reasoning_effort,
        "--sleep",
        str(args.sleep),
        "--max-retries",
        str(args.max_retries),
        "--workers",
        str(args.workers),
        "--seed",
        str(args.seed),
    ]
    if args.llm_config:
        command.extend(["--llm-config", str(args.llm_config)])
    if args.base_url and not args.llm_config:
        command.extend(["--base-url", args.base_url])
    if args.model and not args.llm_config:
        command.extend(["--model", args.model])
    if args.dry_run:
        command.append("--dry-run")
    if args.resume_stages:
        command.append("--resume")
    run_step("llm_extension", command, env=env)


def run_post_filter(args, extended_dir: Path, safe_dir: Path) -> None:
    command = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "post_filter_llm_prompt_bank.py"),
        "--input-dir",
        str(extended_dir),
        "--output-dir",
        str(safe_dir),
        "--splits",
        "train,val",
    ]
    command.append("--check-lineage-text" if args.check_lineage_text else "--no-check-lineage-text")
    run_step("post_filter", command)


def finalize(args, safe_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {
        "prompt_pipeline_version": PIPELINE_VERSION,
        "output_dir": str(output_dir),
        "safe_pool_dir": str(safe_dir),
        "seed": args.seed,
        "id_prefix": args.id_prefix,
        "target_counts": target_counts(args),
        "motion_rich_ratio": args.motion_rich_ratio,
        "splits": {},
        "warnings": [],
    }
    for split, target in target_counts(args).items():
        pool = read_jsonl(safe_dir / f"{split}.jsonl")
        selected, warnings = select_final_records(pool, split, target, args)
        write_jsonl(output_dir / f"{split}.jsonl", selected)
        stats["warnings"].extend(warnings)
        stats["splits"][split] = {
            "safe_pool_records": len(pool),
            "selected_records": len(selected),
            "summary": summarize(selected),
        }
    with (output_dir / "pipeline_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def main():
    args = parse_args()
    if args.oversample_factor < 1.0:
        raise SystemExit("--oversample-factor must be >= 1.0")
    args.vidprom_csv = resolve_path(args.vidprom_csv)
    output_dir = resolve_path(args.output_dir)
    work_dir = resolve_path(args.work_dir)
    clean_dir = work_dir / "01_clean_pool"
    extended_dir = work_dir / "02_llm_extended"
    safe_dir = resolve_path(args.safe_pool_dir) if args.safe_pool_dir else work_dir / "03_safe_pool"

    if args.finalize_only:
        if not args.safe_pool_dir:
            raise SystemExit("--finalize-only requires --safe-pool-dir")
        preflight(known_files(output_dir, "pipeline_stats.json"), args)
        stats = finalize(args, safe_dir, output_dir)
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    pool_counts = {split: pool_count(count, args) for split, count in target_counts(args).items()}
    generated = []
    generated.extend(known_files(clean_dir, "stats.json"))
    generated.extend(known_files(extended_dir, "llm_extension_stats.json"))
    generated.extend(known_files(safe_dir, "post_filter_stats.json", include_rejected=True))
    generated.extend(known_files(output_dir, "pipeline_stats.json"))
    preflight(generated, args)

    clean_dir.mkdir(parents=True, exist_ok=True)
    extended_dir.mkdir(parents=True, exist_ok=True)
    safe_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    build_clean_pool(args, clean_dir, pool_counts)
    run_llm_extension(args, clean_dir, extended_dir)
    run_post_filter(args, extended_dir, safe_dir)
    stats = finalize(args, safe_dir, output_dir)
    stats["work_dir"] = str(work_dir)
    stats["oversample_factor"] = args.oversample_factor
    stats["min_oversample_extra"] = args.min_oversample_extra
    stats["pool_counts_requested"] = pool_counts
    stats["stage_stats"] = {
        "clean": read_json(clean_dir / "stats.json"),
        "llm_extension": read_json(extended_dir / "llm_extension_stats.json"),
        "post_filter": read_json(safe_dir / "post_filter_stats.json"),
    }
    with (output_dir / "pipeline_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
