#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry details, text overlay, watermark, logo, overexposed, "
    "underexposed, frozen motion, duplicate subject, deformed hands, malformed face, "
    "jittery camera, cluttered background"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build a prompt bank for Wan streaming post-training.")
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "rf_prompt_taxonomy_v1.json",
        help="Path to the prompt taxonomy JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory that will receive train/val prompt banks and stats.",
    )
    parser.add_argument("--train-count", type=int, default=20000, help="Number of train prompts.")
    parser.add_argument("--val-count", type=int, default=1000, help="Number of val prompts.")
    parser.add_argument("--seed", type=int, default=20260502, help="Random seed.")
    parser.add_argument("--lang", type=str, default="en", choices=["en"], help="Prompt language.")
    parser.add_argument("--id-prefix", type=str, default="rfv1", help="Prompt id prefix.")
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt attached to every record.",
    )
    return parser.parse_args()


def load_taxonomy(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compose_prompt(rng: random.Random, taxonomy: dict):
    group = rng.choice(taxonomy["subject_groups"])
    style = rng.choice(taxonomy["styles"])
    camera = rng.choice(taxonomy["camera_setups"])
    light = rng.choice(taxonomy["lighting"])
    mood = rng.choice(taxonomy["moods"])
    quality_sample = rng.sample(taxonomy["quality_tags"], k=3)
    subject = rng.choice(group["subjects"])
    action = rng.choice(group["actions"])
    scene = rng.choice(group["scenes"])
    detail = rng.choice(group["details"])

    prompt = (
        f"A {style} of {subject} that {action} {scene}. "
        f"The shot uses {camera} with {light}, and the overall feeling is {mood}. "
        f"{detail}. Emphasize {quality_sample[0]}, {quality_sample[1]}, and {quality_sample[2]}."
    )

    tags = {
        "subject_group": group["name"],
        "subject_type": group["subject_type"],
        "style": style,
        "camera": camera,
        "lighting": light,
        "mood": mood,
        "scene": scene,
    }
    return prompt, tags


def build_records(count: int, split: str, rng: random.Random, taxonomy: dict, args):
    seen_prompts = set()
    records = []
    attempts = 0
    max_attempts = count * 50
    while len(records) < count:
        if attempts > max_attempts:
            raise RuntimeError(f"Unable to generate enough unique prompts for split={split}.")
        attempts += 1
        prompt, tags = compose_prompt(rng, taxonomy)
        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)
        idx = len(records)
        records.append(
            {
                "prompt_id": f"{args.id_prefix}_{split}_{idx:08d}",
                "split": split,
                "lang": args.lang,
                "prompt": prompt,
                "negative_prompt": args.negative_prompt,
                "source": "internal_compositional_taxonomy_v1",
                "license": "internal",
                "seed_hint": rng.randrange(0, 2**31 - 1),
                "tags": tags,
            }
        )
    return records


def write_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    taxonomy = load_taxonomy(args.taxonomy)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_records = build_records(args.train_count, "train", rng, taxonomy, args)
    val_records = build_records(args.val_count, "val", rng, taxonomy, args)

    write_jsonl(args.output_dir / "train.jsonl", train_records)
    write_jsonl(args.output_dir / "val.jsonl", val_records)

    stats = {
        "seed": args.seed,
        "taxonomy": str(args.taxonomy),
        "train_count": len(train_records),
        "val_count": len(val_records),
        "subject_groups": sorted({record["tags"]["subject_group"] for record in train_records + val_records}),
    }
    with (args.output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Wrote prompt banks to {args.output_dir}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
