#!/usr/bin/env python3
import argparse
import csv
import hashlib
import html
import json
import random
import re
from collections import Counter
from pathlib import Path


DEFAULT_VIDPROM_CSV = Path("/vepfs-mlp2/c20250518/241506050/data/VidProm/VidProM_unique.csv")
DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry details, text overlay, watermark, logo, overexposed, "
    "underexposed, frozen motion, duplicate subject, deformed hands, malformed face, "
    "jittery camera, cluttered background"
)
SAFETY_COLUMNS = ("toxicity", "obscene", "identity_attack", "insult", "threat", "sexual_explicit")

TEXT_OR_LOGO_RE = re.compile(
    r"\b(text|font|letter|typography|title|caption|words?|logo|watermark|poster|slogan|write|spelled)\b",
    re.IGNORECASE,
)
ASPECT_RE = re.compile(
    r"(?i)(?:^|\s)(?:--?ar\s*=?\s*\d+:\d+|--?aspect(?:-ratio)?\s*=?\s*\d+:\d+|"
    r"\b\d+:\d+\b|\baspect ratio\b|\bscreen size\b|\bvertical\b|\bhorizontal\b|\bportrait\b|\blandscape\b)"
)
ATTACHMENT_RE = re.compile(r"(?i)(Message:\s*\d+\s*Attachment|<img\b|https?://|cdn\.|discord)")
NEGATIVE_RE = re.compile(r"(?i)(?:^|\s)(-neg\b|negative prompt|bad anatomy|deformed|extra limb|poorly drawn)")
COMMAND_RE = re.compile(r"(?i)(?:^|\s)(--?[a-z][\w-]*)(?:\s+[^,\.;]+)?")
SPACE_RE = re.compile(r"\s+")
SENSITIVE_OR_IP_RE = re.compile(
    r"\b("
    r"marilyn monroe|andrew tate|daffy duck|looney|coca-?cola|pepsi|nike|adidas|maotai|"
    r"mac miller|elon musk|trump|biden|obama|putin|xi jinping|"
    r"jesus|christ|muhammad|buddha|"
    r"star wars|harry potter|marvel|disney|pokemon|spongebob|mickey mouse|"
    r"sexy|nude|naked|porn|erotic"
    r")\b",
    re.IGNORECASE,
)

MOTION_RE = re.compile(
    r"\b("
    r"walk|walking|run|running|move|moving|fly|flying|dance|dancing|turn|turning|jump|jumping|"
    r"ride|riding|drive|driving|flow|flowing|fall|falling|rain|snow|wave|waves|camera|zoom|"
    r"pan|track|tracking|chase|rotate|rotating|spin|spinning|sway|drift|glide|cross|approach|"
    r"leave|enter|swim|swimming|blink|blinking|open|opening|close|closing|roll|rolling"
    r")\b",
    re.IGNORECASE,
)
CAMERA_RE = re.compile(r"\b(camera|shot|zoom|pan|tilt|track|tracking|dolly|handheld|close-up|wide)\b", re.IGNORECASE)

SUBJECT_PATTERNS = {
    "human": r"\b(person|people|man|woman|girl|boy|child|children|baby|dancer|worker|runner|crowd|face|hand|hands)\b",
    "animal": r"\b(cat|kitten|dog|puppy|horse|deer|fox|bird|duck|fish|animal|monkey|bear|lion|tiger|zebra)\b",
    "vehicle": r"\b(car|bus|truck|train|airplane|plane|ship|boat|bicycle|bike|scooter|spaceship|rocket)\b",
    "environment": r"\b(ocean|wave|waves|rain|snow|storm|cloud|clouds|forest|mountain|lake|river|waterfall|grass|sky)\b",
    "object": r"\b(cup|watch|book|guitar|phone|chair|table|camera lens|machine|robot|toy|object)\b",
}
SCENE_PATTERNS = {
    "indoor": r"\b(room|kitchen|bedroom|studio|office|workshop|house|home|indoor|inside)\b",
    "city": r"\b(city|street|market|downtown|building|traffic|subway|station|urban)\b",
    "nature": r"\b(forest|mountain|meadow|field|park|tree|trees|grass|lake|river|beach|sky|clouds)\b",
    "water": r"\b(ocean|sea|lake|river|waterfall|rain|storm|underwater)\b",
    "fantasy": r"\b(fantasy|dragon|fairy|magical|dream|surreal|space|alien|robotic|cyberpunk)\b",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Build a filtered VidProM prompt bank for RF data construction.")
    parser.add_argument("--vidprom-csv", type=Path, default=DEFAULT_VIDPROM_CSV, help="Path to VidProM_unique.csv.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for train/val JSONL and stats.")
    parser.add_argument("--train-count", type=int, default=2000, help="Number of train prompts.")
    parser.add_argument("--val-count", type=int, default=200, help="Number of val prompts.")
    parser.add_argument("--seed", type=int, default=20260503, help="Sampling seed.")
    parser.add_argument("--id-prefix", type=str, default="rfvp", help="Prompt id prefix.")
    parser.add_argument("--min-words", type=int, default=8, help="Minimum prompt word count.")
    parser.add_argument("--max-words", type=int, default=80, help="Maximum prompt word count.")
    parser.add_argument("--max-safety", type=float, default=0.05, help="Maximum allowed safety score.")
    parser.add_argument("--motion-rich-ratio", type=float, default=0.20, help="Target motion-rich fraction per split.")
    parser.add_argument("--keep-aspect-prompts", action="store_true", help="Keep prompts that contain aspect/size hints after cleaning.")
    parser.add_argument("--negative-prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt for records.")
    return parser.parse_args()


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def parse_score(value: str) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def clean_prompt(prompt: str) -> str:
    prompt = html.unescape(prompt or "")
    prompt = ATTACHMENT_RE.sub(" ", prompt)
    prompt = ASPECT_RE.sub(" ", prompt)
    prompt = re.sub(r"(?i)(?:^|\s)(-motion|-camera|-gs)\s+[^,\.;]+", " ", prompt)
    prompt = re.sub(r"(?i)(?:^|\s)-neg\s+.*$", " ", prompt)
    prompt = COMMAND_RE.sub(" ", prompt)
    prompt = SPACE_RE.sub(" ", prompt).strip(" ,.;\t\r\n")
    return prompt


def classify_subject(prompt: str) -> str:
    for label, pattern in SUBJECT_PATTERNS.items():
        if re.search(pattern, prompt, re.IGNORECASE):
            return label
    return "scene"


def classify_scene(prompt: str) -> str:
    for label, pattern in SCENE_PATTERNS.items():
        if re.search(pattern, prompt, re.IGNORECASE):
            return label
    return "unknown"


def classify_motion(prompt: str) -> str:
    lower = prompt.lower()
    if re.search(r"\b(rain|snow|wave|waves|clouds?|smoke|mist|fire|water|wind)\b", lower):
        return "environment_dynamics"
    if re.search(r"\b(hand|hands|tool|cook|paint|draw|write|hold|turning the pages|assemble)\b", lower):
        return "fine_interaction"
    if CAMERA_RE.search(prompt):
        return "camera_motion"
    if MOTION_RE.search(prompt):
        return "subject_translation"
    return "static_low_motion"


def motion_intensity(prompt: str) -> str:
    lower = prompt.lower()
    if re.search(r"\b(explosion|running|chase|fast|rapid|chaos|storm|fight|crash|crowd)\b", lower):
        return "high"
    if MOTION_RE.search(prompt):
        return "medium"
    return "low"


def challenge_tags(prompt: str) -> list[str]:
    tags = []
    lower = prompt.lower()
    if MOTION_RE.search(prompt):
        tags.append("basic_motion")
    if CAMERA_RE.search(prompt):
        tags.append("camera_parallax")
    if re.search(r"\b(walk|run|rain|snow|waves?|spin|cycle|dance)\b", lower):
        tags.append("cyclic_motion")
    if re.search(r"\b(behind|occlud|disappear|reappear|hidden|through the crowd)\b", lower):
        tags.append("occlusion_reentry")
    if re.search(r"\b(same|consistent|identity|returning|again)\b", lower):
        tags.append("identity_persistence")
    if re.search(r"\b(two|three|many|crowd|people|animals|ducklings|group)\b", lower):
        tags.append("multi_entity")
    if classify_motion(prompt) == "fine_interaction":
        tags.append("fine_interaction")
    if classify_motion(prompt) == "environment_dynamics":
        tags.append("environment_dynamics")
    if not tags:
        tags.append("static_low_motion")
    return sorted(set(tags))


def reject_reason(raw_prompt: str, clean: str, scores: dict[str, float], args) -> str | None:
    if not clean:
        return "empty_after_cleaning"
    words = clean.split()
    if len(words) < args.min_words:
        return "too_short"
    if len(words) > args.max_words:
        return "too_long"
    if max(scores.values()) > args.max_safety:
        return "unsafe_score"
    if ATTACHMENT_RE.search(raw_prompt):
        return "attachment_or_url"
    if TEXT_OR_LOGO_RE.search(clean):
        return "text_or_logo_request"
    if SENSITIVE_OR_IP_RE.search(clean):
        return "sensitive_or_ip_term"
    if NEGATIVE_RE.search(raw_prompt):
        return "negative_prompt_fragment"
    if not args.keep_aspect_prompts and ASPECT_RE.search(raw_prompt):
        return "aspect_or_size_hint"
    return None


def make_record(row, clean: str, split: str, index: int, source_bucket: str, args, rng: random.Random):
    raw = html.unescape(row.get("prompt", "")).strip()
    scores = {column: parse_score(row.get(column, "")) for column in SAFETY_COLUMNS}
    prompt_hash = stable_hash(clean.lower())
    tags = {
        "source_bucket": source_bucket,
        "subject_type": classify_subject(clean),
        "scene_type": classify_scene(clean),
        "motion_type": classify_motion(clean),
        "camera_motion": "explicit" if CAMERA_RE.search(clean) else "unknown",
        "motion_intensity": motion_intensity(clean),
        "challenge_tags": challenge_tags(clean),
        "length_bucket": "short" if len(clean.split()) < 16 else "medium" if len(clean.split()) <= 40 else "long",
    }
    return {
        "prompt_id": f"{args.id_prefix}_{split}_{index:08d}",
        "split": split,
        "lang": "en",
        "prompt": clean,
        "negative_prompt": args.negative_prompt,
        "source": "vidprom_filtered_clean_v1",
        "source_uuid": row.get("uuid", ""),
        "raw_prompt": raw,
        "clean_prompt": clean,
        "license": "VidProM",
        "seed_hint": rng.randrange(0, 2**31 - 1),
        "prompt_hash": prompt_hash,
        "safety_scores": scores,
        "tags": tags,
    }


def reservoir_add(bucket: list, item, seen: set[str], target_limit: int, rng: random.Random, counter: Counter):
    key = item["prompt_hash"]
    if key in seen:
        counter["duplicate_clean_prompt"] += 1
        return
    seen.add(key)
    counter["accepted_seen"] += 1
    if len(bucket) < target_limit:
        bucket.append(item)
        return
    j = rng.randrange(counter["accepted_seen"])
    if j < target_limit:
        bucket[j] = item


def load_candidates(args, rng: random.Random):
    natural = []
    motion = []
    target_total = args.train_count + args.val_count
    motion_target = int(round(target_total * args.motion_rich_ratio))
    natural_target = target_total - motion_target
    reserve_factor = 6
    natural_limit = max(natural_target * reserve_factor, natural_target + 100)
    motion_limit = max(motion_target * reserve_factor, motion_target + 100)
    stats = Counter()
    seen = set()

    with args.vidprom_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["rows"] += 1
            raw_prompt = row.get("prompt", "")
            scores = {column: parse_score(row.get(column, "")) for column in SAFETY_COLUMNS}
            clean = clean_prompt(raw_prompt)
            reason = reject_reason(raw_prompt, clean, scores, args)
            if reason:
                stats[f"reject_{reason}"] += 1
                continue
            source_bucket = "vidprom_motion_rich" if MOTION_RE.search(clean) or CAMERA_RE.search(clean) else "vidprom_natural"
            item = {
                "row": row,
                "clean": clean,
                "prompt_hash": stable_hash(clean.lower()),
                "source_bucket": source_bucket,
            }
            if source_bucket == "vidprom_motion_rich":
                reservoir_add(motion, item, seen, motion_limit, rng, stats)
            else:
                reservoir_add(natural, item, seen, natural_limit, rng, stats)

    stats["candidate_natural_reservoir"] = len(natural)
    stats["candidate_motion_reservoir"] = len(motion)
    return natural, motion, stats


def split_records(natural, motion, args, rng: random.Random):
    rng.shuffle(natural)
    rng.shuffle(motion)
    train_motion = int(round(args.train_count * args.motion_rich_ratio))
    val_motion = int(round(args.val_count * args.motion_rich_ratio))
    train_natural = args.train_count - train_motion
    val_natural = args.val_count - val_motion

    if len(natural) < train_natural + val_natural:
        raise RuntimeError(f"Not enough natural candidates: need {train_natural + val_natural}, got {len(natural)}")
    if len(motion) < train_motion + val_motion:
        raise RuntimeError(f"Not enough motion-rich candidates: need {train_motion + val_motion}, got {len(motion)}")

    train_items = natural[:train_natural] + motion[:train_motion]
    val_items = natural[train_natural : train_natural + val_natural] + motion[train_motion : train_motion + val_motion]
    rng.shuffle(train_items)
    rng.shuffle(val_items)

    train_records = [
        make_record(item["row"], item["clean"], "train", idx, item["source_bucket"], args, rng)
        for idx, item in enumerate(train_items)
    ]
    val_records = [
        make_record(item["row"], item["clean"], "val", idx, item["source_bucket"], args, rng)
        for idx, item in enumerate(val_items)
    ]
    return train_records, val_records


def write_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def summarize(records):
    counters = {
        "source_bucket": Counter(),
        "subject_type": Counter(),
        "scene_type": Counter(),
        "motion_type": Counter(),
        "motion_intensity": Counter(),
        "challenge_tags": Counter(),
        "length_bucket": Counter(),
    }
    word_lengths = []
    for record in records:
        tags = record["tags"]
        for key in ("source_bucket", "subject_type", "scene_type", "motion_type", "motion_intensity", "length_bucket"):
            counters[key][tags[key]] += 1
        for tag in tags["challenge_tags"]:
            counters["challenge_tags"][tag] += 1
        word_lengths.append(len(record["prompt"].split()))
    return {
        "counts": {key: dict(value.most_common()) for key, value in counters.items()},
        "word_length": {
            "min": min(word_lengths) if word_lengths else 0,
            "max": max(word_lengths) if word_lengths else 0,
            "avg": round(sum(word_lengths) / len(word_lengths), 2) if word_lengths else 0,
        },
    }


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    natural, motion, filter_stats = load_candidates(args, rng)
    train_records, val_records = split_records(natural, motion, args, rng)

    write_jsonl(args.output_dir / "train.jsonl", train_records)
    write_jsonl(args.output_dir / "val.jsonl", val_records)

    stats = {
        "vidprom_csv": str(args.vidprom_csv),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "train_count": len(train_records),
        "val_count": len(val_records),
        "min_words": args.min_words,
        "max_words": args.max_words,
        "max_safety": args.max_safety,
        "motion_rich_ratio": args.motion_rich_ratio,
        "keep_aspect_prompts": args.keep_aspect_prompts,
        "filter_stats": dict(filter_stats),
        "train_summary": summarize(train_records),
        "val_summary": summarize(val_records),
    }
    with (args.output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
