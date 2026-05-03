#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path


PUBLIC_IP_RE = re.compile(
    r"\b("
    r"travis scott|tupac|notorious|dmx|aaliyah|spider[- ]?man|captain america|sims 4|"
    r"tim burton|stanley kubrick|coca|nike|adidas|pokemon|goku|eren yeager|squidward|bigfoot"
    r")\b",
    re.IGNORECASE,
)
SENSITIVE_RE = re.compile(
    r"\b("
    r"confederate|politician|president|atomic explosion|apocalypse|corpse|skull|blood|"
    r"bombing|burning city|military-style planes|sexual|nude|naked"
    r")\b",
    re.IGNORECASE,
)
NO_TEXT_SENTENCE_RE = re.compile(
    r"\s*(?:ensure|make sure|avoid|with)\s+no\s+(?:readable\s+|visible\s+)?"
    r"(?:text|logos?|signage|watermarks?)[^.;]*(?:[.;]|$)",
    re.IGNORECASE,
)
NO_TEXT_INLINE_RE = re.compile(
    r"\s*,?\s*(?:and\s+)?no\s+(?:readable\s+|visible\s+)?"
    r"(?:text|logos?|signage|watermarks?)[^.;,]*(?=[.;,]|$)",
    re.IGNORECASE,
)
LINEAGE_TEXT_FIELDS = ("raw_prompt", "clean_prompt", "pre_llm_prompt", "extended_prompt")


def parse_args():
    parser = argparse.ArgumentParser(description="Post-filter an LLM-extended VidProM prompt bank.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Input LLM-extended prompt bank directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output safe prompt bank directory.")
    parser.add_argument("--splits", type=str, default="train,val", help="Comma-separated split names.")
    parser.add_argument(
        "--check-lineage-text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also reject if raw/clean/pre-LLM lineage text contains public-IP, brand, or sensitive terms.",
    )
    return parser.parse_args()


def lineage_text(record: dict) -> str:
    return "\n".join(str(record.get(field, "")) for field in LINEAGE_TEXT_FIELDS if record.get(field))


def process_split(split: str, args):
    input_path = args.input_dir / f"{split}.jsonl"
    output_path = args.output_dir / f"{split}.jsonl"
    reject_path = args.output_dir / f"{split}_rejected.jsonl"
    kept = []
    rejected = []
    stats = Counter()

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            reasons = []
            prompt = record.get("prompt", "")
            if record.get("llm_extension", {}).get("used_fallback_prompt"):
                reasons.append("fallback_or_rejected")
            if PUBLIC_IP_RE.search(prompt):
                reasons.append("public_ip_or_brand")
            if args.check_lineage_text and PUBLIC_IP_RE.search(lineage_text(record)):
                reasons.append("public_ip_or_brand_lineage")
            if SENSITIVE_RE.search(prompt):
                reasons.append("sensitive_or_violent")
            if args.check_lineage_text and SENSITIVE_RE.search(lineage_text(record)):
                reasons.append("sensitive_or_violent_lineage")
            if reasons:
                record["usable_for_teacher_generation"] = False
                record["post_filter_reject_reasons"] = reasons
                rejected.append(record)
                for reason in reasons:
                    stats[f"reject_{reason}"] += 1
                continue

            new_prompt = NO_TEXT_SENTENCE_RE.sub("", prompt)
            new_prompt = NO_TEXT_INLINE_RE.sub("", new_prompt).strip()
            if new_prompt != prompt:
                record["prompt_before_postprocess"] = prompt
                record["prompt"] = new_prompt
                record["extended_prompt"] = new_prompt
                record["postprocess_notes"] = ["removed_positive_no_text_clause"]
                stats["scrubbed_no_text_clause"] += 1
            record["usable_for_teacher_generation"] = True
            kept.append(record)

    with output_path.open("w", encoding="utf-8") as f:
        for record in kept:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with reject_path.open("w", encoding="utf-8") as f:
        for record in rejected:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats["input_records"] = len(kept) + len(rejected)
    stats["kept"] = len(kept)
    stats["rejected"] = len(rejected)
    return dict(stats)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "splits": {},
    }
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        stats["splits"][split] = process_split(split, args)
    with (args.output_dir / "post_filter_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
