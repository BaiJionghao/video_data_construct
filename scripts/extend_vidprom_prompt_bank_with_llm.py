#!/usr/bin/env python3
import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DEFAULT_LLM_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "llm_config.json"
SYSTEM_PROMPT = """You rewrite raw text-to-video prompts into concise Wan2.1 text-to-video prompts.

Return only valid JSON with keys:
- prompt: the rewritten English prompt
- reject: boolean
- reject_reason: short string
- tags: an object with subject_type, scene_type, motion_type, motion_intensity, challenge_tags

Rules:
1. Preserve the original intent when it is safe and usable.
2. Write one single-shot 5-second video prompt, not a multi-scene story.
3. Include subject, scene, temporal action, camera behavior, lighting, and one consistency constraint.
4. Keep the rewritten prompt between 35 and 85 English words.
5. Avoid visible text, subtitles, logos, watermarks, UI, brand names, public figures, copyrighted characters, explicit sexual content, graphic violence, and hate.
6. If the input is unsafe, mostly text/logo design, only a platform parameter, too incoherent, or cannot be made into a video prompt, set reject=true.
7. Do not add famous names, brands, copyrighted characters, or new sensitive content.
8. Prefer realistic or neutral video language unless the original asks for animation or stylization.
"""

USER_TEMPLATE = """Rewrite this VidProM prompt for a 5-second Wan2.1 text-to-video teacher generation sample.

Original prompt:
{prompt}

Existing heuristic tags:
{tags}
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Extend a VidProM prompt bank with an OpenAI-compatible chat LLM.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Input prompt bank directory with train.jsonl/val.jsonl.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output prompt bank directory.")
    parser.add_argument(
        "--llm-config",
        type=Path,
        default=DEFAULT_LLM_CONFIG if DEFAULT_LLM_CONFIG.exists() else None,
        help="Optional JSON config with LLM_API_URL/LLM_BASE_MODEL/LLM_API_KEY or base_url/model/api_key.",
    )
    parser.add_argument("--base-url", type=str, default=os.environ.get("LLM_BASE_URL", ""), help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", type=str, default=os.environ.get("LLM_API_KEY", ""), help="API key.")
    parser.add_argument("--model", type=str, default=os.environ.get("LLM_MODEL", ""), help="Model name.")
    parser.add_argument("--splits", type=str, default="train,val", help="Comma-separated split names.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max records per split.")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature.")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Max completion tokens.")
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=os.environ.get("LLM_REASONING_EFFORT", "minimal"),
        help="Optional reasoning effort for compatible models.",
    )
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between requests.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per request.")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent LLM requests.")
    parser.add_argument("--seed", type=int, default=20260503, help="Seed for deterministic dry-run fallback.")
    parser.add_argument("--dry-run", action="store_true", help="Do not call LLM; write deterministic template rewrites.")
    parser.add_argument("--resume", action="store_true", help="Skip already written prompt_ids in output JSONL.")
    return parser.parse_args()


def first_config_value(config: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = config.get(key)
        if value:
            return str(value)
    return ""


def apply_llm_config(args):
    if not args.llm_config:
        return
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


def read_jsonl(path: Path, limit: int | None = None):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                if limit is not None and len(records) >= limit:
                    break
    return records


def read_existing_ids(path: Path):
    ids = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                ids.add(json.loads(line)["prompt_id"])
            except Exception:
                continue
    return ids


def normalize_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    if base_url.endswith("/v1"):
        return base_url + "/chat/completions"
    return base_url + "/v1/chat/completions"


def extract_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        text = text[first : last + 1]
    return json.loads(text)


def call_llm(record, args):
    url = normalize_base_url(args.base_url)
    user_prompt = USER_TEMPLATE.format(
        prompt=record["prompt"],
        tags=json.dumps(record.get("tags", {}), ensure_ascii=False),
    )
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "response_format": {"type": "json_object"},
    }
    if args.reasoning_effort:
        payload["reasoning_effort"] = args.reasoning_effort
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    last_error = None
    for attempt in range(args.max_retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            return extract_json(content)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"LLM request failed for {record['prompt_id']}: {last_error}")


def dry_run_extend(record, rng: random.Random):
    prompt = record["prompt"].strip()
    style = "Realistic video"
    if any(word in prompt.lower() for word in ("anime", "cartoon", "clay", "animation", "animated")):
        style = "Stylized animation video"
    camera = rng.choice(["a locked medium shot", "a gentle forward push-in", "a wide establishing shot"])
    lighting = rng.choice(["soft natural light", "warm afternoon light", "balanced cinematic lighting"])
    rewritten = (
        f"{style}. {prompt}. Over five seconds, the main subject performs one clear, continuous action while the "
        f"background remains coherent. The camera uses {camera} with {lighting}. Keep the subject appearance, "
        f"scene layout, and motion direction consistent across the clip."
    )
    return {
        "prompt": rewritten,
        "reject": False,
        "reject_reason": "",
        "tags": record.get("tags", {}),
    }


def validate_result(result):
    if not isinstance(result, dict):
        return False, "result_not_object"
    if result.get("reject"):
        return True, ""
    prompt = str(result.get("prompt", "")).strip()
    if not prompt:
        return False, "missing_prompt"
    words = prompt.split()
    if len(words) < 25:
        return False, "extended_prompt_too_short"
    if len(words) > 110:
        return False, "extended_prompt_too_long"
    return True, ""


def extend_record(record, result, args):
    out = dict(record)
    out["source"] = "vidprom_filtered_llm_extended_v1" if not args.dry_run else "vidprom_filtered_dryrun_extended_v1"
    out["pre_llm_prompt"] = record["prompt"]
    out["clean_prompt"] = record.get("clean_prompt", record["prompt"])
    out["llm_extension"] = {
        "model": args.model if not args.dry_run else "dry-run-template",
        "base_url": args.base_url if not args.dry_run else "",
        "temperature": args.temperature,
        "reject": bool(result.get("reject", False)),
        "reject_reason": str(result.get("reject_reason", "")),
    }
    if result.get("reject"):
        out["prompt"] = record["prompt"]
        out["llm_extension"]["used_fallback_prompt"] = True
    else:
        out["prompt"] = str(result["prompt"]).strip()
        out["extended_prompt"] = out["prompt"]
        out["llm_extension"]["used_fallback_prompt"] = False
    if isinstance(result.get("tags"), dict):
        tags = dict(record.get("tags", {}))
        tags.update(result["tags"])
        out["tags"] = tags
    return out


def write_record(path: Path, record):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_split(split: str, args, rng: random.Random):
    input_path = args.input_dir / f"{split}.jsonl"
    output_path = args.output_dir / f"{split}.jsonl"
    records = read_jsonl(input_path, args.limit)
    done = read_existing_ids(output_path) if args.resume else set()
    stats = Counter()
    pending = []
    for record in records:
        if record["prompt_id"] in done:
            stats["skipped_resume"] += 1
            continue
        pending.append(record)

    def process_one(record):
        if args.dry_run:
            local_rng = random.Random(args.seed + int(record["prompt_id"].rsplit("_", 1)[-1]))
            result = dry_run_extend(record, local_rng)
        else:
            result = call_llm(record, args)
            if args.sleep > 0:
                time.sleep(args.sleep)
        ok, reason = validate_result(result)
        fallback_reason = ""
        if not ok:
            result = {"prompt": record["prompt"], "reject": True, "reject_reason": reason, "tags": record.get("tags", {})}
            fallback_reason = reason
        if result.get("reject"):
            outcome = "rejected_or_fallback"
        else:
            outcome = "extended"
        out = extend_record(record, result, args)
        return outcome, out, fallback_reason

    if args.workers <= 1:
        for record in pending:
            outcome, out, fallback_reason = process_one(record)
            stats[outcome] += 1
            if fallback_reason:
                stats[f"fallback_{fallback_reason}"] += 1
            write_record(output_path, out)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_one, record) for record in pending]
            for future in as_completed(futures):
                outcome, out, fallback_reason = future.result()
                stats[outcome] += 1
                if fallback_reason:
                    stats[f"fallback_{fallback_reason}"] += 1
                write_record(output_path, out)

    stats["input_records"] = len(records)
    stats["written_records"] = stats["extended"] + stats["rejected_or_fallback"]
    return dict(stats)


def main():
    args = parse_args()
    apply_llm_config(args)
    if not args.dry_run and (not args.base_url or not args.api_key or not args.model):
        raise SystemExit(
            "Non-dry-run mode requires --base-url, --api-key, and --model; "
            "or --llm-config; or LLM_BASE_URL/LLM_API_KEY/LLM_MODEL."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    stats = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "model": args.model if not args.dry_run else "dry-run-template",
        "base_url": args.base_url if not args.dry_run else "",
        "temperature": args.temperature,
        "dry_run": args.dry_run,
        "limit": args.limit,
        "workers": args.workers,
        "splits": {},
    }
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        stats["splits"][split] = process_split(split, args, rng)
    with (args.output_dir / "llm_extension_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
