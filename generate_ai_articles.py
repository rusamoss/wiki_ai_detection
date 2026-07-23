#!/usr/bin/env python3
"""
generate_ai_articles.py

Generates AI-written Wikipedia-style articles via OpenRouter (https://openrouter.ai),
one per title found in a folder of real Wikipedia articles (default:
data/random_2022), using the same prompt for every model: "Write a wikipedia
article about [title]. Do not copy the existing wikipedia article or use it
as a source."

Writes each model's output into its own data/ subfolder (data/claude,
data/gemini, data/gpt), using the same filename as the source article so
run_pangram.py's title/label-based dedup lines up across categories.
Articles already generated for a given model are skipped, not regenerated.
Each generated article's title, model, model slug, prompt, and generation
date are also appended to data/generated_articles.csv.

The real title (not the sanitized filename) is recovered from --source-csv,
the CSV get_test_wiki_pages.py wrote alongside --source-dir, so the prompt
still uses e.g. a colon that had to be stripped from the filename.

Model slugs are OpenRouter identifiers (https://openrouter.ai/models) --
double check these against the current catalog before relying on them, since
exact slugs can change; override a stale guess with --claude-model/
--gemini-model/--gpt-model instead of editing the script.

Usage:
    export OPENROUTER_API_KEY=...        # from https://openrouter.ai/keys
    python generate_ai_articles.py --limit 10
"""

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Any

import requests

from get_test_wiki_pages import (
    DATA_DIR,
    DEFAULT_ARTICLES_DIR,
    DEFAULT_OUT,
    RateLimiter,
    TIMESTAMP_FORMAT,
    sanitize_title,
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OUT_DIR = Path(DATA_DIR)
DEFAULT_GENERATED_CSV = f"{DATA_DIR}/generated_articles.csv"
GENERATED_FIELDNAMES = ["file", "label", "title", "model_slug", "prompt", "date_generated"]

PROMPT_TEMPLATE = (
    "Write a wikipedia article about {title}. Do not copy the existing "
    "wikipedia article or use it as a source."
)

# OpenRouter model slugs -- verify against https://openrouter.ai/models
# before relying on these; override with --claude-model/--gemini-model/
# --gpt-model if a guess here is stale.
DEFAULT_MODELS = {
    "claude": "anthropic/claude-sonnet-5",
    "gemini": "google/gemini-3.6-flash",
    "gpt": "openai/gpt-5.6-terra",
}


def call_openrouter(
    session: requests.Session,
    limiter: RateLimiter,
    model: str,
    prompt: str,
    backoff: float = 2.0,
    max_backoff: float = 120.0,
) -> str:
    """POSTs a chat completion request to OpenRouter, retrying on transient
    failures (429/5xx/timeouts) with exponential backoff, and returns the
    assistant's reply text."""
    while True:
        limiter.wait()
        try:
            resp = session.post(
                OPENROUTER_URL,
                json={"model": model, "messages": [{"role": "user", "content": prompt}]},
                timeout=180,
            )
        except requests.RequestException as e:
            print(f"    [retry] network error calling {model}: {e}; retrying in {backoff:.0f}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)
            continue

        if resp.status_code == 429 or resp.status_code >= 500:
            print(f"    [retry] {model} returned {resp.status_code}; retrying in {backoff:.0f}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)
            continue

        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"OpenRouter error for {model}: {data['error']}")
        return data["choices"][0]["message"]["content"]


def append_generated_row(csv_path: Path, row: dict[str, Any]) -> None:
    """Appends one row to data/generated_articles.csv, writing the header
    first if the file doesn't exist yet (or is empty)."""
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GENERATED_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_titles_by_filename(csv_path: Path) -> dict[str, str]:
    """Maps each article's sanitized ".txt" filename back to its original,
    unsanitized title, from a get_test_wiki_pages.py CSV -- so the prompt
    can use the real title even where the filename had to have characters
    like ":" stripped."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {sanitize_title(row["title"]) + ".txt": row["title"] for row in rows}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--source-dir",
        default=DEFAULT_ARTICLES_DIR,
        help=f"folder of real Wikipedia .txt articles to use as the title list "
             f"(default: {DEFAULT_ARTICLES_DIR})",
    )
    ap.add_argument(
        "--source-csv",
        default=DEFAULT_OUT,
        help=f"the CSV get_test_wiki_pages.py wrote alongside --source-dir, used "
             f"to recover each article's real title from its filename "
             f"(default: {DEFAULT_OUT})",
    )
    ap.add_argument(
        "--limit",
        type=int,
        help="only generate the first N titles (by filename) instead of every "
             "article in --source-dir",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        choices=sorted(DEFAULT_MODELS),
        default=sorted(DEFAULT_MODELS),
        help="which model(s) to generate with (default: all three)",
    )
    ap.add_argument("--claude-model", default=DEFAULT_MODELS["claude"], help="OpenRouter slug used for --models claude")
    ap.add_argument("--gemini-model", default=DEFAULT_MODELS["gemini"], help="OpenRouter slug used for --models gemini")
    ap.add_argument("--gpt-model", default=DEFAULT_MODELS["gpt"], help="OpenRouter slug used for --models gpt")
    ap.add_argument("--min-interval", type=float, default=0.5, help="minimum seconds between OpenRouter requests")
    ap.add_argument(
        "--generated-csv",
        default=DEFAULT_GENERATED_CSV,
        help=f"CSV to append each generated article's title, model, model slug, "
             f"prompt, and generation date to (default: {DEFAULT_GENERATED_CSV})",
    )
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY in your environment (from https://openrouter.ai/keys)")

    model_slugs = {"claude": args.claude_model, "gemini": args.gemini_model, "gpt": args.gpt_model}

    source_dir = Path(args.source_dir)
    txt_files = sorted(source_dir.glob("*.txt"))
    if not txt_files:
        raise SystemExit(f"No .txt files found in {source_dir}")
    if args.limit is not None:
        txt_files = txt_files[:args.limit]

    source_csv = Path(args.source_csv)
    titles_by_filename = load_titles_by_filename(source_csv) if source_csv.exists() else {}

    to_generate: list[tuple[Path, str]] = []
    for txt_path in txt_files:
        for model_key in args.models:
            out_path = DEFAULT_OUT_DIR / model_key / txt_path.name
            if not out_path.exists():
                to_generate.append((txt_path, model_key))

    if not to_generate:
        print("All requested articles already generated; nothing to do.")
        return

    print(f"Generating {len(to_generate)} article(s) across {len(args.models)} "
          f"model(s): {', '.join(args.models)}")
    if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Cancelled.")
        return

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {api_key}"})
    limiter = RateLimiter(min_interval=args.min_interval)

    generated_csv = Path(args.generated_csv)

    succeeded = 0
    for i, (txt_path, model_key) in enumerate(to_generate, 1):
        title = titles_by_filename.get(txt_path.name, txt_path.stem)
        model = model_slugs[model_key]
        prompt = PROMPT_TEMPLATE.format(title=title)
        print(f"[{i}/{len(to_generate)}] {model_key} ({model}): {title}")
        try:
            text = call_openrouter(session, limiter, model, prompt)
        except Exception as e:
            print(f"  [error] {e}; skipping")
            continue
        out_dir = DEFAULT_OUT_DIR / model_key
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / txt_path.name).write_text(text, encoding="utf-8")
        append_generated_row(generated_csv, {
            "file": txt_path.name,
            "label": model_key,
            "title": title,
            "model_slug": model,
            "prompt": prompt,
            "date_generated": time.strftime(TIMESTAMP_FORMAT),
        })
        succeeded += 1

    print(f"\nDone. {succeeded}/{len(to_generate)} succeeded.")


if __name__ == "__main__":
    main()
