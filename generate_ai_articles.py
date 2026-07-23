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

Sometimes a model declines or asks a clarifying question instead of writing
an article (e.g. for an obscure title it isn't confident about) -- this is
detected (see looks_like_refusal) and retried a couple of times, since it
seems to be sampling-variance more than a hard block. If every attempt still
looks like a refusal, nothing is written for that title/model, so a later
run will simply try it again instead of a bad file sitting in the dataset.

"See also", "References", "Notes", and other appendix sections
(get_test_wiki_pages.py's own EXCLUDED_SECTION_HEADINGS) are stripped from
every generated article, matching how real Wikipedia articles are cleaned.
So are other formatting artifacts real cleaned Wikipedia prose never has: a
leading heading that just repeats the article's own title, literal
un-rendered Markdown emphasis (**bold**, *italic*), and trailing whitespace.

Some models (GPT, so far) return actual MediaWiki wikitext -- templates,
wikilinks, wikitables, category tags -- instead of plain prose, sometimes
wrapped in a ```-fenced code block with commentary around it. When that's
detected, the fence is stripped and the wikitext is rendered through the
same live MediaWiki parser and get_test_wiki_pages.py's own
extract_cleaned_prose(), so it's cleaned exactly the same way a real
Wikipedia article is (infoboxes/navboxes/tables/references stripped,
headings and lists kept). That render call needs --contact, same as
get_test_wiki_pages.py, but only if wikitext actually turns up.

The real title (not the sanitized filename) is recovered from --source-csv,
the CSV get_test_wiki_pages.py wrote alongside --source-dir, so the prompt
still uses e.g. a colon that had to be stripped from the filename.

Model slugs are OpenRouter identifiers (https://openrouter.ai/models) --
double check these against the current catalog before relying on them, since
exact slugs can change; override with --claude-model/--gemini-model/--gpt-model.

Usage:
    export OPENROUTER_API_KEY=...        # from https://openrouter.ai/keys
    python generate_ai_articles.py --limit 10
"""

import argparse
import csv
import os
import re
import time
from pathlib import Path
from typing import Any

import requests

from get_test_wiki_pages import (
    DATA_DIR,
    DEFAULT_ARTICLES_DIR,
    DEFAULT_OUT,
    EXCLUDED_SECTION_HEADINGS,
    RateLimiter,
    TIMESTAMP_FORMAT,
    api_get,
    extract_cleaned_prose,
    make_session,
    sanitize_title,
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_OUT_DIR = Path(DATA_DIR)
DEFAULT_GENERATED_CSV = f"{DATA_DIR}/generated_articles.csv"
GENERATED_FIELDNAMES = ["file", "label", "title", "model_slug", "prompt", "date_generated"]

PROMPT_TEMPLATE = (
    "Write a wikipedia article about {title}. Do not copy the existing "
    "wikipedia article or use it as a source. Do not include anything in your output other than the text of the article."
)

# OpenRouter model slugs; override with --claude-model/--gemini-model/--gpt-model
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
    """POSTs a request to OpenRouter, retrying on transient failures, and returns the reply text."""
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


def looks_like_refusal(text: str) -> bool:
    """Whether generated text looks like the model declined or asked a
    clarifying question instead of writing an article."""

    # Curated from common documented LLM refusal/hedging phrasing
    REFUSAL_PHRASES = [
        r"i'?m sorry", r"i am sorry", r"i apologi[sz]e",
        r"as an ai\b", r"as a language model", r"as an assistant",
        r"i cannot\b", r"i can'?t\b",
        r"i am unable to", r"i'?m unable to", r"i am not able to", r"i'?m not able to",
        r"i won'?t\b", r"i will not\b", r"i (?:must|have to) decline",
        r"i (?:don'?t|do not) have\b.{0,40}\b(?:information|details)",
        r"i'?m not familiar with", r"i am not familiar with",
        r"i (?:could not|couldn'?t) find", r"i have no information",
        r"let me know (?:how|if)", r"please (?:provide|share|specify|clarify)",
        r"could you (?:clarify|provide|share|confirm)",
        r"can you (?:clarify|provide|tell me|confirm)",
        r"if you (?:can|could) (?:share|provide|clarify)",
        r"feel free to", r"i'?d be (?:glad|happy) to help",
    ]
    REFUSAL_RE = re.compile("|".join(REFUSAL_PHRASES), re.IGNORECASE)
    # Real encyclopedic prose is strictly third-person
    FIRST_PERSON_RE = re.compile(r"\bi\b|\bi'm\b|\bi'd\b|\bi've\b|\bi'll\b|\bmy\b|\bme\b|\bmine\b|\bwe\b|\bour\b", re.IGNORECASE)
    SECOND_PERSON_RE = re.compile(r"\byou\b|\byour\b", re.IGNORECASE)

    return bool(REFUSAL_RE.search(text) and (FIRST_PERSON_RE.search(text) or SECOND_PERSON_RE.search(text)))


def generate_non_refusing(
    session: requests.Session,
    limiter: RateLimiter,
    model: str,
    prompt: str,
    title: str,
    max_retries: int = 2,
) -> str | None:
    """Calls OpenRouter, retrying up to max_retries times if the reply
    looks like a refusal to make an article."""
    for attempt in range(1, max_retries + 2):
        text = call_openrouter(session, limiter, model, prompt)
        if not looks_like_refusal(text):
            return text
        print(f"    [refused] {title!r} looks like a refusal/clarifying "
              f"question (attempt {attempt}/{max_retries + 1})")
    return None


def strip_dividers(text: str) -> str:
    """Removes "---" section-divider lines Gemini likes"""
    lines = [line for line in text.split("\n") if line.strip() != "---"]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines))


def strip_excluded_sections(text: str) -> str:
    """Removes "See also"/"References"/"Notes"/etc."""
    MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

    kept: list[str] = []
    skip_level: int | None = None
    for line in text.split("\n"):
        heading = MARKDOWN_HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            if skip_level is not None:
                if level <= skip_level:
                    skip_level = None
                else:
                    continue
            if heading.group(2).strip().lower() in EXCLUDED_SECTION_HEADINGS:
                skip_level = level
                continue
            kept.append(line)
            continue
        if skip_level is not None:
            continue
        kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept))


def strip_leading_title_heading(text: str, title: str) -> str:
    """Removes a leading heading that just repeats the title."""
    TITLE_HEADING_RE = re.compile(r"^#{1,6}\s+(.*)$")

    lines = text.split("\n")
    if not lines:
        return text
    heading = TITLE_HEADING_RE.match(lines[0])
    if not heading or heading.group(1).strip().lower() != title.strip().lower():
        return text
    rest = lines[1:]
    while rest and not rest[0].strip():
        rest.pop(0)
    return "\n".join(rest)


def strip_markdown_emphasis(text: str) -> str:
    """Removes Markdown **bold**, *italic*"""

    BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
    ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)")
    BULLET_PREFIX_RE = re.compile(r"^(\s*)\*(\s+)(.*)$")

    lines = []
    for line in text.split("\n"):
        bullet = BULLET_PREFIX_RE.match(line)
        prefix, rest = (f"{bullet.group(1)}*{bullet.group(2)}", bullet.group(3)) if bullet else ("", line)
        rest = BOLD_RE.sub(r"\1", rest)
        rest = ITALIC_RE.sub(r"\1", rest)
        lines.append(prefix + rest)
    return "\n".join(lines)


def strip_trailing_whitespace(text: str) -> str:
    """Strips trailing whitespace from every line."""
    return "\n".join(line.rstrip() for line in text.split("\n"))


def strip_code_fence(text: str) -> str:
    """Strips a wrapping ```-fenced code block (and any content outside it"""
    CODE_FENCE_RE = re.compile(r"```(?:\w+)?\n(.*?)\n```", re.DOTALL)

    match = CODE_FENCE_RE.search(text)
    return match.group(1) if match else text


def looks_like_wikitext(text: str) -> bool:
    """Whether generated text uses MediaWiki wikitext syntax; checks for ==, [[, or {{}}."""
    WIKITEXT_MARKERS_RE = re.compile(r"\{\{|\[\[|^==.+==\s*$|^\{\|", re.MULTILINE)

    return bool(WIKITEXT_MARKERS_RE.search(text))


def render_wikitext_to_prose(session: requests.Session, limiter: RateLimiter, wikitext: str) -> str:
    """Renders wikitext through the same MediaWiki parser used for
    real articles and runs the result through get_test_wiki_pages.py's
    extract_cleaned_prose(), so it's cleaned exactly the same way real Wikipedia
    articles are."""
    data = api_get(session, limiter, {
        "action": "parse",
        "text": wikitext,
        "contentmodel": "wikitext",
        "prop": "text",
        "formatversion": "2",
        "disabletoc": 1,
        "disableeditsection": 1,
    })
    html = data.get("parse", {}).get("text", "")
    if not html:
        return wikitext
    _, cleaned = extract_cleaned_prose(html)
    return cleaned


def append_generated_row(csv_path: Path, row: dict[str, Any]) -> None:
    """Appends one row to data/generated_articles.csv."""
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GENERATED_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_titles_by_filename(csv_path: Path) -> dict[str, str]:
    """Maps each article's sanitized ".txt" filename back to its original,
    unsanitized title, from a get_test_wiki_pages.py CSV."""
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
    ap.add_argument(
        "--generated-csv",
        default=DEFAULT_GENERATED_CSV,
        help=f"CSV to append each generated article's title, model, model slug, "
             f"prompt, and generation date to (default: {DEFAULT_GENERATED_CSV})",
    )
    ap.add_argument(
        "--contact",
        help="required only if a model returns raw wikitext -- identifies this "
             "script to the MediaWiki API (same as get_test_wiki_pages.py's "
             "--contact) when rendering that wikitext to clean prose",
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
    limiter = RateLimiter(min_interval=0.5)

    # Only created if some model actually returns raw wikitext -- most don't.
    wiki_session: requests.Session | None = None
    wiki_limiter = RateLimiter()

    generated_csv = Path(args.generated_csv)

    succeeded = 0
    for i, (txt_path, model_key) in enumerate(to_generate, 1):
        title = titles_by_filename.get(txt_path.name, txt_path.stem)
        model = model_slugs[model_key]
        prompt = PROMPT_TEMPLATE.format(title=title)
        print(f"[{i}/{len(to_generate)}] {model_key} ({model}): {title}")
        try:
            text = generate_non_refusing(session, limiter, model, prompt, title)
        except Exception as e:
            print(f"  [error] {e}; skipping")
            continue
        if text is None:
            print(f"  [skip] {model_key} kept declining {title!r}; not writing a "
                  f"file -- rerun later to retry")
            continue
        if model_key == "gemini":
            text = strip_dividers(text)
        text = strip_code_fence(text)
        text = strip_excluded_sections(text)
        if looks_like_wikitext(text):
            if wiki_session is None:
                if not args.contact:
                    raise SystemExit(
                        f"{model_key}'s output for {title!r} looks like raw wikitext "
                        "and needs to be rendered through the MediaWiki API to clean "
                        "it -- pass --contact so it gets a proper User-Agent."
                    )
                wiki_session = make_session(args.contact)
            print(f"    [wikitext] {title!r} looks like raw wikitext; rendering and cleaning it")
            text = render_wikitext_to_prose(wiki_session, wiki_limiter, text)
        else:
            text = strip_markdown_emphasis(text)
        text = strip_leading_title_heading(text, title)
        text = strip_trailing_whitespace(text)
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
