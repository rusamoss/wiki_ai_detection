#!/usr/bin/env python3
"""
run_pangram.py

Runs Pangram AI-text detection (https://pangram.com) against every .txt
article in a folder, defaulting to the most recently generated subfolder of
--articles-dir if none is given. Uses Pangram's Bulk API, which is cheaper
than one predict() call per file.

Before submitting, each article's (title, --label, revision ID) is checked
against rows already in the destination CSV; anything already scored there
is skipped rather than re-submitted. Results are appended to that CSV
rather than overwriting it, so running this repeatedly against the same
--out (e.g. once per category folder) builds one combined results file.

The revision ID and word count come from --source-csv: the CSV
get_test_wiki_pages.py wrote alongside this folder (it has "title",
"prose_word_count", and "revision_id" columns). Without --source-csv,
articles are deduped by title alone -- fine for e.g. self-generated content
that has no Wikipedia revision -- and cost estimates fall back to a raw
whitespace word count.

Before each bulk submission, prints an estimated cost (5c per 1000 words,
with a 1-credit-per-article minimum) and asks for confirmation. If Pangram
reports the account is out of credits, the script pauses with a link to
buy more instead of failing the run, and retries once you've topped up.

Usage:
    pip install pangram-sdk
    export PANGRAM_API_KEY=...          # from your Pangram account
    python run_pangram.py                                # most recent articles/ subfolder
    python run_pangram.py --folder data/random_2022/20260722_103721 \\
        --source-csv wikipedia_sample.csv --label pre2022 --out combined_results.csv
"""

import argparse
import csv
from pathlib import Path

from pangram import Pangram

CREDITS_URL = "https://www.pangram.com/solutions/api"

FIELDNAMES = [
    "file",
    "label",
    "revision_id",
    "prediction_short",
    "fraction_ai",
    "fraction_ai_assisted",
    "fraction_human",
    "num_ai_segments",
    "confidence",
    "error",
    "full_dict",
]


def window_confidence(result):
    """Pangram reports confidence ("High"/"Medium"/"Low") per text window,
    not as a single top-level field. Most articles fit in one window; when
    there's more than one, join their confidences in order."""
    windows = result.get("windows") or []
    if not windows:
        return None
    return ";".join(w.get("confidence", "") for w in windows)


def build_result_row(name, label, revision_id, result, error, full_dict_source):
    """Builds one destination-CSV row for a bulk-job item, whether it
    succeeded (`result` is Pangram's per-article dict) or failed (`result`
    is None)."""
    return {
        "file": name,
        "label": label,
        "revision_id": revision_id,
        "prediction_short": result.get("prediction_short") if result is not None else None,
        "fraction_ai": result.get("fraction_ai") if result is not None else None,
        "fraction_ai_assisted": result.get("fraction_ai_assisted") if result is not None else None,
        "fraction_human": result.get("fraction_human") if result is not None else None,
        "num_ai_segments": result.get("num_ai_segments") if result is not None else None,
        "confidence": window_confidence(result) if result is not None else None,
        "error": error,
        "full_dict": str(full_dict_source),
    }


def latest_articles_folder(articles_dir):
    subdirs = [p for p in articles_dir.iterdir() if p.is_dir()]
    if not subdirs:
        raise SystemExit(f"No subfolders found in {articles_dir}")
    # Folder names are strftime("%Y%m%d_%H%M%S") timestamps, so lexicographic
    # order is chronological order.
    return max(subdirs, key=lambda p: p.name)


def load_source_metadata(path):
    """Returns {title: {"revision_id": ..., "prose_word_count": ...}} from a
    get_test_wiki_pages.py CSV. "prose_word_count" is that script's own
    prose-only word count (excluding headings/lists), truer than a raw
    whitespace split of the cleaned .txt file."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {
        row["title"]: {
            "revision_id": row.get("revision_id", ""),
            "prose_word_count": row.get("prose_word_count") or "",
        }
        for row in rows
    }


CENTS_PER_1000_WORDS = 5
MIN_CREDITS_PER_ARTICLE = 1  # Pangram bills at least one "1000 word" unit per article


def estimate_cost(texts, word_counts=None):
    """Returns (total_words, total_credits, total_cost_dollars) for a dict
    of {name: text}, at Pangram's pricing: 5 cents per 1000 words, with a
    1-credit minimum per article even if it's under 1000 words. Uses each
    file's word count from `word_counts` (get_test_wiki_pages.py's own
    prose_word_count, when available -- see load_source_metadata) instead
    of a raw whitespace split, since that's the truer prose count; falls
    back to splitting the text for anything not in there."""
    word_counts = word_counts or {}
    total_words = 0
    total_credits = 0
    for name, text in texts.items():
        known = word_counts.get(name)
        words = int(known) if known not in (None, "") else len(text.split())
        total_words += words
        total_credits += max(MIN_CREDITS_PER_ARTICLE, -(-words // 1000))  # ceil division
    return total_words, total_credits, total_credits * CENTS_PER_1000_WORDS / 100


def confirm_bulk_cost(texts, word_counts=None):
    """Prints a cost estimate for submitting `texts` as a Pangram bulk job
    and asks the user to confirm. Returns whether to proceed."""
    total_words, total_credits, total_cost = estimate_cost(texts, word_counts)
    print(f"\nEstimated cost: {len(texts)} article(s), {total_words} words total "
          f"-> {total_credits} credit(s) (~${total_cost:.2f} at "
          f"{CENTS_PER_1000_WORDS}c/1000 words, {MIN_CREDITS_PER_ARTICLE}-credit/article minimum)")
    return input("Proceed with this Pangram bulk submission? [y/N] ").strip().lower() in ("y", "yes")


def call_with_credit_pause(fn, *args, **kwargs):
    """Calls fn(*args, **kwargs), pausing to let the user buy more credits
    and retrying instead of failing the whole run if Pangram reports the
    account is out of credits (HTTP 402)."""
    while True:
        try:
            return fn(*args, **kwargs)
        except ValueError as e:
            if "402" not in str(e):
                raise
            print(f"\n[!] Pangram reports insufficient credits: {e}")
            print(f"    Buy more credits at {CREDITS_URL}")
            input("    Press Enter once you've topped up to retry (Ctrl+C to cancel)... ")


def read_existing_results(out_path):
    """Returns (rows, header) already in the destination CSV, or ([], None)
    if it doesn't exist yet (or is empty)."""
    if not out_path.exists() or out_path.stat().st_size == 0:
        return [], None
    with open(out_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--folder",
        help="folder of .txt articles to scan (default: most recent subfolder of --articles-dir)",
    )
    ap.add_argument(
        "--articles-dir",
        default="articles",
        help="base directory containing timestamped article subfolders",
    )
    ap.add_argument(
        "--out",
        help="CSV path for results, appended to rather than overwritten "
             "(default: <folder>/pangram_results.csv)",
    )
    ap.add_argument(
        "--label",
        default="",
        help="value stamped into every row's 'label' column (e.g. 'suspected_ai', "
             "'pre2022', 'claude') -- makes it easy to tell rows apart after "
             "combining CSVs from separate runs/folders",
    )
    ap.add_argument(
        "--source-csv",
        help="the CSV get_test_wiki_pages.py wrote alongside this folder (has "
             "'title', 'prose_word_count', and 'revision_id' columns), used "
             "to look up each article's revision ID (for dedup) and word "
             "count (for the cost estimate). Without this, articles are "
             "deduped by title alone and costed off a raw whitespace split.",
    )
    args = ap.parse_args()

    folder = Path(args.folder) if args.folder else latest_articles_folder(Path(args.articles_dir))
    if not folder.is_dir():
        raise SystemExit(f"Not a directory: {folder}")

    txt_files = sorted(folder.glob("*.txt"))
    if not txt_files:
        raise SystemExit(f"No .txt files found in {folder}")

    out_path = Path(args.out) if args.out else folder / "pangram_results.csv"

    existing_rows, existing_header = read_existing_results(out_path)
    if existing_header is not None and existing_header != FIELDNAMES:
        raise SystemExit(
            f"{out_path} has a different column layout than expected.\n"
            f"  existing: {existing_header}\n  expected: {FIELDNAMES}\n"
            "Rename/move it aside (or delete it) before appending to it."
        )
    # Keyed by (file, label, revision_id) -- label is part of the identity so
    # e.g. a Claude-generated "X.txt" isn't mistaken for an already-scored
    # real Wikipedia "X.txt" just because they share a title.
    already_scored = {
        (row["file"], row.get("label", ""), row.get("revision_id", ""))
        for row in existing_rows
    }

    source_metadata = load_source_metadata(args.source_csv) if args.source_csv else {}
    if not args.source_csv:
        print("  [warn] no --source-csv given: revision ID is blank for every file "
              "this run, so it won't match an already-scored row that recorded a "
              "real revision ID")

    paths_by_name = {p.name: p for p in txt_files}
    revids = {}
    word_counts = {}
    for name in paths_by_name:
        meta = source_metadata.get(Path(name).stem, {})
        revids[name] = meta.get("revision_id", "")
        word_counts[name] = meta.get("prose_word_count", "")

    to_submit = []
    for name in sorted(paths_by_name):
        if (name, args.label, revids[name]) in already_scored:
            print(f"  [already scored] {name} (label {args.label!r}, revision "
                  f"{revids[name] or 'unknown'}): skipping")
        else:
            to_submit.append(name)

    if not to_submit:
        print("All files already scored; nothing new to submit.")
        return

    # Only read the files that actually need submitting -- anything already
    # scored never needs its text loaded at all.
    to_submit_texts = {name: paths_by_name[name].read_text(encoding="utf-8") for name in to_submit}
    if not confirm_bulk_cost(to_submit_texts, word_counts):
        print("Cancelled; nothing submitted.")
        return

    client = Pangram()  # reads PANGRAM_API_KEY from the environment
    items = [{"id": name, "text": to_submit_texts[name]} for name in to_submit]

    print(f"Submitting {len(items)} new files from {folder} as a bulk job...")
    bulk = call_with_credit_pause(client.submit_bulk, items=items)
    bulk_id = bulk["bulk_id"]

    print(f"Bulk job {bulk_id} submitted, waiting for completion...")
    call_with_credit_pause(client.wait_for_bulk, bulk_id)

    bulk_results = call_with_credit_pause(client.get_bulk_results, bulk_id)

    rows = []
    for item in bulk_results["items"]:
        result = item.get("result") or {}
        name = item.get("id")
        rows.append(build_result_row(name, args.label, revids.get(name, ""),
                                      result, item.get("error"), result))
    for item in bulk_results["failed_items"]:
        name = item.get("id")
        rows.append(build_result_row(name, args.label, revids.get(name, ""),
                                      None, item.get("error"), item))

    rows.sort(key=lambda r: r["file"])
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if existing_header is None:
            writer.writeheader()
        writer.writerows(rows)

    succeeded = sum(1 for r in rows if r["prediction_short"] is not None)
    print(f"\nDone. {succeeded}/{len(rows)} succeeded. Appended to {out_path} "
          f"({len(paths_by_name) - len(to_submit)} already scored, skipped).")


if __name__ == "__main__":
    main()
