#!/usr/bin/env python3
"""
run_pangram.py

Runs Pangram AI-text detection (https://pangram.com) against every .txt
article in a folder (or just the first --limit of them, by filename). With
--all-dirs, runs this once per directory under data/ that directly
contains .txt files (--limit still applies to each one individually)
instead of a single folder -- the normal way to cover every category at
once, since get_test_wiki_pages.py writes each category into its own flat
folder rather than a per-run subfolder. Without --folder or --all-dirs,
falls back to the most recently modified subfolder of --articles-dir (only
relevant if --articles-dir actually has subfolders -- rarely true for the
two conventional categories now). Uses Pangram's Bulk API, which is
cheaper than one predict() call per file.

Before submitting, each article's (title, --label, revision ID) is checked
against rows already in the destination CSV; anything already scored there
is skipped rather than re-submitted. Results (including "date_ran", when
that batch was analyzed) are appended to that CSV rather than overwriting
it (default: data/pangram_results.csv, shared across every folder/category
unless --out says otherwise), so running this repeatedly against different
category folders builds one combined results file. If --label isn't given,
it defaults to the folder's own name (e.g. data/random_2022 -> "random_2022"),
since get_test_wiki_pages.py writes articles straight into a bare category
folder rather than a per-run subfolder.

The revision ID, word count, and URL come from --source-csv: the CSV
get_test_wiki_pages.py wrote alongside this folder (it has "title",
"prose_word_count", "url", "revision_id", and "date_fetched" columns). If
omitted, this is guessed from the label ("random_2022" ->
data/wikipedia_sample.csv, "random_AI_suspected" ->
data/wikipedia_ai_sample.csv) when that file exists. Otherwise -- e.g.
self-generated content, which has no such CSV -- articles are deduped by
title alone and cost estimates fall back to a raw whitespace word count.

Before each bulk submission, prints an estimated cost (5c per 1000 words,
with a 1-credit-per-article minimum) and asks for confirmation. If Pangram
reports the account is out of credits, the script pauses with a link to
buy more instead of failing the run, and retries once you've topped up.

Usage:
    pip install pangram-sdk
    export PANGRAM_API_KEY=...          # from your Pangram account
    python run_pangram.py --folder data/random_2022 --source-csv data/wikipedia_sample.csv
"""

import argparse
import csv
import time
from pathlib import Path

from pangram import Pangram

from get_test_wiki_pages import (
    DATA_DIR,
    DEFAULT_AI_ARTICLES_DIR,
    DEFAULT_AI_OUT,
    DEFAULT_ARTICLES_DIR,
    DEFAULT_OUT,
    TIMESTAMP_FORMAT,
    sanitize_title,
)

CREDITS_URL = "https://www.pangram.com/solutions/api"
DEFAULT_OUT_DIR = Path(DATA_DIR)

FIELDNAMES = [
    "file",
    "label",
    "revision_id",
    "url",
    "prediction_short",
    "fraction_ai",
    "fraction_ai_assisted",
    "fraction_human",
    "num_ai_segments",
    "confidence",
    "error",
    "full_dict",
    "date_ran",
]


def window_confidence(result):
    """Pangram reports confidence ("High"/"Medium"/"Low") per text window,
    not as a single top-level field. Most articles fit in one window; when
    there's more than one, join their confidences in order."""
    windows = result.get("windows") or []
    if not windows:
        return None
    return ";".join(w.get("confidence", "") for w in windows)


def build_result_row(name, label, revision_id, url, result, error, full_dict_source, date_ran):
    """Builds one destination-CSV row for a bulk-job item, whether it
    succeeded (`result` is Pangram's per-article dict) or failed (`result`
    is None)."""
    return {
        "file": name,
        "label": label,
        "revision_id": revision_id,
        "url": url,
        "prediction_short": result.get("prediction_short") if result is not None else None,
        "fraction_ai": result.get("fraction_ai") if result is not None else None,
        "fraction_ai_assisted": result.get("fraction_ai_assisted") if result is not None else None,
        "fraction_human": result.get("fraction_human") if result is not None else None,
        "num_ai_segments": result.get("num_ai_segments") if result is not None else None,
        "confidence": window_confidence(result) if result is not None else None,
        "error": error,
        "full_dict": str(full_dict_source),
        "date_ran": date_ran,
    }


def latest_articles_folder(articles_dir):
    subdirs = [p for p in articles_dir.iterdir() if p.is_dir()]
    if not subdirs:
        raise SystemExit(f"No subfolders found in {articles_dir}")
    # Folder names are strftime("%Y%m%d_%H%M%S") timestamps, so lexicographic
    # order is chronological order.
    return max(subdirs, key=lambda p: p.name)


def find_txt_directories(base_dir):
    """Returns every directory at or under `base_dir` that directly
    contains at least one .txt file, sorted. rglob("*.txt") matches files
    directly in `base_dir` too (not just nested ones), so a `base_dir` with
    no category subfolders -- just .txt files dropped straight into it --
    naturally comes back as [base_dir] with no special-casing needed."""
    return sorted({p.parent for p in base_dir.rglob("*.txt")})


def default_label(folder):
    """Falls back to the folder's own name for --label when none is given
    (e.g. data/random_2022 -> "random_2022", data/claude -> "claude")."""
    return folder.name


# get_test_wiki_pages.py's own --articles-dir/--out defaults (imported, not
# retyped), keyed by the label its folders resolve to via default_label()
# -- lets --source-csv be inferred instead of required, for the two
# conventional categories, and can't drift out of sync with those defaults.
DEFAULT_SOURCE_CSV_BY_LABEL = {
    Path(DEFAULT_ARTICLES_DIR).name: DEFAULT_OUT,
    Path(DEFAULT_AI_ARTICLES_DIR).name: DEFAULT_AI_OUT,
}


def guess_source_csv(label):
    """Returns the conventional get_test_wiki_pages.py output CSV for a
    known category `label`, if that file actually exists in the current
    directory -- otherwise None (e.g. for self-generated content, which
    has no such CSV to guess)."""
    name = DEFAULT_SOURCE_CSV_BY_LABEL.get(label)
    if name and Path(name).exists():
        return name
    return None


def load_source_metadata(path):
    """Returns {title: {"revision_id": ..., "prose_word_count": ..., "url":
    ...}} from a get_test_wiki_pages.py CSV, keyed by sanitize_title(title)
    to match the corresponding .txt filename. "prose_word_count" is that
    script's own prose-only word count (excluding headings/lists), truer
    than a raw whitespace split of the cleaned .txt file."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {
        sanitize_title(row["title"]): {
            "revision_id": row.get("revision_id", ""),
            "prose_word_count": row.get("prose_word_count") or "",
            "url": row.get("url", ""),
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


def process_folder(folder, out_path, args):
    """Runs the full scan/dedup/cost-estimate/submit/append pipeline against
    one folder of .txt articles. Used both for a single --folder run and,
    once per discovered directory, for --all-dirs."""
    label = args.label or default_label(folder)

    txt_files = sorted(folder.glob("*.txt"))
    if not txt_files:
        raise SystemExit(f"No .txt files found in {folder}")

    if args.limit is not None and len(txt_files) > args.limit:
        print(f"  [limit] found {len(txt_files)} .txt files, only looking at "
              f"the first {args.limit}")
        txt_files = txt_files[:args.limit]

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

    source_csv = args.source_csv or guess_source_csv(label)
    if source_csv:
        print(f"  [source-csv] using {source_csv}"
              + ("" if args.source_csv else " (auto-detected from label)"))
    else:
        print("  [warn] no --source-csv given or auto-detected: revision ID/url "
              "blank for every file this run, so it won't match an already-scored "
              "row that recorded a real revision ID")
    source_metadata = load_source_metadata(source_csv) if source_csv else {}

    paths_by_name = {p.name: p for p in txt_files}
    revids = {}
    word_counts = {}
    urls = {}
    for name in paths_by_name:
        meta = source_metadata.get(Path(name).stem, {})
        revids[name] = meta.get("revision_id", "")
        word_counts[name] = meta.get("prose_word_count", "")
        urls[name] = meta.get("url", "")

    to_submit = []
    for name in sorted(paths_by_name):
        if (name, label, revids[name]) in already_scored:
            print(f"  [already scored] {name} (label {label!r}, revision "
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
    date_ran = time.strftime(TIMESTAMP_FORMAT)

    rows = []
    for item in bulk_results["items"]:
        result = item.get("result") or {}
        name = item.get("id")
        rows.append(build_result_row(name, label, revids.get(name, ""), urls.get(name, ""),
                                      result, item.get("error"), result, date_ran))
    for item in bulk_results["failed_items"]:
        name = item.get("id")
        rows.append(build_result_row(name, label, revids.get(name, ""), urls.get(name, ""),
                                      None, item.get("error"), item, date_ran))

    rows.sort(key=lambda r: r["file"])
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if existing_header is None:
            writer.writeheader()
        writer.writerows(rows)

    succeeded = sum(1 for r in rows if r["prediction_short"] is not None)
    print(f"\nDone. {succeeded}/{len(rows)} succeeded. Appended to {out_path} "
          f"({len(paths_by_name) - len(to_submit)} already scored, skipped).")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--folder",
        help="folder of .txt articles to scan (default: most recently modified "
             "subfolder of --articles-dir, if it has any). Ignored if --all-dirs "
             "is given -- prefer that over this for covering every category.",
    )
    ap.add_argument(
        "--articles-dir",
        help="base directory to look for a subfolder in when --folder isn't given "
             "(rarely useful now that get_test_wiki_pages.py's categories are flat "
             "folders, not subfolders -- see --all-dirs)",
    )
    ap.add_argument(
        "--all-dirs",
        action="store_true",
        help=f"process every directory under {DEFAULT_OUT_DIR}/ that directly contains "
             ".txt files (e.g. each category folder), instead of a single --folder -- "
             "--limit still applies to each directory individually",
    )
    ap.add_argument(
        "--out",
        help="CSV path for results, appended to rather than overwritten "
             f"(default: {DEFAULT_OUT_DIR}/pangram_results.csv, shared across "
             "every folder/category)",
    )
    ap.add_argument(
        "--label",
        help="value stamped into every row's 'label' column (e.g. 'suspected_ai', "
             "'pre2022', 'claude') -- makes it easy to tell rows apart after "
             "combining CSVs from separate runs/folders. Defaults to the "
             "folder's own name. With --all-dirs, this is always the "
             "per-directory default, since one label can't fit every directory.",
    )
    ap.add_argument(
        "--source-csv",
        help="the CSV get_test_wiki_pages.py wrote alongside this folder (has "
             "'title', 'prose_word_count', 'url', and 'revision_id' columns), "
             "used to look up each article's revision ID/URL (for dedup and "
             "the output) and word count (for the cost estimate). If omitted, "
             "guessed from the label ('random_2022' -> data/wikipedia_sample.csv, "
             "'random_AI_suspected' -> data/wikipedia_ai_sample.csv) when that "
             "file exists; otherwise articles are deduped by title alone and "
             "costed off a raw whitespace split. With --all-dirs, this is "
             "always guessed per-directory.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        help="only look at the first N .txt files found in each folder (by "
             "filename), even if there are more -- e.g. --limit 25 against a "
             "folder of 50 articles",
    )
    args = ap.parse_args()

    if args.all_dirs and args.label:
        raise SystemExit("--label can't be used with --all-dirs -- each directory needs its own")
    if args.all_dirs and args.source_csv:
        raise SystemExit("--source-csv can't be used with --all-dirs -- each directory needs its own")

    out_path = Path(args.out) if args.out else DEFAULT_OUT_DIR / "pangram_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.all_dirs:
        folders = find_txt_directories(DEFAULT_OUT_DIR)
        if not folders:
            raise SystemExit(f"No .txt files found anywhere under {DEFAULT_OUT_DIR}")
        print(f"[all-dirs] found {len(folders)} director{'y' if len(folders) == 1 else 'ies'} "
              f"with .txt files under {DEFAULT_OUT_DIR}")
        for folder in folders:
            print(f"\n=== {folder} ===")
            process_folder(folder, out_path, args)
        return

    folder = Path(args.folder) if args.folder else latest_articles_folder(Path(args.articles_dir))
    if not folder.is_dir():
        raise SystemExit(f"Not a directory: {folder}")
    process_folder(folder, out_path, args)


if __name__ == "__main__":
    main()
