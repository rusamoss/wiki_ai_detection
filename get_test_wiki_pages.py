#!/usr/bin/env python3
"""
get_test_wiki_pages.py

Generates a random sample of English Wikipedia articles that satisfy:

  1. Created (first revision) at or before the end of October 2022.
  2. Had between 100 and 3750 words of "readable prose" as of the end of
     October 2022; excluding infoboxes, tables, etc.
  3. Are not disambiguation pages, list articles, set index articles,
     or other navigational pages (heuristic: title pattern + pageprops
     + category checks; not a perfect replica of Wikipedia's own
     classification, so spot-check results if precision matters a lot).

Usage:
    python get_test_wiki_pages.py --n 25 --out sample.csv

Optionally, with --ai-n, also collects a separate random sample of articles
from Category:Articles containing suspected AI-generated texts (a Wikipedia
maintenance category for articles human editors have flagged as likely
AI-generated). These are processed the same way, except there's no
creation-date cutoff and each article's *current* revision is used instead
of its Oct-2022 one. Written to their own --ai-articles-dir folder and
--ai-out CSV so they're never mixed with the dated sample:

    python get_test_wiki_pages.py --n 25 --ai-n 25 --out sample.csv --ai-out ai_sample.csv

Articles land in --articles-dir/--ai-articles-dir, and both that folder and its CSV accumulate across
runs -- collecting a title that's already there overwrites its .txt file
and CSV row (with a warning) rather than duplicating it. Each CSV row also
records "date_fetched", when that article's text was (re)written.
"""

import argparse
import csv
import random
import re
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup, Tag

API_URL = "https://en.wikipedia.org/w/api.php"
OCT_SNAPSHOT = "2022-10-31T23:59:59Z"
MIN_WORDS = 100
MAX_WORDS = 3750
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# Default output locations
DATA_DIR = "data"
DEFAULT_ARTICLES_DIR = f"{DATA_DIR}/random_2022"
DEFAULT_OUT = f"{DATA_DIR}/wikipedia_sample.csv"
DEFAULT_AI_ARTICLES_DIR = f"{DATA_DIR}/random_AI_suspected"
DEFAULT_AI_OUT = f"{DATA_DIR}/wikipedia_ai_sample.csv"

# "source" column value for the dated sample
RANDOM_SOURCE_LABEL = f"Random; {time.strftime('%b %Y', time.strptime(OCT_SNAPSHOT, '%Y-%m-%dT%H:%M:%SZ'))} revision"

# Conservative floor for pre-filtering stubs before paying for an
# action=parse call. Even 100 words of plain English text with *zero*
# markup needs on the order of 500-600 bytes (~5-6 bytes/word incl. the
# space); every real revision carries some markup overhead (categories,
# stub templates, etc.) on top of that. So a revision whose raw wikitext
# is under this many bytes cannot plausibly have 100+ prose words once
# rendered, and can be discarded without ever calling action=parse.
MIN_WIKITEXT_BYTES = 550

LIST_TITLE_RE = re.compile(
    r"^(List|Lists|Index|Indices|Outline|Outlines|Glossary|Glossaries|"
    r"Timeline|Timelines|Bibliography|Filmography|Discography) of\b",
    re.IGNORECASE,
)
LIST_CATEGORY_RE = re.compile(
    r"(set index articles|lists of|disambiguation pages|internal set index|"
    r"glossaries|outlines of|indexes of)",
    re.IGNORECASE,
)

# Hidden maintenance container category for suspected-AI-generated text. Its
# actual tagged articles live in monthly dated subcategories (e.g. "...from
# May 2025"), not directly in this page -- see get_category_members().
AI_SUSPECTED_CATEGORY = "Category:Articles containing suspected AI-generated texts"

# Elements that are not "readable prose" -- infoboxes, navboxes, tables, references, images/galleries, TOC, etc.
STRIP_SELECTORS = [
    "table",                      # infoboxes, wikitables, navboxes-as-tables
    ".navbox", ".vertical-navbox", ".navbox-inner",
    ".infobox",
    ".metadata", ".ambox", ".ombox", ".tmbox", ".cmbox", ".dmbox",
    ".hatnote", ".dablink", ".rellink",
    ".sistersitebox",
    ".reflist", ".references", "ol.references", "cite", "sup.reference",
    ".autonumber",                # bare external links MediaWiki renders as "[N]"
    ".mw-editsection",
    ".toc", "#toc",
    ".thumb", ".thumbinner", "figure", "figcaption",
    ".gallery",
    "style", "script",
    ".noprint",
    ".mw-empty-elt",
    # Note: headings (h1-h6) are deliberately NOT stripped
]


def make_session(contact: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": f"AiTestArticleSampler/1.0 ({contact})",
    })
    return s


class RateLimiter:
    """Minimum-interval throttle between successive API requests."""
    def __init__(self, min_interval: float = 0.05) -> None:
        self.min_interval = min_interval
        self.last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        delta = now - self.last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self.last = time.monotonic()


def api_get(
    session: requests.Session,
    limiter: RateLimiter,
    params: dict[str, Any],
    backoff: float = 2.0,
    max_backoff: float = 120.0,
) -> dict[str, Any]:
    """GET against the MediaWiki API. Transient failures (5xx, 429, timeouts,
    dropped connections, maxlag) are retried indefinitely with capped
    exponential backoff. Real API error responses (e.g. a malformed query)
    are raised immediately since retrying won't fix those.
    """
    params = {**params, "format": "json", "maxlag": 5}
    attempt = 0
    while True:
        attempt += 1
        limiter.wait()

        try:
            r = session.get(API_URL, params=params, timeout=45)
        except requests.RequestException as e:
            print(f"  [warn] request failed (attempt {attempt}): {e}, retrying...", file=sys.stderr)
            time.sleep(min(backoff * attempt, max_backoff))
            continue

        if r.status_code != 200:
            print(f"  [warn] HTTP {r.status_code} from API (attempt {attempt}), retrying...", file=sys.stderr)
            time.sleep(min(backoff * attempt, max_backoff))
            continue

        data = r.json()
        if "error" in data:
            if data["error"].get("code") == "maxlag":
                time.sleep(min(backoff * attempt, max_backoff))
                continue
            raise RuntimeError(data["error"])
        return data


def chunked(seq: list[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def print_discarded(titles: Iterable[str], reason: str) -> None:
    for title in sorted(titles):
        print(f"  [discard: {reason}] {title}")


# ---------------------------------------------------------------------
# Step 1: random candidate titles
# ---------------------------------------------------------------------

def get_random_titles(session: requests.Session, limiter: RateLimiter, n: int) -> list[str]:
    titles: list[str] = []
    while len(titles) < n:
        batch = min(500, n - len(titles)) # 500 is the cap on results per req
        data = api_get(session, limiter, {
            "action": "query",
            "list": "random",
            "rnnamespace": 0,
            "rnfilterredir": "nonredirects",
            "rnlimit": batch,
        })
        titles.extend(p["title"] for p in data["query"]["random"])
    return titles


# ---------------------------------------------------------------------
# Step 2/3: batched navigational-page + redirect filter
# ---------------------------------------------------------------------

def filter_navigational_and_redirects(
    session: requests.Session, limiter: RateLimiter, titles: list[str]
) -> dict[str, tuple[int, int]]:
    """Returns {title: (revid, size)} for the subset of `titles` that are
    NOT redirects, NOT disambiguation pages, and NOT list/set-index/outline
    -type pages. prop=info includes each survivor's current lastrevid/length
    at no extra request cost; callers that want the *current* revision (not
    a historical snapshot) can use that directly instead of a separate
    get_revision_ids lookup."""
    survivors: dict[str, tuple[int, int]] = {}
    for chunk in chunked(titles, 50):
        chunk = [t for t in chunk if not LIST_TITLE_RE.search(t)]
        if not chunk:
            continue
        data = api_get(session, limiter, {
            "action": "query",
            "prop": "info|pageprops|categories",
            "titles": "|".join(chunk),
            "cllimit": 500,
            "formatversion": "2",
        })
        pages = data.get("query", {}).get("pages", [])
        for page in pages:
            if page.get("missing") or page.get("invalid"):
                continue
            if page.get("redirect"):
                continue
            props = page.get("pageprops", {})
            if "disambiguation" in props:
                continue
            cats = [c["title"] for c in page.get("categories", [])]
            if any(LIST_CATEGORY_RE.search(c) for c in cats):
                continue
            survivors[page["title"]] = (page["lastrevid"], page["length"])
    return survivors


# ---------------------------------------------------------------------
# Step 4: get one revision per title, as of a given snapshot date. Only
# needed for a historical snapshot: MediaWiki's rvstart/rvdir require a
# single-page query, so titles can't be batched the way
# filter_navigational_and_redirects's prop=info call is. When the *current*
# revision is good enough, filter_navigational_and_redirects's own
# lastrevid/length already cover it -- skip this function entirely.
#
# As a side effect, this also enforces a creation-date cutoff: if a revision
# exists at/before rvstart, the page must have been created at or before
# that point too (its very first revision can't postdate a later one).
# ---------------------------------------------------------------------

def get_revision_ids(
    session: requests.Session, limiter: RateLimiter, titles: Iterable[str], rvstart: str
) -> dict[str, tuple[int, int]]:
    """Returns {title: (revid, size)} for each title's latest revision
    at/before `rvstart`, where `size` is the revision's wikitext byte length
    (rvprop=size costs nothing extra here and lets callers pre-filter stubs
    before paying for an action=parse call).
    """
    result: dict[str, tuple[int, int]] = {}
    for title in titles:
        data = api_get(session, limiter, {
            "action": "query",
            "prop": "revisions",
            "titles": title,
            "rvlimit": 1,
            "rvprop": "ids|size",
            "rvstart": rvstart,
            "rvdir": "older",
            "formatversion": "2",
        })
        pages = data.get("query", {}).get("pages", [])
        if not pages:
            continue
        page = pages[0]
        if page.get("missing") or "revisions" not in page or not page["revisions"]:
            continue
        rev = page["revisions"][0]
        result[page["title"]] = (rev["revid"], rev["size"])
    return result


# ---------------------------------------------------------------------
# Suspected-AI-text category members, used by --ai-n instead of step 1.
# ---------------------------------------------------------------------

def get_category_members(session: requests.Session, limiter: RateLimiter, category: str) -> dict[str, str]:
    """Returns {title: source_category} for all namespace-0 pages in
    `category`, plus one level of its subcategories -- AI_SUSPECTED_CATEGORY
    is a hidden container whose actual tagged articles live in monthly dated
    subcategories (e.g. "...from May 2025"), not directly in the parent.
    `source_category` records exactly which of those a title was found in,
    so callers can see how a title's tag-recency relates to its results."""

    def fetch_members(cat: str, cmtype: str) -> list[str]:
        titles: list[str] = []
        cmcontinue: str | None = None
        while True:
            params = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": cat,
                "cmtype": cmtype,
                "cmlimit": 500,
                "formatversion": "2",
            }
            if cmcontinue:
                params["cmcontinue"] = cmcontinue
            data = api_get(session, limiter, params)
            titles.extend(p["title"] for p in data.get("query", {}).get("categorymembers", []))
            cmcontinue = data.get("continue", {}).get("cmcontinue")
            if not cmcontinue:
                break
        return titles

    members = {title: category for title in fetch_members(category, "page")}
    for subcat in fetch_members(category, "subcat"):
        for title in fetch_members(subcat, "page"):
            members[title] = subcat
    return members


# ---------------------------------------------------------------------
# Step 5: fetch rendered HTML for a specific revision, strip it down to
# prose, and count words. Returns the same cleaned text that gets counted
# so the word count and the on-disk article text never disagree.
# ---------------------------------------------------------------------

def normalize_prose_whitespace(text: str) -> str:
    """Collapses whitespace and removes the spurious space BeautifulSoup's
    get_text(" ") leaves behind where a stripped inline element (e.g. a
    <sup> citation marker) used to sit directly against punctuation, e.g.
    "Shagamu , Ogun" -> "Shagamu, Ogun"."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return text.strip()


def extract_list_lines(list_el: Tag, depth: int = 1) -> list[tuple[int, str]]:
    """Recursively renders a <ul>/<ol>'s items as (depth, text) pairs,
    mirroring wikitext's '*'/'**' nesting. Each item's "own" text excludes
    any nested sub-list (that's collected separately, one level deeper) so
    a parent <li> and its children don't duplicate each other's text."""
    lines: list[tuple[int, str]] = []
    for li in list_el.find_all("li", recursive=False):
        nested_lists: list[Tag] = []
        own_text_parts: list[str] = []
        for child in li.children:
            if getattr(child, "name", None) in ("ul", "ol"):
                nested_lists.append(child)
                continue
            own_text_parts.append(
                child.get_text(" ", strip=True) if hasattr(child, "get_text") else str(child).strip()
            )
        own_text = normalize_prose_whitespace(" ".join(p for p in own_text_parts if p))
        if own_text:
            lines.append((depth, own_text))
        for nested in nested_lists:
            lines.extend(extract_list_lines(nested, depth + 1))
    return lines


HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

# Standard appendix sections that aren't part of an article's prose (backlink
# lists, citations, redlinked navbox templates, etc.) -- excluded entirely,
# heading included, along with everything nested under them until the next
# heading of equal or higher level.
EXCLUDED_SECTION_HEADINGS = {"references", "external links", "see also", "further reading", "notes"}


def extract_cleaned_prose(html: str) -> tuple[int, str]:
    """Returns (word_count, cleaned_text) for rendered article HTML.

    word_count covers everything kept in cleaned_text -- headings, list
    items, and paragraphs alike -- since it's used (via MIN_WORDS/MAX_WORDS
    and run_pangram.py's cost estimate) to size what's actually submitted
    to Pangram, not to approximate the Prosesize gadget's prose-only count.
    """
    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one(".mw-parser-output") or soup

    for el in content.select(",".join(STRIP_SELECTORS)):
        el.decompose()

    words = 0
    parts: list[str] = []
    skip_level: int | None = None  # heading level of an excluded section currently in progress
    # Walk headings, paragraphs, and top-level lists in document order so
    # sections stay attached to the text they introduce. Anything that
    # survived the stripping above but isn't one of these (e.g. stray <div>
    # text) is intentionally left out.
    for el in content.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol"]):
        if el.name in HEADING_TAGS:
            level = int(el.name[1])
            if skip_level is not None:
                if level <= skip_level:
                    skip_level = None  # left the excluded section
                else:
                    continue
            heading_text = normalize_prose_whitespace(el.get_text(" ", strip=True))
            if heading_text.lower() in EXCLUDED_SECTION_HEADINGS:
                skip_level = level
                continue
            if heading_text:
                words += len(re.findall(r"\b[\w'-]+\b", heading_text, flags=re.UNICODE))
                parts.append(heading_text)
            continue

        if skip_level is not None:
            continue

        if el.name in ("ul", "ol"):
            if el.find_parent("li") is not None:
                continue  # nested list; already handled by its ancestor
            list_lines = extract_list_lines(el)
            words += sum(len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE)) for _, text in list_lines)
            lines = [f"{'*' * depth} {text}" for depth, text in list_lines]
            if lines:
                parts.append("\n".join(lines))
            continue

        text = normalize_prose_whitespace(el.get_text(" ", strip=True))
        if not text:
            continue
        words += len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))
        parts.append(text)
    return words, "\n\n".join(parts)


def fetch_cleaned_prose(session: requests.Session, limiter: RateLimiter, revid: int) -> tuple[int, str]:
    data = api_get(session, limiter, {
        "action": "parse",
        "oldid": revid,
        "prop": "text",
        "formatversion": "2",
        "disabletoc": 1,
        "disableeditsection": 1,
    })
    html = data.get("parse", {}).get("text", "")
    if not html:
        return 0, ""
    return extract_cleaned_prose(html)


def sanitize_title(title: str) -> str:
    """Strips characters that aren't valid in a filename."""
    return re.sub(r'[\\/:*?"<>|]', "_", title).strip(" .")


def safe_filename(title: str) -> str:
    return sanitize_title(title) + ".txt"


def make_row(title: str, revid: int | str) -> dict[str, Any]:
    return {
        "title": title,
        "pageid_url_slug": title.replace(" ", "_"),
        "revision_id": revid,
        "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}?oldid={revid}",
        "date_fetched": time.strftime(TIMESTAMP_FORMAT),
    }


def load_existing_by_title(out_path: Path) -> dict[str, dict[str, Any]]:
    """Returns {title: row} for whatever's already in `out_path`, or {} if
    it doesn't exist yet -- the accumulated record collect_sample/
    run_from_csv merge new rows into, so a fresh run doesn't erase rows for
    articles an earlier run already collected into the same bare folder."""
    if not out_path.exists() or out_path.stat().st_size == 0:
        return {}
    with open(out_path, newline="", encoding="utf-8") as f:
        return {row["title"]: row for row in csv.DictReader(f)}


def save_csv(out_path: Path, fieldnames: list[str], rows_by_title: dict[str, dict[str, Any]]) -> None:
    """Rewrites `out_path` from scratch with every row in `rows_by_title`,
    sorted by title. Called after each new/updated row so a crash never
    loses more than the article in progress -- at the cost of re-writing
    the whole accumulated CSV each time, not just the new row."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for title in sorted(rows_by_title):
            writer.writerow(rows_by_title[title])


def emit_row(run_dir: Path, row: dict[str, Any], text: str, progress: str, already_existed: bool) -> None:
    """Saves `text` to run_dir/<title>.txt and prints a "[progress] title
    (N words)" line. `already_existed` -- whether this title was already in
    the accumulated CSV, decided once by the caller."""
    path = run_dir / safe_filename(row["title"])
    if already_existed:
        print(f"  [warn] {row['title']!r} already exists at {path} -- overwriting")
    path.write_text(text, encoding="utf-8")
    print(f"  [{progress}] {row['title']} ({row['prose_word_count']} words)")


# ---------------------------------------------------------------------
# Reuse mode: re-fetch/re-clean text for an already-selected list of
# (title, revision_id) pairs, e.g. from a prior run's output CSV. Skips the
# random-sampling and filtering steps (1-4) entirely.
# ---------------------------------------------------------------------

def run_from_csv(
    session: requests.Session,
    limiter: RateLimiter,
    args: argparse.Namespace,
    run_dir: Path,
    fieldnames: list[str],
) -> list[dict[str, Any]]:
    with open(args.from_csv, newline="", encoding="utf-8") as fin:
        rows_in = list(csv.DictReader(fin))

    out_path = Path(args.out)
    all_rows = load_existing_by_title(out_path)

    results: list[dict[str, Any]] = []
    for row in rows_in:
        title = row["title"]
        revid = row["revision_id"]
        try:
            word_count, text = fetch_cleaned_prose(session, limiter, int(revid))
        except Exception as e:
            print(f"  [warn] {title}: {e}", file=sys.stderr)
            continue

        out_row = make_row(title, revid)
        out_row["prose_word_count"] = word_count
        out_row["source"] = row.get("source") or RANDOM_SOURCE_LABEL
        results.append(out_row)
        emit_row(run_dir, out_row, text, f"{len(results)}/{len(rows_in)}", title in all_rows)
        all_rows[title] = out_row
        save_csv(out_path, fieldnames, all_rows)

    return results


# ---------------------------------------------------------------------
# Shared collection pipeline: both the dated-snapshot sample (main) and the
# suspected-AI-text sample (collect_ai_suspected) are "pull batches of
# candidate titles from somewhere, filter, look up a revision, drop stubs,
# extract prose, keep the ones long enough" -- they differ only in where
# titles come from, which revision they want, and where output goes.
# ---------------------------------------------------------------------

def collect_sample(
    session: requests.Session,
    limiter: RateLimiter,
    run_dir: Path,
    out_path: str | Path,
    batches: Iterable[list[str]],
    *,
    fieldnames: list[str],
    target_n: int,
    rvstart: str | None,
    label: str,
    source: Callable[[str], str],
) -> list[dict[str, Any]]:
    """Pulls candidate titles from `batches` (an iterable of title-list
    batches) until `target_n` qualifying articles are written to `out_path`
    (CSV) and `run_dir` (one .txt per article), or `batches` is exhausted.

    When `rvstart` is given, each survivor's revision is looked up at that
    snapshot date via get_revision_ids (one request per title -- required
    for a historical revision). When `rvstart` is None, the current
    (revid, size) pairs filter_navigational_and_redirects already returns
    are reused directly, skipping that entire extra request stage.

    `source(title)` becomes each row's "source" column -- a callable so
    both a constant value (e.g. `lambda _: "some label"`) and a per-title
    lookup (e.g. a dict's `.get`) fit the same contract.
    """
    out_path = Path(out_path)
    all_rows = load_existing_by_title(out_path)
    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    for batch in batches:
        if len(results) >= target_n:
            break

        batch = [t for t in batch if t not in seen]
        seen.update(batch)
        if not batch:
            continue

        print(f"--- have {len(results)}/{target_n} {label} "
              f"({len(seen)} candidates examined) ---", file=sys.stderr)

        revid_map = filter_navigational_and_redirects(session, limiter, batch)
        print_discarded(set(batch) - set(revid_map), "navigational/redirect")
        if not revid_map:
            continue

        if rvstart is not None:
            revid_map = get_revision_ids(session, limiter, list(revid_map), rvstart=rvstart)
            print_discarded(set(batch) - set(revid_map), "no revision at snapshot date")
            if not revid_map:
                continue

        # Free pre-filter: skip anything whose raw wikitext is too small
        # to plausibly contain MIN_WORDS of prose, without ever calling
        # the expensive action=parse endpoint for it.
        stubs: dict[str, int] = {}
        kept: dict[str, int] = {}
        for title, (revid, size) in revid_map.items():
            if size < MIN_WIKITEXT_BYTES:
                stubs[title] = size
            else:
                kept[title] = revid
        for title in sorted(stubs):
            print_discarded([title], f"stub ({stubs[title]}B wikitext)")
        revid_map = kept
        if not revid_map:
            continue

        for title, revid in revid_map.items():
            try:
                word_count, text = fetch_cleaned_prose(session, limiter, revid)
            except Exception as e:
                print(f"  [warn] {title}: {e}", file=sys.stderr)
                continue
            if word_count <= MIN_WORDS:
                print_discarded([title], f"too short ({word_count} words)")
                continue
            if word_count > MAX_WORDS:
                print_discarded([title], f"too long ({word_count} words)")
                continue

            row = make_row(title, revid)
            row["prose_word_count"] = word_count
            row["source"] = source(title)
            results.append(row)
            emit_row(run_dir, row, text, f"{len(results)}/{target_n}", title in all_rows)
            all_rows[title] = row
            save_csv(out_path, fieldnames, all_rows)

            if len(results) >= target_n:
                break

    print(f"\nDone. Wrote {len(results)} {label} to {out_path} and {run_dir}/ "
          f"(examined {len(seen)} candidate titles).")
    return results


# ---------------------------------------------------------------------
# Suspected-AI-text sample: same pipeline as the dated snapshot, but
# candidates come from AI_SUSPECTED_CATEGORY instead of list=random, and
# each article's *current* revision is used (rvstart=None) with no
# creation-date cutoff. Written to its own --ai-articles-dir folder and
# CSV, kept entirely separate from the dated sample.
# ---------------------------------------------------------------------

AI_FIELDNAMES = ["title", "pageid_url_slug", "revision_id", "prose_word_count",
                 "url", "source", "date_fetched"]


def collect_ai_suspected(
    session: requests.Session, limiter: RateLimiter, args: argparse.Namespace, ai_articles_dir: Path
) -> None:
    print(f"\n--- Collecting {args.ai_n} suspected-AI-text articles from "
          f"{AI_SUSPECTED_CATEGORY} ---", file=sys.stderr)

    # {title: source_category} -- candidate_map.get is already exactly the
    # per-title "source" lookup collect_sample needs (which monthly subcat a
    # title came from).
    candidate_map = get_category_members(session, limiter, AI_SUSPECTED_CATEGORY)
    print(f"  category pool: {len(candidate_map)} articles", file=sys.stderr)
    candidates = list(candidate_map)
    random.shuffle(candidates)

    run_dir = ai_articles_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    batches = (candidates[i:i + args.batch_size] for i in range(0, len(candidates), args.batch_size))
    collect_sample(session, limiter, run_dir, args.ai_out, batches,
                    fieldnames=AI_FIELDNAMES,
                    target_n=args.ai_n, rvstart=None, label="suspected-AI articles",
                    source=candidate_map.get)


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20, help="number of qualifying articles to collect")
    ap.add_argument("--out", default=DEFAULT_OUT,
                     help="CSV path for the dated sample")
    ap.add_argument("--articles-dir", default=DEFAULT_ARTICLES_DIR,
                     help="directory to write the dated sample's cleaned article "
                          "text into -- articles accumulate here across runs; a "
                          "title collecting again overwrites its previous .txt "
                          "file (and CSV row), with a warning")
    ap.add_argument("--batch-size", type=int, default=100,
                     help="random titles pulled per round before filtering")
    ap.add_argument("--min-interval", type=float, default=0.5,
                     help="minimum seconds between successive API requests "
                          "(raise this if you're seeing 429s)")
    ap.add_argument("--contact", required=True,
                     help="email or URL to include in the User-Agent, per WMF API etiquette")
    ap.add_argument("--max-rounds", type=int, default=200,
                     help="safety cap on number of sampling rounds")
    ap.add_argument("--from-csv",
                     help="reuse the (title, revision_id) pairs from an existing "
                          "output CSV instead of drawing a fresh random sample -- "
                          "skips steps 1-4 and just re-fetches/re-cleans text for "
                          "that exact article list")
    ap.add_argument("--ai-n", type=int, default=0,
                     help="also collect this many random articles from "
                          f"{AI_SUSPECTED_CATEGORY!r} (current revision, no "
                          "creation-date cutoff), written to --ai-articles-dir")
    ap.add_argument("--ai-articles-dir", default=DEFAULT_AI_ARTICLES_DIR,
                     help="directory to write the suspected-AI sample's cleaned "
                          "article text into -- see --ai-n; same accumulate/"
                          "overwrite behavior as --articles-dir")
    ap.add_argument("--ai-out", default=DEFAULT_AI_OUT,
                     help="CSV path for the suspected-AI sample (see --ai-n)")

    args = ap.parse_args()

    session = make_session(args.contact)
    limiter = RateLimiter(min_interval=args.min_interval)

    run_dir = Path(args.articles_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = ["title", "pageid_url_slug", "revision_id",
                  "prose_word_count", "url", "source", "date_fetched"]

    if args.from_csv:
        results = run_from_csv(session, limiter, args, run_dir, fieldnames)
        print(f"\nDone. Wrote {len(results)} articles to {args.out} and {run_dir}/ "
              f"(reused titles from {args.from_csv}).")
        if args.ai_n > 0:
            collect_ai_suspected(session, limiter, args, Path(args.ai_articles_dir))
        return

    batches = (get_random_titles(session, limiter, args.batch_size) for _ in range(args.max_rounds))
    collect_sample(session, limiter, run_dir, args.out, batches,
                    fieldnames=fieldnames,
                    target_n=args.n, rvstart=OCT_SNAPSHOT, label="articles",
                    source=lambda _title: RANDOM_SOURCE_LABEL)

    if args.ai_n > 0:
        collect_ai_suspected(session, limiter, args, Path(args.ai_articles_dir))


if __name__ == "__main__":
    main()