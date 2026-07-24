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
    python get_test_wiki_pages.py --2022-n 25 --out sample.csv

Optionally, with --ai-n and/or --ai-drafts-n, also collects separate random
samples from two Wikipedia maintenance categories of likely-AI-generated
text: Category:Articles containing suspected AI-generated texts (--ai-n,
mainspace articles human editors have flagged) and Category:AfC submissions
declined as a large language model output (--ai-drafts-n, Draft-namespace
submissions a reviewer explicitly declined as LLM output). Both are
processed the same way as each other, except there's no creation-date
cutoff, and rather than the *current* revision, each searches its last 50
revisions for the one actually carrying the evidence a human flagged it:
find_decline_revision looks for the AfC reviewing tool's LLM-decline edit
summary, find_ai_tagged_revision looks for an {{AI-generated}}/{{Prod
llm}} tag in the wikitext itself. Falls back to the current revision (with
a warning) if neither turns up within that window. Each is written to its
own folder/CSV pair
(--ai-articles-dir/--ai-out and --ai-drafts-articles-dir/--ai-drafts-out)
so neither is ever mixed with the dated sample or with each other:

    python get_test_wiki_pages.py --2022-n 25 --ai-n 25 --ai-drafts-n 25 --out sample.csv --ai-out ai_sample.csv --ai-drafts-out ai_drafts_sample.csv

-n is shorthand for setting all three (--2022-n, --ai-n, --ai-drafts-n) to
the same count at once, e.g. `-n 25` collects 25 from each. It only takes
effect if none of those three are given individually -- passing any of
them is what makes each opt-in, and -n doesn't override that.

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
DEFAULT_AI_DRAFTS_ARTICLES_DIR = f"{DATA_DIR}/random_AI_suspected_drafts"
DEFAULT_AI_DRAFTS_OUT = f"{DATA_DIR}/wikipedia_ai_drafts_sample.csv"

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

# Random titles pulled per round before filtering
BATCH_SIZE = 100

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

# Hidden maintenance container category for suspected-AI-generated text.
AI_SUSPECTED_CATEGORY = "Category:Articles containing suspected AI-generated texts"
# Draft-namespace submissions an AfC reviewer explicitly declined as LLM output.
AI_SUSPECTED_DRAFTS_CATEGORY = "Category:AfC submissions declined as a large language model output"

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
    '[class*="afc-submission"]',  # AfC review banners/comment boxes on Draft: pages
    # Note: headings (h1-h6) are deliberately NOT stripped
]

# Older-style AfC review comments aren't wrapped in an afc-submission-* box --
# they're a plain "* '''Comment:''' ... ~~~~" bullet, indistinguishable by
# markup from a real list item. But real article prose never ends in a
# MediaWiki auto-signature timestamp, so matching that is a reliable,
# content-based way to catch them instead.
SIGNATURE_TIMESTAMP_RE = re.compile(r"\d{1,2}:\d{2},\s+\d{1,2}\s+\w+\s+\d{4}\s+\(UTC\)\s*$")


def make_session(contact: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": f"AiTestArticleSampler/1.0 ({contact})",
    })
    return s


class RateLimiter:
    """Minimum-interval throttle between successive API requests."""
    def __init__(self, min_interval: float = 0.5) -> None:
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
        chunk = [t for t in chunk if not LIST_TITLE_RE.search(strip_draft_prefix(t))]
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
# Revision finders for the two AI-suspected categories: rather than trust
# the *current* revision, dig back through recent history for the specific
# revision that actually carries the evidence a human flagged this page --
# the tag/decline can predate later edits, or (for AI_SUSPECTED_CATEGORY)
# be removed entirely while category membership lags behind the edit that
# removed it. Both search at most 50 revisions back (newest first) and
# return None -- the caller falls back to the current revision -- if
# nothing turns up in that window.
# ---------------------------------------------------------------------

# The edit summary Wikipedia's AfC reviewing tool (AFCH) leaves when
# declining a submission specifically as LLM-generated content.
AFC_LLM_DECLINE_COMMENT = "Submission appears to be a large language model output"

def find_decline_revision(session: requests.Session, limiter: RateLimiter, title: str) -> tuple[int, int] | None:
    """Returns (revid, size) of the most recent revision whose edit summary
    matches AFC_LLM_DECLINE_COMMENT, searching up to the last 50 revisions
    -- so a declined draft is captured as of that decline, not whatever
    it's since been edited to (further reviewer comments, resubmission,
    etc). None if no match turns up."""
    data = api_get(session, limiter, {
        "action": "query",
        "prop": "revisions",
        "titles": title,
        "rvlimit": 50,
        "rvprop": "ids|comment|size",
        "rvdir": "older",
        "formatversion": "2",
    })
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return None
    for rev in pages[0].get("revisions", []):
        if AFC_LLM_DECLINE_COMMENT.lower() in (rev.get("comment") or "").lower():
            return rev["revid"], rev["size"]
    return None


# Matches {{AI-generated|date=...}} and {{Prod llm/dated|...}} (and the
# hyphen/underscore/space variants MediaWiki treats as equivalent in
# template names) -- the maintenance tags that put an article in
# AI_SUSPECTED_CATEGORY in the first place.
AI_TEMPLATE_RE = re.compile(r"\{\{\s*(ai[- _]generated|prod[- _]?llm)\b", re.IGNORECASE)


def find_ai_tagged_revision(session: requests.Session, limiter: RateLimiter, title: str) -> tuple[int, int] | None:
    """Returns (revid, wikitext size) of the most recent revision whose
    wikitext contains an AI_TEMPLATE_RE match, searching up to the last 50
    revisions. None if no match turns up."""
    data = api_get(session, limiter, {
        "action": "query",
        "prop": "revisions",
        "titles": title,
        "rvlimit": 50,
        "rvprop": "ids|content",
        "rvslots": "main",
        "rvdir": "older",
        "formatversion": "2",
    })
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return None
    for rev in pages[0].get("revisions", []):
        content = rev.get("slots", {}).get("main", {}).get("content", "")
        if AI_TEMPLATE_RE.search(content):
            return rev["revid"], len(content.encode("utf-8"))
    return None


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

    Headings are written back out in Markdown form ("##"/"###"/... per
    h2/h3/... level) rather than as bare text, since MediaWiki articles
    start their sections at h2 -- so a top-level "== Section ==" becomes
    "## Section", matching the heading style most models already use for
    generated articles (see generate_ai_articles.py).
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
                parts.append(f"{'#' * level} {heading_text}")
            continue

        if skip_level is not None:
            continue

        if el.name in ("ul", "ol"):
            if el.find_parent("li") is not None:
                continue  # nested list; already handled by its ancestor
            list_lines = [
                (depth, text) for depth, text in extract_list_lines(el)
                if not SIGNATURE_TIMESTAMP_RE.search(text)
            ]
            words += sum(len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE)) for _, text in list_lines)
            lines = [f"{'*' * depth} {text}" for depth, text in list_lines]
            if lines:
                parts.append("\n".join(lines))
            continue

        text = normalize_prose_whitespace(el.get_text(" ", strip=True))
        if not text or SIGNATURE_TIMESTAMP_RE.search(text):
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


def strip_draft_prefix(title: str) -> str:
    """Strips a leading "Draft:" namespace prefix -- that's the namespace,
    not part of the article's actual title."""
    return title.removeprefix("Draft:")


def make_row(title: str, revid: int | str, wiki_title: str | None = None) -> dict[str, Any]:
    """`wiki_title` is the real page title, namespace prefix included (e.g.
    "Draft:X"), used for pageid_url_slug/url so the link still resolves;
    defaults to `title` when there's no prefix to strip in the first place."""
    wiki_title = wiki_title if wiki_title is not None else title
    return {
        "title": title,
        "pageid_url_slug": wiki_title.replace(" ", "_"),
        "revision_id": revid,
        "url": f"https://en.wikipedia.org/wiki/{wiki_title.replace(' ', '_')}?oldid={revid}",
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
        # pageid_url_slug retains any namespace prefix (e.g. "Draft:") that
        # was stripped from "title" -- recover it so the URL still resolves.
        wiki_title = row.get("pageid_url_slug", title).replace("_", " ")
        try:
            word_count, text = fetch_cleaned_prose(session, limiter, int(revid))
        except Exception as e:
            print(f"  [warn] {title}: {e}", file=sys.stderr)
            continue

        out_row = make_row(title, revid, wiki_title=wiki_title)
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
    revision_finder: Callable[[str], tuple[int, int] | None] | None = None,
) -> list[dict[str, Any]]:
    """Pulls candidate titles from `batches` until `target_n` articles are written to `out_path`
    (CSV) and `run_dir` (one .txt per article), or `batches` is exhausted.

    When `rvstart` is given, each survivor's revision is looked up at that
    snapshot date via get_revision_ids (one request per title -- required
    for a historical revision). When `rvstart` is None, the current
    (revid, size) pairs filter_navigational_and_redirects already returns
    are reused directly -- unless `revision_finder` is also given, in which
    case it overrides that current revision with whatever
    revision_finder(title) finds (e.g. find_decline_revision,
    find_ai_tagged_revision), falling back to the current one with a
    warning if it returns None.

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

        if revision_finder is not None:
            found: dict[str, tuple[int, int]] = {}
            for title in revid_map:
                result = revision_finder(title)
                if result is None:
                    print(f"  [warn] {title}: no matching revision found in "
                          f"last 50; using current revision", file=sys.stderr)
                    found[title] = revid_map[title]
                else:
                    found[title] = result
            revid_map = found

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

        for wiki_title, revid in revid_map.items():
            try:
                word_count, text = fetch_cleaned_prose(session, limiter, revid)
            except Exception as e:
                print(f"  [warn] {wiki_title}: {e}", file=sys.stderr)
                continue
            if word_count <= MIN_WORDS:
                print_discarded([wiki_title], f"too short ({word_count} words)")
                continue
            if word_count > MAX_WORDS:
                print_discarded([wiki_title], f"too long ({word_count} words)")
                continue

            title = strip_draft_prefix(wiki_title)
            row = make_row(title, revid, wiki_title=wiki_title)
            row["prose_word_count"] = word_count
            row["source"] = source(wiki_title)
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
# Category-sourced samples: same pipeline as the dated snapshot, but
# candidates come from a maintenance category instead of list=random, and
# each article's *current* revision is used (rvstart=None) with no
# creation-date cutoff. Written to their own folder and CSV, kept entirely
# separate from the dated sample. Used for both AI_SUSPECTED_CATEGORY
# (mainspace articles tagged as suspected AI text) and
# AI_SUSPECTED_DRAFTS_CATEGORY (Draft-namespace AfC submissions a reviewer
# declined as LLM output) -- get_category_members and collect_sample are
# namespace-agnostic, so the same pipeline covers both.
# ---------------------------------------------------------------------

AI_FIELDNAMES = ["title", "pageid_url_slug", "revision_id", "prose_word_count",
                 "url", "source", "date_fetched"]


def collect_from_category(
    session: requests.Session,
    limiter: RateLimiter,
    category: str,
    target_n: int,
    out_csv: str,
    articles_dir: Path,
    label: str,
    require_prefix: str | None = None,
    revision_finder: Callable[[str], tuple[int, int] | None] | None = None,
) -> None:
    print(f"\n--- Collecting {target_n} {label} from {category} ---", file=sys.stderr)

    # {title: source_category} -- candidate_map.get is already exactly the
    # per-title "source" lookup collect_sample needs (which monthly subcat a
    # title came from).
    candidate_map = get_category_members(session, limiter, category)
    print(f"  category pool: {len(candidate_map)} articles", file=sys.stderr)
    if require_prefix is not None:
        # e.g. AI_SUSPECTED_DRAFTS_CATEGORY also tags stray User:.../sandbox
        # pages alongside real Draft: submissions -- those don't have a
        # corresponding "article title" to extract, so drop them.
        off_prefix = {t for t in candidate_map if not t.startswith(require_prefix)}
        print_discarded(off_prefix, f"not in {require_prefix!r} namespace")
        for title in off_prefix:
            del candidate_map[title]
    candidates = list(candidate_map)
    random.shuffle(candidates)

    articles_dir.mkdir(parents=True, exist_ok=True)

    batches = (candidates[i:i + BATCH_SIZE] for i in range(0, len(candidates), BATCH_SIZE))
    collect_sample(session, limiter, articles_dir, out_csv, batches,
                    fieldnames=AI_FIELDNAMES,
                    target_n=target_n, rvstart=None, label=label,
                    source=candidate_map.get, revision_finder=revision_finder)


# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contact", required=True,
                     help="email or URL to include in the User-Agent, per WMF API etiquette")
    ap.add_argument("--from-csv",
                     help="reuse the articles from an existing "
                          "output CSV instead of drawing a fresh random sample")
    ap.add_argument("--2022-n", type=int, default=0, dest="n2022",
                     help="number of dated (random_2022) articles to collect -- "
                          "each of --2022-n/--ai-n/--ai-drafts-n is independently "
                          "opt-in, so only the ones actually given run")
    ap.add_argument("-n", type=int, default=0, dest="n_all",
                     help="shorthand for collecting this many articles from all three "
                          "categories at once (--2022-n, --ai-n, and --ai-drafts-n) -- "
                          "only takes effect if none of those three are given")
    ap.add_argument("--out", default=DEFAULT_OUT,
                     help="CSV path for the random 2022 sample")
    ap.add_argument("--articles-dir", default=DEFAULT_ARTICLES_DIR,
                     help="directory to write the random 2022 sample's cleaned articles to ")
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
    ap.add_argument("--ai-drafts-n", type=int, default=0,
                     help="also collect this many random articles from "
                          f"{AI_SUSPECTED_DRAFTS_CATEGORY!r} (current revision, "
                          "no creation-date cutoff), written to --ai-drafts-articles-dir")
    ap.add_argument("--ai-drafts-articles-dir", default=DEFAULT_AI_DRAFTS_ARTICLES_DIR,
                     help="directory to write the AfC-declined-as-LLM sample's cleaned "
                          "article text into -- see --ai-drafts-n; same accumulate/"
                          "overwrite behavior as --articles-dir")
    ap.add_argument("--ai-drafts-out", default=DEFAULT_AI_DRAFTS_OUT,
                     help="CSV path for the AfC-declined-as-LLM sample (see --ai-drafts-n)")

    args = ap.parse_args()
    if args.n_all > 0 and not (args.n2022 > 0 or args.ai_n > 0 or args.ai_drafts_n > 0):
        args.n2022 = args.ai_n = args.ai_drafts_n = args.n_all

    session = make_session(args.contact)
    limiter = RateLimiter()

    run_dir = Path(args.articles_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = ["title", "pageid_url_slug", "revision_id",
                  "prose_word_count", "url", "source", "date_fetched"]

    if args.from_csv:
        results = run_from_csv(session, limiter, args, run_dir, fieldnames)
        print(f"\nDone. Wrote {len(results)} articles to {args.out} and {run_dir}/ "
              f"(reused titles from {args.from_csv}).")
        if args.ai_n > 0:
            collect_from_category(session, limiter, AI_SUSPECTED_CATEGORY, args.ai_n,
                                   args.ai_out, Path(args.ai_articles_dir), "suspected-AI articles",
                                   revision_finder=lambda t: find_ai_tagged_revision(session, limiter, t))
        if args.ai_drafts_n > 0:
            collect_from_category(session, limiter, AI_SUSPECTED_DRAFTS_CATEGORY, args.ai_drafts_n,
                                   args.ai_drafts_out, Path(args.ai_drafts_articles_dir),
                                   "AfC-declined-as-LLM drafts", require_prefix="Draft:",
                                   revision_finder=lambda t: find_decline_revision(session, limiter, t))
        return

    if not (args.n2022 > 0 or args.ai_n > 0 or args.ai_drafts_n > 0):
        raise SystemExit("Nothing to do -- pass -n, --2022-n, --ai-n, and/or "
                          "--ai-drafts-n (or --from-csv to reuse an existing sample).")

    if args.n2022 > 0:
        batches = (get_random_titles(session, limiter, BATCH_SIZE) for _ in range(200)) # limit to 200 rounds for safety; lift this if deliberately doing very large run
        collect_sample(session, limiter, run_dir, args.out, batches,
                        fieldnames=fieldnames,
                        target_n=args.n2022, rvstart=OCT_SNAPSHOT, label="articles",
                        source=lambda _title: RANDOM_SOURCE_LABEL)

    if args.ai_n > 0:
        collect_from_category(session, limiter, AI_SUSPECTED_CATEGORY, args.ai_n,
                               args.ai_out, Path(args.ai_articles_dir), "suspected-AI articles",
                               revision_finder=lambda t: find_ai_tagged_revision(session, limiter, t))
    if args.ai_drafts_n > 0:
        collect_from_category(session, limiter, AI_SUSPECTED_DRAFTS_CATEGORY, args.ai_drafts_n,
                               args.ai_drafts_out, Path(args.ai_drafts_articles_dir),
                               "AfC-declined-as-LLM drafts", require_prefix="Draft:",
                               revision_finder=lambda t: find_decline_revision(session, limiter, t))


if __name__ == "__main__":
    main()