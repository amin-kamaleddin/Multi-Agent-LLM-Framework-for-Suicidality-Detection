#!/usr/bin/env python3
"""
build_negative_labels.py

Convert a Reddit API JSON dump (a post + its comment listing) into a CSV of
NEGATIVE-label examples for suicide-detection research: user messages that
discuss emotions/depression but contain NO mention of suicide, self-harm, or
loss of the will to live.

Output columns: ID, users, text, sentiment, time
  - ID        : numeric, contiguous, starts at START_ID
  - users     : numeric, one number per unique (anonymized) author, starts at START_USER
  - text      : the comment (double-quoted); messages > MAX_CHARS are split at
                sentence boundaries into multiple rows
  - sentiment : always "Unsupportive" (double-quoted)
  - time      : the comment's created_utc as an int (0 if unavailable)

Usage:
    python3 build_negative_labels.py input.json output.csv

IMPORTANT — read before reusing:
  * The exclusion filter is deliberately CONSERVATIVE. For negative-label
    creation, wrongly *keeping* a suicidal message is far more damaging than
    wrongly *dropping* a safe one, so this errs toward dropping. Expect some
    innocent messages to be excluded.
  * Regex alone will miss implicit ideation (metaphor, euphemism). This script
    supports a MANUAL_EXCLUDE set of comment IDs for cases a human reviewer
    flags. The IDs currently listed were found by manual review of one specific
    thread; clear or replace them for a different dataset.
  * ALWAYS eyeball the excluded/kept split and spot-check the output before
    trusting it. See the audit block near the bottom.
"""

import argparse
import csv
import html
import json
import re
import sys

# ----------------------------- configuration --------------------------------

START_ID = 1591          # first value in the ID column
START_USER = 2771        # first value in the users column
MAX_CHARS = 600          # messages longer than this are split into multiple rows
SENTIMENT = "Unsupportive"

# Comment IDs a human reviewer judged to contain implicit ideation the regex
# missed. Specific to the original thread — empty this out for new data.
MANUAL_EXCLUDE = {
    "l8adoru", "l89jomg", "l89nivy", "l89w8b1",
    "l8a37sq", "l88ztb8", "l89171x", "l8927cu",
}

# Any comment matching one of these (case-insensitive) is dropped. Covers
# explicit suicide, self-harm, and passive ideation / "loss of will to live".
EXCLUSION_PATTERNS = [
    r'suicid', r'unaliv', r'\bkms\b', r'sewer\s*slide',
    r'self.?harm', r'self.?injur', r'self.?mutilat',
    r'kill(ing)?\s+(my|your|him|her|them|one)\s*sel(f|ves)',
    r'\b(off|offing|top|topping|do|doing)\s+(my|your|him|her|them|one)\s*sel(f|ves)\b',
    r'(take|taking|took|end|ending|ended)\s+(my|your|his|her|their)\s+(own\s+)?life',
    r'end(ing)?\s+(it all|it|things)\b', r'end\s+my\s+life',
    r'(cut|cutting|hurt|hurting|harm|harming|burn|burning|starv\w*)\s+(my|your|him|her|them|one)\s*sel(f|ves)',
    r'want(ed|s)?\s+to\s+die', r'wan(na|t\s+to)\s+die',
    r'wish(ed|es)?\s+(i|he|she|they|to)\s+(was|were|had|would)?\s*(be\s+)?(dead|die|never born)',
    r'better\s+off\s+dead',
    r'rather\s+(be\s+dead|not\s+(be\s+here|exist|wake|live))',
    r'(no|lost|losing|without)\s+(the\s+)?will\s+to\s+live', r'lost\s+the\s+will',
    r'no\s+(point|reason|desire)\s+(in\s+|to\s+)?(living|live|life|be alive|being alive|go on|exist)',
    r'don.?t\s+(want|wanna|wish)\s+to\s+(live|be\s+here|be\s+alive|exist|wake\s+up|go\s+on|continue)',
    r'do\s+not\s+want\s+to\s+(live|be\s+here|exist)',
    r'didn.?t\s+want\s+to\s+(live|wake|be\s+here|exist)',
    r'why\s+(am\s+i|are\s+you|are\s+we|is\s+he|is\s+she)\s+(still\s+)?(alive|here|not\s+dead)',
    r'still\s+not\s+dead', r'why\s+.{0,15}\s+not\s+dead',
    r'can.?t\s+(go\s+on|do\s+this\s+anymore|keep\s+going|take\s+it\s+anymore)',
    r'(tired|done|sick)\s+of\s+(living|being\s+alive|life|existing)',
    r'(hope|wish)\s+(i|he|she|they)\s+(don.?t|do\s+not|never|wouldn.?t)\s+wake',
    r'not\s+wak(e|ing)\s+up',
    r'(disappear|vanish|cease)\s+(forever|to\s+exist|permanently)', r'cease\s+to\s+exist',
    r'feel(ing|s)?\s+like\s+(i.?m\s+)?dying', r'want\s+to\s+be\s+dead',
    r'ready\s+to\s+die', r'welcome\s+death',
    r'\boverdos', r'jump(ed|ing)?\s+off\s+(a\s+)?(bridge|building|roof)',
    r'hang(ed|ing)?\s+(my|him|her)\s*self',
    r'not\s+want\s+to\s+live', r'give\s+up\s+on\s+life', r'end\s+it\s+all',
]
EXCLUDE_RE = re.compile('|'.join(EXCLUSION_PATTERNS), re.IGNORECASE)

# Broad "please look at this by eye" net. These words are NOT auto-excluded;
# they just surface comments a human should review for implicit ideation.
AUDIT_RE = re.compile(
    r'\b(die|dying|died|dead|death|kill|harm|hurt|alive|survive|gun|pills|razor|'
    r'blade|bleed|hang|drown|choke|worthless|hopeless|give up|swallow|morbid|end)\b',
    re.IGNORECASE,
)

# ------------------------------- helpers ------------------------------------

def is_bot_or_empty(comment):
    body = comment.get("body", "").strip()
    if not body or body in ("[deleted]", "[removed]"):
        return True
    if comment.get("author") in ("AutoModerator", "[deleted]"):
        return True
    return False


def clean(text):
    """Decode HTML entities and collapse all whitespace to single spaces."""
    text = html.unescape(text)
    return re.sub(r'\s+', ' ', text).strip()


def split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def chunk_by_max(text, max_chars=MAX_CHARS):
    """Greedily pack sentences into chunks <= max_chars. A single sentence
    longer than max_chars becomes its own (oversized) chunk."""
    if len(text) <= max_chars:
        return [text]
    chunks, cur = [], ""
    for sent in split_sentences(text):
        if not cur:
            cur = sent
        elif len(cur) + 1 + len(sent) <= max_chars:
            cur = cur + " " + sent
        else:
            chunks.append(cur)
            cur = sent
        if len(cur) > max_chars and cur == sent:   # unsplittable long sentence
            chunks.append(cur)
            cur = ""
    if cur:
        chunks.append(cur)
    return chunks


def extract_comments(data):
    """Pull the t1 comment objects out of a Reddit post+comments JSON dump."""
    comments = []
    for listing in data:
        children = listing.get("data", {}).get("children", [])
        for child in children:
            if child.get("kind") == "t1":
                comments.append(child["data"])
    return comments

# -------------------------------- main --------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build a suicide-negative-label CSV from a Reddit JSON dump.")
    ap.add_argument("input_json")
    ap.add_argument("output_csv")
    ap.add_argument("--start-id", type=int, default=START_ID)
    ap.add_argument("--start-user", type=int, default=START_USER)
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS)
    args = ap.parse_args()

    with open(args.input_json, encoding="utf-8") as f:
        data = json.load(f)

    comments = extract_comments(data)

    # ---- filter -------------------------------------------------------------
    kept, excluded, dropped_bot = [], [], 0
    for c in comments:
        if is_bot_or_empty(c):
            dropped_bot += 1
            continue
        if EXCLUDE_RE.search(c.get("body", "")) or c.get("id") in MANUAL_EXCLUDE:
            excluded.append(c)
            continue
        kept.append(c)

    # ---- build rows ---------------------------------------------------------
    user_map, next_user = {}, args.start_user
    rows, next_id = [], args.start_id
    for c in kept:
        author = c.get("author")
        if author not in user_map:
            user_map[author] = next_user
            next_user += 1
        uid = user_map[author]
        t = int(c.get("created_utc") or 0)
        for chunk in chunk_by_max(clean(c.get("body", "")), args.max_chars):
            rows.append((next_id, uid, chunk, SENTIMENT, t))
            next_id += 1

    # ---- write CSV ----------------------------------------------------------
    # Unquoted numeric header; QUOTE_NONNUMERIC quotes text + sentiment (str)
    # and leaves ID/users/time (int) bare.
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        f.write("ID,users,text,sentiment,time\n")
        w = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC, lineterminator="\n")
        for r in rows:
            w.writerow(r)

    # ---- report -------------------------------------------------------------
    print(f"total comments        : {len(comments)}")
    print(f"dropped (bot/empty)   : {dropped_bot}")
    print(f"excluded (suicide/SH) : {len(excluded)}")
    print(f"kept comments         : {len(kept)}")
    print(f"rows written          : {len(rows)}")
    print(f"unique users          : {len(user_map)}")
    if rows:
        print(f"ID range              : {rows[0][0]} -> {rows[-1][0]}")
        print(f"user range            : {min(user_map.values())} -> {max(user_map.values())}")

    # Residual audit: kept comments still containing sensitive tokens. These
    # are NOT excluded — review them by eye and add IDs to MANUAL_EXCLUDE if
    # any contain implicit ideation.
    flagged = [c for c in kept if AUDIT_RE.search(c.get("body", ""))]
    if flagged:
        print(f"\n[audit] {len(flagged)} kept comment(s) contain sensitive tokens — review by hand:")
        for c in flagged:
            print(f"  id={c.get('id')}: {clean(c.get('body',''))[:100]}")


if __name__ == "__main__":
    sys.exit(main())
