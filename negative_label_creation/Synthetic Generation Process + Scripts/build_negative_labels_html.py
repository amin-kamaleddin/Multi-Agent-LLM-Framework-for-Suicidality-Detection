#!/usr/bin/env python3
"""
build_negative_labels_html.py

Convert a saved old.reddit HTML page (a post + its rendered comment tree) into
a CSV of NEGATIVE-label examples for suicide-detection research: user messages
that discuss emotions/depression but contain NO mention of suicide, self-harm,
or loss of the will to live.

This is the HTML-input sibling of build_negative_labels.py. ONLY the input
stage changed (JSON dump -> parsed HTML). The exclusion filter, sentence
chunking, anonymization, CSV format, report, and audit block are identical.

Output columns: ID, users, text, sentiment, time
  - ID        : numeric, contiguous, starts at START_ID
  - users     : numeric, one number per unique (anonymized) author, starts at START_USER
  - text      : the comment (double-quoted); messages > MAX_CHARS are split at
                sentence boundaries into multiple rows
  - sentiment : always "Unsupportive" (double-quoted)
  - time      : the comment's created_utc as an int (0 if unavailable)

Usage:
    python3 build_negative_labels_html.py input.html output.csv

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
import re
import sys
from datetime import datetime

from bs4 import BeautifulSoup

# ----------------------------- configuration --------------------------------

START_ID = 1591          # first value in the ID column
START_USER = 2771        # first value in the users column
MAX_CHARS = 600          # messages longer than this are split into multiple rows
SENTIMENT = "Unsupportive"

# Comment IDs a human reviewer judged to contain implicit ideation / self-harm
# the regex missed. Specific to a single thread — repopulate from the audit
# block below for each new dataset. Populated from manual review of the
# r/AskReddit "What does YOUR depression feel like?" HTML capture: each of
# these expresses a death wish, passive ideation, a suicide method, "better off
# gone/dead", loss of reason to live, or bodily self-harm (disordered eating),
# phrased so the regex above did not catch it.
MANUAL_EXCLUDE = {
    # explicit/active or passive death wish phrased around the pattern
    "e43rdjm",  # "...just want to fucking die god please take me" (word breaks "want to die")
    "e43mvls",  # "wouldn't mind to die ... wouldn't mind if it ... be my last day"
    "e43mnni",  # "Don't care if I live or die."
    "e43w5bi",  # "Basically waiting to die."
    "e43i3sk",  # "...you decide that dying would be the best option."
    "e43b04m",  # "...a mind that's trying to die."
    "e47ufg7",  # "...not sure if I even want to make it through senior alive."
    "e43xvob",  # "'I wish this would end already' ... if I mean my life."
    # "better off gone / more valuable dead" and method contemplation
    "e44c6be",  # "if i just disappeared ... others lives would be better"
    "e43wb81",  # "turn the wrong valve ... more valuable dead than alive"
    "e43v0nq",  # "cannot see any of them genuinely caring for my death"
    "e43k89y",  # intrusive thought naming a suicide method ("hang ourselves...")
    # "no reason / nothing to live for" / resignation to being killed by it
    "e43pdj9",  # "nothing worthwhile to live for. Just pain..."
    "e43hcet",  # "I feel like it'll kill me eventually ... I just want it to stop"
    "e43tdu8",  # "rational mind tells you that it's not killing you quick enough"
    "e43nte4",  # "'life addiction' ... can't stop unless you take drastic action"
    # implied suicide of another, presented as such
    "e43olow",  # "She is dead now. Get help sooner rather than later."
    # bodily self-harm / disordered eating
    "e43trq2",  # "Anorexia, cigs, and pills."
    "e43kjot",  # "eating less than 500 calories a day..."
    "e43otfj",  # self-induced vomiting ("puking my guts out ... better at not doing that")
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
    # In saved old.reddit HTML a deleted/removed account has no data-author
    # attribute (author is None) rather than the literal "[deleted]" the API
    # returns. Both mean "no identifiable author": such comments can't be
    # attributed to a distinct anonymized user, so drop them exactly as the
    # JSON pipeline drops "[deleted]" authors.
    if comment.get("author") in ("AutoModerator", "[deleted]", None, ""):
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


def _iso_to_epoch(iso):
    """Convert an ISO-8601 timestamp (e.g. '2018-08-13T03:02:54+00:00') to an
    int unix epoch. Returns 0 if missing or unparseable."""
    if not iso:
        return 0
    try:
        return int(datetime.fromisoformat(iso).timestamp())
    except ValueError:
        return 0


def extract_comments(soup):
    """Pull comment objects out of a saved old.reddit HTML page.

    Returns a list of dicts with the SAME keys the JSON version produced
    (body, author, id, created_utc) so the downstream filter/build/report/audit
    code is unchanged.

    Each Reddit comment is a `div.thing[data-type="comment"]`. We read only that
    comment's OWN entry (recursive=False on div.entry) so nested reply text is
    not swept into the parent. The visible comment text lives in the <p> tags of
    `div.usertext-body > div.md`; multiple paragraphs are joined into one body.
    Author is `data-author`; the comment id is `data-fullname` minus its "t1_"
    prefix; the time comes from the tagline's <time datetime=...>.
    """
    comments = []
    for thing in soup.select('div.thing[data-type="comment"]'):
        entry = thing.find("div", class_="entry", recursive=False)
        if entry is None:
            continue

        body_div = entry.find("div", class_="usertext-body")
        md = body_div.find("div", class_="md") if body_div else None
        paras = md.find_all("p", recursive=False) if md else []
        body = "\n".join(p.get_text() for p in paras).strip()

        tagline = entry.find("p", class_="tagline")
        time_tag = tagline.find("time") if tagline else None
        created = _iso_to_epoch(time_tag.get("datetime") if time_tag else None)

        fullname = thing.get("data-fullname") or ""
        cid = fullname[3:] if fullname.startswith("t1_") else fullname

        comments.append({
            "body": body,
            "author": thing.get("data-author"),
            "id": cid,
            "created_utc": created,
        })
    return comments

# -------------------------------- main --------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build a suicide-negative-label CSV from a saved old.reddit HTML page.")
    ap.add_argument("input_html")
    ap.add_argument("output_csv")
    ap.add_argument("--start-id", type=int, default=START_ID)
    ap.add_argument("--start-user", type=int, default=START_USER)
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS)
    args = ap.parse_args()

    with open(args.input_html, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")

    comments = extract_comments(soup)

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
