#!/usr/bin/env python3
"""
Generate 537 unique replacement messages for the exact-duplicate rows in
suicidality_full_dataset_posneg_balanced.csv, and write a new CSV with
ONLY the `text` field changed for those rows. ID, users, sentiment, time
are left untouched. All other rows are left byte-identical.

Safety: every generated message is run through a suicide/self-harm keyword
blocklist before being accepted, on top of the fact that none of the
templates reference suicide or self-harm in the first place.
"""
import random
import re
import sys
from pathlib import Path

import pandas as pd

random.seed(42)

SRC_CSV = Path("/mnt/user-data/uploads/suicidality_full_dataset_posneg_balanced.csv")
DUPES_CSV = Path("/home/claude/exact_duplicate_groups.csv")
OUT_CSV = Path("/mnt/user-data/outputs/suicidality_full_dataset_posneg_balanced_deduped.csv")

# ---------------------------------------------------------------------------
# Safety net: reject any candidate containing suicide / self-harm language,
# even though none of the templates below reference it.
# ---------------------------------------------------------------------------
BLOCKLIST_PATTERNS = [
    r"suicid", r"kill myself", r"killing myself", r"end my life", r"end it all",
    r"want to die", r"wanna die", r"self.?harm", r"hurt myself", r"cut myself",
    r"cutting myself", r"overdose", r"take my (own )?life", r"no reason to live",
    r"better off dead", r"not worth living", r"harm myself", r"self.?injur",
]
BLOCKLIST_RE = re.compile("|".join(BLOCKLIST_PATTERNS), re.IGNORECASE)


def is_safe(text: str) -> bool:
    return not BLOCKLIST_RE.search(text)


def is_not_redundant(text: str) -> bool:
    # Guard against "Honestly, ... honestly ..." style repetition when an
    # opener and a clause both happen to use the same filler word.
    return text.lower().count("honestly") <= 1


# ---------------------------------------------------------------------------
# Template bank. Each category pairs opener/clause/closing fragments with a
# topic list, producing a large combinatorial pool of coherent, everyday
# messages that express a specific (non-suicidal) emotion.
# ---------------------------------------------------------------------------
OPENERS = ["", "", "", "Okay so", "Honestly,", "Not gonna lie,", "So,",
           "You know what?", "Update:", "Random thought:", "Ngl,"]

CLOSINGS = ["", "", "", " Anyway, that's my day.", " Just needed to say it somewhere.",
            " Figured I'd share.", " That's all, really.", " Small thing but it matters to me.",
            " Just venting a bit.", " Still thinking about it honestly."]

# Topic pools grouped so that only sensible emotion/topic combinations get
# generated (e.g. "boredom" never gets paired with "the wedding").
TOPICS_MUNDANE_ANNOYANCE = [
    "the traffic this morning", "my commute", "the grocery run", "laundry day",
    "the house cleaning", "the group chat", "the line at the store",
    "the flight delay", "the parking ticket", "the missed bus",
    "the noisy neighbors", "my lost keys", "the flat tire",
    "my broken phone screen", "spilled coffee this morning",
    "the car that wouldn't start", "the slow wifi", "the printer at work",
]
TOPICS_BIG_POSITIVE = [
    "the promotion", "the raise", "the wedding", "the new baby news",
    "graduation", "the anniversary", "the new job", "the new puppy",
    "the college acceptance letter", "moving into the new place",
]
TOPICS_EVERYDAY_ACTIVITY = [
    "my cooking", "the gym", "my workout", "my favorite team's game",
    "the show I've been binging", "that new album", "the concert last night",
    "the book I'm reading", "my garden", "my painting", "the new video game",
    "my knitting project", "the photos I took", "our trip", "the holidays",
    "this weekend", "the new recipe I tried", "board game night",
    "the puzzle we started", "the crossword", "the karaoke night",
    "the road trip", "the camping trip", "the beach day", "trivia night",
    "the museum visit", "the dance class", "the language class",
    "the thrift store find", "the garage sale", "book club",
    "the new song that just dropped", "the DIY project",
]
TOPICS_RELATIONSHIPS = [
    "my roommate", "my sister", "my brother", "my mom", "my dad",
    "my best friend", "my neighbor", "my relationship", "my ex",
    "the family reunion", "the video call with my friend",
    "the surprise party", "the potluck", "the barbecue", "the picnic",
]
TOPICS_WORK_SCHOOL = [
    "work", "my job", "school", "my exam", "my group project",
    "the interview I had", "my first day at the new job",
    "the performance review", "the team meeting", "the presentation",
    "finals week", "back to school", "the layoff rumors at work",
    "my budget this month",
]
TOPICS_WEATHER_HEALTH = [
    "the weather", "the storm last night", "the heat wave", "the snow day",
    "the rainy afternoon", "the sunny weekend", "my allergies", "my cold",
    "my sleep schedule", "my morning routine", "the dentist appointment",
    "the doctor's appointment", "the diet I'm trying",
]
TOPICS_PETS = ["my dog", "my cat"]

# ---------------------------------------------------------------------------
# Emotion categories: (clause templates using {topic}, matching topic pools).
# ---------------------------------------------------------------------------
CATEGORIES = {
    "anger": (
        [
            "I am so annoyed about {topic} right now.",
            "Honestly {topic} is making me furious.",
            "I can't believe how mad {topic} made me today.",
            "{topic} has me absolutely fuming.",
            "I'm seething because of {topic}.",
            "It really ticks me off how {topic} turned out.",
            "I'm so fed up with {topic}.",
            "{topic} pushed me over the edge today, I'm livid.",
        ],
        TOPICS_MUNDANE_ANNOYANCE + TOPICS_RELATIONSHIPS + TOPICS_WORK_SCHOOL,
    ),
    "sadness": (
        [
            "I've been feeling kind of down about {topic}.",
            "{topic} left me feeling really blue today.",
            "Something about {topic} made me tear up a little.",
            "I feel a bit heavy-hearted thinking about {topic}.",
            "I'm just a little sad over {topic} right now.",
            "{topic} has me in a gloomy mood today.",
            "I got unexpectedly emotional about {topic}.",
            "Feeling a little melancholy after {topic}.",
        ],
        TOPICS_RELATIONSHIPS + TOPICS_WEATHER_HEALTH + TOPICS_PETS,
    ),
    "happiness": (
        [
            "I'm absolutely thrilled about {topic}.",
            "{topic} made my whole day so much brighter.",
            "I'm over the moon about {topic} right now.",
            "Honestly {topic} put the biggest smile on my face.",
            "I feel so cheerful after {topic} today.",
            "{topic} has me grinning ear to ear.",
            "I'm delighted with how {topic} turned out.",
            "Nothing beats the joy I got from {topic} today.",
        ],
        TOPICS_BIG_POSITIVE + TOPICS_EVERYDAY_ACTIVITY + TOPICS_PETS,
    ),
    "jealousy": (
        [
            "I have to admit I'm a little jealous about {topic}.",
            "{topic} is making me green with envy, not gonna lie.",
            "I wish I had what they have with {topic}.",
            "Seeing {topic} made me feel a bit envious today.",
            "I'm trying not to compare myself over {topic} but it's hard.",
        ],
        TOPICS_BIG_POSITIVE + TOPICS_RELATIONSHIPS,
    ),
    "love": (
        [
            "I absolutely adore {topic}.",
            "My heart is so full because of {topic}.",
            "I'm so smitten thinking about {topic}.",
            "{topic} means the world to me honestly.",
            "I cherish {topic} more than I can say.",
            "There's nothing like the love I feel around {topic}.",
        ],
        TOPICS_RELATIONSHIPS + TOPICS_PETS,
    ),
    "gratitude": (
        [
            "I'm really grateful for {topic} today.",
            "{topic} reminded me how thankful I am for the little things.",
            "I don't say it enough but I appreciate {topic} so much.",
            "Feeling thankful for {topic} right now.",
        ],
        TOPICS_RELATIONSHIPS + TOPICS_BIG_POSITIVE + TOPICS_PETS,
    ),
    "boredom": (
        [
            "I'm so bored, {topic} is the only thing going on today.",
            "Nothing to do but think about {topic}, so restless right now.",
            "I've been staring at the wall thinking about {topic}, so dull today.",
            "Even {topic} couldn't shake how bored I feel today.",
        ],
        TOPICS_MUNDANE_ANNOYANCE + TOPICS_WEATHER_HEALTH,
    ),
    "excitement": (
        [
            "I can't wait for {topic}, I'm so pumped.",
            "I'm buzzing with excitement over {topic}.",
            "{topic} has me counting down the hours honestly.",
            "So hyped about {topic} right now, can't sit still.",
        ],
        TOPICS_BIG_POSITIVE + TOPICS_EVERYDAY_ACTIVITY,
    ),
    "disappointment": (
        [
            "I'm honestly pretty disappointed about {topic}.",
            "{topic} really let me down today.",
            "I was hoping for more from {topic}, kind of a bummer.",
            "Feeling a bit deflated after {topic} didn't go as planned.",
        ],
        TOPICS_EVERYDAY_ACTIVITY + TOPICS_WORK_SCHOOL + TOPICS_MUNDANE_ANNOYANCE,
    ),
    "frustration": (
        [
            "I'm so frustrated trying to sort out {topic}.",
            "{topic} has me stuck and it's exasperating.",
            "Nothing about {topic} is going smoothly today, so frustrating.",
            "I keep hitting walls with {topic} and it's driving me crazy.",
        ],
        TOPICS_MUNDANE_ANNOYANCE + TOPICS_WORK_SCHOOL,
    ),
    "nostalgia": (
        [
            "{topic} made me nostalgic for the old days.",
            "Thinking about {topic} brought back so many memories.",
            "I found myself reminiscing about {topic} earlier.",
            "{topic} isn't the same as it used to be, kind of miss how it was.",
        ],
        TOPICS_RELATIONSHIPS + TOPICS_EVERYDAY_ACTIVITY + TOPICS_WEATHER_HEALTH,
    ),
    "loneliness": (
        [
            "I miss my friends and {topic} just reminded me of that.",
            "Kind of wish I had someone to share {topic} with today.",
            "Feeling a little lonely, wish I could talk to someone about {topic}.",
            "{topic} would've been more fun with company honestly.",
        ],
        TOPICS_EVERYDAY_ACTIVITY + TOPICS_RELATIONSHIPS,
    ),
    "stress": (
        [
            "I'm so stressed about {topic} right now.",
            "{topic} has me overwhelmed with everything on my plate.",
            "Feeling really anxious about {topic} today.",
            "I can't stop worrying about {topic}, my mind won't slow down.",
        ],
        TOPICS_WORK_SCHOOL + TOPICS_MUNDANE_ANNOYANCE + TOPICS_WEATHER_HEALTH,
    ),
    "relief": (
        [
            "Such a relief that {topic} is finally over.",
            "I feel like a weight came off my shoulders after {topic}.",
            "So glad {topic} worked out in the end, big relief.",
            "Finally I can breathe easy now that {topic} is done.",
        ],
        TOPICS_WORK_SCHOOL + TOPICS_MUNDANE_ANNOYANCE + TOPICS_WEATHER_HEALTH,
    ),
    "pride": (
        [
            "I'm actually really proud of how {topic} went.",
            "{topic} turned out great and I feel accomplished.",
            "Feeling proud of myself for sticking with {topic}.",
            "I surprised myself with {topic} today, feeling good about it.",
        ],
        TOPICS_WORK_SCHOOL + TOPICS_EVERYDAY_ACTIVITY + TOPICS_BIG_POSITIVE,
    ),
    "embarrassment": (
        [
            "I was so embarrassed about {topic} earlier, still cringing.",
            "{topic} was mortifying honestly, I can't stop replaying it.",
            "I turned bright red because of {topic} today.",
            "Still cringing a little over {topic}.",
        ],
        TOPICS_WORK_SCHOOL + TOPICS_RELATIONSHIPS + TOPICS_EVERYDAY_ACTIVITY,
    ),
    "hope": (
        [
            "I'm feeling hopeful about {topic} lately.",
            "{topic} has me optimistic about what's next.",
            "Trying to stay positive about {topic} this week.",
            "I have a good feeling about {topic} honestly.",
        ],
        TOPICS_WORK_SCHOOL + TOPICS_BIG_POSITIVE + TOPICS_RELATIONSHIPS,
    ),
    "contentment": (
        [
            "I feel really at peace with {topic} today.",
            "{topic} left me feeling calm and content.",
            "Simple day, just enjoying {topic} and feeling satisfied.",
            "Nothing fancy, just content after {topic}.",
        ],
        TOPICS_EVERYDAY_ACTIVITY + TOPICS_WEATHER_HEALTH + TOPICS_PETS,
    ),
    "surprise": (
        [
            "I did not expect {topic} at all, caught me off guard.",
            "{topic} totally surprised me today.",
            "Still processing how unexpected {topic} was.",
            "Did not see {topic} coming, honestly shocked.",
        ],
        TOPICS_BIG_POSITIVE + TOPICS_WORK_SCHOOL + TOPICS_MUNDANE_ANNOYANCE,
    ),
    "curiosity": (
        [
            "I'm really curious about {topic} now.",
            "{topic} has me wanting to know more.",
            "Can't stop wondering about {topic} today.",
            "Something about {topic} has me intrigued.",
        ],
        TOPICS_EVERYDAY_ACTIVITY + TOPICS_BIG_POSITIVE,
    ),
    "determination": (
        [
            "I'm determined to get {topic} right this time.",
            "{topic} didn't go well but I'm motivated to try again.",
            "Setting my mind on {topic} and not giving up.",
            "Feeling driven to tackle {topic} head on.",
        ],
        TOPICS_WORK_SCHOOL + TOPICS_EVERYDAY_ACTIVITY,
    ),
}

# Words for which lowercasing the first letter (to splice after an opener)
# would be grammatically wrong -- the pronoun "I" and its contractions.
_I_FORMS = {"i", "i'm", "i've", "i'll", "i'd"}

# Openers that read as a comma-style continuation of the same sentence
# ("Honestly, i'm..." -> lowercase what follows). Openers ending in "?" or ":"
# start a new sentence, so what follows must stay capitalized.
_CONTINUATION_OPENERS = {"okay so", "honestly,", "not gonna lie,", "so,", "ngl,"}


def _cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def _lower_first(s: str) -> str:
    return s[0].lower() + s[1:] if s else s


def _splice_opener(opener: str, body: str) -> str:
    first_word = body.split(" ", 1)[0]
    is_pronoun_i = first_word.lower() in _I_FORMS

    if not opener:
        return body if is_pronoun_i else _cap(body)

    if opener.lower() in _CONTINUATION_OPENERS:
        joined = body if is_pronoun_i else _lower_first(body)
        return f"{opener} {joined}"

    # ":" or "?" openers ("Update:", "Random thought:", "You know what?")
    # start a fresh sentence, so keep/force capitalization.
    return f"{opener} {_cap(body)}"


def build_pool():
    pool = set()
    for clauses, topics in CATEGORIES.values():
        for clause in clauses:
            for topic in topics:
                for opener in OPENERS:
                    for closing in CLOSINGS:
                        body = clause.format(topic=topic)
                        msg = _splice_opener(opener, body) + closing
                        msg = re.sub(r"\s+", " ", msg).strip()
                        msg = _cap(msg)  # final safeguard
                        if 50 <= len(msg) <= 600 and is_safe(msg) and is_not_redundant(msg):
                            pool.add(msg)
    return pool


def main():
    if not SRC_CSV.exists():
        print(f"Missing source csv: {SRC_CSV}")
        sys.exit(1)
    if not DUPES_CSV.exists():
        print(f"Missing duplicate list: {DUPES_CSV}")
        sys.exit(1)

    df = pd.read_csv(SRC_CSV)
    dupes = pd.read_csv(DUPES_CSV)
    dup_ids = dupes["ID"].tolist()
    n_needed = len(dup_ids)
    print(f"Rows to replace: {n_needed}")

    existing_texts = set(df["text"].astype(str).str.strip())

    pool = build_pool()
    print(f"Candidate pool size (unique, length/safety filtered): {len(pool)}")

    # Remove anything that collides with text already present elsewhere in the dataset.
    pool = [m for m in pool if m.strip() not in existing_texts]
    print(f"Candidate pool after removing collisions with existing dataset text: {len(pool)}")

    if len(pool) < n_needed:
        print("ERROR: not enough unique candidates generated. Expand templates.")
        sys.exit(1)

    pool_sorted = sorted(pool)  # deterministic ordering before sampling
    random.shuffle(pool_sorted)
    chosen = pool_sorted[:n_needed]

    assert len(set(chosen)) == n_needed, "Chosen replacements are not internally unique!"
    assert all(is_safe(m) for m in chosen), "Safety filter failure on chosen set!"
    assert all(50 <= len(m) <= 600 for m in chosen), "Length constraint violated!"

    id_to_new_text = dict(zip(dup_ids, chosen))

    # Apply replacement to ONLY the text field of matching IDs; nothing else touched.
    mask = df["ID"].isin(id_to_new_text.keys())
    df.loc[mask, "text"] = df.loc[mask, "ID"].map(id_to_new_text)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    # QUOTE_MINIMAL (default) matches the source file: only quotes fields that
    # need it (contain a comma, quote character, or newline).
    df.to_csv(OUT_CSV, index=False, quoting=0)
    print(f"Wrote: {OUT_CSV}")

    # Sanity report
    print("\nSample of replacements:")
    for i, rid in enumerate(dup_ids[:8]):
        print(f"  ID={rid}: {id_to_new_text[rid]!r}")


if __name__ == "__main__":
    main()
