"""
AMS Meter Comment Pre-Processing for NLP -- offline version.

Why this exists in this form
----------------------------
Oncor's network blocks NLTK's data downloads, so we can't use NLTK's
tagger / lemmatizer / stopwords corpus. This module reproduces the same
preprocessing pipeline using only:

    * the Python standard library (re, collections)
    * pandas / matplotlib
    * an embedded English stopwords list (copied from NLTK)
    * suffix heuristics in place of a real POS tagger

Pipeline (per comment)
----------------------
1. Lowercase + tokenize on letters only (drops punctuation/digits).
2. Optionally drop tokens whose suffix marks them as likely
   adverbs / adjectives / verbs.  Off by default, because for failure
   analysis verbs like "failed", "tripped", "burned" are signal, not noise.
3. Light singularize ("readings" -> "reading", "meters" -> "meter").
4. Drop embedded English stopwords.
5. Drop tokens that match your two custom lists:
       PERMANENT_REMOVE  -> noise you've decided is gone forever
       TEMP_REMOVE       -> words you'll want long-term but want hidden
                            during review so rarer terms surface
6. Drop tokens shorter than `min_len` (default 3).

Iterative workflow
------------------
    run -> top_words() -> inspect
        -> move noise into PERMANENT_REMOVE
        -> move "known-important but loud" words into TEMP_REMOVE
        -> rerun
    repeat until what's left is the vocabulary you actually want for NLP.
"""

import re
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Embedded English stopwords (NLTK's list, ~180 words). No download needed.
# ---------------------------------------------------------------------------
ENGLISH_STOPWORDS = frozenset({
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom",
    "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing",
    "a", "an", "the",
    "and", "but", "if", "or", "because", "as", "until", "while",
    "of", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very",
    "can", "will", "just", "don", "should", "now",
    "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven",
    "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn",
    "wasn", "weren", "won", "wouldn",
    # one- and two-letter fragments left over from contractions
    "ll", "re", "ve", "nt",
})


# ---------------------------------------------------------------------------
# Custom word lists -- edit these as you iterate
# ---------------------------------------------------------------------------

# Pure noise. Stripped permanently.
PERMANENT_REMOVE = {
    # "thank", "please", "asap",
}

# Known-important-but-loud. Hidden during review so rarer terms surface.
# Graduate things from here back into your downstream "clean words" function
# once you're done iterating.
TEMP_REMOVE = {
    # "meter", "outage", "reading", "install",
}


# ---------------------------------------------------------------------------
# Suffix-based POS heuristics (substitute for a real tagger)
# ---------------------------------------------------------------------------

ADJECTIVE_SUFFIXES = ("able", "ible", "ous", "ful", "ive", "less")

def _likely_adverb(w):
    # "quickly", "rapidly" -- but len>4 to spare "fly", "ply", "rely"
    return len(w) > 4 and w.endswith("ly")

def _likely_adjective(w):
    return any(w.endswith(s) and len(w) > len(s) + 2 for s in ADJECTIVE_SUFFIXES)

def _likely_verb_form(w):
    # gerund / past-tense.  Heuristic: false positives like "string", "shed"
    # exist but are caught by the iterative review loop.
    return (len(w) > 5 and w.endswith("ing")) or (len(w) > 4 and w.endswith("ed"))


def _singularize(w):
    """Light plural stripper -- not a real lemmatizer."""
    if len(w) > 4 and w.endswith("ies"):                # batteries -> battery
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("es") and w[-3] in "sxz":   # boxes -> box
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-zA-Z]+")

def clean_comment(
    text,
    drop_modifiers=False,   # True = strip likely adverbs/adjectives by suffix
    drop_verbs=False,       # True = strip likely -ing / -ed forms
    permanent_remove=None,
    temp_remove=None,
    min_len=3,
):
    """
    Clean one comment string into a list of surviving tokens.

    drop_modifiers / drop_verbs default to False because for failure
    analysis the verbs and modifiers ("burned", "tripped", "intermittent")
    are usually signal. Flip them on if you want to inspect the noun-only
    vocabulary in isolation.
    """
    if permanent_remove is None:
        permanent_remove = PERMANENT_REMOVE
    if temp_remove is None:
        temp_remove = TEMP_REMOVE
    if not isinstance(text, str) or not text.strip():
        return []

    tokens = _TOKEN_RE.findall(text.lower())

    out = []
    for tok in tokens:
        if drop_modifiers and (_likely_adverb(tok) or _likely_adjective(tok)):
            continue
        if drop_verbs and _likely_verb_form(tok):
            continue
        tok = _singularize(tok)
        if len(tok) < min_len:               continue
        if tok in ENGLISH_STOPWORDS:         continue
        if tok in permanent_remove:          continue
        if tok in temp_remove:               continue
        out.append(tok)
    return out


def clean_dataframe(
    df,
    text_col="comment",
    id_col="id",
    drop_modifiers=False,
    drop_verbs=False,
    permanent_remove=None,
    temp_remove=None,
):
    """
    Apply clean_comment row-wise. Returns a copy of `df` with an added
    `tokens` column (list of strings per row).
    """
    out = df.copy()
    out["tokens"] = out[text_col].apply(
        lambda t: clean_comment(
            t,
            drop_modifiers=drop_modifiers,
            drop_verbs=drop_verbs,
            permanent_remove=permanent_remove,
            temp_remove=temp_remove,
        )
    )
    return out[[id_col, text_col, "tokens"]]


# ---------------------------------------------------------------------------
# Frequency reporting -- the iteration surface
# ---------------------------------------------------------------------------

def top_words(cleaned_df, n=30, token_col="tokens"):
    """Return (Counter, DataFrame[word,count]) of the top-n tokens corpus-wide."""
    counter = Counter()
    for toks in cleaned_df[token_col]:
        counter.update(toks)
    return counter, pd.DataFrame(counter.most_common(n), columns=["word", "count"])


def plot_top_words(top_df, title="Top words in AMS meter comments"):
    plt.figure(figsize=(10, max(0.3 * len(top_df) + 1, 4)))
    plt.barh(top_df["word"], top_df["count"])
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    d = [
        ["m001", "Meter not communicating after the storm last week, please check ASAP."],
        ["m002", "Customer reports flickering lights. Meter reading looks normal."],
        ["m003", "Replaced meter due to repeated non-com events. Thank you."],
        ["m004", "Outage reported in the area; meter was offline for 6 hours."],
        ["m005", "Display is blank on the meter; suspect failed board."],
    ]
    df = pd.DataFrame(d, columns=["id", "comment"])

    cleaned = clean_dataframe(df)
    print(cleaned)

    counter, top_df = top_words(cleaned, n=20)
    print(top_df)
    plot_top_words(top_df, title="Top words -- review pass")
