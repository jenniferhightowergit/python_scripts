"""
AMS Meter Comment Pre-Processing for NLP
========================================

Iterative cleaning pipeline for free-text comments on AMS meter records.
Designed to feed a DataFrame with columns ["id", "comment"] through a
configurable scrubber so the most informative vocabulary surfaces first.

Pipeline (per comment)
----------------------
1. Lowercase, strip everything that isn't a letter / whitespace.
2. Tokenize.
3. POS-tag and drop adjectives, verbs, adverbs (optionally also nouns).
4. POS-aware lemmatize the survivors.
5. Drop NLTK English stopwords.
6. Drop two custom lists you maintain by hand:
       PERMANENT_REMOVE  -> noise you've decided is gone forever
       TEMP_REMOVE       -> words you know you'll want eventually,
                            but want hidden right now so rarer terms
                            float to the top during review
7. Report top-N word frequencies (table + bar chart) for you to inspect.

Iterative workflow
------------------
    run -> inspect top_words output
        -> move noise into PERMANENT_REMOVE
        -> move "known-important but loud" words into TEMP_REMOVE
        -> rerun
    repeat until what's left is the vocabulary you actually want to model.
"""

import re
import warnings
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore")

# --- one-time NLTK downloads (safe to call repeatedly) ---
for _pkg in (
    "punkt",
    "punkt_tab",
    "stopwords",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "wordnet",
    "omw-1.4",
):
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# ---------------------------------------------------------------------------
# Custom word lists -- edit these as you iterate
# ---------------------------------------------------------------------------

# Words you've decided are pure noise for AMS meter analysis.
# These get stripped permanently. Add to this list as you find more junk.
PERMANENT_REMOVE = {
    # examples -- replace with what shows up as noise in your data
    # "thank", "please", "asap",
}

# Words you know you'll want long-term, but want to hide during review
# so less-common (potentially more informative) terms surface.
# Move things between TEMP_REMOVE and PERMANENT_REMOVE as you learn the
# vocabulary; eventually graduate the keepers into your "clean words"
# function elsewhere in the project.
TEMP_REMOVE = {
    # "meter", "outage", "reading", "install",
}


# ---------------------------------------------------------------------------
# POS handling
# ---------------------------------------------------------------------------

# Penn Treebank tag prefixes:
#   J = adjective, V = verb, R = adverb, N = noun
DROP_POS_DEFAULT = ("J", "V", "R")          # drop ADJ / VERB / ADV, keep nouns
DROP_POS_INC_NOUN = ("J", "V", "R", "N")    # also drop nouns


def _treebank_to_wordnet(tag):
    """Map a Penn Treebank POS tag to a WordNet POS for lemmatization."""
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN  # default


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_comment(
    text,
    drop_pos_prefixes=DROP_POS_DEFAULT,
    permanent_remove=None,
    temp_remove=None,
    min_len=3,
):
    """
    Clean a single comment string and return a list of surviving lemmas.
    """
    if permanent_remove is None:
        permanent_remove = PERMANENT_REMOVE
    if temp_remove is None:
        temp_remove = TEMP_REMOVE

    if not isinstance(text, str) or not text.strip():
        return []

    # 1. lowercase + strip non-letters
    text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())

    # 2. tokenize
    tokens = word_tokenize(text)

    # 3. POS-tag, drop unwanted parts of speech, lemmatize the rest
    kept = []
    for tok, tag in pos_tag(tokens):
        if tag.startswith(drop_pos_prefixes):
            continue
        kept.append(LEMMATIZER.lemmatize(tok, _treebank_to_wordnet(tag)))

    # 4. length / stopword / custom-list filtering
    out = []
    for w in kept:
        if len(w) < min_len:
            continue
        if w in STOP_WORDS:
            continue
        if w in permanent_remove:
            continue
        if w in temp_remove:
            continue
        out.append(w)
    return out


def clean_dataframe(
    df,
    text_col="comment",
    id_col="id",
    drop_nouns=False,
    permanent_remove=None,
    temp_remove=None,
):
    """
    Apply clean_comment row-wise. Returns a copy of `df` with an added
    `tokens` column (list of lemmas per row).

    Parameters
    ----------
    df : DataFrame with at least `id_col` and `text_col`.
    drop_nouns : bool
        If True, also strip nouns. Useful when you want to examine the
        non-noun vocabulary in isolation. Default False.
    """
    drop_pos = DROP_POS_INC_NOUN if drop_nouns else DROP_POS_DEFAULT
    out = df.copy()
    out["tokens"] = out[text_col].apply(
        lambda t: clean_comment(
            t,
            drop_pos_prefixes=drop_pos,
            permanent_remove=permanent_remove,
            temp_remove=temp_remove,
        )
    )
    return out[[id_col, text_col, "tokens"]]


# ---------------------------------------------------------------------------
# Frequency reporting -- the iteration surface
# ---------------------------------------------------------------------------

def top_words(cleaned_df, n=30, token_col="tokens"):
    """
    Return (Counter, DataFrame) of the top-n tokens across the corpus.
    The DataFrame columns are ['word', 'count'].
    """
    counter = Counter()
    for toks in cleaned_df[token_col]:
        counter.update(toks)
    top = counter.most_common(n)
    return counter, pd.DataFrame(top, columns=["word", "count"])


def plot_top_words(top_df, title="Top words in AMS meter comments"):
    """Horizontal bar chart of a top_words() result."""
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
    # Replace with your actual load -- e.g. df = pd.read_csv(...)
    d = [
        ["id1", "Meter is not communicating after the storm last week."],
        ["id2", "Customer reports flickering lights, meter reading looks normal."],
        ["id3", "Replaced meter due to repeated non-com events."],
    ]
    df = pd.DataFrame(d, columns=["id", "comment"])

    # 1. Clean
    cleaned = clean_dataframe(df, text_col="comment", id_col="id", drop_nouns=False)
    print(cleaned)

    # 2. Inspect what's left
    counter, top_df = top_words(cleaned, n=30)
    print(top_df)

    # 3. Plot for review
    plot_top_words(top_df, title="Top 30 words -- review pass")

    # 4. Iterate: copy noise from `top_df` into PERMANENT_REMOVE,
    #    copy known-important-but-loud words into TEMP_REMOVE,
    #    rerun, repeat.
