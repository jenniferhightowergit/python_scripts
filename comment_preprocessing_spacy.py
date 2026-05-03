"""
AMS Meter Comment Pre-Processing for NLP -- spaCy version.

Why this exists
---------------
Iterative cleaning pipeline for free-text comments on AMS meter records.
Designed to feed a DataFrame with columns ["id", "comment"] through a
configurable scrubber so the most informative vocabulary surfaces first.

Pipeline (per comment, via spaCy)
---------------------------------
1. Run text through spaCy's pipeline (tokenize + tag + lemmatize).
2. Drop punctuation, whitespace, digits, and spaCy's built-in stopwords.
3. Drop tokens whose POS tag is in `drop_pos`. Default is to strip
   adjectives, verbs, adverbs, and auxiliaries (matches your original
   script's intent). `drop_nouns=True` on clean_dataframe also strips
   nouns / proper nouns if you want to inspect what's left.
4. Use the lemma (not the surface form) of survivors --
   "communicating", "communicated", "communicates" all collapse to
   "communicate", "meters" -> "meter", "ran" -> "run".
5. Drop tokens shorter than `min_len`.
6. Drop your two custom lists:
       PERMANENT_REMOVE  -> noise you've decided is gone forever
       TEMP_REMOVE       -> known-important-but-loud words you want hidden
                            during review so rarer terms surface

Iterative workflow
------------------
    run -> top_words() -> inspect
        -> move noise into PERMANENT_REMOVE
        -> move "known-important but loud" words into TEMP_REMOVE
        -> rerun
    repeat until what's left is the vocabulary you actually want for NLP.

Setup
-----
One-time install (will use whatever pip proxy is already configured):

    pip install spacy
    python -m spacy download en_core_web_sm

If your coworker uses a bigger model (en_core_web_md or _lg), change
SPACY_MODEL below to match.
"""

import warnings
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import spacy

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Load spaCy model once. Disable components we don't need so it's faster.
# ---------------------------------------------------------------------------
SPACY_MODEL = "en_core_web_sm"

def _load_model(name=SPACY_MODEL):
    try:
        # We only need tokenizer + tagger + lemmatizer.
        # Disable NER and parser for a meaningful speed-up.
        return spacy.load(name, disable=["ner", "parser"])
    except OSError:
        raise OSError(
            f"spaCy model '{name}' isn't installed. Run:\n"
            f"    python -m spacy download {name}\n"
            "If your coworker uses a different model, set SPACY_MODEL "
            "at the top of this file to match."
        )

NLP = _load_model()


# ---------------------------------------------------------------------------
# Custom word lists -- edit these as you iterate
# ---------------------------------------------------------------------------

# Pure noise. Stripped permanently. Add to this list as you find more junk.
PERMANENT_REMOVE = {
    # "thank", "please", "asap",
}

# Known-important-but-loud. Hidden during review so rarer terms surface.
# Graduate words from here back into your downstream "clean words" function
# once you're done iterating.
TEMP_REMOVE = {
    # "meter", "outage", "reading", "install",
}


# ---------------------------------------------------------------------------
# POS classes to drop
# ---------------------------------------------------------------------------
# spaCy uses Universal POS tags. Common ones for our use:
#   ADJ   adjective       VERB  verb           AUX   auxiliary verb
#   ADV   adverb          NOUN  noun           PROPN proper noun
#   NUM   numeral         PRON  pronoun        DET   determiner
#   ADP   preposition     CCONJ conjunction
#
# Default = strip ADJ + VERB + ADV + AUX (matches your original intent
# to remove adjectives, verbs, adverbs).
DROP_POS_DEFAULT = {"ADJ", "VERB", "ADV", "AUX"}
DROP_POS_INC_NOUN = DROP_POS_DEFAULT | {"NOUN", "PROPN"}


# ---------------------------------------------------------------------------
# Core cleaning logic (shared between single-row and batch paths)
# ---------------------------------------------------------------------------

def _filter_doc(doc, drop_pos, permanent_remove, temp_remove, min_len):
    out = []
    for tok in doc:
        if tok.is_punct or tok.is_space:        continue
        if tok.like_num or tok.is_digit:        continue
        if tok.is_stop:                         continue
        if tok.pos_ in drop_pos:                continue
        lemma = tok.lemma_.lower().strip()
        if len(lemma) < min_len:                continue
        if not lemma.isalpha():                 continue
        if lemma in permanent_remove:           continue
        if lemma in temp_remove:                continue
        out.append(lemma)
    return out


def clean_comment(
    text,
    drop_pos=DROP_POS_DEFAULT,
    permanent_remove=None,
    temp_remove=None,
    min_len=3,
):
    """Clean one comment string into a list of surviving lemmas."""
    if permanent_remove is None: permanent_remove = PERMANENT_REMOVE
    if temp_remove is None: temp_remove = TEMP_REMOVE
    if not isinstance(text, str) or not text.strip():
        return []
    return _filter_doc(NLP(text), drop_pos, permanent_remove, temp_remove, min_len)


def clean_dataframe(
    df,
    text_col="comment",
    id_col="id",
    drop_nouns=False,
    permanent_remove=None,
    temp_remove=None,
    min_len=3,
    batch_size=64,
):
    """
    Apply cleaning to every row of `df`. Returns a copy with a new
    `tokens` column (list of lemmas per row). Uses nlp.pipe for speed --
    important once you're processing thousands of comments.

    drop_nouns=True also strips nouns / proper nouns -- useful only when
    you want to inspect non-noun vocabulary in isolation.
    """
    drop_pos = DROP_POS_INC_NOUN if drop_nouns else DROP_POS_DEFAULT
    if permanent_remove is None: permanent_remove = PERMANENT_REMOVE
    if temp_remove is None: temp_remove = TEMP_REMOVE

    texts = df[text_col].fillna("").astype(str).tolist()
    rows = []
    for doc in NLP.pipe(texts, batch_size=batch_size):
        rows.append(_filter_doc(doc, drop_pos, permanent_remove, temp_remove, min_len))

    out = df.copy()
    out["tokens"] = rows
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
