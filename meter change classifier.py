# =============================================================================
# METER CHANGE BINARY CLASSIFIER
# Trains on labeled comments (1 = meter change, 0 = not meter change)
# Produces: confusion matrix, miss analysis, word-level explanations
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# STEP 1: LOAD YOUR LABELED DATA
# Expected columns: 'comment' and 'label' (1 = meter change, 0 = not)
# =============================================================================

df = pd.read_parquet("your_labeled_data.parquet")  # swap in your actual path

# Quick sanity check
print("Label distribution:")
print(df["label"].value_counts())
print(f"\nTotal comments: {len(df)}")


# =============================================================================
# STEP 2: TRAIN / TEST SPLIT
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    df["comment"],
    df["label"],
    test_size=0.2,       # 80% train, 20% test
    random_state=42,
    stratify=df["label"] # keeps same 1/0 ratio in both splits
)

print(f"\nTraining on {len(X_train)} comments, testing on {len(X_test)}")


# =============================================================================
# STEP 3: BUILD AND TRAIN THE MODEL
# =============================================================================

vectorizer = TfidfVectorizer(
    max_features=500,   # top 500 most useful words
    ngram_range=(1, 2), # single words AND two-word phrases
    min_df=2            # ignore words that appear only once
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_vec, y_train)

print("\nModel trained.")


# =============================================================================
# STEP 4: PREDICTIONS + CONFIDENCE SCORES
# =============================================================================

y_pred       = model.predict(X_test_vec)
y_pred_proba = model.predict_proba(X_test_vec)[:, 1]  # probability of being meter change

results_df = X_test.reset_index(drop=True).to_frame()
results_df["true_label"]   = y_test.reset_index(drop=True)
results_df["predicted"]    = y_pred
results_df["confidence"]   = y_pred_proba.round(3)
results_df["correct"]      = results_df["true_label"] == results_df["predicted"]


# =============================================================================
# STEP 5: CONFUSION MATRIX
# =============================================================================
#
#                   PREDICTED
#                   Not MC    Meter Change
# ACTUAL  Not MC  [ TN         FP ]   <- FP = wrongly flagged as meter change
#         MC      [ FN         TP ]   <- FN = missed real meter changes
#
# =============================================================================

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\n" + "="*50)
print("CONFUSION MATRIX")
print("="*50)
print(f"                  Predicted")
print(f"                  Not MC    Meter Change")
print(f"Actual  Not MC  [ {tn:<8}  {fp:<12}]")
print(f"        MC      [ {fn:<8}  {tp:<12}]")
print()
print(f"True Positives  (correctly caught meter changes): {tp}")
print(f"True Negatives  (correctly excluded non-MC):      {tn}")
print(f"False Positives (wrongly flagged as meter change): {fp}")
print(f"False Negatives (missed real meter changes):       {fn}")

print("\n" + classification_report(y_test, y_pred,
      target_names=["Not Meter Change", "Meter Change"]))


# =============================================================================
# STEP 6: WHAT DID WE MISS?
# False Negatives = real meter changes the model didn't catch
# False Positives = comments wrongly labeled as meter change
# =============================================================================

false_negatives = results_df[
    (results_df["true_label"] == 1) & (results_df["predicted"] == 0)
].sort_values("confidence")

false_positives = results_df[
    (results_df["true_label"] == 0) & (results_df["predicted"] == 1)
].sort_values("confidence", ascending=False)

print("\n" + "="*50)
print(f"FALSE NEGATIVES — Missed meter changes ({len(false_negatives)} total)")
print("These are real meter changes the model didn't catch")
print("="*50)
print(false_negatives[["comment", "confidence"]].to_string(index=False))

print("\n" + "="*50)
print(f"FALSE POSITIVES — Wrong flags ({len(false_positives)} total)")
print("These were flagged as meter change but aren't")
print("="*50)
print(false_positives[["comment", "confidence"]].to_string(index=False))


# =============================================================================
# STEP 7: WHY DID THE MODEL FLAG THIS?
# Shows which words pushed a comment toward or away from "meter change"
# =============================================================================

feature_names = vectorizer.get_feature_names_out()
coefficients  = model.coef_[0]

# Top words that DRIVE meter change classification
top_positive = pd.Series(coefficients, index=feature_names)\
    .sort_values(ascending=False).head(20)

# Top words that ARGUE AGAINST meter change
top_negative = pd.Series(coefficients, index=feature_names)\
    .sort_values(ascending=True).head(20)

print("\n" + "="*50)
print("TOP WORDS THAT SIGNAL 'METER CHANGE'")
print("="*50)
for word, score in top_positive.items():
    bar = "█" * int(abs(score) * 5)
    print(f"  {word:<30} {bar}  ({score:.3f})")

print("\n" + "="*50)
print("TOP WORDS THAT SIGNAL 'NOT METER CHANGE'")
print("="*50)
for word, score in top_negative.items():
    bar = "█" * int(abs(score) * 5)
    print(f"  {word:<30} {bar}  ({score:.3f})")


# =============================================================================
# STEP 8: EXPLAIN A SPECIFIC COMMENT
# Useful for walking Willie through "why did this get flagged"
# =============================================================================

def explain_comment(comment_text):
    """
    Shows the confidence score and which words in the comment
    contributed most to the prediction.
    """
    vec      = vectorizer.transform([comment_text])
    pred     = model.predict(vec)[0]
    prob     = model.predict_proba(vec)[0][1]
    label    = "METER CHANGE" if pred == 1 else "NOT METER CHANGE"

    # Find which words in this comment have high coefficients
    feature_index = vec.nonzero()[1]
    word_scores   = [(feature_names[i], coefficients[i]) for i in feature_index]
    word_scores.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\nComment: '{comment_text}'")
    print(f"Prediction: {label}  |  Confidence: {prob:.1%}")
    print("Key words driving this prediction:")
    for word, score in word_scores[:10]:
        direction = "→ meter change" if score > 0 else "→ NOT meter change"
        print(f"  '{word}' {direction}  ({score:.3f})")

# Example usage:
explain_comment("METER EXCHANGED DUE TO FAILED REGISTER")
explain_comment("NO COMM RECEIVED FROM ENDPOINT")


# =============================================================================
# STEP 9: EXPORT FULL RESULTS
# =============================================================================

results_df.to_parquet("meter_change_predictions.parquet", index=False)
print("\nFull results saved to meter_change_predictions.parquet")
