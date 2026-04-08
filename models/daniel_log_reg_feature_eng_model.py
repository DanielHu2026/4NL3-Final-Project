import re
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def make_numeric_features(texts: pd.Series) -> np.ndarray:
    feats = []
    for t in texts.fillna("").astype(str):
        lines = t.splitlines()
        words = re.findall(r"[A-Za-z']+", t.lower())
        n_words = len(words)
        n_unique = len(set(words))
        n_chars = len(t)
        n_lines = len(lines)

        bracket_tags = len(re.findall(r"\[.*?\]", t))
        excls = t.count("!")
        qmarks = t.count("?")
        caps_words = len(re.findall(r"\b[A-Z]{2,}\b", t))

        ttr = (n_unique / n_words) if n_words else 0.0

        feats.append([
            n_words, n_unique, n_chars, n_lines,
            bracket_tags, excls, qmarks, caps_words,
            ttr
        ])
    return np.array(feats, dtype=float)


num_pipe = Pipeline([
    ("extract", FunctionTransformer(lambda X: make_numeric_features(X), validate=False)),
    ("scale", StandardScaler())
])

word_tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9,  
    max_features=50000,
    sublinear_tf=True,
)

char_tfidf = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 6),
    min_df=2,
    max_df=0.9,
    max_features=50000,
    sublinear_tf=True,
)

featurizer = FeatureUnion([
    ("word", word_tfidf),
    ("char", char_tfidf),
    ("num", num_pipe),
])

model = Pipeline([
    ("features", featurizer),
    ("lr", LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        C=8.0,             
    ))
])

# Fit/eval
train = pd.read_csv("train.csv")
test = pd.read_csv("valid.csv")

X_train, y_train = train["lyrics"], train["genre"].str.lower().str.strip()
X_val, y_val = test["lyrics"], test["genre"].str.lower().str.strip()

model.fit(X_train, y_train)
pred = model.predict(X_val)

print({
    "accuracy": accuracy_score(y_val, pred),
    "macro_f1": f1_score(y_val, pred, average="macro")
})
print(classification_report(y_val, pred, digits=4))

# Error Analysis section for artifacts
# calculate probs and CIs
proba = model.predict_proba(X_val)
classes = model.named_steps["lr"].classes_
class_to_idx = {c: i for i, c in enumerate(classes)}


# enumerate through predictions to the get the true and predicted label (to be used for error analysis)
pred_conf = []
true_conf = []
margin = []
for i, true_label in enumerate(y_val):

    pred_label = pred[i]
    pred_idx = class_to_idx[pred_label]
    true_idx = class_to_idx[true_label]


    pred_p = proba[i, pred_idx]
    true_p = proba[i, true_idx]


    pred_conf.append(pred_p)
    true_conf.append(true_p)

    margin.append(pred_p - true_p)


#create the confusion matrix
conf_m = confusion_matrix(y_val, pred, labels=classes)
conf_disp = ConfusionMatrixDisplay(confusion_matrix=conf_m, display_labels=classes)
fig, ax = plt.subplots(figsize=(5, 5))
conf_disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix_val.png", dpi=300, bbox_inches="tight")
plt.close()

# prediction analysis table + sections from confusion matrix
error_df = pd.DataFrame({
    "lyrics": X_val.fillna("").astype(str).values,
    "true_label": y_val.values,
    "pred_label": pred,
    "pred_confidence": pred_conf,
    "true_label_confidence": true_conf,
    "error_margin": margin,
})

# adding additional attributes to the error df to make it more complete and comprehensive
error_df["is_correct"] = error_df["true_label"] == error_df["pred_label"]
error_df["text_length_chars"] = error_df["lyrics"].str.len()
error_df["text_length_words"] = error_df["lyrics"].str.split().str.len()

# Save only the main table of mistakes
incorrectclassified_df = error_df[~error_df["is_correct"]].copy()
incorrectclassified_df = incorrectclassified_df.sort_values(
    by=["pred_confidence", "error_margin"],
    ascending=[False, False]
)
top_wrong_df = incorrectclassified_df.head(20).copy()
top_wrong_df.to_csv("top_misclassified_examples_val.csv", index=False)

print("Saved error analysis artifacts:")
print("- confusion_matrix_val.png")
print("- top_misclassified_examples_val.csv")