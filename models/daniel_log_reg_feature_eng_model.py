import re
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report


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
val = pd.read_csv("val.csv")

X_train, y_train = train["lyrics"], train["genre"].str.lower().str.strip()
X_val, y_val = val["lyrics"], val["genre"].str.lower().str.strip()

model.fit(X_train, y_train)
pred = model.predict(X_val)

print({
    "accuracy": accuracy_score(y_val, pred),
    "macro_f1": f1_score(y_val, pred, average="macro")
})

print(classification_report(y_val, pred, digits=4))