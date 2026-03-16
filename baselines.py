import argparse
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def eval_metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def main(train_path, val_path, test_path, out_test_pred_path, seed=42):
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)

    # expected columns:
    # train/val: lyric_id, genre, lyrics
    # test: lyric_id, lyrics
    for df in (train, val, test):
        df["lyrics"] = df["lyrics"].astype(str).fillna("")

    train["genre"] = train["genre"].astype(str).str.strip().str.lower()
    val["genre"] = val["genre"].astype(str).str.strip().str.lower()

    labels = sorted(train["genre"].unique())
    rng = np.random.default_rng(seed)

    # ------------------------
    # 1) SIMPLE BASELINES
    # ------------------------

    # Majority baseline (most common label in TRAIN)
    majority_label = train["genre"].value_counts().idxmax()
    maj_pred_val = np.full(len(val), majority_label)

    # Random baseline (sample labels according to TRAIN label distribution)
    probs = train["genre"].value_counts(normalize=True).reindex(labels).fillna(0).values
    rand_pred_val = rng.choice(labels, size=len(val), p=probs)

    print("\n=== Simple baselines (on val) ===")
    print(f"Majority label (train): {majority_label}")
    print("Majority:", eval_metrics(val["genre"], maj_pred_val))
    print("Random (train-prior):", eval_metrics(val["genre"], rand_pred_val))

    # ------------------------
    # 2) TRAINED BASELINE
    # ------------------------
    # Logistic Regression with TF-IDF features (simple + strong baseline)
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),   
            min_df=2,
            max_features=50000
        )),
        ("lr", LogisticRegression(
            max_iter=2000,
            solver="lbfgs"
        ))
    ])

    model.fit(train["lyrics"], train["genre"])
    pred_val = model.predict(val["lyrics"])

    print("\n=== Trained baseline (on val) ===")
    print("LogReg + TFIDF:", eval_metrics(val["genre"], pred_val))

    # ------------------------
    # 3) PREDICT ON TEST + SAVE
    # ------------------------
    pred_test = model.predict(test["lyrics"])

    out = pd.DataFrame({
        "lyric_id": test["lyric_id"],
        "genre": pred_test
    })
    out.to_csv(out_test_pred_path, index=False)
    print(f"\nSaved test predictions to: {out_test_pred_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--val", default="val.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--out_test", default="pred_test.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args.train, args.val, args.test, args.out_test, seed=args.seed)