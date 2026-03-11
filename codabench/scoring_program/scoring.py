import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import json
import os

input_dir = "/app/input"
output_dir = "/app/output"

pred = pd.read_csv(os.path.join(input_dir, "prediction.csv"))
truth = pd.read_csv(os.path.join(input_dir, "test_labels.csv"))

merged = pred.merge(truth, on="lyric_id")

y_pred = merged["genre_x"]
y_true = merged["genre_y"]

scores = {
    "macro_f1": f1_score(y_true, y_pred, average="macro"),
    "accuracy": accuracy_score(y_true, y_pred)
}

with open(os.path.join(output_dir, "scores.json"), "w") as f:
    json.dump(scores, f)