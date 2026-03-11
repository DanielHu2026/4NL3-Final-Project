import pandas as pd
import os

from model import Model

input_dir = "/app/input_data"
output_dir = "/app/output"

train = pd.read_csv(os.path.join(input_dir, "train.csv"))
test = pd.read_csv(os.path.join(input_dir, "test.csv"))

X_train = train["lyrics"]
y_train = train["genre"]

X_test = test["lyrics"]

model = Model()
model.fit(X_train, y_train)

pred = model.predict(X_test)

submission = pd.DataFrame({
    "lyric_id": test["lyric_id"],
    "genre": pred
})

submission.to_csv(os.path.join(output_dir, "prediction.csv"), index=False)