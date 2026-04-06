# Model - DistilBERT
# Note: exported collab file as py file

# code from collab
# from google.colab import files
# uploaded = files.upload()

# imports
import re
import random
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# setting the set seed for reproducible results
random.seed(45)
np.random.seed(45)
torch.manual_seed(45)

# model params
MODEL_DIR = "distilbert/distilbert-base-uncased"
MAX_FEATURE_LENGTH = 256
BATCH_SIZE = 16
NUM_EPOCHS = 2
LR = 2e-5
W_DECAY = 0.01
INITIAL_STEPS = 300

# data and result paths (local folder)
TRAIN_CSV_DIR = "train.csv"
VAL_CSV_DIR = "valid.csv"
TEST_CSV_DIR = "test.csv"
TEST_LABELS_PATH = "test_labels.csv"
OUTPUT_DIR = "./distilbert_lyrics_output"

# setting up the label and num/binary conversion dicts
label_to_num = {"p": 0, "m": 1}
num_to_label = {0: "p", 1: "m"}

# setting up the dfs of each dataset
train_df = pd.read_csv(TRAIN_CSV_DIR)
val_df = pd.read_csv(VAL_CSV_DIR)
test_df = pd.read_csv(TEST_CSV_DIR)
test_labels_df = pd.read_csv(TEST_LABELS_PATH)

# mapping the labels of each set to binary
train_df["label"] = train_df["genre"].map(label_to_num)
val_df["label"] = val_df["genre"].map(label_to_num)
test_labels_df["label"] = test_labels_df["genre"].map(label_to_num)


# simple preprocess function to replace empty strings or whitespaces in text
def preprocess_text(text):
    if not text:
        text = ""
    else:
        text = str(text)

    # remove white spaces
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text

##### this section was adapted using chatgpt
# simple function to compute the classification report for each 
def classification_report(eval_pred):
    iterations, labels = eval_pred

    preds = np.argmax(iterations, axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")

    return {"accuracy": acc, "macro_f1": macro_f1}

# use a tokenizer function to better tokenize text in batches for easier training
def tokenize_text_batches(text_b):
    return tokenizer(text_b["lyrics"], truncation=True, max_length=MAX_FEATURE_LENGTH)
#######


# apply the preprocess function to each dataset
train_df["text"] = train_df["lyrics"].apply(preprocess_text)
val_df["text"] = val_df["lyrics"].apply(preprocess_text)
test_df["text"] = test_df["lyrics"].apply(preprocess_text)

# do a inner join to create a new merged eval set to evaluate model predictions
test_eval_df = test_df.merge(test_labels_df[["lyric_id", "genre", "label"]], on="lyric_id", how="inner")

# convert pandas df to hf datasets to better process them and evaluate them late
train_hf = Dataset.from_pandas(train_df[["lyric_id", "lyrics", "label"]].rename(columns={"label": "labels"}),preserve_index=False)
val_hf = Dataset.from_pandas(val_df[["lyric_id", "lyrics", "label"]].rename(columns={"label": "labels"}), preserve_index=False)
test_hf = Dataset.from_pandas(test_df[["lyric_id", "lyrics"]], preserve_index=False)

# use auto tokenizer from the pretrained model (using HF tutorial)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# use a tokenizer function to better tokenize text in batches for easier training
train_hf = train_hf.map(tokenize_text_batches, batched=True)
val_hf = val_hf.map(tokenize_text_batches, batched=True)
test_hf = test_hf.map(tokenize_text_batches, batched=True)

# collect the data using HF datacollector (using HF tutorial)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# load the model from HF
db_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=2, id2label=num_to_label, label2id=label_to_num)

# define the training args for the model
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    weight_decay=W_DECAY,
    warmup_steps=INITIAL_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE
)

# define the trainer method for the model using all args and datasets
model_trainer = Trainer(
    model=db_model,
    args=training_args,
    train_dataset=train_hf,
    eval_dataset=val_hf,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=classification_report
)

# train the model
model_trainer.train()
model_evals = model_trainer.evaluate(eval_dataset = val_hf)

# correctly generate the test predictions using the model
test_pred_output = model_trainer.predict(test_hf)
test_pred_id = np.argmax(test_pred_output.predictions, axis=-1)
test_pred_labels = [num_to_label[i] for i in test_pred_id]
test_predictions_df = pd.DataFrame({"lyric_id": test_df["lyric_id"].values, "pred_genre": test_pred_labels, "pred_label_id": test_pred_id})

# do a inner join to create a new merged eval set to evaluate final model metrics
eval_df = test_predictions_df.merge(test_labels_df[["lyric_id", "genre", "label"]], on="lyric_id", how="inner").rename(columns={"genre": "true_genre", "label": "true_label_id"})

# get the true and predicted labels
genre_true_label = eval_df["true_label_id"].values
genre_pred_label = eval_df["pred_label_id"].values

# compute the two model metrics
test_accuracy = accuracy_score(genre_true_label, genre_pred_label)
test_macro_f1 = f1_score(genre_true_label, genre_pred_label, average="macro")

# print the final metrics
print(f"Accuracy   : {test_accuracy:.4f}")
print(f"Macro F1   : {test_macro_f1:.4f}")