import os
import pandas as pd
import numpy as np

# ---- config ----
DATA_DIR = "./data"
SOURCE_CSV = "train.csv"   # change if needed
OUT_DIR = "./output"

CLEAN_OUT = os.path.join(OUT_DIR, "clean_en_pop_metal.csv")

N_SPLITS = 4
RANDOM_SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)

# ---- load ----
csv_path = os.path.join(DATA_DIR, SOURCE_CSV)
df = pd.read_csv(csv_path)

# ---- clean + filter ----
# standardize a bit (handles weird casing / whitespace)
df["Language"] = df["Language"].astype(str).str.strip().str.lower()
df["Genre"] = df["Genre"].astype(str).str.strip()

# keep only English + Pop/Metal + non-empty lyrics
keep_genres = {"Pop", "Metal"}
clean = (
    df.loc[
        (df["Language"] == "en")
        & (df["Genre"].isin(keep_genres))
        & (df["Lyrics"].notna())
    ]
    .copy()
)

# remove empty/whitespace-only lyrics
clean["Lyrics"] = clean["Lyrics"].astype(str)
clean = clean.loc[clean["Lyrics"].str.strip().ne("")].copy()

# create your own unique IDs
clean = clean.reset_index(drop=True)
clean.insert(0, "lyric_id", np.arange(1, len(clean) + 1, dtype=np.int64))

clean_out = clean[["lyric_id", "Genre", "Lyrics"]].copy()

# save the cleaned sheet first
clean_out.to_csv(CLEAN_OUT, index=False, encoding="utf-8")
print(f"Saved cleaned dataset: {CLEAN_OUT}  (rows={len(clean_out)})")

# ---- shuffle + split into 4 equal annotation files ----
shuffled = clean_out.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# create annotation-only view: ID + lyrics + empty column for annotators
anno = shuffled[["lyric_id", "Lyrics"]].copy()
anno.insert(2, "human_annotation", "")  # empty column for humans to fill in

parts = np.array_split(anno, N_SPLITS)

for i, part in enumerate(parts, start=1):
    out_path = os.path.join(OUT_DIR, f"annotation_part_{i}_of_{N_SPLITS}.csv")
    part.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved: {out_path} (rows={len(part)})")