from pathlib import Path
import numpy as np
import pandas as pd

# Project root = one level up from scripts/
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW / "Data_Entry_2017.csv")

# The 14 official disease labels — "No Finding" is excluded on purpose
DISEASES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]

# Multi-hot encode: one column per disease, 1 if present in Finding Labels
for disease in DISEASES:
    df[disease] = df["Finding Labels"].apply(lambda x: 1 if disease in x else 0)

# Load NIH's official split (patient-level — do not rebuild this yourself)
with open(RAW / "train_val_list.txt") as f:
    train_val_files = set(f.read().splitlines())
with open(RAW / "test_list.txt") as f:
    test_files = set(f.read().splitlines())

train_val_df = df[df["Image Index"].isin(train_val_files)].copy()
test_df = df[df["Image Index"].isin(test_files)].copy()

# Split train_val into train/val by PATIENT ID, not by row
patients = train_val_df["Patient ID"].unique()
rng = np.random.default_rng(42)
rng.shuffle(patients)

n_val_patients = int(len(patients) * 0.1)
val_patients = set(patients[:n_val_patients])

train_df = train_val_df[~train_val_df["Patient ID"].isin(val_patients)]
val_df = train_val_df[train_val_df["Patient ID"].isin(val_patients)]

keep_cols = ["Image Index", "Patient ID"] + DISEASES
train_df[keep_cols].to_csv(PROCESSED / "train.csv", index=False)
val_df[keep_cols].to_csv(PROCESSED / "val.csv", index=False)
test_df[keep_cols].to_csv(PROCESSED / "test.csv", index=False)

print(f"Train: {len(train_df)} images, {train_df['Patient ID'].nunique()} patients")
print(f"Val:   {len(val_df)} images, {val_df['Patient ID'].nunique()} patients")
print(f"Test:  {len(test_df)} images, {test_df['Patient ID'].nunique()} patients")