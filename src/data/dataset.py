from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

DISEASES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]

class ChestXrayDataset(Dataset):
    def __init__(self, csv_path, images_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.images_dir = Path(images_dir)
        self.transform = transform

        # Build a lookup: filename -> full path, since NIH splits images
        # across images_001, images_002, ... folders
        self.image_paths = self._build_image_index()

    def _build_image_index(self):
        index = {}
        for folder in sorted(self.images_dir.glob("images_*")):
            img_subdir = folder / "images"
            for img_path in img_subdir.glob("*.png"):
                index[img_path.name] = img_path
        return index

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["Image Index"]
        img_path = self.image_paths[filename]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(row[DISEASES].values.astype("float32"))
        return image, labels