import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

from src.data.loaders import get_dataloaders
from src.models.model import build_model
from src.training.losses import compute_pos_weight
from src.utils.metrics import compute_auroc

DISEASES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)          # raw logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds = [], []

    for images, labels in tqdm(loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(outputs)   # convert logits -> probabilities HERE
        all_labels.append(labels.cpu().numpy())
        all_preds.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    per_class_auroc, macro_auroc = compute_auroc(all_labels, all_preds, DISEASES)

    avg_loss = running_loss / len(loader.dataset)
    return avg_loss, macro_auroc, per_class_auroc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)

    # pos_weight needs the raw train_df, not just the loader
    import pandas as pd
    train_df = pd.read_csv("data/processed/train.csv")
    pos_weight = compute_pos_weight(train_df, DISEASES).to(device)

    model = build_model(num_classes=14, pretrained=True).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2)

    best_auroc = 0.0
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    num_epochs = 20
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auroc, per_class = evaluate(model, val_loader, criterion, device)

        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val macro-AUROC: {val_auroc:.4f}")
        scheduler.step(val_auroc)

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
            print(f"New best model saved (AUROC: {best_auroc:.4f})")

if __name__ == "__main__":
    main()