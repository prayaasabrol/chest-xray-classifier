import torch
import pandas as pd

from src.data.loaders import get_dataloaders
from src.models.model import build_model
from src.training.train import evaluate, DISEASES

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, _, test_loader = get_dataloaders(batch_size=32)

    model = build_model(num_classes=14, pretrained=False).to(device)
    model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))

    # Reuse pos_weight from train set just so BCEWithLogitsLoss can compute a loss value —
    # doesn't affect AUROC, which is what actually matters here
    from src.training.losses import compute_pos_weight
    train_df = pd.read_csv("data/processed/train.csv")
    pos_weight = compute_pos_weight(train_df, DISEASES).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    test_loss, macro_auroc, per_class_auroc = evaluate(model, test_loader, criterion, device)

    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test macro-AUROC: {macro_auroc:.4f}\n")

    print(f"{'Disease':<20} {'AUROC':>8}")
    print("-" * 30)
    for disease, auroc in sorted(per_class_auroc.items(), key=lambda x: -x[1]):
        print(f"{disease:<20} {auroc:>8.4f}")

if __name__ == "__main__":
    main()