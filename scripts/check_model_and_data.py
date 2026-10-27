import torch
from src.data.loaders import get_dataloaders
from src.models.model import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)
model = build_model(num_classes=14, pretrained=True).to(device)

images, labels = next(iter(train_loader))
images, labels = images.to(device), labels.to(device)

model.eval()
with torch.no_grad():
    output = model(images)

print(output.shape)
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")