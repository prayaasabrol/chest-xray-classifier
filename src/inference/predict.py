from pathlib import Path
import torch
from PIL import Image

from src.models.model import build_model
from src.data.transforms import get_eval_transforms

DISEASES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]

class ChestXrayPredictor:
    def __init__(self, checkpoint_path="checkpoints/best_model.pt", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = build_model(num_classes=14, pretrained=False)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # important: disables dropout/batchnorm training behavior

        self.transform = get_eval_transforms()

    def predict(self, image: Image.Image) -> dict:
        """
        image: a PIL Image (already opened, any mode).
        Returns: {disease_name: probability} for all 14 diseases.
        """
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)  # add batch dim

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).cpu().squeeze(0).tolist()

        return {disease: round(prob, 4) for disease, prob in zip(DISEASES, probs)}