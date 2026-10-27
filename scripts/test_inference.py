from PIL import Image
from src.inference.predict import ChestXrayPredictor

predictor = ChestXrayPredictor()

# Pick any real image path from your test set to try it on
image = Image.open("data/raw/images_001/images/00000013_005.png")
results = predictor.predict(image)

for disease, prob in sorted(results.items(), key=lambda x: -x[1]):
    print(f"{disease:<20} {prob:.4f}")