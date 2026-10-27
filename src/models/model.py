import torch.nn as nn
from torchvision import models

def build_model(num_classes=14, pretrained=True):
    model = models.densenet121(weights="IMAGENET1K_V1" if pretrained else None)

    # DenseNet's final layer is called "classifier" — replace it
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    return model