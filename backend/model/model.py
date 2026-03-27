import torch
import torch.nn as nn
from torchvision import models

class MedicalClassifier(nn.Module):
    def __init__(self, num_classes=3, freeze_layers=True):
        super().__init__()
        # Transfer Learning: EfficientNet-B4
        self.backbone = models.efficientnet_b4(weights="IMAGENET1K_V1")
        
        if freeze_layers:
            for param in list(self.backbone.parameters())[:-30]:
                param.requires_grad = False
        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)