import torch
import torch.nn as nn
from torchvision import models


class BoneAgeNet(nn.Module):
    """ResNet-50 looks at the X-ray, a small layer handles sex, a head predicts age."""

    def __init__(self, pretrained=True):
        super().__init__()

        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        self.backbone.fc = nn.Identity()          # remove the old "name the object" layer

        self.sex_layer = nn.Sequential(nn.Linear(1, 32), nn.ReLU())

        self.head = nn.Sequential(
            nn.Linear(2048 + 32, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )

    def forward(self, x, male):
        picture_features = self.backbone(x)                       # [batch, 2048]
        sex_features = self.sex_layer(male.unsqueeze(1))          # [batch, 32]
        combined = torch.cat([picture_features, sex_features], dim=1)
        return self.head(combined).squeeze(1)                     # [batch]