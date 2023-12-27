from torchvision.models import resnet18, ResNet18_Weights
import torch
import torch.nn as nn

class Resnet18_FC_Changed(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.resnet18 = resnet18(weights=weights, progress=False).eval()
        self.transforms = weights.transforms(antialias=True)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transforms(x)
        return self.resnet18(x)