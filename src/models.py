from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import vgg11_bn, VGG11_BN_Weights
from torchvision.models import googlenet, GoogLeNet_Weights
import torch
import torch.nn as nn

class VGG11_BN_FC_Changed(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights = VGG11_BN_Weights.DEFAULT
        self.model = vgg11_bn(weights=weights, progress=False)
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
        self.transform = weights.transforms(antialias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transform(x)
        return self.model(x)

class Resnet18_FC_Changed(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights=ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights, pretrained=True)
        self.transform = weights.transforms(antialias=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transform(x)
        return self.model(x)
    
class GoogleNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        weights=GoogLeNet_Weights.DEFAULT
        self.model = googlenet(weights=weights, progress=False)
        self.transform = weights.transforms(antialias=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.transform(x)
        return self.model(x)